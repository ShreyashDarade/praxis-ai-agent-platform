# praxis/core/graph_engine.py
"""LangGraph execution engine for a `PlanStep` DAG.

This is the deterministic half of Praxis's execution model, rebuilt on
LangGraph. The Orchestrator previously executed plans with a
hand-rolled level-by-level driver: `compute_levels()` grouped steps
into dependency levels, then each level ran under `asyncio.gather`,
with pauses and resumption managed through in-memory bookkeeping.

That worked, but LangGraph does the same job with properties the
hand-rolled version could not offer:

- **The DAG is the graph.** One node per `PlanStep`, edges taken
  straight from `depends_on`. LangGraph's superstep model runs
  independent nodes concurrently by itself, so the level computation
  becomes a property of the graph rather than something to maintain.
- **`interrupt()` is the pause.** A mutating step raises a real
  interrupt; resuming is `Command(resume=...)` on the same thread.
  Crucially, LangGraph replays the interrupted node from its start on
  resume, so the approval check and the action stay in one function
  instead of being split across a pause site and a separate resume
  path that had to reconstruct the same arguments.
- **Checkpointing is per-superstep and durable**, handled by the
  library rather than by a bespoke serializer.
- **State history gives replay** (`aget_state_history`), which is the
  brief's "replay/debug mode" - previously unavailable for tasks.

**What this module deliberately does not own.** Every
security-relevant decision stays in the Orchestrator and is passed in
as callbacks: risk tiering, approval-record creation and
argument-hash verification, tenancy, audit, execution-mode
enforcement, and capability synthesis. This module only wires them
into a graph and runs it. That split is the point - the migration had
to preserve those properties exactly, so they were not rewritten.

The state channels use explicit reducers because nodes run
concurrently: two steps finishing in the same superstep both write
`results`, and without a merge reducer one would clobber the other.
"""
from __future__ import annotations

import operator
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, TypedDict

import structlog
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph

from praxis.core.execution_graph import PlanStep

_logger = structlog.get_logger(__name__)


def _merge_results(left: dict[int, Any], right: dict[int, Any]) -> dict[int, Any]:
    """Merges per-step results from concurrently-finishing nodes.

    Required rather than optional: LangGraph applies every node's
    update from one superstep to the same channel, so without this
    two steps completing together would overwrite each other's
    output.
    """
    merged = dict(left)
    merged.update(right)
    return merged


def _merge_status(left: dict[int, Any], right: dict[int, Any]) -> dict[int, Any]:
    merged = dict(left)
    merged.update(right)
    return merged


class TaskGraphState(TypedDict, total=False):
    """The graph's state channel.

    Deliberately only JSON-ish data: a checkpointed state must be
    serializable, so live objects (connectors, sessions, skills) stay
    on the Orchestrator and are reached through closures instead.
    """

    # step index -> that step's output
    results: Annotated[dict[int, Any], _merge_results]
    # step index -> "completed" | "failed" | "skipped"
    step_status: Annotated[dict[int, str], _merge_status]
    # URLs surfaced by any completed step (spec §6.1's prior-context rule)
    known_urls: Annotated[list[str], operator.add]
    # Failure reasons, accumulated rather than replaced so a
    # multi-step failure reports all of them.
    errors: Annotated[list[str], operator.add]


# Runs one step. Receives the step index, the step, and the results of
# every step completed so far; returns the state update for that node.
StepRunner = Callable[[int, PlanStep, dict[int, Any], list[str]], Awaitable[dict[str, Any]]]


def build_task_graph(
    steps: list[PlanStep],
    runner: StepRunner,
    *,
    checkpointer: Any | None = None,
) -> Any:
    """Compiles `steps` into an executable LangGraph.

    One node per step; an edge for every `depends_on` entry. A step
    with no dependencies is wired from `START`, which is what makes
    LangGraph run all of them in the first superstep - the same
    concurrency the level-based driver provided, without maintaining
    the levels.

    Compiled per task rather than once: a plan is only known after
    the Planner runs, and compilation is cheap.
    """
    graph = StateGraph(TaskGraphState)

    for index, step in enumerate(steps):
        graph.add_node(_node_name(index), _make_node(index, step, runner))

    terminal_indices = _terminal_indices(steps)

    for index, step in enumerate(steps):
        if not step.depends_on:
            graph.add_edge(START, _node_name(index))
        else:
            for dependency in step.depends_on:
                graph.add_edge(_node_name(dependency), _node_name(index))

    for index in terminal_indices:
        graph.add_edge(_node_name(index), END)

    return graph.compile(checkpointer=checkpointer)


def _node_name(index: int) -> str:
    """Node ids are positional, not skill names: a plan may legitimately
    call the same skill twice, and LangGraph node ids must be unique."""
    return f"step_{index}"


def _terminal_indices(steps: list[PlanStep]) -> list[int]:
    """Steps nothing else depends on - the graph's leaves."""
    depended_upon: set[int] = set()
    for step in steps:
        depended_upon.update(step.depends_on)
    return [index for index in range(len(steps)) if index not in depended_upon]


def _make_node(index: int, step: PlanStep, runner: StepRunner):
    """Wraps one step's runner as a LangGraph node.

    A node that raises would abort the whole graph, so the runner is
    expected to return a state update describing failure instead. The
    guard here is a backstop for a genuinely unexpected error, keeping
    a single step's crash from discarding every sibling's completed
    work.

    **`GraphInterrupt` must pass through untouched.** LangGraph
    signals a pause by raising it - it is control flow, not a failure.
    Catching it (which a bare `except Exception` does) turns every
    approval pause into a crashed step, which is exactly the bug this
    re-raise prevents.
    """

    async def node(state: TaskGraphState) -> dict[str, Any]:
        results = dict(state.get("results", {}))
        known_urls = list(state.get("known_urls", []))
        try:
            return await runner(index, step, results, known_urls)
        except GraphInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - see docstring
            _logger.error(
                "task_graph_node_crashed",
                step=index,
                skill=step.skill_name,
                error=str(exc),
            )
            return {
                "step_status": {index: "failed"},
                "errors": [f"step {index} ('{step.skill_name}') crashed: {exc}"],
            }

    node.__name__ = _node_name(index)
    return node
