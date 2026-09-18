# praxis/agents/loop.py
"""The agentic loop, built on LangGraph's `StateGraph`.

Praxis's Orchestrator executes a **plan** - a DAG decided up front and
run to completion. That is right for deterministic work, and the
product brief asks for exactly that: *"Deterministic workflows for
high-risk operations; agentic workflows for ambiguous
research/analysis."*

This module is the agentic half, for work where the next step depends
on what the last step found: *"Find why customer churn increased last
month"* cannot be decomposed into a correct DAG in advance.

**Why LangGraph rather than a hand-rolled loop.** The first version of
this module was hand-rolled. That was the wrong call: LangGraph is
purpose-built for exactly this shape, is already the library the
product brief's own stack table names for the portable execution
profile, and provides for free several things that are genuinely hard
to get right:

- **Cyclic graphs with conditional routing** - the core loop shape.
- **Bounded recursion** (`recursion_limit`) - a loop does not
  terminate by construction, and this is the backstop.
- **Durable checkpointing per thread**, with `PostgresSaver`
  available, so a loop survives a restart.
- **State history / time-travel** (`get_state_history`) - which is
  precisely the *"replay/debug mode"* the brief requires and which no
  hand-rolled version here had.
- **`interrupt()` / `Command(resume=...)`** - first-class
  human-in-the-loop.

What this module still owns, because LangGraph does **not** provide
it and it is load-bearing here:

- **Stall detection.** LangGraph's `recursion_limit` stops a runaway
  loop, but only after burning every remaining step. A loop taking
  the *same* action and getting the *same* observation has stopped
  making progress, and "stalled" is a different diagnosis from "ran
  out of steps" - so the two get different stop reasons.
- **Contract authorization.** Every tool call is checked against the
  task contract's `authorized_tools`, including tools reached
  indirectly through a `ToolChain`.
- **Budget enforcement** against the shared `BudgetTracker`, so a loop
  cannot out-spend an allowance shared with its siblings.
- **Delegation limits** (depth, recursion, fan-out) via
  `praxis.agents.delegation`.

The deterministic `Orchestrator` is deliberately NOT migrated to
LangGraph: its approval path carries argument-hash binding, tenancy,
and audit semantics that are the security core of this system, and
rewriting it would risk losing those properties for no gain - it
executes a DAG, which is the one shape a plain executor already does
well.
"""
from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any, TypedDict

import structlog
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from praxis.agents.budget import BudgetExhaustedError, BudgetTracker
from praxis.agents.contract import SpecialistResult, TaskContract
from praxis.agents.delegation import DelegationLimitError, DelegationRegistry
from praxis.agents.subagent import AgentContext, SubAgent

_logger = structlog.get_logger(__name__)

DEFAULT_MAX_ITERATIONS = 12

# Argument key under which a policy attaches its hypothesis for an
# action. Owned here rather than in `praxis.agents.policy` because the
# loop is what must strip it before a tool sees it, and what must
# ignore it when deciding whether two actions are the same.
_HYPOTHESIS_KEY = "__hypothesis__"
# How many identical (action, observation) pairs before the loop is
# considered stalled. Two is too eager - a legitimate retry after a
# transient failure repeats once. Three means it has genuinely
# stopped learning anything new.
DEFAULT_STALL_THRESHOLD = 3


class StopReason(str, Enum):
    """Why a loop ended. Every exit path names one."""

    GOAL_REACHED = "goal_reached"
    MAX_ITERATIONS = "max_iterations"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    STALLED = "stalled"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def is_success(self) -> bool:
        return self is StopReason.GOAL_REACHED


class ActionKind(str, Enum):
    """The three ways an iteration can act, plus finishing."""

    SKILL = "skill"
    CHAIN = "chain"
    DELEGATE = "delegate"
    FINISH = "finish"


@dataclass
class Action:
    """What the loop decided to do this iteration."""

    kind: ActionKind
    target: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def fingerprint(self) -> str:
        """Identity of this action, for stall detection.

        Includes the arguments: calling the same tool with *different*
        arguments is progress; calling it with the same ones is not.
        """
        # The policy may attach a hypothesis under a reserved key (see
        # `praxis.agents.policy`). It is prose about the action, not
        # part of it, and must not make a re-worded repeat look like a
        # different action - that would defeat stall detection exactly
        # when it matters.
        identity = {k: v for k, v in self.args.items() if k != _HYPOTHESIS_KEY}
        payload = json.dumps(
            {"kind": self.kind.value, "target": self.target, "args": identity},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class Observation:
    """What came back from acting."""

    succeeded: bool
    content: Any = None
    error: str = ""

    def fingerprint(self) -> str:
        payload = json.dumps(
            {"ok": self.succeeded, "content": self.content, "error": self.error},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class Iteration:
    """One full turn of the loop, kept for the trace."""

    index: int
    action: Action
    observation: Observation
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": {
                "kind": self.action.kind.value,
                "target": self.action.target,
                "args": self.action.args,
                "rationale": self.action.rationale,
            },
            "observation": {
                "succeeded": self.observation.succeeded,
                "content": self.observation.content,
                "error": self.observation.error,
            },
            "started_at": self.started_at.isoformat(),
        }


@dataclass
class LoopResult:
    """The outcome of a whole loop, with its full trace.

    The trace is the point: an agentic loop that produces an answer
    without showing how it got there is unauditable, and the brief
    requires evidence alongside results.
    """

    stop_reason: StopReason
    iterations: list[Iteration] = field(default_factory=list)
    answer: Any = None
    # Iteration indices the answer cites. Empty when the policy cited
    # nothing - which the caller should read as "unsupported".
    evidence: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    spend: dict[str, Any] = field(default_factory=dict)
    # LangGraph thread id, when a checkpointer was configured - the
    # handle for replay/time-travel via `AgentLoop.history`.
    thread_id: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.stop_reason.is_success

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stop_reason": self.stop_reason.value,
            "succeeded": self.succeeded,
            "iterations": [iteration.to_dict() for iteration in self.iterations],
            "iteration_count": self.iteration_count,
            "answer": self.answer,
            "evidence": self.evidence,
            "errors": self.errors,
            "spend": self.spend,
            "thread_id": self.thread_id,
        }


class _LoopState(TypedDict, total=False):
    # Iteration indices a FINISH action cited as the basis of its answer.
    evidence: list[int]
    """The LangGraph state channel for one loop run.

    `iterations` and `errors` use `operator.add` reducers so each node
    *appends* rather than replacing - which is what makes the
    accumulated trace survive checkpoint/resume correctly.
    """

    iterations: Annotated[list[Iteration], operator.add]
    errors: Annotated[list[str], operator.add]
    fingerprints: Annotated[list[str], operator.add]
    answer: Any
    stop_reason: str | None


# The loop asks this for its next move, given the objective and
# everything observed so far. A callable rather than a hardcoded LLM
# call so the decision policy is swappable - an LLM in production, a
# scripted policy in a test, a rule-based one for a deterministic
# workflow.
Policy = Callable[[TaskContract, list[Iteration]], Awaitable[Action]]


class AgentLoop:
    """A bounded observe/decide/act loop, executed by LangGraph."""

    def __init__(
        self,
        *,
        policy: Policy,
        skills: dict[str, Any] | None = None,
        chains: dict[str, Any] | None = None,
        agents: dict[str, SubAgent] | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        stall_threshold: int = DEFAULT_STALL_THRESHOLD,
        delegation: DelegationRegistry | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self._policy = policy
        self._skills = skills or {}
        self._chains = chains or {}
        self._agents = agents or {}
        self._max_iterations = max_iterations
        self._stall_threshold = stall_threshold
        self._delegation = delegation if delegation is not None else DelegationRegistry()
        self._checkpointer = checkpointer
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Compiles the two-node cycle: decide-and-act, then route.

        Deliberately one node rather than separate decide/act nodes:
        a policy failure and an action failure need the same handling
        (record it, let the loop decide what to do next), and
        splitting them would duplicate that logic across two nodes
        for no benefit.
        """
        graph = StateGraph(_LoopState)
        graph.add_node("step", self._step)
        graph.add_edge(START, "step")
        graph.add_conditional_edges(
            "step", self._route, {"step": "step", END: END}
        )
        return graph.compile(checkpointer=self._checkpointer)

    # ---------------------------------------------------------------- #
    # Graph nodes. `_contract`/`_budget`/`_cancelled` are set for the
    # duration of one `run()` - the graph's own state channel carries
    # only JSON-ish trace data, so a checkpointed state stays
    # serializable rather than holding live objects.
    # ---------------------------------------------------------------- #

    async def _step(self, state: _LoopState) -> dict[str, Any]:
        contract = self._contract
        tracker = self._budget
        iterations = list(state.get("iterations", []))

        if self._is_cancelled is not None and self._is_cancelled():
            return {"stop_reason": StopReason.CANCELLED.value}

        if contract.is_expired:
            return {"stop_reason": StopReason.DEADLINE_EXCEEDED.value}

        try:
            tracker.check_deadline()
        except BudgetExhaustedError as exc:
            return {
                "stop_reason": StopReason.BUDGET_EXHAUSTED.value,
                "errors": [str(exc)],
            }

        try:
            action = await self._policy(contract, iterations)
        except BudgetExhaustedError as exc:
            return {
                "stop_reason": StopReason.BUDGET_EXHAUSTED.value,
                "errors": [str(exc)],
            }
        except Exception as exc:  # noqa: BLE001 - a policy failure ends the loop, not the process
            return {
                "stop_reason": StopReason.FAILED.value,
                "errors": [f"policy failed: {exc}"],
            }

        if action.kind is ActionKind.FINISH:
            # The citations travel with the answer. A policy that says
            # which observations an answer rests on is only useful if
            # the loop keeps that list rather than discarding every arg
            # but `answer`, which is what happened before.
            cited = action.args.get("evidence") or []
            return {
                "stop_reason": StopReason.GOAL_REACHED.value,
                "answer": action.args.get("answer"),
                "evidence": [int(i) for i in cited if str(i).lstrip("-").isdigit()],
            }

        try:
            observation = await self._act(action, contract, tracker)
        except BudgetExhaustedError as exc:
            return {
                "stop_reason": StopReason.BUDGET_EXHAUSTED.value,
                "errors": [str(exc)],
            }

        iteration = Iteration(
            index=len(iterations), action=action, observation=observation
        )
        update: dict[str, Any] = {
            "iterations": [iteration],
            "fingerprints": [f"{action.fingerprint()}:{observation.fingerprint()}"],
        }
        if not observation.succeeded and observation.error:
            update["errors"] = [observation.error]
        return update

    def _route(self, state: _LoopState) -> str:
        """Decides whether to loop again, and detects a stall.

        LangGraph's `recursion_limit` handles the runaway case, but
        only by exhausting it. Stall detection stops a loop that has
        stopped learning, and reports a different reason for it.
        """
        if state.get("stop_reason"):
            return END

        fingerprints = state.get("fingerprints", [])
        if fingerprints:
            latest = fingerprints[-1]
            if fingerprints.count(latest) >= self._stall_threshold:
                _logger.info(
                    "agent_loop_stalled",
                    repeats=fingerprints.count(latest),
                    task_id=self._contract.task_id,
                )
                # Recorded on the instance rather than in state: this
                # runs in a routing function, whose return value is
                # the next node, not a state update.
                self._stalled = True
                return END

        if len(state.get("iterations", [])) >= self._max_iterations:
            return END
        return "step"

    # ---------------------------------------------------------------- #
    # Actions
    # ---------------------------------------------------------------- #

    async def _act(
        self, action: Action, contract: TaskContract, tracker: BudgetTracker
    ) -> Observation:
        """Performs one action.

        Returns a failed `Observation` rather than raising for
        ordinary failure - a failed action is information the loop can
        reason about and recover from, which is the entire advantage
        of a loop over a DAG. A `BudgetExhaustedError` is allowed to
        propagate, because that is a hard stop rather than an
        observation.
        """
        try:
            if action.kind is ActionKind.SKILL:
                return await self._act_skill(action, contract)
            if action.kind is ActionKind.CHAIN:
                return await self._act_chain(action, contract)
            if action.kind is ActionKind.DELEGATE:
                return await self._act_delegate(action, contract, tracker)
        except BudgetExhaustedError:
            raise
        except Exception as exc:  # noqa: BLE001 - see docstring
            return Observation(succeeded=False, error=f"{type(exc).__name__}: {exc}")
        return Observation(succeeded=False, error=f"unsupported action kind {action.kind}")

    async def _act_skill(self, action: Action, contract: TaskContract) -> Observation:
        if not contract.authorizes(action.target):
            return Observation(
                succeeded=False,
                error=(
                    f"task is not authorized to use tool '{action.target}'; "
                    f"authorized: {list(contract.authorized_tools)}"
                ),
            )
        skill = self._skills.get(action.target)
        if skill is None:
            return Observation(
                succeeded=False, error=f"no skill named '{action.target}' is available"
            )
        tool_args = {k: v for k, v in action.args.items() if k != _HYPOTHESIS_KEY}
        output = await skill.run(**tool_args, tenant_id=contract.tenant_id)
        return Observation(succeeded=True, content=output)

    async def _act_chain(self, action: Action, contract: TaskContract) -> Observation:
        chain = self._chains.get(action.target)
        if chain is None:
            return Observation(
                succeeded=False, error=f"no tool chain named '{action.target}'"
            )
        # Every tool in the chain is still individually authorized -
        # composing tools must not be a way to reach one the contract
        # did not grant.
        for link in chain.links:
            if not contract.authorizes(link.tool.name):
                return Observation(
                    succeeded=False,
                    error=(
                        f"chain '{action.target}' uses tool '{link.tool.name}', which this "
                        "task is not authorized to use"
                    ),
                )
        result = await chain.run(tenant_id=contract.tenant_id)
        return Observation(succeeded=True, content=result.to_dict())

    async def _act_delegate(
        self, action: Action, contract: TaskContract, tracker: BudgetTracker
    ) -> Observation:
        agent = self._agents.get(action.target)
        if agent is None:
            return Observation(
                succeeded=False, error=f"no agent named '{action.target}' is registered"
            )

        # A delegation must state its own objective. Defaulting to the
        # parent's would make every unqualified delegation
        # self-referential - literally the A->A shape the recursion
        # detector exists to catch - so the loop would trip its own
        # guard on legitimate work. Requiring it is also just correct:
        # handing a specialist the parent's whole objective is
        # duplication, not delegation.
        objective = str(action.args.get("objective") or "").strip()
        if not objective:
            return Observation(
                succeeded=False,
                error=(
                    f"delegation to '{action.target}' specified no objective; a delegated "
                    "task must state what the specialist is being asked to do"
                ),
            )

        child = TaskContract.child_of(
            contract,
            objective=objective,
            authorized_tools=tuple(agent.manifest.tools) or None,
            inputs=dict(action.args.get("inputs") or {}),
        )
        try:
            self._delegation.register_child(child)
        except DelegationLimitError as exc:
            return Observation(succeeded=False, error=str(exc))

        child_budget = tracker.child(agent.manifest.default_budget)
        context = AgentContext(
            contract=child,
            budget=child_budget,
            tools={
                name: skill
                for name, skill in self._skills.items()
                if child.authorizes(name)
            },
            tenant_id=contract.tenant_id,
        )
        result: SpecialistResult = await agent.execute(context)
        return Observation(
            succeeded=result.succeeded,
            content=result.to_dict(),
            error="; ".join(result.errors),
        )

    # ---------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------- #

    async def run(
        self,
        contract: TaskContract,
        *,
        budget: BudgetTracker | None = None,
        is_cancelled: Callable[[], bool] | None = None,
        thread_id: str | None = None,
    ) -> LoopResult:
        """Runs until the goal is reached or a bound stops it."""
        self._contract = contract
        self._budget = budget if budget is not None else BudgetTracker()
        self._is_cancelled = is_cancelled
        self._stalled = False

        self._delegation.register_root(contract)

        resolved_thread = thread_id or contract.task_id
        config: dict[str, Any] = {
            # +2 covers the terminal routing pass, so the iteration
            # ceiling below is what actually stops a normal loop and
            # LangGraph's own limit stays a backstop rather than the
            # primary bound.
            "recursion_limit": self._max_iterations + 2,
        }
        if self._checkpointer is not None:
            config["configurable"] = {"thread_id": resolved_thread}

        try:
            final = await self._graph.ainvoke(
                {"iterations": [], "errors": [], "fingerprints": []}, config
            )
        except GraphRecursionError:
            # LangGraph's backstop fired before our own ceiling - same
            # meaning, reported with our own vocabulary.
            return self._result(
                StopReason.MAX_ITERATIONS, [], [], resolved_thread
            )

        stop_reason = self._resolve_stop_reason(final)
        return self._result(
            stop_reason,
            list(final.get("iterations", [])),
            list(final.get("errors", [])),
            resolved_thread,
            answer=final.get("answer"),
            evidence=list(final.get("evidence", [])),
        )

    def _resolve_stop_reason(self, final: _LoopState) -> StopReason:
        if self._stalled:
            return StopReason.STALLED
        raw = final.get("stop_reason")
        if raw:
            return StopReason(raw)
        return StopReason.MAX_ITERATIONS

    def _result(
        self,
        reason: StopReason,
        iterations: list[Iteration],
        errors: list[str],
        thread_id: str,
        *,
        answer: Any = None,
        evidence: list[int] | None = None,
    ) -> LoopResult:
        _logger.info(
            "agent_loop_finished",
            stop_reason=reason.value,
            iterations=len(iterations),
            errors=len(errors),
        )
        return LoopResult(
            stop_reason=reason,
            iterations=iterations,
            errors=errors,
            answer=answer,
            evidence=list(evidence or []),
            spend=self._budget.snapshot(),
            thread_id=thread_id if self._checkpointer is not None else None,
        )

    def history(self, thread_id: str) -> list[dict[str, Any]]:
        """Every checkpointed state for a thread, newest first.

        This is the brief's *"replay/debug mode"*: it reconstructs
        what the loop believed at each step, which is what makes an
        agentic run debuggable after the fact. Requires a
        checkpointer; returns an empty list without one rather than
        pretending history exists.
        """
        if self._checkpointer is None:
            return []
        config = {"configurable": {"thread_id": thread_id}}
        snapshots = []
        for snapshot in self._graph.get_state_history(config):
            snapshots.append(
                {
                    "step": snapshot.metadata.get("step") if snapshot.metadata else None,
                    "iterations": len(snapshot.values.get("iterations", [])),
                    "stop_reason": snapshot.values.get("stop_reason"),
                    "next": list(snapshot.next),
                }
            )
        return snapshots
