# praxis/core/execution_graph.py
"""Execution graph (spec §3, §8): groups a plan's steps into dependency
levels so independent steps run concurrently (`asyncio.gather`) and
dependent ones wait for their inputs.

`PlanStep` lives here rather than in `praxis.agents.planner` so this
module - like everything else under `praxis.core` - depends on nothing
above it; `Planner` (an `agents`-layer component) imports `PlanStep`
from here, not the other way around.

Arg-resolution convention: an arg value that is exactly the string
`"$<index>.<output_key>"` is resolved, at the point a step is about to
run, from `results[<index>][<output_key>]` - `results` being every
earlier step's already-computed output, keyed by that step's index in
the original plan. `"$<index>"` alone (no `.<output_key>`) resolves to
that step's whole result. Any other string (or non-string value) passes
through unchanged - only this exact `"$"`-prefixed shape is special.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanStep:
    """One step of a plan: run `skill_name` with `args`, after every step
    index in `depends_on` has completed."""

    skill_name: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)


def compute_levels(steps: list[PlanStep]) -> list[list[int]]:
    """Groups step indices into dependency levels.

    Level 0 holds every step with an empty `depends_on`; level N holds
    every step whose `depends_on` are all satisfied by levels `< N` (and
    at least one dependency is *first* satisfied at level `N - 1`).
    Steps within one level have no dependency on each other and are the
    Orchestrator's concurrency unit.

    Raises `ValueError` if a dependency cycle, or a `depends_on` index
    that never resolves (out of range, or a self-reference), leaves any
    step permanently unready - a malformed plan must fail loudly here,
    not hang or silently drop steps.
    """
    step_count = len(steps)
    for index, step in enumerate(steps):
        for dep in step.depends_on:
            if dep == index or not (0 <= dep < step_count):
                raise ValueError(
                    f"step {index} ('{step.skill_name}') has an invalid dependency index {dep}"
                )

    resolved: set[int] = set()
    remaining = set(range(step_count))
    levels: list[list[int]] = []

    while remaining:
        ready = sorted(i for i in remaining if all(d in resolved for d in steps[i].depends_on))
        if not ready:
            raise ValueError("execution graph has a dependency cycle among the remaining steps")
        levels.append(ready)
        resolved.update(ready)
        remaining.difference_update(ready)

    return levels


def resolve_args(args: dict[str, Any], results: dict[int, Any]) -> dict[str, Any]:
    """Resolves every `"$<index>.<output_key>"`-shaped value in `args`
    against `results` (earlier steps' outputs, keyed by index); every
    other value passes through unchanged. See module docstring."""
    return {key: _resolve_value(value, results) for key, value in args.items()}


def _resolve_value(value: Any, results: dict[int, Any]) -> Any:
    if not isinstance(value, str) or not value.startswith("$"):
        return value

    reference = value[1:]
    index_part, _, output_key = reference.partition(".")
    try:
        index = int(index_part)
    except ValueError:
        return value  # doesn't actually match the "$<index>[.<key>]" shape - a literal string

    if index not in results:
        raise KeyError(f"'{value}' references step {index}, which has no result yet")

    step_result = results[index]
    if not output_key:
        return step_result
    if isinstance(step_result, dict) and output_key in step_result:
        return step_result[output_key]
    raise KeyError(
        f"'{value}' references output '{output_key}', which step {index}'s result does not have"
    )


StepExecutor = Callable[[int, PlanStep], Awaitable[Any]]


async def run_graph(steps: list[PlanStep], executor: StepExecutor) -> dict[int, Any]:
    """Runs every step of `steps` to completion and returns each step's
    result keyed by index.

    Steps within one dependency level run concurrently via
    `asyncio.gather`; `executor` is called once per step, after that
    step's `args` have already been resolved (via `resolve_args`)
    against every earlier step's result.

    This is the simple "run everything, no pausing" path - used by
    tests and any caller with no approval/clarification concerns.
    `praxis.core.orchestrator.Orchestrator` does NOT use this: it needs
    to stop mid-graph (on a mutating skill awaiting approval, or a
    failure) and resume later from persisted state, so it drives
    `compute_levels`/`resolve_args` itself instead.
    """
    levels = compute_levels(steps)
    results: dict[int, Any] = {}

    for level in levels:
        resolved_steps = {
            index: PlanStep(
                skill_name=steps[index].skill_name,
                args=resolve_args(steps[index].args, results),
                depends_on=steps[index].depends_on,
            )
            for index in level
        }
        outcomes = await asyncio.gather(
            *(executor(index, resolved_steps[index]) for index in level)
        )
        for index, outcome in zip(level, outcomes, strict=True):
            results[index] = outcome

    return results
