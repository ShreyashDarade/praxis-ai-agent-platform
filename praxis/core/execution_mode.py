# praxis/core/execution_mode.py
"""Execution modes (Prompt §8: read-only, plan-only, dry-run, execute).

The critical property, stated explicitly in the prompt: *"Plan mode
must remove or deny mutating tools, not merely tell the model not to
use them. A to-do list is not an authorization system."*

So each mode is enforced at two independent layers, not one:

1. **Capability narrowing** - `visible_skills()` filters the skill set
   the Planner is even shown, so a restricted mode's plan cannot name a
   mutating skill in the first place.
2. **Execution gating** - `Orchestrator` re-checks `allows_execution()`
   and `allows_mutation()` immediately before running each step, so a
   mutating skill that reached a plan some other way (a synthesized
   skill that declared itself mutating, a replayed plan, a restored
   checkpoint) is still refused at the point of action.

Layer 1 alone would be advisory; layer 2 alone would waste a planning
call producing a plan that cannot run. Both together are what makes the
mode an authorization control rather than a suggestion.
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime import cycle
    from praxis.agents.skill import Skill


class ExecutionMode(str, Enum):
    """How far a task is permitted to go.

    A `str` Enum so it round-trips through the API body, the `tasks.mode`
    column, and a checkpoint payload without conversion.
    """

    # Plan and stop. Produces a checklist and a `completed` task whose
    # result is the plan itself - nothing is executed at all.
    PLAN_ONLY = "plan_only"

    # Execute read-only steps for real, but refuse every mutating one.
    # The honest middle ground: real data is fetched and real analysis
    # happens, no external state changes.
    READ_ONLY = "read_only"

    # Execute read-only steps for real and *simulate* mutating ones,
    # recording what would have happened instead of doing it. Distinct
    # from `read_only`, which fails a mutating step rather than
    # simulating it (Prompt §8 lists simulation/dry-run separately).
    DRY_RUN = "dry_run"

    # Full execution with human approval on mutating steps - the
    # historical behavior and the default.
    EXECUTE = "execute"

    @property
    def allows_execution(self) -> bool:
        """False only for `plan_only`, where no step ever runs."""
        return self is not ExecutionMode.PLAN_ONLY

    @property
    def allows_mutation(self) -> bool:
        """True only for `execute` - the one mode where a mutating skill
        may genuinely touch external state (still gated on approval)."""
        return self is ExecutionMode.EXECUTE

    @property
    def simulates_mutation(self) -> bool:
        """True only for `dry_run`: a mutating step is recorded as a
        simulated outcome rather than executed or failed."""
        return self is ExecutionMode.DRY_RUN

    def describe(self) -> str:
        return _MODE_DESCRIPTIONS[self]


_MODE_DESCRIPTIONS: dict[ExecutionMode, str] = {
    ExecutionMode.PLAN_ONLY: "plan the work and stop; execute nothing",
    ExecutionMode.READ_ONLY: "execute read-only steps; refuse any mutating step",
    ExecutionMode.DRY_RUN: "execute read-only steps; simulate mutating steps without side effects",
    ExecutionMode.EXECUTE: "execute everything, pausing for approval on mutating steps",
}


def visible_skills(skills: list[Skill], mode: ExecutionMode) -> list[Skill]:
    """The skill set the Planner is allowed to see under `mode`.

    Under any mode that cannot actually mutate, mutating skills are
    *removed from the list entirely* rather than left visible with an
    instruction not to use them - this is layer 1 of the two-layer
    enforcement described in the module docstring.
    """
    if mode.allows_mutation or mode.simulates_mutation:
        return list(skills)
    return [skill for skill in skills if skill.risk != "mutating"]


class MutationNotPermittedError(PermissionError):
    """A mutating step was reached under a mode that forbids mutation.

    A `PermissionError` subclass, matching how every other refusal in
    this codebase is typed (`Connector.write`'s read-only guard,
    `praxis.security.policy`'s denials), so a caller catching
    `PermissionError` handles all of them uniformly while a caller that
    cares can name this one.
    """

    def __init__(self, message: str, *, mode: ExecutionMode, skill_name: str) -> None:
        super().__init__(message)
        self.mode = mode
        self.skill_name = skill_name
