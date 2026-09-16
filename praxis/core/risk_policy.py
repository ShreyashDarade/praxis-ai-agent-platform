# praxis/core/risk_policy.py
"""Risk tiering, enforced at the Orchestrator level (spec §8).

"Risk-tiering at the ACTION level ... mutating actions pause for
approval; everything else runs straight through." Deliberately a
free function taking a `Skill`, not a method a skill implements itself
- the whole point is that no skill can opt itself out of the approval
gate; the Orchestrator, not the skill, decides whether to pause,
purely from the skill's own declared `risk` field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from praxis.agents.skill import Skill

PendingInputKind = Literal["approval", "clarification"]


@dataclass
class PendingInput:
    """The shared pause/resume primitive (spec §8): one shape for both
    risk-tiered approval and Planner/Factory-raised clarification -
    "one mechanism, `PendingInput(kind: approval | clarification, ...)`,
    not two".
    """

    kind: PendingInputKind
    detail: str
    options: list[str] | None = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "options": self.options}


def should_pause_for_approval(skill: Skill) -> bool:
    """True iff `skill.risk == "mutating"` - the sole condition for a pause."""
    return skill.risk == "mutating"
