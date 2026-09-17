# praxis/agents/skill.py
"""The `Skill` abstraction (spec §3, §7, §8).

A plain Python ABC, not tied to Claude Agent SDK `skill.md` files - that
richer, LLM-authorable mechanism is Phase 6's concern, when the
Capability Factory starts *synthesizing* new skills from scratch. Every
skill in *this* phase is hand-written and registered as a plain Python
object (see `praxis.agents.skill_registry`); Phase 6 layers synthesis
on top without needing to change this contract.

`inputs`/`outputs` are deliberately loose (`dict[str, str]`, param name
-> a short human-readable type/description string) rather than a full
JSON-schema - their only consumer this phase is the Planner's prompt
(`praxis/llm/prompts/plan_intent/v2.jinja2`), which needs enough for the
LLM to understand what each skill needs/returns, not a validating
schema engine.

`risk` is the one field the Orchestrator's risk-tiering
(`praxis.core.risk_policy.should_pause_for_approval`) reads - every
skill must self-declare `"read_only"` or `"mutating"` truthfully; the
Orchestrator enforces the pause, not the skill (spec §8 "risk-tiering
at the ACTION level").
"""
from __future__ import annotations

import abc
from typing import Any, Literal

Risk = Literal["read_only", "mutating"]


class SkillConfigurationError(RuntimeError):
    """A skill's required configuration/credentials are missing.

    Raised by a skill's own `run()` when it genuinely cannot act (e.g.
    `post_slack_message` with no `slack_bot_token` configured) - never
    swallowed into a silent no-op "pretend it worked" (spec §12: a
    failure is never wrapped as if it were a successful result).
    """


class Skill(abc.ABC):
    """Uniform contract for any hand-written or (later) synthesized skill."""

    name: str
    risk: Risk
    inputs: dict[str, str]
    outputs: dict[str, str]

    @abc.abstractmethod
    async def run(self, **kwargs: Any) -> Any: ...
