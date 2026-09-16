# praxis/agents/planner.py
"""The Planner (spec §7): decomposes an intent into a structured step
list the Orchestrator turns into an execution graph (§8).

Every LLM call goes through the `LLMCatalogue` by purpose (`"planning"`
- never a hardcoded model name) and every prompt through the
`PromptManager` by `name@version` (`plan_intent@v1`) - no inline prompt
string, per spec §7.

The model is asked to answer with a bare JSON array (see
`praxis/llm/prompts/plan_intent/v1.jinja2`), but real models routinely
wrap JSON in a ```json ... ``` fence anyway - `_strip_code_fence`
tolerates that. `parse_plan_response` is a standalone, non-async
helper deliberately kept separate from any real LLM call so tests can
exercise the malformed-JSON-response error path directly, against a
hand-written bad string, without needing to provoke a real bad model
response.
"""
from __future__ import annotations

import json
import re
from typing import Any

from praxis.agents.skill import Skill
from praxis.core.execution_graph import PlanStep
from praxis.llm.catalogue import LLMCatalogue
from praxis.llm.prompt_manager import PromptManager

_PLAN_PROMPT_NAME = "plan_intent"
_PLAN_PROMPT_VERSION = "v1"

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Strips a ```json ... ``` (or bare ``` ... ```) wrapper if present.

    Real models often wrap JSON output in a markdown code fence despite
    being asked not to - this tolerates that instead of failing parsing
    on it. Text with no fence passes through unchanged (just stripped).
    """
    stripped = text.strip()
    match = _CODE_FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped


def parse_plan_response(response: str) -> list[PlanStep]:
    """Parses a Planner LLM response into `PlanStep`s.

    Raises `ValueError` with a clear, specific message (never an opaque
    `json.JSONDecodeError` or `KeyError`) when the response isn't valid
    JSON, isn't a JSON array, or an element is missing the required
    `skill_name` field.
    """
    candidate = _strip_code_fence(response)
    try:
        raw: Any = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Planner response was not valid JSON ({exc}); raw response: {response!r}"
        ) from exc

    if not isinstance(raw, list):
        raise ValueError(
            f"Planner response must be a JSON array of steps, got {type(raw).__name__}: {raw!r}"
        )

    steps: list[PlanStep] = []
    for position, item in enumerate(raw):
        if not isinstance(item, dict) or "skill_name" not in item:
            raise ValueError(
                f"Planner response step {position} is malformed - expected an object with a "
                f"'skill_name' key, got: {item!r}"
            )
        steps.append(
            PlanStep(
                skill_name=item["skill_name"],
                args=dict(item.get("args") or {}),
                depends_on=list(item.get("depends_on") or []),
            )
        )
    return steps


class Planner:
    """Decomposes an intent into a `list[PlanStep]` via a real `"planning"` LLM call."""

    def __init__(self, catalogue: LLMCatalogue, prompt_manager: PromptManager) -> None:
        self._catalogue = catalogue
        self._prompt_manager = prompt_manager

    async def plan(self, intent_text: str, available_skills: list[Skill]) -> list[PlanStep]:
        # inputs/outputs are formatted to a plain string here, in Python,
        # rather than with a nested {% for %} inside the template: Jinja's
        # `trim_blocks=True` (praxis.llm.prompt_manager.PromptManager's
        # Environment setting, shared by every prompt) eats the newline
        # immediately following *any* block tag, including one sitting
        # mid-line right before that line's own terminating newline - a
        # nested per-key loop ending a line triggers exactly that,
        # silently merging it with the next line. Formatting in Python
        # sidesteps the whole class of bug rather than fighting Jinja
        # whitespace control block-tag-by-block-tag.
        prompt = self._prompt_manager.render(
            _PLAN_PROMPT_NAME,
            _PLAN_PROMPT_VERSION,
            intent_text=intent_text,
            skills=[
                {
                    "name": skill.name,
                    "risk": skill.risk,
                    "inputs": _format_params(skill.inputs),
                    "outputs": _format_params(skill.outputs),
                }
                for skill in available_skills
            ],
        )
        response = await self._catalogue.complete("planning", prompt)
        return parse_plan_response(response)


def _format_params(params: dict[str, str]) -> str:
    return ", ".join(f"{name} ({description})" for name, description in params.items())
