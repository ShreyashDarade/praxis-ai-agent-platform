# praxis/agents/answer.py
"""Turning a finished task into an answer a person can read and check.

Before this, a completed task returned
`{"summary": "Task completed: 3/3 step(s) completed.", "steps": [...]}` -
the raw output of every step, and a sentence about arithmetic. That is
a job report, not an answer. A user who asked "what was EMEA revenue
last quarter" got a list of skill outputs to read for themselves.

Two design decisions carry the weight here.

**Evidence is computed, never generated.** The model writes the prose;
the evidence list is built in Python from what actually executed - which
step, which skill, which artifact. A model asked to cite its own sources
will produce plausible citations whether or not they are real, and a
fabricated citation is worse than none because it survives review. So
the one thing a reader uses to check the answer is the one thing the
model is not trusted to produce.

**Failure is part of the answer.** A task where two steps succeeded and
one failed has an answer, and it is not "here are two results". The
failed steps are passed to the composer explicitly and surface as
limitations, because an answer that silently omits what it could not
establish reads exactly like one that established everything.

The composer degrades rather than failing: if the model call is
unavailable, the deterministic summary is still returned with full
evidence attached. An answer without prose is worse than one with it,
and far better than a 500.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import structlog

from praxis.core.execution_graph import PlanStep
from praxis.llm.catalogue import LLMCatalogue
from praxis.llm.prompt_manager import PromptManager
from praxis.security.principal import Principal

_logger = structlog.get_logger(__name__)

_PROMPT_NAME = "compose_answer"
_PROMPT_VERSION = "v1"

# How much of one step's output to show the composer. A query can return
# thousands of rows, and the answer is almost never "every row" - it is
# a figure computed from them. Truncating keeps a large result from
# crowding out the rest of the evidence, and the truncation is disclosed
# in the evidence so a reader knows the model saw a sample.
_MAX_OUTPUT_CHARS = 4000


@dataclass
class Evidence:
    """One thing the answer rests on. Built from execution, not prose."""

    step: int
    skill: str
    summary: str
    artifact_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "skill": self.skill,
            "summary": self.summary,
            "artifact_key": self.artifact_key,
        }


@dataclass
class AssistantAnswer:
    """What the user is shown, and what it is based on."""

    text: str
    evidence: list[Evidence] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "evidence": [e.to_dict() for e in self.evidence],
            "limitations": list(self.limitations),
        }


def _render(value: Any) -> str:
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > _MAX_OUTPUT_CHARS:
        return text[:_MAX_OUTPUT_CHARS] + f"... [truncated, {len(text)} chars total]"
    return text


def _artifact_key_of(output: Any) -> str | None:
    """A stored artifact this step produced, if it produced one.

    Looked up by the key skills actually use (`artifact_key`), so a
    chart or export can be linked from the answer rather than described.
    """
    if isinstance(output, dict):
        for key in ("artifact_key", "artifact", "blob_key"):
            value = output.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def build_evidence(steps: list[PlanStep], results: dict[int, Any]) -> list[Evidence]:
    """The provenance of an answer, derived from what really ran.

    Deliberately in Python. See the module docstring: a model asked to
    cite itself produces citations that look right regardless of whether
    they are.
    """
    evidence: list[Evidence] = []
    for index, step in enumerate(steps):
        if index not in results:
            continue
        evidence.append(
            Evidence(
                step=index,
                skill=step.skill_name,
                summary=_render(results[index]),
                artifact_key=_artifact_key_of(results[index]),
            )
        )
    return evidence


def deterministic_answer(
    intent_text: str, evidence: list[Evidence], errors: list[str]
) -> AssistantAnswer:
    """The answer when no model is available to write prose.

    Still a real answer: it carries every result and every failure. The
    point of the fallback is that losing the model costs fluency, not
    information.
    """
    if evidence and not errors:
        text = (
            f"Completed {len(evidence)} step(s) for: {intent_text}\n\n"
            + "\n\n".join(f"- {e.skill}: {e.summary}" for e in evidence)
        )
    elif evidence:
        text = (
            f"Partially completed: {intent_text}\n\n"
            + "\n\n".join(f"- {e.skill}: {e.summary}" for e in evidence)
        )
    else:
        text = f"No step produced a result for: {intent_text}"
    return AssistantAnswer(text=text, evidence=evidence, limitations=list(errors))


async def compose_answer(
    *,
    intent_text: str,
    steps: list[PlanStep],
    results: dict[int, Any],
    errors: list[str],
    catalogue: LLMCatalogue,
    prompt_manager: PromptManager,
    history: str = "",
    principal: Principal | None = None,
) -> AssistantAnswer:
    """Writes the assistant's reply to one turn.

    `errors` are the steps that did not succeed; they are shown to the
    composer rather than hidden, so the answer can say what it could not
    establish instead of quietly answering a narrower question than the
    one asked.
    """
    evidence = build_evidence(steps, results)

    if not evidence and not errors:
        return AssistantAnswer(
            text="Nothing ran for this request, so there is no answer to give.",
            evidence=[],
            limitations=["No step produced a result."],
        )

    evidence_block = "\n\n".join(
        f"Step {e.step} ({e.skill}):\n{e.summary}" for e in evidence
    ) or "(no step produced a result)"

    try:
        prompt = prompt_manager.render(
            _PROMPT_NAME,
            _PROMPT_VERSION,
            intent_text=intent_text,
            history=history,
            evidence_block=evidence_block,
            failures="\n".join(f"- {e}" for e in errors),
        )
        raw = await catalogue.complete("answering", prompt, principal=principal)
        payload = json.loads(_strip_fence(raw))
        text = str(payload.get("answer") or "").strip()
        limitations = [str(x) for x in (payload.get("limitations") or []) if str(x).strip()]
        if not text:
            raise ValueError("composer returned an empty answer")
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the task
        _logger.warning("answer_composition_failed", error=str(exc))
        return deterministic_answer(intent_text, evidence, errors)

    # Failures are appended rather than left to the model's discretion:
    # a step that did not run is a limitation whether or not the composer
    # thought to mention it.
    for error in errors:
        if not any(error in limitation for limitation in limitations):
            limitations.append(error)

    return AssistantAnswer(text=text, evidence=evidence, limitations=limitations)


def _strip_fence(raw: str) -> str:
    """Models wrap JSON in a code fence more often than not."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -len("```")]
    return text.strip()
