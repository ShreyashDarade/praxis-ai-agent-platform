# praxis/agents/critic.py
"""The critic/reviewer: independent validation of a specialist's result.

Prompt §1 requires *"agent output validation and critic/reviewer
agents"*, and §7's skill catalogue names a `result-validation` skill
owned by an *"Independent reviewer"*.

**Two-tier by design, and the ordering matters.** Machine-checkable
criteria are evaluated first, deterministically, with no model call at
all. Only the criteria that genuinely cannot be checked mechanically
(*"the diagnosis cites the metric it is based on"*) go to an LLM.

That ordering is the whole point rather than an optimization:

- A deterministic check cannot be talked out of its verdict, which
  matters because the thing being reviewed is itself model output.
- If every criterion is machine-checkable, the critic makes **zero**
  LLM calls - so validation is free in the common case and cannot
  itself exhaust a budget.
- The LLM tier is explicitly advisory-with-teeth: it returns a
  verdict plus reasoning, and a `fail` from it fails the criterion,
  but it is never asked to re-judge something a deterministic check
  already settled.

The critic never *repairs* a result. It reports. Silently fixing a
specialist's output would destroy the signal that the specialist is
unreliable, which is the most valuable thing this produces.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import structlog

from praxis.agents.contract import AcceptanceCriterion, SpecialistResult, TaskContract

_logger = structlog.get_logger(__name__)

_REVIEW_PROMPT_NAME = "review_result"
_REVIEW_PROMPT_VERSION = "v1"


@dataclass
class CriterionVerdict:
    """One criterion's outcome."""

    description: str
    passed: bool
    reason: str = ""
    checked_by: str = "machine"


@dataclass
class ReviewReport:
    """The critic's full verdict on one result."""

    accepted: bool
    verdicts: list[CriterionVerdict] = field(default_factory=list)
    llm_calls: int = 0

    @property
    def failures(self) -> list[CriterionVerdict]:
        return [verdict for verdict in self.verdicts if not verdict.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "llm_calls": self.llm_calls,
            "verdicts": [
                {
                    "description": v.description,
                    "passed": v.passed,
                    "reason": v.reason,
                    "checked_by": v.checked_by,
                }
                for v in self.verdicts
            ],
        }


# --------------------------------------------------------------------- #
# Machine-checkable criteria.
#
# Each takes (result, criterion) and returns (passed, reason). Adding a
# new check is one dict entry - the same open/closed shape used for
# chart builders and connector factories.
# --------------------------------------------------------------------- #

CheckFn = Callable[[SpecialistResult, AcceptanceCriterion], tuple[bool, str]]


def _check_succeeded(result: SpecialistResult, _c: AcceptanceCriterion) -> tuple[bool, str]:
    return result.succeeded, "" if result.succeeded else "specialist reported failure"


def _check_no_errors(result: SpecialistResult, _c: AcceptanceCriterion) -> tuple[bool, str]:
    return (not result.errors), f"errors reported: {result.errors}" if result.errors else ""


def _check_has_evidence(result: SpecialistResult, _c: AcceptanceCriterion) -> tuple[bool, str]:
    return bool(result.evidence), "" if result.evidence else "no evidence was attached"


def _check_output_keys(result: SpecialistResult, criterion: AcceptanceCriterion) -> tuple[bool, str]:
    expected = criterion.expected or []
    missing = [key for key in expected if key not in result.results]
    return (not missing), f"missing result key(s): {missing}" if missing else ""


def _check_non_empty_result(result: SpecialistResult, _c: AcceptanceCriterion) -> tuple[bool, str]:
    populated = any(
        value not in (None, "", [], {}) for value in result.results.values()
    )
    return populated, "" if populated else "every result value was empty"


def _check_min_rows(result: SpecialistResult, criterion: AcceptanceCriterion) -> tuple[bool, str]:
    minimum = int(criterion.expected or 1)
    for value in result.results.values():
        if isinstance(value, list) and len(value) >= minimum:
            return True, ""
    return False, f"no result field held at least {minimum} row(s)"


def _check_has_artifact(result: SpecialistResult, _c: AcceptanceCriterion) -> tuple[bool, str]:
    return bool(result.artifacts), "" if result.artifacts else "no artifact was produced"


CRITERION_CHECKS: dict[str, CheckFn] = {
    "succeeded": _check_succeeded,
    "no_errors": _check_no_errors,
    "has_evidence": _check_has_evidence,
    "output_keys": _check_output_keys,
    "non_empty_result": _check_non_empty_result,
    "min_rows": _check_min_rows,
    "has_artifact": _check_has_artifact,
}


class Critic:
    """Reviews a `SpecialistResult` against its contract's criteria.

    `catalogue`/`prompt_manager` are optional: a critic constructed
    without them evaluates only the machine-checkable criteria and
    records any judgement-based criterion as *unverified* rather than
    silently passing it. Marking it unverified-and-failing is the
    honest behavior - claiming a criterion passed because nothing was
    available to check it would be exactly the "wrap failure as
    success" pattern spec §12 forbids.
    """

    def __init__(
        self,
        catalogue: Any | None = None,
        prompt_manager: Any | None = None,
        *,
        model_purpose: str = "planning",
    ) -> None:
        self._catalogue = catalogue
        self._prompt_manager = prompt_manager
        self._model_purpose = model_purpose

    async def review(
        self, contract: TaskContract, result: SpecialistResult
    ) -> ReviewReport:
        """Evaluates every acceptance criterion; returns the report.

        A contract with no criteria is accepted with an empty verdict
        list - nothing was asked of the result, so there is nothing to
        fail. That is different from "checked and found fine", and the
        empty `verdicts` list says so.
        """
        verdicts: list[CriterionVerdict] = []
        judgement_criteria: list[AcceptanceCriterion] = []

        for criterion in contract.acceptance_criteria:
            if criterion.is_machine_checkable:
                check = CRITERION_CHECKS.get(criterion.check or "")
                if check is None:
                    verdicts.append(
                        CriterionVerdict(
                            description=criterion.description,
                            passed=False,
                            reason=f"unknown machine check '{criterion.check}'",
                            checked_by="machine",
                        )
                    )
                    continue
                passed, reason = check(result, criterion)
                verdicts.append(
                    CriterionVerdict(
                        description=criterion.description,
                        passed=passed,
                        reason=reason,
                        checked_by="machine",
                    )
                )
            else:
                judgement_criteria.append(criterion)

        llm_calls = 0
        if judgement_criteria:
            if self._catalogue is None or self._prompt_manager is None:
                verdicts.extend(
                    CriterionVerdict(
                        description=criterion.description,
                        passed=False,
                        reason=(
                            "no reviewer model is configured, so this judgement-based "
                            "criterion could not be verified"
                        ),
                        checked_by="unverified",
                    )
                    for criterion in judgement_criteria
                )
            else:
                verdicts.extend(await self._review_by_model(contract, result, judgement_criteria))
                llm_calls = 1

        return ReviewReport(
            accepted=all(verdict.passed for verdict in verdicts),
            verdicts=verdicts,
            llm_calls=llm_calls,
        )

    async def _review_by_model(
        self,
        contract: TaskContract,
        result: SpecialistResult,
        criteria: list[AcceptanceCriterion],
    ) -> list[CriterionVerdict]:
        """One model call covering every judgement criterion at once.

        Batched rather than one call per criterion: the reviewer needs
        the same context for all of them, and N calls would multiply
        cost for no additional signal.
        """
        prompt = self._prompt_manager.render(
            _REVIEW_PROMPT_NAME,
            _REVIEW_PROMPT_VERSION,
            objective=contract.objective,
            criteria=[criterion.description for criterion in criteria],
            result=json.dumps(result.to_dict(), indent=2, default=str)[:8000],
        )
        raw = await self._catalogue.complete(self._model_purpose, prompt)

        try:
            parsed = _parse_review_response(raw)
        except ValueError as exc:
            # An unparseable reviewer response fails the criteria
            # rather than passing them: a reviewer that did not produce
            # a usable verdict has not approved anything.
            _logger.warning("critic_response_unparseable", error=str(exc))
            return [
                CriterionVerdict(
                    description=criterion.description,
                    passed=False,
                    reason=f"reviewer response could not be parsed: {exc}",
                    checked_by="llm",
                )
                for criterion in criteria
            ]

        verdicts: list[CriterionVerdict] = []
        for index, criterion in enumerate(criteria):
            entry = parsed[index] if index < len(parsed) else None
            if entry is None:
                verdicts.append(
                    CriterionVerdict(
                        description=criterion.description,
                        passed=False,
                        reason="reviewer did not return a verdict for this criterion",
                        checked_by="llm",
                    )
                )
                continue
            verdicts.append(
                CriterionVerdict(
                    description=criterion.description,
                    passed=bool(entry.get("passed")),
                    reason=str(entry.get("reason", "")),
                    checked_by="llm",
                )
            )
        return verdicts


_FENCE = "```"


def _parse_review_response(raw: str) -> list[dict[str, Any]]:
    """Parses the reviewer's JSON array, tolerating a code fence.

    Mirrors `praxis.agents.planner._strip_code_fence`'s own reasoning:
    real models wrap JSON in a fence despite being asked not to, and
    failing on that would be brittle rather than strict.
    """
    text = raw.strip()
    if text.startswith(_FENCE):
        lines = [line for line in text.splitlines() if not line.strip().startswith(_FENCE)]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc

    if not isinstance(parsed, list):
        raise ValueError(f"expected a JSON array of verdicts, got {type(parsed).__name__}")
    return [entry for entry in parsed if isinstance(entry, dict)]
