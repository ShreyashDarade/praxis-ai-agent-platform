# praxis/core/completion.py
"""The completion gate: "every step ran" is not "the question was answered".

Before this, a task was `completed` the moment its last step returned
without raising. That conflates two different claims. A plan can run
cleanly and still not answer the intent - the wrong column summed, a
delegation whose critic rejected the result but whose skill returned
normally, a `plan_only`-shaped output where numbers were asked for.
Each of those was reported as success, and a success is never
retried, so the one path that could have fixed it never ran.

The gate checks a finished attempt in two layers:

- **Machine checks, always.** Deterministic, free, and sufficient to
  refute the obvious cases: any step error; no step produced output
  at all; a step whose output *itself* reports failure (`succeeded:
  false`, or an `error` key) - which is exactly how a rejected
  delegation looks from outside. A result that fails here is refuted
  without spending a model call to confirm what is already known.

- **A model judgement, when a reviewer is configured.** "Does this
  result answer the objective as asked?" - judged by a model that did
  not produce the result, against the actual output, through
  `praxis.agents.critic.Critic`. This is the check that catches a
  clean run of the wrong plan. It is opt-in by construction: the
  orchestrator is built with or without a reviewer, so a deployment
  chooses it once and a test double never makes a network call.

A refuted verdict is not the end of the task; it is a failure with a
reason, and the harness re-plans from it exactly as it would from a
step error. That is the point: the diagnosis gets checked, not only
the patch.

When no reviewer is configured the judgement is recorded as
`unavailable`, not as passed. The verification block on the task says
which layer decided, so nobody reads a machine-only pass as a model's
endorsement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from praxis.agents.contract import AcceptanceCriterion, SpecialistResult, TaskContract
from praxis.agents.critic import Critic
from praxis.core.execution_graph import PlanStep
from praxis.core.harness import Verdict

_logger = structlog.get_logger(__name__)

_ANSWERS_INTENT = (
    "The result answers the objective as asked - not a plan, not an error message, "
    "not a placeholder, and not an answer to an easier or different question. Judge "
    "what the objective ASKED FOR: if it asked for a number, the number must be "
    "present; if it asked for a check or validation, a verdict with its findings IS "
    "the answer; if it asked for a chart or artifact, the chart specification or "
    "artifact reference is the answer. Data the objective itself supplied (rows, "
    "definitions, parameters) are inputs, not findings, and the result is not "
    "required to repeat them."
)


@dataclass
class GateVerdict:
    """What the gate concluded, and on what basis."""

    verdict: Verdict
    reason: str
    checks: list[dict[str, Any]] = field(default_factory=list)
    # "machine" when only deterministic checks ran, "model" when a
    # reviewer also judged, "unavailable" when a judgement was wanted
    # and no reviewer was configured.
    judged_by: str = "machine"

    @property
    def accepted(self) -> bool:
        return self.verdict is Verdict.CONFIRMED

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "accepted": self.accepted,
            "reason": self.reason,
            "judged_by": self.judged_by,
            "checks": self.checks,
        }


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes, list, tuple, dict, set)):
        return len(value) == 0
    return False


def _self_reported_failure(output: Any) -> str | None:
    """A step output that says, in its own structure, that it failed.

    The delegation skill returns normally with `succeeded: false` when
    the critic rejects a specialist's result; a connector wrapper may
    return `{"error": ...}`. From the orchestrator's side the step
    *completed*. The gate reads the output's own account instead.
    """
    if not isinstance(output, dict):
        return None
    if output.get("succeeded") is False:
        errors = output.get("errors") or []
        return "the step reports succeeded=false" + (
            f": {errors[0]}" if errors else ""
        )
    if isinstance(output.get("error"), str) and output["error"].strip():
        return f"the step's output carries an error: {output['error'][:200]}"
    return None


class CompletionGate:
    """Decides whether a finished attempt actually completed its task."""

    def __init__(self, critic: Critic | None = None) -> None:
        # A critic constructed without a catalogue can only evaluate
        # machine-checkable criteria; the gate treats that the same as
        # no critic for the judgement layer.
        self._critic = critic

    async def verify(
        self,
        *,
        task_id: str,
        intent: str,
        steps: list[PlanStep],
        results: dict[int, Any],
        errors: list[str],
    ) -> GateVerdict:
        checks: list[dict[str, Any]] = []

        # -- layer 1: machine ------------------------------------------ #
        if errors:
            checks.append({"check": "no_errors", "passed": False, "detail": errors[0][:300]})
            return GateVerdict(
                Verdict.REFUTED, f"a step failed: {errors[0][:300]}", checks, "machine"
            )
        checks.append({"check": "no_errors", "passed": True})

        outputs = [results.get(index) for index in range(len(steps))]
        if steps and all(_is_empty(output) for output in outputs):
            checks.append({"check": "produced_output", "passed": False})
            return GateVerdict(
                Verdict.REFUTED,
                "every step ran but none produced any output; there is nothing to "
                "answer from",
                checks,
                "machine",
            )
        checks.append({"check": "produced_output", "passed": True})

        for index, output in enumerate(outputs):
            failure = _self_reported_failure(output)
            if failure:
                skill = steps[index].skill_name if index < len(steps) else "?"
                checks.append(
                    {"check": "no_self_reported_failure", "passed": False,
                     "step": index, "skill": skill, "detail": failure}
                )
                return GateVerdict(
                    Verdict.REFUTED,
                    f"step {index} ('{skill}') completed but {failure}",
                    checks,
                    "machine",
                )
        checks.append({"check": "no_self_reported_failure", "passed": True})

        # -- layer 2: judgement ---------------------------------------- #
        if self._critic is None or not self._critic_can_judge():
            checks.append({"check": "answers_intent", "passed": None, "detail": "no reviewer configured"})
            return GateVerdict(
                Verdict.CONFIRMED,
                "machine checks passed; no reviewer is configured to judge whether the "
                "result answers the intent",
                checks,
                "unavailable",
            )

        contract = TaskContract(
            objective=intent,
            task_id=task_id,
            acceptance_criteria=[AcceptanceCriterion(description=_ANSWERS_INTENT)],
        )
        specialist_result = SpecialistResult(
            task_id=task_id,
            succeeded=True,
            results={
                "steps": [
                    {"step": index, "skill": step.skill_name, "output": results.get(index)}
                    for index, step in enumerate(steps)
                ]
            },
        )
        try:
            report = await self._critic.review(contract, specialist_result)
        except Exception as exc:  # noqa: BLE001 - a reviewer outage must not fail a good result
            # Recorded as unresolved rather than as a pass: the check
            # did not happen. Recorded rather than raised because a
            # result that passed every machine check should not be
            # failed by the reviewer being down.
            _logger.warning("completion_gate_reviewer_failed", task_id=task_id, error=str(exc))
            checks.append({"check": "answers_intent", "passed": None, "detail": f"reviewer failed: {exc}"})
            return GateVerdict(
                Verdict.CONFIRMED,
                f"machine checks passed; the reviewer could not run ({exc})",
                checks,
                "unavailable",
            )

        failures = report.failures
        if failures:
            reason = "; ".join(v.reason for v in failures) or "the reviewer rejected the result"
            checks.append({"check": "answers_intent", "passed": False, "detail": reason})
            _logger.info("completion_gate_refuted", task_id=task_id, reason=reason[:200])
            return GateVerdict(
                Verdict.REFUTED,
                f"the result does not answer the intent: {reason}",
                checks,
                "model",
            )

        passed_reason = next((v.reason for v in report.verdicts if v.passed), "")
        checks.append({"check": "answers_intent", "passed": True, "detail": passed_reason})
        return GateVerdict(Verdict.CONFIRMED, passed_reason or "the result answers the intent", checks, "model")

    def _critic_can_judge(self) -> bool:
        critic = self._critic
        return (
            critic is not None
            and getattr(critic, "_catalogue", None) is not None
            and getattr(critic, "_prompt_manager", None) is not None
        )
