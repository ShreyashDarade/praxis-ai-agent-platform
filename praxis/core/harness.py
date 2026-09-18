# praxis/core/harness.py
"""The evidence ledger: what a task tried, what it learned, and why it
is allowed - or not allowed - to try again.

Praxis already retried a failed plan with the failure text as context.
That is necessary and not sufficient. Fed only prose about what went
wrong, a planner can produce a plan that *reads* differently and *is*
identical - the same skill with the same arguments, re-described - and
the loop would spend another model call and another minute learning
nothing. The retry had no obligation to be different, and nothing
checked whether it was.

This module makes three things structural rather than hoped for:

- **Findings are claims with evidence and a verdict.** Each step a
  plan ran becomes a `Claim`: *this skill, with these arguments,
  produces this*. The observation decides the verdict - `confirmed`
  when it produced a result, `refuted` when it failed, with the
  actual error as the evidence - and `unresolved` when it never ran
  because an earlier step failed first. Those are three different
  states, and collapsing them into "the task failed" is how a system
  ends up re-trying a step that already worked and never re-trying
  the one that was never reached.

- **A retry must differ from what was refuted.** Every attempt's plan
  is fingerprinted on the skills it calls and the arguments it passes.
  A retry whose fingerprint matches a refuted attempt is a stall, and
  `Ledger.check_retry` refuses it. The difference between attempts is
  *computed* (`plan_diff`) and recorded as the retry's justification:
  a model asked to state what changed can state anything, whereas a
  diff of the two plans cannot.

- **Rejected hypotheses are kept.** The ledger persists with the task
  and its refuted approaches feed the next planning prompt and the
  task's episodic memory, so neither this run nor a later run of the
  same intent rediscovers a dead end.

When the harness stops a task - attempts exhausted, or stalled - it
produces a `blocker` that says concretely what was tried, what each
attempt established, and what remains unresolved. "Failed" with a
stack of errors is a symptom list; the blocker is a diagnosis.

Nothing in here calls a model or touches the database. It is plain
bookkeeping over what the orchestrator observed, which is what makes
its verdicts trustworthy: they record what happened, not what a model
believes happened.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from praxis.core.execution_graph import PlanStep

# The orchestrator's wording for a step starved by an upstream failure:
# `'$1.sql' references step 1, which has no result yet`.
_KNOCK_ON_RE = re.compile(r"references step (\d+), which has no result yet")

# How much of an observation to retain as evidence. A claim's
# evidence must be readable by whoever inspects the ledger and by the
# planner on retry; a 40 KB result set is neither.
_MAX_EVIDENCE_CHARS = 2000


class Verdict(str, Enum):
    """What the evidence says about a claim.

    Three states, deliberately. "Not reproduced" is not "refuted": a
    step that never ran because its dependency failed has established
    nothing either way, and treating it as disproven would steer the
    next attempt away from an approach that may be right.
    """

    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    UNRESOLVED = "unresolved"


class StopCause(str, Enum):
    """Why the harness stopped driving a task."""

    GOAL_REACHED = "goal_reached"
    STALLED = "stalled"
    ATTEMPTS_EXHAUSTED = "attempts_exhausted"
    PLANNER_CANNOT_REPLAN = "planner_cannot_replan"
    PAUSED = "paused"
    CANCELLED = "cancelled"


def _trim(value: Any) -> Any:
    """Evidence small enough to keep and read."""
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        text = str(value)
    if len(text) <= _MAX_EVIDENCE_CHARS:
        return value
    return {
        "truncated": True,
        "chars": len(text),
        "head": text[:_MAX_EVIDENCE_CHARS],
    }


@dataclass
class Claim:
    """One thing an attempt asserted and what the evidence said.

    `statement` is what the step was expected to establish;
    `evidence` is what actually came back. The verdict is derived from
    the evidence by `Ledger.record_attempt`, never set by a caller
    reasoning about it in prose.
    """

    statement: str
    verdict: Verdict
    evidence: Any = None
    reason: str = ""
    step: int | None = None
    skill: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "skill": self.skill,
            "statement": self.statement,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass
class Attempt:
    """One plan-and-execute pass, as the ledger saw it."""

    index: int
    fingerprint: str
    plan: list[dict[str, Any]]
    claims: list[Claim] = field(default_factory=list)
    outcome: str = ""
    # Empty on the first attempt. On a retry, the computed difference
    # from the most recent refuted attempt - what actually changed.
    justification: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def refuted(self) -> list[Claim]:
        return [claim for claim in self.claims if claim.verdict is Verdict.REFUTED]

    @property
    def confirmed(self) -> list[Claim]:
        return [claim for claim in self.claims if claim.verdict is Verdict.CONFIRMED]

    @property
    def unresolved(self) -> list[Claim]:
        return [claim for claim in self.claims if claim.verdict is Verdict.UNRESOLVED]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "fingerprint": self.fingerprint,
            "plan": self.plan,
            "outcome": self.outcome,
            "justification": self.justification,
            "claims": [claim.to_dict() for claim in self.claims],
            "started_at": self.started_at.isoformat(),
        }


class RetryRefused(Exception):
    """A proposed retry would repeat a refuted approach unchanged."""

    def __init__(self, message: str, *, matches_attempt: int) -> None:
        super().__init__(message)
        self.matches_attempt = matches_attempt


# --------------------------------------------------------------------- #
# Plan identity
# --------------------------------------------------------------------- #


def _normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Arguments as identity, not as text.

    Two plans that differ only in the whitespace of a SQL string, or
    in the order of keys, are the same plan. Normalizing strings by
    collapsing whitespace and lowercasing catches the common case of
    a model re-emitting the same query with cosmetic changes and
    calling it a new approach.
    """
    normalized: dict[str, Any] = {}
    for key in sorted(args):
        value = args[key]
        if isinstance(value, str):
            normalized[key] = " ".join(value.split()).lower()
        elif isinstance(value, dict):
            normalized[key] = _normalize_args(value)
        else:
            normalized[key] = value
    return normalized


def plan_fingerprint(steps: list[PlanStep]) -> str:
    """A stable identity for a plan's *approach*.

    Skills called and arguments passed, in order, with dependencies.
    The rationale a planner might attach is deliberately excluded: a
    plan is what it does, not what it says about itself.
    """
    payload = [
        {
            "skill": step.skill_name,
            "args": _normalize_args(dict(step.args)),
            "depends_on": list(step.depends_on),
        }
        for step in steps
    ]
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def plan_diff(previous: list[dict[str, Any]], current: list[PlanStep]) -> str:
    """What changed between a refuted plan and its replacement, in words.

    This is the retry's justification. It is computed rather than
    requested so that it is true: a planner asked "what did you
    change?" can answer anything, but the diff of the two plans is
    exactly what changed and nothing else.
    """
    lines: list[str] = []
    prev_by_index = {i: entry for i, entry in enumerate(previous)}
    for index, step in enumerate(current):
        before = prev_by_index.get(index)
        if before is None:
            lines.append(f"step {index}: added {step.skill_name}({_short(step.args)})")
            continue
        if before.get("skill_name") != step.skill_name:
            lines.append(
                f"step {index}: {before.get('skill_name')} -> {step.skill_name}"
            )
            continue
        before_args = _normalize_args(dict(before.get("args") or {}))
        after_args = _normalize_args(dict(step.args))
        changed = sorted(
            key
            for key in set(before_args) | set(after_args)
            if before_args.get(key) != after_args.get(key)
        )
        if changed:
            lines.append(
                f"step {index} ({step.skill_name}): changed argument(s) {changed}"
            )
    if len(previous) > len(current):
        dropped = [
            f"{entry.get('skill_name')}" for entry in previous[len(current):]
        ]
        lines.append(f"dropped step(s): {dropped}")
    return "; ".join(lines) if lines else ""


def _short(args: dict[str, Any], limit: int = 80) -> str:
    text = json.dumps(args, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


# --------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------- #


class Ledger:
    """Per-task record of attempts, claims and verdicts."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.attempts: list[Attempt] = []
        self.stop_cause: StopCause | None = None
        self.blocker: dict[str, Any] | None = None
        # The completion gate's verdict on the final result, when one
        # ran. Distinct from step-level claims: every step can succeed
        # and the result can still fail to answer the intent.
        self.verification: dict[str, Any] | None = None

    # -- retry admission -------------------------------------------- #

    def check_retry(self, steps: list[PlanStep]) -> str:
        """Admits a retry, returning its computed justification.

        Raises `RetryRefused` when the proposed plan is the same
        approach as one already refuted. "Same" is by fingerprint -
        skills and normalized arguments - so a re-worded rationale or
        re-spaced query does not count as a change.

        The first attempt is always admitted with an empty
        justification; there is nothing yet for it to differ from.
        """
        fingerprint = plan_fingerprint(steps)
        refuted = [attempt for attempt in self.attempts if attempt.refuted]
        for attempt in refuted:
            if attempt.fingerprint == fingerprint:
                raise RetryRefused(
                    f"the proposed plan is identical to attempt {attempt.index}, which "
                    f"was already refuted: "
                    + "; ".join(claim.reason for claim in attempt.refuted),
                    matches_attempt=attempt.index,
                )
        if not refuted:
            return ""
        latest = refuted[-1]
        return plan_diff(latest.plan, steps) or "(plan differs in structure)"

    # -- recording ---------------------------------------------------- #

    def open_attempt(
        self, steps: list[PlanStep], *, justification: str = ""
    ) -> Attempt:
        attempt = Attempt(
            index=len(self.attempts),
            fingerprint=plan_fingerprint(steps),
            plan=[
                {
                    "skill_name": step.skill_name,
                    "args": dict(step.args),
                    "depends_on": list(step.depends_on),
                }
                for step in steps
            ],
            justification=justification,
        )
        self.attempts.append(attempt)
        return attempt

    def record_outcome(
        self,
        attempt: Attempt,
        *,
        status: str,
        step_status: dict[int, str],
        results: dict[int, Any],
        errors: list[str],
    ) -> None:
        """Turns one attempt's execution into claims with verdicts.

        The verdict comes from what the step did, read off the
        orchestrator's own status and results - not from an error
        message being interpreted. A step that completed is confirmed
        (its output is the evidence); one that failed is refuted (its
        error is the evidence); one that never ran is unresolved.
        """
        attempt.outcome = status
        # Errors are prefixed "step N: ..." or "step N ('skill') ..." by
        # the orchestrator; index them so each refuted claim carries
        # its own reason rather than the whole list.
        by_step: dict[int, str] = {}
        for error in errors:
            head = error.split(":", 1)[0]
            digits = "".join(ch for ch in head if ch.isdigit())
            if digits.isdigit():
                by_step.setdefault(int(digits), error)

        # An error no step owns - the completion gate refuting the whole
        # result - is a claim about the plan as a whole. Recording it as
        # refuted is what lets `check_retry` catch a plan whose steps
        # all pass and whose result still does not answer the intent;
        # without it that plan has no refuted claim to match and would
        # be admitted unchanged, forever.
        attributed = set(by_step.values())
        if status == "failed":
            for error in errors:
                if error not in attributed:
                    attempt.claims.append(
                        Claim(
                            statement="the plan as a whole answers the intent",
                            verdict=Verdict.REFUTED,
                            evidence=error[:_MAX_EVIDENCE_CHARS],
                            reason=error,
                        )
                    )
                    break

        for index, entry in enumerate(attempt.plan):
            skill = str(entry.get("skill_name"))
            statement = f"{skill}({_short(entry.get('args') or {})}) produces a usable result"
            state = step_status.get(index, "pending")
            if state == "completed":
                attempt.claims.append(
                    Claim(
                        statement=statement,
                        verdict=Verdict.CONFIRMED,
                        evidence=_trim(results.get(index)),
                        reason="the step ran and returned a result",
                        step=index,
                        skill=skill,
                    )
                )
            elif state == "failed" and _KNOCK_ON_RE.search(by_step.get(index, "")):
                # The orchestrator reports a step whose dependency failed
                # as "failed" too, with an error naming the missing
                # result. That step never executed, so nothing about its
                # approach was tested: it is unresolved, and calling it
                # refuted would steer the next plan away from a step that
                # may be exactly right once its input exists.
                upstream = _KNOCK_ON_RE.search(by_step[index]).group(1)
                attempt.claims.append(
                    Claim(
                        statement=statement,
                        verdict=Verdict.UNRESOLVED,
                        reason=f"never ran: it depends on step {upstream}, which failed first",
                        step=index,
                        skill=skill,
                    )
                )
            elif state == "failed":
                reason = by_step.get(index) or (errors[0] if errors else "the step failed")
                attempt.claims.append(
                    Claim(
                        statement=statement,
                        verdict=Verdict.REFUTED,
                        evidence=reason,
                        reason=reason,
                        step=index,
                        skill=skill,
                    )
                )
            else:
                attempt.claims.append(
                    Claim(
                        statement=statement,
                        verdict=Verdict.UNRESOLVED,
                        reason=f"never ran (status '{state}'); an earlier step failed first",
                        step=index,
                        skill=skill,
                    )
                )

    # -- what the next attempt should know ---------------------------- #

    def refuted_summary(self) -> str:
        """Every refuted claim so far, for the planning prompt.

        Grouped by attempt so the planner can see not just *that* an
        approach failed but that it was one coherent approach - and
        what, if anything, the retry after it changed.
        """
        lines: list[str] = []
        for attempt in self.attempts:
            if not attempt.refuted:
                continue
            lines.append(f"Attempt {attempt.index}" + (
                f" (changed from the previous attempt: {attempt.justification})"
                if attempt.justification else ""
            ) + ":")
            for claim in attempt.confirmed:
                lines.append(f"  - CONFIRMED step {claim.step} {claim.skill}: worked")
            for claim in attempt.refuted:
                where = f"step {claim.step} {claim.skill}" if claim.step is not None else "the whole plan"
                lines.append(f"  - REFUTED  {where}: {claim.reason}")
            for claim in attempt.unresolved:
                lines.append(f"  - UNRESOLVED step {claim.step} {claim.skill}: never ran")
        return "\n".join(lines)

    def refuted_approaches(self) -> list[dict[str, Any]]:
        """Compact form for episodic memory: which skills, which
        arguments, why they were refuted."""
        found: list[dict[str, Any]] = []
        for attempt in self.attempts:
            for claim in attempt.refuted:
                found.append(
                    {
                        "attempt": attempt.index,
                        "skill": claim.skill,
                        "args": (attempt.plan[claim.step] or {}).get("args")
                        if claim.step is not None and claim.step < len(attempt.plan)
                        else None,
                        "reason": claim.reason[:300],
                    }
                )
        return found

    # -- stopping ------------------------------------------------------ #

    def stop(self, cause: StopCause, *, detail: str = "") -> None:
        self.stop_cause = cause
        if cause in (StopCause.STALLED, StopCause.ATTEMPTS_EXHAUSTED,
                     StopCause.PLANNER_CANNOT_REPLAN):
            self.blocker = self._build_blocker(cause, detail)

    def _build_blocker(self, cause: StopCause, detail: str) -> dict[str, Any]:
        """The concrete blocker, for a person to act on.

        Answers three questions a bare "failed" does not: what was
        tried, what each attempt established, and what is still
        unknown. `unresolved` is the actionable part - it is the list
        of things no attempt got far enough to test.
        """
        unresolved = [
            {"skill": claim.skill, "statement": claim.statement}
            for attempt in self.attempts
            for claim in attempt.unresolved
        ]
        return {
            "cause": cause.value,
            "detail": detail,
            "attempts": len(self.attempts),
            "tried": [
                {
                    "attempt": attempt.index,
                    "approach": [entry.get("skill_name") for entry in attempt.plan],
                    "justification": attempt.justification,
                    "outcome": attempt.outcome,
                    "refuted": [claim.reason[:200] for claim in attempt.refuted],
                }
                for attempt in self.attempts
            ],
            "unresolved": unresolved,
        }

    # -- serialization ------------------------------------------------- #

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None, task_id: str) -> Ledger:
        """Rehydrates a persisted ledger.

        Needed because the in-memory ledger lives only as long as
        `drive_task` is running. A task that paused for approval and
        is resumed later - possibly in another process - settles
        through `_settle` with no in-memory ledger, and building a
        fresh one there would overwrite the attempts recorded before
        the pause with an empty history.
        """
        ledger = cls(task_id)
        if not payload:
            return ledger
        for raw in payload.get("attempts") or ():
            attempt = Attempt(
                index=int(raw.get("index", len(ledger.attempts))),
                fingerprint=str(raw.get("fingerprint", "")),
                plan=list(raw.get("plan") or []),
                outcome=str(raw.get("outcome", "")),
                justification=str(raw.get("justification", "")),
            )
            started = raw.get("started_at")
            if started:
                try:
                    attempt.started_at = datetime.fromisoformat(started)
                except ValueError:  # pragma: no cover - tolerate a foreign timestamp
                    pass
            for claim in raw.get("claims") or ():
                try:
                    verdict = Verdict(claim.get("verdict"))
                except ValueError:
                    verdict = Verdict.UNRESOLVED
                attempt.claims.append(
                    Claim(
                        statement=str(claim.get("statement", "")),
                        verdict=verdict,
                        evidence=claim.get("evidence"),
                        reason=str(claim.get("reason", "")),
                        step=claim.get("step"),
                        skill=claim.get("skill"),
                    )
                )
            ledger.attempts.append(attempt)
        cause = payload.get("stop_cause")
        if cause:
            try:
                ledger.stop_cause = StopCause(cause)
            except ValueError:
                ledger.stop_cause = None
        ledger.blocker = payload.get("blocker")
        ledger.verification = payload.get("verification")
        return ledger

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "stop_cause": self.stop_cause.value if self.stop_cause else None,
            "blocker": self.blocker,
            "verification": self.verification,
        }
