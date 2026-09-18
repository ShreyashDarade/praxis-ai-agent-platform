# praxis/security/approval.py
"""Identity-bound, expiring, idempotent, argument-bound approvals.

Prompt §8: "Approval records must bind approver, tenant, exact
action/arguments, expiry, and idempotency key. Recheck authorization
when a paused or scheduled task resumes."

All five bindings are real here:

- **approver** - `ApprovalRecord.approver_user_id`, taken from the
  authenticated `Principal`, never from the request body.
- **tenant** - the record carries the task's tenant, and a decision
  from a principal in another tenant is refused.
- **exact arguments** - `action_hash` is a SHA-256 over the canonical
  `(skill_name, resolved_args)` the operator was actually shown. At
  resume time the orchestrator recomputes it from the args it is about
  to run; a mismatch (the plan changed underneath the approval) is a
  hard refusal, not a warning.
- **expiry** - a pending approval past `expires_at` can no longer be
  decided; the stale-approval sweep fails the task instead.
- **idempotency key** - unique per `(task, step, action_hash)`, so a
  duplicated or replayed approve call resolves to the *same* record
  rather than authorizing a second execution.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from praxis.memory.models import ApprovalRecord
from praxis.security.principal import Principal


class ApprovalBindingError(Exception):
    """The approval being decided does not match the action about to run.

    Carries both hashes so the failure is diagnosable (which is the
    point of binding at all) without re-deriving them at the call site.
    """

    def __init__(self, message: str, *, expected_hash: str, actual_hash: str) -> None:
        super().__init__(message)
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash


class ApprovalExpiredError(Exception):
    """The approval window closed before a decision was recorded."""

    def __init__(self, message: str, *, expired_at: datetime) -> None:
        super().__init__(message)
        self.expired_at = expired_at


@dataclass(frozen=True)
class ApprovalRequest:
    """What an operator is being asked to approve - the exact shape whose
    hash gets bound into the record."""

    task_id: str
    step_index: int
    skill_name: str
    args: dict[str, Any]

    @property
    def action(self) -> str:
        return f"{self.skill_name}(step {self.step_index})"


def compute_action_hash(skill_name: str, args: dict[str, Any]) -> str:
    """Canonical SHA-256 over the action + its exact arguments.

    `sort_keys=True` makes the hash independent of dict ordering;
    `default=str` keeps it total over non-JSON values (a datetime, a
    set) rather than raising on a payload the orchestrator legitimately
    produced.
    """
    payload = json.dumps({"skill": skill_name, "args": args}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_idempotency_key(task_id: str, step_index: int, action_hash: str) -> str:
    """Stable across retries of the *same* approval, distinct across
    different actions - so replaying an approve call is a no-op rather
    than a second authorization."""
    return hashlib.sha256(
        f"{task_id}:{step_index}:{action_hash}".encode("utf-8")
    ).hexdigest()[:64]


async def create_pending_approval(
    session: AsyncSession,
    *,
    request: ApprovalRequest,
    tenant_id: str,
    ttl_seconds: int,
) -> ApprovalRecord:
    """Creates (or returns the existing) pending record for this exact action.

    Idempotent by construction: pausing the same step on the same task
    with the same args twice - e.g. after a process restart replays the
    pause - resolves to the one record rather than a duplicate.
    """
    action_hash = compute_action_hash(request.skill_name, request.args)
    idempotency_key = compute_idempotency_key(request.task_id, request.step_index, action_hash)

    existing = (
        await session.execute(
            select(ApprovalRecord).where(ApprovalRecord.idempotency_key == idempotency_key)
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    record = ApprovalRecord(
        tenant_id=tenant_id,
        task_id=request.task_id,
        step_index=request.step_index,
        action=request.action,
        action_hash=action_hash,
        idempotency_key=idempotency_key,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
    )
    session.add(record)
    await session.flush()
    return record


async def decide_approval(
    session: AsyncSession,
    *,
    task_id: str,
    step_index: int,
    approved: bool,
    principal: Principal,
) -> ApprovalRecord:
    """Records `principal`'s decision on the pending approval.

    Returns the already-decided record unchanged when one exists (the
    idempotency guarantee): a duplicate approve call never flips a
    decision, never re-authorizes, and never errors.

    Raises `KeyError` when no pending approval exists, and
    `ApprovalExpiredError` when the window has closed.
    """
    record = (
        await session.execute(
            select(ApprovalRecord)
            .where(
                ApprovalRecord.task_id == task_id,
                ApprovalRecord.step_index == step_index,
                ApprovalRecord.tenant_id == principal.tenant_id,
            )
            .order_by(ApprovalRecord.created_at.desc())
        )
    ).scalars().first()

    if record is None:
        raise KeyError(
            f"no approval record for task '{task_id}' step {step_index} in tenant "
            f"'{principal.tenant_id}'"
        )

    if record.decided_at is not None:
        return record

    expires_at = record.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at <= datetime.now(timezone.utc):
        raise ApprovalExpiredError(
            f"approval for task '{task_id}' step {step_index} expired at {expires_at.isoformat()}",
            expired_at=expires_at,
        )

    record.approved = approved
    record.approver_user_id = principal.user_id
    record.decided_at = datetime.now(timezone.utc)
    await session.flush()
    return record


async def verify_approval_binding(
    session: AsyncSession,
    *,
    task_id: str,
    step_index: int,
    skill_name: str,
    args: dict[str, Any],
) -> ApprovalRecord:
    """Re-checks, at resume time, that the approval on file was granted
    for exactly the action about to run.

    This is the control that makes an approval un-replayable against
    different arguments: the orchestrator calls this immediately before
    executing a previously-paused step, with the args it actually
    resolved, and a divergence raises rather than executing.
    """
    record = (
        await session.execute(
            select(ApprovalRecord)
            .where(
                ApprovalRecord.task_id == task_id,
                ApprovalRecord.step_index == step_index,
            )
            .order_by(ApprovalRecord.created_at.desc())
        )
    ).scalars().first()

    if record is None:
        raise KeyError(f"no approval record for task '{task_id}' step {step_index}")

    actual_hash = compute_action_hash(skill_name, args)
    if record.action_hash != actual_hash:
        raise ApprovalBindingError(
            (
                f"approval for task '{task_id}' step {step_index} was granted for a different "
                "action/arguments than the ones about to run"
            ),
            expected_hash=record.action_hash,
            actual_hash=actual_hash,
        )
    return record
