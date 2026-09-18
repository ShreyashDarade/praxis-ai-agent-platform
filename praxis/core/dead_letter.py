# praxis/core/dead_letter.py
"""The dead-letter queue for failed background work (Prompt §9).

Before this, a failed scheduled job logged an exception and vanished.
That is acceptable for a health scan (the next one runs in five
minutes) and unacceptable for anything the user asked for: a Monday
report that failed should be visible and retryable on Tuesday, not
silently absent.

**Why a DLQ rather than just retrying in place.** Retry handles
transient failure; a DLQ handles the case where retry has been
exhausted and a human needs to know. Conflating them produces either
infinite retries of a permanently-broken job, or silent loss of one
that merely needed a credential refreshed. So: bounded retries with
exponential backoff, and then the entry parks in the queue with its
full failure history until someone replays or discards it.

Entries record the *whole* failure history rather than only the last
error - "it failed three times with a timeout, then once with an auth
error" is a materially different diagnosis from "it failed with an
auth error", and only the history distinguishes them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import structlog
from sqlalchemy import select

from praxis.memory.db import PostgresStore
from praxis.memory.models import DeadLetterEntry
from praxis.observability.metrics import DLQ_DEPTH, metrics

_logger = structlog.get_logger(__name__)


class DeadLetterState(str, Enum):
    PENDING_RETRY = "pending_retry"
    DEAD = "dead"
    REPLAYED = "replayed"
    DISCARDED = "discarded"


# Exponential backoff, capped. Uncapped doubling reaches absurd delays
# within a dozen attempts; capping at an hour keeps a recovering
# dependency from waiting days for the next attempt.
DEFAULT_MAX_ATTEMPTS = 3
_BASE_BACKOFF_SECONDS = 30
_MAX_BACKOFF_SECONDS = 3600


def backoff_delay(attempt: int) -> int:
    """Seconds to wait before attempt number `attempt` (1-based)."""
    return min(_BASE_BACKOFF_SECONDS * (2 ** max(0, attempt - 1)), _MAX_BACKOFF_SECONDS)


@dataclass
class FailureRecord:
    """One failed attempt."""

    attempt: int
    error: str
    error_type: str
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "error": self.error[:2000],
            "error_type": self.error_type,
            "occurred_at": self.occurred_at.isoformat(),
        }


class DeadLetterQueue:
    """Records, retries, and replays failed background work."""

    def __init__(
        self, store: PostgresStore, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS
    ) -> None:
        self._store = store
        self._max_attempts = max_attempts

    async def record_failure(
        self,
        *,
        tenant_id: str,
        job_type: str,
        job_key: str,
        payload: dict[str, Any],
        error: Exception,
    ) -> DeadLetterEntry:
        """Records a failure, scheduling a retry or parking the entry.

        Idempotent per `job_key`: the same failing job accumulates
        attempts on one entry rather than creating a new row each
        time, which is what makes the failure history readable.
        """
        now = datetime.now(UTC)
        failure = FailureRecord(
            attempt=1, error=str(error), error_type=type(error).__name__
        )

        async with self._store.session() as session:
            entry = (
                await session.execute(
                    select(DeadLetterEntry).where(
                        DeadLetterEntry.tenant_id == tenant_id,
                        DeadLetterEntry.job_key == job_key,
                        DeadLetterEntry.state.in_(
                            [DeadLetterState.PENDING_RETRY.value, DeadLetterState.DEAD.value]
                        ),
                    )
                )
            ).scalar_one_or_none()

            if entry is None:
                entry = DeadLetterEntry(
                    tenant_id=tenant_id,
                    job_type=job_type,
                    job_key=job_key,
                    payload=payload,
                    attempts=1,
                    failures=[failure.to_dict()],
                    state=DeadLetterState.PENDING_RETRY.value,
                    next_retry_at=now + timedelta(seconds=backoff_delay(1)),
                )
                session.add(entry)
            else:
                entry.attempts += 1
                failure = FailureRecord(
                    attempt=entry.attempts,
                    error=str(error),
                    error_type=type(error).__name__,
                )
                # Reassigned rather than appended in place: SQLAlchemy
                # does not track in-place mutation of a plain JSON
                # column (the same reason `Task.checklist` is
                # reassigned in the orchestrator).
                entry.failures = [*entry.failures, failure.to_dict()]
                if entry.attempts >= self._max_attempts:
                    entry.state = DeadLetterState.DEAD.value
                    entry.next_retry_at = None
                else:
                    entry.next_retry_at = now + timedelta(
                        seconds=backoff_delay(entry.attempts)
                    )
            await session.commit()

        _logger.warning(
            "dead_letter_recorded",
            job_type=job_type,
            job_key=job_key,
            attempts=entry.attempts,
            state=entry.state,
            error=str(error),
        )
        metrics.increment(DLQ_DEPTH, labels={"job_type": job_type})
        return entry

    async def due_for_retry(self, *, tenant_id: str | None = None) -> list[DeadLetterEntry]:
        """Entries whose backoff has elapsed."""
        now = datetime.now(UTC)
        async with self._store.session() as session:
            stmt = select(DeadLetterEntry).where(
                DeadLetterEntry.state == DeadLetterState.PENDING_RETRY.value,
                DeadLetterEntry.next_retry_at <= now,
            )
            if tenant_id is not None:
                stmt = stmt.where(DeadLetterEntry.tenant_id == tenant_id)
            return list((await session.execute(stmt)).scalars().all())

    async def dead_entries(self, *, tenant_id: str) -> list[DeadLetterEntry]:
        """Entries that exhausted their retries and need a human."""
        async with self._store.session() as session:
            return list(
                (
                    await session.execute(
                        select(DeadLetterEntry)
                        .where(
                            DeadLetterEntry.tenant_id == tenant_id,
                            DeadLetterEntry.state == DeadLetterState.DEAD.value,
                        )
                        .order_by(DeadLetterEntry.updated_at.desc())
                    )
                ).scalars().all()
            )

    async def mark_replayed(self, entry_id: str) -> None:
        """Marks an entry successfully replayed.

        Kept rather than deleted: the record that something failed
        four times and then succeeded is exactly the operational
        history worth retaining.
        """
        await self._set_state(entry_id, DeadLetterState.REPLAYED)

    async def discard(self, entry_id: str) -> None:
        """Abandons an entry deliberately, as a recorded decision."""
        await self._set_state(entry_id, DeadLetterState.DISCARDED)

    async def _set_state(self, entry_id: str, state: DeadLetterState) -> None:
        async with self._store.session() as session:
            entry = await session.get(DeadLetterEntry, entry_id)
            if entry is None:
                raise KeyError(f"no dead-letter entry with id '{entry_id}'")
            entry.state = state.value
            entry.next_retry_at = None
            await session.commit()
        _logger.info("dead_letter_state_changed", entry_id=entry_id, state=state.value)

    async def depth(self, *, tenant_id: str) -> dict[str, int]:
        """How much is waiting, by state - the number an operator
        actually watches."""
        async with self._store.session() as session:
            rows = (
                await session.execute(
                    select(DeadLetterEntry.state).where(
                        DeadLetterEntry.tenant_id == tenant_id
                    )
                )
            ).scalars().all()
        counts: dict[str, int] = {}
        for state in rows:
            counts[state] = counts.get(state, 0) + 1
        return counts
