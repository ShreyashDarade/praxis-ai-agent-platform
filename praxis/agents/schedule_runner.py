# praxis/agents/schedule_runner.py
"""The driver that actually fires due schedules (brief §9).

`praxis.agents.schedules` decides *when* a schedule should fire and
*whether* a given slot may start; `praxis.api.routes.schedules` lets a
user define one. This module is the part that wakes up, finds what is
due, and runs it - without which the other two are a definition nobody
executes.

The design is a **polling claim loop**, not an in-process APScheduler
job per schedule. Three reasons, in order of weight:

1. **Survives restart.** Due-ness lives in `schedules.next_run_at`, a
   column. A process that dies mid-slot loses nothing; the next process
   to poll sees the same row still due. In-memory APScheduler jobs
   would vanish with the process, and a weekly report that silently
   stopped after a deploy is exactly the failure this must not have.
2. **Survives horizontal scaling.** Several API processes polling the
   same table is fine, because the claim is a unique insert on
   `schedule_runs.idempotency_key` - two workers racing the same slot
   produce the same key and the loser's insert collides. Deduplication
   is by database constraint, not by a lock anybody has to remember to
   take.
3. **One moving part.** Registering N APScheduler jobs and keeping them
   in step with N rows that a user can edit at any time is a
   synchronization problem; polling a column is not.

APScheduler is still doing the genuinely hard part - `next_fire_time`
delegates to its trigger, so cron semantics, timezones and DST are its
code, not a hand-rolled calendar. This module only decides *that* a row
is due, never *when* it should have been.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from praxis.agents.schedules import (
    MissedRunPolicy,
    OverlapPolicy,
    Schedule,
    ScheduleError,
    ScheduleState,
    missed_fire_times,
    next_fire_time,
    run_idempotency_key,
    should_start_run,
)
from praxis.memory.models import ScheduleRecord, ScheduleRun

_logger = structlog.get_logger(__name__)

# Statuses a `ScheduleRun` row can hold. Strings rather than an Enum
# column so an older process reading a newer status degrades to "some
# status I don't recognise" instead of failing to load the row at all.
RUN_PENDING = "pending"
RUN_RUNNING = "running"
RUN_SUCCEEDED = "succeeded"
RUN_FAILED = "failed"
RUN_SKIPPED = "skipped"

_ACTIVE_STATUSES = (RUN_PENDING, RUN_RUNNING)

# A launcher turns one due slot into a running task and returns its id.
# Injected rather than imported so this module does not depend on the
# Orchestrator: the runner's job is deciding what is due, and a test can
# prove that without standing up an LLM-backed planner.
TaskLauncher = Callable[[Schedule, datetime], Awaitable[str]]


@dataclass
class FiredSlot:
    """One slot this poll decided about."""

    schedule_id: str
    scheduled_for: datetime
    started: bool
    reason: str = ""
    task_id: str | None = None
    error: str = ""


@dataclass
class PollResult:
    """What one `poll_once` did - returned rather than only logged, so a
    caller (and a test) can assert on it."""

    slots: list[FiredSlot] = field(default_factory=list)

    @property
    def started_count(self) -> int:
        return sum(1 for s in self.slots if s.started)

    @property
    def skipped_count(self) -> int:
        return sum(1 for s in self.slots if not s.started)

    def to_dict(self) -> dict[str, Any]:
        return {
            "started": self.started_count,
            "skipped": self.skipped_count,
            "slots": [
                {
                    "schedule_id": s.schedule_id,
                    "scheduled_for": s.scheduled_for.isoformat(),
                    "started": s.started,
                    "reason": s.reason,
                    "task_id": s.task_id,
                    "error": s.error,
                }
                for s in self.slots
            ],
        }


def _record_to_schedule(record: ScheduleRecord) -> Schedule:
    return Schedule(
        schedule_id=record.id,
        name=record.name,
        intent_text=record.intent_text,
        tenant_id=record.tenant_id,
        principal_user_id=record.principal_user_id,
        interval_seconds=record.interval_seconds,
        cron=record.cron,
        timezone_name=record.timezone_name,
        state=ScheduleState(record.state),
        overlap=OverlapPolicy(record.overlap_policy),
        missed_run=MissedRunPolicy(record.missed_run_policy),
        connector_name=record.connector_name,
        last_run_at=record.last_run_at,
        next_run_at=record.next_run_at,
    )


async def _has_active_run(session: Any, schedule_id: str) -> bool:
    row = (
        await session.execute(
            select(ScheduleRun.id)
            .where(ScheduleRun.schedule_id == schedule_id)
            .where(ScheduleRun.status.in_(_ACTIVE_STATUSES))
            .limit(1)
        )
    ).first()
    return row is not None


async def _claim_slot(
    session: Any, schedule: Schedule, scheduled_for: datetime
) -> ScheduleRun | None:
    """Inserts the run row that claims this slot, or returns None if
    another worker already claimed it.

    The claim *is* the insert: `idempotency_key` is unique, so the race
    is resolved by the database rather than by anything this process has
    to coordinate. A nested transaction keeps the collision from
    poisoning the outer session.
    """
    key = run_idempotency_key(schedule.schedule_id, scheduled_for)
    run = ScheduleRun(
        tenant_id=schedule.tenant_id,
        schedule_id=schedule.schedule_id,
        idempotency_key=key,
        scheduled_for=scheduled_for,
        status=RUN_PENDING,
    )
    try:
        async with session.begin_nested():
            session.add(run)
            await session.flush()
    except IntegrityError:
        return None
    return run


async def poll_once(
    session: Any,
    launcher: TaskLauncher,
    *,
    now: datetime | None = None,
    limit: int = 100,
) -> PollResult:
    """Finds every due schedule, fires what should fire, advances the rest.

    Returns a `PollResult` describing each slot and why it did or did
    not start. Commits once at the end: every row this touched moves
    together, so a crash mid-poll leaves slots unclaimed and re-pollable
    rather than half-advanced.
    """
    now = now or datetime.now(UTC)
    result = PollResult()

    due = (
        (
            await session.execute(
                select(ScheduleRecord)
                .where(ScheduleRecord.state == ScheduleState.ACTIVE.value)
                .where(ScheduleRecord.next_run_at.is_not(None))
                .where(ScheduleRecord.next_run_at <= now)
                .order_by(ScheduleRecord.next_run_at.asc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )

    for record in due:
        try:
            schedule = _record_to_schedule(record)
        except (ScheduleError, ValueError) as exc:
            # A row that can no longer produce a valid schedule (an
            # invalid cron written by an older release, a timezone the
            # host no longer knows) is paused rather than retried
            # forever at every poll.
            record.state = ScheduleState.PAUSED.value
            record.next_run_at = None
            _logger.error(
                "pausing unfireable schedule",
                schedule_id=record.id,
                error=str(exc),
            )
            continue

        slots = _slots_to_run(schedule, record, now)

        for scheduled_for in slots:
            start, reason = should_start_run(
                schedule, is_previous_run_active=await _has_active_run(session, record.id)
            )
            if not start:
                result.slots.append(
                    FiredSlot(
                        schedule_id=record.id,
                        scheduled_for=scheduled_for,
                        started=False,
                        reason=reason,
                    )
                )
                continue

            run = await _claim_slot(session, schedule, scheduled_for)
            if run is None:
                result.slots.append(
                    FiredSlot(
                        schedule_id=record.id,
                        scheduled_for=scheduled_for,
                        started=False,
                        reason="slot already claimed by another worker",
                    )
                )
                continue

            try:
                task_id = await launcher(schedule, scheduled_for)
            except Exception as exc:  # noqa: BLE001 - one bad schedule must not stop the poll
                run.status = RUN_FAILED
                run.detail = f"{type(exc).__name__}: {exc}"
                run.finished_at = datetime.now(UTC)
                result.slots.append(
                    FiredSlot(
                        schedule_id=record.id,
                        scheduled_for=scheduled_for,
                        started=False,
                        reason="launch failed",
                        error=run.detail,
                    )
                )
                _logger.error(
                    "schedule launch failed", schedule_id=record.id, error=run.detail
                )
                continue

            run.status = RUN_RUNNING
            run.started_at = datetime.now(UTC)
            run.task_id = task_id
            record.last_run_at = scheduled_for
            result.slots.append(
                FiredSlot(
                    schedule_id=record.id,
                    scheduled_for=scheduled_for,
                    started=True,
                    reason=reason,
                    task_id=task_id,
                )
            )

        # Always advance, even when every slot was skipped: a schedule
        # whose `next_run_at` stayed in the past would be "due" at every
        # poll forever, re-deciding the same skipped slot indefinitely.
        try:
            record.next_run_at = next_fire_time(schedule, now)
        except ScheduleError:
            # A trigger with no future fire time will never be due
            # again. Paused (not cancelled): cancelled reads as a
            # deliberate user action, and this was not one.
            record.state = ScheduleState.PAUSED.value
            record.next_run_at = None
            _logger.info("schedule has no future fire time; pausing", schedule_id=record.id)

    await session.commit()
    return result


def _slots_to_run(
    schedule: Schedule, record: ScheduleRecord, now: datetime
) -> list[datetime]:
    """Which slots this poll should act on for one schedule.

    Normally exactly one: the slot that came due. When the process was
    down long enough that several elapsed, the schedule's
    `MissedRunPolicy` decides - which is what keeps a weekend outage
    from producing 200 catch-up runs on Monday.
    """
    due_at = record.next_run_at
    if due_at is None:
        return []
    if due_at.tzinfo is None:
        due_at = due_at.replace(tzinfo=UTC)

    missed = missed_fire_times(schedule, since=due_at, now=now)
    if not missed:
        return [due_at]
    if schedule.missed_run is MissedRunPolicy.RUN_ONCE:
        # The catch-up helper already collapsed to the latest missed
        # slot; the originally-due one is superseded by it.
        return missed
    return [due_at, *missed]
