# praxis/agents/schedules.py
"""User-defined recurring tasks and the dead-letter queue (Prompt §9).

Two gaps this closes. The scheduler previously ran exactly two
built-in jobs (a health scan and an approval sweep), so the product
brief's own example - *"Compare this month's metrics with last month
and schedule a report every Monday"* - was literally unimplementable.
And there was no dead-letter queue, so a failed background job simply
vanished.

Prompt §9 is specific about what a real scheduler must handle:
*"time-zone/DST rules, overlap handling, missed-run policy, credential
refresh, deduplication, and cancellation."* Each is addressed here:

- **Time zone / DST.** Delegated to APScheduler's `CronTrigger` /
  `IntervalTrigger`, which are already a dependency of this project
  (they back `praxis.agents.scheduler`). Cron parsing and
  DST-correct next-fire-time computation are genuinely hard - "every
  Monday 9am" must stay local 9am across a DST boundary, which a
  naive UTC-offset calculation gets wrong twice a year - and
  APScheduler has solved it properly for over a decade. Hand-rolling
  a cron parser here was the wrong instinct and was removed; this
  module owns only the policy APScheduler does *not* provide
  (overlap, missed-run, dedup, credential handling).
- **Overlap.** `OverlapPolicy` decides what happens when a run is
  still going at the next fire time: skip, queue, or run concurrently.
  Defaulting to `SKIP` is the safe choice - a slow weekly report
  should not stack up N copies of itself.
- **Missed runs.** After downtime, `MissedRunPolicy` decides whether
  to fire once for the gap, fire for every missed slot, or skip.
  `RUN_ONCE` is the default because firing 200 catch-up runs after a
  weekend outage is almost never what anyone wants.
- **Deduplication.** Each run gets a deterministic idempotency key
  from `(schedule_id, scheduled_for)`, so two workers racing on the
  same slot cannot both execute it.
- **Cancellation.** A schedule is paused/deleted by state, and an
  in-flight run is cancelled through the orchestrator's own
  cancellation path.
- **Credential refresh** is deliberately NOT solved by storing
  credentials on the schedule: a run executes as a stored principal
  reference, and credentials are resolved fresh at fire time from the
  live store - so a revoked key stops the schedule at its next run
  rather than continuing on a stale copy.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger


class ScheduleState(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class OverlapPolicy(str, Enum):
    """What to do when the previous run is still in flight."""

    SKIP = "skip"
    QUEUE = "queue"
    CONCURRENT = "concurrent"


class MissedRunPolicy(str, Enum):
    """What to do about fire times that elapsed while nothing was running."""

    RUN_ONCE = "run_once"
    RUN_ALL = "run_all"
    SKIP = "skip"


class ScheduleError(Exception):
    """An invalid schedule definition."""


@dataclass
class Schedule:
    """A recurring task definition.

    Deliberately stores `intent_text` rather than a pre-built plan: the
    plan should be produced fresh at each fire time, because the data,
    the available skills, and the connectors may all have changed
    since the schedule was created. Freezing a plan would make a
    weekly report quietly go stale in a way nobody would notice.
    """

    schedule_id: str
    name: str
    intent_text: str
    tenant_id: str
    # Which stored user this runs as. A reference, not a credential -
    # see the module docstring on credential refresh.
    principal_user_id: str | None = None
    interval_seconds: int | None = None
    # Restricted cron: minute hour day-of-month month day-of-week.
    cron: str | None = None
    timezone_name: str = "UTC"
    state: ScheduleState = ScheduleState.ACTIVE
    overlap: OverlapPolicy = OverlapPolicy.SKIP
    missed_run: MissedRunPolicy = MissedRunPolicy.RUN_ONCE
    connector_name: str | None = None
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.interval_seconds is None and self.cron is None:
            raise ScheduleError("a schedule needs either interval_seconds or cron")
        if self.interval_seconds is not None and self.cron is not None:
            raise ScheduleError("a schedule cannot have both interval_seconds and cron")
        if self.interval_seconds is not None and self.interval_seconds < 1:
            raise ScheduleError("interval_seconds must be >= 1")
        try:
            ZoneInfo(self.timezone_name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ScheduleError(f"unknown timezone '{self.timezone_name}'") from exc
        # Build the trigger eagerly so an invalid cron expression is
        # rejected at definition time rather than at 3am on the first
        # fire, when nobody is watching.
        self.trigger()

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_name)

    def trigger(self) -> CronTrigger | IntervalTrigger:
        """The APScheduler trigger this schedule is defined by.

        Constructed on demand rather than stored: triggers are cheap,
        and keeping the dataclass free of a non-serializable field
        means a `Schedule` still round-trips cleanly through
        `to_dict`/persistence.
        """
        if self.interval_seconds is not None:
            return IntervalTrigger(seconds=self.interval_seconds, timezone=self.zone)
        try:
            return CronTrigger.from_crontab(self.cron or "", timezone=self.zone)
        except ValueError as exc:
            raise ScheduleError(f"invalid cron expression {self.cron!r}: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "name": self.name,
            "intent_text": self.intent_text,
            "tenant_id": self.tenant_id,
            "principal_user_id": self.principal_user_id,
            "interval_seconds": self.interval_seconds,
            "cron": self.cron,
            "timezone": self.timezone_name,
            "state": self.state.value,
            "overlap": self.overlap.value,
            "missed_run": self.missed_run.value,
            "connector_name": self.connector_name,
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
        }


def next_fire_time(schedule: Schedule, after: datetime) -> datetime:
    """The next time `schedule` should fire, strictly after `after`.

    Delegates entirely to APScheduler's trigger, which computes the
    next fire time in the schedule's own timezone and handles DST
    transitions correctly - so "every Monday 9am" stays local 9am on
    both sides of a clock change, rather than drifting by an hour
    twice a year as a fixed-offset calculation would.
    """
    if after.tzinfo is None:
        after = after.replace(tzinfo=UTC)

    fire_time = schedule.trigger().get_next_fire_time(None, after)
    if fire_time is None:
        raise ScheduleError(
            f"schedule '{schedule.name}' has no future fire time"
        )
    return fire_time.astimezone(UTC)


def missed_fire_times(
    schedule: Schedule, *, since: datetime, now: datetime
) -> list[datetime]:
    """Fire times that elapsed while nothing was running.

    Applies the schedule's `MissedRunPolicy`, which is what stops a
    weekend outage turning into 200 catch-up runs on Monday morning.
    """
    if schedule.missed_run is MissedRunPolicy.SKIP:
        return []

    missed: list[datetime] = []
    cursor = since
    # Bounded so a very old `since` cannot produce an unbounded list
    # even under RUN_ALL.
    for _ in range(1000):
        cursor = next_fire_time(schedule, cursor)
        if cursor > now:
            break
        missed.append(cursor)

    if not missed:
        return []
    if schedule.missed_run is MissedRunPolicy.RUN_ONCE:
        return [missed[-1]]
    return missed


def run_idempotency_key(schedule_id: str, scheduled_for: datetime) -> str:
    """A deterministic key for one (schedule, slot) pair.

    Two workers racing on the same slot compute the same key, so the
    second one's insert collides rather than executing a duplicate
    run - deduplication by construction rather than by locking.
    """
    stamp = scheduled_for.astimezone(UTC).replace(microsecond=0).isoformat()
    return hashlib.sha256(f"{schedule_id}:{stamp}".encode()).hexdigest()[:48]


def should_start_run(
    schedule: Schedule, *, is_previous_run_active: bool
) -> tuple[bool, str]:
    """Applies the overlap policy. Returns `(start, reason)`.

    A reason string either way, because "the weekly report did not run
    this week" needs an answer an operator can read.
    """
    if schedule.state is not ScheduleState.ACTIVE:
        return False, f"schedule is {schedule.state.value}"
    if not is_previous_run_active:
        return True, "no previous run in flight"
    if schedule.overlap is OverlapPolicy.CONCURRENT:
        return True, "overlap policy permits concurrent runs"
    if schedule.overlap is OverlapPolicy.QUEUE:
        return False, "previous run still active; this slot is queued"
    return False, "previous run still active; this slot is skipped"
