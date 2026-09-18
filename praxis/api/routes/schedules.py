# praxis/api/routes/schedules.py
"""Schedule CRUD (brief §9's "users can schedule recurring tasks").

`praxis.agents.schedules` already models what a schedule *is* - cron and
interval triggers, timezones and DST, overlap and missed-run policies,
run deduplication. This module is the surface that lets a user actually
create one, which is what makes the capability real rather than merely
modelled.

Two things are deliberately validated here rather than at fire time:

**The trigger is built on write.** `Schedule.__post_init__` constructs
the APScheduler trigger eagerly, so an invalid cron expression or an
unknown timezone is a 400 on the create call - not a 3am failure on the
first fire, when nobody is watching.

**`next_run_at` is computed and stored on write.** The runner selects
due schedules by that column, so a schedule that never had it computed
would simply never fire, silently. Storing it at creation means a
schedule is due from the moment it exists.

Tenant-scoped throughout, with the same 404-not-403 rule the task and
dashboard routes use: reporting 403 for another tenant's schedule would
confirm it exists.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from praxis.agents.schedules import (
    MissedRunPolicy,
    OverlapPolicy,
    Schedule,
    ScheduleError,
    ScheduleState,
    next_fire_time,
)
from praxis.api.dependencies import authorize_resource_or_404, get_settings, require
from praxis.config import Settings
from praxis.memory.db import PostgresStore
from praxis.memory.models import ScheduleRecord, ScheduleRun
from praxis.security.policy import Permission
from praxis.security.principal import Principal

router = APIRouter(prefix="/schedules", tags=["schedules"])


class CreateScheduleRequest(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    intent_text: str = Field(min_length=1, description="What to run, in natural language")
    interval_seconds: int | None = None
    cron: str | None = Field(
        default=None, description="Restricted cron: minute hour day-of-month month day-of-week"
    )
    timezone_name: str = "UTC"
    overlap: OverlapPolicy = OverlapPolicy.SKIP
    missed_run: MissedRunPolicy = MissedRunPolicy.RUN_ONCE
    connector_name: str | None = None


class UpdateScheduleRequest(BaseModel):
    name: str | None = None
    intent_text: str | None = None
    interval_seconds: int | None = None
    cron: str | None = None
    timezone_name: str | None = None
    state: ScheduleState | None = None
    overlap: OverlapPolicy | None = None
    missed_run: MissedRunPolicy | None = None


def _to_schedule(record: ScheduleRecord) -> Schedule:
    """Rebuilds the domain object from its row.

    Going through `Schedule` rather than reading the row's columns
    directly is what gets the trigger semantics (timezone, DST, cron
    validation) applied consistently everywhere - the row is storage,
    the `Schedule` is the behaviour.
    """
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


def _serialize(record: ScheduleRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "name": record.name,
        "intent_text": record.intent_text,
        "interval_seconds": record.interval_seconds,
        "cron": record.cron,
        "timezone_name": record.timezone_name,
        "state": record.state,
        "overlap": record.overlap_policy,
        "missed_run": record.missed_run_policy,
        "connector_name": record.connector_name,
        "last_run_at": record.last_run_at.isoformat() if record.last_run_at else None,
        "next_run_at": record.next_run_at.isoformat() if record.next_run_at else None,
    }


@router.post("", status_code=201)
async def create_schedule(
    body: CreateScheduleRequest,
    principal: Annotated[Principal, Depends(require(Permission.SCHEDULE_WRITE))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Creates a recurring task, rejecting an unfireable definition."""
    now = datetime.now(UTC)
    try:
        schedule = Schedule(
            schedule_id="pending",
            name=body.name,
            intent_text=body.intent_text,
            tenant_id=principal.tenant_id,
            principal_user_id=principal.user_id,
            interval_seconds=body.interval_seconds,
            cron=body.cron,
            timezone_name=body.timezone_name,
            overlap=body.overlap,
            missed_run=body.missed_run,
            connector_name=body.connector_name,
        )
        first_run = next_fire_time(schedule, now)
    except ScheduleError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            record = ScheduleRecord(
                tenant_id=principal.tenant_id,
                name=body.name,
                intent_text=body.intent_text,
                principal_user_id=principal.user_id,
                interval_seconds=body.interval_seconds,
                cron=body.cron,
                timezone_name=body.timezone_name,
                state=ScheduleState.ACTIVE.value,
                overlap_policy=body.overlap.value,
                missed_run_policy=body.missed_run.value,
                connector_name=body.connector_name,
                next_run_at=first_run,
            )
            session.add(record)
            await session.commit()
            return _serialize(record)
    finally:
        await store.dispose()


@router.get("")
async def list_schedules(
    principal: Annotated[Principal, Depends(require(Permission.SCHEDULE_READ))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """The caller's own tenant's schedules - filtered in the query."""
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            rows = (
                (
                    await session.execute(
                        select(ScheduleRecord)
                        .where(ScheduleRecord.tenant_id == principal.tenant_id)
                        .order_by(ScheduleRecord.next_run_at.asc())
                    )
                )
                .scalars()
                .all()
            )
            return {"schedules": [_serialize(row) for row in rows]}
    finally:
        await store.dispose()


@router.get("/{schedule_id}")
async def get_schedule(
    schedule_id: str,
    principal: Annotated[Principal, Depends(require(Permission.SCHEDULE_READ))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            record = await session.get(ScheduleRecord, schedule_id)
            record = await authorize_resource_or_404(
                principal,
                Permission.SCHEDULE_READ,
                resource_type="schedule",
                resource_id=schedule_id,
                record=record,
            )
            runs = (
                (
                    await session.execute(
                        select(ScheduleRun)
                        .where(ScheduleRun.schedule_id == schedule_id)
                        .order_by(ScheduleRun.scheduled_for.desc())
                        .limit(20)
                    )
                )
                .scalars()
                .all()
            )
            return {
                **_serialize(record),
                "recent_runs": [
                    {
                        "scheduled_for": run.scheduled_for.isoformat(),
                        "status": run.status,
                        "task_id": run.task_id,
                    }
                    for run in runs
                ],
            }
    finally:
        await store.dispose()


@router.patch("/{schedule_id}")
async def update_schedule(
    schedule_id: str,
    body: UpdateScheduleRequest,
    principal: Annotated[Principal, Depends(require(Permission.SCHEDULE_WRITE))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Updates a schedule, re-validating the trigger and recomputing
    `next_run_at` whenever anything that decides when it fires changed.

    Recomputing is not optional: a schedule whose cron was edited but
    whose `next_run_at` still pointed at the old slot would fire on the
    old cadence indefinitely, which reads as the edit having silently
    failed.
    """
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            record = await session.get(ScheduleRecord, schedule_id)
            record = await authorize_resource_or_404(
                principal,
                Permission.SCHEDULE_WRITE,
                resource_type="schedule",
                resource_id=schedule_id,
                record=record,
            )

            fields = body.model_dump(exclude_unset=True)
            timing_changed = bool(
                {"interval_seconds", "cron", "timezone_name", "state"} & fields.keys()
            )

            if "name" in fields:
                record.name = fields["name"]
            if "intent_text" in fields:
                record.intent_text = fields["intent_text"]
            if "interval_seconds" in fields:
                record.interval_seconds = fields["interval_seconds"]
            if "cron" in fields:
                record.cron = fields["cron"]
            if "timezone_name" in fields:
                record.timezone_name = fields["timezone_name"]
            if "state" in fields:
                record.state = ScheduleState(fields["state"]).value
            if "overlap" in fields:
                record.overlap_policy = OverlapPolicy(fields["overlap"]).value
            if "missed_run" in fields:
                record.missed_run_policy = MissedRunPolicy(fields["missed_run"]).value

            if timing_changed:
                try:
                    schedule = _to_schedule(record)
                except ScheduleError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                if schedule.state is ScheduleState.ACTIVE:
                    record.next_run_at = next_fire_time(schedule, datetime.now(UTC))
                else:
                    # A paused schedule has no next run. Leaving a stale
                    # one would make it fire the moment it is resumed,
                    # for a slot that passed while it was paused.
                    record.next_run_at = None

            await session.commit()
            return _serialize(record)
    finally:
        await store.dispose()


@router.delete("/{schedule_id}")
async def delete_schedule(
    schedule_id: str,
    principal: Annotated[Principal, Depends(require(Permission.SCHEDULE_WRITE))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            record = await session.get(ScheduleRecord, schedule_id)
            record = await authorize_resource_or_404(
                principal,
                Permission.SCHEDULE_WRITE,
                resource_type="schedule",
                resource_id=schedule_id,
                record=record,
            )
            await session.delete(record)
            await session.commit()
            return {"id": schedule_id, "deleted": True}
    finally:
        await store.dispose()
