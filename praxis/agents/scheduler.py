# praxis/agents/scheduler.py
"""A thin wrapper around APScheduler (spec §7's "Scheduled agents": "any
task ... can be registered ... via APScheduler"; spec §18: "Scheduled
health scan (APScheduler, e.g. every 5 min) writes health history to
Postgres").

`Scheduler` deliberately wraps `BackgroundScheduler` (thread-based, not
`AsyncIOScheduler`): jobs are registered at `praxis.api.main` *import*
time, before any asyncio event loop uvicorn will actually run even
exists yet - tying job scheduling to "whatever the current event loop
is at import time" would risk silently binding to the wrong loop by the
time uvicorn starts serving. A background thread has no such lifecycle
coupling, mirrors `praxis.cli.init()`'s own `asyncio.run(...)`-per-step
posture for bridging a sync context into async work, and is exactly
`add_interval_job`'s expected shape below: `func` is a plain, callable
sync function bridging to `asyncio.run(...)` for the actual async job
logic itself, e.g. `praxis.api.main`'s `record_health_scan`/
`sweep_stale_approvals`.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from apscheduler.job import Job
from apscheduler.schedulers.background import BackgroundScheduler


class Scheduler:
    """Registers and runs interval jobs against a real APScheduler
    `BackgroundScheduler` - one real instance per process (constructed
    and `start()`-ed once at `praxis.api.main` import time, mirroring
    how `praxis.agents.skill_registry.discover_skills()` and
    `praxis.ingestion.parsers.registry.discover_parsers()` are driven
    from that same module-level setup)."""

    def __init__(self) -> None:
        self._scheduler = BackgroundScheduler()

    def add_interval_job(
        self, func: Callable[[], Any], seconds: int, *, job_id: str | None = None
    ) -> Job:
        """Registers `func` (a plain, zero-arg sync callable) to run
        every `seconds` seconds. Returns the real APScheduler `Job` -
        callers/tests can inspect it directly (`job.trigger`,
        `job.next_run_time`) rather than this wrapper re-exposing every
        APScheduler detail itself."""
        return self._scheduler.add_job(func, "interval", seconds=seconds, id=job_id)

    def get_jobs(self) -> list[Job]:
        return self._scheduler.get_jobs()

    def start(self) -> None:
        if not self._scheduler.running:
            self._scheduler.start()

    def shutdown(self, *, wait: bool = False) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=wait)

    @property
    def running(self) -> bool:
        return self._scheduler.running
