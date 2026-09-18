"""FastAPI application entrypoint (spec §14).

This module owns the app's shared wiring - observability config, parser/
skill discovery, the embedder, health-check registration/aggregation,
the connector registry, the scheduler, and the `Orchestrator` singleton
- everything `praxis/api/routes/*.py`'s actual route handlers are built
on top of (each imports `from praxis.api import main` and calls back
into the functions/singletons defined here, e.g. `main._get_orchestrator()`,
`main.run_health_checks()`). Splitting the route handlers themselves out
into `praxis/api/routes/` keeps this file focused on that shared wiring
instead of growing one handler per endpoint forever; nothing about the
module-level singletons below changed in that split - `main._orchestrator`/
`main._connector_registry`/`main.HEALTH_CHECKS` etc. are still the exact
attributes tests reach into directly.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
from docker.errors import DockerException
from fastapi import FastAPI
from sqlalchemy import select

from praxis.agents import procedure_registry, skill_registry
from praxis.agents.capability_factory import CapabilityFactory
from praxis.agents.conversation import ConversationService
from praxis.agents.planner import Planner
from praxis.agents.schedule_runner import poll_once
from praxis.agents.scheduler import Scheduler
from praxis.agents.subagent import discover_agents
from praxis.config import Settings
from praxis.connectors.bootstrap import build_registry
from praxis.connectors.registry import ConnectorRegistry
from praxis.connectors.sql.sql_connector import SQLConnector
from praxis.core.dead_letter import DeadLetterQueue
from praxis.core.exceptions import ApprovalTimeoutError
from praxis.core.interfaces import HealthStatus
from praxis.core.orchestrator import Orchestrator
from praxis.core.completion import CompletionGate
from praxis.agents.critic import Critic
from praxis.ingestion.embedders.sentence_transformer_embedder import get_default_embedder
from praxis.ingestion.parsers import registry as parser_registry
from praxis.llm.catalogue import LLMCatalogue
from praxis.llm.prompt_manager import PromptManager
from praxis.memory.db import PostgresStore
from praxis.memory.graph_store import PgGraphStore
from praxis.memory.models import HealthRecord, Task
from praxis.observability.logging import configure_logging
from praxis.observability.tracing import configure_tracing
from praxis.sandbox.executor import DockerSandboxExecutor
from praxis.security.principal import Principal
from praxis.security.provisioning import ensure_default_tenant

# Observability (spec §10) - configured once, at import time, before
# anything else in this module can possibly log or open a span; both
# are idempotent, mirroring parser_registry.discover_parsers()'s own
# "safe to call repeatedly" posture below.
configure_logging()
configure_tracing()
_logger = structlog.get_logger(__name__)

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "agents" / "skills"

app = FastAPI(title="Praxis")

# Trigger self-registration of every built-in parser (text/document/
# image-OCR/tabular) exactly once at import time - mirrors how
# `_build_connector_registry()` below drives connector self-registration
# via `build_registry`. Safe to call repeatedly (idempotent).
parser_registry.discover_parsers()

# Same discovery pattern, for the hand-written skills this phase ships
# (spec §7; praxis/agents/skills/) - one new file with a
# `register_skill(...)` call is how a new skill gets added, not an edit
# here.
skill_registry.discover_skills()

# Declarative capabilities: every `skills/**/SKILL.md` that composes
# already-registered tools becomes a runnable skill. Loaded AFTER the
# Python skills above, deliberately - a procedure can only compose
# tools that already exist, so the things it composes have to be
# registered first. A malformed one is logged and skipped rather
# than taking startup down.
_loaded_procedures = procedure_registry.load_from_disk()
if _loaded_procedures:
    _logger.info("declarative_skills_loaded", skills=_loaded_procedures)

# Same discovery pattern for specialist sub-agents (Phase 15): one new
# file in `praxis/agents/specialists/` with a `register_agent(...)`
# call is how a specialist is added, with no edit here.
discover_agents()

# Loading the sentence-transformers model is expensive (first use may
# download weights) - force it to load once here, at module level,
# rather than on the first request. `get_default_embedder()` (not a
# fresh `SentenceTransformerEmbedder()`) so this process's one loaded
# model is shared with `praxis.agents.skills.retrieve_documents`
# (Phase 5) instead of each loading its own copy. `praxis.api.routes.
# attachments` reaches this same instance via `main._embedder`.
_embedder = get_default_embedder()

# Later phases append connector/sandbox checks here - OCP, mirrors cli.py's INIT_STEPS.
HealthCheck = Callable[[], Awaitable[HealthStatus]]
HEALTH_CHECKS: list[HealthCheck] = []


def register_health_check(check: HealthCheck) -> None:
    HEALTH_CHECKS.append(check)


async def _database_check() -> HealthStatus:
    settings = Settings()
    store = PostgresStore(settings)
    status = await store.health()
    await store.dispose()
    return status


register_health_check(_database_check)


def _build_connector_registry() -> ConnectorRegistry:
    # Constructing Settings() can fail (database_url is required) if the
    # process env / .env isn't configured yet - e.g. during test
    # collection, before any test's fixture has set env vars. Connector
    # registration must not take the whole API module import down over
    # that; an empty registry is exactly the "degrades gracefully"
    # posture spec §6 asks for.
    try:
        settings = Settings()
    except Exception:  # noqa: BLE001
        return ConnectorRegistry()
    registry = build_registry(settings)

    # Phase 11 (spec §16.2's "customer-db"): a narrow, explicitly
    # demo-labeled exception to `SQLConnector`'s own "no global DSN"
    # design principle (see its docstring) - registers the one demo
    # connector this phase's dashboard walkthrough needs, by the exact
    # name (`"customer-db"`) spec §16.2's own example `/intent` payload
    # names, only when this one demo-only setting is actually
    # configured; skipped (not failed) otherwise, exactly like every
    # other optional connector above.
    if settings.demo_customer_db_dsn:
        registry.register(
            SQLConnector(dsn=settings.demo_customer_db_dsn, name="customer-db", read_only=True)
        )
    return registry


# Connector *membership* is fixed at process startup (deployment config,
# spec §19 step 4) - unlike _database_check above, which re-reads
# Settings() per request because DB reachability is a live condition.
# Each connector's own health() is still re-evaluated on every /health
# call, via the closures all_health_checks() returns.
#
# Phase 11: retained as a proper module-level singleton (not discarded
# after registering health checks, as before this phase) - `_get_orchestrator`
# below hands this exact registry to the Orchestrator so `connector_name`
# resolution (spec §16.2) reaches the very same connectors `/health`
# already reports on, never a second, independently-built registry.
_connector_registry: ConnectorRegistry = _build_connector_registry()

for _connector_health_check in _connector_registry.all_health_checks():
    register_health_check(_connector_health_check)


async def run_health_checks() -> list[HealthStatus]:
    """The one, real health-check aggregation - runs every registered
    `HEALTH_CHECKS` entry, isolating a check that raises rather than
    letting it take down the whole aggregation. Shared by `GET /health`
    (`praxis.api.routes.health`) and the scheduled health scan
    (`record_health_scan` below, spec §18) - deliberately the *same*
    function, not two copies of this loop, so the on-demand and the
    scheduled view of "what's healthy" can never silently drift apart
    (spec §10: "`GET /health` plus the `HealthMonitor` ... exposed both
    on-demand and on schedule").
    """
    results = []
    for check in HEALTH_CHECKS:
        try:
            results.append(await check())
        except Exception as exc:  # noqa: BLE001 - one bad check must not take down /health
            results.append(HealthStatus(name=check.__name__, healthy=False, detail=str(exc)))
    return results


# ------------------------------------------------------------------ #
# Scheduled agents (spec §7, §10, §12, §18): a real health scan and a
# real approval-timeout sweep, both registered with
# `praxis.agents.scheduler.Scheduler` below, at this module's own
# import time - mirroring `parser_registry.discover_parsers()` and
# `skill_registry.discover_skills()`'s own "wired in at module-level
# setup" posture.
# ------------------------------------------------------------------ #


async def record_health_scan() -> list[HealthStatus]:
    """The §18 scheduled health scan: runs `run_health_checks()` - the
    exact same aggregation `GET /health` uses, never a duplicate - then
    writes one `HealthRecord` row per component to the real database,
    which is what a future Observability view (§10) reads as health
    history. Returns the results too, so a test (or any other caller
    wanting the outcome without waiting on the scheduler) can call this
    directly and inspect exactly what happened.
    """
    results = await run_health_checks()

    settings = Settings()
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            for result in results:
                session.add(
                    HealthRecord(
                        component=result.name, healthy=result.healthy, detail=result.detail
                    )
                )
            await session.commit()
    finally:
        await store.dispose()

    _logger.info(
        "health_scan_completed",
        components=len(results),
        healthy=all(r.healthy for r in results),
    )
    return results


async def sweep_stale_approvals() -> list[str]:
    """The approval-timeout sweep (spec §12's `ApprovalTimeoutError`):
    finds every `Task` still `awaiting_approval`/`awaiting_clarification`
    whose `updated_at` is older than `Settings.approval_timeout_seconds`,
    marks each one `failed` with the error's detail recorded in
    `Task.result` - surfaced to the user via `GET /tasks/{id}`, never
    silently left paused forever (spec §12) - and logs it. Returns the
    swept task ids, so a test (or the sync job wrapper below) can
    inspect the real outcome.
    """
    settings = Settings()
    store = PostgresStore(settings)
    cutoff = datetime.now(UTC) - timedelta(seconds=settings.approval_timeout_seconds)
    swept: list[str] = []
    try:
        async with store.session() as session:
            stmt = select(Task).where(
                Task.status.in_(["awaiting_approval", "awaiting_clarification"]),
                Task.updated_at < cutoff,
            )
            stale_tasks = (await session.execute(stmt)).scalars().all()

            for task in stale_tasks:
                waited_seconds = (datetime.now(UTC) - task.updated_at).total_seconds()
                error = ApprovalTimeoutError(
                    f"task '{task.id}' was left '{task.status}' past the "
                    f"{settings.approval_timeout_seconds}s approval timeout",
                    task_id=task.id,
                    waited_seconds=waited_seconds,
                    detail=f"last updated at {task.updated_at.isoformat()}",
                )
                task.status = "failed"
                task.pending_input = None
                task.result = {"error": str(error)}
                swept.append(task.id)
                _logger.warning(
                    "approval_timeout", task_id=task.id, waited_seconds=waited_seconds
                )

            await session.commit()
    finally:
        await store.dispose()

    return swept


def _health_scan_job() -> None:
    # A plain sync callable APScheduler's BackgroundScheduler runs in its
    # own worker thread - bridges to the real async job via a fresh
    # asyncio.run(...), exactly like praxis.cli.init()'s own steps do for
    # the same reason. Any failure is logged, never left to crash the
    # scheduler's worker thread and silently take every future run of
    # every other job down with it.
    try:
        asyncio.run(record_health_scan())
    except Exception:  # noqa: BLE001
        _logger.exception("health_scan_job_failed")


def _approval_timeout_sweep_job() -> None:
    try:
        asyncio.run(sweep_stale_approvals())
    except Exception:  # noqa: BLE001
        _logger.exception("approval_timeout_sweep_job_failed")


async def run_due_schedules() -> dict[str, int]:
    """Fires every schedule whose slot has come due.

    The gap this closes: `POST /schedules` persisted a recurring task
    and `poll_once` knew how to find due ones, but nothing in the
    shipped application ever called it. A user could create a weekly
    report and it would simply never run - persistence without a worker
    is a configuration screen, not a feature.

    The launcher is supplied here rather than inside the poller because
    only the application knows how to start real work. It runs each due
    schedule as its own task under the schedule's own stored identity,
    so a recurring job executes with the tenant that created it and not
    with system rights.

    Failures go to the dead-letter queue rather than a log line: a
    schedule fires with nobody watching, so "the Monday report did not
    run" has to be discoverable on Tuesday and retryable.
    """
    settings = Settings()
    store = PostgresStore(settings)
    orchestrator = _get_orchestrator()
    dead_letters = DeadLetterQueue(store)

    async def _launch(schedule: Any, scheduled_for: datetime) -> str:
        # The schedule's own principal, reconstructed from what was
        # stored when it was created. Deliberately NOT the system
        # principal: a recurring task must not quietly acquire more
        # rights than the person who scheduled it.
        principal = Principal(
            tenant_id=schedule.tenant_id,
            user_id=schedule.principal_user_id,
            email="",
            roles=("analyst",),
        )
        return await orchestrator.start_task(
            schedule.intent_text,
            connector_name=schedule.connector_name,
            principal=principal,
        )

    try:
        async with store.session() as session:
            result = await poll_once(session, _launch, dead_letters=dead_letters)
    finally:
        await store.dispose()

    if result.started_count:
        _logger.info(
            "schedules_fired",
            started=result.started_count,
            skipped=result.skipped_count,
        )
    return {"started": result.started_count, "skipped": result.skipped_count}


def _schedule_poll_job() -> None:
    # Same sync-callable bridge as the health scan above, and the same
    # reason for swallowing: a failure here must not take the
    # scheduler's worker thread down and silently stop every other job.
    try:
        asyncio.run(run_due_schedules())
    except Exception:  # noqa: BLE001
        _logger.exception("schedule_poll_job_failed")


def _build_scheduler() -> Scheduler | None:
    """Genuinely optional, degrades gracefully - same posture as
    `_build_connector_registry` above: `Settings()` can fail (e.g.
    during test collection, before any fixture has set env vars) and
    that must not take this whole module's import down.
    """
    try:
        settings = Settings()
    except Exception:  # noqa: BLE001
        return None

    scheduler = Scheduler()
    scheduler.add_interval_job(
        _health_scan_job, settings.health_scan_interval_seconds, job_id="health_scan"
    )
    scheduler.add_interval_job(
        _approval_timeout_sweep_job,
        settings.approval_timeout_seconds,
        job_id="approval_timeout_sweep",
    )
    scheduler.add_interval_job(
        _schedule_poll_job,
        settings.schedule_poll_interval_seconds,
        job_id="schedule_poll",
    )
    return scheduler


# Built at import (so the jobs are inspectable) but deliberately NOT
# started here - `_lifespan` starts it when the application actually
# runs.
#
# Starting background work at import time was a real defect, not a
# style preference: the schedule poller queries the database every
# minute, and importing this module for any reason - a CLI command, a
# test that only wanted the app object - silently began doing that.
# Against a test database being created and dropped between cases, a
# poller running in a worker thread produces failures that move around
# and vanish when the same tests are run alone, which is the most
# expensive kind of bug to chase.
_scheduler = _build_scheduler()


# Orchestrator wiring (spec §3, §14). Unlike a per-request `PostgresStore`
# (see `praxis.api.routes.attachments.upload_attachment`), this is a true
# module-level singleton, constructed lazily (not at import time, for the
# same "Settings() may not be configured yet" reason `_build_connector_registry`
# defers) but memoized thereafter: `Orchestrator` holds in-memory execution-
# graph state (`praxis.core.orchestrator._TaskState`) across requests -
# a task pausing to `awaiting_approval` on one request and being
# resumed by a later `POST /tasks/{id}/approve` request must reach the
# *same* Orchestrator instance, not a fresh one per call.
_orchestrator: Orchestrator | None = None


def _build_capability_factory(settings: Settings, store: PostgresStore) -> CapabilityFactory | None:
    """Genuinely optional, degrades gracefully - same posture as
    `_build_connector_registry` above and `praxis.connectors.bootstrap`
    (spec §6's "degrades gracefully" applied to §19 step 5's sandbox
    check): a `DockerSandboxExecutor` needs a real, reachable Docker
    daemon, which isn't guaranteed in every deployment. When it isn't
    reachable, the Orchestrator simply gets no `CapabilityFactory` and
    falls back to its pre-Phase-6 behavior (an unregistered skill fails
    the task immediately with a clear message) rather than this whole
    endpoint failing to construct.
    """
    try:
        sandbox = DockerSandboxExecutor(docker_host=settings.docker_host)
    except DockerException:
        return None
    return CapabilityFactory(
        LLMCatalogue(),
        PromptManager(),
        sandbox,
        PgGraphStore(store),
        store,
        _SKILLS_DIR,
        sandbox_timeout_seconds=settings.sandbox_timeout_seconds,
    )


def _get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        settings = Settings()
        store = PostgresStore(settings)
        planner = Planner(LLMCatalogue(), PromptManager())
        capability_factory = _build_capability_factory(settings, store)
        # Phase 11 (spec §16.1 step 8's "the lineage graph records every
        # skill/tool used") - constructed unconditionally, unlike
        # `capability_factory` above, since lineage recording has no
        # Docker dependency to degrade around.
        graph_store = PgGraphStore(store)
        _orchestrator = Orchestrator(
            store,
            planner,
            skill_registry,
            capability_factory,
            connector_registry=_connector_registry,
            graph_store=graph_store,
            completion_gate=_completion_gate(),
        )
    return _orchestrator


def _completion_gate() -> CompletionGate:
    """The gate that decides whether a finished task answered its intent.

    With `verify_completion` on, a reviewing `Critic` judges the result
    against the intent through a real model; off, only the gate's
    deterministic checks run. Built here rather than defaulted inside
    the Orchestrator so the decision to spend a model call per task is
    a deployment's, made once, and never a test double's.
    """
    if not Settings().verify_completion:
        return CompletionGate()
    return CompletionGate(Critic(LLMCatalogue(), PromptManager()))


_conversation_service: ConversationService | None = None


def get_conversation_service() -> ConversationService:
    """The shared chat service, built over the same Orchestrator.

    One instance per process because it holds the in-flight turn set:
    a per-request service would let Python garbage-collect a running
    turn the moment the request that started it returned, which is the
    exact failure `asyncio.create_task` has without a retained handle.
    """
    global _conversation_service
    if _conversation_service is None:
        from praxis.agents.conversation import ConversationService

        _conversation_service = ConversationService(
            PostgresStore(Settings()),
            _get_orchestrator(),
            catalogue=LLMCatalogue(),
            prompt_manager=PromptManager(),
        )
    return _conversation_service


# Route modules are imported (and their routers included) last, after
# every singleton/helper function above is defined - each of
# `praxis.api.routes.{health,attachments,tasks}` does `from praxis.api
# import main` and calls back into the names above (e.g.
# `main._get_orchestrator()`, `main.run_health_checks()`) from inside its
# own handler functions, never at its own import time, so the exact
# order of these two statements relative to each other doesn't matter -
# but living here, at the bottom, keeps this file reading top-to-bottom
# as "build the shared wiring, then mount the routes on top of it".
from praxis.api.routes import (  # noqa: E402
    admin,
    agents,
    attachments,
    conversations,
    dashboards,
    health,
    investigations,
    memory,
    schedules,
    semantic,
    skills,
    tasks,
)

app.include_router(health.router)
app.include_router(attachments.router)
app.include_router(tasks.router)
app.include_router(admin.router)
app.include_router(dashboards.router)
app.include_router(schedules.router)
app.include_router(conversations.router)
app.include_router(skills.router)
app.include_router(agents.router)
app.include_router(memory.router)
app.include_router(semantic.router)
app.include_router(investigations.router)


async def _bootstrap_default_tenant() -> None:
    """Guarantees `DEFAULT_TENANT_ID` resolves to a real tenant row.

    The Alembic migration inserts it, but a database built by
    `Base.metadata.create_all` (every test, and any quick local
    scratch DB) has no migration history - this makes both paths
    converge on the same invariant: there is always a real default
    tenant, never a dangling tenant id.

    Degrades gracefully, exactly like `_build_connector_registry` and
    `_build_scheduler` above: an unconfigured/unreachable database must
    not prevent the app from starting.
    """
    try:
        settings = Settings()
    except Exception:  # noqa: BLE001
        return
    store = PostgresStore(settings)
    try:
        await ensure_default_tenant(store)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("default_tenant_bootstrap_skipped", error=str(exc))
    finally:
        await store.dispose()


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hooks (the modern replacement for `on_event`)."""
    await _bootstrap_default_tenant()
    if _scheduler is not None:
        # Started here rather than at import: background jobs belong to
        # a running application, not to the act of importing a module.
        _scheduler.start()
    try:
        yield
    finally:
        if _scheduler is not None:
            _scheduler.shutdown()


# Assigned after the fact rather than passed to `FastAPI(...)` above,
# because the routers - and the singletons they close over - are built
# between that constructor call and here; keeping the constructor at
# the top of the module is what makes `app` importable by the route
# modules in the first place.
app.router.lifespan_context = _lifespan
