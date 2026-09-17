"""FastAPI application entrypoint (spec §14)."""
from __future__ import annotations

import asyncio
import mimetypes
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import structlog
from docker.errors import DockerException
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from sqlalchemy import select

from praxis.agents import skill_registry
from praxis.agents.capability_factory import CapabilityFactory
from praxis.agents.planner import Planner
from praxis.agents.scheduler import Scheduler
from praxis.config import Settings
from praxis.connectors.bootstrap import build_registry
from praxis.connectors.registry import ConnectorRegistry
from praxis.connectors.sql.connector import SQLConnector
from praxis.core.events import task_event_bus, task_state_snapshot
from praxis.core.exceptions import ApprovalTimeoutError
from praxis.core.interfaces import HealthStatus
from praxis.core.orchestrator import Orchestrator
from praxis.ingestion.embedders.sentence_transformer_embedder import get_default_embedder
from praxis.ingestion.parsers import registry as parser_registry
from praxis.ingestion.pipeline import ingest
from praxis.llm.catalogue import LLMCatalogue
from praxis.llm.prompt_manager import PromptManager
from praxis.memory.blob_store import LocalBlobStore
from praxis.memory.db import PostgresStore
from praxis.memory.graph_store import PgGraphStore
from praxis.memory.models import HealthRecord, Task
from praxis.memory.vector_store import PgVectorStore
from praxis.observability.logging import configure_logging
from praxis.observability.tracing import configure_tracing
from praxis.sandbox.executor import DockerSandboxExecutor

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

# Same discovery pattern, for the two hand-written skills this phase
# ships (spec §7; praxis/agents/skills/) - one new file with a
# `register_skill(...)` call is how skill #3 gets added, not an edit
# here.
skill_registry.discover_skills()

# Loading the sentence-transformers model is expensive (first use may
# download weights) - force it to load once here, at module level,
# rather than on the first request. `get_default_embedder()` (not a
# fresh `SentenceTransformerEmbedder()`) so this process's one loaded
# model is shared with `praxis.agents.skills.retrieve_documents`
# (Phase 5) instead of each loading its own copy.
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
    letting it take down the whole aggregation. Shared by `/health`
    (below) and the scheduled health scan (`record_health_scan`, spec
    §18) - deliberately the *same* function, not two copies of this
    loop, so the on-demand and the scheduled view of "what's healthy"
    can never silently drift apart (spec §10: "`GET /health` plus the
    `HealthMonitor` ... exposed both on-demand and on schedule").
    """
    results = []
    for check in HEALTH_CHECKS:
        try:
            results.append(await check())
        except Exception as exc:  # noqa: BLE001 - one bad check must not take down /health
            results.append(HealthStatus(name=check.__name__, healthy=False, detail=str(exc)))
    return results


@app.get("/health")
async def health() -> JSONResponse:
    results = await run_health_checks()

    overall = all(r.healthy for r in results)
    body = {
        "healthy": overall,
        "components": [
            {"name": r.name, "healthy": r.healthy, "detail": r.detail} for r in results
        ],
    }
    return JSONResponse(content=body, status_code=200 if overall else 503)


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
    exact same aggregation `/health` uses, never a duplicate - then
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
                    HealthRecord(component=result.name, healthy=result.healthy, detail=result.detail)
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
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.approval_timeout_seconds)
    swept: list[str] = []
    try:
        async with store.session() as session:
            stmt = select(Task).where(
                Task.status.in_(["awaiting_approval", "awaiting_clarification"]),
                Task.updated_at < cutoff,
            )
            stale_tasks = (await session.execute(stmt)).scalars().all()

            for task in stale_tasks:
                waited_seconds = (datetime.now(timezone.utc) - task.updated_at).total_seconds()
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
        _approval_timeout_sweep_job, settings.approval_timeout_seconds, job_id="approval_timeout_sweep"
    )
    scheduler.start()
    return scheduler


_scheduler = _build_scheduler()


@app.post("/attachments", status_code=201)
async def upload_attachment(file: UploadFile = File(...)) -> dict[str, Any]:
    """File upload -> Ingestion pipeline (spec §5, §14).

    Settings/PostgresStore are constructed fresh per request rather than
    once at module import time - the same posture `_database_check`
    above already takes, so `PRAXIS_DATABASE_URL` can be pointed at a
    different database per test (or per deployment reload) without
    re-importing this module. The embedder and parser registry, in
    contrast, don't depend on Settings and are built once above.

    No `DocumentEnrichment` is constructed/passed here by default - it
    would put a real LLM call on the hot path of every upload (cost and
    latency neither the spec nor this endpoint's existing contract
    calls for). `ingest()`'s `enrichment` parameter stays available for
    a future deployment-level toggle; this handler already surfaces a
    `summary`/`topics` field on the response whenever `ingest()` did
    return one, so wiring enrichment in later needs no change here.
    """
    data = await file.read()
    mime_type = file.content_type or "application/octet-stream"
    source = file.filename or "upload"

    try:
        settings = Settings()
    except Exception as exc:  # noqa: BLE001 - a clear 500 beats a bare import-time traceback
        raise HTTPException(status_code=500, detail=f"configuration error: {exc}") from exc

    db = PostgresStore(settings)
    try:
        blob_store = LocalBlobStore(settings.blob_store_root)
        vector_store = PgVectorStore(db)
        attachment_id = await ingest(
            data,
            mime_type,
            source,
            blob_store=blob_store,
            parser_registry=parser_registry,
            embedder=_embedder,
            vector_store=vector_store,
            db=db,
        )
    except ValueError as exc:
        # e.g. no parser registered for this mime type - a client error,
        # not a server fault (spec §12's "never wrap failure as if it
        # were success" - this must not come back as a 201).
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"ingestion failed: {exc}") from exc
    finally:
        await db.dispose()

    response: dict[str, Any] = {"attachment_id": attachment_id, "status": "indexed"}
    # `attachment_id` is an `IngestResult` (a `str` subclass) - `summary`
    # is only non-None when an `enrichment` was passed to `ingest()` and
    # actually ran; never drop it on the floor when it did (spec §5
    # step 4, this endpoint's docstring above).
    if attachment_id.summary is not None:
        response["summary"] = attachment_id.summary
        response["topics"] = attachment_id.topics or []
    return response


@app.get("/artifacts/{key:path}")
async def get_artifact(key: str) -> Response:
    """Fetches a stored artifact - a rendered chart from `create_chart`
    (spec §13) today, and, going forward, whatever else lands in the
    same blob store under its own key - as a real binary response with
    the right `Content-Type`, rather than a client having to reach for
    the blob store directly.

    `key:path` (not a plain `{key}`) so a real artifact key - which is
    itself a relative path with a slash in it, e.g.
    `"charts/<uuid>.png"` - is matched whole, not truncated at the first
    `/`.

    `LocalBlobStore` already refuses a `..`-containing/absolute key at
    the store layer (`ValueError`, see `praxis.memory.blob_store`) - this
    route's only job is to turn that, and a genuinely missing key
    (`FileNotFoundError`), into a clean `400`/`404` rather than letting
    either propagate as an unhandled `500` (spec §12: "never a silent
    no-op or a 500").
    """
    settings = Settings()
    blob_store = LocalBlobStore(settings.blob_store_root)
    try:
        data = await blob_store.get(key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid artifact key: {exc}") from exc
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"no artifact stored under key '{key}'") from None

    mime_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
    return Response(content=data, media_type=mime_type)


# Orchestrator wiring (spec §3, §14). Unlike `upload_attachment`'s
# per-request `PostgresStore`, this is a true module-level singleton,
# constructed lazily (not at import time, for the same "Settings() may
# not be configured yet" reason `_build_connector_registry` defers)
# but memoized thereafter: `Orchestrator` holds in-memory execution-
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
        )
    return _orchestrator


class IntentRequest(BaseModel):
    text: str
    # Phase 11 (spec §16.2's literal example payload: `{"text": "...",
    # "connector": "customer-db"}`) - optional, forwarded verbatim to
    # `Orchestrator.start_task`'s own `connector_name` (see its
    # docstring for resolution/failure semantics).
    connector: str | None = None


class ApproveRequest(BaseModel):
    approved: bool


class ClarifyRequest(BaseModel):
    answer: str


@app.post("/intent", status_code=201)
async def create_intent(body: IntentRequest) -> dict[str, Any]:
    """Ad-hoc ask -> a new `Task`, run through the Orchestrator (spec §14)."""
    orchestrator = _get_orchestrator()
    task_id = await orchestrator.start_task(body.text, connector_name=body.connector)
    return {"task_id": task_id}


@app.post("/webhook/alert", status_code=201)
async def webhook_alert(request: Request) -> dict[str, Any]:
    """Async incident trigger (spec §14; spec §16.1 step 1: "Alertmanager
    fires -> `POST /webhook/alert` -> API creates a `Task` ..., assigns
    `correlation_id`, logs the event").

    Rides the *identical* `Orchestrator.start_task()` path `/intent`
    above already uses - same `Task` creation, same `correlation_id`
    assignment, same `task_started` logging - never a parallel "webhook
    task" system (spec §7's "rides the identical Orchestrator path"
    principle for a scheduled trigger, generalized here to a webhook
    trigger: both just have to produce an `intent_text` string, then
    hand it to the one real path).

    The body is read and validated by hand (a plain `Request`, not a
    pydantic model) rather than via FastAPI's usual declarative body
    parsing, specifically so *every* malformed shape - missing `alerts`,
    an empty list, invalid JSON, wrong field types - comes back as a
    clear `400` (spec: "not a silent no-op or a 500"), rather than
    pydantic's usual `422` for a schema mismatch or an unhandled `500`
    for, say, `.json()` failing on a non-JSON body.

    Real Alertmanager webhook payloads batch multiple alerts into one
    call (`{"alerts": [...]}`, alongside fields this MVP has no use for
    - `version`, `groupKey`, `receiver`, `groupLabels`, ...). This
    handler deliberately turns only the *first* alert in the batch into
    one `Task`, rather than either (a) silently dropping the rest, or
    (b) fanning out one `Task` per alert - a single, reasonable MVP
    choice, not a full multi-alert dispatcher; a later phase's natural
    extension point is right here, with no change to how any one alert
    becomes an `Intent`.
    """
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001 - an unparseable body is a 400, not a 500
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="webhook payload must be a JSON object")

    alerts = payload.get("alerts")
    if not isinstance(alerts, list) or not alerts:
        raise HTTPException(
            status_code=400, detail="webhook payload must include a non-empty 'alerts' list"
        )

    alert = alerts[0]
    if not isinstance(alert, dict):
        raise HTTPException(status_code=400, detail="each entry in 'alerts' must be a JSON object")

    labels = alert.get("labels")
    labels = labels if isinstance(labels, dict) else {}
    annotations = alert.get("annotations")
    annotations = annotations if isinstance(annotations, dict) else {}

    alertname = labels.get("alertname") or "unknown_alert"
    summary = annotations.get("summary") or annotations.get("description") or "no summary provided"
    intent_text = f"Investigate alert '{alertname}': {summary}"

    orchestrator = _get_orchestrator()
    task_id = await orchestrator.start_task(intent_text)
    return {"task_id": task_id}


@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict[str, Any]:
    """Task status/result, including the live checklist and any pending
    approval/clarification detail (spec §7, §9, §14)."""
    settings = Settings()
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            task = await session.get(Task, task_id)
    finally:
        await store.dispose()

    if task is None:
        raise HTTPException(status_code=404, detail=f"no task with id '{task_id}'")

    return {
        "status": task.status,
        "checklist": task.checklist,
        "pending_input": task.pending_input,
        "result": task.result,
    }


@app.post("/tasks/{task_id}/approve")
async def approve_task(task_id: str, body: ApproveRequest) -> dict[str, Any]:
    """Resumes a mutating task paused on approval (spec §8, §14 - the interrupt)."""
    orchestrator = _get_orchestrator()
    try:
        await orchestrator.resume_after_approval(task_id, body.approved)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        # Not currently awaiting approval (wrong status, unknown to this
        # process, ...) - a conflict with the resource's current state,
        # not a missing resource or a bad request body.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"task_id": task_id}


@app.post("/tasks/{task_id}/clarify")
async def clarify_task(task_id: str, body: ClarifyRequest) -> dict[str, Any]:
    """Answers a pending `ClarificationRequest` and resumes (spec §8, §14).

    Nothing in this phase's scope ever raises a `ClarificationRequest`
    (no Planner-side clarification logic is required yet - see the
    phase brief's scope note): this endpoint is still a real,
    contract-correct implementation rather than an omission - it
    validates the task exists and is genuinely `awaiting_clarification`
    before doing anything, exactly like `approve_task` above. It simply
    has no caller in this phase that ever reaches that state, so its
    "success" path (clearing the pause and recording the answer) stays
    real but untested-via-a-live-caller until a later phase's Planner/
    Factory actually raises one.
    """
    settings = Settings()
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            task = await session.get(Task, task_id)
            if task is None:
                raise HTTPException(status_code=404, detail=f"no task with id '{task_id}'")
            if task.status != "awaiting_clarification":
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"task '{task_id}' is not awaiting clarification "
                        f"(status: '{task.status}')"
                    ),
                )
            task.pending_input = None
            task.status = "running"
            task.result = {**(task.result or {}), "clarification_answer": body.answer}
            await session.commit()
    finally:
        await store.dispose()
    return {"task_id": task_id}


# A task in either of these is done, for good, no further event will
# ever be published for it (`Orchestrator._advance`/`_process_level`/
# `_synthesize_missing_skill`/`resume_after_approval` - see
# praxis/core/orchestrator.py) - the one condition `stream_task` below
# uses to know when to stop forwarding events and close the socket.
_TERMINAL_TASK_STATUSES = {"completed", "failed"}


@app.websocket("/tasks/{task_id}/stream")
async def stream_task(websocket: WebSocket, task_id: str) -> None:
    """Live status/log stream (spec §14: "including checklist item
    transitions and any new `ClarificationRequest` the instant it's
    raised").

    Every message this sends - the very first one included - is
    `praxis.core.events.task_state_snapshot`'s shape, identical to `GET
    /tasks/{task_id}`'s own response body above: a client renders a WS
    frame with the exact same code it already uses for a polled `GET`.

    Sequence: reject immediately (before ever accepting the handshake)
    if `task_id` doesn't exist - spec: "close with a clear error code/
    reason rather than hanging". Otherwise accept, subscribe to
    `task_event_bus`, *then* re-read the task's current durable state
    and send it (so a client connecting mid-task isn't left waiting for
    the next change), then forward every event `Orchestrator` publishes
    for this task - real state transitions, never a fake heartbeat -
    until one arrives in a terminal status, send that, and close
    cleanly.

    Subscribing *before* re-reading current state (rather than the
    other way round) closes a race that would otherwise be able to hang
    this route forever: if the task finished in the gap between the
    existence check and the subscribe call, subscribing first guarantees
    either the terminal event is already waiting in the queue, or the
    fresh read immediately below already shows the terminal status -
    never a subscribe that's too late to ever see one more event because
    the task was already done and nothing will ever publish again.
    """
    settings = Settings()
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            exists = await session.get(Task, task_id)
        if exists is None:
            await websocket.close(code=4404, reason=f"no task with id '{task_id}'")
            return

        await websocket.accept()
        queue = task_event_bus.subscribe(task_id)
        # A task paused on `awaiting_approval`/`awaiting_clarification`
        # (spec §8) is real, but *not* terminal - this route must keep
        # the connection open across a pause exactly like it would
        # across any other in-progress state (more events may still
        # arrive, e.g. once someone calls `POST /tasks/{id}/approve`).
        # That means the loop below can be waiting on `queue.get()`
        # indefinitely with nothing else ever going to publish again in
        # this test/run - so a client-initiated disconnect must be
        # noticed concurrently, or this coroutine (and the `TestClient`
        # session waiting on it to finish) would hang forever. Hence the
        # second, concurrently-awaited task below, whose only job is to
        # notice `{"type": "websocket.disconnect"}` the moment it
        # arrives - this route never expects the client to *send*
        # anything meaningful of its own.
        disconnect_task = asyncio.ensure_future(_wait_for_client_disconnect(websocket))
        try:
            async with store.session() as session:
                task = await session.get(Task, task_id)
            assert task is not None  # just confirmed to exist, immediately above

            await websocket.send_json(task_state_snapshot(task))

            if task.status not in _TERMINAL_TASK_STATUSES:
                while True:
                    get_event = asyncio.ensure_future(queue.get())
                    done, _pending = await asyncio.wait(
                        {get_event, disconnect_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if disconnect_task in done:
                        get_event.cancel()
                        break
                    event = get_event.result()
                    await websocket.send_json(event)
                    if event["status"] in _TERMINAL_TASK_STATUSES:
                        break
        except WebSocketDisconnect:
            # The client hung up early - nothing left to stream to, and
            # nothing to clean up beyond the `finally` below.
            pass
        finally:
            if not disconnect_task.done():
                disconnect_task.cancel()
            task_event_bus.unsubscribe(task_id, queue)

        try:
            await websocket.close(code=1000)
        except RuntimeError:
            # Already closed (e.g. the client disconnected first, above) -
            # closing twice is a no-op, not an error worth surfacing.
            pass
    finally:
        await store.dispose()


async def _wait_for_client_disconnect(websocket: WebSocket) -> None:
    """Resolves the moment the client hangs up - the half of `stream_task`
    above's disconnect race that lets a still-paused (non-terminal) task's
    stream close promptly instead of blocking on `queue.get()` forever
    once nobody is listening any more."""
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
