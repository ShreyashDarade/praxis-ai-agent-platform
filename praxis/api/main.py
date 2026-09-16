"""FastAPI application entrypoint (spec §14)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

from docker.errors import DockerException
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from praxis.agents import skill_registry
from praxis.agents.capability_factory import CapabilityFactory
from praxis.agents.planner import Planner
from praxis.config import Settings
from praxis.connectors.bootstrap import build_registry
from praxis.connectors.registry import ConnectorRegistry
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
from praxis.memory.models import Task
from praxis.memory.vector_store import PgVectorStore
from praxis.sandbox.executor import DockerSandboxExecutor

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
    return build_registry(settings)


# Connector *membership* is fixed at process startup (deployment config,
# spec §19 step 4) - unlike _database_check above, which re-reads
# Settings() per request because DB reachability is a live condition.
# Each connector's own health() is still re-evaluated on every /health
# call, via the closures all_health_checks() returns.
for _connector_health_check in _build_connector_registry().all_health_checks():
    register_health_check(_connector_health_check)


@app.get("/health")
async def health() -> JSONResponse:
    results = []
    for check in HEALTH_CHECKS:
        try:
            results.append(await check())
        except Exception as exc:  # noqa: BLE001 - one bad check must not take down /health
            results.append(HealthStatus(name=check.__name__, healthy=False, detail=str(exc)))

    overall = all(r.healthy for r in results)
    body = {
        "healthy": overall,
        "components": [
            {"name": r.name, "healthy": r.healthy, "detail": r.detail} for r in results
        ],
    }
    return JSONResponse(content=body, status_code=200 if overall else 503)


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
        _orchestrator = Orchestrator(store, planner, skill_registry, capability_factory)
    return _orchestrator


class IntentRequest(BaseModel):
    text: str


class ApproveRequest(BaseModel):
    approved: bool


class ClarifyRequest(BaseModel):
    answer: str


@app.post("/intent", status_code=201)
async def create_intent(body: IntentRequest) -> dict[str, Any]:
    """Ad-hoc ask -> a new `Task`, run through the Orchestrator (spec §14)."""
    orchestrator = _get_orchestrator()
    task_id = await orchestrator.start_task(body.text)
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
