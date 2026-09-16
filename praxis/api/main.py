"""FastAPI application entrypoint (spec §14)."""
from __future__ import annotations

from typing import Awaitable, Callable

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from praxis.config import Settings
from praxis.connectors.bootstrap import build_registry
from praxis.connectors.registry import ConnectorRegistry
from praxis.core.interfaces import HealthStatus
from praxis.ingestion.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from praxis.ingestion.parsers import registry as parser_registry
from praxis.ingestion.pipeline import ingest
from praxis.memory.blob_store import LocalBlobStore
from praxis.memory.db import PostgresStore
from praxis.memory.vector_store import PgVectorStore

app = FastAPI(title="Praxis")

# Trigger self-registration of every built-in parser (text/document/
# image-OCR/tabular) exactly once at import time - mirrors how
# `_build_connector_registry()` below drives connector self-registration
# via `build_registry`. Safe to call repeatedly (idempotent).
parser_registry.discover_parsers()

# Loading the sentence-transformers model is expensive (first use may
# download weights) - build it once here, at module level, rather than
# per-request inside `upload_attachment` below (spec §5 step 5's
# environment note; mirrors SentenceTransformerEmbedder's own
# load-once-at-construction discipline).
_embedder = SentenceTransformerEmbedder()

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
async def upload_attachment(file: UploadFile = File(...)) -> dict[str, str]:
    """File upload -> Ingestion pipeline (spec §5, §14).

    Settings/PostgresStore are constructed fresh per request rather than
    once at module import time - the same posture `_database_check`
    above already takes, so `PRAXIS_DATABASE_URL` can be pointed at a
    different database per test (or per deployment reload) without
    re-importing this module. The embedder and parser registry, in
    contrast, don't depend on Settings and are built once above.
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

    return {"attachment_id": attachment_id, "status": "indexed"}
