# praxis/api/routes/attachments.py
"""File/artifact routes (spec §5, §13, §14) - split out of `praxis.api.main`:

- `POST /attachments`: File upload -> Ingestion pipeline (spec §5).
- `GET /artifacts/{key:path}`: fetches a stored artifact (spec §13, e.g.
  a rendered chart from `create_chart`).

Both build their own `Settings`/`PostgresStore`/`LocalBlobStore` fresh
per request - the same posture `praxis.api.main._database_check` takes -
so `PRAXIS_DATABASE_URL`/`PRAXIS_BLOB_STORE_ROOT` can differ per test or
per deployment reload without re-importing this module. The embedder and
parser registry, in contrast, don't depend on per-request `Settings` and
are built once, at `praxis.api.main` import time - this module reaches
that shared instance via `main._embedder` rather than constructing its
own.
"""
from __future__ import annotations

import mimetypes
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response

from praxis.api.dependencies import require
from praxis.config import Settings
from praxis.ingestion.parsers import registry as parser_registry
from praxis.ingestion.pipeline import ingest
from praxis.memory.blob_store import LocalBlobStore, is_key_in_tenant
from praxis.memory.db import PostgresStore
from praxis.memory.vector_store import PgVectorStore
from praxis.security.policy import Permission
from praxis.security.principal import Principal

router = APIRouter()


@router.post("/attachments", status_code=201)
async def upload_attachment(
    principal: Annotated[Principal, Depends(require(Permission.ATTACHMENT_UPLOAD))],
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """File upload -> Ingestion pipeline (spec §5, §14).

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

    # Imported inside the handler, not at module scope: `main` imports
    # this module (to mount the router), so a module-level import here
    # is a genuine cycle that resolves today only because of import
    # ordering. Deferring it to call time removes the cycle without
    # moving the shared embedder out of `main`, which is built once at
    # import time precisely so every request shares one model.
    from praxis.api import main

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
            embedder=main._embedder,
            vector_store=vector_store,
            db=db,
            tenant_id=principal.tenant_id,
            uploaded_by_user_id=principal.user_id,
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


@router.get("/artifacts/{key:path}")
async def get_artifact(
    key: str,
    principal: Annotated[Principal, Depends(require(Permission.ATTACHMENT_READ))],
) -> Response:
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

    **Tenant isolation (Phase 12)**: artifact keys produced by skills are
    namespaced per tenant (`praxis.memory.blob_store.tenant_artifact_key`),
    and a key outside the caller's own namespace is reported as 404 -
    never 403, which would confirm another tenant's artifact exists.
    """
    settings = Settings()
    if not is_key_in_tenant(key, principal.tenant_id):
        raise HTTPException(status_code=404, detail=f"no artifact stored under key '{key}'")
    blob_store = LocalBlobStore(settings.blob_store_root)
    try:
        data = await blob_store.get(key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid artifact key: {exc}") from exc
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"no artifact stored under key '{key}'"
        ) from None

    mime_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
    return Response(content=data, media_type=mime_type)
