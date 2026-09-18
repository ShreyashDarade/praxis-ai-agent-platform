# praxis/ingestion/pipeline.py
"""Ingestion & RAG orchestration (spec §5): upload -> parse -> chunk -> enrich -> embed -> index -> retrieve.

Step 4 (Enrich) now runs between parse and chunk, per the spec's
lifecycle ordering, via an optional `DocumentEnrichment` (Phase 4's LLM
Catalogue + Prompt Manager). It is genuinely optional: `enrichment`
defaults to `None`, so every existing caller that doesn't pass one gets
byte-for-byte the same behavior as before this phase - enrichment is
additive, not mandatory on every ingest (mirrors spec §5 step 4's
"only when relations are meaningful"/explicitly requested framing).

Chunker selection by mime type is business logic that lives here, not
behind an extension-point registry like `ParserRegistry` - adding a new
*parser* is "one new file, zero core edits" (spec §5/§6's pattern), but
there are only ever two chunking strategies (prose vs. tabular-summary),
so a registry for that would be ceremony without a real extension point
behind it.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

from praxis.core.interfaces import BlobStore, Chunker, Embedder, Parser, VectorStore
from praxis.ingestion.chunkers.recursive_chunker import RecursiveChunker
from praxis.ingestion.chunkers.table_aware_chunker import TableAwareChunker
from praxis.ingestion.enrichment.document_enrichment import DocumentEnrichment, EnrichmentResult
from praxis.memory.db import PostgresStore
from praxis.memory.models import DEFAULT_TENANT_ID, Attachment
from praxis.safety.untrusted import detect_injection_markers

logger = logging.getLogger(__name__)


class IngestResult(str):
    """`ingest()`'s return value.

    Subclasses `str` rather than being a plain dataclass so this stays
    fully backward compatible: every existing/pre-existing caller that
    treats `ingest()`'s return value as a bare `attachment_id` string
    (equality checks, use as a `session.get()` primary key, dict keys,
    string formatting, ...) keeps working completely unchanged. The
    `summary`/`topics` attributes are purely additive, populated only
    when an `enrichment` was passed in and actually ran.
    """

    summary: str | None
    topics: list[str] | None

    def __new__(
        cls, attachment_id: str, summary: str | None = None, topics: list[str] | None = None
    ) -> "IngestResult":
        obj = super().__new__(cls, attachment_id)
        obj.summary = summary
        obj.topics = topics
        return obj


# The two tabular mime types TabularParser handles (spec §5 step 2) get
# the row/section-aware chunker; everything else gets the prose one.
_TABULAR_MIME_TYPES = frozenset(
    {
        "text/csv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
)

# Chunkers are stateless and cheap to construct, but reused across calls
# so `ingest()` doesn't re-build a text splitter on every attachment.
_RECURSIVE_CHUNKER = RecursiveChunker()
_TABLE_AWARE_CHUNKER = TableAwareChunker()


def _chunker_for(mime_type: str) -> Chunker:
    if mime_type in _TABULAR_MIME_TYPES:
        return _TABLE_AWARE_CHUNKER
    return _RECURSIVE_CHUNKER


class ParserRegistryLike(Protocol):
    """What `ingest()` needs from a parser registry - satisfied by the
    `praxis.ingestion.parsers.registry` module itself (a module is a
    valid Python object; no wrapper class is required), or any stand-in
    exposing the same function in a test."""

    def get_parser_for(self, mime_type: str) -> Parser: ...


async def _set_status(db: PostgresStore, attachment_id: str, status: str) -> None:
    async with db.session() as session:
        attachment = await session.get(Attachment, attachment_id)
        if attachment is None:  # pragma: no cover - defensive, row is always created first
            return
        attachment.status = status
        await session.commit()


async def ingest(
    data: bytes,
    mime_type: str,
    source: str,
    *,
    blob_store: BlobStore,
    parser_registry: ParserRegistryLike,
    embedder: Embedder,
    vector_store: VectorStore,
    db: PostgresStore,
    chunker: Chunker | None = None,
    enrichment: DocumentEnrichment | None = None,
    tenant_id: str = DEFAULT_TENANT_ID,
    uploaded_by_user_id: str | None = None,
) -> IngestResult:
    """Runs one attachment through the full ingestion pipeline; returns an
    `IngestResult` (an `Attachment.id` string, usable exactly as one).

    `chunker` is normally left unset - the right strategy (`TableAwareChunker`
    vs `RecursiveChunker`) is picked from `mime_type` internally. It's
    still accepted as an explicit override for callers/tests that need
    to force a specific chunker.

    `enrichment` is optional and defaults to `None` (no enrichment,
    identical to this pipeline's pre-Phase-4 behavior). When provided,
    `enrichment.enrich(text)` runs after parsing and before chunking
    (spec §5's parse -> chunk -> enrich -> embed ordering), and the
    resulting summary/topics are carried back on the returned
    `IngestResult` rather than dropped.
    """
    async with db.session() as session:
        attachment = Attachment(
            tenant_id=tenant_id,
            uploaded_by_user_id=uploaded_by_user_id,
            mime_type=mime_type,
            source=source,
            size_bytes=len(data),
            status="uploaded",
        )
        session.add(attachment)
        await session.commit()
        attachment_id = attachment.id

    try:
        await _set_status(db, attachment_id, "processing")

        # The attachment id is already a fresh UUID (spec §9's provenance
        # key builds on it below) - safe and collision-free as a flat
        # blob key with no path-traversal risk.
        await blob_store.put(attachment_id, data)

        parser = parser_registry.get_parser_for(mime_type)
        text = await parser.parse(data, mime_type)

        # Phase 13 (Prompt §4: "Treat extracted content as untrusted
        # data, never executable instructions"). Uploaded documents
        # were previously the one untrusted-content path with no
        # injection handling at all - web content had it, uploads did
        # not, even though an uploaded PDF is exactly as attacker-
        # controllable as a fetched page.
        #
        # Detection is recorded, not acted on by scrubbing: silently
        # rewriting a user's document would be worse than flagging it.
        # The marker names are persisted on the chunk metadata below so
        # a retrieval consumer can decide how much to trust a chunk,
        # and so an operator can audit what was flagged.
        injection_markers = detect_injection_markers(text)
        if injection_markers:
            logger.warning(
                "prompt_injection_markers_detected in attachment %s (source=%s): %s",
                attachment_id,
                source,
                injection_markers,
            )

        enrichment_result: EnrichmentResult | None = None
        if enrichment is not None:
            enrichment_result = await enrichment.enrich(text)

        chosen_chunker = chunker if chunker is not None else _chunker_for(mime_type)
        chunks = chosen_chunker.chunk(text)

        if chunks:
            embeddings = await embedder.embed(chunks)
            for chunk_index, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                doc_id = f"{attachment_id}:{chunk_index}"
                metadata: dict[str, Any] = {
                    "source": source,
                    "mime_type": mime_type,
                    "chunk_index": chunk_index,
                    "content": chunk_text,
                    "attachment_id": attachment_id,
                    "tenant_id": tenant_id,
                    # Carried per-chunk so a retrieval consumer can see
                    # that this text came from a document flagged for
                    # injection markers, rather than that signal being
                    # lost at ingest time.
                    "untrusted": True,
                    "injection_markers": injection_markers,
                }
                await vector_store.upsert(doc_id, embedding, metadata, tenant_id=tenant_id)

        await _set_status(db, attachment_id, "indexed")
    except Exception:
        logger.exception("ingestion failed for attachment %s (source=%s)", attachment_id, source)
        await _set_status(db, attachment_id, "failed")
        raise

    if enrichment_result is not None:
        return IngestResult(attachment_id, summary=enrichment_result.summary, topics=enrichment_result.topics)
    return IngestResult(attachment_id)


async def retrieve(
    query_text: str,
    *,
    top_k: int,
    embedder: Embedder,
    vector_store: VectorStore,
    tenant_id: str = DEFAULT_TENANT_ID,
) -> list[dict[str, Any]]:
    """Embeds `query_text` and returns the `top_k` nearest chunks (spec §5 step 6).

    `tenant_id` is pushed down into the store's own query rather than
    post-filtering here, so another tenant's chunks never consume the
    `top_k` budget (see `praxis.memory.vector_store.PgVectorStore`).
    """
    (query_embedding,) = await embedder.embed([query_text])
    return await vector_store.similarity_search(
        query_embedding, top_k=top_k, tenant_id=tenant_id
    )
