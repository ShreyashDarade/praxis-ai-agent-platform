# praxis/ingestion/pipeline.py
"""Ingestion & RAG orchestration (spec §5): upload -> parse -> chunk -> embed -> index -> retrieve.

Step 4 (Enrich) is deliberately skipped per this phase's scope - it
needs the LLM Catalogue (Phase 4) and no LLM-calling code belongs in
this phase.

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
from praxis.memory.db import PostgresStore
from praxis.memory.models import Attachment

logger = logging.getLogger(__name__)

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
) -> str:
    """Runs one attachment through the full ingestion pipeline; returns its `Attachment.id`.

    `chunker` is normally left unset - the right strategy (`TableAwareChunker`
    vs `RecursiveChunker`) is picked from `mime_type` internally. It's
    still accepted as an explicit override for callers/tests that need
    to force a specific chunker.
    """
    async with db.session() as session:
        attachment = Attachment(mime_type=mime_type, source=source, size_bytes=len(data), status="uploaded")
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
                }
                await vector_store.upsert(doc_id, embedding, metadata)

        await _set_status(db, attachment_id, "indexed")
    except Exception:
        logger.exception("ingestion failed for attachment %s (source=%s)", attachment_id, source)
        await _set_status(db, attachment_id, "failed")
        raise

    return attachment_id


async def retrieve(
    query_text: str,
    *,
    top_k: int,
    embedder: Embedder,
    vector_store: VectorStore,
) -> list[dict[str, Any]]:
    """Embeds `query_text` and returns the `top_k` nearest chunks (spec §5 step 6)."""
    (query_embedding,) = await embedder.embed([query_text])
    return await vector_store.similarity_search(query_embedding, top_k=top_k)
