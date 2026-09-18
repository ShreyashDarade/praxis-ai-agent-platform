# praxis/memory/vector_store.py
"""pgvector-backed `VectorStore` (spec §2, §5 step 5, §9).

Reuses `PostgresStore.session()` rather than holding its own engine -
same Postgres instance as everything else per spec §2's MVP deployment
("This means the MVP *deploys* exactly one external service
(PostgreSQL with the pgvector extension)").
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select

from praxis.core.interfaces import VectorStore
from praxis.memory.db import PostgresStore
from praxis.memory.models import DEFAULT_TENANT_ID, VectorChunk


class PgVectorStore(VectorStore):
    """`VectorChunk` rows via pgvector's cosine-distance operator (`<=>`).

    **Tenant isolation (Phase 12)**: every row carries a `tenant_id`, and
    `similarity_search` filters on it *in the SQL WHERE clause*, not by
    post-filtering results in Python. That distinction is load-bearing:
    post-filtering would let another tenant's chunks consume the `top_k`
    budget (silently degrading recall, and leaking the *existence* of
    neighbouring data through result counts), whereas an indexed
    predicate means a tenant's search only ever ranks its own rows.
    """

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    async def upsert(
        self,
        doc_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
        *,
        tenant_id: str = DEFAULT_TENANT_ID,
    ) -> None:
        # `VectorChunk.doc_id` has no unique constraint at the schema
        # level (see praxis/memory/models.py) - upsert semantics are
        # implemented here in application code: delete any existing
        # row(s) for this doc_id, then insert the new one, both inside
        # one transaction, so a second upsert of the same doc_id updates
        # in place rather than accumulating duplicates (spec §5/§9
        # provenance dedup requirement).
        #
        # The `VectorStore` interface has no dedicated "content"
        # parameter - the chunk's text travels inside `metadata["content"]`
        # and is split back out here into `VectorChunk.content`'s own
        # column, so `chunk_metadata` stays pure metadata (source,
        # mime_type, chunk_index, ...) with no duplicated text.
        metadata = dict(metadata)
        content = str(metadata.pop("content", ""))
        async with self._store.session() as session:
            async with session.begin():
                await session.execute(
                    delete(VectorChunk).where(
                        VectorChunk.doc_id == doc_id, VectorChunk.tenant_id == tenant_id
                    )
                )
                session.add(
                    VectorChunk(
                        tenant_id=tenant_id,
                        doc_id=doc_id,
                        content=content,
                        embedding=embedding,
                        chunk_metadata=metadata,
                    )
                )

    async def similarity_search(
        self,
        embedding: list[float],
        top_k: int = 5,
        *,
        tenant_id: str = DEFAULT_TENANT_ID,
    ) -> list[dict[str, Any]]:
        distance = VectorChunk.embedding.cosine_distance(embedding).label("distance")
        stmt = (
            select(VectorChunk, distance)
            .where(VectorChunk.tenant_id == tenant_id)
            .order_by(distance)
            .limit(top_k)
        )

        async with self._store.session() as session:
            result = await session.execute(stmt)
            rows = result.all()

        return [
            {
                "doc_id": chunk.doc_id,
                "content": chunk.content,
                "metadata": chunk.chunk_metadata,
                # Cosine distance is in [0, 2]; report similarity
                # (higher = closer) so callers rank results the
                # intuitive way without knowing the underlying metric.
                "score": 1.0 - float(dist),
            }
            for chunk, dist in rows
        ]
