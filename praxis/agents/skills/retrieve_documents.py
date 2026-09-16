# praxis/agents/skills/retrieve_documents.py
"""`retrieve_documents`: a read-only skill wrapping Phase 3's real
retrieval path (`praxis.ingestion.pipeline.retrieve`), so the
Orchestrator can hand a task's real ingested/embedded content to a
running plan exactly as `GET /attachments`'s RAG path already does.

Dependencies (embedder, `PostgresStore`, `PgVectorStore`) are
constructed the same way `praxis/api/main.py` builds them for the
`/attachments` endpoint: the embedder via the shared process-wide
singleton (`get_default_embedder()` - loading the sentence-transformers
model is expensive, and both this module and `praxis.api.main` sharing
one instance avoids loading it twice), the DB store fresh per call
(mirrors `praxis.api.main._database_check`'s per-request `Settings()`
posture, so `PRAXIS_DATABASE_URL` can differ per test/deployment
without re-importing this module).
"""
from __future__ import annotations

from typing import Any

from praxis.agents.skill import Skill
from praxis.agents.skill_registry import register_skill
from praxis.config import Settings
from praxis.ingestion.embedders.sentence_transformer_embedder import get_default_embedder
from praxis.ingestion.pipeline import retrieve
from praxis.memory.db import PostgresStore
from praxis.memory.vector_store import PgVectorStore

_DEFAULT_TOP_K = 5


class RetrieveDocumentsSkill(Skill):
    name = "retrieve_documents"
    risk = "read_only"
    inputs = {"query": "text to search for"}
    outputs = {"results": "list of matching document chunks"}

    async def run(self, **kwargs: Any) -> Any:
        query = kwargs["query"]
        settings = Settings()
        db = PostgresStore(settings)
        try:
            vector_store = PgVectorStore(db)
            results = await retrieve(
                query,
                top_k=_DEFAULT_TOP_K,
                embedder=get_default_embedder(),
                vector_store=vector_store,
            )
        finally:
            await db.dispose()
        return {"results": results}


register_skill(RetrieveDocumentsSkill())
