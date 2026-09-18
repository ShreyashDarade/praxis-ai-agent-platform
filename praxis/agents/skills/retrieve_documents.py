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
from praxis.memory.models import DEFAULT_TENANT_ID
from praxis.memory.vector_store import PgVectorStore

_DEFAULT_TOP_K = 5


class RetrieveDocumentsSkill(Skill):
    name = "retrieve_documents"
    risk = "read_only"
    inputs = {
        "query": (
            "text to search for. Semantic retrieval over document prose - use "
            "it for questions answered by what a document *says*. Do NOT use it "
            "to compute a number from an uploaded spreadsheet (a sum, count, "
            "average, min/max or ranking): similarity search returns text that "
            "resembles an answer, which is not the same as the correct total. "
            "Use `query_table` for that."
        )
    }
    outputs = {"results": "list of matching document chunks"}

    async def run(self, **kwargs: Any) -> Any:
        query = kwargs["query"]
        # Phase 12: the Orchestrator injects `tenant_id` out of band on
        # every skill call. Retrieval is filtered to that tenant inside
        # the vector store's own SQL - one tenant can never retrieve,
        # cite, or even see the existence of another's chunks.
        tenant_id = kwargs.get("tenant_id") or DEFAULT_TENANT_ID
        settings = Settings()
        db = PostgresStore(settings)
        try:
            vector_store = PgVectorStore(db)
            results = await retrieve(
                query,
                top_k=_DEFAULT_TOP_K,
                embedder=get_default_embedder(),
                vector_store=vector_store,
                tenant_id=tenant_id,
            )
        finally:
            await db.dispose()
        return {"results": results}


register_skill(RetrieveDocumentsSkill())
