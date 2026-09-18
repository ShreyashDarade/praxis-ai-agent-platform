# praxis/agents/skills/query_table.py
"""`query_table`: exact SQL over an uploaded spreadsheet (brief §4).

The counterpart to `retrieve_documents`. That skill answers questions
by similarity over embedded prose, which is right for "what does the
Q3 report say about churn" and wrong for "what was total EMEA revenue".
This one answers the second kind by running real SQL, with real types,
over the uploaded file's original bytes - the brief's "do not rely on
embeddings for exact aggregations".

Which to use is not a judgement the model has to make well for the
system to be safe: the two skills' descriptions below name the split
explicitly, and this one refuses anything that is not a bounded read,
so the worst case of picking wrong is a wrong-shaped answer rather than
a plausible-looking wrong number.

`risk = "read_only"`: `SqlGuard` validates every query as a read before
the file is even opened, and the attachment's bytes are fetched from
the blob store rather than the database, so nothing here can mutate
anything.
"""
from __future__ import annotations

from typing import Any

from praxis.agents.skill import Skill
from praxis.agents.skill_registry import register_skill
from praxis.config import Settings
from praxis.ingestion.tables import TABLE_NAME, describe_uploaded_table, query_table
from praxis.memory.blob_store import LocalBlobStore
from praxis.memory.db import PostgresStore
from praxis.memory.models import Attachment


class QueryTableSkill(Skill):
    name = "query_table"
    risk = "read_only"
    inputs = {
        "attachment_id": "id of an uploaded CSV/XLSX attachment to query",
        "sql": (
            f"a read-only SQL SELECT over the uploaded table, which is named "
            f"`{TABLE_NAME}` - e.g. "
            f"'SELECT region, SUM(revenue) AS total FROM {TABLE_NAME} GROUP BY region'. "
            "Use this rather than semantic retrieval whenever the answer is a "
            "computed number (a sum, count, average, min/max, or ranking): "
            "embeddings retrieve similar text, which is not the same as the "
            "correct total. Omit to get the table's schema instead."
        ),
    }
    outputs = {
        "rows": "the query's result rows",
        "schema": "the table's typed column schema",
        "query": "the SQL that actually ran, including any applied row bound",
    }

    async def run(self, **kwargs: Any) -> Any:
        attachment_id = kwargs["attachment_id"]
        sql = (kwargs.get("sql") or "").strip()

        settings = Settings()
        db = PostgresStore(settings)
        try:
            async with db.session() as session:
                attachment = await session.get(Attachment, attachment_id)
                if attachment is None:
                    raise KeyError(f"no attachment with id '{attachment_id}'")
                mime_type = attachment.mime_type
        finally:
            await db.dispose()

        blob_store = LocalBlobStore(settings.blob_store_root)
        data = await blob_store.get(attachment_id)

        # No SQL means "tell me what I can ask" - the schema, not a
        # guessed query. A model that has to invent a query blind writes
        # one against columns that may not exist.
        if not sql:
            schema = await describe_uploaded_table(data, mime_type)
            return {"rows": [], "schema": schema.to_dict(), "query": ""}

        return await query_table(data, mime_type, sql)


register_skill(QueryTableSkill())
