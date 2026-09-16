# praxis/memory/graph_store.py
"""Postgres-backed `GraphStore` (spec §2, §9): edge tables in the same
Postgres instance as everything else - mirrors `PgVectorStore`'s
pattern (praxis/memory/vector_store.py): constructed with a
`PostgresStore`, reuses its `.session()` rather than holding its own
engine.

Stores connector schemas already introspected and the lineage graph of
which skill/tool was created or used for which task (spec §9) - this
phase adds the store itself; those specific write sites land in later
phases (Capability Factory, connector introspection caching).
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import select

from praxis.core.interfaces import GraphStore
from praxis.memory.db import PostgresStore
from praxis.memory.models import GraphEdge


class PgGraphStore(GraphStore):
    """`GraphEdge` rows: a lightweight knowledge graph (spec §9)."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    async def add_edge(
        self, source: str, relation: str, target: str, metadata: dict[str, Any] | None = None
    ) -> None:
        async with self._store.session() as session:
            async with session.begin():
                session.add(
                    GraphEdge(
                        source=source,
                        relation=relation,
                        target=target,
                        edge_metadata=dict(metadata) if metadata is not None else {},
                    )
                )

    async def neighbors(self, node: str, relation: str | None = None) -> list[dict[str, Any]]:
        stmt = select(GraphEdge).where(GraphEdge.source == node)
        if relation is not None:
            stmt = stmt.where(GraphEdge.relation == relation)

        async with self._store.session() as session:
            result = await session.execute(stmt)
            edges = result.scalars().all()

        return [
            {
                "source": edge.source,
                "relation": edge.relation,
                "target": edge.target,
                "metadata": edge.edge_metadata,
            }
            for edge in edges
        ]
