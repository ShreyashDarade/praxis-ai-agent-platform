# praxis/connectors/postgres/postgres_connector.py
"""Connector for an arbitrary, user-specified Postgres database (spec §6, §16.2).

This is distinct from praxis.memory.db.PostgresStore, which is Praxis's
own metadata store reached via SQLAlchemy. PostgresConnector talks
directly to asyncpg and is meant to be pointed at *any* external
Postgres database a task needs data from - e.g. the "customer-db" in
the ad-hoc dashboard walkthrough (spec §16.2) - not Praxis's own
schema. Because there is no single external DB, there's no global
Settings field for it: callers construct and register one per DSN they
need (see praxis/connectors/bootstrap.py's docstring).

``describe()`` is the "generic MCP describe call" the Capability
Factory relies on to introspect an unfamiliar database before
synthesizing a tool against it (spec §3, §6).
"""
from __future__ import annotations

import re
from typing import Any

import asyncpg

from praxis.core.interfaces import Connector, ConnectorDescription, HealthStatus

# Defense in depth (spec §6 "safety net independent of LLM-authored
# code"): even when the DB grants are misconfigured, or a synthesized
# tool builds this query itself, a read-only connector refuses anything
# that merely *looks* like a mutation before it ever reaches the wire.
_MUTATING_QUERY = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE)\b", re.IGNORECASE
)


class PostgresConnector(Connector):
    """Read (and, if configured, write) access to one external Postgres database.

    ``dsn`` is a plain asyncpg-style DSN (``postgresql://user:pass@host:port/db``),
    not a SQLAlchemy ``postgresql+asyncpg://`` URL - this connector talks to
    asyncpg directly, with no SQLAlchemy dependency.
    """

    def __init__(self, dsn: str, name: str = "postgres", read_only: bool = True) -> None:
        self.name = name
        self.read_only = read_only
        self._dsn = dsn

    async def describe(self) -> ConnectorDescription:
        conn = await asyncpg.connect(self._dsn)
        try:
            rows = await conn.fetch(
                """
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position
                """
            )
        finally:
            await conn.close()

        schema: dict[str, Any] = {}
        for row in rows:
            columns = schema.setdefault(row["table_name"], [])
            columns.append({"name": row["column_name"], "type": row["data_type"]})
        return ConnectorDescription(kind="postgres", schema=schema)

    async def read(self, query: str, **params: Any) -> Any:
        if self.read_only and _MUTATING_QUERY.match(query):
            raise PermissionError(
                f"connector '{self.name}' is registered read-only; "
                f"refusing a query that looks like a mutation"
            )
        conn = await asyncpg.connect(self._dsn)
        try:
            # Keyword params are bound positionally, in the order given,
            # to the query's $1, $2, ... placeholders.
            rows = await conn.fetch(query, *params.values())
        finally:
            await conn.close()
        return [dict(row) for row in rows]

    async def health(self) -> HealthStatus:
        try:
            conn = await asyncpg.connect(self._dsn)
            try:
                await conn.fetchval("SELECT 1")
            finally:
                await conn.close()
            return HealthStatus(name=self.name, healthy=True)
        except Exception as exc:  # noqa: BLE001 - a health check must never raise
            return HealthStatus(name=self.name, healthy=False, detail=str(exc))
