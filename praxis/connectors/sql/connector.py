# praxis/connectors/sql/connector.py
"""Connector for an arbitrary, user-specified SQL database of any dialect (spec §6).

`praxis.connectors.postgres.connector.PostgresConnector` talks to
asyncpg directly, which is exactly what makes it Postgres-only. This
connector goes through SQLAlchemy's async engine instead - the same
dialect abstraction `praxis.memory.db.PostgresStore` already uses for
Praxis's own metadata store - so the *same* code works against
Postgres, MySQL, SQLite, MSSQL, etc., provided the right async driver
is installed for whichever DSN is passed in (e.g.
``postgresql+asyncpg://``, ``sqlite+aiosqlite://``,
``mysql+aiomysql://``). ``describe()`` in particular uses SQLAlchemy's
generic ``inspect()`` rather than Postgres-specific
``information_schema`` queries, so introspection works identically
regardless of dialect.

Like `PostgresConnector`, there is no global `Settings` field for this:
"connect to any SQL database" has no single DSN to gate a factory on,
so instantiating one is left to whoever needs it (a task, a test), not
auto-registered by `praxis.connectors.bootstrap`.
"""
from __future__ import annotations

import re
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection as SyncConnection
from sqlalchemy.ext.asyncio import create_async_engine

from praxis.core.interfaces import Connector, ConnectorDescription, HealthStatus

# Same defense-in-depth guard as PostgresConnector, dialect-independent:
# refuse anything that merely *looks* like a mutation before it ever
# reaches the wire, regardless of which SQL dialect is on the other end.
_MUTATING_QUERY = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE)\b", re.IGNORECASE
)


class SQLConnector(Connector):
    """Read (and, if configured, write) access to one external SQL database,
    of whichever dialect ``dsn`` names.

    ``dsn`` is a full SQLAlchemy async URL, not a driver-native DSN -
    that dialect prefix (``postgresql+asyncpg://``, ``sqlite+aiosqlite://``,
    ...) is what lets SQLAlchemy pick the right async driver and dialect.
    """

    def __init__(self, dsn: str, name: str, read_only: bool = True) -> None:
        self.name = name
        self.read_only = read_only
        self._dsn = dsn

    async def describe(self) -> ConnectorDescription:
        engine = create_async_engine(self._dsn)
        try:
            async with engine.connect() as conn:
                schema = await conn.run_sync(_introspect)
        finally:
            await engine.dispose()
        return ConnectorDescription(kind="sql", schema=schema)

    async def read(self, query: str, **params: Any) -> Any:
        if self.read_only and _MUTATING_QUERY.match(query):
            raise PermissionError(
                f"connector '{self.name}' is registered read-only; "
                f"refusing a query that looks like a mutation"
            )
        engine = create_async_engine(self._dsn)
        try:
            async with engine.connect() as conn:
                # text(...) binds named (:param) placeholders, translated
                # to each dialect's native parameter style - unlike
                # PostgresConnector's positional $1/$2 asyncpg style,
                # this is what stays portable across dialects.
                result = await conn.execute(text(query), params)
                # A SELECT-shaped query returns rows; a mutating one (only
                # reachable at all when read_only=False, past the guard
                # above) doesn't - SQLAlchemy's CursorResult raises if
                # .mappings() is called on the latter, unlike asyncpg's
                # fetch(), which just returns an empty list either way.
                rows = result.mappings().all() if result.returns_rows else []
                # engine.connect() autobegins an implicit transaction on
                # first execute; it must be committed explicitly or a
                # mutation made through here would be rolled back when
                # the connection closes below. A no-op for a plain SELECT.
                await conn.commit()
        finally:
            await engine.dispose()
        return [dict(row) for row in rows]

    async def health(self) -> HealthStatus:
        try:
            engine = create_async_engine(self._dsn)
            try:
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
            finally:
                await engine.dispose()
            return HealthStatus(name=self.name, healthy=True)
        except Exception as exc:  # noqa: BLE001 - a health check must never raise
            return HealthStatus(name=self.name, healthy=False, detail=str(exc))


def _introspect(sync_conn: SyncConnection) -> dict[str, Any]:
    """Runs on a sync-facade connection via ``AsyncConnection.run_sync`` since
    SQLAlchemy's ``inspect()`` is itself a synchronous API.
    """
    inspector = inspect(sync_conn)
    schema: dict[str, Any] = {}
    for table_name in inspector.get_table_names():
        schema[table_name] = [
            {"name": column["name"], "type": str(column["type"])}
            for column in inspector.get_columns(table_name)
        ]
    return schema
