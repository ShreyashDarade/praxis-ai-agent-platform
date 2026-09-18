# praxis/connectors/sql/sql_connector.py
"""Connector for an arbitrary, user-specified SQL database of any dialect (spec §6).

`praxis.connectors.postgres.postgres_connector.PostgresConnector` talks to
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

Like `PostgresConnector`, there is no *general-purpose* global `Settings`
field for this: "connect to any SQL database" has no single DSN to gate
a factory on, so instantiating one is left to whoever needs it (a task,
a test), not auto-registered by `praxis.connectors.bootstrap`. Phase 11
adds exactly one narrow, explicitly demo-labeled exception to this -
`Settings.demo_customer_db_dsn`, registered as `"customer-db"` by
`praxis.api.main` (not by `praxis.connectors.bootstrap.build_registry`
itself) purely for spec §16.2's dashboard walkthrough - see that
setting's own docstring; it is not a template for a second one.
"""
from __future__ import annotations

import re
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection as SyncConnection
from sqlalchemy.ext.asyncio import create_async_engine

from praxis.connectors.errors import describe_exception
from praxis.core.interfaces import Connector, ConnectorDescription, HealthStatus
from praxis.safety.sql_guard import QueryCostLimits, SqlGuard

# Same defense-in-depth guard as PostgresConnector, dialect-independent:
# refuse anything that merely *looks* like a mutation before it ever
# reaches the wire, regardless of which SQL dialect is on the other end.
# Kept alongside the Phase 13 `SqlGuard` (see `read`'s docstring) as a
# cheap first gate, not replaced by it.
_MUTATING_QUERY = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE)\b", re.IGNORECASE
)

# SQLAlchemy dialect name -> sqlglot dialect name. Only the mappings
# that actually differ or that this project has verified are listed;
# anything unlisted is passed to sqlglot as `None` (its permissive
# default grammar), which parses standard SQL correctly and simply
# offers no dialect-specific handling.
_SQLGLOT_DIALECTS: dict[str, str] = {
    "postgresql": "postgres",
    "sqlite": "sqlite",
    "mysql": "mysql",
    "mssql": "tsql",
    "oracle": "oracle",
    "snowflake": "snowflake",
    "bigquery": "bigquery",
    "duckdb": "duckdb",
    "clickhouse": "clickhouse",
}


class SQLConnector(Connector):
    """Read (and, if configured, write) access to one external SQL database,
    of whichever dialect ``dsn`` names.

    ``dsn`` is a full SQLAlchemy async URL, not a driver-native DSN -
    that dialect prefix (``postgresql+asyncpg://``, ``sqlite+aiosqlite://``,
    ...) is what lets SQLAlchemy pick the right async driver and dialect.
    """

    def __init__(
        self,
        dsn: str,
        name: str,
        read_only: bool = True,
        *,
        query_limits: QueryCostLimits | None = None,
    ) -> None:
        self.name = name
        self.read_only = read_only
        self._dsn = dsn
        # Phase 13: per-connector query bounds. A deployment that wants
        # a table allow-list or a tighter row cap for one specific
        # database passes its own limits here rather than changing a
        # global - so one permissive connector can never relax another.
        self._guard = SqlGuard(query_limits)

    def _engine_dialect_name(self) -> str:
        """The SQLAlchemy dialect name implied by this connector's DSN.

        Derived from the URL rather than by opening a connection: the
        guard needs the dialect on every `read()`, and paying for a
        connection just to learn something the DSN already states would
        be wasteful.
        """
        scheme = self._dsn.split("://", 1)[0]
        return scheme.split("+", 1)[0].lower()

    @property
    def dsn(self) -> str:
        """The DSN this connector was constructed with, exposed read-only.

        Lets a caller that already legitimately knows this connector's
        identity - e.g. `CapabilityFactory`, deciding whether a
        synthesized skill can reach this connector's real data using
        only the Python standard library (spec §8/§19/§20's sandbox
        constraint - see its own docstring) - inspect it without a
        separate, parallel channel to the same config value.
        """
        return self._dsn

    async def describe(self) -> ConnectorDescription:
        """`schema` is `{"dialect": <SQLAlchemy dialect name>, "tables": {...}}`.

        The `dialect` key exists for a real reason found in this phase's
        own testing, not speculatively: a synthesized skill's generated
        SQL needs to know it's targeting e.g. `"sqlite"` rather than
        `"postgresql"` (SQLite has no `date_trunc`, a Postgres-only
        function - a real Capability Factory run against this connector
        generated Postgres-flavored SQL and failed against a real SQLite
        DB before this field existed). `CapabilityFactory` JSON-dumps
        this whole dict into the synthesis prompt regardless of its
        internal shape, so surfacing the dialect here is what lets the
        model generate dialect-correct SQL without guessing from the
        DSN string alone. Nested under `"tables"` rather than a sibling
        of each table name, so a real table happening to be named
        `"dialect"` can never collide with this key.
        """
        engine = create_async_engine(self._dsn)
        try:
            dialect_name = engine.dialect.name
            async with engine.connect() as conn:
                tables = await conn.run_sync(_introspect)
        finally:
            await engine.dispose()
        return ConnectorDescription(kind="sql", schema={"dialect": dialect_name, "tables": tables})

    async def read(self, query: str, **params: Any) -> Any:
        """Runs `query`, after real SQL validation and row bounding.

        **Phase 13 (Prompt §8)**: the query is parsed with `sqlglot`
        against this connector's *actual* dialect before it is sent.
        That catches, with a specific error, what the old
        regex-prefix check could not: stacked statements
        (`SELECT 1; DROP TABLE users`), a mutation hidden behind a CTE,
        an unbounded scan, or a cartesian product. A `SELECT` with no
        `LIMIT` gets one injected at the guard's `max_rows`.

        The regex check is deliberately *kept* as a cheap first gate
        rather than replaced: if `sqlglot` ever fails to parse a
        dialect-specific form, the guard raises `unparseable` and the
        query is refused - so the two controls fail in the same safe
        direction rather than the parser becoming a single point of
        bypass.
        """
        if self.read_only and _MUTATING_QUERY.match(query):
            raise PermissionError(
                f"connector '{self.name}' is registered read-only; "
                f"refusing a query that looks like a mutation"
            )

        dialect = _SQLGLOT_DIALECTS.get(self._engine_dialect_name())
        if self.read_only:
            query = self._guard.validate_read(query, dialect=dialect)
        else:
            self._guard.assert_write_allowed(query, dialect=dialect, read_only=False)

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
            return HealthStatus(name=self.name, healthy=False, detail=describe_exception(exc))


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
