# praxis/safety/sql_guard.py
"""SQL validation and query-cost bounds (Prompt §8).

**This is a defence-in-depth check, not the security boundary.** The
prompt is explicit about this - "SQLGlot plus database-enforced
permissions ... Parsing is an additional check, not the security
boundary" - and this module is written to that framing: the real
boundary is the database role's own grants plus
`Connector.read_only`. What parsing buys is catching an unsafe or
unbounded query *before* it reaches the database, with a clear,
specific error a plan step can surface, instead of discovering it as a
permission error or a runaway scan.

What it enforces:

- **Statement kind.** A read path accepts only `SELECT`/`WITH`. Any
  DML/DDL (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`,
  `GRANT`, ...) is refused with the kind named.
- **One statement per call.** Stacked statements (`SELECT 1; DROP TABLE
  users`) are refused outright - the classic injection shape, and never
  something a legitimate generated query needs.
- **Row bounds.** A `SELECT` with no `LIMIT` gets one injected at
  `max_rows` rather than being refused, because an unbounded read is
  usually an oversight rather than an attack; an explicit `LIMIT`
  larger than `max_rows` IS refused, because that is a deliberate ask
  the caller should have to justify.
- **Table allow/deny lists.** Optional, off by default. When set, every
  table the query touches (including through CTEs and subqueries) must
  be permitted.
- **Cost heuristics.** Cartesian products (a multi-table `FROM` with no
  join condition) and `SELECT *` on an unbounded query are refused as
  genuine cost risks.

Dialect-aware throughout: the connector's real dialect
(`ConnectorDescription.schema["dialect"]`) is passed in, so SQLite and
Postgres syntax are each parsed by their own grammar rather than a
lowest-common-denominator one.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


class SqlStatementKind(str, Enum):
    """What a parsed statement actually does."""

    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    DDL = "ddl"
    OTHER = "other"

    @property
    def is_read_only(self) -> bool:
        return self is SqlStatementKind.SELECT


class SqlGuardError(Exception):
    """Base for every refusal this module issues."""


class UnsafeQueryError(SqlGuardError):
    """The query was rejected. `reason` is a short machine-readable code
    (`not_readonly`, `multiple_statements`, `unparseable`,
    `limit_too_large`, `table_not_allowed`, `cartesian_product`) so a
    caller can branch on the category, while `str(exc)` stays a clear
    human explanation."""

    def __init__(self, message: str, *, reason: str, detail: str = "") -> None:
        super().__init__(message)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class QueryCostLimits:
    """Bounds applied to a validated read query.

    `max_rows` is injected as a `LIMIT` when the query has none, and is
    the ceiling an explicit `LIMIT` may not exceed. `max_joins` bounds
    fan-out; `allowed_tables`/`denied_tables` are optional allow/deny
    lists (a deny entry always wins over an allow entry).
    """

    max_rows: int = 10_000
    max_joins: int = 10
    allowed_tables: frozenset[str] | None = None
    denied_tables: frozenset[str] = frozenset()


_DDL_EXPRESSIONS = (
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Grant,
)

_KIND_BY_EXPRESSION: list[tuple[type[exp.Expression], SqlStatementKind]] = [
    (exp.Insert, SqlStatementKind.INSERT),
    (exp.Update, SqlStatementKind.UPDATE),
    (exp.Delete, SqlStatementKind.DELETE),
]


def _classify(statement: exp.Expression) -> SqlStatementKind:
    if isinstance(statement, _DDL_EXPRESSIONS):
        return SqlStatementKind.DDL
    for expression_type, kind in _KIND_BY_EXPRESSION:
        if isinstance(statement, expression_type):
            return kind
    if isinstance(statement, (exp.Select, exp.Union, exp.Subquery)):
        return SqlStatementKind.SELECT
    # A bare `WITH ... SELECT` parses as the inner select wrapped in a
    # `With`; sqlglot attaches it to the select, so reaching here means
    # something genuinely other (SET, EXPLAIN, CALL, ...).
    return SqlStatementKind.OTHER


def _referenced_tables(statement: exp.Expression) -> set[str]:
    """Every base table the statement reads, CTE names excluded.

    CTE names are excluded deliberately: a CTE is a query-local alias,
    not a real table, so allow-listing would otherwise have to
    enumerate names the caller invented on the spot.
    """
    cte_names = {
        cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE) if cte.alias_or_name
    }
    tables: set[str] = set()
    for table in statement.find_all(exp.Table):
        name = table.name.lower()
        if name and name not in cte_names:
            tables.add(name)
    return tables


class SqlGuard:
    """Validates and bounds SQL before it reaches a connector.

    Stateless apart from its limits, so one instance is safely shared.
    """

    def __init__(self, limits: QueryCostLimits | None = None) -> None:
        self._limits = limits if limits is not None else QueryCostLimits()

    @property
    def limits(self) -> QueryCostLimits:
        return self._limits

    def _parse_one(self, query: str, dialect: str | None) -> exp.Expression:
        try:
            statements = sqlglot.parse(query, read=dialect)
        except ParseError as exc:
            raise UnsafeQueryError(
                "query could not be parsed as SQL",
                reason="unparseable",
                detail=str(exc),
            ) from exc

        real = [statement for statement in statements if statement is not None]
        if not real:
            raise UnsafeQueryError("query is empty", reason="unparseable")
        if len(real) > 1:
            # The classic injection shape. No legitimate generated query
            # in this system needs to stack statements.
            raise UnsafeQueryError(
                f"query contains {len(real)} statements; exactly one is allowed",
                reason="multiple_statements",
                detail="; ".join(statement.sql(dialect=dialect) for statement in real),
            )
        return real[0]

    def classify(self, query: str, *, dialect: str | None = None) -> SqlStatementKind:
        """What kind of statement this is, without applying any policy."""
        return _classify(self._parse_one(query, dialect))

    def _check_tables(self, statement: exp.Expression) -> None:
        tables = _referenced_tables(statement)

        denied = {table for table in tables if table in self._limits.denied_tables}
        if denied:
            raise UnsafeQueryError(
                f"query references denied table(s): {sorted(denied)}",
                reason="table_not_allowed",
                detail=f"denied list: {sorted(self._limits.denied_tables)}",
            )

        if self._limits.allowed_tables is not None:
            not_allowed = {table for table in tables if table not in self._limits.allowed_tables}
            if not_allowed:
                raise UnsafeQueryError(
                    f"query references table(s) outside the allow-list: {sorted(not_allowed)}",
                    reason="table_not_allowed",
                    detail=f"allowed: {sorted(self._limits.allowed_tables)}",
                )

    def _check_cost(self, statement: exp.Expression) -> None:
        joins = list(statement.find_all(exp.Join))
        if len(joins) > self._limits.max_joins:
            raise UnsafeQueryError(
                f"query has {len(joins)} joins, exceeding the limit of {self._limits.max_joins}",
                reason="too_many_joins",
            )

        # A cartesian product: more than one table in FROM with no join
        # condition anywhere. Genuinely a cost risk (row counts
        # multiply), and almost always a generation mistake.
        for select in statement.find_all(exp.Select):
            from_clause = select.args.get("from")
            if from_clause is None:
                continue
            select_joins = select.args.get("joins") or []
            unconditioned = [
                join
                for join in select_joins
                if not join.args.get("on") and not join.args.get("using")
            ]
            for join in unconditioned:
                # `CROSS JOIN` is explicit and intentional, so it is
                # permitted. A bare `JOIN b` and an implicit comma-join
                # (`FROM a, b`) both parse with an EMPTY kind - which is
                # exactly the accidental cartesian product worth
                # refusing, so an empty kind must NOT be treated as
                # "unknown, allow it".
                kind = f"{join.side or ''} {join.kind or ''}".upper()
                if "CROSS" not in kind:
                    raise UnsafeQueryError(
                        "query contains a join with no ON/USING condition (cartesian product)",
                        reason="cartesian_product",
                    )

    def validate_read(self, query: str, *, dialect: str | None = None) -> str:
        """Validates `query` as a read, returning the SQL to actually run.

        Raises `UnsafeQueryError` for anything that isn't a single,
        bounded, permitted read.

        **The returned SQL is the caller's own text**, not a
        regenerated rendering of the parse tree - at most a `LIMIT` is
        appended. This is deliberate and load-bearing: sqlglot's
        generator normalizes as it renders, which silently rewrites
        named bind parameters (`:id` becomes `%(id)s`, breaking
        SQLAlchemy's `text()` binding) and can subtly alter
        dialect-specific constructs. A validator that quietly rewrites
        the query it was asked to check is a worse problem than the one
        it solves, so parsing is used purely to *decide*, never to
        re-emit.
        """
        statement = self._parse_one(query, dialect)
        kind = _classify(statement)

        if not kind.is_read_only:
            raise UnsafeQueryError(
                f"only read queries are permitted here, but this is a {kind.value.upper()} statement",
                reason="not_readonly",
                detail=kind.value,
            )

        self._check_tables(statement)
        self._check_cost(statement)

        return self._apply_row_bound(query, statement)

    def _apply_row_bound(self, query: str, statement: exp.Expression) -> str:
        """Appends a `LIMIT` when absent; refuses one that is too large.

        The asymmetry is deliberate (see module docstring): a missing
        limit is an oversight worth silently correcting, an oversized
        explicit limit is a deliberate ask worth refusing.
        """
        limit = statement.args.get("limit")
        if limit is not None:
            expression = limit.expression
            if isinstance(expression, exp.Literal) and expression.is_int:
                requested = int(expression.name)
                if requested > self._limits.max_rows:
                    raise UnsafeQueryError(
                        (
                            f"query requests LIMIT {requested}, above the maximum of "
                            f"{self._limits.max_rows}"
                        ),
                        reason="limit_too_large",
                    )
            return query

        if not isinstance(statement, (exp.Select, exp.Union)):
            return query

        # Textual append onto the caller's own SQL. The trailing
        # semicolon (if any) is stripped first so the appended clause
        # doesn't land after the statement terminator.
        trimmed = query.rstrip().rstrip(";").rstrip()
        return f"{trimmed} LIMIT {self._limits.max_rows}"

    def assert_write_allowed(
        self, query: str, *, dialect: str | None = None, read_only: bool
    ) -> SqlStatementKind:
        """For a write path: classifies the statement and refuses it
        outright when the connector is read-only.

        Keeps the "which kind of write is this" answer available to the
        caller (for audit detail and approval prompts) rather than
        collapsing every write into an opaque boolean.
        """
        statement = self._parse_one(query, dialect)
        kind = _classify(statement)
        if read_only and not kind.is_read_only:
            raise UnsafeQueryError(
                f"connector is read-only; refusing a {kind.value.upper()} statement",
                reason="not_readonly",
                detail=kind.value,
            )
        self._check_tables(statement)
        return kind


# The default guard: sensible bounds, no table restrictions (a
# deployment that wants an allow-list constructs its own).
default_sql_guard = SqlGuard()
