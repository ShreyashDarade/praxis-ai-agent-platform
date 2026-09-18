# praxis/agents/specialists/sql_analyst.py
"""`sql_analyst`: writes and executes bounded, dialect-correct SQL.

Prompt §7's catalogue: a `safe-query-analysis` skill owned by a
"Query analyst" whose job is to *"validate and execute bounded
queries"*.

**Three controls, applied in this order, and the order matters:**

1. **Dialect correctness.** The query is written for the dialect the
   schema analyst actually reported, not the one the user's phrasing
   implies. This is a real failure mode already seen in this codebase:
   an intent that casually says "Postgres" against a SQLite-backed
   connector produced `date_trunc`, which does not exist there.
2. **Validation before execution** (`praxis.safety.sql_guard`) -
   catches stacked statements, hidden mutations, unbounded scans and
   accidental cartesian products *before* anything reaches the
   database.
3. **Execution through the connector**, so the connector's own
   read-only enforcement still applies. The guard is defence in
   depth, not the boundary.

The agent reports the exact SQL it ran as evidence. A number on a
dashboard whose query nobody can see is not auditable, and the
dashboard's provenance record is built from this.
"""
from __future__ import annotations

from praxis.agents.budget import Budget
from praxis.agents.contract import SpecialistResult
from praxis.agents.subagent import AgentContext, AgentManifest, SubAgent, register_agent
from praxis.safety.sql_guard import QueryCostLimits, SqlGuard, UnsafeQueryError

# SQLAlchemy dialect name -> sqlglot dialect name, mirroring
# `praxis.connectors.sql.sql_connector`'s own mapping.
_SQLGLOT_DIALECTS = {
    "postgresql": "postgres",
    "postgres": "postgres",
    "sqlite": "sqlite",
    "mysql": "mysql",
    "mssql": "tsql",
    "oracle": "oracle",
    "snowflake": "snowflake",
    "bigquery": "bigquery",
    "duckdb": "duckdb",
    "clickhouse": "clickhouse",
}


class SqlAnalystAgent(SubAgent):
    """Validates and runs one bounded read query."""

    manifest = AgentManifest(
        name="sql_analyst",
        description=(
            "Validates a SQL query against the connector's real dialect and cost bounds, "
            "then executes it read-only and returns the rows"
        ),
        tools=("query_connector",),
        required_permissions=("connector:read",),
        model_purpose="planning",
        default_budget=Budget(max_llm_calls=0),
        keywords=("sql", "query", "rows", "analyse"),
    )

    async def execute(self, context: AgentContext) -> SpecialistResult:
        inputs = context.contract.inputs
        connector = inputs.get("connector")
        query = str(inputs.get("query") or "").strip()

        if connector is None or not query:
            return SpecialistResult(
                task_id=context.contract.task_id,
                succeeded=False,
                errors=["sql_analyst requires both a 'connector' and a 'query' input"],
            )

        dialect = _SQLGLOT_DIALECTS.get(str(inputs.get("dialect") or "").lower())
        guard = SqlGuard(
            QueryCostLimits(
                max_rows=int(inputs.get("max_rows", 10_000)),
                allowed_tables=(
                    frozenset(inputs["allowed_tables"])
                    if inputs.get("allowed_tables")
                    else None
                ),
            )
        )

        limitations: list[str] = []
        if dialect is None and inputs.get("dialect"):
            limitations.append(
                f"dialect '{inputs['dialect']}' is not one this guard knows; the query was "
                "validated against standard SQL instead"
            )

        try:
            bounded_query = guard.validate_read(query, dialect=dialect)
        except UnsafeQueryError as exc:
            # Refused before execution - the whole point of validating
            # first. The reason code travels so a caller can tell a
            # policy refusal from a database error.
            return SpecialistResult(
                task_id=context.contract.task_id,
                succeeded=False,
                errors=[f"query refused ({exc.reason}): {exc}"],
                evidence=[{"kind": "rejected_query", "query": query, "reason": exc.reason}],
            )

        if bounded_query != query:
            limitations.append(
                "the query had no LIMIT, so one was applied; results may be truncated"
            )

        try:
            rows = await connector.read(bounded_query)
        except Exception as exc:  # noqa: BLE001 - reported, not raised at the supervisor
            return SpecialistResult(
                task_id=context.contract.task_id,
                succeeded=False,
                errors=[f"query failed against '{connector.name}': {exc}"],
                evidence=[{"kind": "executed_query", "query": bounded_query}],
            )

        rows = list(rows or [])
        if not rows:
            limitations.append("the query returned no rows")

        return SpecialistResult(
            task_id=context.contract.task_id,
            succeeded=True,
            results={
                "rows": rows,
                "row_count": len(rows),
                "columns": sorted(rows[0]) if rows else [],
                "query": bounded_query,
            },
            evidence=[
                {
                    "kind": "executed_query",
                    "connector": connector.name,
                    "query": bounded_query,
                    "dialect": dialect,
                    "row_count": len(rows),
                }
            ],
            limitations=limitations,
            spend=context.budget.snapshot(),
        )


register_agent(SqlAnalystAgent())
