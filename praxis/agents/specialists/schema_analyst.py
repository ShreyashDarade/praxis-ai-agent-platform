# praxis/agents/specialists/schema_analyst.py
"""`schema_analyst`: inspects a connector's schema and data safely.

Prompt §2's dashboard lifecycle opens with *"retrieve schemas,
metadata, relationships, table statistics, and sample data safely"*,
and §7's catalogue names a `schema-profiling` skill owned by a
"Schema analyst".

**Why a specialist rather than a skill.** A skill would be
`describe() -> schema`. This agent has to *judge*: which tables are
relevant to the objective, which columns look like measures versus
dimensions, whether a column is a usable time axis. That judgement is
what the downstream SQL analyst and chart designer consume, and it is
why the output carries `limitations` - "I could not tell whether
`status` is an enum" is information the next specialist needs, not a
failure.

**Safety.** Profiling reads only through the connector's own
`describe()` and bounded `read()` calls, so the connector's read-only
enforcement and the SQL guard's row bounds both still apply. Sample
rows are capped hard: a profile that pulls a million rows to "see the
data" would be its own incident.
"""
from __future__ import annotations

from typing import Any

from praxis.agents.budget import Budget
from praxis.agents.contract import SpecialistResult
from praxis.agents.subagent import AgentContext, AgentManifest, SubAgent, register_agent

# A profile needs shape, not volume. Ten rows shows nulls, formats,
# and cardinality hints; a thousand shows the same thing at a hundred
# times the cost and risk.
_SAMPLE_ROW_LIMIT = 10

# Column names that conventionally indicate a time axis. A heuristic,
# and reported as such - the agent says which columns it *believes*
# are temporal rather than asserting it.
_TIME_HINTS = (
    "date", "time", "created", "updated", "occurred", "timestamp", "day", "week", "month",
)

# Types that can be aggregated numerically.
_MEASURE_TYPES = (
    "int", "numeric", "decimal", "float", "double", "real", "money", "bigint", "smallint",
)


def _looks_temporal(name: str, type_name: str) -> bool:
    lowered = name.lower()
    return "date" in type_name.lower() or "time" in type_name.lower() or any(
        hint in lowered for hint in _TIME_HINTS
    )


def _looks_like_measure(name: str, type_name: str) -> bool:
    return any(candidate in type_name.lower() for candidate in _MEASURE_TYPES)


class SchemaAnalystAgent(SubAgent):
    """Profiles a connector's schema for downstream specialists."""

    manifest = AgentManifest(
        name="schema_analyst",
        description=(
            "Inspects a connector's tables, columns and sample rows, and reports which "
            "look like measures, dimensions and time axes"
        ),
        tools=("query_connector",),
        required_permissions=("connector:read",),
        model_purpose="planning",
        default_budget=Budget(max_llm_calls=0),
        keywords=("schema", "profile", "tables", "columns", "introspect"),
    )

    async def execute(self, context: AgentContext) -> SpecialistResult:
        connector = context.contract.inputs.get("connector")
        if connector is None:
            return SpecialistResult(
                task_id=context.contract.task_id,
                succeeded=False,
                errors=["schema_analyst requires a 'connector' input"],
            )

        limitations: list[str] = []
        evidence: list[dict[str, Any]] = []

        try:
            description = await connector.describe()
        except Exception as exc:  # noqa: BLE001 - a specialist reports, it does not crash its supervisor
            return SpecialistResult(
                task_id=context.contract.task_id,
                succeeded=False,
                errors=[f"could not introspect connector '{connector.name}': {exc}"],
            )

        schema = description.schema or {}
        dialect = schema.get("dialect")
        tables = schema.get("tables") or {}
        if not tables:
            limitations.append(
                "the connector reported no tables; it may be empty or may not support "
                "schema introspection"
            )

        profiled: dict[str, Any] = {}
        for table_name, columns in tables.items():
            column_list = list(columns or [])
            measures = [
                column["name"]
                for column in column_list
                if _looks_like_measure(column.get("name", ""), str(column.get("type", "")))
            ]
            temporal = [
                column["name"]
                for column in column_list
                if _looks_temporal(column.get("name", ""), str(column.get("type", "")))
            ]
            dimensions = [
                column["name"]
                for column in column_list
                if column["name"] not in measures and column["name"] not in temporal
            ]
            profiled[table_name] = {
                "columns": column_list,
                "likely_measures": measures,
                "likely_time_axes": temporal,
                "likely_dimensions": dimensions,
            }
            if not measures:
                limitations.append(
                    f"table '{table_name}' has no obviously numeric column, so a metric over "
                    "it will need an explicit definition (e.g. COUNT(*))"
                )

        # Sample rows, strictly bounded - see the module docstring.
        sample_of = context.contract.inputs.get("sample_table")
        samples: list[dict[str, Any]] = []
        if sample_of and sample_of in tables:
            try:
                samples = await connector.read(
                    f"SELECT * FROM {sample_of} LIMIT {_SAMPLE_ROW_LIMIT}"
                )
                evidence.append(
                    {
                        "kind": "sample_rows",
                        "table": sample_of,
                        "row_count": len(samples),
                        "limit": _SAMPLE_ROW_LIMIT,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                limitations.append(f"could not sample '{sample_of}': {exc}")

        evidence.append(
            {"kind": "schema", "connector": connector.name, "table_count": len(tables)}
        )

        return SpecialistResult(
            task_id=context.contract.task_id,
            succeeded=True,
            results={
                "dialect": dialect,
                "tables": profiled,
                "table_names": sorted(tables),
                "samples": samples,
            },
            evidence=evidence,
            limitations=limitations,
            spend=context.budget.snapshot(),
        )


register_agent(SchemaAnalystAgent())
