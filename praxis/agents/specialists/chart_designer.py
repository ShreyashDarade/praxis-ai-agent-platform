# praxis/agents/specialists/chart_designer.py
"""`chart_designer`: picks a chart type and encoding that fit the data.

Prompt §2 lists a "chart designer" among the specialists a dashboard
request should spawn.

**Deterministic, not generative, and that is the point.** The obvious
implementation asks a model "what chart should I use?". This one
decides from the *shape of the data* instead:

- a single row with one measure is a KPI card, not a one-bar chart;
- a temporal dimension means a line;
- a small categorical dimension means a bar;
- a large categorical dimension means a bar that has to be truncated,
  and says so rather than rendering 4,000 unreadable ticks;
- two measures and no dimension is a scatter;
- anything it cannot classify falls back to a table, which can always
  represent tabular data faithfully.

Choosing deterministically means the same data always produces the
same chart, which makes a dashboard reproducible and a regression
visible. It also costs nothing, so a 12-panel dashboard does not make
12 model calls to answer a question the data already answers.
"""
from __future__ import annotations

from typing import Any

from praxis.agents.budget import Budget
from praxis.agents.contract import SpecialistResult
from praxis.agents.subagent import AgentContext, AgentManifest, SubAgent, register_agent

# Above this many distinct categories a bar chart stops being
# readable; the designer still picks bar but flags the truncation
# rather than silently emitting an unreadable axis.
_CATEGORY_READABILITY_LIMIT = 30

_TIME_HINTS = ("date", "time", "week", "month", "day", "quarter", "year", "period")


def _is_temporal(column: str) -> bool:
    lowered = column.lower()
    return any(hint in lowered for hint in _TIME_HINTS)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _classify_columns(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Splits columns into (measures, dimensions) from real values.

    Inferred from the data rather than from declared types, because
    this runs on a query result whose column types are whatever the
    driver produced.
    """
    if not rows:
        return [], []
    first = rows[0]
    measures = [name for name, value in first.items() if _is_numeric(value)]
    dimensions = [name for name in first if name not in measures]
    return measures, dimensions


class ChartDesignerAgent(SubAgent):
    """Chooses a chart type and encoding for a result set."""

    manifest = AgentManifest(
        name="chart_designer",
        description=(
            "Chooses a chart type and role encoding from the shape of a result set, "
            "deterministically and without a model call"
        ),
        tools=("create_chart",),
        model_purpose="planning",
        default_budget=Budget(max_llm_calls=0),
        keywords=("chart", "visualise", "plot", "graph", "encoding"),
    )

    async def execute(self, context: AgentContext) -> SpecialistResult:
        rows = list(context.contract.inputs.get("rows") or [])
        if not rows:
            return SpecialistResult(
                task_id=context.contract.task_id,
                succeeded=False,
                errors=["chart_designer needs at least one row to choose a chart for"],
            )

        preferred = context.contract.inputs.get("preferred_chart_type")
        measures, dimensions = _classify_columns(rows)
        limitations: list[str] = []

        if not measures:
            # Nothing numeric to plot - a table is the honest answer
            # rather than inventing a count.
            return SpecialistResult(
                task_id=context.contract.task_id,
                succeeded=True,
                results={
                    "chart_type": "table",
                    "encoding": {},
                    "alt_text": f"Table of {len(rows)} rows with no numeric column to chart",
                    "rationale": "no numeric column, so the data is shown as a table",
                },
                limitations=["no numeric column was found, so no chart could be drawn"],
                spend=context.budget.snapshot(),
            )

        measure = measures[0]

        # One row, one measure: a KPI card states the number plainly.
        # A single-bar chart would be strictly worse.
        if len(rows) == 1 and not dimensions:
            chart_type = "kpi_card"
            encoding = {"value": measure}
            rationale = "a single numeric value reads best as a KPI card"
        elif dimensions:
            dimension = next(
                (name for name in dimensions if _is_temporal(name)), dimensions[0]
            )
            distinct = len({str(row.get(dimension)) for row in rows})
            if _is_temporal(dimension):
                chart_type = "line"
                rationale = f"'{dimension}' is a time axis, so a trend line fits"
            else:
                chart_type = "bar"
                rationale = f"'{dimension}' is categorical, so a bar comparison fits"
                if distinct > _CATEGORY_READABILITY_LIMIT:
                    limitations.append(
                        f"'{dimension}' has {distinct} distinct values, past the "
                        f"{_CATEGORY_READABILITY_LIMIT} a bar chart stays readable at; "
                        "consider aggregating or filtering"
                    )
            encoding = {"x": dimension, "y": measure}
        elif len(measures) >= 2:
            chart_type = "scatter"
            encoding = {"x": measures[0], "y": measures[1]}
            rationale = "two numeric columns and no dimension suggests a relationship plot"
        else:
            chart_type = "table"
            encoding = {}
            rationale = "the shape did not match a known chart, so a table is used"
            limitations.append("the data shape was not recognised; falling back to a table")

        if preferred and preferred != chart_type:
            # The caller's preference wins, but the mismatch is
            # recorded - silently overriding either way would hide a
            # real disagreement about how to read the data.
            limitations.append(
                f"caller requested '{preferred}' but the data shape suggests "
                f"'{chart_type}'; using the requested type"
            )
            chart_type = preferred

        alt_text = _describe(chart_type, encoding, rows)

        return SpecialistResult(
            task_id=context.contract.task_id,
            succeeded=True,
            results={
                "chart_type": chart_type,
                "encoding": encoding,
                "alt_text": alt_text,
                "rationale": rationale,
                "measures": measures,
                "dimensions": dimensions,
            },
            evidence=[{"kind": "chart_choice", "chart_type": chart_type, "rows": len(rows)}],
            limitations=limitations,
            spend=context.budget.snapshot(),
        )


def _describe(chart_type: str, encoding: dict[str, str], rows: list[dict[str, Any]]) -> str:
    """Builds the accessible description.

    Generated here rather than left to the caller because a panel
    without alt text is rejected by dashboard validation - making it
    the designer's responsibility means it can never be forgotten.
    """
    if chart_type == "kpi_card":
        value = rows[0].get(encoding.get("value", ""))
        return f"Key figure: {encoding.get('value')} is {value}"
    if chart_type == "table":
        return f"Table of {len(rows)} rows"
    x = encoding.get("x", "")
    y = encoding.get("y", "")
    return f"{chart_type.capitalize()} chart of {y} by {x}, {len(rows)} data points"


register_agent(ChartDesignerAgent())
