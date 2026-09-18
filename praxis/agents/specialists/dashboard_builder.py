# praxis/agents/specialists/dashboard_builder.py
"""`dashboard_builder`: assembles and validates a dashboard spec.

The last specialist in Prompt §2's pipeline, and the one that turns
the others' output into the thing the user asked for: *"save the
dashboard configuration and allow scheduled refreshes"*.

**It builds data, never code.** The output is a `DashboardSpec` -
validated against the chart types the renderer actually supports
before it is returned, so an unrenderable dashboard is a caught error
rather than a broken page. This is the concrete expression of the
brief's rule against *"arbitrary model-generated browser
JavaScript"*.

Each panel carries the provenance its SQL analyst reported - the
exact query, row count, connector, and any validation warnings - so a
number on the finished dashboard can always be traced back to how it
was computed and what was doubted about it.
"""
from __future__ import annotations

from datetime import UTC, datetime

from praxis.agents.budget import Budget
from praxis.agents.contract import SpecialistResult
from praxis.agents.subagent import AgentContext, AgentManifest, SubAgent, register_agent
from praxis.analytics.dashboard import (
    DashboardSpec,
    PanelSpec,
    QueryProvenance,
    RefreshMode,
    validate_dashboard_spec,
)
from praxis.analytics.visualize import PlotlyVisualizer


class DashboardBuilderAgent(SubAgent):
    """Assembles validated panels into a saveable dashboard spec."""

    manifest = AgentManifest(
        name="dashboard_builder",
        description=(
            "Assembles panels into a validated, saveable dashboard specification with "
            "per-panel query provenance and accessible descriptions"
        ),
        tools=("create_chart",),
        required_permissions=("dashboard:write",),
        model_purpose="planning",
        default_budget=Budget(max_llm_calls=0),
        keywords=("dashboard", "assemble", "panels", "save"),
    )

    async def execute(self, context: AgentContext) -> SpecialistResult:
        inputs = context.contract.inputs
        title = str(inputs.get("title") or context.contract.objective)[:512]
        panel_inputs = list(inputs.get("panels") or [])

        if not panel_inputs:
            return SpecialistResult(
                task_id=context.contract.task_id,
                succeeded=False,
                errors=["dashboard_builder needs at least one panel to assemble"],
            )

        limitations: list[str] = []
        panels: list[PanelSpec] = []

        for raw in panel_inputs:
            refresh_seconds = raw.get("refresh_interval_seconds")
            panels.append(
                PanelSpec(
                    title=str(raw.get("title") or "Untitled panel"),
                    chart_type=str(raw.get("chart_type") or "table"),
                    encoding=dict(raw.get("encoding") or {}),
                    metrics=list(raw.get("metrics") or []),
                    dimensions=list(raw.get("dimensions") or []),
                    connector=str(raw.get("connector") or ""),
                    query=str(raw.get("query") or ""),
                    # Alt text is required by validation; the chart
                    # designer supplies it, and a missing one is
                    # surfaced rather than silently defaulted to
                    # something meaningless.
                    alt_text=str(raw.get("alt_text") or ""),
                    refresh=(
                        RefreshMode.SCHEDULED if refresh_seconds else RefreshMode.ON_LOAD
                    ),
                    refresh_interval_seconds=refresh_seconds,
                    provenance=QueryProvenance(
                        connector=str(raw.get("connector") or ""),
                        query=str(raw.get("query") or ""),
                        metrics=list(raw.get("metrics") or []),
                        dimensions=list(raw.get("dimensions") or []),
                        row_count=raw.get("row_count"),
                        executed_at=datetime.now(UTC),
                        validation_warnings=list(raw.get("validation_warnings") or []),
                    ),
                )
            )

        spec = DashboardSpec(
            title=title,
            description=str(inputs.get("description") or ""),
            tenant_id=context.tenant_id or context.contract.tenant_id,
            owner=str(inputs.get("owner") or ""),
            panels=panels,
        )

        supported = set(PlotlyVisualizer.supported_chart_types())
        problems = validate_dashboard_spec(
            spec,
            supported_chart_types=supported,
            required_roles_for=PlotlyVisualizer.required_roles(),
        )
        if problems:
            # An unrenderable dashboard is a caught error, not a
            # broken page discovered by whoever opens it.
            return SpecialistResult(
                task_id=context.contract.task_id,
                succeeded=False,
                errors=problems,
                results={"spec": spec.to_dict()},
            )

        return SpecialistResult(
            task_id=context.contract.task_id,
            succeeded=True,
            results={"spec": spec.to_dict(), "panel_count": len(panels)},
            evidence=[
                {
                    "kind": "dashboard",
                    "title": title,
                    "panels": [panel.title for panel in panels],
                }
            ],
            limitations=limitations,
            spend=context.budget.snapshot(),
        )


register_agent(DashboardBuilderAgent())
