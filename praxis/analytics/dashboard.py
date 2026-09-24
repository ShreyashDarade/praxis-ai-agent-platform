# praxis/analytics/dashboard.py
"""The declarative dashboard specification (Prompt §2).

Prompt §2 is emphatic about the form this must take: *"Use a validated
declarative dashboard specification, not arbitrary model-generated
browser JavaScript. Define chart schemas, renderer capability
negotiation, accessible fallbacks, query provenance, refresh
semantics, and export artifacts."*

So a dashboard here is **data, not code**. An LLM produces a
`DashboardSpec` (JSON), which is validated against the registered
chart types and the semantic layer before anything renders. Nothing
the model emits is ever executed - the renderer only ever interprets a
spec it has already validated. That is the difference between "the
model suggested a chart" and "the model can run code in the operator's
browser".

Each piece the prompt names:

- **Chart schemas** - `PanelSpec.chart_type` must name a type the
  visualizer actually registers, and `encoding` is validated against
  that type's real required roles.
- **Renderer capability negotiation** - `negotiate` picks the best
  chart a given renderer supports, falling back down a declared
  preference chain rather than failing outright.
- **Accessible fallbacks** - every panel carries `alt_text` and can
  degrade to a `table`, because a PNG chart is unreadable to a screen
  reader and a spec that cannot degrade is not accessible.
- **Query provenance** - each panel records the metric/dimension names
  and the exact query that produced it, so a number on a dashboard can
  always be traced to how it was computed.
- **Refresh semantics** - declared per panel, with the staleness
  question answered from the semantic layer rather than guessed.
- **Export artifacts** - `to_dict`/`from_dict` round-trip losslessly,
  which is what makes a dashboard storable, diffable, and re-renderable.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class RefreshMode(str, Enum):
    """When a panel's data is recomputed."""

    MANUAL = "manual"
    ON_LOAD = "on_load"
    SCHEDULED = "scheduled"


@dataclass
class FilterSpec:
    """A dashboard-level filter applied to every panel that accepts it.

    `dimension` names a semantic-layer dimension rather than a raw
    column: a filter written against a column would bypass the
    semantic layer's grain and join validation, which is precisely
    what that layer exists to prevent.
    """

    dimension: str
    operator: str = "eq"
    value: Any = None
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "operator": self.operator,
            "value": self.value,
            "label": self.label or self.dimension,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FilterSpec:
        return cls(
            dimension=payload["dimension"],
            operator=payload.get("operator", "eq"),
            value=payload.get("value"),
            label=payload.get("label", ""),
        )


@dataclass
class QueryProvenance:
    """How a panel's numbers were actually produced (Prompt §2).

    Recorded at render time, not declared up front: the point is to
    answer "where did this number come from" after the fact, which
    means capturing what really ran rather than what was intended.
    """

    connector: str = ""
    query: str = ""
    metrics: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    row_count: int | None = None
    executed_at: datetime | None = None
    validation_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector": self.connector,
            "query": self.query,
            "metrics": self.metrics,
            "dimensions": self.dimensions,
            "row_count": self.row_count,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "validation_warnings": self.validation_warnings,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> QueryProvenance:
        executed = payload.get("executed_at")
        return cls(
            connector=payload.get("connector", ""),
            query=payload.get("query", ""),
            metrics=list(payload.get("metrics") or []),
            dimensions=list(payload.get("dimensions") or []),
            row_count=payload.get("row_count"),
            executed_at=datetime.fromisoformat(executed) if executed else None,
            validation_warnings=list(payload.get("validation_warnings") or []),
        )


@dataclass
class PanelSpec:
    """One chart/table/KPI on a dashboard."""

    title: str
    chart_type: str
    encoding: dict[str, str] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    panel_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    connector: str = ""
    query: str = ""
    filters: list[FilterSpec] = field(default_factory=list)
    refresh: RefreshMode = RefreshMode.ON_LOAD
    refresh_interval_seconds: int | None = None
    # Accessibility is a required property of the spec, not a
    # nice-to-have: a rendered PNG is opaque to a screen reader, so a
    # panel that cannot describe itself is not shippable.
    alt_text: str = ""
    # The chart types this panel may degrade to, in preference order,
    # when a renderer cannot draw `chart_type`. `table` is the
    # universal floor - any tabular result can always be shown as one.
    fallback_chart_types: list[str] = field(default_factory=lambda: ["table"])
    provenance: QueryProvenance = field(default_factory=QueryProvenance)
    width: int = 6
    height: int = 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "title": self.title,
            "chart_type": self.chart_type,
            "encoding": self.encoding,
            "metrics": self.metrics,
            "dimensions": self.dimensions,
            "connector": self.connector,
            "query": self.query,
            "filters": [f.to_dict() for f in self.filters],
            "refresh": self.refresh.value,
            "refresh_interval_seconds": self.refresh_interval_seconds,
            "alt_text": self.alt_text,
            "fallback_chart_types": self.fallback_chart_types,
            "provenance": self.provenance.to_dict(),
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PanelSpec:
        return cls(
            panel_id=payload.get("panel_id") or str(uuid.uuid4()),
            title=payload["title"],
            chart_type=payload["chart_type"],
            encoding=dict(payload.get("encoding") or {}),
            metrics=list(payload.get("metrics") or []),
            dimensions=list(payload.get("dimensions") or []),
            connector=payload.get("connector", ""),
            query=payload.get("query", ""),
            filters=[FilterSpec.from_dict(f) for f in payload.get("filters") or []],
            refresh=RefreshMode(payload.get("refresh", RefreshMode.ON_LOAD.value)),
            refresh_interval_seconds=payload.get("refresh_interval_seconds"),
            alt_text=payload.get("alt_text", ""),
            fallback_chart_types=list(payload.get("fallback_chart_types") or ["table"]),
            provenance=QueryProvenance.from_dict(payload.get("provenance") or {}),
            width=payload.get("width", 6),
            height=payload.get("height", 4),
        )


@dataclass
class DashboardSpec:
    """A complete, validated, storable dashboard definition."""

    title: str
    panels: list[PanelSpec] = field(default_factory=list)
    dashboard_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    filters: list[FilterSpec] = field(default_factory=list)
    tenant_id: str = ""
    owner: str = ""
    # Spec-format version, so a stored dashboard read back by a later
    # release can be migrated rather than silently misinterpreted.
    spec_version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dashboard_id": self.dashboard_id,
            "spec_version": self.spec_version,
            "title": self.title,
            "description": self.description,
            "tenant_id": self.tenant_id,
            "owner": self.owner,
            "filters": [f.to_dict() for f in self.filters],
            "panels": [panel.to_dict() for panel in self.panels],
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DashboardSpec:
        created = payload.get("created_at")
        return cls(
            dashboard_id=payload.get("dashboard_id") or str(uuid.uuid4()),
            spec_version=payload.get("spec_version", 1),
            title=payload["title"],
            description=payload.get("description", ""),
            tenant_id=payload.get("tenant_id", ""),
            owner=payload.get("owner", ""),
            filters=[FilterSpec.from_dict(f) for f in payload.get("filters") or []],
            panels=[PanelSpec.from_dict(p) for p in payload.get("panels") or []],
            created_at=(
                datetime.fromisoformat(created) if created else datetime.now(UTC)
            ),
        )


class DashboardValidationError(Exception):
    """The spec is not renderable. Carries every problem, not the first."""

    def __init__(self, message: str, *, problems: list[str]) -> None:
        super().__init__(message)
        self.problems = problems


def validate_dashboard_spec(
    spec: DashboardSpec,
    *,
    supported_chart_types: set[str],
    required_roles_for: dict[str, tuple[str, ...]] | None = None,
) -> list[str]:
    """Validates a spec against what the renderer can actually draw.

    Returns the list of problems (empty when valid) rather than
    raising, because a dashboard-authoring flow wants to show an
    author everything wrong at once. `assert_valid_dashboard_spec`
    is the raising variant for a call site that cannot proceed.
    """
    problems: list[str] = []
    roles = required_roles_for or {}

    if not spec.panels:
        problems.append("dashboard has no panels")

    seen_ids: set[str] = set()
    for panel in spec.panels:
        if panel.panel_id in seen_ids:
            problems.append(f"duplicate panel_id '{panel.panel_id}'")
        seen_ids.add(panel.panel_id)

        if panel.chart_type not in supported_chart_types:
            problems.append(
                f"panel '{panel.title}' uses unsupported chart_type "
                f"'{panel.chart_type}' (supported: {sorted(supported_chart_types)})"
            )
        else:
            missing = [
                role
                for role in roles.get(panel.chart_type, ())
                if role not in panel.encoding
            ]
            if missing:
                problems.append(
                    f"panel '{panel.title}' ({panel.chart_type}) is missing required "
                    f"encoding role(s) {missing}"
                )

        if not panel.alt_text:
            problems.append(
                f"panel '{panel.title}' has no alt_text; a rendered chart is opaque to a "
                "screen reader without one"
            )

        if panel.refresh is RefreshMode.SCHEDULED and not panel.refresh_interval_seconds:
            problems.append(
                f"panel '{panel.title}' declares scheduled refresh but no interval"
            )

        unsupported_fallbacks = [
            fallback
            for fallback in panel.fallback_chart_types
            if fallback not in supported_chart_types
        ]
        if unsupported_fallbacks:
            problems.append(
                f"panel '{panel.title}' declares unsupported fallback chart type(s) "
                f"{unsupported_fallbacks}"
            )

    return problems


def assert_valid_dashboard_spec(
    spec: DashboardSpec,
    *,
    supported_chart_types: set[str],
    required_roles_for: dict[str, tuple[str, ...]] | None = None,
) -> None:
    """Raises `DashboardValidationError` if `spec` is not renderable."""
    problems = validate_dashboard_spec(
        spec,
        supported_chart_types=supported_chart_types,
        required_roles_for=required_roles_for,
    )
    if problems:
        raise DashboardValidationError(
            f"{len(problems)} problem(s) with dashboard '{spec.title}'", problems=problems
        )


def negotiate(panel: PanelSpec, renderer_supports: set[str]) -> str:
    """Picks the chart type to actually draw for this renderer.

    Renderer capability negotiation (Prompt §2): a renderer that cannot
    draw a Sankey should fall back down the panel's declared
    preference chain rather than failing the whole dashboard. Raises
    only when even `table` is unavailable, which would mean a renderer
    that can draw nothing at all.
    """
    if panel.chart_type in renderer_supports:
        return panel.chart_type
    for fallback in panel.fallback_chart_types:
        if fallback in renderer_supports:
            return fallback
    raise DashboardValidationError(
        (
            f"renderer supports none of panel '{panel.title}'s chart types "
            f"({[panel.chart_type, *panel.fallback_chart_types]})"
        ),
        problems=[f"no renderable chart type for panel '{panel.panel_id}'"],
    )

