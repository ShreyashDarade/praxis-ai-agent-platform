# praxis/analytics/refresh.py
"""Actually re-executing a saved dashboard's panels (brief §2's "save the
dashboard configuration and allow scheduled refreshes").

Saving a dashboard and stamping `last_refreshed_at` is not a refresh: a
dashboard whose timestamp moves while its numbers do not is *worse*
than one that admits it is stale, because the timestamp is the thing a
reader trusts when deciding whether to act on a figure. This module
runs each panel's query for real and records what actually ran.

Three properties it is built around:

**One bad panel does not blank the dashboard.** Panel failures are
captured per panel, not raised. A dashboard with nine good panels and
one broken query should render nine panels and one clearly-failed one -
the alternative (the whole refresh raising) means a single dropped
table takes down an executive's whole view, which is both worse
operationally and worse for trust than a visibly broken tile.

**Provenance is recorded from the execution, not from the spec.** The
`QueryProvenance` written back carries the query the guard actually
approved (which may carry an appended row bound), the real row count,
and the real timestamp - so "where did this number come from" is
answered by what ran, not by what was intended. This is the whole point
of `QueryProvenance` existing as a separate record from `PanelSpec.query`.

**Refresh is read-only, always.** Every panel query goes through
`SqlGuard.validate_read` before it reaches a connector, so a saved
dashboard cannot become a persistence mechanism for a mutation: the
spec is stored data, and stored data that later executes with write
power is the shape of a stored-injection bug. A panel whose query is
not a read is failed, not run.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

import structlog

from praxis.analytics.dashboard import DashboardSpec, PanelSpec, QueryProvenance, RefreshMode
from praxis.core.interfaces import Connector
from praxis.safety.sql_guard import SqlGuard, SqlGuardError

_logger = structlog.get_logger(__name__)

# A refresh is a background/scheduled operation, so a single wedged
# connector must not hold the whole dashboard open indefinitely. This is
# a ceiling, not a target - a panel that needs longer than this is a
# panel whose query needs fixing.
DEFAULT_PANEL_TIMEOUT_SECONDS = 30.0


class ConnectorResolver(Protocol):
    """Resolves a panel's `connector` name to a live `Connector`.

    A Protocol rather than a concrete registry so a caller can scope
    which connectors a refresh may reach - a scheduled refresh running
    on behalf of a tenant must not be able to resolve a name outside
    that tenant's configured set just because the string matched.
    """

    def __call__(self, name: str) -> Connector: ...


@dataclass
class PanelRefreshResult:
    """What happened to one panel."""

    panel_id: str
    title: str
    succeeded: bool
    rows: list[dict[str, Any]] = field(default_factory=list)
    provenance: QueryProvenance = field(default_factory=QueryProvenance)
    error: str = ""
    skipped_reason: str = ""

    @property
    def skipped(self) -> bool:
        return bool(self.skipped_reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "title": self.title,
            "succeeded": self.succeeded,
            "row_count": len(self.rows),
            "provenance": self.provenance.to_dict(),
            "error": self.error,
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class DashboardRefreshResult:
    """The outcome of refreshing a whole dashboard."""

    dashboard_id: str
    panels: list[PanelRefreshResult] = field(default_factory=list)
    refreshed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def succeeded_count(self) -> int:
        return sum(1 for p in self.panels if p.succeeded)

    @property
    def failed_count(self) -> int:
        return sum(1 for p in self.panels if not p.succeeded and not p.skipped)

    @property
    def skipped_count(self) -> int:
        return sum(1 for p in self.panels if p.skipped)

    @property
    def fully_succeeded(self) -> bool:
        """True only when nothing failed.

        A refresh where some panels failed is deliberately *not*
        reported as success: the caller stamping `last_refreshed_at`
        needs to know the dashboard is partly stale.
        """
        return self.failed_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dashboard_id": self.dashboard_id,
            "refreshed_at": self.refreshed_at.isoformat(),
            "succeeded": self.succeeded_count,
            "failed": self.failed_count,
            "skipped": self.skipped_count,
            "fully_succeeded": self.fully_succeeded,
            "panels": [p.to_dict() for p in self.panels],
        }


def _normalize_rows(result: Any) -> list[dict[str, Any]]:
    """Coerces a connector's `read()` result into rows.

    Connectors legitimately differ in shape - a SQL connector returns a
    list of row mappings, others wrap it. Anything not recognisable as
    rows is reported as zero rows rather than guessed at, because a
    silently mis-shaped result would be charted as if it were data.
    """
    if result is None:
        return []
    if isinstance(result, dict):
        for key in ("rows", "result", "data"):
            inner = result.get(key)
            if isinstance(inner, list):
                return [dict(r) for r in inner if isinstance(r, dict)]
        return []
    if isinstance(result, list):
        return [dict(r) for r in result if isinstance(r, dict)]
    return []


async def refresh_panel(
    panel: PanelSpec,
    resolve_connector: ConnectorResolver,
    *,
    guard: SqlGuard | None = None,
    dialect: str | None = None,
    timeout_seconds: float = DEFAULT_PANEL_TIMEOUT_SECONDS,
) -> PanelRefreshResult:
    """Re-executes one panel's query and returns rows plus real provenance.

    Never raises for a panel-level problem - an unresolvable connector,
    a query the guard refuses, a connector error or a timeout all come
    back as `succeeded=False` with `error` set. See the module docstring
    for why one panel's failure must not take the dashboard with it.
    """
    guard = guard or SqlGuard()
    result = PanelRefreshResult(panel_id=panel.panel_id, title=panel.title, succeeded=False)

    if not panel.query.strip():
        # A panel with no query is not broken - a static/markdown tile
        # legitimately has nothing to run. Skipped, not failed.
        result.skipped_reason = "panel declares no query"
        return result
    if not panel.connector.strip():
        result.error = "panel has a query but names no connector"
        return result

    try:
        connector = resolve_connector(panel.connector)
    except Exception as exc:  # noqa: BLE001 - resolver shape is caller-defined
        result.error = f"connector '{panel.connector}' is not available: {exc}"
        return result

    # Read-only enforcement happens before anything touches a connector,
    # so a stored write can never reach one even momentarily.
    try:
        safe_query = guard.validate_read(panel.query, dialect=dialect)
    except SqlGuardError as exc:
        result.error = f"query refused by SQL guard: {exc}"
        return result

    started = datetime.now(UTC)
    try:
        raw = await asyncio.wait_for(connector.read(safe_query), timeout=timeout_seconds)
    except TimeoutError:
        result.error = f"query exceeded {timeout_seconds:g}s"
        return result
    except Exception as exc:  # noqa: BLE001 - any connector failure is this panel's failure
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    rows = _normalize_rows(raw)
    result.rows = rows
    result.succeeded = True
    # Provenance describes the execution: the approved query (which may
    # differ from `panel.query` by an appended row bound), the real
    # count, the real time.
    result.provenance = QueryProvenance(
        connector=panel.connector,
        query=safe_query,
        metrics=list(panel.metrics),
        dimensions=list(panel.dimensions),
        row_count=len(rows),
        executed_at=started,
    )
    return result


async def refresh_dashboard(
    spec: DashboardSpec,
    resolve_connector: ConnectorResolver,
    *,
    guard: SqlGuard | None = None,
    dialect: str | None = None,
    only_scheduled: bool = False,
    timeout_seconds: float = DEFAULT_PANEL_TIMEOUT_SECONDS,
    write_back: bool = True,
) -> DashboardRefreshResult:
    """Re-executes every panel on `spec`, concurrently.

    Panels are independent queries against (possibly) different
    connectors, so they run concurrently - a twelve-panel dashboard
    refreshing serially would take twelve times longer for no reason.
    `return_exceptions` is not needed because `refresh_panel` already
    converts every panel-level failure into a result.

    `only_scheduled=True` restricts the run to panels declaring
    `RefreshMode.SCHEDULED`, which is what a timer-driven refresh wants:
    an `on_load` panel is refreshed by being loaded, and re-running it
    on a schedule would bill the query twice for one view.

    `write_back=True` records each panel's real provenance onto the spec
    in place, so a caller that persists `spec` afterwards stores what
    actually ran.
    """
    guard = guard or SqlGuard()

    selected: list[PanelSpec] = [
        panel
        for panel in spec.panels
        if not only_scheduled or panel.refresh is RefreshMode.SCHEDULED
    ]

    results = await asyncio.gather(
        *(
            refresh_panel(
                panel,
                resolve_connector,
                guard=guard,
                dialect=dialect,
                timeout_seconds=timeout_seconds,
            )
            for panel in selected
        )
    )

    if write_back:
        by_id = {panel.panel_id: panel for panel in selected}
        for result in results:
            if result.succeeded:
                by_id[result.panel_id].provenance = result.provenance

    outcome = DashboardRefreshResult(dashboard_id=spec.dashboard_id, panels=list(results))
    if outcome.failed_count:
        _logger.warning(
            "dashboard refresh completed with failures",
            dashboard_id=spec.dashboard_id,
            succeeded=outcome.succeeded_count,
            failed=outcome.failed_count,
            skipped=outcome.skipped_count,
        )
    return outcome
