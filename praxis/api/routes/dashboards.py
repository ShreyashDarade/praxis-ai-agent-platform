# praxis/api/routes/dashboards.py
"""Dashboard CRUD (Prompt §2's "save the dashboard configuration").

A dashboard is stored as one validated `DashboardSpec` document. Every
write re-validates against the renderer's *real* capabilities before
persisting, so an unrenderable dashboard can never reach storage -
the failure surfaces to whoever is creating it, not to whoever later
opens it.

Tenant-scoped throughout, with the same 404-not-403 rule the task
routes use: reporting 403 for another tenant's dashboard would
confirm it exists.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from praxis.analytics.dashboard import (
    DashboardSpec,
    DashboardValidationError,
    assert_valid_dashboard_spec,
)
from praxis.analytics.refresh import refresh_dashboard as run_refresh
from praxis.analytics.visualize import PlotlyVisualizer
from praxis.api.dependencies import authorize_resource_or_404, get_settings, require
from praxis.config import Settings
from praxis.connectors.bootstrap import build_registry
from praxis.memory.db import PostgresStore
from praxis.memory.models import DashboardRecord
from praxis.security.policy import Permission
from praxis.security.principal import Principal

router = APIRouter(prefix="/dashboards", tags=["dashboards"])


class SaveDashboardRequest(BaseModel):
    """A whole dashboard spec, as the declarative document it is."""

    spec: dict[str, Any] = Field(description="A DashboardSpec document")
    refresh_interval_seconds: int | None = None


def _validate(spec: DashboardSpec) -> None:
    """Rejects a spec the renderer could not actually draw."""
    try:
        assert_valid_dashboard_spec(
            spec,
            supported_chart_types=set(PlotlyVisualizer.supported_chart_types()),
            required_roles_for=PlotlyVisualizer.required_roles(),
        )
    except DashboardValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": str(exc), "problems": exc.problems},
        ) from exc


@router.post("", status_code=201)
async def create_dashboard(
    body: SaveDashboardRequest,
    principal: Annotated[Principal, Depends(require(Permission.DASHBOARD_WRITE))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Validates and saves a new dashboard."""
    try:
        spec = DashboardSpec.from_dict({**body.spec, "tenant_id": principal.tenant_id})
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"malformed dashboard spec: {exc}") from exc

    _validate(spec)

    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            record = DashboardRecord(
                tenant_id=principal.tenant_id,
                title=spec.title,
                description=spec.description,
                owner_user_id=principal.user_id,
                spec_version=spec.spec_version,
                spec=spec.to_dict(),
                refresh_interval_seconds=body.refresh_interval_seconds,
            )
            session.add(record)
            await session.commit()
            return {"id": record.id, "title": record.title, "panels": len(spec.panels)}
    finally:
        await store.dispose()


@router.get("")
async def list_dashboards(
    principal: Annotated[Principal, Depends(require(Permission.DASHBOARD_READ))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """The caller's own tenant's dashboards - filtered in the query."""
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            rows = (
                await session.execute(
                    select(DashboardRecord)
                    .where(DashboardRecord.tenant_id == principal.tenant_id)
                    .order_by(DashboardRecord.updated_at.desc())
                )
            ).scalars().all()
    finally:
        await store.dispose()

    return {
        "dashboards": [
            {
                "id": row.id,
                "title": row.title,
                "description": row.description,
                "panels": len(row.spec.get("panels", [])),
                "last_refreshed_at": (
                    row.last_refreshed_at.isoformat() if row.last_refreshed_at else None
                ),
            }
            for row in rows
        ]
    }


@router.get("/{dashboard_id}")
async def get_dashboard(
    dashboard_id: str,
    principal: Annotated[Principal, Depends(require(Permission.DASHBOARD_READ))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            record = await session.get(DashboardRecord, dashboard_id)
    finally:
        await store.dispose()

    record = await authorize_resource_or_404(
        principal,
        Permission.DASHBOARD_READ,
        resource_type="dashboard",
        resource_id=dashboard_id,
        record=record,
    )
    return {"id": record.id, "spec": record.spec}


@router.put("/{dashboard_id}")
async def update_dashboard(
    dashboard_id: str,
    body: SaveDashboardRequest,
    principal: Annotated[Principal, Depends(require(Permission.DASHBOARD_WRITE))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Replaces a dashboard's spec, re-validating before it is stored."""
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            record = await session.get(DashboardRecord, dashboard_id)
            record = await authorize_resource_or_404(
                principal,
                Permission.DASHBOARD_WRITE,
                resource_type="dashboard",
                resource_id=dashboard_id,
                record=record,
            )

            try:
                spec = DashboardSpec.from_dict(
                    {**body.spec, "tenant_id": principal.tenant_id}
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400, detail=f"malformed dashboard spec: {exc}"
                ) from exc
            _validate(spec)

            record.title = spec.title
            record.description = spec.description
            record.spec = spec.to_dict()
            record.spec_version = spec.spec_version
            if body.refresh_interval_seconds is not None:
                record.refresh_interval_seconds = body.refresh_interval_seconds
            await session.commit()
            return {"id": record.id, "title": record.title}
    finally:
        await store.dispose()


@router.post("/{dashboard_id}/refresh")
async def refresh_dashboard(
    dashboard_id: str,
    principal: Annotated[Principal, Depends(require(Permission.DASHBOARD_READ))],
    settings: Annotated[Settings, Depends(get_settings)],
    execute: bool = True,
) -> dict[str, Any]:
    """Re-executes the dashboard's panels and records what actually ran.

    `execute=True` (the default) runs every panel's query through the
    SQL guard against its connector, writes the real provenance back
    onto the stored spec, and only then stamps `last_refreshed_at`.
    Panels fail independently - see `praxis.analytics.refresh`.

    `execute=False` stamps the timestamp without running anything. That
    is for a caller that has *already* refreshed the data by another
    route and only needs the marker; it is not the default, because a
    timestamp that moves while the numbers do not is worse than an
    honestly stale one.

    `last_refreshed_at` is advanced even on a partial failure, with
    `fully_succeeded: false` in the response - the read did happen, and
    pretending otherwise would make a dashboard with one permanently
    broken panel look permanently un-refreshed.
    """
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            record = await session.get(DashboardRecord, dashboard_id)
            record = await authorize_resource_or_404(
                principal,
                Permission.DASHBOARD_READ,
                resource_type="dashboard",
                resource_id=dashboard_id,
                record=record,
            )

            outcome: dict[str, Any] = {}
            if execute:
                spec = DashboardSpec.from_dict(record.spec)
                registry = build_registry(settings)
                result = await run_refresh(spec, registry.get)
                # The spec now carries each panel's real provenance.
                record.spec = spec.to_dict()
                outcome = result.to_dict()

            record.last_refreshed_at = datetime.now(UTC)
            await session.commit()
            return {
                "id": record.id,
                "last_refreshed_at": record.last_refreshed_at.isoformat(),
                "executed": execute,
                **outcome,
            }
    finally:
        await store.dispose()


@router.delete("/{dashboard_id}")
async def delete_dashboard(
    dashboard_id: str,
    principal: Annotated[Principal, Depends(require(Permission.DASHBOARD_WRITE))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            record = await session.get(DashboardRecord, dashboard_id)
            record = await authorize_resource_or_404(
                principal,
                Permission.DASHBOARD_WRITE,
                resource_type="dashboard",
                resource_id=dashboard_id,
                record=record,
            )
            await session.delete(record)
            await session.commit()
    finally:
        await store.dispose()
    return {"id": dashboard_id, "deleted": True}
