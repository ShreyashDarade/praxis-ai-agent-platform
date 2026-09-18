# praxis/api/dependencies.py
"""FastAPI dependencies: identity, authorization, and the shared store.

This is the *only* place an inbound HTTP request becomes a
`Principal`. Routes never read headers themselves and never construct a
`Principal` - they declare `principal: Principal = Depends(...)` and a
required `Permission`, which keeps the authorization decision
declarative and impossible to forget silently (a route with no
permission dependency is visibly different from one with it).

**Auth-disabled mode.** When `Settings.auth_enabled` is false (the
default, and what this repo's whole existing test suite runs under),
every request resolves to `SYSTEM_PRINCIPAL` inside the real default
tenant. The `PolicyEngine` check and the audit write still happen - only
the credential step is skipped - so the authorization path is exercised
identically in both modes rather than being a branch that is never run
in development and then fails in production.
"""
from __future__ import annotations

from typing import Annotated, Callable

import structlog
from fastapi import Depends, Header, HTTPException, Request

from praxis.config import Settings
from praxis.memory.db import PostgresStore
from praxis.security.audit import AuditLogger
from praxis.security.authentication import (
    AuthenticationError,
    authenticate_api_key,
    extract_credential,
)
from praxis.security.policy import (
    Permission,
    PermissionDeniedError,
    PolicyEngine,
    TenantIsolationError,
    policy_engine,
)
from praxis.security.principal import SYSTEM_PRINCIPAL, Principal

_logger = structlog.get_logger(__name__)


def get_settings() -> Settings:
    """A fresh `Settings` per request - the same posture every route in
    this app already takes, so `PRAXIS_*` can be repointed per test or
    per deployment reload without re-importing anything."""
    return Settings()


def get_store(settings: Annotated[Settings, Depends(get_settings)]) -> PostgresStore:
    """A per-request `PostgresStore`.

    Deliberately *not* disposed here: FastAPI resolves this once per
    request and the engine's own pool is what manages connections. The
    routes that need a longer-lived store (the orchestrator) use
    `praxis.api.main`'s module-level singleton instead.
    """
    return PostgresStore(settings)


async def current_principal(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """Resolves the authenticated principal for this request.

    Returns 401 with a uniform body on any authentication failure - the
    specific `reason` is logged, never returned, so an unauthenticated
    caller cannot distinguish "unknown key" from "revoked key".
    """
    if not settings.auth_enabled:
        principal = SYSTEM_PRINCIPAL
        request.state.principal = principal
        return principal

    credential = extract_credential(x_api_key, authorization)
    store = PostgresStore(settings)
    try:
        principal = await authenticate_api_key(store, credential)
    except AuthenticationError as exc:
        _logger.warning(
            "authentication_failed",
            reason=exc.reason,
            path=request.url.path,
            method=request.method,
        )
        raise HTTPException(
            status_code=401,
            detail="authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    finally:
        await store.dispose()

    request.state.principal = principal
    return principal


CurrentPrincipal = Annotated[Principal, Depends(current_principal)]


def require(permission: Permission) -> Callable[..., object]:
    """Builds a dependency enforcing `permission`, auditing either way.

    Usage::

        @router.post("/intent", dependencies=[Depends(require(Permission.TASK_CREATE))])

    or, when the handler also needs the principal::

        principal: Principal = Depends(require(Permission.TASK_CREATE))

    The returned dependency yields the `Principal` so both forms work
    from one definition.
    """

    async def _dependency(
        request: Request,
        principal: CurrentPrincipal,
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> Principal:
        decision = policy_engine.check(
            principal,
            permission,
            resource_type=request.url.path,
        )

        store = PostgresStore(settings)
        try:
            await AuditLogger(store).record_decision(
                principal=principal,
                decision=decision,
                action=permission.value,
                detail={"method": request.method, "path": request.url.path},
            )
        finally:
            await store.dispose()

        if not decision.allowed:
            _logger.warning(
                "authorization_denied",
                permission=permission.value,
                principal=principal.describe(),
                path=request.url.path,
            )
            raise HTTPException(status_code=403, detail=decision.reason)
        return principal

    return _dependency


async def authorize_resource(
    principal: Principal,
    permission: Permission,
    *,
    resource_type: str,
    resource_id: str,
    resource_tenant_id: str,
    engine: PolicyEngine = policy_engine,
) -> None:
    """Second-stage ABAC check, once a route has actually loaded the row.

    The `require(...)` dependency can only check RBAC (the resource
    isn't loaded yet at dependency-resolution time); this is what
    enforces "and it belongs to your tenant".

    This decision - not the earlier route-level one - is the
    security-relevant one for a cross-tenant attempt, so it writes its
    own audit row carrying the *real* resource id. Without that, the
    audit trail would only ever show "someone called GET /tasks/{id}"
    with no record of which resource was actually reached for.

    A cross-tenant reference raises 404 rather than 403 on purpose: a
    403 would confirm the resource exists in *some* tenant, which is
    itself a cross-tenant information leak. The audit row records the
    real `TenantIsolationError` reason regardless.
    """
    decision = engine.check(
        principal,
        permission,
        resource_type=resource_type,
        resource_id=resource_id,
        resource_tenant_id=resource_tenant_id,
    )

    store = PostgresStore(Settings())
    try:
        await AuditLogger(store).record_decision(principal=principal, decision=decision)
    finally:
        await store.dispose()

    if decision.allowed:
        return

    if resource_tenant_id != principal.tenant_id:
        _logger.warning(
            "tenant_isolation_denied",
            resource_type=resource_type,
            resource_id=resource_id,
            principal_tenant=principal.tenant_id,
            resource_tenant=resource_tenant_id,
        )
        raise HTTPException(
            status_code=404, detail=f"no {resource_type} with id '{resource_id}'"
        )
    raise HTTPException(status_code=403, detail=decision.reason)
