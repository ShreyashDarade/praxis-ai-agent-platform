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

from collections.abc import Callable
from typing import Annotated, TypeVar

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
    UNAVAILABLE_REASON,
    Permission,
    PolicyDecision,
    PolicyEngine,
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
        # The resource *type*, not the request path. `AuditLog.
        # resource_type` is a String(64), and a path longer than that
        # (every `/artifacts/t/<uuid>/...` fetch, for one) made the
        # INSERT raise - which `AuditLogger.record` deliberately
        # swallows so an audit failure cannot break the caller. The two
        # together meant those requests were silently unaudited.
        #
        # Every `Permission` is spelled `<resource>:<verb>`, so the part
        # before the colon is the type and is bounded by construction.
        # The full path is still recorded, in `detail` below, where the
        # column is unbounded.
        resource_type = permission.value.split(":", 1)[0]
        decision = policy_engine.check(
            principal,
            permission,
            resource_type=resource_type,
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
    resource_tenant_id: str | None,
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
    itself a cross-tenant information leak.

    **`resource_tenant_id=None` means the row was not found**, and a
    route must call this in that case too rather than raising 404 on its
    own. The two outcomes are then indistinguishable from outside in
    every channel a tenant can observe:

    - the HTTP response is the same 404;
    - an audit row is written either way, so the *presence* of a row is
      not itself an oracle - previously, probing a real id owned by
      someone else produced a row while probing a nonexistent id
      produced none, and an admin reading their own `/admin/audit` could
      tell the two apart;
    - the audited reason is the same string for both.

    The real distinction survives only in the structlog warning below,
    which goes to operators rather than into a tenant-readable table.
    Recording the not-found case is also a gain in its own right:
    sustained probing for ids that do not exist is what enumeration
    looks like.
    """
    if resource_tenant_id is None:
        decision = PolicyDecision(
            allowed=False,
            permission=permission.value,
            principal=principal.describe(),
            tenant_id=principal.tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            # Deliberately identical to the cross-tenant reason.
            reason=UNAVAILABLE_REASON,
        )
    else:
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


_ResourceT = TypeVar("_ResourceT")


async def authorize_resource_or_404(
    principal: Principal,
    permission: Permission,
    *,
    resource_type: str,
    resource_id: str,
    record: _ResourceT | None,
    engine: PolicyEngine = policy_engine,
) -> _ResourceT:
    """`authorize_resource`, for the usual "load the row then check it" shape.

    Replaces the pattern every route was repeating::

        if record is None:
            raise HTTPException(404, ...)          # <- audited nowhere
        await authorize_resource(..., record.tenant_id)

    The early raise was the bug: a probe for an id that exists in
    another tenant produced an audit row, and a probe for one that
    exists nowhere produced none - so an admin reading their own
    `/admin/audit` could tell the two apart, which is exactly the
    existence oracle the 404 is there to deny. Routing both through
    `authorize_resource` makes the observable trail identical.

    Returns the record, so the caller gets the non-`None` narrowing for
    free rather than needing a second check.
    """
    await authorize_resource(
        principal,
        permission,
        resource_type=resource_type,
        resource_id=resource_id,
        resource_tenant_id=None if record is None else getattr(record, "tenant_id", None),
        engine=engine,
    )
    if record is None:
        # Unreachable: `authorize_resource` raises 404 for a `None`
        # tenant. Kept so the return type is honest rather than relying
        # on a `NoReturn` the checker cannot infer conditionally.
        raise HTTPException(status_code=404, detail=f"no {resource_type} with id '{resource_id}'")
    return record
