# praxis/api/routes/admin.py
"""Tenant/user/API-key administration and audit retrieval (Prompt §11).

Every route here requires `Permission.TENANT_ADMIN` (audit read requires
`AUDIT_READ`), and every one is tenant-scoped: an admin administers
*their own* tenant. Creating a whole new tenant is the single exception
and is deliberately restricted to the `system` principal - i.e. to the
CLI (`praxis tenant-create`) or an auth-disabled deployment - because a
tenant admin being able to mint sibling tenants would defeat the
isolation boundary rather than operate inside it.

The plaintext API key is returned exactly once, in the creation
response, and never stored or logged (`praxis.security.api_key`).
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from praxis.api.dependencies import get_settings, require
from praxis.config import Settings
from praxis.memory.db import PostgresStore
from praxis.memory.models import ApiKey, AuditLog, Tenant, User
from praxis.security import provisioning
from praxis.security.policy import Permission, Role
from praxis.security.principal import Principal

router = APIRouter(prefix="/admin", tags=["admin"])


class CreateTenantRequest(BaseModel):
    slug: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=256)


class CreateUserRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    display_name: str = ""
    roles: list[str] = Field(default_factory=lambda: [Role.VIEWER.value])


class IssueKeyRequest(BaseModel):
    user_id: str
    name: str = ""
    # `None` means "inherit the user's full role permissions"; a list
    # *narrows* them (never widens - see `PolicyEngine`).
    scopes: list[str] | None = None
    expires_in_days: int | None = None


@router.post("/tenants", status_code=201)
async def create_tenant(
    body: CreateTenantRequest,
    principal: Annotated[Principal, Depends(require(Permission.TENANT_ADMIN))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Creates a new tenant. System principal only - see module docstring."""
    if not principal.is_system:
        raise HTTPException(
            status_code=403,
            detail=(
                "creating a tenant is restricted to the system principal; use the "
                "`praxis tenant-create` CLI"
            ),
        )
    store = PostgresStore(settings)
    try:
        tenant = await provisioning.create_tenant(store, slug=body.slug, name=body.name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        await store.dispose()
    return {"id": tenant.id, "slug": tenant.slug, "name": tenant.name}


@router.get("/tenants/current")
async def current_tenant(
    principal: Annotated[Principal, Depends(require(Permission.TENANT_ADMIN))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """The caller's own tenant - never another's."""
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            tenant = await session.get(Tenant, principal.tenant_id)
    finally:
        await store.dispose()
    if tenant is None:
        raise HTTPException(status_code=404, detail="tenant not found")
    return {"id": tenant.id, "slug": tenant.slug, "name": tenant.name, "active": tenant.active}


@router.post("/users", status_code=201)
async def create_user(
    body: CreateUserRequest,
    principal: Annotated[Principal, Depends(require(Permission.TENANT_ADMIN))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Creates a user inside the caller's own tenant."""
    store = PostgresStore(settings)
    try:
        user = await provisioning.create_user(
            store,
            tenant_id=principal.tenant_id,
            email=body.email,
            display_name=body.display_name,
            roles=body.roles,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await store.dispose()
    return {
        "id": user.id,
        "tenant_id": user.tenant_id,
        "email": user.email,
        "roles": user.roles,
    }


@router.get("/users")
async def list_users(
    principal: Annotated[Principal, Depends(require(Permission.TENANT_ADMIN))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            users = (
                await session.execute(select(User).where(User.tenant_id == principal.tenant_id))
            ).scalars().all()
    finally:
        await store.dispose()
    return {
        "users": [
            {"id": u.id, "email": u.email, "roles": u.roles, "active": u.active} for u in users
        ]
    }


@router.get("/api-keys")
async def list_api_keys(
    principal: Annotated[Principal, Depends(require(Permission.TENANT_ADMIN))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Every API key in the caller's own tenant.

    Keys could be issued and revoked but never listed, so an admin had
    no way to see what existed - which makes revocation a guess. The
    hash is never returned: a key's plaintext exists once, in the issue
    response, and nothing here can reconstruct it. What is returned is
    what an operator actually needs to decide whether to revoke
    something - who holds it, when it was last used, and whether it is
    still live.
    """
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            rows = (
                (
                    await session.execute(
                        select(ApiKey)
                        .where(ApiKey.tenant_id == principal.tenant_id)
                        .order_by(ApiKey.created_at.desc())
                    )
                )
                .scalars()
                .all()
            )
    finally:
        await store.dispose()

    return {
        "api_keys": [
            {
                "id": key.id,
                "user_id": key.user_id,
                "name": key.name,
                "scopes": list(key.scopes or []),
                "revoked": key.revoked_at is not None,
                "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None,
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "created_at": key.created_at.isoformat(),
            }
            for key in rows
        ]
    }


@router.post("/api-keys", status_code=201)
async def issue_api_key(
    body: IssueKeyRequest,
    principal: Annotated[Principal, Depends(require(Permission.TENANT_ADMIN))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Issues an API key for a user in the caller's own tenant.

    `api_key` in the response is the only time the plaintext exists
    outside the caller's hands - it is not recoverable afterwards.
    """
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            user = await session.get(User, body.user_id)
        if user is None or user.tenant_id != principal.tenant_id:
            # Same 404-not-403 reasoning as `authorize_resource`: a 403
            # would confirm the user exists in another tenant.
            raise HTTPException(status_code=404, detail=f"no user with id '{body.user_id}'")

        plaintext, record = await provisioning.issue_api_key(
            store,
            user_id=body.user_id,
            name=body.name,
            scopes=body.scopes,
            expires_in_days=body.expires_in_days,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await store.dispose()

    return {
        "id": record.id,
        "api_key": plaintext,
        "user_id": record.user_id,
        "scopes": record.scopes,
        "expires_at": record.expires_at.isoformat() if record.expires_at else None,
        "warning": "store this key now; it cannot be retrieved again",
    }


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    principal: Annotated[Principal, Depends(require(Permission.TENANT_ADMIN))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Revokes a key. Effective on the very next request - authentication
    reads the row every time, with no cache to go stale."""
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            record = await session.get(ApiKey, key_id)
        if record is None or record.tenant_id != principal.tenant_id:
            raise HTTPException(status_code=404, detail=f"no api key with id '{key_id}'")
        await provisioning.revoke_api_key(store, key_id=key_id)
    finally:
        await store.dispose()
    return {"id": key_id, "revoked": True}


@router.get("/audit")
async def read_audit_log(
    principal: Annotated[Principal, Depends(require(Permission.AUDIT_READ))],
    settings: Annotated[Settings, Depends(get_settings)],
    limit: int = 100,
    action: str | None = None,
    resource_id: str | None = None,
) -> dict[str, Any]:
    """The caller's own tenant's audit trail, newest first.

    Tenant-filtered in the query itself - there is no code path that
    returns another tenant's audit rows.
    """
    limit = max(1, min(limit, 1000))
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            stmt = (
                select(AuditLog)
                .where(AuditLog.tenant_id == principal.tenant_id)
                .order_by(AuditLog.created_at.desc())
                .limit(limit)
            )
            if action is not None:
                stmt = stmt.where(AuditLog.action == action)
            if resource_id is not None:
                stmt = stmt.where(AuditLog.resource_id == resource_id)
            rows = (await session.execute(stmt)).scalars().all()
    finally:
        await store.dispose()

    return {
        "entries": [
            {
                "id": row.id,
                "action": row.action,
                "resource_type": row.resource_type,
                "resource_id": row.resource_id,
                "decision": row.decision,
                "reason": row.reason,
                "actor_user_id": row.actor_user_id,
                "detail": row.detail,
                "correlation_id": row.correlation_id,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]
    }
