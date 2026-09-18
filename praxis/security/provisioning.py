# praxis/security/provisioning.py
"""Creating tenants, users, and API keys.

Kept out of `praxis.api` deliberately: provisioning must be reachable
from the CLI (`praxis tenant-create`, `praxis key-issue`) on a
deployment where the HTTP surface isn't up yet, or where the first
admin key doesn't exist yet and therefore *couldn't* authenticate to an
HTTP endpoint that requires one. The API's admin routes call straight
into these same functions.

`ensure_default_tenant` is idempotent and safe to call on every
startup: it is what guarantees `DEFAULT_TENANT_ID` always resolves to a
real row, including on a database created by `Base.metadata.create_all`
(every test) rather than by the Alembic migration.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from praxis.memory.db import PostgresStore
from praxis.memory.models import (
    DEFAULT_TENANT_ID,
    DEFAULT_TENANT_SLUG,
    ApiKey,
    Tenant,
    User,
)
from praxis.security.api_key import generate_api_key
from praxis.security.policy import Role


async def ensure_default_tenant(store: PostgresStore) -> Tenant:
    """Creates the well-known default tenant if absent; returns it either way."""
    async with store.session() as session:
        tenant = await session.get(Tenant, DEFAULT_TENANT_ID)
        if tenant is not None:
            return tenant
        tenant = Tenant(
            id=DEFAULT_TENANT_ID,
            slug=DEFAULT_TENANT_SLUG,
            name="Default Tenant",
            active=True,
        )
        session.add(tenant)
        await session.commit()
        return tenant


async def create_tenant(store: PostgresStore, *, slug: str, name: str) -> Tenant:
    """Creates a new tenant. Raises `ValueError` on a duplicate slug
    rather than silently returning the existing one - provisioning a
    tenant that already exists is a real operator mistake worth
    surfacing."""
    async with store.session() as session:
        existing = (
            await session.execute(select(Tenant).where(Tenant.slug == slug))
        ).scalar_one_or_none()
        if existing is not None:
            raise ValueError(f"a tenant with slug '{slug}' already exists (id: {existing.id})")
        tenant = Tenant(slug=slug, name=name, active=True)
        session.add(tenant)
        await session.commit()
        return tenant


async def create_user(
    store: PostgresStore,
    *,
    tenant_id: str,
    email: str,
    display_name: str = "",
    roles: list[str] | None = None,
) -> User:
    """Creates a user inside `tenant_id`.

    Validates every role against `Role` up front - a typo'd role would
    otherwise silently grant nothing (see
    `PolicyEngine.effective_permissions`, which ignores unknown roles by
    design), producing a user who mysteriously can't do anything.
    """
    resolved_roles = roles if roles is not None else [Role.VIEWER.value]
    known = {role.value for role in Role}
    unknown = [role for role in resolved_roles if role not in known]
    if unknown:
        raise ValueError(f"unknown role(s) {unknown}; known roles: {sorted(known)}")

    async with store.session() as session:
        tenant = await session.get(Tenant, tenant_id)
        if tenant is None:
            raise ValueError(f"no tenant with id '{tenant_id}'")
        user = User(
            tenant_id=tenant_id,
            email=email,
            display_name=display_name or email,
            roles=resolved_roles,
            active=True,
        )
        session.add(user)
        await session.commit()
        return user


async def issue_api_key(
    store: PostgresStore,
    *,
    user_id: str,
    name: str = "",
    scopes: list[str] | None = None,
    expires_in_days: int | None = None,
) -> tuple[str, ApiKey]:
    """Issues a key for `user_id`; returns `(plaintext, record)`.

    The plaintext is returned to the caller once and never stored - see
    `praxis.security.api_key`.
    """
    plaintext, key_hash = generate_api_key()
    async with store.session() as session:
        user = await session.get(User, user_id)
        if user is None:
            raise ValueError(f"no user with id '{user_id}'")
        record = ApiKey(
            tenant_id=user.tenant_id,
            user_id=user.id,
            name=name or "api key",
            key_hash=key_hash,
            scopes=scopes,
            expires_at=(
                datetime.now(UTC) + timedelta(days=expires_in_days)
                if expires_in_days is not None
                else None
            ),
        )
        session.add(record)
        await session.commit()
        return plaintext, record


async def revoke_api_key(store: PostgresStore, *, key_id: str) -> ApiKey:
    """Marks a key revoked. Revocation takes effect on the next request -
    there is no key cache to invalidate, deliberately: authentication
    reads the row every time so a revoked credential can never be
    honoured from a stale cache (Prompt's "revoked credentials"
    acceptance test)."""
    async with store.session() as session:
        record = await session.get(ApiKey, key_id)
        if record is None:
            raise ValueError(f"no api key with id '{key_id}'")
        record.revoked_at = datetime.now(UTC)
        await session.commit()
        return record
