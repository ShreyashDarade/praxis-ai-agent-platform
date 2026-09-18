# praxis/security/authentication.py
"""Turning a raw credential into a verified `Principal`.

Kept in `praxis.security` rather than `praxis.api` so the CLI, the
scheduler, and any future entrypoint authenticate through the exact
same code path as HTTP - there is no second, looser way to become a
principal.

Every lookup reads the `api_keys` row fresh (no cache): revocation and
expiry must take effect on the very next request, which is precisely
what a credential-revocation control is for.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from praxis.memory.db import PostgresStore
from praxis.memory.models import ApiKey, Tenant, User
from praxis.security.api_key import hash_api_key, looks_like_api_key
from praxis.security.principal import Principal


class AuthenticationError(Exception):
    """A credential was absent, malformed, unknown, expired, or revoked.

    `reason` is a short machine-readable code (`missing`, `malformed`,
    `unknown`, `expired`, `revoked`, `inactive`) so the API can log the
    real cause while returning a deliberately uniform 401 body - never
    telling an unauthenticated caller which of those it was.
    """

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


async def authenticate_api_key(store: PostgresStore, plaintext: str | None) -> Principal:
    """Verifies `plaintext` and returns the `Principal` it identifies.

    Also stamps `last_used_at`, which is what makes an unused/leaked key
    visible to an operator reviewing credentials.
    """
    if not plaintext:
        raise AuthenticationError("no API key supplied", reason="missing")
    if not looks_like_api_key(plaintext):
        raise AuthenticationError("malformed API key", reason="malformed")

    key_hash = hash_api_key(plaintext)
    now = datetime.now(timezone.utc)

    async with store.session() as session:
        record = (
            await session.execute(select(ApiKey).where(ApiKey.key_hash == key_hash))
        ).scalar_one_or_none()
        if record is None:
            raise AuthenticationError("unknown API key", reason="unknown")
        if record.revoked_at is not None:
            raise AuthenticationError("API key has been revoked", reason="revoked")
        if record.expires_at is not None:
            expires_at = record.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at <= now:
                raise AuthenticationError("API key has expired", reason="expired")

        user = await session.get(User, record.user_id)
        if user is None or not user.active:
            raise AuthenticationError("API key's user is inactive or missing", reason="inactive")

        tenant = await session.get(Tenant, user.tenant_id)
        if tenant is None or not tenant.active:
            raise AuthenticationError("tenant is inactive or missing", reason="inactive")

        record.last_used_at = now
        await session.commit()

        return Principal(
            tenant_id=user.tenant_id,
            user_id=user.id,
            email=user.email,
            roles=tuple(user.roles or ()),
            scopes=tuple(record.scopes) if record.scopes is not None else None,
            api_key_id=record.id,
            is_system=False,
        )


def extract_credential(
    api_key_header: str | None, authorization_header: str | None
) -> str | None:
    """Pulls the raw key out of either supported header.

    `X-API-Key: <key>` is the primary form; `Authorization: Bearer <key>`
    is accepted because it is what most HTTP clients and API gateways
    send by default.
    """
    if api_key_header:
        return api_key_header.strip()
    if authorization_header:
        parts = authorization_header.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    return None
