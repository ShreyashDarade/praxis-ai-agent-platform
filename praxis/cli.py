"""Operator commands, including `praxis init` (spec §19)."""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig

from praxis.config import Settings
from praxis.connectors.bootstrap import build_registry
from praxis.memory.db import PostgresStore
from praxis.memory.models import DEFAULT_TENANT_ID
from praxis.security.policy import Role
from praxis.security.provisioning import (
    create_tenant,
    create_user,
    ensure_default_tenant,
    issue_api_key,
    revoke_api_key,
)

app = typer.Typer(help="Praxis operator CLI")


@app.callback(invoke_without_command=True)
def main() -> None:
    """Praxis operator CLI."""
    pass


@dataclass
class InitStep:
    name: str
    run: Callable[[Settings], Coroutine[Any, Any, str]]


async def _validate_config(settings: Settings) -> str:
    return "config OK (database_url configured)"


def _alembic_config() -> AlembicConfig:
    # Bare Config (no alembic.ini) so this works regardless of cwd; the
    # connection itself is resolved by migrations/env.py's own
    # Settings().database_url, never hardcoded here.
    repo_root = Path(__file__).resolve().parent.parent
    config = AlembicConfig()
    config.set_main_option("script_location", str(repo_root / "migrations"))
    return config


async def _apply_schema(settings: Settings) -> str:
    # alembic.command.upgrade is synchronous; run it off the event loop
    # thread so this async step doesn't block it.
    await asyncio.to_thread(alembic_command.upgrade, _alembic_config(), "head")
    return "schema applied (alembic upgrade head)"


async def _check_database(settings: Settings) -> str:
    store = PostgresStore(settings)
    status = await store.health()
    await store.dispose()
    if not status.healthy:
        raise RuntimeError(f"database unreachable: {status.detail}")
    return "database reachable"


async def _register_connectors(settings: Settings) -> str:
    # Register example connectors from spec §16 if their config is
    # present; skipped, not failed, for any left unconfigured (spec §19
    # step 4) - an empty registry is a valid, expected state, not an
    # init failure.
    registry = build_registry(settings)
    connectors = registry.all()
    if not connectors:
        return "0 connectors registered (none configured)"
    names = ", ".join(connector.name for connector in connectors)
    return f"{len(connectors)} connector(s) registered: {names}"


# Later phases append to this list (sandbox verification, prompt
# seeding) rather than editing init() itself - OCP.
async def _ensure_default_tenant(settings: Settings) -> str:
    """Phase 12: guarantees the default tenant row exists.

    Runs as an `INIT_STEP` (OCP - appended, not an edit to `init()`)
    because every tenant-scoped column's `server_default` points at it:
    a database where this row is missing would have rows referencing a
    tenant that doesn't exist.
    """
    store = PostgresStore(settings)
    try:
        tenant = await ensure_default_tenant(store)
        return f"default tenant present ({tenant.slug}, id={tenant.id})"
    finally:
        await store.dispose()


# Later phases append to this list (sandbox verification, prompt
# seeding) rather than editing init() itself - OCP.
INIT_STEPS: list[InitStep] = [
    InitStep("validate config", _validate_config),
    InitStep("apply schema", _apply_schema),
    InitStep("check database", _check_database),
    InitStep("ensure default tenant", _ensure_default_tenant),
    InitStep("register connectors", _register_connectors),
]


@app.command()
def init() -> None:
    """Idempotent bootstrap: validate config, apply the schema, verify the database is reachable."""
    try:
        settings = Settings()
    except Exception as exc:  # noqa: BLE001 - construction must fail as cleanly as any step
        typer.echo(f"[FAIL] validate config: {exc}")
        raise typer.Exit(code=1) from exc

    for step in INIT_STEPS:
        try:
            detail = asyncio.run(step.run(settings))
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"[FAIL] {step.name}: {exc}")
            raise typer.Exit(code=1) from exc
        typer.echo(f"[ OK ] {step.name}: {detail}")
    typer.echo("praxis init complete.")


# ---------------------------------------------------------------- #
# Phase 12: tenant/user/key provisioning (Prompt §11).
#
# These live on the CLI rather than being HTTP-only for a concrete
# reason: with `PRAXIS_AUTH_ENABLED=true` there is no way to
# authenticate to the admin API until a first key exists, and that
# first key has to be minted by something that isn't itself behind the
# auth wall. `praxis.api.routes.admin` calls the very same
# `praxis.security.provisioning` functions.
# ---------------------------------------------------------------- #


@app.command("tenant-create")
def tenant_create(slug: str, name: str = "") -> None:
    """Creates a new tenant."""
    settings = Settings()
    store = PostgresStore(settings)

    async def _run() -> str:
        try:
            tenant = await create_tenant(store, slug=slug, name=name or slug)
            return f"{tenant.id}\t{tenant.slug}\t{tenant.name}"
        finally:
            await store.dispose()

    try:
        typer.echo(asyncio.run(_run()))
    except ValueError as exc:
        typer.echo(f"[FAIL] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("user-create")
def user_create(
    email: str,
    tenant_id: str = typer.Option(DEFAULT_TENANT_ID, help="Tenant to create the user in"),
    role: str = typer.Option(Role.ADMIN.value, help="Role name (admin/operator/analyst/viewer)"),
    display_name: str = "",
) -> None:
    """Creates a user in a tenant."""
    settings = Settings()
    store = PostgresStore(settings)

    async def _run() -> str:
        try:
            user = await create_user(
                store,
                tenant_id=tenant_id,
                email=email,
                display_name=display_name,
                roles=[role],
            )
            return f"{user.id}\t{user.email}\t{user.roles}"
        finally:
            await store.dispose()

    try:
        typer.echo(asyncio.run(_run()))
    except ValueError as exc:
        typer.echo(f"[FAIL] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("key-issue")
def key_issue(
    user_id: str,
    name: str = typer.Option("", help="A label for this key"),
    scope: list[str] = typer.Option(None, help="Narrow this key to specific permissions"),
    expires_in_days: int = typer.Option(None, help="Key lifetime in days (default: no expiry)"),
) -> None:
    """Issues an API key. The plaintext is printed once and never stored."""
    settings = Settings()
    store = PostgresStore(settings)

    async def _run() -> str:
        try:
            plaintext, record = await issue_api_key(
                store,
                user_id=user_id,
                name=name,
                scopes=list(scope) if scope else None,
                expires_in_days=expires_in_days,
            )
            return f"{record.id}\t{plaintext}"
        finally:
            await store.dispose()

    try:
        key_line = asyncio.run(_run())
    except ValueError as exc:
        typer.echo(f"[FAIL] {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(key_line)
    typer.echo("store this key now; it cannot be retrieved again")


@app.command("key-revoke")
def key_revoke(key_id: str) -> None:
    """Revokes an API key. Effective on the next request."""
    settings = Settings()
    store = PostgresStore(settings)

    async def _run() -> str:
        try:
            record = await revoke_api_key(store, key_id=key_id)
            return f"{record.id}\trevoked"
        finally:
            await store.dispose()

    try:
        typer.echo(asyncio.run(_run()))
    except ValueError as exc:
        typer.echo(f"[FAIL] {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
