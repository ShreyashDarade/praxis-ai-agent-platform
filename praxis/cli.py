"""Operator commands, including `praxis init` (spec §19)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import typer
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig

from praxis.config import Settings
from praxis.memory.db import PostgresStore

app = typer.Typer(help="Praxis operator CLI")


@app.callback(invoke_without_command=True)
def main() -> None:
    """Praxis operator CLI."""
    pass


@dataclass
class InitStep:
    name: str
    run: Callable[[Settings], Awaitable[str]]


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


# Later phases append to this list (connector registration, sandbox
# verification, prompt seeding) rather than editing init() itself - OCP.
INIT_STEPS: list[InitStep] = [
    InitStep("validate config", _validate_config),
    InitStep("apply schema", _apply_schema),
    InitStep("check database", _check_database),
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


if __name__ == "__main__":
    app()
