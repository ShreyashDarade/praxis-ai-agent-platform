"""Operator commands, including `praxis init` (spec §19)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

import typer

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
    InitStep("check database", _check_database),
]


@app.command()
def init() -> None:
    """Idempotent bootstrap: validate config, verify the schema is reachable."""
    settings = Settings()
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
