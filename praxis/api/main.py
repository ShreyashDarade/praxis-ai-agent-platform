"""FastAPI application entrypoint (spec §14)."""
from __future__ import annotations

from typing import Awaitable, Callable

from fastapi import FastAPI

from praxis.config import Settings
from praxis.core.interfaces import HealthStatus
from praxis.memory.db import PostgresStore

app = FastAPI(title="Praxis")

# Later phases append connector/sandbox checks here - OCP, mirrors cli.py's INIT_STEPS.
HealthCheck = Callable[[], Awaitable[HealthStatus]]
HEALTH_CHECKS: list[HealthCheck] = []


def register_health_check(check: HealthCheck) -> None:
    HEALTH_CHECKS.append(check)


async def _database_check() -> HealthStatus:
    settings = Settings()
    store = PostgresStore(settings)
    status = await store.health()
    await store.dispose()
    return status


register_health_check(_database_check)


@app.get("/health")
async def health() -> dict:
    results = [await check() for check in HEALTH_CHECKS]
    overall = all(r.healthy for r in results)
    return {
        "healthy": overall,
        "components": [
            {"name": r.name, "healthy": r.healthy, "detail": r.detail} for r in results
        ],
    }
