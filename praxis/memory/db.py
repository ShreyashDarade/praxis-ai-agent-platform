"""Async engine/session factory - the RelationalStore implementation (spec §2)."""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from praxis.config import Settings
from praxis.core.interfaces import HealthStatus, RelationalStore


class PostgresStore(RelationalStore):
    def __init__(self, settings: Settings) -> None:
        self._engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._session_factory() as session:
            yield session

    async def health(self) -> HealthStatus:
        try:
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
            return HealthStatus(name="database", healthy=True)
        except Exception as exc:  # noqa: BLE001 - a health check must never raise
            return HealthStatus(name="database", healthy=False, detail=str(exc))

    async def dispose(self) -> None:
        await self._engine.dispose()
