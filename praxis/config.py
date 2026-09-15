"""Deployment-wide settings (spec §2; §19 step 2 - config validation)."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PRAXIS_", env_file=".env", extra="ignore")

    database_url: str = Field(
        ...,
        description="Async SQLAlchemy URL, e.g. postgresql+asyncpg://user:pass@host/db",
    )
    docker_host: str | None = Field(
        default=None, description="Override for the Docker daemon socket"
    )
    sandbox_timeout_seconds: int = Field(default=30, ge=1, le=300)
