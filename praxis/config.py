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

    # Optional connector credentials (spec §6, §19 step 4). Each is
    # genuinely optional: an unset value means that connector is simply
    # not registered (praxis.connectors.bootstrap.build_registry skips
    # it, not fails) - the platform degrades gracefully, per spec §6.
    # No field here for the generic Postgres connector's DSN: connecting
    # to an arbitrary external Postgres DB (spec §16.2) is a per-task/
    # per-connector-registration concern, not a single global setting -
    # a settings field would wrongly imply there's only ever one.
    github_token: str | None = Field(
        default=None, description="GitHub personal access token for GitHubConnector"
    )
    slack_bot_token: str | None = Field(
        default=None, description="Slack bot token (xoxb-...) for SlackConnector"
    )
    prometheus_url: str | None = Field(
        default=None, description="Base URL of a Prometheus server for PrometheusConnector"
    )
