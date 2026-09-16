"""Deployment-wide settings (spec §2; §19 step 2 - config validation)."""
from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPServerConfig(BaseModel):
    """One entry in ``Settings.mcp_servers``: everything needed to build one
    ``MCPConnector`` (spec §6's "register it as an MCP server ... no core
    code change").

    A pydantic ``BaseModel`` (not a dataclass) so pydantic-settings can
    parse a whole list of these straight out of a JSON-encoded env var.
    Exactly one of ``command`` (stdio) or ``url`` (HTTP/SSE) is expected
    to be set per entry - the same validation
    ``praxis.connectors.mcp.connector.MCPConnector.__init__`` enforces
    when `praxis.connectors.bootstrap.build_registry` constructs one
    from this config.
    """

    name: str
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None


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

    # Universal MCP connectors (spec §6): each entry becomes one
    # MCPConnector, registered by praxis.connectors.bootstrap.build_registry
    # directly (not via the self-registering-factory mechanism the four
    # Phase 2 connectors use, since one entry here is one *instance*, not
    # one *type*). Setting, e.g.,
    #   PRAXIS_MCP_SERVERS='[{"name": "example", "command": "npx", "args": ["-y", "some-mcp-server"]}]'
    # registers that MCP server as a connector with zero code changes -
    # the concrete proof that adding connector #N is a config entry, not
    # a new Python class. Defaults to empty: no MCP servers configured.
    # This is deliberately unlike the missing SQLConnector setting above:
    # "any SQL DB" has no single global DSN, but "which MCP servers this
    # deployment keeps connected" genuinely is a global, enumerable list
    # - each entry names one specific, already-known server - so it
    # belongs on Settings the way github_token/slack_bot_token/
    # prometheus_url do, not left to per-task construction.
    mcp_servers: list[MCPServerConfig] = Field(
        default_factory=list,
        description="MCP servers to register as connectors; see MCPServerConfig",
    )
