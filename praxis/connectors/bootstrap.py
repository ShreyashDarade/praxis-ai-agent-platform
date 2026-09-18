# praxis/connectors/bootstrap.py
"""Builds a ConnectorRegistry from Settings (spec §6 "degrades gracefully", §19 step 4).

This module never names a connector class, and never grows an
`if settings.x: registry.register(SomeConnector(...))` line per
connector - that pattern is exactly the "core code change" spec §6
says a new connector must not require. Instead:

1. `_discover_connector_modules()` scans every immediate subdirectory
   of `praxis/connectors/` for a `<name>_connector.py` file and
   imports it. Importing it runs that module's own
   `register_connector_factory(...)` call (see e.g.
   `praxis.connectors.github.github_connector`), which is the
   connector registering *itself* into `praxis.connectors.factory`.
2. `build_registry()` then just asks every registered factory "are you
   configured?" (via `Settings`) and builds the ones that are.

Adding connector #5 is entirely one new file:
`praxis/connectors/<name>/<name>_connector.py`, implementing
`Connector` and ending with a `register_connector_factory(...)` call.
Nothing in this file, or any other core file, changes.

The generic PostgresConnector (and, for the same reason, the
dialect-agnostic SQLConnector) is deliberately not part of this
self-registering set: per spec §6/§16.2, "connect to any [Postgres/SQL]
DB" has no single global DSN to gate a factory on, so registering one
is left to whoever actually needs it (a later phase's task, or a test)
- see `praxis.connectors.postgres.postgres_connector`'s and
`praxis.connectors.sql.sql_connector`'s docstrings.

Universal MCP connectors are a second, separate registration pass,
below - genuinely data-driven (a for-loop over `settings.mcp_servers`),
not the self-registering-factory pattern, because a factory is
one-per-connector-*type* ("is this type configured at all") while an
MCP server entry is one-per-*instance* (a deployment may want several,
each with its own name/command/url). `MCPConnector` itself never calls
`register_connector_factory` - see
`praxis.connectors.mcp.mcp_connector`'s docstring.
"""
from __future__ import annotations

import importlib
import pkgutil

import structlog

import praxis.connectors as _connectors_pkg
from praxis.config import Settings
from praxis.connectors import factory
from praxis.connectors.mcp.mcp_connector import MCPConnector
from praxis.connectors.registry import ConnectorRegistry
from praxis.connectors.resilience import (
    ConnectorResilience,
    RateLimit,
    ResilientConnector,
)
from praxis.core.interfaces import Connector

_logger = structlog.get_logger(__name__)

_DISCOVERED = False


def _discover_connector_modules() -> None:
    """Import every `praxis.connectors.<name>.<name>_connector` module once.

    Idempotent and safe to call repeatedly (e.g. once per test) -
    Python's own import cache means a second call is a no-op, so no
    factory ever gets registered twice.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    for module_info in pkgutil.iter_modules(_connectors_pkg.__path__):
        if not module_info.ispkg:
            continue
        connector_module = f"praxis.connectors.{module_info.name}.{module_info.name}_connector"
        try:
            importlib.import_module(connector_module)
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if missing == connector_module or missing.startswith(
                f"praxis.connectors.{module_info.name}"
            ):
                # Not every subpackage is a connector (e.g. this file's
                # own package has no <name>_connector.py at its own
                # level) - skip silently rather than treating "no
                # connector module here" as an error.
                continue
            # The connector module exists but one of its imports does
            # not. For a connector backed by an optional extra (`s3`
            # needs `aioboto3`) that is the expected state of a base
            # install, so it must not crash discovery - but it is a
            # different fact from "there is no connector here", and
            # collapsing the two would also hide a genuine typo'd import
            # inside a connector. Logged at INFO, and the connector is
            # simply unavailable.
            _logger.info(
                "connector unavailable: optional dependency not installed",
                connector=module_info.name,
                missing_module=missing,
            )
            continue
    _DISCOVERED = True


def _resilience_for(settings: Settings) -> ConnectorResilience | None:
    """The shared rate limiter + breaker set for one registry, or None.

    One per registry rather than one global: two tenants' registries
    must not share a breaker, or one tenant's outage would reject the
    other's calls to a connector that is working fine for them.
    """
    if not settings.connector_circuit_breaker_enabled and (
        settings.connector_rate_limit_per_second is None
    ):
        return None

    rate_limit = (
        RateLimit(max_calls=int(settings.connector_rate_limit_per_second), period_seconds=1.0)
        if settings.connector_rate_limit_per_second
        else None
    )
    return ConnectorResilience(
        default_rate_limit=rate_limit,
        # A threshold this high effectively disables opening while
        # keeping the rate limiter, for a deployment that wants
        # throttling without breaking.
        failure_threshold=(
            settings.connector_failure_threshold
            if settings.connector_circuit_breaker_enabled
            else 2**31
        ),
        reset_timeout_seconds=settings.connector_circuit_reset_seconds,
    )


def build_registry(settings: Settings) -> ConnectorRegistry:
    _discover_connector_modules()
    registry = ConnectorRegistry()
    resilience = _resilience_for(settings)

    def _register(connector: Connector) -> None:
        # Wrapping happens here, once, rather than inside each
        # connector: resilience is a property of *calling* a connector,
        # so none of the thirteen has to know about it.
        registry.register(
            ResilientConnector(connector, resilience) if resilience is not None else connector
        )

    for connector_factory in factory.all_factories():
        if connector_factory.is_configured(settings):
            _register(connector_factory.build(settings))

    # Second, separate pass: one MCPConnector per configured
    # `MCPServerConfig` entry. Genuinely data-driven - a for-loop over a
    # list - so registering server #2, #3, ... never touches this
    # function again, same "no core code change" property the
    # self-registering factories above give the four Phase 2 connectors.
    for server in settings.mcp_servers:
        _register(
            MCPConnector(
                name=server.name,
                command=server.command,
                args=server.args,
                url=server.url,
            )
        )
    return registry
