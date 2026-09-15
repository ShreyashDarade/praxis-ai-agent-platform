# praxis/connectors/registry.py
"""In-process registry of connected external systems (spec §6).

Deliberately one-directional: this module imports only from
praxis.core.interfaces, never from praxis.api - callers (e.g.
praxis.api.main, praxis.cli) import the registry, not the reverse, so
the connector layer stays usable from any entrypoint (API, CLI, a
future scheduler) without pulling in FastAPI.
"""
from __future__ import annotations

from typing import Awaitable, Callable

from praxis.core.interfaces import Connector, HealthStatus

HealthCheck = Callable[[], Awaitable[HealthStatus]]


class ConnectorRegistry:
    """Keys registered connectors by name; the source of truth for 'what's connected'."""

    def __init__(self) -> None:
        self._connectors: dict[str, Connector] = {}

    def register(self, connector: Connector) -> None:
        if connector.name in self._connectors:
            raise ValueError(
                f"a connector named '{connector.name}' is already registered"
            )
        self._connectors[connector.name] = connector

    def get(self, name: str) -> Connector:
        try:
            return self._connectors[name]
        except KeyError:
            raise KeyError(f"no connector named '{name}' is registered") from None

    def all(self) -> list[Connector]:
        return list(self._connectors.values())

    def all_health_checks(self) -> list[HealthCheck]:
        """One zero-arg async callable per registered connector, calling its own .health().

        Shaped exactly like what praxis.api.main.register_health_check
        expects, so callers can do::

            for check in registry.all_health_checks():
                register_health_check(check)
        """
        return [_health_check_for(connector) for connector in self._connectors.values()]


def _health_check_for(connector: Connector) -> HealthCheck:
    async def check() -> HealthStatus:
        return await connector.health()

    # A meaningful __name__ matters here: praxis.api.main's /health
    # endpoint falls back to `check.__name__` if a check raises
    # unexpectedly (connector.health() itself must never raise, but the
    # fallback should still be legible rather than a generic "check").
    check.__name__ = f"{connector.name}_health_check"
    return check
