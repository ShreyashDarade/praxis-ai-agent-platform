# praxis/connectors/factory.py
"""Self-registering connector factories (spec §6 universality).

The gap this closes: `bootstrap.py` used to import every connector
class by name and hardcode an `if settings.x: registry.register(...)`
per one. That meant adding connector #5 required *editing* that
function - a core-code change, contradicting spec §6's "adding a new
connector type is registering a new MCP server + a config entry - no
core code change."

The fix is a self-registration pattern, mirroring how `praxis.cli`'s
`INIT_STEPS` and `praxis.api.main`'s `HEALTH_CHECKS` are lists other
modules *append* to rather than edit: each connector module calls
`register_connector_factory(...)` once, at import time, describing how
to tell if it's configured and how to build itself. `bootstrap.py`
never names a connector class - it discovers connector modules by
scanning the package (see `bootstrap.py`) and asks each registered
factory "are you configured?" A brand new connector type is entirely
one new file (a new `praxis/connectors/<name>/connector.py` ending in
a `register_connector_factory(...)` call); nothing else changes.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from praxis.config import Settings
from praxis.core.interfaces import Connector

IsConfigured = Callable[[Settings], bool]
Build = Callable[[Settings], Connector]


@dataclass(frozen=True)
class ConnectorFactory:
    name: str
    is_configured: IsConfigured
    build: Build


_FACTORIES: list[ConnectorFactory] = []


def register_connector_factory(factory: ConnectorFactory) -> None:
    """Called by a connector module at import time to register itself.

    Registering the same factory name twice (e.g. the module got
    imported and somehow re-registered) is a programming error, not a
    runtime condition to degrade gracefully from - fail loudly.
    """
    if any(f.name == factory.name for f in _FACTORIES):
        raise ValueError(f"a connector factory named '{factory.name}' is already registered")
    _FACTORIES.append(factory)


def all_factories() -> list[ConnectorFactory]:
    return list(_FACTORIES)


def required(value: str | None, *, setting: str) -> str:
    """Narrows a `Settings` field a factory's own `is_configured` has
    already established is present.

    Every connector's `build` reads optional settings, and it only ever
    runs after `build_registry` has asked its `is_configured`. That
    invariant is split across two separate lambdas, so nothing checks
    that the pair agree - and when they disagree the `None` travels all
    the way into a connector and surfaces as an unauthenticated 401 or a
    connection to the string "None". This turns that into one message
    naming the setting, at the point of construction.
    """
    if not value:
        raise ValueError(
            f"connector factory reported '{setting}' as configured, but it is unset; "
            "the factory's is_configured and build disagree"
        )
    return value
