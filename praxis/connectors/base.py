# praxis/connectors/base.py
"""Re-exports of the Connector contract (spec §6).

The ABC itself lives in praxis.core.interfaces - the core module every
subsystem depends on (spec §4 DIP: connectors depend on core, never the
other way around). This module exists purely so connector code can do
``from praxis.connectors.base import Connector`` without reaching into
``praxis.core`` directly, mirroring the repo layout in spec §15
(``connectors/base.py (Connector interface)``).
"""
from __future__ import annotations

from praxis.core.interfaces import Connector, ConnectorDescription, HealthStatus

__all__ = ["Connector", "ConnectorDescription", "HealthStatus"]
