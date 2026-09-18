# praxis/connectors/prometheus/prometheus_connector.py
"""Prometheus HTTP API connector (spec §6, §15, §16.1's fetch-metrics step)."""
from __future__ import annotations

from typing import Any

import httpx

from praxis.connectors.errors import describe_exception
from praxis.connectors.factory import ConnectorFactory, register_connector_factory, required
from praxis.core.interfaces import Connector, ConnectorDescription, HealthStatus


class PrometheusConnector(Connector):
    def __init__(self, base_url: str, name: str = "prometheus", read_only: bool = True) -> None:
        self.name = name
        self.read_only = read_only
        self._base_url = base_url.rstrip("/")

    async def describe(self) -> ConnectorDescription:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self._base_url}/api/v1/label/__name__/values")
            response.raise_for_status()
            data = response.json()
        return ConnectorDescription(
            kind="prometheus", schema={"metric_names": data.get("data", [])}
        )

    async def read(self, query: str, **params: Any) -> Any:
        """Runs ``query`` as a PromQL instant query, returning the parsed JSON result."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{self._base_url}/api/v1/query", params={"query": query, **params}
            )
            response.raise_for_status()
            return response.json()

    async def health(self) -> HealthStatus:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self._base_url}/-/healthy")
            if response.status_code == 200:
                return HealthStatus(name=self.name, healthy=True)
            return HealthStatus(
                name=self.name,
                healthy=False,
                detail=f"unexpected status {response.status_code}",
            )
        except Exception as exc:  # noqa: BLE001 - a health check must never raise
            return HealthStatus(name=self.name, healthy=False, detail=describe_exception(exc))


register_connector_factory(
    ConnectorFactory(
        name="prometheus",
        is_configured=lambda settings: bool(settings.prometheus_url),
        build=lambda settings: PrometheusConnector(
            base_url=required(settings.prometheus_url, setting="prometheus_url")
        ),
    )
)
