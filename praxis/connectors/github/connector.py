# praxis/connectors/github/connector.py
"""GitHub REST API connector (spec §6, §15, §16).

Talks to https://api.github.com directly via httpx - GitHub's REST API
is simple enough (bearer token + JSON) that a dedicated SDK adds
nothing httpx doesn't already give the rest of this project.
"""
from __future__ import annotations

from typing import Any

import httpx

from praxis.connectors.factory import ConnectorFactory, register_connector_factory
from praxis.core.interfaces import Connector, ConnectorDescription, HealthStatus

_BASE_URL = "https://api.github.com"


class GitHubConnector(Connector):
    def __init__(self, token: str, name: str = "github", read_only: bool = True) -> None:
        self.name = name
        self.read_only = read_only
        self._token = token

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=10.0,
        )

    async def describe(self) -> ConnectorDescription:
        async with self._client() as client:
            response = await client.get("/user")
            response.raise_for_status()
            data = response.json()
        return ConnectorDescription(
            kind="github", schema={"authenticated_as": data.get("login")}
        )

    async def read(self, query: str, **params: Any) -> Any:
        """``query`` is a repo path like ``"owner/repo"``; returns its GitHub repo info."""
        async with self._client() as client:
            response = await client.get(f"/repos/{query}")
            response.raise_for_status()
            return response.json()

    async def health(self) -> HealthStatus:
        try:
            async with self._client() as client:
                response = await client.get("/rate_limit")
            if response.status_code == 200:
                return HealthStatus(name=self.name, healthy=True)
            return HealthStatus(
                name=self.name,
                healthy=False,
                detail=f"unexpected status {response.status_code}",
            )
        except Exception as exc:  # noqa: BLE001 - a health check must never raise
            return HealthStatus(name=self.name, healthy=False, detail=str(exc))


register_connector_factory(
    ConnectorFactory(
        name="github",
        is_configured=lambda settings: bool(settings.github_token),
        build=lambda settings: GitHubConnector(token=settings.github_token),
    )
)
