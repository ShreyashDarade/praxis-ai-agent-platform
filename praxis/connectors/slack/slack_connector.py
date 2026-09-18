# praxis/connectors/slack/slack_connector.py
"""Slack connector (spec §6, §16.1), built on the official slack_sdk AsyncWebClient.

Unlike the other three connectors, Slack defaults to ``read_only=False``
(spec's "post the postmortem through the Slack connector" walkthrough,
§16.1 step 9) - posting a message is Slack's primary write action and
the whole reason to register this connector for many tasks. The
read_only flag is still fully honored: an operator can register a
read-only Slack connector, and ``write()`` refuses exactly like any
other connector in that configuration (spec §6 "safety net independent
of LLM-authored code").
"""
from __future__ import annotations

from typing import Any

from slack_sdk.web.async_client import AsyncWebClient

from praxis.connectors.factory import ConnectorFactory, register_connector_factory
from praxis.core.interfaces import Connector, ConnectorDescription, HealthStatus


class SlackConnector(Connector):
    def __init__(self, bot_token: str, name: str = "slack", read_only: bool = False) -> None:
        self.name = name
        self.read_only = read_only
        self._client = AsyncWebClient(token=bot_token)

    async def describe(self) -> ConnectorDescription:
        response = await self._client.auth_test()
        return ConnectorDescription(
            kind="slack",
            schema={
                "team": response.get("team"),
                "team_id": response.get("team_id"),
                "user": response.get("user"),
                "user_id": response.get("user_id"),
            },
        )

    async def read(self, query: str, **params: Any) -> Any:
        """``query`` is a channel ID/name; returns its recent messages."""
        response = await self._client.conversations_history(channel=query, **params)
        return response.get("messages", [])

    async def write(self, action: str, **params: Any) -> Any:
        # The read_only guard applies uniformly to every action, exactly
        # like the base Connector.write() - overriding this method only
        # to add a real implementation for "post_message" must not weaken
        # that guard for any other configuration.
        if self.read_only:
            raise PermissionError(f"connector '{self.name}' is registered read-only")
        if action == "post_message":
            response = await self._client.chat_postMessage(**params)
            return response.data
        raise NotImplementedError(f"unsupported action '{action}' for connector '{self.name}'")

    async def health(self) -> HealthStatus:
        try:
            response = await self._client.auth_test()
            if response.get("ok", False):
                return HealthStatus(name=self.name, healthy=True)
            return HealthStatus(name=self.name, healthy=False, detail=str(response.data))
        except Exception as exc:  # noqa: BLE001 - a health check must never raise
            return HealthStatus(name=self.name, healthy=False, detail=str(exc))


register_connector_factory(
    ConnectorFactory(
        name="slack",
        is_configured=lambda settings: bool(settings.slack_bot_token),
        build=lambda settings: SlackConnector(bot_token=settings.slack_bot_token),
    )
)
