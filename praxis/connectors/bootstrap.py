# praxis/connectors/bootstrap.py
"""Builds a ConnectorRegistry from Settings (spec §6 "degrades gracefully", §19 step 4).

Each of the three settings-backed connectors (GitHub, Slack,
Prometheus) is registered only if its required config is present;
anything left unconfigured is skipped, not failed - a fresh dev
environment with zero credentials configured yields a valid, empty
registry rather than an error (see praxis.cli's "register connectors"
step and praxis.api.main's module-level wiring, both of which call this).

The generic PostgresConnector is deliberately NOT auto-registered here:
per spec §6/§16.2, "connect to any Postgres DB" has no single global
DSN, so registering one is left to whoever actually needs it (a later
phase's task, or a test) - see praxis.connectors.postgres.connector's
docstring.
"""
from __future__ import annotations

from praxis.config import Settings
from praxis.connectors.github.connector import GitHubConnector
from praxis.connectors.prometheus.connector import PrometheusConnector
from praxis.connectors.registry import ConnectorRegistry
from praxis.connectors.slack.connector import SlackConnector


def build_registry(settings: Settings) -> ConnectorRegistry:
    registry = ConnectorRegistry()

    if settings.github_token:
        registry.register(GitHubConnector(token=settings.github_token))

    if settings.slack_bot_token:
        registry.register(SlackConnector(bot_token=settings.slack_bot_token))

    if settings.prometheus_url:
        registry.register(PrometheusConnector(base_url=settings.prometheus_url))

    return registry
