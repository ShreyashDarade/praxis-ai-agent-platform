# praxis/connectors/mcp/mcp_connector.py
"""Universal connector: wraps any real Model Context Protocol server (spec §6).

Phase 2's four connectors are each a hand-written, stack-specific
integration (asyncpg for Postgres, httpx against GitHub's/Prometheus's
own REST APIs, slack_sdk for Slack). Adding connector #6 for a brand
new *kind* of system still meant writing a new Python class with
system-specific API knowledge.

MCPConnector closes that gap: spec §6 says a new connector can be
"register[ed] ... as an MCP server behind one Connector interface". Any
system with a real MCP server - and mature ones already exist for many
systems - is reachable through this one class, with *zero* new Python
code. Registering a new one is a `praxis.config.Settings.mcp_servers`
config entry (see `praxis.connectors.bootstrap`'s data-driven
registration loop), not a new connector module.

Unlike the other four connectors, this one is not part of the
self-registering-factory set (`praxis.connectors.factory`): a factory
is one-per-connector-*type*, gated on "is this type configured at all".
Here one *type* (MCPConnector) can back arbitrarily many named
instances - one per `MCPServerConfig` entry - so bootstrap.py
instantiates them directly in a loop instead.

Connection lifecycle: every method below opens a fresh MCP session
(spawns the stdio subprocess, or opens the HTTP/SSE connection) and
tears it down before returning. No session is held between calls. This
is a deliberate, documented trade-off - the same one
`praxis.api.main`'s `_database_check` and every other connector's
per-call connect/dispose pattern already makes - not an oversight. It
keeps `Connector`/`HealthCheckable` free of any new lifecycle methods
(`connect`/`close`), which would force every other, simpler connector
to grow a lifecycle it doesn't need.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, ContentBlock

from praxis.connectors.errors import describe_exception
from praxis.core.interfaces import Connector, ConnectorDescription, HealthStatus


class MCPConnector(Connector):
    """Talks to one MCP server - a local subprocess over stdio, or a remote
    server over HTTP/SSE - as a plain Praxis ``Connector``.

    Exactly one of ``command`` (stdio) or ``url`` (HTTP/SSE) must be given.
    """

    def __init__(
        self,
        name: str,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        read_only: bool = True,
    ) -> None:
        if command and url:
            raise ValueError(
                "MCPConnector accepts either 'command' (stdio) or 'url' (HTTP/SSE), "
                "not both"
            )
        if not command and not url:
            raise ValueError(
                "MCPConnector requires either 'command' (stdio) or 'url' (HTTP/SSE)"
            )
        self.name = name
        self.read_only = read_only
        self._command = command
        self._args = list(args) if args else []
        self._url = url

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[ClientSession]:
        """Opens one MCP session for the duration of the `with` block, then tears it down.

        Handles both transports this connector supports: a local server
        launched over stdio (``self._command``), or a remote server
        reached over streamable HTTP (``self._url``).
        """
        if self._command is not None:
            params = StdioServerParameters(command=self._command, args=self._args)
            async with (
                stdio_client(params) as (read_stream, write_stream),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                yield session
        else:
            assert self._url is not None  # guaranteed by __init__'s validation
            async with (
                streamable_http_client(self._url) as (read_stream, write_stream, _),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                yield session

    async def describe(self) -> ConnectorDescription:
        async with self._session() as session:
            result = await session.list_tools()
        return ConnectorDescription(
            kind="mcp", schema={"tools": [tool.name for tool in result.tools]}
        )

    async def read(self, query: str, **params: Any) -> Any:
        """``query`` is an MCP tool name; ``params`` are that tool's arguments."""
        async with self._session() as session:
            result = await session.call_tool(query, params)
        return _unwrap(result)

    async def write(self, action: str, **params: Any) -> Any:
        if self.read_only:
            raise PermissionError(f"connector '{self.name}' is registered read-only")
        # For a fully generic MCP server there is no structural difference
        # between a "read" tool and a "write" tool - which tool names a
        # task's risk tier permits is a Praxis-side policy decision, not
        # something this connector can distinguish. Once past the
        # read_only gate, invoking a tool is mechanically identical.
        return await self.read(action, **params)

    async def health(self) -> HealthStatus:
        try:
            async with self._session() as session:
                await session.list_tools()
            return HealthStatus(name=self.name, healthy=True)
        except Exception as exc:  # noqa: BLE001 - a health check must never raise
            return HealthStatus(name=self.name, healthy=False, detail=describe_exception(exc))


def _unwrap(result: CallToolResult) -> Any:
    """Turns an SDK ``CallToolResult`` into a plain dict/list/str, and turns a
    tool-level failure into a raised exception (unlike ``health()``, ``read()``/
    ``write()`` must let a failing tool call propagate as a clear failure).

    Unwraps ``content`` rather than preferring the SDK's optional
    ``structuredContent`` field: ``content`` is the one part of a
    ``CallToolResult`` every MCP server always populates, while whether
    ``structuredContent`` is set - and how a bare scalar gets wrapped
    inside it - is itself an SDK/framework convention (e.g. FastMCP only
    sets it for tools whose return type it can turn into an output
    schema, and wraps a bare scalar as ``{"result": ...}``). Relying on
    ``content`` alone keeps this connector's unwrapping genuinely
    server-agnostic instead of leaning on one framework's behavior.
    """
    if result.isError:
        raise RuntimeError(f"MCP tool call failed: {_unwrap_content(result.content)}")
    return _unwrap_content(result.content)


def _unwrap_content(content: list[ContentBlock]) -> Any:
    values = [_unwrap_block(block) for block in content]
    if len(values) == 1:
        return values[0]
    return values


def _unwrap_block(block: ContentBlock) -> Any:
    if block.type == "text":
        # Many MCP tools return JSON-encoded structured data as text;
        # give callers the parsed form when possible, the raw string
        # otherwise, rather than making every caller re-parse it.
        try:
            return json.loads(block.text)
        except (json.JSONDecodeError, ValueError):
            return block.text
    return block.model_dump()
