# praxis/connectors/web/search_provider.py
"""Pluggable web search backends (spec §2's "Web search: MCP search
connector (Tavily)" row - "Also drops in via the same interface: Brave,
SerpAPI, Bing, any MCP search server"). `WebConnector.search` depends
only on the `SearchProvider` ABC below, never on `TavilySearchProvider`
directly - swapping providers is writing one new small adapter class,
never touching `WebConnector`.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

import httpx


@dataclass
class SearchResultItem:
    title: str
    url: str
    content: str


class SearchProvider(abc.ABC):
    @abc.abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[SearchResultItem]: ...


_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_DEFAULT_TIMEOUT_SECONDS = 15.0


class TavilySearchProvider(SearchProvider):
    """Tavily's search API.

    Request/response shape verified against Tavily's own published API
    reference (`POST /search`) at development time, not guessed from
    memory: the request body is JSON `{"api_key": ..., "query": ...,
    "max_results": ...}`; the response is JSON `{"results": [{"title":
    ..., "url": ..., "content": ...}, ...], ...}` (Tavily's response
    also includes other top-level fields such as `answer`/
    `response_time`, deliberately ignored here - only `results` is
    consumed). **No live Tavily API key is configured in this
    environment** - see `WebConnector.search`'s docstring for the
    "not configured" posture this leads to. `tests/connectors/web/
    test_search_provider.py` therefore proves the request-shaping and
    response-parsing logic against a `respx`-mocked, realistic Tavily
    response, exactly like this codebase's existing pattern for every
    other credential-gated connector (e.g.
    `tests/connectors/github/test_github_connector.py`) - a real live-
    API end-to-end call needs a key this environment doesn't have, and
    is explicitly not attempted here rather than silently skipped
    without saying so.
    """

    def __init__(self, api_key: str, *, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS) -> None:
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    async def search(self, query: str, max_results: int = 5) -> list[SearchResultItem]:
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(
                _TAVILY_SEARCH_URL,
                json={"api_key": self._api_key, "query": query, "max_results": max_results},
            )
            response.raise_for_status()
            data = response.json()

        return [
            SearchResultItem(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
            )
            for item in data.get("results", [])
        ]
