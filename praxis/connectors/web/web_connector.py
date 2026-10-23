# praxis/connectors/web/web_connector.py
"""`WebConnector`: web search/read/crawl behind the same `Connector`
interface as every other external system Praxis talks to (spec §6.1's
closing line: "no special-cased subsystem, just a connector whose
`read()` implementation happens to carry these extra guarantees because
of what it reads from - the open web, not a system you configured").

This is the most safety-dense connector in the codebase - every method
that touches an external URL goes through the full pipeline this
subpackage builds up, module by module:

- `ssrf.py` - DNS-resolve + validate + reject credentials, before any
  connection, on every hop.
- `guarded_fetch.py` - SSRF-safe pinned fetch, manual bounded redirects,
  content-type + byte-cap enforcement, streamed.
- `robots.py` - per-origin robots.txt, cached.
- `extract.py` - fetched bytes -> plain text.
- `untrusted.py` - plain text -> explicitly tagged untrusted content,
  applied to every distilled result this connector ever returns.
- The distillation pass itself (spec §5's "distill before it reaches
  the context window") - a cheap-tier (`"routing"` purpose) LLM call
  through the same `LLMCatalogue`/`PromptManager` every other
  LLM-calling component in this codebase uses (Phase 4/6/7's own
  pattern, e.g. `praxis.ingestion.enrichment.document_enrichment`).

**The one rule this whole phase exists to enforce** (spec §6.1's
"prior-context-only fetch", spec §20's "exfiltration via a synthesized
fetch is closed structurally, not by review"): `read_page`/`crawl`
refuse outright - `PriorContextViolationError`, before any network
activity at all - to fetch a URL that isn't already in the caller-
supplied `known_urls` set. `known_urls` is never supplied by an LLM's
own freshly-generated output; it is populated exclusively by
`praxis.core.orchestrator.Orchestrator` from the task's validated
intent text and prior tool results (regex-extracted, see that module),
and passed into every skill call as a plain kwarg - a synthesized or
prompt-injected tool cannot manufacture membership in this set merely
by generating a URL string, because this check runs here, in the
connector, regardless of what any generated code contains.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from praxis.connectors.factory import ConnectorFactory, register_connector_factory, required
from praxis.connectors.web import guarded_fetch, robots, untrusted
from praxis.connectors.web.errors import (
    FETCH_ERRORS,
    PriorContextViolationError,
    RobotsDisallowedError,
    SearchProviderNotConfiguredError,
)
from praxis.connectors.web.extract import extract_text
from praxis.connectors.web.search_provider import SearchProvider, TavilySearchProvider
from praxis.core.interfaces import Connector, ConnectorDescription, HealthStatus
from praxis.llm.catalogue import LLMCatalogue
from praxis.llm.prompt_manager import PromptManager

_DISTILL_PROMPT_NAME = "distill_page"
_DISTILL_PROMPT_VERSION = "v1"
_DEFAULT_USER_AGENT = "PraxisBot"


class WebConnector(Connector):
    read_only = True

    def __init__(
        self,
        *,
        search_provider: SearchProvider | None = None,
        catalogue: LLMCatalogue | None = None,
        prompt_manager: PromptManager | None = None,
        name: str = "web",
        user_agent: str = _DEFAULT_USER_AGENT,
    ) -> None:
        self.name = name
        self._search_provider = search_provider
        self._catalogue = catalogue if catalogue is not None else LLMCatalogue()
        self._prompt_manager = prompt_manager if prompt_manager is not None else PromptManager()
        self._user_agent = user_agent

    async def describe(self) -> ConnectorDescription:
        return ConnectorDescription(
            kind="web",
            schema={
                "capabilities": ["search", "read", "crawl"],
                "search_configured": self._search_provider is not None,
            },
        )

    async def read(self, query: str, **params: Any) -> Any:
        """Generic `Connector.read()` conformance (spec §6): dispatches
        on `params["mode"]` ("search"|"read"|"crawl"). The three named
        async methods below (`search`/`read_page`/`crawl`) are what
        every real caller in this codebase actually uses directly (the
        three skills in `praxis.agents.skills`); this exists only so
        `WebConnector` satisfies the same generic contract every other
        connector in the registry does.
        """
        mode = params.get("mode", "search")
        if mode == "search":
            queries = params.get("queries") or [query]
            return await self.search(
                objective=params.get("objective", query),
                queries=queries,
                max_results=params.get("max_results", 5),
            )
        if mode == "read":
            return await self.read_page(
                query,
                question=params.get("question", query),
                known_urls=params.get("known_urls") or set(),
            )
        if mode == "crawl":
            return await self.crawl(
                query,
                question=params.get("question", query),
                known_urls=params.get("known_urls") or set(),
            )
        raise ValueError(f"WebConnector.read: unknown mode '{mode}' (expected search/read/crawl)")

    async def health(self) -> HealthStatus:
        # "The web" has no single health endpoint to poll (unlike every
        # other connector in this registry) - never raises; reports
        # whether a search provider is configured as the one cheap,
        # meaningful signal available, matching every other connector's
        # "health() never raises" discipline.
        detail = "" if self._search_provider is not None else "no search provider configured"
        return HealthStatus(name=self.name, healthy=True, detail=detail)

    # ------------------------------------------------------------------ #
    # search
    # ------------------------------------------------------------------ #

    async def search(self, objective: str, queries: list[str], max_results: int = 5) -> list[dict]:
        """Runs every query in `queries` against the configured search
        provider, deduping results by URL across queries. `objective`
        is accepted (per this phase's declared signature) for future
        provider implementations that can use it to steer relevance,
        even though `TavilySearchProvider` itself doesn't use it today.

        Raises `SearchProviderNotConfiguredError` - never a silent empty
        list disguised as "no results found" - when no provider is
        configured (spec §7: web access is an explicit deployment-level
        toggle, off by default).
        """
        if self._search_provider is None:
            raise SearchProviderNotConfiguredError(
                "web_search requires a configured search provider (e.g. PRAXIS_TAVILY_API_KEY "
                "set so a TavilySearchProvider can be constructed); refusing to return an empty "
                "result set disguised as a real search"
            )

        seen_urls: set[str] = set()
        results: list[dict] = []
        for query in queries:
            items = await self._search_provider.search(query, max_results=max_results)
            for item in items:
                if item.url in seen_urls:
                    continue
                seen_urls.add(item.url)
                results.append({"title": item.title, "url": item.url, "content": item.content})
        return results

    # ------------------------------------------------------------------ #
    # read_page
    # ------------------------------------------------------------------ #

    async def read_page(self, url: str, question: str, known_urls: set[str]) -> str:
        """Guarded-fetch -> robots check -> extract -> distill -> wrap.

        Raises `PriorContextViolationError` if `url` is not already in
        `known_urls` (see this module's docstring - checked first, before
        any network activity), `RobotsDisallowedError` if the origin's
        robots.txt disallows it, or any `guarded_fetch` error
        (`ssrf.SSRFRejectedError`, `TooManyRedirectsError`,
        `DisallowedContentTypeError`, `ByteCapExceededError`) - every one
        a distinguishable typed failure, never a fabricated success.
        """
        if url not in known_urls:
            raise PriorContextViolationError(
                f"refusing to fetch '{url}': it does not appear in the task's validated prior "
                "context (the intent text or an earlier tool result) - only URLs already "
                "established by prior context may be fetched (spec §6.1 prior-context-only fetch)",
                url=url,
            )

        if not await robots.is_allowed(url, user_agent=self._user_agent):
            raise RobotsDisallowedError(
                f"robots.txt disallows fetching '{url}' for user agent '{self._user_agent}'",
                url=url,
                user_agent=self._user_agent,
            )

        fetch_result = await guarded_fetch.fetch(url)
        text = extract_text(fetch_result.content, fetch_result.content_type)

        distilled = await self._distill(text, question=question, source=url)
        return untrusted.wrap_untrusted(distilled, source=url)

    # ------------------------------------------------------------------ #
    # crawl
    # ------------------------------------------------------------------ #

    async def crawl(
        self,
        start_url: str,
        question: str,
        known_urls: set[str],
        *,
        max_pages: int = 5,
        max_depth: int = 2,
        max_concurrency: int = 2,
        wall_clock_seconds: float = 25.0,
        max_total_bytes: int = 6_000_000,
    ) -> str:
        """Same-host BFS crawl from `start_url`, bounded on every axis
        the spec names (page count, depth, concurrency, wall-clock,
        total bytes), degrading to whatever was collected so far if the
        wall-clock runs out - never hanging. One distillation runs over
        the combined collected text at the end, not one per page.

        Only `start_url` is checked against `known_urls` (spec §6.1
        explicitly scopes the prior-context rule to the seed URL - links
        discovered mid-crawl are not required to already be known).
        """
        if start_url not in known_urls:
            raise PriorContextViolationError(
                f"refusing to crawl from '{start_url}': it does not appear in the task's "
                "validated prior context (spec §6.1 prior-context-only fetch)",
                url=start_url,
            )

        start_host = urlsplit(start_url).netloc
        deadline = time.monotonic() + wall_clock_seconds

        visited: set[str] = set()
        collected: list[tuple[str, str]] = []
        total_bytes = 0
        semaphore = asyncio.Semaphore(max(1, max_concurrency))
        state_lock = asyncio.Lock()

        async def _fetch_one(url: str, depth: int) -> list[str]:
            nonlocal total_bytes
            async with semaphore:
                if time.monotonic() >= deadline:
                    return []
                try:
                    if not await robots.is_allowed(url, user_agent=self._user_agent):
                        return []
                    fetch_result = await guarded_fetch.fetch(url)
                except FETCH_ERRORS:
                    return []
                except Exception:
                    # A raw network error (timeout, connection reset) on
                    # one page must not abort the whole crawl - it's
                    # simply a page that yields nothing, same posture as
                    # a policy-blocked page.
                    return []

                async with state_lock:
                    if total_bytes + len(fetch_result.content) > max_total_bytes:
                        return []
                    total_bytes += len(fetch_result.content)

                text = extract_text(fetch_result.content, fetch_result.content_type)
                async with state_lock:
                    collected.append((url, text))

                if depth >= max_depth:
                    return []
                return _extract_same_host_links(
                    fetch_result.content, fetch_result.content_type, url, start_host
                )

        frontier: list[tuple[str, int]] = [(start_url, 0)]

        while frontier and len(visited) < max_pages and time.monotonic() < deadline:
            batch: list[tuple[str, int]] = []
            batch_urls: set[str] = set()
            while frontier and len(visited) + len(batch) < max_pages:
                candidate_url, depth = frontier.pop(0)
                if candidate_url in visited or candidate_url in batch_urls:
                    continue
                batch.append((candidate_url, depth))
                batch_urls.add(candidate_url)

            if not batch:
                break

            visited.update(batch_urls)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*(_fetch_one(u, d) for u, d in batch)),
                    timeout=remaining,
                )
            except TimeoutError:
                # Wall-clock exhausted mid-batch: whatever individual
                # pages had already completed and appended to
                # `collected` before the timeout fired are kept (spec:
                # "degrades to partial results ... rather than hanging")
                # - only the links this now-abandoned batch would have
                # discovered are lost, never the whole crawl.
                break

            next_depth = {u: d + 1 for u, d in batch}
            for (batch_url, _), links in zip(batch, batch_results, strict=True):
                for link in links:
                    if link not in visited and len(visited) < max_pages:
                        frontier.append((link, next_depth[batch_url]))

        combined_text = "\n\n---\n\n".join(
            f"[Source: {page_url}]\n{page_text}" for page_url, page_text in collected
        )

        if not combined_text.strip():
            distilled = "(no content was successfully retrieved during the crawl)"
        else:
            source_label = f"crawl starting at {start_url} ({len(collected)} page(s) fetched)"
            distilled = await self._distill(combined_text, question=question, source=source_label)

        return untrusted.wrap_untrusted(distilled, source=f"crawl:{start_url}")

    # ------------------------------------------------------------------ #
    # shared distillation (spec §5: "distill before it reaches the
    # context window" - a cheap-tier LLM call, never the raw fetched
    # text handed to the orchestrating LLM)
    # ------------------------------------------------------------------ #

    async def _distill(self, text: str, *, question: str, source: str) -> str:
        prompt = self._prompt_manager.render(
            _DISTILL_PROMPT_NAME,
            _DISTILL_PROMPT_VERSION,
            text=text,
            question=question,
            source=source,
        )
        return await self._catalogue.complete("routing", prompt)


def _extract_same_host_links(
    content: bytes, content_type: str, base_url: str, start_host: str
) -> list[str]:
    base_type = content_type.split(";")[0].strip().lower()
    if base_type not in ("text/html", "application/xhtml+xml"):
        return []
    try:
        soup = BeautifulSoup(content, "html.parser")
    except Exception:  # noqa: BLE001 - a malformed page yields no links, never a crawl crash
        return []

    links: list[str] = []
    seen: set[str] = set()
    for tag in soup.find_all("a", href=True):
        raw_href = tag["href"]
        if not isinstance(raw_href, str):
            continue
        href = raw_href.strip()
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlsplit(absolute)
        if parsed.scheme not in ("http", "https") or parsed.netloc != start_host:
            continue
        normalized = absolute.split("#")[0]
        if normalized in seen:
            continue
        seen.add(normalized)
        links.append(normalized)
    return links


register_connector_factory(
    ConnectorFactory(
        name="web",
        is_configured=lambda settings: bool(settings.tavily_api_key),
        build=lambda settings: WebConnector(
            search_provider=TavilySearchProvider(
                api_key=required(settings.tavily_api_key, setting="tavily_api_key")
            )
        ),
    )
)
