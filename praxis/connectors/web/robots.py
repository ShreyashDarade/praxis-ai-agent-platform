# praxis/connectors/web/robots.py
"""robots.txt enforcement, cached per origin (spec §6.1: "robots.txt and
content-type honored: cached per origin, checked before fetch").

Fetches `robots.txt` itself through `guarded_fetch.fetch` rather than a
second, separate `httpx` call - robots.txt IS an external URL fetch,
so it gets the exact same SSRF-safe pinned-redirect-bounded treatment
as the real content fetch it's gating, per spec §6.1's "one shared,
required baseline rather than each tool inventing its own." Any
guarded-fetch failure fetching robots.txt itself (unreachable, SSRF-
rejected, wrong content type, byte cap) is treated exactly like a
missing robots.txt: allow by default (spec: "404 (allow-by-default)") -
Praxis can't honor a disallow rule it can't safely retrieve, and
refusing every fetch to an origin merely because *that origin's*
robots.txt was unreachable would be a much larger, unintended
availability regression than the spec asks for here.

Caching: `InMemoryCache` keyed by origin (`scheme://host[:port]`), a
short TTL (spec's own suggested "e.g. 900s") - a second `is_allowed`
call to the same origin within the TTL never re-fetches robots.txt at
all, real cache-hit, not just "would be fast anyway."
"""
from __future__ import annotations

from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from praxis.cache.memory_cache import InMemoryCache
from praxis.connectors.web import guarded_fetch
from praxis.connectors.web.errors import FETCH_ERRORS

DEFAULT_USER_AGENT = "PraxisBot"
ROBOTS_CACHE_TTL_SECONDS = 900
_ROBOTS_MAX_BYTES = 200_000

# Module-level, process-wide - deliberately NOT task-scoped (unlike the
# Orchestrator's per-task tool-result cache, spec §11): the whole point
# is that repeated fetches to the same site *across* tasks don't each
# re-fetch robots.txt, and robots.txt rules aren't task-specific data.
_robots_cache = InMemoryCache()


async def _fetch_robots_text(origin: str) -> str:
    robots_url = f"{origin}/robots.txt"
    try:
        result = await guarded_fetch.fetch(
            robots_url,
            max_bytes=_ROBOTS_MAX_BYTES,
            allowed_content_types=("text/plain", "text/html"),
        )
    except (*FETCH_ERRORS, httpx.HTTPError):
        # Any guarded-fetch policy refusal (SSRF-rejected, disallowed
        # content type, byte cap, too many redirects) OR a raw network
        # failure (timeout, connection refused, DNS hiccup) fetching
        # robots.txt itself - both degrade to "allow by default", never
        # propagate up and block the real content fetch this is gating.
        return ""
    if result.status_code >= 400:
        return ""  # e.g. a real 404 - allow by default, not "disallow everything"
    return result.content.decode("utf-8", errors="replace")


async def is_allowed(url: str, user_agent: str = DEFAULT_USER_AGENT) -> bool:
    """True if `user_agent` may fetch `url` per its origin's robots.txt
    (real `urllib.robotparser.RobotFileParser` semantics - not a
    hand-rolled parser)."""
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"

    cached = await _robots_cache.get(origin)
    if cached is None:
        robots_text = await _fetch_robots_text(origin)
        await _robots_cache.set(origin, robots_text, ttl_seconds=ROBOTS_CACHE_TTL_SECONDS)
    else:
        robots_text = cached

    parser = RobotFileParser()
    parser.parse(robots_text.splitlines())
    return parser.can_fetch(user_agent, url)
