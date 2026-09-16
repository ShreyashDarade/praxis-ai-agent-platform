# praxis/connectors/web/errors.py
"""Shared, distinguishable failure types for the web connector's
guarded-fetch pipeline (spec §6.1, §12): "a blocked or errored fetch
must be a distinguishable typed outcome, never a string that could be
mistaken for real content." Every one of these is raised by
`guarded_fetch.py`, `robots.py`, or `connector.py` at the exact point a
fetch is refused or aborted - never caught and silently downgraded into
a plain string result anywhere in this subpackage. `ssrf.SSRFRejectedError`
is the one exception in this family defined in its own module (`ssrf.py`
has no reason to import this one back), but is a `FetchError` too so a
caller can catch `FetchError` for "any guarded-fetch policy refusal"
uniformly - see the note on `FetchError` below.
"""
from __future__ import annotations

from praxis.connectors.web.ssrf import SSRFRejectedError

__all__ = [
    "FetchError",
    "FETCH_ERRORS",
    "SSRFRejectedError",
    "TooManyRedirectsError",
    "DisallowedContentTypeError",
    "ByteCapExceededError",
    "RobotsDisallowedError",
    "PriorContextViolationError",
    "SearchProviderNotConfiguredError",
]


class FetchError(Exception):
    """Base class for every guarded-fetch-pipeline-specific failure
    below - lets a caller catch "any policy-level fetch refusal" in one
    except clause, distinct from e.g. a raw `httpx` network error.

    `SSRFRejectedError` (defined in `ssrf.py`, a lower-level module this
    one depends on rather than the other way around) is deliberately
    NOT made a subclass of this - plain `Exception` subclasses can't be
    retroactively registered as virtual subclasses the way an `abc.ABC`
    can. `FETCH_ERRORS` below is the tuple to catch when a caller wants
    "any guarded-fetch policy refusal, SSRF included" in one place.
    """


# Every caller in this subpackage that wants "any policy-level refusal
# from the guarded-fetch pipeline, SSRF rejection included" catches this
# tuple, rather than `FetchError` alone - see the class docstring above.
FETCH_ERRORS = (FetchError, SSRFRejectedError)


class TooManyRedirectsError(FetchError):
    """Raised when a redirect chain exceeds `max_hops` (spec §6.1: "a
    small hop limit"). `hops` is how many redirects had actually been
    followed when the limit was hit; `max_hops` is the configured cap.
    """

    def __init__(self, message: str, *, hops: int, max_hops: int) -> None:
        super().__init__(message)
        self.hops = hops
        self.max_hops = max_hops


class DisallowedContentTypeError(FetchError):
    """Raised when a response's `Content-Type` header, or the actual
    sniffed bytes of its body, doesn't match an allowed content type
    (spec §6.1: "verified against both headers and sniffed bytes")."""

    def __init__(self, message: str, *, content_type: str, url: str) -> None:
        super().__init__(message)
        self.content_type = content_type
        self.url = url


class ByteCapExceededError(FetchError):
    """Raised the instant a streamed response body exceeds `max_bytes`.

    `partial_content` is exactly the bytes actually read off the wire
    before aborting - strictly bounded by (approximately) `max_bytes`,
    never the full body read first and truncated after the fact (spec
    §6.1: "streamed and aborted past a byte cap rather than reading an
    unbounded body into memory") - kept as an attribute so a test (or a
    caller wanting best-effort partial data) can inspect exactly what
    had been read at the moment of abort.
    """

    def __init__(self, message: str, *, max_bytes: int, partial_content: bytes) -> None:
        super().__init__(message)
        self.max_bytes = max_bytes
        self.partial_content = partial_content


class RobotsDisallowedError(FetchError):
    """Raised when a target URL's origin's robots.txt disallows fetching
    it for the given user agent (spec §6.1: "robots.txt ... honored,
    cached per origin, checked before fetch")."""

    def __init__(self, message: str, *, url: str, user_agent: str) -> None:
        super().__init__(message)
        self.url = url
        self.user_agent = user_agent


class PriorContextViolationError(PermissionError, FetchError):
    """Raised when `web_read`/`web_crawl` is asked to fetch a URL that
    does not already appear in the task's validated prior context (spec
    §6.1's "prior-context-only fetch") - the one rule this entire phase
    exists to enforce; see `WebConnector.read_page`/`.crawl`. Subclasses
    `PermissionError` (matching the base `Connector.write()`'s own
    read-only-violation posture elsewhere in this codebase) as well as
    `FetchError`, so either an "is this a permission problem" or an
    "is this a guarded-fetch policy refusal" catch clause sees it.
    """

    def __init__(self, message: str, *, url: str) -> None:
        super().__init__(message)
        self.url = url


class SearchProviderNotConfiguredError(FetchError):
    """Raised by `WebConnector.search()` when no search provider is
    configured (e.g. no `PRAXIS_TAVILY_API_KEY` set) - the same posture
    as `praxis.agents.skill.SkillConfigurationError` for
    `post_slack_message` when Slack is unconfigured: a clear, typed
    failure, never a silent empty-results no-op that looks like a
    genuine "no results found" (spec §12 distinguishes that as
    `barren`, a different outcome from "can't even try")."""
