# praxis/connectors/web/guarded_fetch.py
"""SSRF-safe, bounded fetch of a single URL (spec §6.1).

Every safety property below is enforced in this one function, in this
order, on the *initial* URL and again identically on *every redirect
hop* - never only on the first request:

1. Reject embedded credentials (`ssrf.assert_no_embedded_credentials`).
2. Reject a non-http(s) scheme outright (`file://`, `gopher://`, ...).
3. Resolve the hostname and validate the resolved IP
   (`ssrf.resolve_and_validate`) - real DNS resolution done by this
   code, never left to whatever the underlying HTTP client would have
   resolved on its own moments later.
4. Pin the actual connection to that validated IP: the request is sent
   to the validated IP address directly (`httpx.URL.copy_with(host=...)`),
   while the original hostname travels as both the `Host` header and the
   TLS SNI (`extensions={"sni_hostname": ...}`, honored by `httpcore`'s
   connection classes) - so the certificate is still verified against
   the real hostname, but the TCP connection cannot be silently rebound
   to a different address between validation and connection (the whole
   point of "pin the connection to that validated IP" in spec §6.1).
   `follow_redirects=False` is set on the client; redirects are read and
   re-validated one hop at a time by this same function, up to
   `max_hops` - never `follow_redirects=True`.
5. The response `Content-Type` header is checked against
   `allowed_content_types` *before* any body is read.
6. The body is streamed (never `response.aread()`-then-truncate): each
   chunk is checked against the running byte total, aborting
   (`ByteCapExceededError`, carrying exactly what was read so far) the
   instant `max_bytes` would be exceeded; the *first* non-empty chunk is
   also sniffed against the claimed content-type family
   (`_sniff_looks_like`) so a mislabeled `Content-Type` header alone
   can't smuggle a disallowed body past step 5.

**A note on how this is tested** (see `tests/connectors/web/
test_guarded_fetch.py`): `respx` intercepts at `httpcore`'s connection-
pool layer, one level *above* where a real TCP socket would actually be
opened - so a mocked test genuinely proves steps 1-3 and 5-6 (an SSRF
rejection mid-redirect-chain, a byte-cap abort, a content-type
rejection, the hop limit), and proves the *pinned URL* respx received is
the validated IP with the original `Host` header (step 4's visible,
request-level effect) - but it cannot prove a real socket only ever
connects to that IP on a live network, since no real socket is opened
under a mock at all. That last mile is inherent to testing above the
transport layer with any mocking approach, not a gap specific to this
implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from praxis.connectors.web import ssrf
from praxis.connectors.web.errors import (
    ByteCapExceededError,
    DisallowedContentTypeError,
    TooManyRedirectsError,
)

DEFAULT_ALLOWED_CONTENT_TYPES: tuple[str, ...] = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "application/pdf",
)
DEFAULT_MAX_HOPS = 3
DEFAULT_MAX_BYTES = 2_000_000
DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_USER_AGENT = "PraxisBot/1.0 (+web_read; guarded-fetch)"

_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
_CHUNK_SIZE = 8192


@dataclass
class FetchResult:
    content: bytes
    content_type: str
    final_url: str
    status_code: int


def _content_type_family(content_type: str) -> str | None:
    """Maps a `Content-Type` header value to a coarse family used both
    for the allow-list check and the byte-sniff check below. `None`
    means "not one of the families this fetch pipeline understands" -
    callers treat that as disallowed."""
    base = content_type.split(";")[0].strip().lower()
    if base == "application/pdf":
        return "pdf"
    if base in ("text/html", "application/xhtml+xml"):
        return "html"
    if base.startswith("text/"):
        return "text"
    return None


def _sniff_looks_like(family: str, sample: bytes) -> bool:
    """Cheap sniff of the first bytes actually read off the wire,
    confirming the body plausibly matches its claimed content-type
    family (spec §6.1: "verified against both headers and sniffed
    bytes") - catches a server whose `Content-Type` header lies about
    what it's actually serving."""
    if not sample:
        return True
    if family == "pdf":
        return sample[:5] == b"%PDF-"
    # html/text: a real text/HTML page is overwhelmingly printable
    # ASCII/UTF-8 - reject a sample that looks like binary junk (a high
    # proportion of control bytes other than common whitespace).
    control_bytes = sum(1 for b in sample if b < 9 or (13 < b < 32))
    return control_bytes / len(sample) < 0.05


def _allowed_families(allowed_content_types: tuple[str, ...]) -> set[str]:
    families = set()
    for content_type in allowed_content_types:
        family = _content_type_family(content_type)
        if family is not None:
            families.add(family)
    return families


async def fetch(
    url: str,
    *,
    max_hops: int = DEFAULT_MAX_HOPS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    allowed_content_types: tuple[str, ...] = DEFAULT_ALLOWED_CONTENT_TYPES,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_USER_AGENT,
) -> FetchResult:
    """Fetches `url` under every safety property described in this
    module's docstring, returning a `FetchResult` on success. Raises
    `ssrf.SSRFRejectedError`, `TooManyRedirectsError`,
    `DisallowedContentTypeError`, or `ByteCapExceededError` -
    distinguishable typed failures, never a string result (spec §12).
    """
    allowed_families = _allowed_families(allowed_content_types)
    current_url = url
    hops = 0

    async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=False) as client:
        while True:
            ssrf.assert_no_embedded_credentials(current_url)
            parsed = httpx.URL(current_url)
            if parsed.scheme not in ("http", "https"):
                raise ssrf.SSRFRejectedError(
                    f"scheme '{parsed.scheme}' is not allowed for a guarded fetch "
                    f"(url: {current_url})",
                    host=parsed.host or "",
                    reason="disallowed_scheme",
                )
            hostname = parsed.host
            if not hostname:
                raise ssrf.SSRFRejectedError(
                    f"URL has no host: {current_url}", host="", reason="no_host"
                )

            # Real DNS resolution + range validation, on THIS hop's own
            # host - re-run every single time this loop iterates, i.e.
            # on the initial URL and again on every redirect target.
            validated_ip = ssrf.resolve_and_validate(hostname)

            # Pin the connection: the actual request goes to the
            # validated IP; the original hostname is preserved for the
            # Host header and TLS SNI (see module docstring, step 4).
            pinned_url = parsed.copy_with(host=validated_ip)
            headers = {"Host": hostname, "User-Agent": user_agent}
            request = client.build_request(
                "GET", pinned_url, headers=headers, extensions={"sni_hostname": hostname}
            )

            response = await client.send(request, stream=True)
            try:
                if (
                    response.status_code in _REDIRECT_STATUS_CODES
                    and "location" in response.headers
                ):
                    location = response.headers["location"]
                    hops += 1
                    if hops > max_hops:
                        raise TooManyRedirectsError(
                            f"exceeded {max_hops} redirect hop(s) fetching '{url}' "
                            f"(next hop would be to '{location}')",
                            hops=hops,
                            max_hops=max_hops,
                        )
                    # Resolved against the *unpinned* current URL, not
                    # the pinned one, so a relative Location header
                    # resolves against the real hostname/path.
                    current_url = str(httpx.URL(current_url).join(location))
                    continue

                content_type_header = response.headers.get("content-type", "")
                family = _content_type_family(content_type_header)
                if family is None or family not in allowed_families:
                    raise DisallowedContentTypeError(
                        f"content type '{content_type_header}' is not allowed "
                        f"fetching '{current_url}'",
                        content_type=content_type_header,
                        url=current_url,
                    )

                collected = bytearray()
                sniffed = False
                async for chunk in response.aiter_bytes(_CHUNK_SIZE):
                    if not chunk:
                        continue
                    collected.extend(chunk)
                    if not sniffed:
                        sniffed = True
                        if not _sniff_looks_like(family, bytes(collected)):
                            raise DisallowedContentTypeError(
                                f"response body for '{current_url}' does not look like the "
                                f"declared content type '{content_type_header}'",
                                content_type=content_type_header,
                                url=current_url,
                            )
                    if len(collected) > max_bytes:
                        raise ByteCapExceededError(
                            f"response body for '{current_url}' exceeded the {max_bytes}-byte cap",
                            max_bytes=max_bytes,
                            partial_content=bytes(collected),
                        )

                return FetchResult(
                    content=bytes(collected),
                    content_type=content_type_header,
                    final_url=current_url,
                    status_code=response.status_code,
                )
            finally:
                await response.aclose()
