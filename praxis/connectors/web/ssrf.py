# praxis/connectors/web/ssrf.py
"""SSRF-safe hostname/IP validation (spec §6.1): "resolve and pin the
connection to a validated IP (never trust a hostname that can rebind
mid-request), reject loopback/private/link-local/metadata addresses and
credentials-in-URL."

`resolve_and_validate` is the one function every fetch attempt in this
subpackage calls - once for the initial URL, and again for every
redirect hop (`guarded_fetch.fetch`) - so a hostname is never connected
to without this check running immediately beforehand, on the exact
target of that specific connection attempt.

Real IP-range validation via the stdlib `ipaddress` module - no
hand-rolled CIDR math. A literal IP address in the URL (e.g.
`http://127.0.0.1/`) is validated directly with **no DNS lookup at
all** (`ipaddress.ip_address` parses it locally); a real hostname goes
through `socket.getaddrinfo`, and *every* address it resolves to is
checked - if any one of them is disallowed, the whole hostname is
rejected outright, rather than picking only the "good" address and
silently ignoring that the same name also resolves to a private one
(a multi-A-record bypass otherwise: some DNS answers deliberately mix a
public IP with a private one specifically to slip past a validator that
only checks the first result).
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit


class SSRFRejectedError(Exception):
    """Raised for any hostname/IP/URL rejected by SSRF-safety validation.

    `host` is the hostname/IP that was rejected; `reason` is a short,
    stable machine-readable code (e.g. `"disallowed_ip_range"`,
    `"embedded_credentials"`, `"dns_resolution_failed"`,
    `"disallowed_scheme"`) - kept as attributes, not folded away, so a
    caller/test can distinguish exactly why a fetch was refused, per
    spec §12's "a blocked outcome must be visibly one of those, not a
    string a caller has to string-match."
    """

    def __init__(self, message: str, *, host: str, reason: str) -> None:
        super().__init__(message)
        self.host = host
        self.reason = reason


def _is_disallowed_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True for loopback (127.0.0.0/8, ::1), private-use (10/8,
    172.16/12, 192.168/16, fc00::/7), link-local (169.254.0.0/16 -
    which is exactly the range the cloud-metadata address
    169.254.169.254 lives in - and fe80::/10), and every other IANA
    special-purpose range `ipaddress` itself recognizes (reserved,
    multicast, unspecified `0.0.0.0`/`::`)."""
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def assert_no_embedded_credentials(url: str) -> None:
    """Rejects a URL of the form `scheme://user:pass@host/...` (spec
    §6.1: "reject credentials-in-URL") - a real attack surface for a
    synthesized or prompt-injected fetch: e.g. embedding an internal
    system's Basic-Auth credentials directly in a constructed URL.
    """
    parts = urlsplit(url)
    if parts.username or parts.password:
        raise SSRFRejectedError(
            f"URL contains embedded credentials, which is never allowed: {url}",
            host=parts.hostname or "",
            reason="embedded_credentials",
        )


def resolve_and_validate(hostname: str) -> str:
    """Resolves `hostname` and validates every resolved address is
    allowed, returning one validated IP (as a string) to pin the actual
    connection to.

    A bracketed IPv6 literal (`[::1]`, as it appears in a URL's
    authority component) is unwrapped before parsing. Raises
    `SSRFRejectedError` if `hostname` is itself a disallowed IP literal,
    if DNS resolution fails outright, or if resolution succeeds but
    yields no usable address, or if it succeeds and *any* resolved
    address is disallowed.
    """
    bare = hostname.strip("[]")

    try:
        literal = ipaddress.ip_address(bare)
    except ValueError:
        literal = None

    if literal is not None:
        # No DNS query at all for a literal IP - nothing to resolve.
        if _is_disallowed_ip(literal):
            raise SSRFRejectedError(
                f"IP address '{hostname}' is not allowed",
                host=hostname,
                reason="disallowed_ip_range",
            )
        return str(literal)

    try:
        addrinfo = socket.getaddrinfo(bare, None)
    except socket.gaierror as exc:
        raise SSRFRejectedError(
            f"could not resolve hostname '{hostname}': {exc}",
            host=hostname,
            reason="dns_resolution_failed",
        ) from exc

    resolved_ips = {str(info[4][0]) for info in addrinfo}
    if not resolved_ips:
        raise SSRFRejectedError(
            f"hostname '{hostname}' resolved to no addresses", host=hostname, reason="no_addresses"
        )

    validated: list[str] = []
    for ip_str in resolved_ips:
        ip_obj = ipaddress.ip_address(ip_str)
        if _is_disallowed_ip(ip_obj):
            raise SSRFRejectedError(
                f"hostname '{hostname}' resolves to disallowed address {ip_str}; refusing the "
                "whole hostname rather than silently picking a different resolved address",
                host=hostname,
                reason="disallowed_ip_range",
            )
        validated.append(ip_str)

    return validated[0]
