# praxis/connectors/errors.py
"""Shared, distinguishable failure types for the data-source connectors
(spec §6, §12).

The three connectors written in Phase 2 (github, slack, prometheus) let
their transport library's own exception escape - an `httpx.HTTPStatusError`
or a `SlackApiError`. That was tolerable when every connector spoke HTTP
through the same one client library; it stops being tolerable once the
set spans four unrelated transports (httpx, motor/pymongo, aioboto3/
botocore, the neo4j Bolt driver), because a caller would then have to
know which library each connector happens to be built on just to tell
"I could not reach the server" apart from "the server rejected your
query" apart from "you never configured this".

So the six connectors added here map their transport's errors onto the
three-way taxonomy below at the boundary, and never let a raw driver
exception escape from `describe()`/`read()`/`write()`. The distinction
is the one an autonomous caller actually has to act on:

- `ConnectorConfigurationError` - the call could not even be attempted.
  Nothing was sent. Retrying is pointless; an operator has to change
  config. (Also what a read-only connector raises for a write it will
  never perform - see `ConnectorReadOnlyError`.)
- `ConnectorUnavailableError` - the request was attempted and the far
  side could not be reached or did not answer in time. Retrying may
  well work, which is exactly what `praxis.connectors.retry.
  call_with_retry` is for.
- `ConnectorQueryError` - the far side was reached and answered, and
  the answer was a refusal: bad syntax, a missing collection/index/
  bucket, a permission denial, a result that blew a configured bound.
  Retrying the identical request will fail identically.

Every one of these is raised at the exact point the operation is
refused or fails. None of them is ever caught and downgraded into an
empty list, an empty dict, or a `None` that a caller could mistake for
a real, empty-but-successful result - the failure mode this taxonomy
exists to make impossible.

These deliberately do NOT subclass `praxis.core.exceptions.ConnectorError`.
That type has a fixed constructor carrying a retry *attempt count*,
because it is what `call_with_retry` raises once a bounded retry loop is
exhausted - it describes a wrapper's give-up, one level above a single
call's failure. A connector raising one directly would have to invent an
`attempts` number for a call that was never retried. The two compose
instead: a single call raises one of these, and `call_with_retry`
wrapping that call raises `ConnectorError` carrying this one's `str()`
as its `detail`.
"""
from __future__ import annotations

__all__ = [
    "DataSourceError",
    "ConnectorConfigurationError",
    "ConnectorReadOnlyError",
    "ConnectorUnavailableError",
    "ConnectorQueryError",
]


class DataSourceError(Exception):
    """Base for every failure below - lets a caller catch "this connector
    failed, in a way the connector itself classified" in one clause,
    distinct from a bug (a `TypeError`, an `AttributeError`) escaping the
    connector's own code.

    `connector_name` is the registered name of the connector that
    failed, not its kind, so a deployment with two REST connectors can
    tell which one broke. `detail` is the underlying cause's own text
    (the driver's exception message, the server's error body) kept
    verbatim as an attribute rather than only folded into the message
    string, so a caller or a log line can inspect the real cause without
    re-parsing a formatted message.
    """

    def __init__(self, message: str, *, connector_name: str, detail: str = "") -> None:
        super().__init__(f"{message}: {detail}" if detail else message)
        self.connector_name = connector_name
        self.detail = detail


class ConnectorConfigurationError(DataSourceError):
    """The call could not be attempted because the connector is missing
    something an operator has to supply - a database name, a bucket, a
    credential the factory's `is_configured` gate did not cover.

    Raised in preference to a silent no-op (spec §6's "degrades
    gracefully" is about *not registering* an unconfigured connector at
    bootstrap, never about a registered connector quietly returning
    nothing at call time). Subclasses nothing builtin deliberately: this
    is not a `ValueError` about the *caller's* argument, it is about the
    deployment's own configuration, and conflating the two would let a
    generic `except ValueError` around a query swallow a misconfiguration
    that no query change can fix.
    """


class ConnectorReadOnlyError(ConnectorConfigurationError, PermissionError):
    """A write (or a read whose content would mutate the far side) was
    refused because this connector is registered read-only.

    Subclasses `PermissionError` so it is caught by exactly the same
    `except PermissionError` clause that the base
    `praxis.core.interfaces.Connector.write()`'s own refusal - and
    `PostgresConnector`/`SQLConnector`'s mutation-shaped-query refusal -
    already raise, keeping one uniform "the safety net said no" catch
    across old and new connectors. Subclasses
    `ConnectorConfigurationError` as well because it is, genuinely, a
    configuration outcome: the same call against the same connector
    registered `read_only=False` would have been attempted.
    """


class ConnectorUnavailableError(DataSourceError):
    """The far side could not be reached, or did not answer within the
    connector's timeout. Nothing is known about whether the operation
    took effect (for a write, this is genuinely ambiguous - a request
    that timed out may still have been applied server-side, which is why
    this is a distinct type rather than folded into
    `ConnectorQueryError`).
    """


class ConnectorQueryError(DataSourceError):
    """The far side was reached and refused: malformed query, unknown
    collection/index/bucket/label, permission denied at the data layer,
    or a result that exceeded a bound this connector enforces.

    `status` carries the transport-level or driver-level code when the
    far side supplied one (an HTTP status, a MongoDB/botocore error
    code, a Neo4j error code) and is `None` when it did not - kept as an
    attribute so a caller can branch on "404, the thing does not exist"
    versus "403, I am not allowed" without string-matching `detail`.
    """

    def __init__(
        self,
        message: str,
        *,
        connector_name: str,
        detail: str = "",
        status: int | str | None = None,
    ) -> None:
        super().__init__(message, connector_name=connector_name, detail=detail)
        self.status = status


def describe_exception(exc: BaseException) -> str:
    """The cause's text, guaranteed non-empty.

    Several of the exceptions that matter most here stringify to the
    empty string. `httpx.ConnectError` is the important one: it is what
    a refused connection raises, which is the single most common outage
    shape a connector sees. `str(exc)` on it yields `""`, so an error
    built from it reads "connector 'ledger' could not GET '/invoices'"
    with no cause attached, and a `health()` verdict whose `detail` is
    blank - exactly when someone is trying to find out what broke.

    Falling back to the class name is not as good as a message, but
    "ConnectError" tells an operator it was the connection rather than
    the query, which is the distinction they need first. Where the
    exception does carry text, that text is used unchanged.
    """
    text = str(exc).strip()
    if text:
        return text
    return type(exc).__name__
