# praxis/connectors/s3/s3_connector.py
"""S3-compatible object-storage connector, built on aioboto3 (spec §6).

**Why aioboto3 rather than httpx + hand-rolled SigV4.** Signing an S3
request is not "add a header". AWS Signature Version 4 requires a
canonical request built from a URI-encoded-but-not-doubly-encoded path,
query parameters sorted by byte order, headers lowercased, trimmed of
interior whitespace and joined, a signed-headers list that must exactly
match what goes on the wire, a payload hash (or the literal
`UNSIGNED-PAYLOAD`), and a derived signing key chained through four
HMACs. Every S3-compatible vendor has its own small deviations, and
every mistake produces the same opaque `SignatureDoesNotMatch`. That is
a cryptographic protocol implementation, and putting one in this
codebase to avoid a dependency would be trading a maintained,
AWS-authored implementation for a novel one nobody here can test
against the dozen providers this is meant to work with.

`aioboto3` is a thin async wrapper over `aiobotocore`, which is itself
botocore with an aiohttp transport - so the request model, the signer,
the endpoint resolution and the per-provider quirks are all
upstream-maintained. It installed cleanly here (aiohttp was already
present for `SlackConnector`'s own transport), so the "if aioboto3 is
heavy, hand-roll it" fallback did not have to be taken.

**Scope, and why this is not `S3BlobStore`.** `praxis.memory.s3_blob_store`
is also built on aioboto3 and also talks to S3, and the two are
deliberately separate. A `praxis.core.interfaces.BlobStore` is *Praxis's
own* artifact storage: Praxis chooses the bucket, writes the keys, and
knows the layout, so its interface is `put`/`get` over opaque bytes. This
is a `Connector`: an external store that somebody else's data already
lives in, which Praxis reads and must first *describe* to find out what
is there. Sharing an implementation would mean one of them carrying the
other's concerns - a blob store that has to introspect, or a connector
that cannot be pointed at an arbitrary bucket.
"""
from __future__ import annotations

from typing import Any

import aioboto3
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
)

from praxis.connectors.errors import (
    ConnectorConfigurationError,
    ConnectorQueryError,
    ConnectorReadOnlyError,
    ConnectorUnavailableError,
    describe_exception,
)
from praxis.connectors.factory import (
    ConnectorFactory,
    register_connector_factory,
    required,
)
from praxis.core.interfaces import Connector, ConnectorDescription, HealthStatus

# botocore requires a region to build a client even for providers that
# ignore it entirely (MinIO, Ceph RGW). us-east-1 is the conventional
# stand-in and is what every S3-compatible implementation accepts.
_DEFAULT_REGION = "us-east-1"

# `read()` decodes an object into a string, so the whole body lands in
# memory and, usually, in an LLM's context window after that. Objects
# past this are refused rather than truncated - a half-read CSV or a
# truncated JSON document is worse than no document, because it parses.
_MAX_OBJECT_BYTES = 5_000_000

# `describe()`'s bounded peek into the default bucket. One page of
# `list_objects_v2` and no pagination: a description exists to tell a
# caller what kind of thing lives here, and a bucket with a million
# keys would otherwise turn `describe()` into thousands of round trips.
_MAX_SAMPLED_KEYS = 100

# Returned by `ListAllMyBuckets` for a key that is correctly configured
# but scoped to specific buckets - extremely common, and not an
# unhealthy connector. See `health()`.
_ACCESS_DENIED_CODES = frozenset({"AccessDenied", "AccessDeniedException", "Forbidden", "403"})

# botocore's own defaults are 60s connect, 60s read and up to 5 attempts
# in "legacy" retry mode - tuned for a long-running batch job, not for
# one step of an agent's task graph, where a dead endpoint would stall
# the whole task for minutes before reporting anything. "standard" mode
# is botocore's own bounded, jittered policy and is what AWS recommends
# over "legacy"; three attempts is the bound Praxis already uses for a
# connector call elsewhere (`praxis.connectors.retry.DEFAULT_MAX_ATTEMPTS`).
_DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
_DEFAULT_READ_TIMEOUT_SECONDS = 30
_DEFAULT_MAX_ATTEMPTS = 3


class S3Connector(Connector):
    """Read (and, if configured, write) access to one S3-compatible endpoint.

    `endpoint_url` unset means real AWS S3; anything else (MinIO, Ceph
    RGW, R2, B2) needs its own endpoint. A fresh client is created per
    operation through `aioboto3.Session.client(...)`'s async context
    manager, for the same event-loop-affinity reason documented on
    `praxis.connectors.mongodb.mongodb_connector.MongoDBConnector`: the
    underlying aiohttp session belongs to the loop that created it, and
    `Connector` has no `close()` hook through which a cached one could
    be torn down.
    """

    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        *,
        name: str = "s3",
        read_only: bool = True,
        endpoint_url: str | None = None,
        region: str | None = None,
        default_bucket: str | None = None,
        connect_timeout_seconds: int = _DEFAULT_CONNECT_TIMEOUT_SECONDS,
        read_timeout_seconds: int = _DEFAULT_READ_TIMEOUT_SECONDS,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self.name = name
        self.read_only = read_only
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._endpoint_url = endpoint_url
        self._region = region or _DEFAULT_REGION
        self._default_bucket = default_bucket
        self._config = Config(
            connect_timeout=connect_timeout_seconds,
            read_timeout=read_timeout_seconds,
            retries={"max_attempts": max_attempts, "mode": "standard"},
        )
        # The *session* is cached; the *client* is not. A botocore
        # session owns the loader cache for service models, and S3's
        # model is a multi-megabyte JSON document whose parse dominates
        # the cost of a small request - rebuilding a session per call
        # measurably costs seconds. Unlike a client, a session holds no
        # aiohttp connector and so has no event-loop affinity, which is
        # what makes caching this one safe where caching the client
        # would not be.
        self._session = aioboto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=self._region,
        )

    def _client(self) -> Any:
        """An async context manager yielding a configured S3 client.

        Returns the context manager rather than the client because
        `aioboto3`'s client is only valid inside `async with` - its
        aiohttp session is opened on enter and closed on exit, and a
        client extracted from it would fail on first use.
        """
        return self._session.client(
            "s3", endpoint_url=self._endpoint_url, config=self._config
        )

    async def describe(self) -> ConnectorDescription:
        """Lists buckets, plus a bounded sample of keys in the default bucket.

        The key sample is the object-store analogue of a table's
        columns: a bucket name alone says nothing about what is in it,
        whereas a hundred keys immediately show the prefix layout
        (`year=2024/month=03/...`) and the file types a task would be
        reading. It is only fetched when a `default_bucket` is
        configured, because without one there is no non-arbitrary
        bucket to sample.

        **A scoped credential often cannot list buckets at all.**
        `ListAllMyBuckets` is an account-level permission that most
        well-scoped keys deliberately lack, and several S3-compatible
        providers do not implement it. That is reported as
        `"buckets": None` with a `"buckets_error"` explaining why,
        while the key sample still goes ahead - a `describe()` that
        raised here would make a perfectly usable, correctly-scoped
        connector look broken.
        """
        schema: dict[str, Any] = {
            "endpoint_url": self._endpoint_url or "https://s3.amazonaws.com (AWS)",
            "region": self._region,
            "read_only": self.read_only,
            "default_bucket": self._default_bucket,
        }

        async with self._client() as client:
            try:
                listing = await client.list_buckets()
                schema["buckets"] = [bucket["Name"] for bucket in listing.get("Buckets", [])]
            except ClientError as exc:
                if _error_code(exc) not in _ACCESS_DENIED_CODES:
                    raise _map_boto_error(exc, self.name, "list buckets") from exc
                schema["buckets"] = None
                schema["buckets_error"] = (
                    "this credential may not list buckets (ListAllMyBuckets denied); "
                    "that does not affect reading objects from buckets it is scoped to"
                )
            except (BotoCoreError, OSError) as exc:
                raise _map_boto_error(exc, self.name, "list buckets") from exc

            if self._default_bucket:
                try:
                    objects = await client.list_objects_v2(
                        Bucket=self._default_bucket, MaxKeys=_MAX_SAMPLED_KEYS
                    )
                except (BotoCoreError, ClientError, OSError) as exc:
                    raise _map_boto_error(
                        exc, self.name, f"list objects in '{self._default_bucket}'"
                    ) from exc
                schema["sampled_keys"] = [item["Key"] for item in objects.get("Contents", [])]
                schema["sampled_keys_truncated"] = bool(objects.get("IsTruncated"))

        return ConnectorDescription(kind="s3", schema=schema)

    async def read(self, query: str, **params: Any) -> Any:
        """Fetches one object and returns its decoded text.

        `query` addresses the object in one of three ways:

        - ``s3://bucket/path/to/object.json`` - fully explicit.
        - ``path/to/object.json`` with ``bucket="..."`` as a keyword -
          also fully explicit.
        - ``path/to/object.json`` alone - uses `default_bucket`, and
          raises `ConnectorConfigurationError` when none is configured.

        There is deliberately no "split the first path segment off as
        the bucket" convenience, because `reports/2024/q1.csv` is a
        perfectly ordinary key and guessing that `reports` is a bucket
        would read the wrong object - or, worse, succeed against a real
        bucket of that name.

        Returns `str`, not `bytes`: this connector's job is to make an
        object's *contents* available to a task, and everything
        downstream (an LLM prompt, a task result, a JSON API response)
        needs text. An object that is not valid UTF-8 raises
        `ConnectorQueryError` rather than being decoded with
        replacement characters - silently corrupting a Parquet file
        into mojibake and handing it over as "the data" is exactly the
        fabricated-success failure this package's error taxonomy
        exists to prevent. Binary objects are out of scope for this
        connector.
        """
        bucket, key = self._resolve_location(query, params.pop("bucket", None))
        if params:
            raise ConnectorQueryError(
                f"connector '{self.name}' read got unexpected arguments",
                connector_name=self.name,
                detail=f"unrecognised: {sorted(params)}; expected 'bucket'",
            )

        async with self._client() as client:
            try:
                response = await client.get_object(Bucket=bucket, Key=key)
            except (BotoCoreError, ClientError, OSError) as exc:
                raise _map_boto_error(exc, self.name, f"get '{key}' from '{bucket}'") from exc

            # Checked from the response metadata *before* the body is
            # streamed, so an oversized object costs one round trip and
            # no memory rather than being downloaded and then rejected.
            content_length = response.get("ContentLength")
            if isinstance(content_length, int) and content_length > _MAX_OBJECT_BYTES:
                raise ConnectorQueryError(
                    f"connector '{self.name}' refuses to read '{key}' from '{bucket}'",
                    connector_name=self.name,
                    detail=(
                        f"object is {content_length} bytes; the per-object bound is "
                        f"{_MAX_OBJECT_BYTES}"
                    ),
                )
            try:
                body = await response["Body"].read()
            except (BotoCoreError, ClientError, OSError) as exc:
                raise _map_boto_error(
                    exc, self.name, f"read body of '{key}' from '{bucket}'"
                ) from exc

        # A provider that omits ContentLength (some S3-compatible
        # implementations do on chunked responses) still gets bounded,
        # just after the transfer rather than before it.
        if len(body) > _MAX_OBJECT_BYTES:
            raise ConnectorQueryError(
                f"connector '{self.name}' refuses to read '{key}' from '{bucket}'",
                connector_name=self.name,
                detail=f"object is {len(body)} bytes; the per-object bound is {_MAX_OBJECT_BYTES}",
            )

        try:
            return body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConnectorQueryError(
                f"connector '{self.name}' could not decode '{key}' from '{bucket}' as UTF-8 text",
                connector_name=self.name,
                detail=f"{exc}; this connector reads text objects only",
            ) from exc

    async def write(self, action: str, **params: Any) -> Any:
        """`action` is `"put_object"` or `"delete_object"`.

        Both take the same addressing as `read()`: `key` (optionally an
        `s3://bucket/key` URI) plus an optional `bucket`. `put_object`
        additionally requires `body` (`str` or `bytes`; a `str` is
        encoded UTF-8) and takes an optional `content_type`.
        """
        if self.read_only:
            raise ConnectorReadOnlyError(
                f"connector '{self.name}' is registered read-only; refusing action '{action}'",
                connector_name=self.name,
            )
        if action not in ("put_object", "delete_object"):
            raise NotImplementedError(
                f"unsupported action '{action}' for connector '{self.name}' "
                "(expected 'put_object' or 'delete_object')"
            )

        key_argument = params.pop("key", None)
        if not isinstance(key_argument, str) or not key_argument:
            raise ConnectorQueryError(
                f"connector '{self.name}' action '{action}' requires a non-empty 'key'",
                connector_name=self.name,
            )
        bucket, key = self._resolve_location(key_argument, params.pop("bucket", None))

        if action == "delete_object":
            _reject_extra(params, self.name, action, "key/bucket")
            async with self._client() as client:
                try:
                    await client.delete_object(Bucket=bucket, Key=key)
                except (BotoCoreError, ClientError, OSError) as exc:
                    raise _map_boto_error(
                        exc, self.name, f"delete '{key}' from '{bucket}'"
                    ) from exc
            return {"bucket": bucket, "key": key, "deleted": True}

        body = params.pop("body", None)
        if isinstance(body, str):
            body = body.encode("utf-8")
        if not isinstance(body, bytes):
            raise ConnectorQueryError(
                f"connector '{self.name}' action 'put_object' requires 'body' as str or bytes",
                connector_name=self.name,
                detail=f"got {type(body).__name__}",
            )
        content_type = params.pop("content_type", None)
        _reject_extra(params, self.name, action, "key/bucket/body/content_type")

        extra = {"ContentType": content_type} if content_type else {}
        async with self._client() as client:
            try:
                await client.put_object(Bucket=bucket, Key=key, Body=body, **extra)
            except (BotoCoreError, ClientError, OSError) as exc:
                raise _map_boto_error(exc, self.name, f"put '{key}' into '{bucket}'") from exc
        return {"bucket": bucket, "key": key, "bytes_written": len(body)}

    async def health(self) -> HealthStatus:
        """`HeadBucket` on the default bucket, or `ListBuckets` without one.

        The distinction matters: a credential scoped to a single bucket
        can head that bucket but usually cannot list the account's
        buckets, so checking the configured bucket is both the cheaper
        and the more accurate signal when one exists.

        Falling back to `ListBuckets`, an `AccessDenied` is reported as
        *healthy* with an explanatory detail. That is a considered
        call: an AccessDenied is proof the endpoint is reachable, the
        request was well-formed and the signature verified - everything
        a health check can establish - and the only thing it says
        beyond that is that this particular credential is scoped, which
        is good practice rather than a fault. An invalid key or a bad
        signature returns a *different* error code and is correctly
        reported unhealthy.
        """
        try:
            async with self._client() as client:
                if self._default_bucket:
                    await client.head_bucket(Bucket=self._default_bucket)
                    return HealthStatus(name=self.name, healthy=True)
                try:
                    await client.list_buckets()
                except ClientError as exc:
                    if _error_code(exc) in _ACCESS_DENIED_CODES:
                        return HealthStatus(
                            name=self.name,
                            healthy=True,
                            detail="reachable; credential may not list buckets",
                        )
                    raise
                return HealthStatus(name=self.name, healthy=True)
        except Exception as exc:  # noqa: BLE001 - a health check must never raise
            return HealthStatus(name=self.name, healthy=False, detail=describe_exception(exc))

    def _resolve_location(self, location: str, bucket: Any) -> tuple[str, str]:
        """`(bucket, key)` from an `s3://` URI, an explicit bucket, or the default."""
        if location.startswith("s3://"):
            remainder = location[len("s3://") :]
            uri_bucket, separator, uri_key = remainder.partition("/")
            if not uri_bucket or not separator or not uri_key:
                raise ConnectorQueryError(
                    f"connector '{self.name}' got a malformed s3:// URI",
                    connector_name=self.name,
                    detail=f"expected s3://<bucket>/<key>, got {location!r}",
                )
            return uri_bucket, uri_key

        if isinstance(bucket, str) and bucket:
            return bucket, location
        if bucket is not None:
            raise ConnectorQueryError(
                f"connector '{self.name}' got a non-string 'bucket'",
                connector_name=self.name,
                detail=f"got {type(bucket).__name__}",
            )
        if self._default_bucket:
            return self._default_bucket, location

        raise ConnectorConfigurationError(
            f"connector '{self.name}' has no bucket for key '{location}'",
            connector_name=self.name,
            detail=(
                "pass an s3://bucket/key URI, a bucket= keyword, or set PRAXIS_S3_CONNECTOR_BUCKET"
            ),
        )


def _error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


def _map_boto_error(exc: Exception, connector_name: str, operation: str) -> Exception:
    """botocore's exception surface -> this package's three-way taxonomy.

    `ClientError` means the service answered with an error document, so
    its `Error.Code` (`NoSuchKey`, `NoSuchBucket`, `AccessDenied`,
    `SignatureDoesNotMatch`) is carried through as the `status` -
    those are the codes a caller actually branches on, and they are far
    more specific than the HTTP status beside them.

    `EndpointConnectionError` and a bare `OSError` (DNS failure, reset
    connection - aiohttp raises these straight through) are transport
    failures. `NoCredentialsError` is the one `BotoCoreError` that is a
    configuration fault rather than a transport one, so it is split
    out; everything else in that tree (`ConnectTimeoutError`,
    `ReadTimeoutError`, ...) is treated as unavailability, which is
    what it is.
    """
    if isinstance(exc, ClientError):
        return ConnectorQueryError(
            f"connector '{connector_name}' could not {operation}",
            connector_name=connector_name,
            detail=describe_exception(exc),
            status=_error_code(exc) or None,
        )
    if isinstance(exc, NoCredentialsError):
        return ConnectorConfigurationError(
            f"connector '{connector_name}' has no usable credentials for {operation}",
            connector_name=connector_name,
            detail=describe_exception(exc),
        )
    if isinstance(exc, (EndpointConnectionError, BotoCoreError, OSError)):
        return ConnectorUnavailableError(
            f"connector '{connector_name}' could not reach the S3 endpoint to {operation}",
            connector_name=connector_name,
            detail=describe_exception(exc),
        )
    return ConnectorQueryError(
        f"connector '{connector_name}' failed to {operation}",
        connector_name=connector_name,
        detail=f"{type(exc).__name__}: {exc}",
    )


def _reject_extra(params: dict[str, Any], connector_name: str, action: str, expected: str) -> None:
    if params:
        raise ConnectorQueryError(
            f"connector '{connector_name}' action '{action}' got unexpected arguments",
            connector_name=connector_name,
            detail=f"unrecognised: {sorted(params)}; expected {expected}",
        )


register_connector_factory(
    ConnectorFactory(
        name="s3",
        is_configured=lambda settings: bool(
            settings.s3_connector_access_key_id and settings.s3_connector_secret_access_key
        ),
        build=lambda settings: S3Connector(
            access_key_id=required(
                settings.s3_connector_access_key_id, setting="s3_connector_access_key_id"
            ),
            secret_access_key=required(
                settings.s3_connector_secret_access_key,
                setting="s3_connector_secret_access_key",
            ),
            endpoint_url=settings.s3_connector_endpoint_url,
            region=settings.s3_connector_region,
            default_bucket=settings.s3_connector_bucket,
        ),
    )
)
