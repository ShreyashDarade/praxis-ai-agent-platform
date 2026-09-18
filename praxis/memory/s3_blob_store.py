# praxis/memory/s3_blob_store.py
"""`S3BlobStore`: the S3-compatible `BlobStore` (spec §2's "object/file
storage", and the swap path `LocalBlobStore`'s docstring anticipates).

`LocalBlobStore` is correct for one machine with one disk. Artifacts
outlive the process that made them and have to be readable from
whichever worker serves `GET /artifacts/{key}` next, so a real
deployment needs shared object storage. Built on `aioboto3`, so it
talks to S3, MinIO, Cloudflare R2, Ceph RGW - anything with an S3 API -
by pointing `endpoint_url` at it.

**The contract that must not move when a deployment swaps backends.**
Three things are load-bearing and are implemented to match
`LocalBlobStore` exactly, not approximately:

1. `put()` returns the *key* it was given, never a URL, a path or a
   bucket-qualified name. `Attachment`/`VectorChunk` provenance stores
   that key, and a later `get()` - possibly against a different backend
   entirely - is handed it back unchanged.
2. `get()` on a missing key raises `FileNotFoundError`. Botocore's own
   failure is a `ClientError` carrying an error code in a response
   dict, which no existing caller catches; letting that escape would
   mean swapping the backend changed the exception type every caller
   handles. It is mapped here, at the boundary.
3. An invalid key raises `ValueError`, from the same
   `praxis.memory.blob_store.validate_blob_key` the local backend uses,
   so a key refused by one store is refused by both.

**Tenant namespacing is unchanged and still the caller's job.** Keys
are written verbatim under `prefix`, so `tenant_artifact_key()` still
produces `t/<tenant_id>/...` and `is_key_in_tenant()` still answers
correctly for a key read back out of this store. `prefix` exists for
sharing one bucket between deployments, and is *not* a tenant boundary:
tenant ownership is a property of the key, checkable without a database
lookup, which is exactly what `is_key_in_tenant` relies on.

**Honest limitation.** Everything here is verified against key
construction, tenant namespacing and error mapping - the last of those
end to end through a real `aioboto3` client against a real HTTP
endpoint (see `tests/memory/test_s3_blob_store.py`). Behavior that
needs a live S3 account - IAM, SSE-KMS, versioning, multipart
thresholds for very large objects - is not exercised here and is not
claimed to be.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

import aioboto3
import structlog
from botocore.exceptions import ClientError

from praxis.core.interfaces import BlobStore
from praxis.memory.blob_store import validate_blob_key

_logger = structlog.get_logger(__name__)

# What S3 and its clones report for "that key is not there". `NoSuchKey`
# comes from GetObject, `NoSuchBucket` is included deliberately (a
# caller asked for a blob and there is nothing to read - the distinction
# between a missing key and a missing bucket is an operator's problem,
# logged below, not a different outcome for the read path), and the bare
# `404`/`NotFound` codes are what HeadObject and several S3-compatible
# gateways return instead of a named code.
_MISSING_ERROR_CODES = frozenset({"NoSuchKey", "NoSuchBucket", "404", "NotFound"})


def _is_missing(exc: ClientError) -> bool:
    """Whether a botocore `ClientError` means "no such object".

    Reads the error code from the response dict rather than catching
    `client.exceptions.NoSuchKey`, because that exception class only
    exists on a constructed client instance and differs between
    S3-compatible implementations - the wire-level error code does not.
    Also inspects the raw HTTP status, since a gateway returning 404
    with an empty or nonstandard body would otherwise fall through to a
    re-raise.
    """
    error = exc.response.get("Error", {}) if isinstance(exc.response, dict) else {}
    if str(error.get("Code", "")) in _MISSING_ERROR_CODES:
        return True
    metadata = exc.response.get("ResponseMetadata", {}) if isinstance(exc.response, dict) else {}
    return int(metadata.get("HTTPStatusCode", 0) or 0) == 404


class S3BlobStore(BlobStore):
    """Stores blobs as objects in an S3-compatible bucket.

    The client is built lazily on first use and then kept, rather than
    created per call: each `aioboto3` client owns an `aiohttp` session
    and a fresh TCP pool, so a per-call client would pay a full
    connection setup for every artifact read. The cost of keeping it is
    that `close()` must be called on shutdown to release the session;
    not calling it leaks it for the life of the process, which is
    survivable but is a leak, so it is stated rather than hidden.

    Credentials are deliberately not required here. Left unset,
    botocore's standard chain applies - environment variables, shared
    config, instance/task role - which is how a deployment avoids
    putting long-lived keys in application config at all. They are
    accepted for MinIO and for tests, where there is no such chain.
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        endpoint_url: str | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        botocore_config: Any | None = None,
        session: aioboto3.Session | None = None,
    ) -> None:
        if not bucket:
            raise ValueError("S3BlobStore requires a bucket name")
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._client_kwargs: dict[str, Any] = {}
        if endpoint_url is not None:
            self._client_kwargs["endpoint_url"] = endpoint_url
        # `botocore.config.Config` passthrough, not a test affordance:
        # MinIO and several other S3-compatible services only serve
        # path-style addressing (`Config(s3={"addressing_style":
        # "path"})`), and a deployment that wants different retry or
        # timeout behavior has nowhere else to say so.
        if botocore_config is not None:
            self._client_kwargs["config"] = botocore_config
        self._session = session if session is not None else aioboto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
        self._exit_stack = AsyncExitStack()
        self._client: Any | None = None
        # Two concurrent first calls would otherwise each build a
        # client and one would be dropped un-closed.
        self._client_lock = asyncio.Lock()

    def object_key(self, key: str) -> str:
        """The bucket-relative object name for a caller-supplied key.

        Public because it is the one place the mapping from Praxis key
        to S3 object name is defined, and an operator inspecting a
        bucket needs to be able to compute it. Validation runs first, so
        an invalid key fails identically here and on the local backend.
        """
        validate_blob_key(key)
        return f"{self._prefix}/{key}" if self._prefix else key

    async def _get_client(self) -> Any:
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = await self._exit_stack.enter_async_context(
                        self._session.client("s3", **self._client_kwargs)
                    )
        return self._client

    async def put(self, key: str, data: bytes) -> str:
        """Writes `data` at `key` and returns `key` unchanged."""
        object_key = self.object_key(key)
        client = await self._get_client()
        await client.put_object(Bucket=self._bucket, Key=object_key, Body=data)
        return key

    async def get(self, key: str) -> bytes:
        """Reads the blob at `key`.

        Raises `FileNotFoundError` when there is no such object - the
        same exception `LocalBlobStore.get` raises via `read_bytes()`,
        so a caller's error handling is unaffected by which backend is
        configured. Every other `ClientError` (denied, throttled,
        unreachable) propagates unchanged: those are not "absent", and
        reporting them as absent would turn a credentials mistake into
        a silent data-loss report.
        """
        object_key = self.object_key(key)
        client = await self._get_client()
        try:
            response = await client.get_object(Bucket=self._bucket, Key=object_key)
        except ClientError as exc:
            if _is_missing(exc):
                _logger.info(
                    "s3_blob_missing",
                    bucket=self._bucket,
                    object_key=object_key,
                    code=str(exc.response.get("Error", {}).get("Code", "")),
                )
                raise FileNotFoundError(key) from exc
            raise
        return await response["Body"].read()

    async def close(self) -> None:
        """Releases the underlying client and its connection pool."""
        await self._exit_stack.aclose()
        self._exit_stack = AsyncExitStack()
        self._client = None


def s3_blob_store_from_settings(settings: Any) -> S3BlobStore | None:
    """Builds an `S3BlobStore` from `Settings`, or `None` when no bucket
    is configured.

    `None` rather than an error, matching every other optional backend
    in `praxis.config`: a deployment that has not configured object
    storage uses `LocalBlobStore`, which is a supported configuration
    (spec §2 names it as the MVP backend), not a misconfiguration.
    """
    bucket = getattr(settings, "s3_bucket", None)
    if not bucket:
        return None
    return S3BlobStore(
        bucket,
        prefix=getattr(settings, "s3_key_prefix", "") or "",
        endpoint_url=getattr(settings, "s3_endpoint_url", None),
        region_name=getattr(settings, "s3_region", None),
    )
