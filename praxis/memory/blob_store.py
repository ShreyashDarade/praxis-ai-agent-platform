# praxis/memory/blob_store.py
"""Local-filesystem `BlobStore` (spec §2: "Local filesystem (scratch dir)").

`put(key, data)` writes under `root/key` and returns `key` itself
(unchanged) - not an absolute path. This is the contract callers must
rely on: `Attachment`/`VectorChunk` provenance data stores the *key*,
and a later `get(key)` (here, or against a swapped-in `BlobStore`
backend per spec §2's fsspec-based swap path - S3, GCS, ...) is handed
that same key back, never a local filesystem path that wouldn't mean
anything against a different backend.
"""
from __future__ import annotations

from pathlib import Path, PureWindowsPath

from praxis.core.interfaces import BlobStore

# Phase 12: every artifact a skill produces is written under
# `t/<tenant_id>/...` so tenant ownership is a property of the key
# itself, checkable without a database lookup on the read path
# (`GET /artifacts/{key}`). Attachment blobs keep their historical flat
# `<attachment_id>` key: those are already looked up through the
# `attachments` table, which carries a real `tenant_id` column, so the
# ownership check there reads the row rather than parsing the key.
_TENANT_KEY_PREFIX = "t"


def tenant_artifact_key(tenant_id: str, relative_key: str) -> str:
    """Namespaces `relative_key` under `tenant_id`.

    e.g. `tenant_artifact_key("abc", "charts/x.png") -> "t/abc/charts/x.png"`.
    """
    return f"{_TENANT_KEY_PREFIX}/{tenant_id}/{relative_key.lstrip('/')}"


def is_key_in_tenant(key: str, tenant_id: str) -> bool:
    """True iff `key` lives inside `tenant_id`'s artifact namespace.

    A key with no tenant prefix at all is treated as **not** belonging
    to any tenant and is therefore refused - failing closed. Legacy
    flat keys predate tenant namespacing and are reachable only through
    a row lookup that carries its own `tenant_id`, never through this
    path.
    """
    return key.startswith(f"{_TENANT_KEY_PREFIX}/{tenant_id}/")


def validate_blob_key(key: str) -> None:
    """Rejects any key no `BlobStore` backend may accept; raises
    `ValueError` with the reason.

    Shared by every backend rather than reimplemented per backend, and
    for a specific reason: a key must be accepted or refused
    *identically* whichever store is configured, or swapping
    `LocalBlobStore` for `S3BlobStore` silently changes which uploads
    succeed. Keys come from user-supplied filenames (spec §5 step 1), so
    this is a security boundary on the local backend - `..` escapes the
    scratch root - and a namespacing boundary on the remote ones, where
    an S3 key is literal but a prefix is not a container: `t/a/../b`
    stored verbatim escapes `t/a/`'s notional namespace the moment
    anything normalizes it, including several S3-compatible gateways.

    Checks both POSIX and Windows separators. The local backend runs on
    whatever the deployment's OS is, so treating `a\\..\\b` as one
    opaque segment (which a pure-POSIX split does) would leave a real
    traversal open on Windows.
    """
    if not key:
        raise ValueError("invalid blob key (must not be empty)")

    normalized = key.replace("\\", "/")
    if normalized.startswith("/") or PureWindowsPath(key).is_absolute():
        raise ValueError(f"invalid blob key (must be a relative path): {key!r}")
    if ".." in normalized.split("/"):
        raise ValueError(f"invalid blob key (must not contain '..'): {key!r}")


class LocalBlobStore(BlobStore):
    """Stores blobs as files under `root`, keyed by a caller-supplied relative key."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        # Path-traversal guard: `key` ultimately comes from a
        # user-supplied filename in a real deployment (spec §5 step 1's
        # upload path), so it must never be allowed to escape `root` via
        # ".." segments or be treated as an absolute path in its own
        # right. The key-shape half of that check is shared with every
        # other backend (`validate_blob_key`); the resolved-path half
        # below is specific to a filesystem and catches what a symlink
        # under `root` could still reach.
        validate_blob_key(key)

        resolved = (self._root / Path(key)).resolve()
        if resolved != self._root and self._root not in resolved.parents:
            raise ValueError(f"invalid blob key (escapes blob store root): {key!r}")
        return resolved

    async def put(self, key: str, data: bytes) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    async def get(self, key: str) -> bytes:
        path = self._resolve(key)
        return path.read_bytes()
