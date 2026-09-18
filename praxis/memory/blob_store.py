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

from pathlib import Path

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
        # right.
        candidate = Path(key)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"invalid blob key (must be a relative path with no '..'): {key!r}")

        resolved = (self._root / candidate).resolve()
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
