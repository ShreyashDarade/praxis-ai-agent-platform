# praxis/ingestion/embedders/sentence_transformer_embedder.py
"""`sentence-transformers`-backed `Embedder` (spec §2, §5 step 5).

The MVP-deployed embedding backend per spec §2: "`sentence-transformers`
(open-source, runs locally - no vendor API dependency for the MVP
default)", model `all-MiniLM-L6-v2` (384-dim, matching `vector_chunks`'
`Vector(384)` column - spec's environment note).

**Embedding cache (Phase 7, spec §11)**: `embed()` is fronted by a
`Cache` keyed on a content hash (`model_name` + the exact input text) -
re-embedding an already-seen exact string within the cache's TTL skips
the real (CPU-bound, `asyncio.to_thread`-dispatched) `model.encode()`
call entirely for that text. A batch with a mix of cached and uncached
texts only ever runs `model.encode()` on the uncached subset.
`real_encode_calls` counts actual `model.encode()` invocations (not
individual texts) - real, load-bearing testability, matching
`praxis.llm.catalogue.LLMCatalogue.real_api_calls`' same posture: a test
asserts on this counter, never on wall-clock timing.
"""
from __future__ import annotations

import asyncio
import hashlib

from sentence_transformers import SentenceTransformer

from praxis.cache.memory_cache import InMemoryCache
from praxis.core.interfaces import Cache, Embedder

_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# No expiry-by-default: a given (model, exact text) pair always embeds
# to the same vector for a fixed model - the only reason to ever expire
# this cache is memory growth over a very long-running process, which
# an explicit TTL can still be set for via the constructor.
DEFAULT_CACHE_TTL_SECONDS: int | None = None


def _content_key(model_name: str, text: str) -> str:
    payload = f"{model_name}\x00{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SentenceTransformerEmbedder(Embedder):
    """Wraps a local `SentenceTransformer` model; loaded once, reused across calls."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        *,
        cache: Cache | None = None,
        cache_ttl_seconds: int | None = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self._model_name = model_name
        # Loading the model is slow (first use downloads weights from
        # the HF hub) - load it once here rather than per-call. Callers
        # that want lazy/deferred loading can construct this class lazily
        # themselves; the class itself always loads eagerly at
        # construction so `.embed()` never pays a surprise first-call
        # latency mid-request.
        self._model = SentenceTransformer(model_name)
        self._cache: Cache = cache if cache is not None else InMemoryCache()
        self._cache_ttl_seconds = cache_ttl_seconds
        # Real, load-bearing testability (see module docstring) - counts
        # actual `model.encode()` calls, never cache hits.
        self.real_encode_calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        cache_keys = [_content_key(self._model_name, text) for text in texts]
        results: list[list[float] | None] = [await self._cache.get(key) for key in cache_keys]

        missing_indices = [index for index, value in enumerate(results) if value is None]
        if missing_indices:
            missing_texts = [texts[index] for index in missing_indices]
            # .encode() is sync/CPU-bound - run it off the event loop
            # thread so embedding a batch doesn't block other async work.
            vectors = await asyncio.to_thread(self._model.encode, missing_texts)
            self.real_encode_calls += 1
            for position, index in enumerate(missing_indices):
                vector = vectors[position].tolist()
                results[index] = vector
                await self._cache.set(
                    cache_keys[index], vector, ttl_seconds=self._cache_ttl_seconds
                )

        return results  # type: ignore[return-value]  # every entry is filled by this point


_default_instance: SentenceTransformerEmbedder | None = None


def get_default_embedder() -> SentenceTransformerEmbedder:
    """A process-wide, lazily-constructed `SentenceTransformerEmbedder` singleton.

    Every caller that just wants "the" default embedder - `praxis.api.main`'s
    `/attachments` handler and `praxis.agents.skills.retrieve_documents`'s
    skill alike (Phase 5) - should call this rather than each
    constructing its own `SentenceTransformerEmbedder()`: loading the
    model is expensive, and two independent module-level instances would
    silently load the same weights twice for no benefit. Callers that
    genuinely need an isolated instance (e.g. a test forcing a specific
    `model_name`) still construct `SentenceTransformerEmbedder(...)` directly.
    """
    global _default_instance
    if _default_instance is None:
        _default_instance = SentenceTransformerEmbedder()
    return _default_instance

