# praxis/ingestion/embedders/sentence_transformer_embedder.py
"""`sentence-transformers`-backed `Embedder` (spec §2, §5 step 5).

The MVP-deployed embedding backend per spec §2: "`sentence-transformers`
(open-source, runs locally - no vendor API dependency for the MVP
default)", model `all-MiniLM-L6-v2` (384-dim, matching `vector_chunks`'
`Vector(384)` column - spec's environment note).
"""
from __future__ import annotations

import asyncio

from sentence_transformers import SentenceTransformer

from praxis.core.interfaces import Embedder

_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SentenceTransformerEmbedder(Embedder):
    """Wraps a local `SentenceTransformer` model; loaded once, reused across calls."""

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        # Loading the model is slow (first use downloads weights from
        # the HF hub) - load it once here rather than per-call. Callers
        # that want lazy/deferred loading can construct this class lazily
        # themselves; the class itself always loads eagerly at
        # construction so `.embed()` never pays a surprise first-call
        # latency mid-request.
        self._model = SentenceTransformer(model_name)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # .encode() is sync/CPU-bound - run it off the event loop thread
        # so embedding a batch doesn't block other async work.
        vectors = await asyncio.to_thread(self._model.encode, texts)
        return [vector.tolist() for vector in vectors]
