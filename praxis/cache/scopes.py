# praxis/cache/scopes.py
"""The named cache scopes, their TTL rationale, and their invalidation rules.

`praxis.cache.memory_cache.InMemoryCache`'s docstring names the four
scopes spec §11 called for (LLM response, embedding, connector/schema
introspection, tool result). The product brief (§10, "Caching, storage,
and performance") asks for three more - semantic/query, retrieval, and
dashboard-result - so this module is the single place every scope name,
its default TTL, and the rule that invalidates it are written down,
instead of a bare string literal repeated at each call site.

Two things these constants are deliberately *not*:

- They are not a cache registry. Each scope still owns its own `Cache`
  instance (the reason is on `InMemoryCache`: a policy tuned for the
  task-scoped tool-result cache must never leak into the long-lived
  schema cache). These are labels and policy, not plumbing.
- They are not enforced at runtime by the `Cache` ABC. `Cache.get`/`set`
  take an opaque key; nothing stops a caller passing an unlabelled one.
  What is enforced is `require_exact_match()` below, which any future
  semantic-similarity cache must call before matching a query against a
  stored entry by embedding distance.

**Why "semantic caching" gets its own gate.** An exact-match cache can
only ever return the answer to precisely the question that was asked.
A *semantic* cache returns the answer to a question that was merely
*similar*, which is a completely different risk: "revenue last month"
and "revenue last quarter" are close in embedding space and are not the
same number. The brief states the rule directly - *"Do not apply
semantic caching to exact financial or operational metrics without
strict validity rules"* - so every scope below declares whether
similarity matching is permitted at all, and the default for an
unrecognized scope is "no".
"""
from __future__ import annotations

from typing import Final

LLM_RESPONSE: Final = "llm_response"
"""Exact-match cache over one completion (`praxis.llm.catalogue`).

TTL 1h. A model's provider-side weights and safety behavior can change
under a stable model id, so an unbounded entry would let a long-running
process serve the same answer for its whole lifetime; an hour is short
enough that drift is re-discovered within one deployment window and long
enough to absorb the several identical calls one task's retries make.

Invalidation is by key, not by sweep: the key already carries the model
id, the prompt text, `max_tokens` and every request kwarg, so changing
any of them is a different entry rather than a stale one. Never
similarity-matched - that is `SEMANTIC_QUERY`'s job, under its rules.
"""

EMBEDDING: Final = "embedding"
"""Exact-match cache over `(embedding model, text) -> vector`.

TTL 24h. Unlike every other scope here, this one has no freshness
argument at all: an embedding model is deterministic, so a cached vector
is never *wrong*, only occupying memory. The TTL is therefore a memory
bound, honestly - `InMemoryCache` has no LRU eviction, so without one a
long-lived ingestion process would accumulate every vector it ever
computed.

Invalidation is a key change: the model id (and therefore the
dimension) is part of the key, so switching embedding models cannot
serve vectors from the old model. A model/dimension change is an index
migration, not a cache flush.
"""

CONNECTOR_SCHEMA: Final = "connector_schema"
"""Exact-match cache over a connector's introspected schema
(`praxis.agents.capability_factory`).

TTL 5 minutes - spec §11's "short TTL". Long enough to skip
re-introspection across the several syntheses one task triggers
back-to-back, short enough that a column actually added to a live table
is picked up well within a deployment's lifetime. The durable record
(the GraphStore `describes` edge) is unbounded; this is only the hot
path in front of it.

The key includes a digest of the connector's DSN, not just its name:
the dashboard demo re-registers "customer-db" against a different
SQLite file every run, and a name-only key silently served one run's
schema to the next (see `CapabilityFactory._connector_identity`).
Never similarity-matched: a schema that is *nearly* right generates SQL
that fails, or worse, silently reads the wrong column.
"""

TOOL_RESULT: Final = "tool_result"
"""Exact-match cache over one read-only skill invocation
(`praxis.core.orchestrator`).

No TTL, because the cache *instance* is the TTL: a fresh cache is
created per task and discarded with it, so an entry cannot outlive the
task that produced it. A time-based expiry inside that window would
only introduce a way for two identical steps of one plan to disagree.

**Only read-only skills.** The brief: *"Never cache mutating tool
execution as though it were a reusable answer; use idempotency records
instead."* Caching a mutation means the second call returns the first
call's receipt without performing the second effect - a silently
dropped write. Never similarity-matched, for the same reason and more
so: arguments that look alike are not the same arguments.
"""

SEMANTIC_QUERY: Final = "semantic_query"
"""Similarity-matched cache over natural-language questions and the
narrative answers to them.

TTL 15 minutes. This is the one scope where a *near* match may be
served, so its TTL is deliberately the shortest of the long-lived
scopes: the blast radius of a wrong hit here is an answer to a question
nobody asked, and a short window bounds how long one can persist.

Two conditions, both required, before a stored entry may be served for
a merely-similar question:

1. The candidate's similarity must clear an explicit threshold chosen
   for the deployment - not "nearest neighbour wins", which always
   returns something.
2. The question must not resolve to an exact figure. Explanations,
   definitions, and summaries may be similarity-matched; a number a
   person will act on may not. Route those to `DASHBOARD_RESULT` or
   `TOOL_RESULT`, which are exact-match only.

Invalidated by source version: the key carries the semantic-layer and
source-data versions the answer was computed against, so redefining a
metric or reloading a source produces new keys rather than stale hits.
"""

RETRIEVAL: Final = "retrieval"
"""Similarity-matched cache over a retrieval request and the chunk ids
it returned.

TTL 5 minutes, which is short for a reason that is not staleness of the
*text*: it is staleness of *permission*. The brief requires that
"deletion and permission revocation must propagate to chunks, vectors,
graph facts, caches, and derived artifacts", and a revocation that only
takes effect when a cache entry expires is a revocation that did not
happen. A short TTL bounds that window; it does not replace the
requirement to re-authorize.

**Store ids, never chunk text.** A cached retrieval result is a list of
chunk identifiers that must be re-read and re-authorized on every hit,
so a document deleted or unshared since the entry was written drops out
of the result even while the entry is still live. Caching the text
itself would make the cache a second, unpoliced copy of the corpus.

Similarity matching is permitted here because the retrieval step is
itself approximate and its output is re-ranked and re-authorized
downstream - the cache is a shortcut to a candidate set, not to an
answer.
"""

DASHBOARD_RESULT: Final = "dashboard_result"
"""Exact-match cache over one dashboard panel's computed result set.

TTL 60 seconds as a *ceiling only*. The real bound is the panel's own
declared freshness - `praxis.semantic.model.Metric.freshness_sla_seconds`
- and a caller must use the smaller of the two. A metric declaring a
30-second SLA that is served from a 60-second-old cache entry has had
its SLA broken by the cache, which is exactly the failure the SLA
exists to make visible.

This scope is the brief's named example of what must never be
similarity-matched: these are exact financial and operational figures.
A dashboard that answers "revenue this quarter" with last quarter's
cached number because the two questions embedded closely is not a
performance optimization, it is a wrong number presented with the same
confidence as a right one.

Invalidated explicitly, not only by TTL: the key carries the semantic
model version, the metric and dimension names, the filter set, and the
source-data version, so a redefined metric or a reloaded source is a
different key. A panel whose underlying source announces a new version
should have its tenant prefix swept (`CacheKey.tenant_prefix`).
"""

ALL_SCOPES: Final[tuple[str, ...]] = (
    LLM_RESPONSE,
    EMBEDDING,
    CONNECTOR_SCHEMA,
    TOOL_RESULT,
    SEMANTIC_QUERY,
    RETRIEVAL,
    DASHBOARD_RESULT,
)

DEFAULT_TTL_SECONDS: Final[dict[str, int | None]] = {
    LLM_RESPONSE: 3600,
    EMBEDDING: 86_400,
    CONNECTOR_SCHEMA: 300,
    TOOL_RESULT: None,
    SEMANTIC_QUERY: 900,
    RETRIEVAL: 300,
    DASHBOARD_RESULT: 60,
}

SEMANTIC_MATCH_SCOPES: Final[frozenset[str]] = frozenset({SEMANTIC_QUERY, RETRIEVAL})
"""The only scopes a similarity-matched lookup may ever serve from.

Everything else - LLM responses, embeddings, connector schemas, tool
results and dashboard results - is exact-match only. `EMBEDDING` is in
the exact-match set for a different reason than the rest: matching an
embedding request by the similarity of its *input text* would return
the vector for a different string, which is not an approximation of the
answer, it is a different answer.
"""

EXACT_MATCH_ONLY_SCOPES: Final[frozenset[str]] = frozenset(ALL_SCOPES) - SEMANTIC_MATCH_SCOPES


class SemanticCachingNotPermittedError(ValueError):
    """Raised when a similarity-matched lookup is attempted against a
    scope that must only ever be served on an exact key match.

    A `ValueError` subclass so it reads as the programming error it is -
    the scope a lookup runs against is chosen in code, never by a user
    or a model, so reaching this is a bug to fix, not a condition to
    degrade from.
    """


def ttl_for(scope: str) -> int | None:
    """The default TTL in seconds for `scope`, or `None` for the scopes
    that deliberately have no expiry.

    Raises `KeyError` for an unknown scope rather than inventing a
    default: silently applying someone else's TTL to a new scope is how
    a cache ends up serving data far past its freshness rule.
    """
    return DEFAULT_TTL_SECONDS[scope]


def semantic_matching_allowed(scope: str) -> bool:
    """Whether `scope` may serve an entry stored for a merely *similar*
    request.

    An unrecognized scope returns `False` - failing closed, so adding a
    new scope without deciding this question yields the safe answer
    rather than the convenient one.
    """
    return scope in SEMANTIC_MATCH_SCOPES


def require_exact_match(scope: str) -> None:
    """Guard a similarity-matched lookup; raises for a scope that must
    be exact-match only.

    Call this at the top of any semantic/embedding-distance cache
    lookup. It is the enforcement point for the brief's *"Do not apply
    semantic caching to exact financial or operational metrics"*, which
    is otherwise a rule written in a document and nowhere in the code.
    """
    if not semantic_matching_allowed(scope):
        raise SemanticCachingNotPermittedError(
            f"cache scope '{scope}' is exact-match only and must never be served from a "
            f"similarity match; similarity matching is permitted only for "
            f"{sorted(SEMANTIC_MATCH_SCOPES)}"
        )
