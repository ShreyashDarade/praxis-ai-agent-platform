# praxis/memory/memory_types.py
"""The typed memory subsystem (Prompt §5).

Prompt §5 enumerates memory kinds Praxis had no representation for:
*"user preference memory, project/workspace memory, agent-specific
memory, shared team memory, episodic memory for previous task
outcomes"*, plus *"memory expiration, correction, provenance, and
access control"* and *"memory isolation between tenants, users,
agents, and workspaces"*.

What already existed covered only two of these: semantic memory
(pgvector chunks from ingested documents) and a lineage graph. There
was nowhere to record "this user prefers revenue in thousands", "the
last three times we investigated this alert it was the cache", or
"this workspace's fiscal year starts in April".

**Scope is the isolation mechanism, and it is not advisory.** Every
entry carries a `MemoryScope` naming the tenant plus, depending on
kind, a user / workspace / agent. `MemoryStore.recall` builds its
WHERE clause from the scope, so one user's preferences are not
reachable from another's session even within a tenant - the same
push-the-predicate-into-SQL discipline the vector store already uses.

**Provenance and correction are first-class**, because the prompt is
explicit that unverified agent conclusions must not become trusted
facts. Every entry records where it came from (`source`) and how much
to trust it (`confidence`), and superseding an entry *supersedes*
rather than deletes, so a correction leaves an auditable trail of
what was previously believed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import structlog
from sqlalchemy import and_, or_, select

from praxis.memory.db import PostgresStore
from praxis.memory.models import DEFAULT_TENANT_ID, MemoryEntry

_logger = structlog.get_logger(__name__)


class MemoryKind(str, Enum):
    """The kinds the prompt enumerates.

    `SEMANTIC` is present for completeness but is **not** stored here:
    semantic knowledge lives in the vector store, which is built for
    similarity search. Listing it keeps the taxonomy honest rather
    than implying this module owns every kind.
    """

    PREFERENCE = "preference"
    WORKSPACE = "workspace"
    AGENT = "agent"
    TEAM = "team"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    SEMANTIC = "semantic"


class MemorySource(str, Enum):
    """Where a memory came from - the basis for how far to trust it."""

    USER_STATED = "user_stated"
    OPERATOR_CONFIGURED = "operator_configured"
    OBSERVED = "observed"
    AGENT_INFERRED = "agent_inferred"

    @property
    def is_verified(self) -> bool:
        """Whether this came from a human or from the system's own
        observation, rather than from a model's inference.

        The distinction the prompt insists on: *"do not store
        unverified agent conclusions as trusted facts"*. An inferred
        memory is still worth keeping - it just must not be presented
        with the same authority as something a user said.
        """
        return self in (
            MemorySource.USER_STATED,
            MemorySource.OPERATOR_CONFIGURED,
            MemorySource.OBSERVED,
        )


@dataclass(frozen=True)
class MemoryScope:
    """Who a memory belongs to. The isolation key.

    Deliberately explicit rather than inferred from context: a memory
    written with the wrong scope is a cross-user data leak, and
    "whatever scope the ambient context happened to have" is exactly
    how that happens.
    """

    tenant_id: str = DEFAULT_TENANT_ID
    user_id: str | None = None
    workspace_id: str | None = None
    agent_name: str | None = None

    def matches_kind(self, kind: MemoryKind) -> bool:
        """Whether this scope carries the identifier its kind requires.

        A preference with no user, or an agent memory with no agent,
        would be unaddressable - readable by everyone in the tenant,
        which is precisely the isolation failure this guards against.
        """
        if kind is MemoryKind.PREFERENCE:
            return self.user_id is not None
        if kind is MemoryKind.WORKSPACE:
            return self.workspace_id is not None
        if kind is MemoryKind.AGENT:
            return self.agent_name is not None
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "workspace_id": self.workspace_id,
            "agent_name": self.agent_name,
        }


class MemoryScopeError(Exception):
    """A memory was written with a scope its kind cannot address."""


@dataclass
class Memory:
    """One recalled memory, in application terms."""

    key: str
    value: Any
    kind: MemoryKind
    scope: MemoryScope
    source: MemorySource = MemorySource.AGENT_INFERRED
    confidence: float = 1.0
    entry_id: str = ""
    created_at: datetime | None = None
    expires_at: datetime | None = None
    superseded_by: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def is_trusted(self) -> bool:
        """High-confidence and from a verified source.

        Used to decide whether a memory may be stated as fact or must
        be hedged - the practical expression of the prompt's rule
        about unverified conclusions.
        """
        return self.source.is_verified and self.confidence >= 0.8

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "key": self.key,
            "value": self.value,
            "kind": self.kind.value,
            "scope": self.scope.to_dict(),
            "source": self.source.value,
            "confidence": self.confidence,
            "trusted": self.is_trusted,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "superseded_by": self.superseded_by,
            "detail": self.detail,
        }


class MemoryStore:
    """Typed, scoped, expiring memory with provenance and correction."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    async def remember(
        self,
        *,
        key: str,
        value: Any,
        kind: MemoryKind,
        scope: MemoryScope,
        source: MemorySource = MemorySource.AGENT_INFERRED,
        confidence: float = 1.0,
        ttl_seconds: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> Memory:
        """Stores a memory, superseding any live entry with the same key.

        Superseding rather than overwriting is what makes correction
        auditable: the previous belief is still on record, marked as
        replaced by this one, so "what did we think last week and why
        did that change" is answerable.
        """
        if not scope.matches_kind(kind):
            raise MemoryScopeError(
                f"a '{kind.value}' memory requires the matching scope identifier; got "
                f"{scope.to_dict()}"
            )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")

        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=ttl_seconds) if ttl_seconds else None

        async with self._store.session() as session:
            previous = (
                await session.execute(
                    self._scoped_query(key=key, kind=kind, scope=scope, include_expired=True)
                )
            ).scalars().first()

            entry = MemoryEntry(
                tenant_id=scope.tenant_id,
                user_id=scope.user_id,
                workspace_id=scope.workspace_id,
                agent_name=scope.agent_name,
                kind=kind.value,
                key=key,
                value={"value": value},
                source=source.value,
                confidence=confidence,
                expires_at=expires_at,
                detail=dict(detail or {}),
            )
            session.add(entry)
            await session.flush()

            if previous is not None and previous.superseded_by is None:
                previous.superseded_by = entry.id

            await session.commit()
            entry_id = entry.id
            created_at = entry.created_at

        _logger.info(
            "memory_stored",
            key=key,
            kind=kind.value,
            source=source.value,
            superseded=previous.id if previous is not None else None,
        )
        return Memory(
            key=key,
            value=value,
            kind=kind,
            scope=scope,
            source=source,
            confidence=confidence,
            entry_id=entry_id,
            created_at=created_at,
            expires_at=expires_at,
            detail=dict(detail or {}),
        )

    def _scoped_query(
        self,
        *,
        kind: MemoryKind | None,
        scope: MemoryScope,
        key: str | None = None,
        include_expired: bool = False,
        include_superseded: bool = False,
    ):
        """Builds the scoped SELECT.

        Every scope identifier becomes a SQL predicate rather than a
        post-filter, for the same reason the vector store pushes its
        tenant filter down: post-filtering leaks existence through
        result counts and lets another scope's rows consume any limit.
        """
        conditions = [MemoryEntry.tenant_id == scope.tenant_id]

        if scope.user_id is not None:
            # Tenant-wide entries (no user) remain visible to a
            # user-scoped read - a team memory is meant to be shared.
            conditions.append(
                or_(MemoryEntry.user_id == scope.user_id, MemoryEntry.user_id.is_(None))
            )
        else:
            conditions.append(MemoryEntry.user_id.is_(None))

        if scope.workspace_id is not None:
            conditions.append(
                or_(
                    MemoryEntry.workspace_id == scope.workspace_id,
                    MemoryEntry.workspace_id.is_(None),
                )
            )
        if scope.agent_name is not None:
            conditions.append(
                or_(
                    MemoryEntry.agent_name == scope.agent_name,
                    MemoryEntry.agent_name.is_(None),
                )
            )

        if kind is not None:
            conditions.append(MemoryEntry.kind == kind.value)
        if key is not None:
            conditions.append(MemoryEntry.key == key)
        if not include_superseded:
            conditions.append(MemoryEntry.superseded_by.is_(None))
        if not include_expired:
            conditions.append(
                or_(
                    MemoryEntry.expires_at.is_(None),
                    MemoryEntry.expires_at > datetime.now(UTC),
                )
            )

        return (
            select(MemoryEntry)
            .where(and_(*conditions))
            .order_by(MemoryEntry.created_at.desc())
        )

    @staticmethod
    def _to_memory(entry: MemoryEntry) -> Memory:
        return Memory(
            key=entry.key,
            value=(entry.value or {}).get("value"),
            kind=MemoryKind(entry.kind),
            scope=MemoryScope(
                tenant_id=entry.tenant_id,
                user_id=entry.user_id,
                workspace_id=entry.workspace_id,
                agent_name=entry.agent_name,
            ),
            source=MemorySource(entry.source),
            confidence=entry.confidence,
            entry_id=entry.id,
            created_at=entry.created_at,
            expires_at=entry.expires_at,
            superseded_by=entry.superseded_by,
            detail=entry.detail or {},
        )

    async def recall(
        self,
        *,
        scope: MemoryScope,
        kind: MemoryKind | None = None,
        key: str | None = None,
        limit: int = 50,
        min_confidence: float = 0.0,
        trusted_only: bool = False,
    ) -> list[Memory]:
        """Recalls live, in-scope memories, newest first.

        `trusted_only` is the switch a caller flips when a memory is
        about to be *stated as fact* rather than merely considered -
        the practical expression of the prompt's rule against treating
        agent inference as verified knowledge.
        """
        async with self._store.session() as session:
            rows = (
                await session.execute(
                    self._scoped_query(kind=kind, scope=scope, key=key).limit(limit)
                )
            ).scalars().all()

        memories = [self._to_memory(row) for row in rows]
        memories = [m for m in memories if m.confidence >= min_confidence]
        if trusted_only:
            memories = [m for m in memories if m.is_trusted]
        return memories

    async def recall_one(
        self, *, scope: MemoryScope, kind: MemoryKind, key: str
    ) -> Memory | None:
        found = await self.recall(scope=scope, kind=kind, key=key, limit=1)
        return found[0] if found else None

    async def history(
        self, *, scope: MemoryScope, kind: MemoryKind, key: str
    ) -> list[Memory]:
        """Every version of one memory, including superseded ones.

        What makes correction auditable rather than destructive.
        """
        async with self._store.session() as session:
            rows = (
                await session.execute(
                    self._scoped_query(
                        kind=kind,
                        scope=scope,
                        key=key,
                        include_expired=True,
                        include_superseded=True,
                    )
                )
            ).scalars().all()
        return [self._to_memory(row) for row in rows]

    async def forget(
        self, *, scope: MemoryScope, kind: MemoryKind, key: str
    ) -> int:
        """Hard-deletes every version of a memory.

        Genuinely deletes, unlike correction: this is the
        right-to-be-forgotten path, where leaving a tombstone
        containing the value would defeat the purpose.
        """
        async with self._store.session() as session:
            rows = (
                await session.execute(
                    self._scoped_query(
                        kind=kind,
                        scope=scope,
                        key=key,
                        include_expired=True,
                        include_superseded=True,
                    )
                )
            ).scalars().all()
            for row in rows:
                await session.delete(row)
            await session.commit()
        _logger.info("memory_forgotten", key=key, kind=kind.value, count=len(rows))
        return len(rows)

    async def purge_expired(self, *, tenant_id: str | None = None) -> int:
        """Removes entries past their TTL.

        Expiry is enforced on read as well (`_scoped_query` filters
        them out), so this is housekeeping rather than a correctness
        control - an unpurged expired entry is already invisible.
        """
        now = datetime.now(UTC)
        async with self._store.session() as session:
            stmt = select(MemoryEntry).where(
                MemoryEntry.expires_at.is_not(None), MemoryEntry.expires_at <= now
            )
            if tenant_id is not None:
                stmt = stmt.where(MemoryEntry.tenant_id == tenant_id)
            rows = (await session.execute(stmt)).scalars().all()
            for row in rows:
                await session.delete(row)
            await session.commit()
        return len(rows)

    async def record_episode(
        self,
        *,
        scope: MemoryScope,
        task_id: str,
        intent: str,
        outcome: str,
        summary: str,
        detail: dict[str, Any] | None = None,
    ) -> Memory:
        """Records what happened on a task - episodic memory.

        Keyed by intent rather than task id so that *recalling* is
        useful: the question a future task asks is "what happened last
        time we were asked something like this", not "what happened in
        task 9f3a".
        """
        return await self.remember(
            key=f"episode:{intent[:180]}",
            value={"task_id": task_id, "outcome": outcome, "summary": summary},
            kind=MemoryKind.EPISODIC,
            scope=scope,
            source=MemorySource.OBSERVED,
            confidence=1.0,
            detail=detail,
        )

