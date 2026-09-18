# praxis/core/checkpoint.py
"""A LangGraph checkpointer backed by Praxis's own database.

Durable checkpointing is what makes a task paused on approval survive
a process restart. LangGraph provides this, and the obvious choice was
its official `AsyncPostgresSaver` - but that is built on `psycopg`,
whose async mode **cannot run on Windows' ProactorEventLoop**
(`psycopg.InterfaceError`, verified directly on this machine). The
workaround would be forcing `WindowsSelectorEventLoopPolicy`
process-wide, which is a global change to satisfy one component and
risks subtle breakage elsewhere.

So rather than fight the library or fork its storage, this implements
LangGraph's own documented extension point -
`BaseCheckpointSaver` - over the SQLAlchemy/asyncpg engine Praxis
already uses everywhere else. That means:

- One database driver, already working on every supported platform.
- One connection pool, already configured and health-checked.
- LangGraph's real serializer (`JsonPlusSerializer`) does the
  encoding, so checkpoint fidelity is the library's concern, not a
  bespoke one - which is exactly what the previous hand-rolled
  checkpointer got wrong (it degraded unserializable values to
  `repr()`).

Two tables, mirroring LangGraph's own schema shape:
`langgraph_checkpoints` (the state snapshots) and
`langgraph_writes` (pending channel writes for a partially-completed
superstep). Both are tenant-scoped, because a thread id is a task id
and tasks belong to tenants.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

import structlog
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from sqlalchemy import delete, select

from praxis.memory.db import PostgresStore
from praxis.memory.models import DEFAULT_TENANT_ID, LangGraphCheckpoint, LangGraphWrite

_logger = structlog.get_logger(__name__)


class PraxisCheckpointSaver(BaseCheckpointSaver):
    """LangGraph checkpointing over Praxis's existing Postgres store.

    Only the async methods are implemented. LangGraph calls the sync
    ones (`get_tuple`, `put`, ...) only from sync entrypoints, and
    Praxis drives every graph with `ainvoke` - so a sync path here
    would be dead code that could silently diverge from the async
    one. Calling a sync method raises rather than pretending.
    """

    def __init__(self, store: PostgresStore, *, tenant_id: str = DEFAULT_TENANT_ID) -> None:
        super().__init__(serde=JsonPlusSerializer())
        self._store = store
        self._tenant_id = tenant_id

    def for_tenant(self, tenant_id: str) -> "PraxisCheckpointSaver":
        """A saver scoped to one tenant.

        Cheap to construct and shares the connection pool, so the
        Orchestrator makes one per task rather than threading a tenant
        argument through LangGraph's interface (which has no slot for
        it).
        """
        return PraxisCheckpointSaver(self._store, tenant_id=tenant_id)

    # ---------------------------------------------------------------- #
    # Reads
    # ---------------------------------------------------------------- #

    async def aget_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        async with self._store.session() as session:
            stmt = select(LangGraphCheckpoint).where(
                LangGraphCheckpoint.thread_id == thread_id,
                LangGraphCheckpoint.checkpoint_ns == checkpoint_ns,
            )
            if checkpoint_id:
                stmt = stmt.where(LangGraphCheckpoint.checkpoint_id == checkpoint_id)
            else:
                # No explicit id means "the latest", which is ordered
                # by insertion rather than by id: checkpoint ids are
                # UUIDs and are not lexically ordered by time.
                stmt = stmt.order_by(LangGraphCheckpoint.seq.desc())
            row = (await session.execute(stmt.limit(1))).scalar_one_or_none()

            if row is None:
                return None

            writes = (
                await session.execute(
                    select(LangGraphWrite)
                    .where(
                        LangGraphWrite.thread_id == thread_id,
                        LangGraphWrite.checkpoint_ns == checkpoint_ns,
                        LangGraphWrite.checkpoint_id == row.checkpoint_id,
                    )
                    .order_by(LangGraphWrite.idx)
                )
            ).scalars().all()

        return self._to_tuple(row, writes)

    def _to_tuple(
        self, row: LangGraphCheckpoint, writes: Sequence[LangGraphWrite]
    ) -> CheckpointTuple:
        checkpoint = self.serde.loads_typed((row.checkpoint_type, row.checkpoint))
        metadata = self.serde.loads_typed((row.metadata_type, row.checkpoint_metadata))
        parent_config = (
            {
                "configurable": {
                    "thread_id": row.thread_id,
                    "checkpoint_ns": row.checkpoint_ns,
                    "checkpoint_id": row.parent_checkpoint_id,
                }
            }
            if row.parent_checkpoint_id
            else None
        )
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": row.thread_id,
                    "checkpoint_ns": row.checkpoint_ns,
                    "checkpoint_id": row.checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=[
                (
                    write.task_id,
                    write.channel,
                    self.serde.loads_typed((write.value_type, write.value)),
                )
                for write in writes
            ],
        )

    async def alist(
        self,
        config: dict[str, Any] | None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Lists checkpoints newest-first - what powers time-travel
        and the replay/debug view."""
        async with self._store.session() as session:
            stmt = select(LangGraphCheckpoint).where(
                LangGraphCheckpoint.tenant_id == self._tenant_id
            )
            if config is not None:
                stmt = stmt.where(
                    LangGraphCheckpoint.thread_id == config["configurable"]["thread_id"],
                    LangGraphCheckpoint.checkpoint_ns
                    == config["configurable"].get("checkpoint_ns", ""),
                )
            if before is not None:
                before_seq = (
                    await session.execute(
                        select(LangGraphCheckpoint.seq).where(
                            LangGraphCheckpoint.checkpoint_id
                            == before["configurable"]["checkpoint_id"]
                        )
                    )
                ).scalar_one_or_none()
                if before_seq is not None:
                    stmt = stmt.where(LangGraphCheckpoint.seq < before_seq)

            stmt = stmt.order_by(LangGraphCheckpoint.seq.desc())
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = (await session.execute(stmt)).scalars().all()

            for row in rows:
                writes = (
                    await session.execute(
                        select(LangGraphWrite)
                        .where(
                            LangGraphWrite.thread_id == row.thread_id,
                            LangGraphWrite.checkpoint_ns == row.checkpoint_ns,
                            LangGraphWrite.checkpoint_id == row.checkpoint_id,
                        )
                        .order_by(LangGraphWrite.idx)
                    )
                ).scalars().all()
                yield self._to_tuple(row, writes)

    # ---------------------------------------------------------------- #
    # Writes
    # ---------------------------------------------------------------- #

    async def aput(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict[str, Any]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_id = config["configurable"].get("checkpoint_id")
        checkpoint_id = checkpoint["id"]

        checkpoint_type, checkpoint_blob = self.serde.dumps_typed(checkpoint)
        metadata_type, metadata_blob = self.serde.dumps_typed(dict(metadata))

        async with self._store.session() as session:
            existing = (
                await session.execute(
                    select(LangGraphCheckpoint).where(
                        LangGraphCheckpoint.thread_id == thread_id,
                        LangGraphCheckpoint.checkpoint_ns == checkpoint_ns,
                        LangGraphCheckpoint.checkpoint_id == checkpoint_id,
                    )
                )
            ).scalar_one_or_none()

            if existing is None:
                session.add(
                    LangGraphCheckpoint(
                        tenant_id=self._tenant_id,
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        checkpoint_id=checkpoint_id,
                        parent_checkpoint_id=parent_id,
                        checkpoint=checkpoint_blob,
                        checkpoint_type=checkpoint_type,
                        checkpoint_metadata=metadata_blob,
                        metadata_type=metadata_type,
                    )
                )
            else:
                existing.checkpoint = checkpoint_blob
                existing.checkpoint_type = checkpoint_type
                existing.checkpoint_metadata = metadata_blob
                existing.metadata_type = metadata_type
                existing.parent_checkpoint_id = parent_id
            await session.commit()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Records pending channel writes for a partially-completed
        superstep - what lets an interrupted step resume without
        losing its siblings' completed work."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        async with self._store.session() as session:
            for index, (channel, value) in enumerate(writes):
                value_type, value_blob = self.serde.dumps_typed(value)
                existing = (
                    await session.execute(
                        select(LangGraphWrite).where(
                            LangGraphWrite.thread_id == thread_id,
                            LangGraphWrite.checkpoint_ns == checkpoint_ns,
                            LangGraphWrite.checkpoint_id == checkpoint_id,
                            LangGraphWrite.task_id == task_id,
                            LangGraphWrite.idx == index,
                        )
                    )
                ).scalar_one_or_none()
                if existing is None:
                    session.add(
                        LangGraphWrite(
                            tenant_id=self._tenant_id,
                            thread_id=thread_id,
                            checkpoint_ns=checkpoint_ns,
                            checkpoint_id=checkpoint_id,
                            task_id=task_id,
                            task_path=task_path,
                            idx=index,
                            channel=channel,
                            value=value_blob,
                            value_type=value_type,
                        )
                    )
                else:
                    existing.channel = channel
                    existing.value = value_blob
                    existing.value_type = value_type
            await session.commit()

    async def adelete_thread(self, thread_id: str) -> None:
        """Drops every checkpoint for a thread - called when a task
        reaches a terminal state and has nothing left to resume."""
        async with self._store.session() as session:
            await session.execute(
                delete(LangGraphWrite).where(LangGraphWrite.thread_id == thread_id)
            )
            await session.execute(
                delete(LangGraphCheckpoint).where(
                    LangGraphCheckpoint.thread_id == thread_id
                )
            )
            await session.commit()

    # ---------------------------------------------------------------- #
    # Sync surface - deliberately unavailable (see class docstring)
    # ---------------------------------------------------------------- #

    def get_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        raise NotImplementedError(
            "PraxisCheckpointSaver is async-only; drive the graph with ainvoke/astream"
        )

    def put(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "PraxisCheckpointSaver is async-only; drive the graph with ainvoke/astream"
        )

    def put_writes(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "PraxisCheckpointSaver is async-only; drive the graph with ainvoke/astream"
        )

    def list(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "PraxisCheckpointSaver is async-only; drive the graph with ainvoke/astream"
        )
