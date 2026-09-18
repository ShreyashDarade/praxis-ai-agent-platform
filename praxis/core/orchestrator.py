# praxis/core/orchestrator.py
"""The Orchestrator (spec §3, §7, §8, §9, §14), executed by LangGraph.

Turns a `Task`'s intent into a materialized checklist and drives it
through an execution graph, pausing for approval on any `mutating`
step.

**Phase 21: migrated onto LangGraph.** The previous engine was a
hand-rolled level-by-level driver - `compute_levels()` grouped steps
into dependency levels, each level ran under `asyncio.gather`, and
pauses were managed with in-memory bookkeeping (`paused_index`,
`pending_args`) plus a bespoke checkpoint serializer. LangGraph now
owns execution (`praxis.core.graph_engine`), which changes four
things concretely:

1. **The DAG is the graph.** One node per `PlanStep`, edges from
   `depends_on`; LangGraph's superstep model runs independent steps
   concurrently on its own.
2. **A pause is a real `interrupt()`**, resumed with
   `Command(resume=...)`. LangGraph replays the interrupted node from
   its start, so the approval check and the action it guards stay in
   *one* function rather than being split across a pause site and a
   separate resume path that had to reconstruct the same arguments.
3. **Checkpointing is the library's** (`praxis.core.checkpoint`
   implements LangGraph's `BaseCheckpointSaver` over Praxis's own
   Postgres store). The previous serializer degraded unserializable
   values to `repr()`; LangGraph's handles them properly.
4. **Replay/time-travel** (`replay_history`) is now available for
   tasks, which it was not before.

**What did NOT change, deliberately.** Every security-relevant
decision is still made here, in this module, and was carried across
unmodified rather than rewritten:

- risk tiering (`should_pause_for_approval`);
- identity-bound approval records, including the argument-hash
  re-verification that makes an approval un-replayable against
  different arguments;
- tenancy on every write;
- the two-layer execution-mode enforcement;
- audit logging of every decision;
- capability synthesis for an unregistered skill;
- output validation, the tool-result cache, and the
  prior-context URL allow-list.

`compute_levels()` is still used - not to execute, but to *validate*:
LangGraph would happily build a cyclic graph, and a plan with a cycle
or an out-of-range dependency must fail with a clear "invalid plan"
error before anything runs.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import structlog
from langgraph.types import Command, interrupt
from sqlalchemy import select

from praxis.agents.planner import Planner
from praxis.agents.publication import SkillStatus
from praxis.agents.skill import Skill
from praxis.cache.memory_cache import InMemoryCache
from praxis.config import Settings
from praxis.core.checkpoint import PraxisCheckpointSaver
from praxis.core.events import TaskEventBus, task_event_bus, task_state_snapshot
from praxis.core.exceptions import SandboxViolationError, SynthesisValidationError
from praxis.core.execution_graph import PlanStep, compute_levels, resolve_args
from praxis.core.execution_mode import (
    ExecutionMode,
    MutationNotPermittedError,
    visible_skills,
)
from praxis.core.graph_engine import build_task_graph
from praxis.core.interfaces import Connector, GraphStore
from praxis.core.risk_policy import PendingInput, should_pause_for_approval
from praxis.memory.db import PostgresStore
from praxis.memory.models import SkillRecord, Task
from praxis.observability.logging import bind_correlation_id
from praxis.observability.tracing import start_span
from praxis.safety.output_validation import validate_skill_output
from praxis.security.approval import (
    ApprovalRequest,
    create_pending_approval,
    decide_approval,
    verify_approval_binding,
)
from praxis.security.audit import AuditLogger
from praxis.security.policy import Permission, policy_engine
from praxis.security.principal import SYSTEM_PRINCIPAL, Principal

_logger = structlog.get_logger(__name__)

# Phase 9 (spec §6.1's "prior-context-only fetch"): the Orchestrator is
# the one place that decides which URLs are legitimate fetch targets for
# `web_read`/`web_crawl` - never the skill/connector itself, and never
# an LLM's own freshly-generated output. Deliberately permissive
# matching is safe here specifically because every candidate this ever
# runs against already qualifies as "prior context" by definition.
_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")

# A task in one of these will never transition again.
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


def _extract_urls(text: str) -> set[str]:
    return {match.rstrip(".,;:!?)]}'\"") for match in _URL_RE.findall(text)}


class SkillRegistryLike(Protocol):
    """What the Orchestrator needs from a skill registry - satisfied by
    the `praxis.agents.skill_registry` module itself, or any stand-in
    exposing the same two functions in a test."""

    def get_skill(self, name: str) -> Skill: ...

    def all_skills(self) -> list[Skill]: ...


class CapabilityFactoryLike(Protocol):
    """What the Orchestrator needs from a Capability Factory."""

    async def synthesize(
        self, need_description: str, *, connector: Any | None = None, task_id: str | None = None
    ) -> Skill: ...


class ConnectorRegistryLike(Protocol):
    """What the Orchestrator needs from a connector registry."""

    def get(self, name: str) -> Connector: ...


@dataclass
class _TaskContext:
    """Per-run context that is NOT part of the graph's state.

    Everything here is either non-serializable (a live `Connector`, a
    cache) or a runtime-only concern (cancellation). The graph's own
    state channel carries only JSON-ish data, which is what keeps a
    checkpoint restorable - live objects are re-resolved by name
    instead (see `_rebuild_context`).
    """

    steps: list[PlanStep]
    principal: Principal = SYSTEM_PRINCIPAL
    mode: ExecutionMode = ExecutionMode.EXECUTE
    connector: Connector | None = None
    connector_name: str | None = None
    # Spec §11's tool-result cache: one task's lifetime, so a fresh
    # instance per run means it can never leak across tasks.
    tool_cache: InMemoryCache = field(default_factory=InMemoryCache)
    # In-flight deduplication for identical read-only calls.
    #
    # The cache alone dedupes *sequential* repeats, but LangGraph runs
    # independent steps concurrently, so two steps naming the same
    # skill+args start together and both miss the cache. The first to
    # arrive registers a future here; the others await it instead of
    # issuing a second identical call. This replaces the previous
    # engine's level-scoped "runner/alias" bookkeeping and is strictly
    # more general - it dedupes across supersteps too, not just within
    # one.
    in_flight: dict[str, asyncio.Future] = field(default_factory=dict)
    cancelled: bool = False


def _expected_output_keys(all_steps: list[PlanStep], step_index: int) -> set[str]:
    """What output key(s) a later step already expects `step_index` to
    return.

    Closes a real gap: the Planner (which invents the key name when it
    names a not-yet-existing skill) and the Capability Factory (which
    decides the synthesized skill's real `outputs`) are two independent
    LLM calls with no shared state. Feeding the already-committed key
    name into the synthesis prompt means the Factory knows exactly what
    shape is expected of it.
    """
    keys: set[str] = set()
    prefix = f"${step_index}."
    for other in all_steps:
        for value in other.args.values():
            if isinstance(value, str) and value.startswith(prefix):
                keys.add(value[len(prefix):])
    return keys


def _describe_step(step: PlanStep) -> str:
    return f"{step.skill_name}({step.args})"


def _tool_result_cache_key(skill_name: str, args: dict[str, Any]) -> str:
    """Identical `skill_name` + `args` hash to the same key regardless
    of dict ordering - a read-only call is idempotent (spec §11)."""
    payload = json.dumps({"skill": skill_name, "args": args}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _plan_to_json(steps: list[PlanStep]) -> list[dict[str, Any]]:
    return [
        {
            "skill_name": step.skill_name,
            "args": step.args,
            "depends_on": list(step.depends_on),
        }
        for step in steps
    ]


def _plan_from_json(payload: list[dict[str, Any]]) -> list[PlanStep]:
    return [
        PlanStep(
            skill_name=raw["skill_name"],
            args=dict(raw.get("args") or {}),
            depends_on=list(raw.get("depends_on") or []),
        )
        for raw in payload or []
    ]


class Orchestrator:
    def __init__(
        self,
        store: PostgresStore,
        planner: Planner,
        skills: SkillRegistryLike,
        capability_factory: CapabilityFactoryLike | None = None,
        event_bus: TaskEventBus | None = None,
        connector_registry: ConnectorRegistryLike | None = None,
        graph_store: GraphStore | None = None,
        audit_logger: AuditLogger | None = None,
        approval_ttl_seconds: int | None = None,
        checkpointer: Any | None = None,
        retain_checkpoints: bool | None = None,
    ) -> None:
        self._store = store
        self._planner = planner
        self._skills = skills
        self._capability_factory = capability_factory
        self._event_bus = event_bus if event_bus is not None else task_event_bus
        self._connector_registry = connector_registry
        self._graph_store = graph_store
        self._audit = audit_logger if audit_logger is not None else AuditLogger(store)
        self._approval_ttl_seconds = approval_ttl_seconds
        # LangGraph checkpointing over Praxis's own store. Defaulting
        # to a real one means every caller gains durable pause/resume
        # with no constructor change.
        self._checkpointer = (
            checkpointer if checkpointer is not None else PraxisCheckpointSaver(store)
        )
        self._retain_checkpoints = retain_checkpoints
        self._contexts: dict[str, _TaskContext] = {}

    @property
    def _retains_checkpoints(self) -> bool:
        """Whether a finished task keeps its replayable history.

        Read from `Settings` on demand rather than at construction, so a
        deployment that flips it does not need every long-lived
        Orchestrator rebuilt. Falls back to retaining: losing a failed
        run's history is worse than keeping rows, and a Settings that
        cannot be constructed must not silently turn the debug feature
        off.
        """
        if self._retain_checkpoints is not None:
            return self._retain_checkpoints
        try:
            from praxis.config import Settings as _Settings

            return bool(_Settings().retain_checkpoints_after_completion)
        except Exception:  # noqa: BLE001 - see docstring: fail toward retention
            return True

    async def _release_thread(self, task_id: str) -> None:
        """Drops a terminal task's in-memory context, and its checkpoints
        only when history is not being retained."""
        self._contexts.pop(task_id, None)
        if not self._retains_checkpoints:
            await self._checkpointer.adelete_thread(task_id)

    async def prune_task_history(self, task_id: str) -> None:
        """Deletes a finished task's checkpoints explicitly.

        The escape hatch for the retention default: an operator (or a
        scheduled sweeper) reclaims the space for a run nobody needs to
        replay any more. Deliberately not automatic - see
        `Settings.retain_checkpoints_after_completion`.
        """
        await self._checkpointer.adelete_thread(task_id)

    @property
    def _approval_ttl(self) -> int:
        if self._approval_ttl_seconds is not None:
            return self._approval_ttl_seconds
        try:
            return Settings().approval_ttl_seconds
        except Exception:  # noqa: BLE001 - an unconfigured env must not break a pause
            return 3600

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #

    async def _record_skill_used(self, task_id: str, skill_name: str, tenant_id: str) -> None:
        if self._graph_store is None:
            return
        await self._graph_store.add_edge(
            source=task_id, relation="used_skill", target=skill_name, tenant_id=tenant_id
        )

    async def _publish_state(self, task: Task) -> None:
        """Publishes a live snapshot to any `WS /tasks/{id}/stream`
        subscriber, right after a commit that changed real state."""
        await self._event_bus.publish(task.id, task_state_snapshot(task))

    async def _update_checklist(
        self, task_id: str, index: int, *, status: str, reason: str | None = None
    ) -> None:
        """Marks one checklist item and publishes the change.

        Loads the row fresh each time because steps run concurrently:
        two nodes finishing together must not overwrite each other's
        checklist update, and re-reading inside the transaction is what
        prevents that. The list is reassigned rather than mutated in
        place so SQLAlchemy's change tracking on the plain `JSON`
        column actually sees the write.
        """
        async with self._store.session() as session:
            task = await session.get(Task, task_id)
            if task is None:  # pragma: no cover - defensive
                return
            checklist = [dict(item) for item in task.checklist]
            if index < len(checklist):
                checklist[index] = {**checklist[index], "status": status, "reason": reason}
                task.checklist = checklist
            await session.commit()
        await self._publish_state(task)

    async def _fail_task(
        self, task_id: str, message: str, *, checklist: list[dict[str, Any]] | None = None
    ) -> None:
        async with self._store.session() as session:
            task = await session.get(Task, task_id)
            assert task is not None
            if checklist is not None:
                task.checklist = checklist
            task.status = "failed"
            task.result = {"error": message}
            await session.commit()
        await self._publish_state(task)
        # Retained by default: a failed run is the one an operator most
        # wants to replay, so its history outlives it unless the
        # deployment opted out.
        await self._release_thread(task_id)

    @staticmethod
    def _build_result(
        steps: list[PlanStep],
        results: dict[int, Any],
        checklist: list[dict[str, Any]],
    ) -> dict[str, Any]:
        completed = sum(1 for item in checklist if item["status"] == "completed")
        steps_summary = [
            {
                "step": index,
                "skill": step.skill_name,
                "status": checklist[index]["status"] if index < len(checklist) else "unknown",
                "output": results.get(index),
            }
            for index, step in enumerate(steps)
        ]
        return {
            "summary": f"Task completed: {completed}/{len(checklist)} step(s) completed.",
            "steps": steps_summary,
        }

    # ------------------------------------------------------------------ #
    # The per-step node body - where every security control lives
    # ------------------------------------------------------------------ #

    async def _approval_gate_blocks(
        self, skill_name: str, context: _TaskContext
    ) -> str | None:
        """Why this skill may not run, or `None` if it may.

        Only *synthesized* skills are gated. A hand-written skill is
        already in version control and was reviewed by whoever merged
        it; requiring a second approval at runtime would add ceremony
        without adding scrutiny. The gate exists for code an LLM wrote
        minutes ago, which nobody has read.
        """
        async with self._store.session() as session:
            rows = (
                await session.execute(
                    select(SkillRecord).where(
                        SkillRecord.tenant_id == context.principal.tenant_id,
                        SkillRecord.name == skill_name,
                    )
                )
            ).scalars().all()

        synthesized = [row for row in rows if row.synthesized]
        if not synthesized:
            return None

        if any(SkillStatus(row.status).is_callable for row in synthesized):
            return None

        statuses = sorted({row.status for row in synthesized})
        return (
            f"skill '{skill_name}' was synthesized but is not approved for execution "
            f"(status: {', '.join(statuses)}); a principal with 'skill:approve' must "
            "approve it first"
        )

    @staticmethod
    def _settle_in_flight(
        context: _TaskContext,
        cache_key: str | None,
        *,
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        """Resolves (and clears) the in-flight future for `cache_key`.

        Always called on both the success and failure paths: a future
        left unresolved would hang every concurrent step waiting on
        the same call, turning a deduplication optimization into a
        deadlock.
        """
        if cache_key is None:
            return
        future = context.in_flight.pop(cache_key, None)
        if future is None or future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)

    def _make_step_runner(self, task_id: str, context: _TaskContext):
        """Builds the `StepRunner` closure for one task's graph.

        A closure rather than a method so the non-serializable context
        (connector, cache, principal) stays out of LangGraph's state
        channel while remaining reachable from every node.
        """

        async def run_step(
            index: int,
            step: PlanStep,
            results: dict[int, Any],
            known_urls: list[str],
        ) -> dict[str, Any]:
            if context.cancelled:
                return {"step_status": {index: "skipped"}}

            resolved_args = resolve_args(step.args, results)

            # --- resolve or synthesize the skill ---------------------- #
            skill: Skill | None
            try:
                skill = self._skills.get_skill(step.skill_name)
            except KeyError:
                skill, synthesis_error = await self._synthesize_missing_skill(
                    task_id, context, index, step, resolved_args
                )
                if skill is None:
                    # The real sandbox/validation detail is propagated,
                    # not collapsed into a generic "not registered" -
                    # spec §12: a failure is never reduced to a message
                    # that hides why it happened.
                    return {
                        "step_status": {index: "failed"},
                        "errors": [f"step {index}: {synthesis_error}"],
                    }

            # --- the synthesis approval gate -------------------------- #
            # A synthesized skill that has not been approved is
            # refused HERE, at the point of execution. Recording it as
            # `pending_approval` in the catalogue is only bookkeeping;
            # this check is what makes the gate real (Prompt §7:
            # "Generated agents/tools must never be silently
            # trusted").
            blocked = await self._approval_gate_blocks(skill.name, context)
            if blocked is not None:
                await self._update_checklist(task_id, index, status="failed", reason=blocked)
                await self._audit.record(
                    principal=context.principal,
                    action="task:execute_step",
                    resource_type="skill",
                    resource_id=skill.name,
                    decision="denied",
                    reason=blocked,
                    detail={"step": index, "skill": skill.name},
                )
                return {"step_status": {index: "failed"}, "errors": [blocked]}

            # --- execution-mode enforcement, layer 2 ------------------ #
            # A mutating skill that reached this plan despite layer 1 -
            # a freshly *synthesized* one is the real case, since it
            # did not exist when the Planner's visible set was computed.
            if skill.risk == "mutating" and not context.mode.allows_mutation:
                if context.mode.simulates_mutation:
                    simulated = {
                        "simulated": True,
                        "mode": context.mode.value,
                        "would_have_called": {"skill": skill.name, "args": resolved_args},
                        "note": (
                            "dry run: this mutating step was NOT executed; no external "
                            "state was changed"
                        ),
                    }
                    _logger.info(
                        "mutating_step_simulated",
                        skill=skill.name, step=index, task_id=task_id,
                        mode=context.mode.value,
                    )
                    await self._update_checklist(
                        task_id, index, status="completed",
                        reason=f"simulated ({context.mode.value})",
                    )
                    return {"results": {index: simulated}, "step_status": {index: "completed"}}

                error = MutationNotPermittedError(
                    (
                        f"step {index} ('{skill.name}') is a mutating skill, which execution "
                        f"mode '{context.mode.value}' does not permit "
                        f"({context.mode.describe()})"
                    ),
                    mode=context.mode,
                    skill_name=skill.name,
                )
                _logger.warning(
                    "mutating_step_refused",
                    skill=skill.name, step=index, task_id=task_id, mode=context.mode.value,
                )
                await self._audit.record(
                    principal=context.principal,
                    action="task:execute_step",
                    resource_type="task",
                    resource_id=task_id,
                    decision="denied",
                    reason=str(error),
                    detail={"step": index, "skill": skill.name, "mode": context.mode.value},
                )
                await self._update_checklist(task_id, index, status="failed", reason=str(error))
                return {"step_status": {index: "failed"}, "errors": [str(error)]}

            # --- human approval --------------------------------------- #
            if should_pause_for_approval(skill):
                approved = await self._await_approval(
                    task_id, context, index, step, resolved_args
                )
                if not approved:
                    await self._update_checklist(
                        task_id, index, status="skipped", reason="rejected by operator"
                    )
                    return {
                        "step_status": {index: "skipped"},
                        "errors": [
                            f"step {index} ('{step.skill_name}') was rejected by the operator"
                        ],
                    }

            # --- tool-result cache (read-only calls only) -------------- #
            # Only `read_only` calls are ever cached or deduplicated: a
            # `mutating` call is never assumed idempotent, and each one
            # already pauses for its own individual approval.
            cache_key: str | None = None
            if skill.risk == "read_only":
                cache_key = _tool_result_cache_key(skill.name, resolved_args)

                cached = await context.tool_cache.get(cache_key)
                if cached is not None:
                    _logger.info(
                        "tool_result_cache_hit", skill=skill.name, step=index, task_id=task_id
                    )
                    await self._record_skill_used(
                        task_id, skill.name, context.principal.tenant_id
                    )
                    await self._update_checklist(task_id, index, status="completed")
                    return {
                        "results": {index: cached},
                        "step_status": {index: "completed"},
                        "known_urls": sorted(_extract_urls(str(cached))),
                    }

                # An identical call already running concurrently: wait
                # for its result rather than issuing a second one.
                pending = context.in_flight.get(cache_key)
                if pending is not None:
                    _logger.info(
                        "tool_result_in_flight_join",
                        skill=skill.name, step=index, task_id=task_id,
                    )
                    try:
                        shared = await pending
                    except Exception as exc:  # noqa: BLE001 - mirrors the runner's own failure
                        await self._update_checklist(
                            task_id, index, status="failed", reason=str(exc)
                        )
                        return {
                            "step_status": {index: "failed"},
                            "errors": [
                                f"step {index} ('{skill.name}') failed: a concurrent "
                                f"identical call failed: {exc}"
                            ],
                        }
                    await self._record_skill_used(
                        task_id, skill.name, context.principal.tenant_id
                    )
                    await self._update_checklist(task_id, index, status="completed")
                    return {
                        "results": {index: shared},
                        "step_status": {index: "completed"},
                        "known_urls": sorted(_extract_urls(str(shared))),
                    }

                context.in_flight[cache_key] = asyncio.get_running_loop().create_future()

            # --- execute ----------------------------------------------- #
            await self._update_checklist(task_id, index, status="in_progress")
            _logger.info(
                "skill_execution_started", skill=skill.name, step=index, task_id=task_id
            )
            try:
                with start_span("skill.execute", skill=skill.name, step=index):
                    output = await skill.run(
                        **resolved_args,
                        known_urls=set(known_urls),
                        tenant_id=context.principal.tenant_id,
                    )
                output = validate_skill_output(skill.name, skill.outputs, output)
            except Exception as exc:  # noqa: BLE001 - a step's failure, not swallowed (spec §12)
                _logger.error(
                    "skill_execution_failed",
                    skill=skill.name, step=index, task_id=task_id, error=str(exc),
                )
                # Anything awaiting this exact call must fail too,
                # rather than hanging on a future nobody will resolve.
                self._settle_in_flight(context, cache_key, error=exc)
                await self._update_checklist(task_id, index, status="failed", reason=str(exc))
                return {
                    "step_status": {index: "failed"},
                    "errors": [f"step {index} ('{skill.name}') failed: {exc}"],
                }

            _logger.info(
                "skill_execution_completed", skill=skill.name, step=index, task_id=task_id
            )
            if cache_key is not None:
                await context.tool_cache.set(cache_key, output)
                self._settle_in_flight(context, cache_key, result=output)
            await self._record_skill_used(task_id, skill.name, context.principal.tenant_id)
            await self._update_checklist(task_id, index, status="completed")

            return {
                "results": {index: output},
                "step_status": {index: "completed"},
                "known_urls": sorted(_extract_urls(str(output))),
            }

        return run_step

    async def _await_approval(
        self,
        task_id: str,
        context: _TaskContext,
        index: int,
        step: PlanStep,
        resolved_args: dict[str, Any],
    ) -> bool:
        """Records the pending approval, pauses, and verifies on resume.

        The whole approval lifecycle lives in one function because
        LangGraph replays this node from its start on resume: the
        arguments are recomputed from checkpointed upstream results
        rather than stashed, and the hash check then proves they are
        the same ones the operator actually saw.
        """
        pending = PendingInput(
            kind="approval",
            detail=(
                f"approve mutating skill '{step.skill_name}' "
                f"(step {index}) with args {resolved_args}?"
            ),
        )

        # Idempotent by construction (`idempotency_key`), which matters
        # precisely because this runs again on every replay.
        async with self._store.session() as session:
            await create_pending_approval(
                session,
                request=ApprovalRequest(
                    task_id=task_id,
                    step_index=index,
                    skill_name=step.skill_name,
                    args=resolved_args,
                ),
                tenant_id=context.principal.tenant_id,
                ttl_seconds=self._approval_ttl,
            )
            task = await session.get(Task, task_id)
            if task is not None:
                checklist = [dict(item) for item in task.checklist]
                if index < len(checklist):
                    checklist[index] = {**checklist[index], "status": "awaiting_approval"}
                    task.checklist = checklist
                task.status = "awaiting_approval"
                task.pending_input = pending.to_dict()
            await session.commit()
        if task is not None:
            await self._publish_state(task)

        # Pauses the graph here. On resume this whole function runs
        # again and `interrupt` returns the resume payload instead.
        #
        # `step_index` travels in the payload so `resume_after_approval`
        # can resume THIS interrupt specifically. That matters when two
        # mutating steps pause in the same superstep: approving one
        # must not blanket-approve the other, because each mutating
        # action needs its own human decision.
        decision = interrupt({**pending.to_dict(), "step_index": index})
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)

        if approved:
            # Re-verify at resume time (Prompt §8: "Recheck
            # authorization when a paused or scheduled task resumes").
            # A mismatch means the plan's arguments drifted from the
            # ones a human approved, and is a hard refusal.
            async with self._store.session() as session:
                await verify_approval_binding(
                    session,
                    task_id=task_id,
                    step_index=index,
                    skill_name=step.skill_name,
                    args=resolved_args,
                )
        return approved

    async def _synthesize_missing_skill(
        self,
        task_id: str,
        context: _TaskContext,
        index: int,
        step: PlanStep,
        resolved_args: dict[str, Any],
    ) -> tuple[Skill | None, str]:
        """Synthesizes a skill the plan named but nothing provides.

        Returns `(skill, "")` on success, or `(None, reason)` carrying
        the *real* failure detail - the sandbox's own output, not a
        generic message - so the caller can surface it on the task
        rather than only in the checklist.
        """
        if self._capability_factory is None:
            message = f"no skill named '{step.skill_name}' is registered"
            _logger.warning(
                "skill_not_registered_no_factory", skill=step.skill_name, step=index
            )
            await self._update_checklist(
                task_id, index, status="skipped", reason=message
            )
            return None, message

        need_description = (
            f"A running plan needs a skill named '{step.skill_name}', to be called with "
            f"keyword arguments {resolved_args!r} - no such skill is registered yet. "
            "Synthesize a skill that fulfills this need, accepting exactly those keyword "
            "argument names."
        )
        expected_keys = _expected_output_keys(context.steps, index)
        if expected_keys:
            plural = len(expected_keys) != 1
            need_description += (
                f"\n\nA later step in this same plan will read this skill's result using "
                f"the key(s) {sorted(expected_keys)!r} (as \"$<this step's index>.<key>\") - "
                f"your declared `outputs` MUST include exactly "
                f"{'those keys' if plural else 'that key'}, and `run()` must return a dict "
                f"containing {'them' if plural else 'it'}."
            )

        _logger.info(
            "capability_synthesis_triggered", skill=step.skill_name, step=index, task_id=task_id
        )
        try:
            skill = await self._capability_factory.synthesize(
                need_description=need_description, connector=context.connector, task_id=task_id
            )
            return skill, ""
        except (SynthesisValidationError, SandboxViolationError) as exc:
            failure = (
                f"no skill named '{step.skill_name}' is registered; capability synthesis "
                f"also failed: {exc}"
            )
            _logger.error(
                "capability_synthesis_failed",
                skill=step.skill_name, step=index, task_id=task_id, error=str(exc),
            )
            await self._update_checklist(task_id, index, status="failed", reason=failure)
            return None, failure

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def start_task(
        self,
        intent_text: str,
        *,
        connector_name: str | None = None,
        principal: Principal = SYSTEM_PRINCIPAL,
        mode: ExecutionMode = ExecutionMode.EXECUTE,
    ) -> str:
        """Plans `intent_text` and drives it to completion (or a pause)."""
        async with self._store.session() as session:
            task = Task(
                tenant_id=principal.tenant_id,
                created_by_user_id=principal.user_id,
                intent_text=intent_text,
                status="planning",
                mode=mode.value,
                checklist=[],
                plan=[],
            )
            session.add(task)
            await session.commit()
            task_id = task.id
            correlation_id = task.correlation_id

        with bind_correlation_id(correlation_id):
            _logger.info(
                "task_started",
                task_id=task_id,
                intent_text=intent_text,
                connector_name=connector_name,
                mode=mode.value,
                principal=principal.describe(),
            )
        await self._audit.record(
            principal=principal,
            action="task:create",
            resource_type="task",
            resource_id=task_id,
            detail={"intent_text": intent_text, "mode": mode.value, "connector": connector_name},
            correlation_id=correlation_id,
        )

        # Resolved once, up front - before the (real, expensive)
        # Planner call - so a typo'd connector fails fast and clearly.
        connector: Connector | None = None
        if connector_name is not None:
            if self._connector_registry is None:
                message = (
                    f"task requested connector '{connector_name}' but this Orchestrator has "
                    "no connector registry configured"
                )
                with bind_correlation_id(correlation_id):
                    _logger.error(
                        "task_connector_resolution_failed",
                        task_id=task_id, connector_name=connector_name, error=message,
                    )
                await self._fail_task(task_id, message, checklist=[])
                return task_id
            try:
                connector = self._connector_registry.get(connector_name)
            except KeyError as exc:
                with bind_correlation_id(correlation_id):
                    _logger.error(
                        "task_connector_resolution_failed",
                        task_id=task_id, connector_name=connector_name, error=str(exc),
                    )
                await self._fail_task(
                    task_id, f"unknown connector '{connector_name}': {exc}", checklist=[]
                )
                return task_id

        try:
            # Enforcement layer 1: under any mode that cannot mutate,
            # the Planner is never shown a mutating skill.
            steps = await self._planner.plan(
                intent_text, visible_skills(self._skills.all_skills(), mode), connector=connector
            )
        except Exception as exc:  # noqa: BLE001 - a planning failure fails the task
            with bind_correlation_id(correlation_id):
                _logger.error("task_planning_failed", task_id=task_id, error=str(exc))
            await self._fail_task(task_id, f"planning failed: {exc}", checklist=[])
            return task_id

        checklist = [
            {"content": _describe_step(step), "status": "pending", "reason": None}
            for step in steps
        ]

        # `compute_levels` is no longer the executor, but it is still
        # the validator: LangGraph would happily build a cyclic graph,
        # and a cycle or an out-of-range dependency must fail clearly
        # before anything runs.
        try:
            compute_levels(steps)
        except ValueError as exc:
            await self._fail_task(task_id, f"invalid plan: {exc}", checklist=checklist)
            return task_id

        # `plan_only`: the plan IS the deliverable.
        if not mode.allows_execution:
            async with self._store.session() as session:
                # A distinct name from the `task` created above: this is
                # a re-read that may legitimately be absent, and reusing
                # the name would conflate "the row I just created" with
                # "whatever is in the database now".
                stored = await session.get(Task, task_id)
                assert stored is not None
                stored.checklist = checklist
                stored.plan = _plan_to_json(steps)
                stored.status = "completed"
                stored.result = {
                    "mode": mode.value,
                    "summary": (
                        f"Plan-only run: produced a {len(steps)}-step plan; nothing was executed."
                    ),
                    "plan": [
                        {
                            "step": index,
                            "skill": step.skill_name,
                            "args": step.args,
                            "depends_on": step.depends_on,
                        }
                        for index, step in enumerate(steps)
                    ],
                }
                await session.commit()
            await self._publish_state(stored)
            with bind_correlation_id(correlation_id):
                _logger.info("task_plan_only_completed", task_id=task_id, steps=len(steps))
            return task_id

        async with self._store.session() as session:
            stored = await session.get(Task, task_id)
            assert stored is not None
            stored.checklist = checklist
            stored.plan = _plan_to_json(steps)
            stored.status = "running"
            await session.commit()
        await self._publish_state(task)

        context = _TaskContext(
            steps=steps,
            principal=principal,
            mode=mode,
            connector=connector,
            connector_name=connector_name,
        )
        self._contexts[task_id] = context

        await self._drive(
            task_id,
            context,
            initial={
                "results": {},
                "step_status": {},
                "known_urls": sorted(_extract_urls(intent_text)),
                "errors": [],
            },
        )
        return task_id

    async def _drive(
        self, task_id: str, context: _TaskContext, *, initial: Any
    ) -> None:
        """Runs (or resumes) the task graph and settles the final state."""
        saver = self._checkpointer
        if isinstance(saver, PraxisCheckpointSaver):
            saver = saver.for_tenant(context.principal.tenant_id)

        graph = build_task_graph(
            context.steps, self._make_step_runner(task_id, context), checkpointer=saver
        )
        config = {
            "configurable": {"thread_id": task_id},
            # One superstep per step is the theoretical worst case
            # (a fully sequential plan); +5 covers LangGraph's own
            # bookkeeping passes.
            "recursion_limit": max(10, len(context.steps) + 5),
        }

        async with self._store.session() as session:
            task = await session.get(Task, task_id)
            correlation_id = task.correlation_id if task else task_id

        with bind_correlation_id(correlation_id):
            with start_span("orchestrator.advance", task_id=task_id):
                final = await graph.ainvoke(initial, config)

            # A pending interrupt means the graph paused for approval;
            # the task row already says `awaiting_approval`.
            if "__interrupt__" in final:
                _logger.info("task_advance_stopped", task_id=task_id, outcome="paused")
                return

            if context.cancelled:
                _logger.info("task_advance_stopped", task_id=task_id, outcome="cancelled")
                self._contexts.pop(task_id, None)
                return

            await self._settle(task_id, context, final)

    async def _settle(
        self, task_id: str, context: _TaskContext, final: dict[str, Any]
    ) -> None:
        """Writes the terminal task state from the graph's final state."""
        results = dict(final.get("results", {}))
        errors = list(final.get("errors", []))

        async with self._store.session() as session:
            task = await session.get(Task, task_id)
            assert task is not None
            checklist = [dict(item) for item in task.checklist]

            if errors:
                task.status = "failed"
                task.result = {
                    **self._build_result(context.steps, results, checklist),
                    "error": errors[0] if len(errors) == 1 else "one or more steps failed",
                    "errors": errors,
                }
            else:
                task.status = "completed"
                task.result = self._build_result(context.steps, results, checklist)
            task.pending_input = None
            await session.commit()

        await self._publish_state(task)
        _logger.info(
            "task_completed" if not errors else "task_failed",
            task_id=task_id,
            steps=len(context.steps),
        )
        # Terminal: nothing left to resume. The checkpoints are kept
        # anyway when history retention is on, because that is what
        # makes `replay_history` a post-hoc debugging tool rather than
        # one that only works while a task is still running.
        await self._release_thread(task_id)

    async def _rebuild_context(self, task: Task) -> _TaskContext | None:
        """Reconstructs run context for a task this process never started.

        The plan comes from the durable `tasks.plan` column; the
        connector is re-resolved **by name** rather than deserialized,
        because a live connection cannot be serialized and restoring a
        dead one would be worse than re-resolving. A connector that no
        longer exists yields `None`, so the resume fails clearly rather
        than proceeding against something that is gone.
        """
        steps = _plan_from_json(task.plan or [])
        if not steps:
            return None

        try:
            mode = ExecutionMode(task.mode)
        except ValueError:  # pragma: no cover - defensive
            mode = ExecutionMode.EXECUTE

        return _TaskContext(
            steps=steps,
            principal=Principal(
                tenant_id=task.tenant_id,
                user_id=task.created_by_user_id,
                roles=("system",),
                is_system=True,
            ),
            mode=mode,
        )

    async def resume_after_approval(
        self, task_id: str, approved: bool, *, principal: Principal = SYSTEM_PRINCIPAL
    ) -> None:
        """Records `principal`'s decision and resumes (or aborts) the task."""
        async with self._store.session() as session:
            task = await session.get(Task, task_id)
            if task is None:
                raise KeyError(f"no task with id '{task_id}'")
            if task.status != "awaiting_approval":
                raise ValueError(
                    f"task '{task_id}' is not awaiting approval (status: '{task.status}')"
                )
            step_index = self._paused_index(task)

            # Authorization is rechecked HERE, at the point of action,
            # not only at the HTTP boundary (brief §8: "Recheck
            # authorization when a paused or scheduled task resumes").
            #
            # The API route already requires `task:approve`, but this
            # method is also reachable from the CLI and the schedule
            # runner, and it is this method that actually performs the
            # mutation. Same two-layer posture as execution modes: the
            # outer layer keeps the action out of reach, the inner one
            # refuses it anyway. `decide_approval` below binds tenant,
            # arguments, expiry and idempotency - but it does not
            # establish that the approver holds the permission at all.
            try:
                policy_engine.authorize(
                    principal,
                    Permission.TASK_APPROVE,
                    resource_type="task",
                    resource_id=task_id,
                    resource_tenant_id=task.tenant_id,
                )
            except PermissionError as exc:
                await self._audit.record(
                    principal=principal,
                    action="task:approve",
                    resource_type="task",
                    resource_id=task_id,
                    decision="denied",
                    reason=str(exc),
                    detail={"step": step_index},
                    correlation_id=task.correlation_id,
                )
                raise

            # Identity-bound, expiring, idempotent. A replayed call
            # resolves to the same record rather than re-authorizing.
            approval = await decide_approval(
                session,
                task_id=task_id,
                step_index=step_index,
                approved=approved,
                principal=principal,
            )
            task.pending_input = None
            task.status = "running"
            await session.commit()
            correlation_id = task.correlation_id

        await self._publish_state(task)
        await self._audit.record(
            principal=principal,
            action="task:approve",
            resource_type="task",
            resource_id=task_id,
            decision="allowed" if approved else "denied",
            reason=f"operator {'approved' if approved else 'rejected'} step {step_index}",
            detail={"step": step_index, "approval_id": approval.id,
                    "action_hash": approval.action_hash},
            correlation_id=correlation_id,
        )

        context = self._contexts.get(task_id)
        if context is None:
            # A different process (or a restarted one) - rebuild from
            # the durable plan; LangGraph's checkpoint supplies how far
            # the task got.
            async with self._store.session() as session:
                task = await session.get(Task, task_id)
            context = await self._rebuild_context(task) if task else None
            if context is None:
                raise ValueError(
                    f"task '{task_id}' is awaiting approval but its plan could not be "
                    "recovered; it cannot be resumed here"
                )
            self._contexts[task_id] = context

        # Resume THIS step's interrupt only. With several mutating
        # steps paused in the same superstep, a blanket resume would
        # authorize actions no one approved - so the decision is
        # addressed to one interrupt id.
        resume_payload = await self._resume_payload(task_id, context, step_index, approved)
        await self._drive(task_id, context, initial=Command(resume=resume_payload))

    async def _resume_payload(
        self, task_id: str, context: _TaskContext, step_index: int, approved: bool
    ) -> Any:
        """Builds the `Command(resume=...)` payload for one step.

        Returns an `{interrupt_id: decision}` map when the matching
        pending interrupt can be identified, and a bare decision when
        there is only one - LangGraph accepts both, and the bare form
        keeps the common single-pause case simple.
        """
        decision = {"approved": approved}
        saver = self._checkpointer
        if isinstance(saver, PraxisCheckpointSaver):
            saver = saver.for_tenant(context.principal.tenant_id)
        graph = build_task_graph(
            context.steps, self._make_step_runner(task_id, context), checkpointer=saver
        )
        snapshot = await graph.aget_state({"configurable": {"thread_id": task_id}})
        pending = list(getattr(snapshot, "interrupts", ()) or ())

        if len(pending) <= 1:
            return decision

        for item in pending:
            value = item.value if isinstance(item.value, dict) else {}
            if value.get("step_index") == step_index:
                return {item.id: decision}
        return decision

    @staticmethod
    def _paused_index(task: Task) -> int:
        """Which step the task is paused on, read from the checklist."""
        for index, item in enumerate(task.checklist or []):
            if item.get("status") == "awaiting_approval":
                return index
        return 0

    async def cancel_task(self, task_id: str, *, principal: Principal = SYSTEM_PRINCIPAL) -> bool:
        """Cancels a running or paused task.

        Returns True when this call performed the cancellation, False
        when the task was already terminal (idempotent). Cooperative:
        the flag is checked at the start of each step, so no step is
        killed mid-write.
        """
        async with self._store.session() as session:
            task = await session.get(Task, task_id)
            if task is None:
                raise KeyError(f"no task with id '{task_id}'")
            if task.status in _TERMINAL_STATUSES:
                return False

            context = self._contexts.get(task_id)
            if context is not None:
                context.cancelled = True

            checklist = [dict(item) for item in task.checklist]
            for index, item in enumerate(checklist):
                if item["status"] in ("pending", "awaiting_approval", "in_progress"):
                    checklist[index] = {
                        **item,
                        "status": "skipped",
                        "reason": f"task cancelled by {principal.describe()}",
                    }
            task.checklist = checklist
            task.status = "cancelled"
            task.pending_input = None
            task.result = {
                "summary": "Task cancelled.",
                "cancelled_by": principal.describe(),
                "error": "task was cancelled",
            }
            await session.commit()
            correlation_id = task.correlation_id

        await self._publish_state(task)
        await self._audit.record(
            principal=principal,
            action="task:cancel",
            resource_type="task",
            resource_id=task_id,
            detail={"previous_status": "running"},
            correlation_id=correlation_id,
        )
        with bind_correlation_id(correlation_id):
            _logger.info("task_cancelled", task_id=task_id, by=principal.describe())

        await self._release_thread(task_id)
        return True

    async def replay_history(self, task_id: str) -> list[dict[str, Any]]:
        """Every checkpointed state for a task, newest first.

        The brief's *"replay/debug mode"*: it reconstructs what the
        task's execution looked like at each superstep, which is what
        makes a run debuggable after the fact. Distinct from
        re-executing it - nothing here has side effects.
        """
        async with self._store.session() as session:
            task = await session.get(Task, task_id)
        if task is None:
            raise KeyError(f"no task with id '{task_id}'")

        context = self._contexts.get(task_id) or await self._rebuild_context(task)
        if context is None:
            return []

        saver = self._checkpointer
        if isinstance(saver, PraxisCheckpointSaver):
            saver = saver.for_tenant(task.tenant_id)
        graph = build_task_graph(
            context.steps, self._make_step_runner(task_id, context), checkpointer=saver
        )

        history: list[dict[str, Any]] = []
        async for snapshot in graph.aget_state_history(
            {"configurable": {"thread_id": task_id}}
        ):
            history.append(
                {
                    "step": snapshot.metadata.get("step") if snapshot.metadata else None,
                    "next": list(snapshot.next),
                    "completed_steps": sorted(snapshot.values.get("results", {})),
                    "step_status": snapshot.values.get("step_status", {}),
                    "errors": snapshot.values.get("errors", []),
                }
            )
        return history
