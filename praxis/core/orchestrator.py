# praxis/core/orchestrator.py
"""The Orchestrator (spec §3, §7, §8, §9, §14): turns a `Task`'s intent
into a materialized checklist and drives it through the execution
graph, pausing for approval on any `mutating` step.

**Capability synthesis (Phase 6, spec §3's "NO MATCH -> synthesize")**:
when a `PlanStep` names a skill that isn't registered, and a
`CapabilityFactory` was actually supplied (see `capability_factory`
below), the Orchestrator asks it to synthesize one - real sandbox
validation and all (`praxis.agents.capability_factory`) - before giving
up. Per spec §8/§20, synthesis itself is never gated behind approval
(only a *mutating* skill's actual execution is, exactly the same as any
hand-written skill); if synthesis fails validation
(`SynthesisValidationError`), the step - and the task - fails with the
real detail, never silently. When no `CapabilityFactory` is supplied
(the default - e.g. every pre-Phase-6 test in this file, and any
deployment without Docker reachable), the old behavior holds exactly:
an unregistered skill fails the task immediately, with a clear message.

**Connector-aware synthesis (Phase 11, spec §16.2)**: `start_task` now
takes an optional `connector_name` - when a caller (e.g. `POST /intent`'s
`connector` field) names one, it's resolved through the
`ConnectorRegistry` this Orchestrator was constructed with, *once*, up
front, and the resolved `Connector` object is threaded through this
task's whole run so that if/when a plan step needs synthesis,
`capability_factory.synthesize(connector=...)` gets the real object -
real schema introspection, real lineage edges, exactly like
`praxis.agents.capability_factory`'s own connector-aware tests already
exercise, just reached from here instead of only from a caller that
constructs the Factory directly. An unknown `connector_name` is a
clear, typed failure (the task fails with the real detail), never a
silent ignore - this Orchestrator has no way to tell "the caller made a
typo" apart from "proceed without one" otherwise. A bare `PlanStep`
still carries no *per-step* connector signal (the Planner decomposes
intents in general, not just connector-shaped ones) - one connector per
task, resolved once, is the real, principled scope this phase's
walkthrough actually needs, not a heuristic guess at a finer grain
nothing in this codebase asks for yet.

**Short-term vs. long-term state (spec §9)**: the execution graph's
live state (`_TaskState` - resolved levels, in-flight args, partial
results) is genuinely short-term - held only in `Orchestrator._active`,
in-memory, keyed by task id, and dropped the moment a task reaches a
terminal or paused state that doesn't need it further. The long-term,
durable record - `checklist`, `pending_input`, `result`, `status` - is
what actually lives in the `tasks` table and is all `GET /tasks/{id}`
ever reads. This is a deliberate MVP trade-off: a process restart while
a task is `awaiting_approval` loses that task's resumability (the DB
still shows exactly what it was waiting on, but `resume_after_approval`
against a fresh process would find no in-memory state) - acceptable for
this phase, and the natural place a later phase would add durable
graph-state persistence if that trade-off ever needs revisiting.

**Why a level can't "partially" pause**: within one dependency level,
steps are scanned in index order; the moment a step needs approval (or
names an unregistered skill), the Orchestrator stops *before starting
any step in that level* - even ones that would otherwise run
concurrently alongside it - rather than running some now and leaving
others stranded mid-level. Once nothing in the level needs to pause,
every remaining step in it runs concurrently via `asyncio.gather`,
exactly per spec §8.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from praxis.agents.planner import Planner
from praxis.agents.skill import Skill
from praxis.cache.memory_cache import InMemoryCache
from praxis.config import Settings
from praxis.core.events import TaskEventBus, task_event_bus, task_state_snapshot
from praxis.core.execution_graph import PlanStep, compute_levels, resolve_args
from praxis.core.execution_mode import (
    ExecutionMode,
    MutationNotPermittedError,
    visible_skills,
)
from praxis.core.exceptions import SandboxViolationError, SynthesisValidationError
from praxis.core.interfaces import Connector, GraphStore
from praxis.core.risk_policy import PendingInput, should_pause_for_approval
from praxis.memory.db import PostgresStore
from praxis.memory.models import Task
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
from praxis.security.principal import SYSTEM_PRINCIPAL, Principal

_logger = structlog.get_logger(__name__)

# Phase 9 (spec §6.1's "prior-context-only fetch"): the Orchestrator is
# the one place that decides which URLs are legitimate fetch targets for
# `web_read`/`web_crawl` - never the skill/connector itself, and never
# an LLM's own freshly-generated output. `_extract_urls` regex-extracts
# every http(s) URL substring out of a piece of *already-validated*
# text (the task's own intent text, or an earlier step's own result) -
# deliberately permissive matching is safe here specifically because
# every candidate this ever runs against already qualifies as "prior
# context" by definition; over-matching only ever widens what a later
# fetch is allowed to target with text that was already trusted input,
# it never admits anything an LLM invented on its own. Trailing prose
# punctuation commonly following a URL in natural-language text (a
# period, comma, closing paren/bracket/quote) is stripped, since it's
# never actually part of the URL.
_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")


def _extract_urls(text: str) -> set[str]:
    return {match.rstrip(".,;:!?)]}'\"") for match in _URL_RE.findall(text)}


class SkillRegistryLike(Protocol):
    """What the Orchestrator needs from a skill registry - satisfied by
    the `praxis.agents.skill_registry` module itself (mirrors
    `praxis.ingestion.pipeline.ParserRegistryLike`), or any stand-in
    exposing the same two functions in a test."""

    def get_skill(self, name: str) -> Skill: ...

    def all_skills(self) -> list[Skill]: ...


class CapabilityFactoryLike(Protocol):
    """What the Orchestrator needs from a Capability Factory - satisfied
    by the real `praxis.agents.capability_factory.CapabilityFactory`, or
    any stand-in exposing the same one method in a test (mirrors
    `SkillRegistryLike` above)."""

    async def synthesize(
        self, need_description: str, *, connector: Any | None = None, task_id: str | None = None
    ) -> Skill: ...


class ConnectorRegistryLike(Protocol):
    """What the Orchestrator needs from a connector registry (Phase 11,
    spec §16.2) - satisfied by the real
    `praxis.connectors.registry.ConnectorRegistry`, or any stand-in
    exposing this one method in a test (mirrors `SkillRegistryLike`/
    `CapabilityFactoryLike` above)."""

    def get(self, name: str) -> Connector: ...


@dataclass
class _TaskState:
    """In-memory execution-graph state for one in-flight task (see module docstring).

    `tool_cache` (spec §11's tool-result cache scope, "idempotent
    read-only calls within a single task") is a fresh `InMemoryCache`
    per task - this state object's own lifetime already IS one task's
    execution (created in `start_task`, dropped in `_advance`/
    `resume_after_approval` on a terminal outcome per the module
    docstring's "short-term vs. long-term state" note), so a fresh
    instance here naturally means the cache never outlives, or leaks
    across, one task run.
    """

    steps: list[PlanStep]
    levels: list[list[int]]
    results: dict[int, Any] = field(default_factory=dict)
    next_level: int = 0
    paused_index: int | None = None
    pending_args: dict[int, dict[str, Any]] = field(default_factory=dict)
    tool_cache: InMemoryCache = field(default_factory=InMemoryCache)
    # Phase 9 (spec §6.1): seeded from the task's own intent text in
    # `start_task`, grown by `_merge_known_urls` every time a step's
    # result is recorded below - passed into every skill call as the
    # `known_urls` kwarg (see `_run_one` and `resume_after_approval`),
    # which is exactly what `web_read`/`web_crawl` need to enforce
    # "may only fetch a URL that already appears in validated task
    # input or a prior tool result."
    known_urls: set[str] = field(default_factory=set)
    # Phase 11 (spec §16.2): the real `Connector` this task's
    # `connector_name` (if any) resolved to at `start_task` time - `None`
    # for the (still overwhelmingly common) case where a task names no
    # connector at all. Threaded into `capability_factory.synthesize()`
    # by `_synthesize_missing_skill` below whenever this task's plan
    # needs a fresh capability synthesized.
    connector: Connector | None = None
    # Phase 12: who this task runs as, and therefore which tenant every
    # row it writes (graph edges, artifacts, vector chunks) belongs to.
    principal: Principal = SYSTEM_PRINCIPAL
    # Phase 13: the execution mode, re-checked before every step rather
    # than only at plan time (see `praxis.core.execution_mode`).
    mode: ExecutionMode = ExecutionMode.EXECUTE
    # Phase 14: cooperative cancellation. `cancel_task` sets this; the
    # level loop checks it between levels and refuses to start further
    # work. A step already in flight is allowed to finish - killing a
    # half-applied external mutation mid-write would be worse than one
    # extra completed step.
    cancelled: bool = False
    cancelled_by: str | None = None


def _expected_output_keys(all_steps: list[PlanStep], step_index: int) -> set[str]:
    """Scans every step's args for a `"$<step_index>.<key>"` reference -
    i.e. what output key(s) a later step in this same plan already
    expects `step_index`'s (about-to-be-synthesized) skill to return.

    Closes a real gap that would otherwise exist purely because the
    Planner (which invents this key name, when it names a step whose
    skill doesn't exist yet - spec §7/Phase 11's `plan_intent@v2`) and
    the Capability Factory (which is what actually decides the
    synthesized skill's real declared `outputs`) are two independent LLM
    calls with no shared state between them - without this, a
    downstream step could reference an output key the synthesized skill
    never actually declared, and `resolve_args` would raise a `KeyError`
    at run time for reasons neither LLM call could see coming. Feeding
    the real, already-committed key name(s) back into the synthesis
    `need_description` (see `_synthesize_missing_skill`) means the
    Factory's LLM call knows exactly what shape is already expected of
    it.
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


# A task in one of these will never transition again - `cancel_task`
# treats a duplicate cancel of one of these as a no-op rather than an
# error, and `praxis.api.routes.tasks`'s WS stream uses the same set to
# know when to stop forwarding.
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


def _tool_result_cache_key(skill_name: str, args: dict[str, Any]) -> str:
    """Cache key for the tool-result scope: identical `skill_name` +
    `args` (a read-only skill call is idempotent per spec §11) hash to
    the same key, regardless of the args dict's key insertion order."""
    payload = json.dumps({"skill": skill_name, "args": args}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    ) -> None:
        self._store = store
        self._planner = planner
        self._skills = skills
        self._capability_factory = capability_factory
        # Defaults to the process-global bus (`praxis.core.events.
        # task_event_bus`) - the same instance the `WS /tasks/{id}/
        # stream` route subscribes to (praxis.api.main) - so every
        # pre-existing caller (every test in this file included)
        # continues to work unchanged, with live events simply
        # published nowhere in particular unless something subscribes.
        # A test wanting an isolated bus can still pass its own.
        self._event_bus = event_bus if event_bus is not None else task_event_bus
        # Phase 11 (spec §16.2): additive, defaults to `None` exactly
        # like `capability_factory` above - every pre-Phase-11 caller
        # (every test in this file that doesn't pass one) gets byte-for-
        # byte the old behavior: a task naming no `connector_name` never
        # touches this at all, and a task that does with no registry
        # configured fails clearly (see `start_task`) rather than
        # silently proceeding as if no connector had been named.
        self._connector_registry = connector_registry
        # Phase 11 (spec §16.1 step 8's "the lineage graph records every
        # skill/tool used" - a general property of every task, not
        # something reserved for freshly-synthesized skills, which is
        # all Phase 6's `CapabilityFactory._register()` records on its
        # own). Additive and optional, same posture as every dependency
        # above: `None` (the default) means lineage recording is simply
        # skipped, byte-for-byte the pre-Phase-11 behavior for every
        # test/caller that doesn't pass one.
        self._graph_store = graph_store
        # Phase 12: the audit trail. Optional and defaulting to one
        # built over this Orchestrator's own store, so every caller
        # (including every pre-Phase-12 test) gets real auditing with no
        # constructor change, while a test wanting to assert on audit
        # rows can inject its own.
        self._audit = audit_logger if audit_logger is not None else AuditLogger(store)
        # Resolved lazily-but-once: reading Settings() here would make
        # constructing an Orchestrator fail on an unconfigured
        # environment, which several tests deliberately do.
        self._approval_ttl_seconds = approval_ttl_seconds
        self._active: dict[str, _TaskState] = {}

    @property
    def _approval_ttl(self) -> int:
        if self._approval_ttl_seconds is not None:
            return self._approval_ttl_seconds
        try:
            return Settings().approval_ttl_seconds
        except Exception:  # noqa: BLE001 - an unconfigured env must not break a pause
            return 3600

    async def _record_skill_used(self, task_id: str, skill_name: str, tenant_id: str) -> None:
        if self._graph_store is None:
            return
        await self._graph_store.add_edge(
            source=task_id, relation="used_skill", target=skill_name, tenant_id=tenant_id
        )

    async def _publish_state(self, task: Task) -> None:
        """Publishes a live snapshot of `task` to any `WS /tasks/{id}/
        stream` subscriber (spec §14) - called right after every commit
        below that actually changes `status`/`checklist`/`pending_input`/
        `result`, never as a periodic heartbeat. Safe to read `task`'s
        attributes here, straight off the same ORM object just
        committed, with no extra query - `PostgresStore`'s session
        factory sets `expire_on_commit=False` (praxis/memory/db.py)
        specifically so this kind of post-commit read never needs one.
        """
        await self._event_bus.publish(task.id, task_state_snapshot(task))

    # ------------------------------------------------------------------ #
    # Checklist helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _mark_item(task: Task, index: int, *, status: str, reason: str | None = None) -> None:
        # Reassigns a fresh list (rather than mutating task.checklist[index]
        # in place) so SQLAlchemy's change tracking on the plain `JSON`
        # column actually sees the write - in-place mutation of a JSON
        # column's Python value is invisible to the ORM without
        # `sqlalchemy.ext.mutable`, which this schema deliberately
        # doesn't add (see praxis/memory/models.py).
        checklist = [dict(item) for item in task.checklist]
        checklist[index] = {**checklist[index], "status": status, "reason": reason}
        task.checklist = checklist

    @staticmethod
    def _merge_known_urls(state: _TaskState, value: Any) -> None:
        """Grows `state.known_urls` from a just-recorded step result
        (spec §6.1: "... or a prior tool result"). Called at every point
        in this file that assigns into `state.results[...]` - a cache
        hit, a freshly-run step, or an alias copying another step's
        outcome - so a later step's `known_urls` kwarg always reflects
        every URL surfaced by any step that has completed so far,
        regardless of which of those three paths produced it."""
        state.known_urls |= _extract_urls(str(value))

    @staticmethod
    def _build_result(state: _TaskState, checklist: list[dict[str, Any]]) -> dict[str, Any]:
        completed = sum(1 for item in checklist if item["status"] == "completed")
        steps_summary = [
            {
                "step": index,
                "skill": step.skill_name,
                "status": checklist[index]["status"],
                "output": state.results.get(index),
            }
            for index, step in enumerate(state.steps)
        ]
        return {
            "summary": f"Task completed: {completed}/{len(checklist)} step(s) completed.",
            "steps": steps_summary,
        }

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
        """Plans `intent_text` and drives it to completion (or a pause).

        `principal` decides which tenant every row this task writes
        belongs to, and is recorded on the task itself. `mode` decides
        how far the task may go - and is enforced twice, once by
        narrowing the skills the Planner can even see and again before
        each step actually runs (see `praxis.core.execution_mode`).
        """
        async with self._store.session() as session:
            task = Task(
                tenant_id=principal.tenant_id,
                created_by_user_id=principal.user_id,
                intent_text=intent_text,
                status="planning",
                mode=mode.value,
                checklist=[],
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

        # Phase 11 (spec §16.2): resolved once, up front - before the
        # (real, potentially expensive) Planner call - so a caller-typo'd
        # or unconfigured connector name fails fast and clearly, exactly
        # like a Planner failure below, rather than silently proceeding
        # with no connector and only surfacing the mismatch much later,
        # deep inside a confusing synthesis failure.
        connector: Connector | None = None
        if connector_name is not None:
            if self._connector_registry is None:
                message = (
                    f"task requested connector '{connector_name}' but this Orchestrator has "
                    "no connector registry configured"
                )
                with bind_correlation_id(correlation_id):
                    _logger.error(
                        "task_connector_resolution_failed", task_id=task_id,
                        connector_name=connector_name, error=message,
                    )
                await self._fail_task(task_id, message, checklist=[])
                return task_id
            try:
                connector = self._connector_registry.get(connector_name)
            except KeyError as exc:
                with bind_correlation_id(correlation_id):
                    _logger.error(
                        "task_connector_resolution_failed", task_id=task_id,
                        connector_name=connector_name, error=str(exc),
                    )
                await self._fail_task(task_id, f"unknown connector '{connector_name}': {exc}", checklist=[])
                return task_id

        try:
            # Enforcement layer 1 (spec: Prompt §8's "Plan mode must
            # REMOVE or deny mutating tools"): under any mode that
            # cannot mutate, the Planner is never even shown a mutating
            # skill, so it cannot produce a plan naming one.
            steps = await self._planner.plan(
                intent_text,
                visible_skills(self._skills.all_skills(), mode),
                connector=connector,
            )
        except Exception as exc:  # noqa: BLE001 - a planning failure must fail the task, not crash the caller
            with bind_correlation_id(correlation_id):
                _logger.error("task_planning_failed", task_id=task_id, error=str(exc))
            await self._fail_task(task_id, f"planning failed: {exc}", checklist=[])
            return task_id

        checklist = [
            {"content": _describe_step(step), "status": "pending", "reason": None} for step in steps
        ]

        try:
            levels = compute_levels(steps)
        except ValueError as exc:
            await self._fail_task(task_id, f"invalid plan: {exc}", checklist=checklist)
            return task_id

        # `plan_only`: the plan IS the deliverable. The task completes
        # immediately with the full plan as its result and nothing is
        # ever executed - not "executed with mutations skipped", which
        # would be a different and much weaker guarantee.
        if not mode.allows_execution:
            async with self._store.session() as session:
                task = await session.get(Task, task_id)
                assert task is not None
                task.checklist = checklist
                task.status = "completed"
                task.result = {
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
                    "levels": levels,
                }
                await session.commit()
            await self._publish_state(task)
            with bind_correlation_id(correlation_id):
                _logger.info("task_plan_only_completed", task_id=task_id, steps=len(steps))
            return task_id

        async with self._store.session() as session:
            task = await session.get(Task, task_id)
            assert task is not None
            task.checklist = checklist
            task.status = "running"
            await session.commit()
        await self._publish_state(task)

        self._active[task_id] = _TaskState(
            steps=steps,
            levels=levels,
            known_urls=_extract_urls(intent_text),
            connector=connector,
            principal=principal,
            mode=mode,
        )
        await self._advance(task_id)
        return task_id

    async def cancel_task(self, task_id: str, *, principal: Principal = SYSTEM_PRINCIPAL) -> bool:
        """Cancels a running or paused task (Prompt §1).

        Returns True when this call performed the cancellation, False
        when the task was already in a terminal state (idempotent - a
        duplicate cancel is not an error).

        Cooperative by design: the flag is checked between levels and
        before each step starts, so no step is ever killed mid-write.
        Any in-flight level finishes, then the task stops.
        """
        async with self._store.session() as session:
            task = await session.get(Task, task_id)
            if task is None:
                raise KeyError(f"no task with id '{task_id}'")

            if task.status in _TERMINAL_STATUSES:
                return False

            state = self._active.get(task_id)
            if state is not None:
                state.cancelled = True
                state.cancelled_by = principal.describe()

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
            partial = (
                self._build_result(state, task.checklist)
                if state is not None
                else {"summary": "Task cancelled before any execution state existed."}
            )
            task.result = {
                **partial,
                "cancelled_by": principal.describe(),
                "error": "task was cancelled",
            }
            await session.commit()

        await self._publish_state(task)
        await self._audit.record(
            principal=principal,
            action="task:cancel",
            resource_type="task",
            resource_id=task_id,
            detail={"previous_status": "running"},
        )
        with bind_correlation_id(task.correlation_id):
            _logger.info("task_cancelled", task_id=task_id, by=principal.describe())
        self._active.pop(task_id, None)
        return True

    async def resume_after_approval(
        self, task_id: str, approved: bool, *, principal: Principal = SYSTEM_PRINCIPAL
    ) -> None:
        """Records `principal`'s decision and resumes (or aborts) the task.

        Two security controls run here before anything executes
        (Prompt §8): the decision is written as an identity-bound,
        expiring, idempotent `ApprovalRecord`, and the approval's
        `action_hash` is re-verified against the arguments actually
        about to run - so an approval can never be replayed against
        different arguments than the human saw.
        """
        async with self._store.session() as session:
            # DB existence/status checked first (KeyError for a genuinely
            # unknown task -> 404; ValueError for a known task in the
            # wrong state -> 409, e.g. already completed) - only once
            # both hold do we consult the in-memory execution state,
            # whose own absence (e.g. a process restart) is a *different*
            # 409 (spec §12: never conflate "doesn't exist" with "can't
            # be resumed right now").
            task = await session.get(Task, task_id)
            if task is None:
                raise KeyError(f"no task with id '{task_id}'")
            if task.status != "awaiting_approval":
                raise ValueError(
                    f"task '{task_id}' is not awaiting approval (status: '{task.status}')"
                )

            state = self._active.get(task_id)
            if state is None or state.paused_index is None:
                raise ValueError(
                    f"task '{task_id}' is marked awaiting_approval in storage but has no "
                    "in-memory execution state in this process (likely a process restart); "
                    "it cannot be resumed here"
                )

            with bind_correlation_id(task.correlation_id):
                _logger.info("resume_after_approval_requested", task_id=task_id, approved=approved)

                paused_index = state.paused_index
                step = state.steps[paused_index]

                # Record the decision as an identity-bound, expiring,
                # idempotent approval record BEFORE acting on it. A
                # replayed call resolves to the same record (no second
                # authorization); an expired one raises rather than
                # executing on a stale human decision.
                approval = await decide_approval(
                    session,
                    task_id=task_id,
                    step_index=paused_index,
                    approved=approved,
                    principal=principal,
                )
                await self._audit.record(
                    principal=principal,
                    action="task:approve",
                    resource_type="task",
                    resource_id=task_id,
                    decision="allowed" if approved else "denied",
                    reason=f"operator {'approved' if approved else 'rejected'} step {paused_index}",
                    detail={
                        "step": paused_index,
                        "skill": step.skill_name,
                        "approval_id": approval.id,
                        "action_hash": approval.action_hash,
                    },
                    correlation_id=task.correlation_id,
                )

                if not approved:
                    self._mark_item(task, paused_index, status="skipped", reason="rejected by operator")
                    task.status = "failed"
                    task.pending_input = None
                    task.result = {
                        "error": (
                            f"step {paused_index} ('{step.skill_name}') was rejected by the "
                            "operator; task aborted"
                        )
                    }
                    _logger.info("task_rejected_by_operator", task_id=task_id, step=paused_index)
                    await session.commit()
                    await self._publish_state(task)
                    self._active.pop(task_id, None)
                    return

                skill = self._skills.get_skill(step.skill_name)  # guaranteed registered - checked before the pause
                resolved_args = state.pending_args[paused_index]

                # Re-verify authorization at resume time (Prompt §8:
                # "Recheck authorization when a paused or scheduled task
                # resumes"). The binding check raises
                # `ApprovalBindingError` if the arguments drifted from
                # the ones the operator actually approved.
                await verify_approval_binding(
                    session,
                    task_id=task_id,
                    step_index=paused_index,
                    skill_name=step.skill_name,
                    args=resolved_args,
                )

                self._mark_item(task, paused_index, status="in_progress")
                task.status = "running"
                task.pending_input = None
                await session.commit()
                await self._publish_state(task)

                _logger.info("skill_execution_started", skill=skill.name, step=paused_index, task_id=task_id)
                try:
                    with start_span("skill.execute", skill=skill.name, step=paused_index):
                        # `known_urls` (spec §6.1) is injected here, out
                        # of band from `resolved_args` - never merged
                        # into it, so it never perturbs `pending_args`
                        # (already captured before the pause) or any
                        # tool-result cache key computed from it.
                        output = await skill.run(
                            **resolved_args,
                            known_urls=set(state.known_urls),
                            tenant_id=state.principal.tenant_id,
                        )
                        output = validate_skill_output(skill.name, skill.outputs, output)
                except Exception as exc:  # noqa: BLE001 - a skill's own failure, not swallowed (spec §12)
                    _logger.error(
                        "skill_execution_failed", skill=skill.name, step=paused_index,
                        task_id=task_id, error=str(exc),
                    )
                    self._mark_item(task, paused_index, status="failed", reason=str(exc))
                    task.status = "failed"
                    # Preserve whatever steps *did* complete before this
                    # one failed (per _build_result, keyed off state.results
                    # and the checklist's current per-item status) rather
                    # than discarding them - a failed task's real, already-
                    # gathered data (e.g. a fetched metric) stays visible
                    # via task.result["steps"], matching this system's
                    # "nothing silently vanishes" checklist philosophy
                    # (spec §7) instead of collapsing to a bare error string.
                    task.result = {
                        **self._build_result(state, task.checklist),
                        "error": f"step {paused_index} ('{step.skill_name}') failed: {exc}",
                    }
                    await session.commit()
                    await self._publish_state(task)
                    self._active.pop(task_id, None)
                    return
                _logger.info("skill_execution_completed", skill=skill.name, step=paused_index, task_id=task_id)

                state.results[paused_index] = output
                self._merge_known_urls(state, output)
                state.paused_index = None
                self._mark_item(task, paused_index, status="completed")
                await self._record_skill_used(task_id, skill.name, state.principal.tenant_id)
                await session.commit()
                await self._publish_state(task)

        await self._advance(task_id)

    # ------------------------------------------------------------------ #
    # Internal execution-graph driver
    # ------------------------------------------------------------------ #

    async def _fail_task(self, task_id: str, message: str, *, checklist: list[dict[str, Any]]) -> None:
        async with self._store.session() as session:
            task = await session.get(Task, task_id)
            assert task is not None
            task.checklist = checklist
            task.status = "failed"
            task.result = {"error": message}
            await session.commit()
        await self._publish_state(task)

    async def _advance(self, task_id: str) -> None:
        state = self._active[task_id]
        async with self._store.session() as session:
            task = await session.get(Task, task_id)
            assert task is not None

            with bind_correlation_id(task.correlation_id):
                with start_span("orchestrator.advance", task_id=task_id):
                    for level_pos in range(state.next_level, len(state.levels)):
                        # Cooperative cancellation checkpoint: checked
                        # between levels, so a cancel lands promptly
                        # without ever interrupting a step mid-write.
                        if state.cancelled:
                            _logger.info(
                                "task_advance_stopped", task_id=task_id, outcome="cancelled"
                            )
                            self._active.pop(task_id, None)
                            return
                        outcome = await self._process_level(session, task, state, level_pos)
                        if outcome in ("paused", "failed"):
                            _logger.info("task_advance_stopped", task_id=task_id, outcome=outcome)
                            if outcome == "failed":
                                self._active.pop(task_id, None)
                            return
                        state.next_level = level_pos + 1

                    task.status = "completed"
                    task.result = self._build_result(state, task.checklist)
                    await session.commit()
                await self._publish_state(task)
                _logger.info("task_completed", task_id=task_id)
        self._active.pop(task_id, None)

    async def _synthesize_missing_skill(
        self,
        session: AsyncSession,
        task: Task,
        state: _TaskState,
        index: int,
        step: PlanStep,
        resolved_args: dict[str, Any],
    ) -> Skill | None:
        """Called from `_process_level` the moment `step.skill_name` isn't
        registered. Returns the newly-synthesized `Skill` on success, or
        `None` after already marking the step/task `failed` (mirroring
        every other failure branch in `_process_level`, which returns
        `"failed"` right after calling this)."""
        message = f"no skill named '{step.skill_name}' is registered"
        if self._capability_factory is None:
            _logger.warning("skill_not_registered_no_factory", skill=step.skill_name, step=index)
            self._mark_item(task, index, status="skipped", reason=message)
            task.status = "failed"
            task.result = {"error": f"step {index}: {message}"}
            await session.commit()
            await self._publish_state(task)
            return None

        need_description = (
            f"A running plan needs a skill named '{step.skill_name}', to be called with "
            f"keyword arguments {resolved_args!r} - no such skill is registered yet. "
            "Synthesize a skill that fulfills this need, accepting exactly those keyword "
            "argument names."
        )
        # Phase 11: fold in whatever key(s) a later step in this same
        # plan already committed to reading from this step's result -
        # see `_expected_output_keys`'s own docstring for why this
        # matters (the Planner and the Factory are two independent LLM
        # calls with no shared state otherwise).
        expected_keys = _expected_output_keys(state.steps, index)
        if expected_keys:
            plural = len(expected_keys) != 1
            need_description += (
                f"\n\nA later step in this same plan will read this skill's result using "
                f"the key(s) {sorted(expected_keys)!r} (as \"$<this step's index>.<key>\") - "
                f"your declared `outputs` MUST include exactly {'those keys' if plural else 'that key'}, "
                f"and `run()` must return a dict containing {'them' if plural else 'it'}."
            )
        _logger.info("capability_synthesis_triggered", skill=step.skill_name, step=index, task_id=task.id)
        try:
            return await self._capability_factory.synthesize(
                need_description=need_description, connector=state.connector, task_id=task.id
            )
        except (SynthesisValidationError, SandboxViolationError) as exc:
            # Both are surfaced identically at the Orchestrator level
            # (spec §12): a validation failure exhausts its own bounded
            # retry inside the Factory; a sandbox violation (e.g. an
            # OOM-killed synthesis attempt) is never retried by the
            # Factory at all - either way, this step - and the task -
            # fails with the real detail, never silently.
            failure = f"{message}; capability synthesis also failed: {exc}"
            _logger.error(
                "capability_synthesis_failed", skill=step.skill_name, step=index,
                task_id=task.id, error=str(exc),
            )
            self._mark_item(task, index, status="failed", reason=failure)
            task.status = "failed"
            task.result = {"error": f"step {index}: {failure}"}
            await session.commit()
            await self._publish_state(task)
            return None

    async def _process_level(
        self, session: AsyncSession, task: Task, state: _TaskState, level_pos: int
    ) -> str:
        level = state.levels[level_pos]
        todo = [index for index in level if index not in state.results]
        if not todo:
            return "ok"

        # Tool-result cache (spec §11: "idempotent read-only calls within
        # a single task") - `state.tool_cache` lives exactly as long as
        # this one task. Only `read_only` skills are ever cached (a
        # `mutating` skill's call is never assumed idempotent, and each
        # one already pauses for its own individual approval). A cache
        # hit resolves the step immediately, with no execution at all.
        # Two-or-more steps sharing the same (skill, args) that are BOTH
        # still cache misses in this same level are deduplicated here
        # too (`aliases`) - only the first ("runner") actually calls
        # `skill.run()`; the rest copy its outcome once it completes,
        # rather than each racing to miss the cache concurrently.
        run_plan: dict[int, tuple[Skill, dict[str, Any]]] = {}
        cache_keys: dict[int, str] = {}
        runner_for_key: dict[str, int] = {}
        aliases: dict[int, int] = {}

        for index in todo:
            step = state.steps[index]
            resolved_args = resolve_args(step.args, state.results)
            try:
                skill = self._skills.get_skill(step.skill_name)
            except KeyError:
                skill = await self._synthesize_missing_skill(
                    session, task, state, index, step, resolved_args
                )
                if skill is None:
                    return "failed"

            # Enforcement layer 2 (see `praxis.core.execution_mode`): a
            # mutating skill that reached this plan despite layer 1 -
            # a freshly *synthesized* skill that declared itself
            # mutating is the real case, since it did not exist when
            # the Planner's visible set was computed - is refused or
            # simulated here, at the point of action.
            if skill.risk == "mutating" and not state.mode.allows_mutation:
                if state.mode.simulates_mutation:
                    simulated = {
                        "simulated": True,
                        "mode": state.mode.value,
                        "would_have_called": {"skill": skill.name, "args": resolved_args},
                        "note": (
                            "dry run: this mutating step was NOT executed; no external state "
                            "was changed"
                        ),
                    }
                    _logger.info(
                        "mutating_step_simulated",
                        skill=skill.name, step=index, task_id=task.id, mode=state.mode.value,
                    )
                    state.results[index] = simulated
                    self._mark_item(
                        task, index, status="completed", reason=f"simulated ({state.mode.value})"
                    )
                    continue

                error = MutationNotPermittedError(
                    (
                        f"step {index} ('{skill.name}') is a mutating skill, which execution "
                        f"mode '{state.mode.value}' does not permit "
                        f"({state.mode.describe()})"
                    ),
                    mode=state.mode,
                    skill_name=skill.name,
                )
                _logger.warning(
                    "mutating_step_refused",
                    skill=skill.name, step=index, task_id=task.id, mode=state.mode.value,
                )
                await self._audit.record(
                    principal=state.principal,
                    action="task:execute_step",
                    resource_type="task",
                    resource_id=task.id,
                    decision="denied",
                    reason=str(error),
                    detail={"step": index, "skill": skill.name, "mode": state.mode.value},
                    correlation_id=task.correlation_id,
                )
                self._mark_item(task, index, status="failed", reason=str(error))
                task.status = "failed"
                task.result = {"error": str(error)}
                await session.commit()
                await self._publish_state(task)
                return "failed"

            if should_pause_for_approval(skill):
                pending = PendingInput(
                    kind="approval",
                    detail=(
                        f"approve mutating skill '{step.skill_name}' "
                        f"(step {index}) with args {resolved_args}?"
                    ),
                )
                # Create the durable, argument-bound approval record at
                # pause time, so the exact action+args the operator is
                # about to be shown is what gets hashed and bound -
                # not something recomputed later from possibly-drifted
                # state (`praxis.security.approval`).
                await create_pending_approval(
                    session,
                    request=ApprovalRequest(
                        task_id=task.id,
                        step_index=index,
                        skill_name=step.skill_name,
                        args=resolved_args,
                    ),
                    tenant_id=state.principal.tenant_id,
                    ttl_seconds=self._approval_ttl,
                )
                self._mark_item(task, index, status="awaiting_approval")
                task.status = "awaiting_approval"
                task.pending_input = pending.to_dict()
                await session.commit()
                await self._publish_state(task)
                state.paused_index = index
                state.pending_args[index] = resolved_args
                return "paused"

            if skill.risk == "read_only":
                cache_key = _tool_result_cache_key(skill.name, resolved_args)
                cached_result = await state.tool_cache.get(cache_key)
                if cached_result is not None:
                    _logger.info("tool_result_cache_hit", skill=skill.name, step=index, task_id=task.id)
                    state.results[index] = cached_result
                    self._merge_known_urls(state, cached_result)
                    self._mark_item(task, index, status="completed")
                    await self._record_skill_used(task.id, skill.name, state.principal.tenant_id)
                    continue
                if cache_key in runner_for_key:
                    aliases[index] = runner_for_key[cache_key]
                    cache_keys[index] = cache_key
                    continue
                runner_for_key[cache_key] = index
                cache_keys[index] = cache_key

            run_plan[index] = (skill, resolved_args)

        for index in run_plan:
            self._mark_item(task, index, status="in_progress")
        await session.commit()
        await self._publish_state(task)

        async def _run_one(index: int) -> tuple[int, Any, BaseException | None]:
            skill, args = run_plan[index]
            _logger.info("skill_execution_started", skill=skill.name, step=index, task_id=task.id)
            try:
                with start_span("skill.execute", skill=skill.name, step=index):
                    # `known_urls` (spec §6.1) injected out of band here
                    # too - see the identical note in
                    # `resume_after_approval` above.
                    result = await skill.run(
                        **args,
                        known_urls=set(state.known_urls),
                        tenant_id=state.principal.tenant_id,
                    )
                    # Phase 13 (Prompt §8's "output validation"): a
                    # skill that didn't return what it declared has
                    # failed, and is reported as a step failure rather
                    # than silently propagating a result a downstream
                    # `$n.key` reference will later fail to resolve.
                    # Matters most for synthesized skills, where the
                    # implementation is LLM-authored.
                    result = validate_skill_output(skill.name, skill.outputs, result)
            except Exception as exc:  # noqa: BLE001 - captured per-step, not swallowed (spec §12)
                _logger.error(
                    "skill_execution_failed", skill=skill.name, step=index,
                    task_id=task.id, error=str(exc),
                )
                return index, None, exc
            _logger.info("skill_execution_completed", skill=skill.name, step=index, task_id=task.id)
            return index, result, None

        outcomes = await asyncio.gather(*(_run_one(index) for index in run_plan))

        any_failed = False
        for index, output, exc in outcomes:
            if exc is not None:
                self._mark_item(task, index, status="failed", reason=str(exc))
                any_failed = True
            else:
                state.results[index] = output
                self._merge_known_urls(state, output)
                self._mark_item(task, index, status="completed")
                await self._record_skill_used(
                    task.id, run_plan[index][0].name, state.principal.tenant_id
                )
                cache_key = cache_keys.get(index)
                if cache_key is not None:
                    await state.tool_cache.set(cache_key, output)

        # Propagate each runner's outcome to every alias sharing its
        # (skill, args) signature within this same level.
        for alias_index, runner_index in aliases.items():
            if runner_index in state.results:
                state.results[alias_index] = state.results[runner_index]
                self._merge_known_urls(state, state.results[alias_index])
                self._mark_item(task, alias_index, status="completed")
                await self._record_skill_used(
                    task.id, state.steps[alias_index].skill_name, state.principal.tenant_id
                )
            else:
                reason = next(
                    (str(exc) for index, _, exc in outcomes if index == runner_index and exc is not None),
                    "a duplicate step sharing this skill/args failed",
                )
                self._mark_item(task, alias_index, status="failed", reason=reason)
                any_failed = True
        await session.commit()
        await self._publish_state(task)

        if any_failed:
            task.status = "failed"
            task.result = {"error": "one or more steps failed", "checklist": task.checklist}
            await session.commit()
            await self._publish_state(task)
            return "failed"
        return "ok"
