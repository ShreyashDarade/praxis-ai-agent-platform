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

**An honest gap, not a heuristic (see the phase report)**: synthesis
triggered from here is never given a `connector` - the Planner (§7)
doesn't currently produce any signal for "this step is about connector
X," so there is no principled way for the Orchestrator to guess one
without inventing a fake heuristic. `CapabilityFactory.synthesize`
still fully supports connector-aware synthesis (schema introspection +
caching, lineage edges) for a caller that *does* know the connector -
see `praxis.agents.capability_factory`'s own tests - this is specifically
about what the Orchestrator itself can determine from a bare
`PlanStep`, which today is nothing.

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
from praxis.core.events import TaskEventBus, task_event_bus, task_state_snapshot
from praxis.core.execution_graph import PlanStep, compute_levels, resolve_args
from praxis.core.exceptions import SandboxViolationError, SynthesisValidationError
from praxis.core.risk_policy import PendingInput, should_pause_for_approval
from praxis.memory.db import PostgresStore
from praxis.memory.models import Task
from praxis.observability.logging import bind_correlation_id
from praxis.observability.tracing import start_span

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


def _describe_step(step: PlanStep) -> str:
    return f"{step.skill_name}({step.args})"


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
        self._active: dict[str, _TaskState] = {}

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

    async def start_task(self, intent_text: str) -> str:
        async with self._store.session() as session:
            task = Task(intent_text=intent_text, status="planning", checklist=[])
            session.add(task)
            await session.commit()
            task_id = task.id
            correlation_id = task.correlation_id

        with bind_correlation_id(correlation_id):
            _logger.info("task_started", task_id=task_id, intent_text=intent_text)

        try:
            steps = await self._planner.plan(intent_text, self._skills.all_skills())
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

        async with self._store.session() as session:
            task = await session.get(Task, task_id)
            assert task is not None
            task.checklist = checklist
            task.status = "running"
            await session.commit()
        await self._publish_state(task)

        self._active[task_id] = _TaskState(
            steps=steps, levels=levels, known_urls=_extract_urls(intent_text)
        )
        await self._advance(task_id)
        return task_id

    async def resume_after_approval(self, task_id: str, approved: bool) -> None:
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
                    del self._active[task_id]
                    return

                skill = self._skills.get_skill(step.skill_name)  # guaranteed registered - checked before the pause
                resolved_args = state.pending_args[paused_index]
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
                        output = await skill.run(**resolved_args, known_urls=set(state.known_urls))
                except Exception as exc:  # noqa: BLE001 - a skill's own failure, not swallowed (spec §12)
                    _logger.error(
                        "skill_execution_failed", skill=skill.name, step=paused_index,
                        task_id=task_id, error=str(exc),
                    )
                    self._mark_item(task, paused_index, status="failed", reason=str(exc))
                    task.status = "failed"
                    task.result = {"error": f"step {paused_index} ('{step.skill_name}') failed: {exc}"}
                    await session.commit()
                    await self._publish_state(task)
                    del self._active[task_id]
                    return
                _logger.info("skill_execution_completed", skill=skill.name, step=paused_index, task_id=task_id)

                state.results[paused_index] = output
                self._merge_known_urls(state, output)
                state.paused_index = None
                self._mark_item(task, paused_index, status="completed")
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
                        outcome = await self._process_level(session, task, state, level_pos)
                        if outcome in ("paused", "failed"):
                            _logger.info("task_advance_stopped", task_id=task_id, outcome=outcome)
                            if outcome == "failed":
                                del self._active[task_id]
                            return
                        state.next_level = level_pos + 1

                    task.status = "completed"
                    task.result = self._build_result(state, task.checklist)
                    await session.commit()
                await self._publish_state(task)
                _logger.info("task_completed", task_id=task_id)
        del self._active[task_id]

    async def _synthesize_missing_skill(
        self,
        session: AsyncSession,
        task: Task,
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
        _logger.info("capability_synthesis_triggered", skill=step.skill_name, step=index, task_id=task.id)
        try:
            return await self._capability_factory.synthesize(
                need_description=need_description, task_id=task.id
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
                skill = await self._synthesize_missing_skill(session, task, index, step, resolved_args)
                if skill is None:
                    return "failed"

            if should_pause_for_approval(skill):
                pending = PendingInput(
                    kind="approval",
                    detail=(
                        f"approve mutating skill '{step.skill_name}' "
                        f"(step {index}) with args {resolved_args}?"
                    ),
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
                    result = await skill.run(**args, known_urls=set(state.known_urls))
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
