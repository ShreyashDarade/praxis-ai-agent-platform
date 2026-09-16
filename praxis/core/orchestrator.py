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
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from praxis.agents.planner import Planner
from praxis.agents.skill import Skill
from praxis.core.execution_graph import PlanStep, compute_levels, resolve_args
from praxis.core.exceptions import SynthesisValidationError
from praxis.core.risk_policy import PendingInput, should_pause_for_approval
from praxis.memory.db import PostgresStore
from praxis.memory.models import Task


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
    """In-memory execution-graph state for one in-flight task (see module docstring)."""

    steps: list[PlanStep]
    levels: list[list[int]]
    results: dict[int, Any] = field(default_factory=dict)
    next_level: int = 0
    paused_index: int | None = None
    pending_args: dict[int, dict[str, Any]] = field(default_factory=dict)


def _describe_step(step: PlanStep) -> str:
    return f"{step.skill_name}({step.args})"


class Orchestrator:
    def __init__(
        self,
        store: PostgresStore,
        planner: Planner,
        skills: SkillRegistryLike,
        capability_factory: CapabilityFactoryLike | None = None,
    ) -> None:
        self._store = store
        self._planner = planner
        self._skills = skills
        self._capability_factory = capability_factory
        self._active: dict[str, _TaskState] = {}

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

        try:
            steps = await self._planner.plan(intent_text, self._skills.all_skills())
        except Exception as exc:  # noqa: BLE001 - a planning failure must fail the task, not crash the caller
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

        self._active[task_id] = _TaskState(steps=steps, levels=levels)
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
                await session.commit()
                del self._active[task_id]
                return

            skill = self._skills.get_skill(step.skill_name)  # guaranteed registered - checked before the pause
            resolved_args = state.pending_args[paused_index]
            self._mark_item(task, paused_index, status="in_progress")
            task.status = "running"
            task.pending_input = None
            await session.commit()

            try:
                output = await skill.run(**resolved_args)
            except Exception as exc:  # noqa: BLE001 - a skill's own failure, not swallowed (spec §12)
                self._mark_item(task, paused_index, status="failed", reason=str(exc))
                task.status = "failed"
                task.result = {"error": f"step {paused_index} ('{step.skill_name}') failed: {exc}"}
                await session.commit()
                del self._active[task_id]
                return

            state.results[paused_index] = output
            state.paused_index = None
            self._mark_item(task, paused_index, status="completed")
            await session.commit()

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

    async def _advance(self, task_id: str) -> None:
        state = self._active[task_id]
        async with self._store.session() as session:
            task = await session.get(Task, task_id)
            assert task is not None

            for level_pos in range(state.next_level, len(state.levels)):
                outcome = await self._process_level(session, task, state, level_pos)
                if outcome in ("paused", "failed"):
                    if outcome == "failed":
                        del self._active[task_id]
                    return
                state.next_level = level_pos + 1

            task.status = "completed"
            task.result = self._build_result(state, task.checklist)
            await session.commit()
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
            self._mark_item(task, index, status="skipped", reason=message)
            task.status = "failed"
            task.result = {"error": f"step {index}: {message}"}
            await session.commit()
            return None

        need_description = (
            f"A running plan needs a skill named '{step.skill_name}', to be called with "
            f"keyword arguments {resolved_args!r} - no such skill is registered yet. "
            "Synthesize a skill that fulfills this need, accepting exactly those keyword "
            "argument names."
        )
        try:
            return await self._capability_factory.synthesize(
                need_description=need_description, task_id=task.id
            )
        except SynthesisValidationError as exc:
            failure = f"{message}; capability synthesis also failed: {exc}"
            self._mark_item(task, index, status="failed", reason=failure)
            task.status = "failed"
            task.result = {"error": f"step {index}: {failure}"}
            await session.commit()
            return None

    async def _process_level(
        self, session: AsyncSession, task: Task, state: _TaskState, level_pos: int
    ) -> str:
        level = state.levels[level_pos]
        todo = [index for index in level if index not in state.results]
        if not todo:
            return "ok"

        run_plan: dict[int, tuple[Skill, dict[str, Any]]] = {}
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
                state.paused_index = index
                state.pending_args[index] = resolved_args
                return "paused"

            run_plan[index] = (skill, resolved_args)

        for index in run_plan:
            self._mark_item(task, index, status="in_progress")
        await session.commit()

        async def _run_one(index: int) -> tuple[int, Any, BaseException | None]:
            skill, args = run_plan[index]
            try:
                return index, await skill.run(**args), None
            except Exception as exc:  # noqa: BLE001 - captured per-step, not swallowed (spec §12)
                return index, None, exc

        outcomes = await asyncio.gather(*(_run_one(index) for index in run_plan))

        any_failed = False
        for index, output, exc in outcomes:
            if exc is not None:
                self._mark_item(task, index, status="failed", reason=str(exc))
                any_failed = True
            else:
                state.results[index] = output
                self._mark_item(task, index, status="completed")
        await session.commit()

        if any_failed:
            task.status = "failed"
            task.result = {"error": "one or more steps failed", "checklist": task.checklist}
            await session.commit()
            return "failed"
        return "ok"
