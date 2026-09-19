# praxis/agents/delegation.py
"""Delegation limits: concurrency, nesting, recursion, orphan cleanup.

Prompt §1 names these explicitly: *"Specify concurrency limits, nesting
limits, recursive-task detection, total descendant cost limits,
worker-pool capacity, orphan cleanup, and cancellation propagation."*

Each control here exists because a specific, real failure mode is
otherwise unbounded:

- **Nesting limit** - a specialist that delegates to a specialist that
  delegates... Without a depth bound this terminates only when
  something else breaks.
- **Recursive-task detection** - depth alone does not catch A->B->A
  cycling at shallow depth with a slowly-mutating objective. Detection
  hashes the *objective* along the ancestry chain, so a task that
  re-asks a question already being asked above it is refused.
- **Concurrency limit / worker pool** - bounded parallel fan-out, so a
  100-way `asyncio.gather` cannot exhaust connections or rate limits.
- **Orphan cleanup** - a parent that dies must not leave children
  running and writing. The registry tracks parentage so descendants
  can be cancelled as a set.

Descendant *cost* limits live in `praxis.agents.budget` (they are a
budget concern, tracked on the same rollup ledger), not here.
"""
from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import structlog

from praxis.agents.contract import TaskContract

_logger = structlog.get_logger(__name__)

T = TypeVar("T")

# Deliberately conservative defaults. A supervisor -> specialist ->
# helper chain is 3 levels; anything deeper is far more likely to be
# runaway delegation than genuine decomposition.
DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_CONCURRENCY = 8
DEFAULT_MAX_CHILDREN_PER_TASK = 20


class DelegationLimitError(Exception):
    """A delegation was refused by a structural limit.

    `limit` names which one (`depth`, `children`, `recursion`) so a
    caller can distinguish "this decomposition is too deep" from "this
    task is asking itself the same question again".
    """

    def __init__(self, message: str, *, limit: str, detail: str = "") -> None:
        super().__init__(message)
        self.limit = limit
        self.detail = detail


def _objective_fingerprint(objective: str) -> str:
    """A normalized hash of an objective, for cycle detection.

    Normalized (lowercased, whitespace-collapsed) so trivially reworded
    repeats - the usual shape of an LLM-driven delegation loop - still
    collide. This is intentionally a *coarse* check: a false positive
    refuses one delegation with a clear error, while a false negative
    allows an infinite loop, so erring toward collision is the safer
    direction.
    """
    normalized = " ".join(objective.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@dataclass
class DelegationLimits:
    max_depth: int = DEFAULT_MAX_DEPTH
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    max_children_per_task: int = DEFAULT_MAX_CHILDREN_PER_TASK


@dataclass
class _TaskNode:
    contract: TaskContract
    fingerprint: str
    children: list[str] = field(default_factory=list)
    cancelled: bool = False


class DelegationRegistry:
    """Tracks the live delegation tree for one orchestrator.

    Holds only in-flight structure (parentage, objectives, cancellation
    flags) - never results, which belong to the task record. That
    keeps this cheap enough to consult on every delegation.
    """

    def __init__(self, limits: DelegationLimits | None = None) -> None:
        self._limits = limits if limits is not None else DelegationLimits()
        self._nodes: dict[str, _TaskNode] = {}
        self._semaphore = asyncio.Semaphore(self._limits.max_concurrency)

    @property
    def limits(self) -> DelegationLimits:
        return self._limits

    def register_root(self, contract: TaskContract) -> None:
        self._nodes[contract.task_id] = _TaskNode(
            contract=contract, fingerprint=_objective_fingerprint(contract.objective)
        )

    def _ancestry(self, task_id: str) -> list[_TaskNode]:
        """Every ancestor of `task_id`, nearest first."""
        chain: list[_TaskNode] = []
        seen: set[str] = set()
        cursor = self._nodes.get(task_id)
        while cursor is not None and cursor.contract.parent_id is not None:
            parent_id = cursor.contract.parent_id
            if parent_id in seen:  # pragma: no cover - defensive against a malformed tree
                break
            seen.add(parent_id)
            parent = self._nodes.get(parent_id)
            if parent is None:
                break
            chain.append(parent)
            cursor = parent
        return chain

    def authorize_delegation(self, child: TaskContract) -> None:
        """Checks every structural limit before a child is admitted.

        Raises `DelegationLimitError` rather than returning a verdict:
        a limit a caller can forget to consult is not a limit.
        """
        if child.depth > self._limits.max_depth:
            raise DelegationLimitError(
                (
                    f"delegation depth {child.depth} exceeds the maximum of "
                    f"{self._limits.max_depth}"
                ),
                limit="depth",
                detail=child.objective[:200],
            )

        parent_id = child.parent_id
        if parent_id is not None:
            parent = self._nodes.get(parent_id)
            if (
                parent is not None
                and len(parent.children) + 1 > self._limits.max_children_per_task
            ):
                raise DelegationLimitError(
                    (
                        f"task '{parent_id}' already has {len(parent.children)} children, "
                        f"at the maximum of {self._limits.max_children_per_task}"
                    ),
                    limit="children",
                )

            fingerprint = _objective_fingerprint(child.objective)
            # Cycle detection: the child's own parent chain must not
            # already contain this objective. Depth alone would let
            # A->B->A recur indefinitely at shallow depth.
            for ancestor in [parent, *self._ancestry(parent_id)] if parent else []:
                if ancestor.fingerprint == fingerprint:
                    raise DelegationLimitError(
                        (
                            "recursive delegation detected: this objective is already being "
                            f"pursued by ancestor task '{ancestor.contract.task_id}'"
                        ),
                        limit="recursion",
                        detail=child.objective[:200],
                    )

    def register_child(self, child: TaskContract) -> None:
        """Admits `child` after `authorize_delegation` has passed."""
        self.authorize_delegation(child)
        self._nodes[child.task_id] = _TaskNode(
            contract=child, fingerprint=_objective_fingerprint(child.objective)
        )
        if child.parent_id is not None and child.parent_id in self._nodes:
            self._nodes[child.parent_id].children.append(child.task_id)

    def descendants_of(self, task_id: str) -> list[str]:
        """Every descendant id, breadth-first. Used by cancellation
        propagation and orphan cleanup."""
        found: list[str] = []
        queue = list(self._nodes.get(task_id, _TaskNode(TaskContract(""), "")).children)
        while queue:
            current = queue.pop(0)
            found.append(current)
            node = self._nodes.get(current)
            if node is not None:
                queue.extend(node.children)
        return found

    def cancel_tree(self, task_id: str) -> list[str]:
        """Marks `task_id` and every descendant cancelled.

        Returns the ids actually marked, so the caller can log and
        audit exactly what a cancellation reached rather than assuming
        it reached everything.
        """
        cancelled: list[str] = []
        for node_id in [task_id, *self.descendants_of(task_id)]:
            node = self._nodes.get(node_id)
            if node is not None and not node.cancelled:
                node.cancelled = True
                cancelled.append(node_id)
        return cancelled

    def is_cancelled(self, task_id: str) -> bool:
        node = self._nodes.get(task_id)
        return node.cancelled if node is not None else False

    def reap_orphans(self, live_root_ids: set[str]) -> list[str]:
        """Drops tracked tasks whose root is no longer live.

        Without this the registry grows forever in a long-running
        process, and - worse - a dead parent's children stay
        un-cancelled, which is exactly the orphan case the prompt calls
        out. Returns the reaped ids.
        """
        reaped: list[str] = []
        for task_id, node in list(self._nodes.items()):
            root = task_id
            cursor = node
            while cursor.contract.parent_id is not None:
                parent = self._nodes.get(cursor.contract.parent_id)
                if parent is None:
                    break
                root = parent.contract.task_id
                cursor = parent
            if root not in live_root_ids:
                self._nodes.pop(task_id, None)
                reaped.append(task_id)
        if reaped:
            _logger.info("delegation_orphans_reaped", count=len(reaped))
        return reaped

    async def run_bounded(self, coro_factory: Callable[[], Awaitable[T]]) -> T:
        """Runs one delegated coroutine under the concurrency limit.

        A semaphore rather than an unbounded `gather`: bounded fan-out
        is what keeps a wide plan from exhausting database connections
        or tripping provider rate limits.
        """
        async with self._semaphore:
            return await coro_factory()

    async def run_all_bounded(
        self, coro_factories: list[Callable[[], Awaitable[T]]]
    ) -> list[T | BaseException]:
        """Runs many delegated coroutines with bounded concurrency,
        returning results positionally.

        Exceptions are *returned*, not raised, so one specialist's
        failure never discards its siblings' completed work - the
        partial-result aggregation the prompt requires.
        """
        return await asyncio.gather(
            *(self.run_bounded(factory) for factory in coro_factories),
            return_exceptions=True,
        )


def enforce_authorized_tools(contract: TaskContract, tool_name: str) -> None:
    """Raises `PermissionError` if `contract` does not authorize `tool_name`.

    A standalone function, deliberately: it is called at the point of
    *use* (`AgentContext.tool`) rather than being implied by
    construction, so the enforcement is visible at the call site rather
    than being an invisible property of an object graph.
    """
    if not contract.authorizes(tool_name):
        raise PermissionError(
            f"task '{contract.task_id}' is not authorized to use tool '{tool_name}'"
        )


def aggregate_results(outcomes: list[Any]) -> dict[str, Any]:
    """Folds a fan-out's outcomes into one summary (Prompt §1's
    "partial-result aggregation and failure propagation").

    Both halves are preserved deliberately: successes are not discarded
    because a sibling failed, and failures are not hidden because most
    siblings succeeded. The caller decides what to do with a partial.
    """
    succeeded: list[Any] = []
    failed: list[str] = []
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            failed.append(str(outcome))
        elif getattr(outcome, "succeeded", True):
            succeeded.append(outcome)
        else:
            failed.extend(getattr(outcome, "errors", []) or ["unspecified failure"])
    return {
        "total": len(outcomes),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "results": succeeded,
        "errors": failed,
        "partial": bool(succeeded) and bool(failed),
    }

