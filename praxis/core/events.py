# praxis/core/events.py
"""In-process pub/sub for live task-state updates (spec §14's
`WS /tasks/{id}/stream`: "Live status/log stream, including checklist
item transitions and any new `ClarificationRequest` the instant it's
raised").

A single process-global `TaskEventBus` instance (`task_event_bus`,
constructed once below) is shared between `praxis.core.orchestrator.
Orchestrator` (the publisher - see its `_publish_state`, called right
after every commit that changes a `Task`'s `status`/`checklist`/
`pending_input`/`result`) and the `WS /tasks/{id}/stream` route in
`praxis.api.main` (the subscriber). This is deliberately in-memory
only, mirroring `Orchestrator._active`'s own "short-term state"
trade-off (see that module's docstring): a process restart drops every
subscriber, which is fine because a WS client that lost its connection
just reconnects and gets the task's *current* durable state (from
Postgres) as its first message anyway - never a fake heartbeat, always
a real snapshot.

Each subscriber gets its own `asyncio.Queue` so one slow consumer can
never block another subscriber, or block the Orchestrator's own
execution - `publish()` always uses `put_nowait`, bounded by a generous
maxsize; a full queue drops the new event for that one slow subscriber
rather than ever raising into the Orchestrator's own execution path (it
will still get the *next* event, and the route's own re-read of current
state on connect means no subscriber is ever stuck believing a stale
status forever).
"""
from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Protocol


class _TaskLike(Protocol):
    """What `task_state_snapshot` needs - satisfied by the real
    `praxis.memory.models.Task` ORM instance, with no import-time
    dependency on it (keeps this module usable from anywhere without
    pulling in the whole memory/models stack)."""

    id: str
    status: str
    checklist: list[Any]
    pending_input: dict[str, Any] | None
    result: dict[str, Any] | None


def task_state_snapshot(task: _TaskLike) -> dict[str, Any]:
    """The one snapshot shape shared by every `TaskEventBus.publish` call
    in `Orchestrator` *and* by the `WS /tasks/{id}/stream` route's own
    initial/final messages in `praxis.api.main` - deliberately identical
    to `GET /tasks/{id}`'s response body, so a client renders a WS frame
    with the exact same code it already uses for a polled `GET`."""
    return {
        "status": task.status,
        "checklist": task.checklist,
        "pending_input": task.pending_input,
        "result": task.result,
    }


class TaskEventBus:
    """A minimal in-process pub/sub, keyed by task id (see module
    docstring). `subscribe`/`unsubscribe`/`publish` are the entire
    surface - deliberately no "topics" or wildcard subscriptions; one
    task's WS stream only ever cares about that one task's own events.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    def subscribe(self, task_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._subscribers.setdefault(task_id, []).append(queue)
        return queue

    def unsubscribe(self, task_id: str, queue: asyncio.Queue) -> None:
        subscribers = self._subscribers.get(task_id)
        if not subscribers:
            return
        with contextlib.suppress(ValueError):
            subscribers.remove(queue)
        if not subscribers:
            self._subscribers.pop(task_id, None)

    async def publish(self, task_id: str, event: dict[str, Any]) -> None:
        for queue in list(self._subscribers.get(task_id, ())):
            # A full queue drops this event for that one slow subscriber
            # rather than raising into the publisher - see the module
            # docstring for why that trade is the right one here.
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(event)


# The process-global bus `Orchestrator` publishes to by default (see its
# `event_bus` constructor parameter) and the WS route subscribes to -
# mirrors `praxis.agents.skill_registry`'s own module-level singleton
# posture, for the same reason: both the API's request handlers and the
# Orchestrator singleton (`praxis.api.main._get_orchestrator`) must
# genuinely share the one instance, not each get their own.
task_event_bus = TaskEventBus()
