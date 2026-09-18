# praxis/api/routes/tasks.py
"""Task lifecycle routes (spec §7, §8, §9, §14) - split out of
`praxis.api.main`, which still owns the `Orchestrator` singleton
(`main._get_orchestrator()`) every route here drives.

- `POST /intent`: ad-hoc ask -> a new `Task`.
- `POST /webhook/alert`: async incident trigger (spec §16.1 step 1),
  riding the identical `Orchestrator.start_task()` path `/intent` uses.
- `GET /tasks/{task_id}`: status/result, including the live checklist
  and any pending approval/clarification detail.
- `POST /tasks/{task_id}/approve`: resumes a mutating task paused on
  approval (spec §8's interrupt).
- `POST /tasks/{task_id}/clarify`: answers a pending
  `ClarificationRequest` and resumes (spec §8).
- `WS /tasks/{task_id}/stream`: live status/log stream.
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from praxis.api import main
from praxis.config import Settings
from praxis.core.events import task_event_bus, task_state_snapshot
from praxis.memory.db import PostgresStore
from praxis.memory.models import Task

router = APIRouter()


class IntentRequest(BaseModel):
    text: str
    # Phase 11 (spec §16.2's literal example payload: `{"text": "...",
    # "connector": "customer-db"}`) - optional, forwarded verbatim to
    # `Orchestrator.start_task`'s own `connector_name` (see its
    # docstring for resolution/failure semantics).
    connector: str | None = None


class ApproveRequest(BaseModel):
    approved: bool


class ClarifyRequest(BaseModel):
    answer: str


@router.post("/intent", status_code=201)
async def create_intent(body: IntentRequest) -> dict[str, Any]:
    """Ad-hoc ask -> a new `Task`, run through the Orchestrator (spec §14)."""
    orchestrator = main._get_orchestrator()
    task_id = await orchestrator.start_task(body.text, connector_name=body.connector)
    return {"task_id": task_id}


@router.post("/webhook/alert", status_code=201)
async def webhook_alert(request: Request) -> dict[str, Any]:
    """Async incident trigger (spec §14; spec §16.1 step 1: "Alertmanager
    fires -> `POST /webhook/alert` -> API creates a `Task` ..., assigns
    `correlation_id`, logs the event").

    Rides the *identical* `Orchestrator.start_task()` path `/intent`
    above already uses - same `Task` creation, same `correlation_id`
    assignment, same `task_started` logging - never a parallel "webhook
    task" system (spec §7's "rides the identical Orchestrator path"
    principle for a scheduled trigger, generalized here to a webhook
    trigger: both just have to produce an `intent_text` string, then
    hand it to the one real path).

    The body is read and validated by hand (a plain `Request`, not a
    pydantic model) rather than via FastAPI's usual declarative body
    parsing, specifically so *every* malformed shape - missing `alerts`,
    an empty list, invalid JSON, wrong field types - comes back as a
    clear `400` (spec: "not a silent no-op or a 500"), rather than
    pydantic's usual `422` for a schema mismatch or an unhandled `500`
    for, say, `.json()` failing on a non-JSON body.

    Real Alertmanager webhook payloads batch multiple alerts into one
    call (`{"alerts": [...]}`, alongside fields this MVP has no use for
    - `version`, `groupKey`, `receiver`, `groupLabels`, ...). This
    handler deliberately turns only the *first* alert in the batch into
    one `Task`, rather than either (a) silently dropping the rest, or
    (b) fanning out one `Task` per alert - a single, reasonable MVP
    choice, not a full multi-alert dispatcher; a later phase's natural
    extension point is right here, with no change to how any one alert
    becomes an `Intent`.
    """
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001 - an unparseable body is a 400, not a 500
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="webhook payload must be a JSON object")

    alerts = payload.get("alerts")
    if not isinstance(alerts, list) or not alerts:
        raise HTTPException(
            status_code=400, detail="webhook payload must include a non-empty 'alerts' list"
        )

    alert = alerts[0]
    if not isinstance(alert, dict):
        raise HTTPException(status_code=400, detail="each entry in 'alerts' must be a JSON object")

    labels = alert.get("labels")
    labels = labels if isinstance(labels, dict) else {}
    annotations = alert.get("annotations")
    annotations = annotations if isinstance(annotations, dict) else {}

    alertname = labels.get("alertname") or "unknown_alert"
    summary = annotations.get("summary") or annotations.get("description") or "no summary provided"
    intent_text = f"Investigate alert '{alertname}': {summary}"

    orchestrator = main._get_orchestrator()
    task_id = await orchestrator.start_task(intent_text)
    return {"task_id": task_id}


@router.get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict[str, Any]:
    """Task status/result, including the live checklist and any pending
    approval/clarification detail (spec §7, §9, §14)."""
    settings = Settings()
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            task = await session.get(Task, task_id)
    finally:
        await store.dispose()

    if task is None:
        raise HTTPException(status_code=404, detail=f"no task with id '{task_id}'")

    return {
        "status": task.status,
        "checklist": task.checklist,
        "pending_input": task.pending_input,
        "result": task.result,
    }


@router.post("/tasks/{task_id}/approve")
async def approve_task(task_id: str, body: ApproveRequest) -> dict[str, Any]:
    """Resumes a mutating task paused on approval (spec §8, §14 - the interrupt)."""
    orchestrator = main._get_orchestrator()
    try:
        await orchestrator.resume_after_approval(task_id, body.approved)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        # Not currently awaiting approval (wrong status, unknown to this
        # process, ...) - a conflict with the resource's current state,
        # not a missing resource or a bad request body.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"task_id": task_id}


@router.post("/tasks/{task_id}/clarify")
async def clarify_task(task_id: str, body: ClarifyRequest) -> dict[str, Any]:
    """Answers a pending `ClarificationRequest` and resumes (spec §8, §14).

    Nothing in this phase's scope ever raises a `ClarificationRequest`
    (no Planner-side clarification logic is required yet - see the
    phase brief's scope note): this endpoint is still a real,
    contract-correct implementation rather than an omission - it
    validates the task exists and is genuinely `awaiting_clarification`
    before doing anything, exactly like `approve_task` above. It simply
    has no caller in this phase that ever reaches that state, so its
    "success" path (clearing the pause and recording the answer) stays
    real but untested-via-a-live-caller until a later phase's Planner/
    Factory actually raises one.
    """
    settings = Settings()
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            task = await session.get(Task, task_id)
            if task is None:
                raise HTTPException(status_code=404, detail=f"no task with id '{task_id}'")
            if task.status != "awaiting_clarification":
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"task '{task_id}' is not awaiting clarification "
                        f"(status: '{task.status}')"
                    ),
                )
            task.pending_input = None
            task.status = "running"
            task.result = {**(task.result or {}), "clarification_answer": body.answer}
            await session.commit()
    finally:
        await store.dispose()
    return {"task_id": task_id}


# A task in either of these is done, for good, no further event will
# ever be published for it (`Orchestrator._advance`/`_process_level`/
# `_synthesize_missing_skill`/`resume_after_approval` - see
# praxis/core/orchestrator.py) - the one condition `stream_task` below
# uses to know when to stop forwarding events and close the socket.
_TERMINAL_TASK_STATUSES = {"completed", "failed"}


@router.websocket("/tasks/{task_id}/stream")
async def stream_task(websocket: WebSocket, task_id: str) -> None:
    """Live status/log stream (spec §14: "including checklist item
    transitions and any new `ClarificationRequest` the instant it's
    raised").

    Every message this sends - the very first one included - is
    `praxis.core.events.task_state_snapshot`'s shape, identical to `GET
    /tasks/{task_id}`'s own response body above: a client renders a WS
    frame with the exact same code it already uses for a polled `GET`.

    Sequence: reject immediately (before ever accepting the handshake)
    if `task_id` doesn't exist - spec: "close with a clear error code/
    reason rather than hanging". Otherwise accept, subscribe to
    `task_event_bus`, *then* re-read the task's current durable state
    and send it (so a client connecting mid-task isn't left waiting for
    the next change), then forward every event `Orchestrator` publishes
    for this task - real state transitions, never a fake heartbeat -
    until one arrives in a terminal status, send that, and close
    cleanly.

    Subscribing *before* re-reading current state (rather than the
    other way round) closes a race that would otherwise be able to hang
    this route forever: if the task finished in the gap between the
    existence check and the subscribe call, subscribing first guarantees
    either the terminal event is already waiting in the queue, or the
    fresh read immediately below already shows the terminal status -
    never a subscribe that's too late to ever see one more event because
    the task was already done and nothing will ever publish again.
    """
    settings = Settings()
    store = PostgresStore(settings)
    try:
        async with store.session() as session:
            exists = await session.get(Task, task_id)
        if exists is None:
            await websocket.close(code=4404, reason=f"no task with id '{task_id}'")
            return

        await websocket.accept()
        queue = task_event_bus.subscribe(task_id)
        # A task paused on `awaiting_approval`/`awaiting_clarification`
        # (spec §8) is real, but *not* terminal - this route must keep
        # the connection open across a pause exactly like it would
        # across any other in-progress state (more events may still
        # arrive, e.g. once someone calls `POST /tasks/{id}/approve`).
        # That means the loop below can be waiting on `queue.get()`
        # indefinitely with nothing else ever going to publish again in
        # this test/run - so a client-initiated disconnect must be
        # noticed concurrently, or this coroutine (and the `TestClient`
        # session waiting on it to finish) would hang forever. Hence the
        # second, concurrently-awaited task below, whose only job is to
        # notice `{"type": "websocket.disconnect"}` the moment it
        # arrives - this route never expects the client to *send*
        # anything meaningful of its own.
        disconnect_task = asyncio.ensure_future(_wait_for_client_disconnect(websocket))
        try:
            async with store.session() as session:
                task = await session.get(Task, task_id)
            assert task is not None  # just confirmed to exist, immediately above

            await websocket.send_json(task_state_snapshot(task))

            if task.status not in _TERMINAL_TASK_STATUSES:
                while True:
                    get_event = asyncio.ensure_future(queue.get())
                    done, _pending = await asyncio.wait(
                        {get_event, disconnect_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if disconnect_task in done:
                        get_event.cancel()
                        break
                    event = get_event.result()
                    await websocket.send_json(event)
                    if event["status"] in _TERMINAL_TASK_STATUSES:
                        break
        except WebSocketDisconnect:
            # The client hung up early - nothing left to stream to, and
            # nothing to clean up beyond the `finally` below.
            pass
        finally:
            if not disconnect_task.done():
                disconnect_task.cancel()
            task_event_bus.unsubscribe(task_id, queue)

        try:
            await websocket.close(code=1000)
        except RuntimeError:
            # Already closed (e.g. the client disconnected first, above) -
            # closing twice is a no-op, not an error worth surfacing.
            pass
    finally:
        await store.dispose()


async def _wait_for_client_disconnect(websocket: WebSocket) -> None:
    """Resolves the moment the client hangs up - the half of `stream_task`
    above's disconnect race that lets a still-paused (non-terminal) task's
    stream close promptly instead of blocking on `queue.get()` forever
    once nobody is listening any more."""
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
