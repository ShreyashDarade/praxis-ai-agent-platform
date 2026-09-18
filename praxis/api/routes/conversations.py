# praxis/api/routes/conversations.py
"""The chat surface (brief: a conversation, not a task submission).

`POST /intent` takes one sentence, runs it to completion, and returns a
task id when there is nothing left to watch. That is a job queue. These
routes are the product the brief actually describes: a thread you can
keep talking to, where a follow-up knows what came before and the reply
is an answer rather than a list of step outputs.

Three differences from `/intent`, each deliberate:

- **A message returns immediately.** Both ids come back as soon as the
  rows exist, because the ids are what a client needs in order to watch
  the work - and they are worthless once it has finished.
- **The reply is a message.** When the task settles, the composer turns
  what actually ran into prose with evidence and limitations, and that
  becomes the assistant's turn (`praxis.agents.answer`).
- **History reaches planning.** Prior turns are put in front of the
  planner, which is the whole difference between "group that by region"
  meaning something and meaning nothing.

Tenant- and user-scoped throughout. A conversation is private to its
creator: two users in one tenant do not see each other's threads, and
"not yours" and "does not exist" are the same 404 for the usual reason.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from praxis.agents.conversation import ConversationNotFoundError
from praxis.api.dependencies import get_settings, require
from praxis.config import Settings
from praxis.core.execution_mode import ExecutionMode
from praxis.security.policy import Permission
from praxis.security.principal import Principal

router = APIRouter(prefix="/conversations", tags=["conversations"])


class CreateConversationRequest(BaseModel):
    title: str = Field(default="", max_length=512)
    connector_name: str | None = Field(
        default=None,
        description="Data source this thread is about, so follow-ups need not name it",
    )


class SendMessageRequest(BaseModel):
    text: str = Field(min_length=1, description="What the user is saying")
    attachment_ids: list[str] = Field(
        default_factory=list,
        description="Uploads this turn is scoped to, so 'these invoices' means these",
    )
    mode: ExecutionMode = Field(
        default=ExecutionMode.EXECUTE, description="read_only | plan_only | dry_run | execute"
    )


def _service():
    """Built per request from the shared singletons.

    Imported inside the function for the same reason the other route
    modules do it: `main` imports this module to mount the router, so a
    module-level import would be a genuine cycle.
    """
    from praxis.api import main

    return main.get_conversation_service()


def _serialize_message(message: Any) -> dict[str, Any]:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "task_id": message.task_id,
        "attachment_ids": list(message.attachment_ids or []),
        "evidence": list(message.evidence or []),
        "limitations": list(message.limitations or []),
        "created_at": message.created_at.isoformat(),
    }


def _serialize_conversation(conversation: Any) -> dict[str, Any]:
    return {
        "id": conversation.id,
        "title": conversation.title,
        "connector_name": conversation.connector_name,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
    }


@router.post("", status_code=201)
async def create_conversation(
    body: CreateConversationRequest,
    principal: Annotated[Principal, Depends(require(Permission.TASK_CREATE))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Opens a thread."""
    conversation = await _service().create(
        principal, title=body.title, connector_name=body.connector_name
    )
    return _serialize_conversation(conversation)


@router.get("")
async def list_conversations(
    principal: Annotated[Principal, Depends(require(Permission.TASK_READ))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """The caller's own threads - filtered in the query, not after."""
    rows = await _service().list_for(principal)
    return {"conversations": [_serialize_conversation(row) for row in rows]}


@router.get("/{conversation_id}/messages")
async def list_messages(
    conversation_id: str,
    principal: Annotated[Principal, Depends(require(Permission.TASK_READ))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    try:
        rows = await _service().messages(principal, conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "conversation_id": conversation_id,
        "messages": [_serialize_message(row) for row in rows],
    }


@router.post("/{conversation_id}/messages", status_code=202)
async def send_message(
    conversation_id: str,
    body: SendMessageRequest,
    principal: Annotated[Principal, Depends(require(Permission.TASK_CREATE))],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Says something, and starts the work behind it.

    **202, not 200**: the turn is accepted and running, not finished.
    The response carries `task_id` so a client can subscribe to
    `WS /tasks/{id}/stream` immediately, and the assistant's reply
    appears as a new message when the work settles.
    """
    try:
        started = await _service().send(
            principal,
            conversation_id,
            body.text,
            attachment_ids=body.attachment_ids,
            mode=body.mode,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return started.to_dict()
