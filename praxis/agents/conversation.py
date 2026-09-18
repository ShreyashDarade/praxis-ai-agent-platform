# praxis/agents/conversation.py
"""Conversations: the surface that turns a task runner into a commander.

The brief describes a user who types a request, gets an answer, and then
says "now group that by region". Every piece needed to *execute* that
second request already existed - planning, skills, connectors,
approvals, checkpoints. What did not exist was the thing that makes it a
second request rather than a first one: somewhere for the exchange to
live.

This module is that. It is deliberately thin, and deliberately owns
three things the orchestrator should not know about:

**History reaches planning.** A follow-up is planned with the prior
turns in front of it, which is the entire difference between "group that
by region" meaning something and meaning nothing. The history is passed
as text rather than as a re-derived plan: re-planning from scratch with
context is honest, whereas patching the previous plan would silently
inherit decisions the user may have been reacting against.

**A turn returns before the work finishes.** `POST /intent` awaited the
whole task, so a client got the id only once there was nothing left to
watch. A conversation creates the user message and the task, starts
execution in the background, and returns immediately - the ids are what
the client needs to subscribe, and they are useless after the fact.

**The answer is written back as a message.** When the task settles, the
composer turns what actually ran into prose with evidence and
limitations (`praxis.agents.answer`), and that becomes the assistant's
turn. A task whose result nobody converts into an answer is a job
report, which is what this replaces.

Failure is a turn too. If the task fails, or the process dies mid-run,
the conversation must not simply stop having a reply - so every
terminal outcome writes an assistant message, including the ones nobody
wants.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select

from praxis.agents.answer import AssistantAnswer, compose_answer
from praxis.core.execution_graph import PlanStep
from praxis.core.execution_mode import ExecutionMode
from praxis.llm.catalogue import LLMCatalogue
from praxis.llm.prompt_manager import PromptManager
from praxis.memory.db import PostgresStore
from praxis.memory.models import Conversation, Message, Task
from praxis.security.principal import Principal

_logger = structlog.get_logger(__name__)

# How many prior turns to put in front of the planner. Enough for a
# follow-up to resolve what "that" refers to, bounded so a long thread
# does not grow the prompt without limit. Context compression for
# genuinely long conversations is a separate, unbuilt concern, and this
# bound is what keeps its absence from becoming a runaway cost.
DEFAULT_HISTORY_TURNS = 10

_MAX_HISTORY_CHARS_PER_MESSAGE = 2000


class ConversationNotFoundError(KeyError):
    """No such conversation for this principal.

    A `KeyError` so existing handlers map it to 404 unchanged. Not
    found and not yours are the same error on purpose: telling them
    apart would confirm another user's conversation exists.
    """


@dataclass
class TurnStarted:
    """What a client gets back the moment it sends a message.

    Both ids matter immediately: `message_id` identifies the turn, and
    `task_id` is what a progress stream subscribes to. Returning them
    only after the work finished - which is what awaiting execution
    did - made the stream pointless.
    """

    conversation_id: str
    message_id: str
    task_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "task_id": self.task_id,
        }


def _truncate(text: str) -> str:
    if len(text) <= _MAX_HISTORY_CHARS_PER_MESSAGE:
        return text
    return text[:_MAX_HISTORY_CHARS_PER_MESSAGE] + "... [truncated]"


class ConversationService:
    """Creates conversations, runs turns, and writes the replies."""

    def __init__(
        self,
        store: PostgresStore,
        orchestrator: Any,
        *,
        catalogue: LLMCatalogue | None = None,
        prompt_manager: PromptManager | None = None,
        history_turns: int = DEFAULT_HISTORY_TURNS,
    ) -> None:
        self._store = store
        self._orchestrator = orchestrator
        self._catalogue = catalogue if catalogue is not None else LLMCatalogue()
        self._prompts = prompt_manager if prompt_manager is not None else PromptManager()
        self._history_turns = history_turns
        # Background turns, kept referenced so Python cannot garbage
        # collect a running task out from under itself - a real failure
        # mode of `asyncio.create_task` with no retained handle.
        self._running: set[asyncio.Task] = set()

    # -- conversations -------------------------------------------------

    async def create(
        self, principal: Principal, *, title: str = "", connector_name: str | None = None
    ) -> Conversation:
        async with self._store.session() as session:
            conversation = Conversation(
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                title=title,
                connector_name=connector_name,
            )
            session.add(conversation)
            await session.commit()
            await session.refresh(conversation)
            return conversation

    async def get(self, principal: Principal, conversation_id: str) -> Conversation:
        """Loads a conversation the principal actually owns."""
        async with self._store.session() as session:
            conversation = await session.get(Conversation, conversation_id)
        if (
            conversation is None
            or conversation.tenant_id != principal.tenant_id
            or (conversation.user_id is not None and conversation.user_id != principal.user_id)
        ):
            # One error for both cases - see ConversationNotFoundError.
            raise ConversationNotFoundError(
                f"no conversation with id '{conversation_id}'"
            )
        return conversation

    async def list_for(self, principal: Principal, *, limit: int = 50) -> list[Conversation]:
        async with self._store.session() as session:
            rows = (
                (
                    await session.execute(
                        select(Conversation)
                        .where(Conversation.tenant_id == principal.tenant_id)
                        .where(Conversation.user_id == principal.user_id)
                        .order_by(Conversation.updated_at.desc())
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            return list(rows)

    async def messages(
        self, principal: Principal, conversation_id: str, *, limit: int = 100
    ) -> list[Message]:
        await self.get(principal, conversation_id)  # authorizes, or raises
        async with self._store.session() as session:
            rows = (
                (
                    await session.execute(
                        select(Message)
                        .where(Message.conversation_id == conversation_id)
                        .where(Message.tenant_id == principal.tenant_id)
                        .order_by(Message.created_at.asc())
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            return list(rows)

    # -- turns ---------------------------------------------------------

    async def _history_text(self, conversation_id: str, tenant_id: str) -> str:
        """Prior turns, oldest first, as plain text for the planner.

        Text rather than structured turns because it is going into a
        prompt: the planner needs to understand what was asked and
        answered, not to re-execute it.
        """
        async with self._store.session() as session:
            rows = (
                (
                    await session.execute(
                        select(Message)
                        .where(Message.conversation_id == conversation_id)
                        .where(Message.tenant_id == tenant_id)
                        .order_by(Message.created_at.desc())
                        .limit(self._history_turns)
                    )
                )
                .scalars()
                .all()
            )
        return "\n".join(
            f"{row.role}: {_truncate(row.content)}" for row in reversed(list(rows))
        )

    async def send(
        self,
        principal: Principal,
        conversation_id: str,
        text: str,
        *,
        attachment_ids: list[str] | None = None,
        mode: ExecutionMode = ExecutionMode.EXECUTE,
    ) -> TurnStarted:
        """Records the user's turn and starts the work behind it.

        Returns as soon as both rows exist. Execution continues in the
        background and writes the assistant's reply when it settles.
        """
        conversation = await self.get(principal, conversation_id)
        history = await self._history_text(conversation_id, principal.tenant_id)

        async with self._store.session() as session:
            message = Message(
                tenant_id=principal.tenant_id,
                conversation_id=conversation_id,
                role="user",
                content=text,
                attachment_ids=list(attachment_ids or []),
            )
            session.add(message)
            # Bumped so a conversation list orders by real activity.
            stored = await session.get(Conversation, conversation_id)
            if stored is not None:
                stored.updated_at = datetime.now(UTC)
            await session.commit()
            message_id = message.id

        intent = self._compose_intent(text, history, attachment_ids or [])
        task_id = await self._orchestrator.create_task(
            intent,
            principal=principal,
            mode=mode,
            connector_name=conversation.connector_name,
            conversation_id=conversation_id,
        )

        async with self._store.session() as session:
            stored_message = await session.get(Message, message_id)
            if stored_message is not None:
                stored_message.task_id = task_id
                await session.commit()

        runner = asyncio.create_task(
            self._run_turn(
                principal=principal,
                conversation_id=conversation_id,
                task_id=task_id,
                intent_text=text,
                history=history,
            )
        )
        self._running.add(runner)
        runner.add_done_callback(self._running.discard)

        return TurnStarted(
            conversation_id=conversation_id, message_id=message_id, task_id=task_id
        )

    @staticmethod
    def _compose_intent(text: str, history: str, attachment_ids: list[str]) -> str:
        """What the planner is actually asked to plan.

        The user's words come last and are labelled, so a follow-up is
        planned *as* a follow-up rather than as a fresh request that
        happens to mention earlier ones.
        """
        parts: list[str] = []
        if history:
            parts.append("Earlier in this conversation:\n" + history)
        if attachment_ids:
            parts.append(
                "The user has attached these uploads to this message; use them as "
                "the source rather than searching more broadly: "
                + ", ".join(attachment_ids)
            )
        parts.append("The user now asks:\n" + text)
        return "\n\n".join(parts)

    async def _run_turn(
        self,
        *,
        principal: Principal,
        conversation_id: str,
        task_id: str,
        intent_text: str,
        history: str,
    ) -> None:
        """Drives one task to completion and writes the reply.

        Every terminal outcome produces an assistant message, including
        failure: a conversation whose reply simply never arrives is
        indistinguishable from one still working, and a user cannot tell
        whether to wait.
        """
        try:
            await self._orchestrator.drive_task(task_id)
        except Exception as exc:  # noqa: BLE001 - a turn must always answer
            _logger.exception("conversation_turn_failed", task_id=task_id)
            await self._write_assistant_message(
                principal,
                conversation_id,
                task_id,
                AssistantAnswer(
                    text=(
                        "That request could not be completed. "
                        f"The run stopped with: {type(exc).__name__}: {exc}"
                    ),
                    limitations=["The task did not finish."],
                ),
            )
            return

        answer = await self._answer_for(principal, task_id, intent_text, history)
        if answer is not None:
            await self._write_assistant_message(
                principal, conversation_id, task_id, answer
            )

    async def _answer_for(
        self, principal: Principal, task_id: str, intent_text: str, history: str
    ) -> AssistantAnswer | None:
        """Composes the reply, or `None` when the task is still pending.

        A task paused for approval has not failed and has no answer yet.
        Writing one would be wrong twice over: it would claim a result
        that does not exist, and it would make the approval look
        already-resolved.
        """
        async with self._store.session() as session:
            task = await session.get(Task, task_id)
        if task is None:
            return None
        if task.status in ("awaiting_approval", "awaiting_clarification"):
            pending = task.pending_input or {}
            detail = str(pending.get("detail") or "waiting for your decision")
            return AssistantAnswer(
                text=detail,
                limitations=["This request is paused until you respond."],
            )
        if task.status not in ("completed", "failed", "cancelled"):
            return None

        result = task.result or {}
        steps = [
            PlanStep(
                skill_name=str(entry.get("skill_name", "")),
                args=dict(entry.get("args") or {}),
                depends_on=list(entry.get("depends_on") or []),
            )
            for entry in (task.plan or [])
        ]
        results = {
            int(item["step"]): item.get("output")
            for item in result.get("steps", [])
            if isinstance(item, dict) and "step" in item
        }
        errors = [str(e) for e in (result.get("errors") or [])]
        if task.status == "cancelled":
            errors.append("The run was cancelled.")

        return await compose_answer(
            intent_text=intent_text,
            steps=steps,
            results=results,
            errors=errors,
            catalogue=self._catalogue,
            prompt_manager=self._prompts,
            history=history,
            principal=principal,
        )

    async def _write_assistant_message(
        self,
        principal: Principal,
        conversation_id: str,
        task_id: str,
        answer: AssistantAnswer,
    ) -> Message:
        async with self._store.session() as session:
            message = Message(
                tenant_id=principal.tenant_id,
                conversation_id=conversation_id,
                role="assistant",
                content=answer.text,
                task_id=task_id,
                evidence=[e.to_dict() for e in answer.evidence],
                limitations=list(answer.limitations),
            )
            session.add(message)
            stored = await session.get(Conversation, conversation_id)
            if stored is not None:
                stored.updated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(message)
            return message

    async def wait_for_turn(self, timeout: float = 120.0) -> None:
        """Waits for in-flight turns. For tests and graceful shutdown."""
        if not self._running:
            return
        await asyncio.wait(set(self._running), timeout=timeout)
