# praxis/agents/policy.py
"""The model-backed decision function `AgentLoop` runs on.

`AgentLoop` was complete except for the one thing that makes it an
*agent*: a `Policy`, the callable it asks "what next?" each iteration.
Only tests ever supplied one, so the loop - stall detection, contract
authorization, budget enforcement, delegation limits, checkpointed
replay - could not be reached from a real request at all. This module
is that missing piece.

Three properties are deliberate:

- **It sees only what the contract authorizes.** The prompt lists the
  contract's own `authorized_tools`, not the whole registry, so the
  model is not invited to reach for something the loop would then
  refuse. `AgentLoop._act_skill` re-checks anyway - the prompt shapes
  the decision, the contract enforces it - but showing a tool that
  cannot be called wastes an iteration on a refusal.
- **It is shown its own failures.** Each iteration's action *and*
  observation go back into the next prompt, including errors. That is
  the whole advantage of a loop over a plan: a query that failed
  naming the columns that do exist is enough to get the next query
  right, and the prompt says so explicitly.
- **Finishing empty-handed is a legitimate answer.** The prompt tells
  the model to finish and say what blocked it when the objective
  cannot be met. A policy that could only finish *successfully* would
  keep acting until it exhausted the iteration ceiling, and then the
  loop would report "ran out of steps" for a question that was
  actually unanswerable - a worse diagnosis, arrived at more
  expensively.

Parse failures raise. `AgentLoop._step` catches a policy exception and
stops with `StopReason.FAILED` carrying the message, which is the
honest outcome: a decision that could not be read is not a decision,
and guessing an action from a malformed response would be inventing
one.
"""
from __future__ import annotations

import json
import re
from typing import Any

import structlog

from praxis.agents.contract import TaskContract
from praxis.agents.loop import Action, ActionKind, Iteration
from praxis.llm.catalogue import LLMCatalogue
from praxis.llm.prompt_manager import PromptManager

_logger = structlog.get_logger(__name__)

_PROMPT_NAME = "next_action"
_PROMPT_VERSION = "v1"
_LLM_PURPOSE = "planning"

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

# How much of one observation to show. A loop that fetched 10k rows
# must not spend its whole context replaying them, and the shape plus
# the head of the data is what the next decision actually turns on.
_MAX_OBSERVATION_CHARS = 15000


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    match = _CODE_FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped


def parse_action(response: str) -> Action:
    """Parses one policy response into an `Action`.

    Raises `ValueError` with a specific message - never an opaque
    `JSONDecodeError` or `KeyError` - because the message becomes the
    loop's recorded failure reason and is the only thing whoever reads
    the trace will have.
    """
    candidate = _strip_fence(response)
    try:
        raw: Any = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"policy response was not valid JSON ({exc}); raw response: {response!r}"
        ) from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"policy response must be a JSON object describing one action, got "
            f"{type(raw).__name__}: {raw!r}"
        )

    kind_raw = str(raw.get("kind") or "").strip().lower()
    try:
        kind = ActionKind(kind_raw)
    except ValueError:
        raise ValueError(
            f"unknown action kind '{kind_raw}'; valid kinds are "
            f"{[k.value for k in ActionKind]}"
        ) from None

    args = raw.get("args") or {}
    if not isinstance(args, dict):
        raise ValueError(f"action 'args' must be an object, got {type(args).__name__}")

    target = str(raw.get("target") or "").strip()
    if kind is not ActionKind.FINISH and not target:
        raise ValueError(f"a '{kind.value}' action must name a target")

    return Action(
        kind=kind,
        target=target,
        args=dict(args),
        rationale=str(raw.get("rationale") or ""),
    )


def _summarize(value: Any) -> str:
    """One observation, trimmed to what the next decision needs."""
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        text = str(value)
    if len(text) <= _MAX_OBSERVATION_CHARS:
        return text
    return f"{text[:_MAX_OBSERVATION_CHARS]}... [truncated, {len(text)} chars total]"


def format_history(iterations: list[Iteration]) -> str:
    """The trace so far, as the prompt sees it.

    Failures are rendered as prominently as successes: an error is the
    most informative thing an iteration can produce, and burying it
    would waste it.
    """
    lines: list[str] = []
    for iteration in iterations:
        action = iteration.action
        observation = iteration.observation
        lines.append(
            f"Iteration {iteration.index}: {action.kind.value} '{action.target}' "
            f"with args {json.dumps(action.args, default=str)}"
        )
        if observation.succeeded:
            lines.append(f"  -> succeeded: {_summarize(observation.content)}")
        else:
            lines.append(f"  -> FAILED: {observation.error}")
    return "\n".join(lines)


class ModelPolicy:
    """Chooses the loop's next action with a real model call.

    Constructed with the tool and specialist *descriptions* rather
    than the objects themselves: the policy decides, `AgentLoop` acts,
    and keeping the live objects out of here means the decision cannot
    accidentally bypass the loop's authorization and budget checks by
    calling something directly.
    """

    def __init__(
        self,
        *,
        catalogue: LLMCatalogue | None = None,
        prompt_manager: PromptManager | None = None,
        tools: list[dict[str, Any]] | None = None,
        agents: list[dict[str, Any]] | None = None,
    ) -> None:
        self._catalogue = catalogue if catalogue is not None else LLMCatalogue()
        self._prompts = prompt_manager if prompt_manager is not None else PromptManager()
        self._tools = tools or []
        self._agents = agents or []

    async def __call__(
        self, contract: TaskContract, iterations: list[Iteration]
    ) -> Action:
        # Only tools this contract actually authorizes. See the module
        # docstring: offering one it does not would spend an iteration
        # discovering the refusal.
        authorized = [
            tool for tool in self._tools if contract.authorizes(str(tool.get("name", "")))
        ]

        prompt = self._prompts.render(
            _PROMPT_NAME,
            _PROMPT_VERSION,
            objective=contract.objective,
            inputs=json.dumps(contract.inputs, default=str) if contract.inputs else "",
            tools=authorized,
            agents=self._agents,
            history=format_history(iterations),
        )
        response = await self._catalogue.complete(_LLM_PURPOSE, prompt)
        action = parse_action(response)
        _logger.info(
            "agent_loop_action_chosen",
            task_id=contract.task_id,
            iteration=len(iterations),
            kind=action.kind.value,
            target=action.target,
        )
        return action
