# praxis/agents/toollink.py
"""Tool links: composing tools without generating code.

Prompt §7 names four ways to get a capability, in preference order:
*"reuse an existing tool, discover an approved MCP tool, **compose
existing tools programmatically**, or generate a genuinely new
executable tool. Prefer the first three."*

Praxis had the first and the fourth. Composition - the third, and the
one that should be reached for far more often than synthesis - had no
representation at all: the only way to express "run this query, then
chart its result" was either a Planner-authored multi-step plan or a
newly synthesized skill.

**A tool link is declarative wiring, not generated code.** A
`ToolLink` says "call tool A, take key `rows` from its output, feed it
to tool B's `data` parameter". That is data, validated before
anything runs, and it is strictly safer than synthesizing a skill to
do the same thing: no new code exists, so there is nothing to
sandbox, review, or approve.

**Why the mapping is validated up front.** The declared `outputs` of
the producing tool and the declared `inputs` of the consuming tool are
both known statically. A chain wiring `rows -> data` can be checked
for the producer actually declaring `rows` and the consumer actually
accepting `data` *before* the first tool runs - turning what would
otherwise be a mid-execution `KeyError` after real work has already
happened into a refusal at build time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import structlog

_logger = structlog.get_logger(__name__)


class LinkableTool(Protocol):
    """What a tool must expose to participate in a link.

    Satisfied by `praxis.agents.skill.Skill` as-is - no adapter, no
    registration, no change to the `Skill` contract. Composition works
    with the tools that already exist, which is the point.
    """

    name: str
    inputs: dict[str, str]
    outputs: dict[str, str]

    async def run(self, **kwargs: Any) -> Any: ...


class ToolLinkError(Exception):
    """A chain is not wirable, or failed mid-execution.

    `stage` names which step, so a failure in a five-tool chain says
    where rather than just that something went wrong.
    """

    def __init__(self, message: str, *, stage: str = "", reason: str = "") -> None:
        super().__init__(message)
        self.stage = stage
        self.reason = reason


@dataclass(frozen=True)
class ToolLink:
    """One step: a tool plus how its inputs are wired.

    `static_args` are fixed values. `mapped_args` wire an upstream
    output into this tool's input: `{"data": "query.rows"}` means
    "this tool's `data` parameter comes from the step named `query`'s
    `rows` output".

    Referencing a step *by name* rather than by index is deliberate -
    an index-based reference silently points at the wrong step the
    moment anyone reorders the chain, and reordering is exactly what
    someone editing a composition does.
    """

    tool: LinkableTool
    step_name: str
    static_args: dict[str, Any] = field(default_factory=dict)
    mapped_args: dict[str, str] = field(default_factory=dict)
    # When set, the chain stops (successfully) if this returns False
    # for the step's output - the guard that lets a composition say
    # "if there are no rows, don't bother charting them".
    continue_if: str | None = None


@dataclass
class ToolChainResult:
    """What a whole chain produced."""

    outputs: dict[str, Any] = field(default_factory=dict)
    steps_run: list[str] = field(default_factory=list)
    stopped_early_at: str | None = None
    stop_reason: str = ""

    @property
    def final(self) -> Any:
        """The last step's output - the usual thing a caller wants."""
        if not self.steps_run:
            return None
        return self.outputs.get(self.steps_run[-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "outputs": self.outputs,
            "steps_run": self.steps_run,
            "stopped_early_at": self.stopped_early_at,
            "stop_reason": self.stop_reason,
        }


def _resolve_reference(reference: str, produced: dict[str, Any]) -> Any:
    """Resolves `step.key` (or bare `step`) against prior outputs."""
    step_name, _, key = reference.partition(".")
    if step_name not in produced:
        raise ToolLinkError(
            f"reference '{reference}' names step '{step_name}', which has not run",
            stage=step_name,
            reason="unknown_step",
        )
    value = produced[step_name]
    if not key:
        return value
    if not isinstance(value, dict) or key not in value:
        available = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ToolLinkError(
            f"step '{step_name}' produced no key '{key}' (available: {available})",
            stage=step_name,
            reason="unknown_output_key",
        )
    return value[key]


class ToolChain:
    """A validated, declarative composition of existing tools."""

    def __init__(self, links: list[ToolLink], *, name: str = "chain") -> None:
        self.name = name
        self._links = list(links)
        self.validate()

    @property
    def links(self) -> list[ToolLink]:
        return list(self._links)

    def validate(self) -> None:
        """Checks the whole chain is wirable before anything runs.

        Every failure here would otherwise surface mid-execution,
        after earlier tools had already done real work (and possibly
        mutated something).
        """
        seen: dict[str, LinkableTool] = {}

        for link in self._links:
            if link.step_name in seen:
                raise ToolLinkError(
                    f"duplicate step name '{link.step_name}' in chain '{self.name}'",
                    stage=link.step_name,
                    reason="duplicate_step",
                )

            declared_inputs = set(link.tool.inputs or {})
            supplied = set(link.static_args) | set(link.mapped_args)

            unknown = supplied - declared_inputs
            if declared_inputs and unknown:
                raise ToolLinkError(
                    (
                        f"step '{link.step_name}' supplies argument(s) {sorted(unknown)} "
                        f"that tool '{link.tool.name}' does not declare "
                        f"(declared: {sorted(declared_inputs)})"
                    ),
                    stage=link.step_name,
                    reason="unknown_argument",
                )

            missing = declared_inputs - supplied
            if missing:
                raise ToolLinkError(
                    (
                        f"step '{link.step_name}' does not supply required input(s) "
                        f"{sorted(missing)} for tool '{link.tool.name}'"
                    ),
                    stage=link.step_name,
                    reason="missing_argument",
                )

            for target, reference in link.mapped_args.items():
                source_step, _, key = reference.partition(".")
                if source_step not in seen:
                    raise ToolLinkError(
                        (
                            f"step '{link.step_name}' maps '{target}' from '{reference}', but "
                            f"no earlier step is named '{source_step}'"
                        ),
                        stage=link.step_name,
                        reason="unknown_step",
                    )
                if key:
                    producer_outputs = set(seen[source_step].outputs or {})
                    if producer_outputs and key not in producer_outputs:
                        raise ToolLinkError(
                            (
                                f"step '{link.step_name}' maps '{target}' from "
                                f"'{reference}', but step '{source_step}' "
                                f"(tool '{seen[source_step].name}') declares outputs "
                                f"{sorted(producer_outputs)}"
                            ),
                            stage=link.step_name,
                            reason="unknown_output_key",
                        )

            seen[link.step_name] = link.tool

    async def run(self, **injected: Any) -> ToolChainResult:
        """Executes the chain, threading outputs into later inputs.

        `injected` values (e.g. the orchestrator's `tenant_id` and
        `known_urls`) are passed to every tool, exactly as the
        orchestrator already does for a single skill - so a composed
        tool is subject to the same tenant scoping and URL allow-list
        as one called directly.
        """
        result = ToolChainResult()

        for link in self._links:
            kwargs = dict(link.static_args)
            for target, reference in link.mapped_args.items():
                kwargs[target] = _resolve_reference(reference, result.outputs)

            _logger.info("tool_link_step_started", chain=self.name, step=link.step_name)
            try:
                output = await link.tool.run(**kwargs, **injected)
            except Exception as exc:
                raise ToolLinkError(
                    f"step '{link.step_name}' (tool '{link.tool.name}') failed: {exc}",
                    stage=link.step_name,
                    reason="tool_failed",
                ) from exc

            result.outputs[link.step_name] = output
            result.steps_run.append(link.step_name)

            if link.continue_if is not None and not _guard_passes(link.continue_if, output):
                result.stopped_early_at = link.step_name
                result.stop_reason = (
                    f"guard '{link.continue_if}' was not satisfied by step "
                    f"'{link.step_name}'"
                )
                _logger.info(
                    "tool_link_stopped_early",
                    chain=self.name,
                    step=link.step_name,
                    guard=link.continue_if,
                )
                break

        return result


# Guards are named predicates, not expressions: evaluating a
# caller-supplied expression string would be an arbitrary-code
# execution path in something an LLM may well have authored. A small
# closed set covers the real cases and cannot be escaped.
_GUARDS: dict[str, Any] = {
    "non_empty": lambda value: bool(value)
    and (not isinstance(value, dict) or any(bool(v) for v in value.values())),
    "has_rows": lambda value: bool(
        value.get("rows") if isinstance(value, dict) else value
    ),
    "succeeded": lambda value: bool(
        value.get("succeeded", True) if isinstance(value, dict) else value
    ),
}


def _guard_passes(guard: str, output: Any) -> bool:
    check = _GUARDS.get(guard)
    if check is None:
        raise ToolLinkError(
            f"unknown guard '{guard}'; known guards: {sorted(_GUARDS)}",
            reason="unknown_guard",
        )
    return bool(check(output))
