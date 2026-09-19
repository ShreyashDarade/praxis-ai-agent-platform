# praxis/agents/procedure.py
"""Skills that are configuration, not code.

The brief's preferred order for a new capability is: reuse an existing
tool, discover an approved one, **compose existing tools**, and only
then generate new executable code. Praxis had the first and the last.
This module is the middle one, and it is the brief's main lever for
"less code": a procedure that combines already-approved tools needs a
SKILL.md and nothing else.

**Why this is safe in a way generated Python is not.** A procedure
cannot execute arbitrary code. It can only name tools that already
exist and were already approved, wire their inputs together, and run
them in order. Everything a normal skill is subject to - the
Orchestrator's approval gate for mutating steps, tenant injection,
output validation - applies unchanged, because a procedure *is* a
`Skill` and its steps are ordinary tool calls. The worst a bad
procedure can do is call permitted tools in a silly order.

That is also why the validation below is strict and happens at load
time rather than at run time. A procedure naming a tool that does not
exist, or declaring itself read-only while calling something mutating,
is refused when it is submitted - not discovered halfway through a run
after earlier steps have already done real work.

**Risk is derived, never trusted.** A manifest that says
`risk: read_only` while step 2 calls a mutating tool is rejected
outright rather than quietly corrected. Trusting the declaration would
let a procedure launder a mutating action past the approval gate, which
is exactly the shape of defect the MCP read path had.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import structlog

from praxis.agents.manifest import SkillManifest
from praxis.agents.skill import Skill
from praxis.agents.toollink import ToolChain, ToolLink, ToolLinkError

_logger = structlog.get_logger(__name__)


# How a step refers to one of the procedure's own inputs, as opposed to
# an earlier step's output.
_INPUT_PREFIX = "$."


@dataclass(frozen=True)
class _StepSpec:
    """One declared step, before its input references are resolved."""

    tool: Skill
    step_name: str
    static_args: dict[str, Any]
    from_step: dict[str, str]
    from_input: dict[str, str]
    continue_if: str | None = None


class ProcedureError(Exception):
    """A declarative procedure cannot be compiled into a runnable chain."""

    def __init__(self, message: str, *, skill: str = "", reason: str = "") -> None:
        super().__init__(message)
        self.skill = skill
        self.reason = reason


class SkillLookup:
    """What a procedure needs in order to resolve its tools.

    A tiny protocol-shaped wrapper rather than importing the registry
    module directly, so a test can compile a procedure against a
    handful of fake tools without touching global state.
    """

    def __init__(self, get_skill: Any) -> None:
        self._get_skill = get_skill

    def get(self, name: str) -> Skill:
        return self._get_skill(name)


def _derive_risk(tools: list[Skill]) -> Literal["read_only", "mutating"]:
    """The real risk of a composition: the riskiest thing in it.

    A chain that calls one mutating tool is a mutating chain, whatever
    its author wrote in the frontmatter.
    """
    if any(t.risk == "mutating" for t in tools):
        return "mutating"
    return "read_only"


class ProcedureSkill(Skill):
    """A `Skill` whose behaviour is declared in a SKILL.md, not coded.

    Compiles the manifest's `steps` into a `ToolChain` at construction,
    so an unwirable procedure fails when it is loaded rather than when
    someone finally runs it.
    """

    def __init__(self, manifest: SkillManifest, lookup: SkillLookup) -> None:
        self.manifest = manifest
        self.name = manifest.name
        self.description = manifest.description
        self._declared_inputs = dict(manifest.inputs)
        self._declared_outputs = dict(manifest.outputs)

        if not manifest.steps:
            raise ProcedureError(
                f"procedure '{manifest.name}' declares no steps; a SKILL.md without "
                "steps is documentation, not a runnable capability",
                skill=manifest.name,
                reason="no_steps",
            )

        specs: list[_StepSpec] = []
        tools: list[Skill] = []
        for index, step in enumerate(manifest.steps):
            tool_name = str(step.get("tool") or "").strip()
            if not tool_name:
                raise ProcedureError(
                    f"procedure '{manifest.name}' step {index} names no tool",
                    skill=manifest.name,
                    reason="missing_tool",
                )
            try:
                tool = lookup.get(tool_name)
            except KeyError as exc:
                raise ProcedureError(
                    (
                        f"procedure '{manifest.name}' step {index} calls '{tool_name}', "
                        "which is not a registered skill; a procedure may only compose "
                        "tools that already exist"
                    ),
                    skill=manifest.name,
                    reason="unknown_tool",
                ) from exc

            tools.append(tool)

            # `from:` carries two different kinds of reference, and they
            # resolve at different times:
            #
            #   data: query.rows     -> an earlier STEP's output, known
            #                           only while the chain runs
            #   sql: $.sql           -> one of this PROCEDURE's own
            #                           inputs, known only when someone
            #                           calls it
            #
            # Step references are `ToolLink.mapped_args`, which the
            # chain resolves. Input references cannot be, because the
            # chain has never heard of the procedure - so they are
            # recorded here and substituted into `static_args` at call
            # time, when the values actually exist.
            wiring = dict(step.get("from") or {})
            from_step = {
                str(k): str(v)
                for k, v in wiring.items()
                if not str(v).startswith(_INPUT_PREFIX)
            }
            from_input = {
                str(k): str(v)[len(_INPUT_PREFIX):]
                for k, v in wiring.items()
                if str(v).startswith(_INPUT_PREFIX)
            }
            unknown_inputs = set(from_input.values()) - set(self._declared_inputs)
            if unknown_inputs:
                raise ProcedureError(
                    (
                        f"procedure '{manifest.name}' step '{step.get('name') or index}' "
                        f"reads input(s) {sorted(unknown_inputs)} that the procedure does "
                        "not declare under `inputs`"
                    ),
                    skill=manifest.name,
                    reason="unknown_input",
                )

            specs.append(
                _StepSpec(
                    tool=tool,
                    step_name=str(step.get("name") or f"step_{index}"),
                    static_args=dict(step.get("args") or {}),
                    from_step=from_step,
                    from_input=from_input,
                    continue_if=step.get("continue_if"),
                )
            )

        actual_risk = _derive_risk(tools)
        if manifest.risk != actual_risk and actual_risk == "mutating":
            # Refused, not corrected. A procedure that under-declares its
            # risk would otherwise slip a mutating action past the
            # Orchestrator's approval gate, and silently fixing it would
            # hide that someone wrote it down wrong.
            mutating = [t.name for t in tools if t.risk == "mutating"]
            raise ProcedureError(
                (
                    f"procedure '{manifest.name}' declares risk '{manifest.risk}' but "
                    f"calls mutating tool(s) {mutating}; declare 'risk: mutating'"
                ),
                skill=manifest.name,
                reason="risk_understated",
            )
        self.risk = actual_risk

        self._specs = specs
        # Validated once, now, against placeholder values for the
        # inputs - so an unwirable procedure is refused at load time
        # rather than discovered by whoever first runs it.
        try:
            self._build_chain({name: None for name in self._declared_inputs}).validate()
        except ToolLinkError as exc:
            raise ProcedureError(
                f"procedure '{manifest.name}' is not wirable: {exc}",
                skill=manifest.name,
                reason="not_wirable",
            ) from exc

        self._tool_names = [t.name for t in tools]

    @property
    def inputs(self) -> dict[str, str]:  # type: ignore[override]
        return dict(self._declared_inputs)

    @property
    def outputs(self) -> dict[str, str]:  # type: ignore[override]
        return dict(self._declared_outputs)

    @property
    def tool_names(self) -> list[str]:
        """The tools this procedure composes. Used by the approval view."""
        return list(self._tool_names)

    def _build_chain(self, supplied: dict[str, Any]) -> ToolChain:
        """Materialises the chain for one call, substituting inputs."""
        links = [
            ToolLink(
                tool=spec.tool,
                step_name=spec.step_name,
                static_args={
                    **spec.static_args,
                    **{
                        target: supplied.get(source)
                        for target, source in spec.from_input.items()
                    },
                },
                mapped_args=dict(spec.from_step),
                continue_if=spec.continue_if,
            )
            for spec in self._specs
        ]
        return ToolChain(links, name=self.name)

    async def run(self, **kwargs: Any) -> Any:
        """Runs the composed chain.

        Every keyword the Orchestrator injects - `tenant_id` above all -
        is forwarded to each tool by `ToolChain.run`, so a composed
        capability is scoped exactly as a directly-called one is.
        """
        # The procedure's declared inputs are consumed here, by being
        # substituted into the steps that asked for them. They are then
        # removed from what is forwarded, so a tool cannot receive the
        # same argument twice - once substituted and once injected -
        # which Python rejects outright.
        supplied = {name: kwargs.get(name) for name in self._declared_inputs}
        forwarded = {k: v for k, v in kwargs.items() if k not in self._declared_inputs}
        result = await self._build_chain(supplied).run(**forwarded)
        return {
            "outputs": result.outputs,
            "steps_run": result.steps_run,
            "stopped_early_at": result.stopped_early_at,
        }


def compile_procedure(manifest: SkillManifest, lookup: SkillLookup) -> ProcedureSkill:
    """Builds a runnable procedure, or raises `ProcedureError`."""
    procedure = ProcedureSkill(manifest, lookup)
    _logger.info(
        "procedure_compiled",
        skill=procedure.name,
        risk=procedure.risk,
        tools=procedure.tool_names,
    )
    return procedure

