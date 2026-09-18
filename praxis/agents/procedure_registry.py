# praxis/agents/procedure_registry.py
"""Loading declarative skills - from disk at startup, and from the API.

Until now `skills/*/SKILL.md` were documentation. `SkillManifest` could
parse that format, and nothing called the parser outside a unit test,
so the coverage matrix's "SKILL.md format: Built" described a parser
rather than a runtime capability. This module is what makes the format
load.

Two entry points, one path:

- `load_from_disk()` scans the `skills/` tree at startup, so a
  procedure that ships with the deployment is available without an API
  call.
- `register_procedure()` takes markdown submitted through
  `POST /skills`, which is how a user adds one at runtime.

Both compile through `praxis.agents.procedure`, so a procedure loaded
from disk gets exactly the same validation as one posted over HTTP -
tools must exist, the chain must be wirable, and declared risk must
match what the composed tools actually do.

**Approval applies to procedures too, and for a narrower reason.** A
procedure cannot execute arbitrary code - it can only call already-
approved tools - so it is genuinely less dangerous than generated
Python. It is not harmless: it chooses *which* permitted tools run,
with which arguments, and a mutating chain is still a mutating chain.
So it goes through the same catalogue and the same gate, and the same
`require_skill_approval` setting governs both. The difference in risk
shows up in how easy it is to review - a reviewer reads five lines of
YAML instead of auditing a module - not in whether review happens.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from praxis.agents import skill_registry
from praxis.agents.manifest import ManifestError, SkillManifest, compute_code_hash
from praxis.agents.procedure import ProcedureError, ProcedureSkill, SkillLookup
from praxis.agents.publication import SkillStatus
from praxis.memory.db import PostgresStore
from praxis.memory.models import DEFAULT_TENANT_ID, SkillRecord

_logger = structlog.get_logger(__name__)

# Procedures that ship with the product, INSIDE the package.
#
# They were briefly at the repository root, on the reasoning that
# configuration is not code and so does not belong in the Python
# package. That reasoning was aesthetic and the consequence was a real
# defect: hatchling packages `praxis/` only, so a built wheel contained
# none of them and `pip install praxis` silently had no built-in
# procedures at all - they worked from a repo checkout and nowhere
# else.
#
# `praxis/llm/prompts/*.jinja2` is the precedent that settles it. Those
# are content too, and they live inside the package precisely so they
# ship. A `.md` file in a package directory is a data file; it is not
# importable and being there does not make it code.
BUILTIN_PROCEDURES_DIR = Path(__file__).resolve().parent.parent / "procedures"


def operator_procedures_dir() -> Path | None:
    """An additional directory this deployment supplies, if configured.

    This is the part that genuinely is deployment configuration: a
    mounted volume of procedures an operator maintains separately from
    the product. Loaded after the built-ins, so an operator can add
    capabilities without rebuilding an image - and, because the
    registry keeps the first registration of a name, cannot silently
    shadow a shipped one.
    """
    try:
        from praxis.config import Settings

        configured = Settings().procedures_dir
    except Exception:  # noqa: BLE001 - unconfigured is not an error here
        return None
    return Path(configured) if configured else None


def _lookup() -> SkillLookup:
    return SkillLookup(skill_registry.get_skill)


def compile_markdown(text: str, *, source_path: str = "") -> ProcedureSkill:
    """Parses and compiles one SKILL.md into a runnable procedure.

    Raises `ManifestError` for a malformed document and
    `ProcedureError` for one that parses but cannot be wired - two
    genuinely different problems, kept apart so the API can report
    which one happened.
    """
    manifest = SkillManifest.from_markdown(text, source_path=source_path)
    return ProcedureSkill(manifest, _lookup())


def load_from_disk(directory: Path | str | None = None) -> list[str]:
    """Registers every `SKILL.md` found. Returns their names.

    With no `directory`, loads the procedures that ship inside the
    package and then any the deployment configured, in that order.

    A file that fails to parse or compile is logged and skipped rather
    than taking startup down: one malformed procedure must not stop the
    application from serving every other capability. The log line names
    the file, so the failure is findable rather than merely survived.
    """
    if directory is not None:
        roots = [Path(directory)]
    else:
        roots = [BUILTIN_PROCEDURES_DIR]
        operator_dir = operator_procedures_dir()
        if operator_dir is not None:
            roots.append(operator_dir)

    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(sorted(root.rglob("SKILL.md")))

    loaded: list[str] = []
    for path in paths:
        try:
            procedure = compile_markdown(path.read_text(encoding="utf-8"), source_path=str(path))
        except (ManifestError, ProcedureError) as exc:
            _logger.warning(
                "procedure_skipped", path=str(path), error=str(exc)
            )
            continue

        try:
            skill_registry.register_skill(procedure)
        except ValueError:
            # Already registered - a re-scan, or a hand-written skill
            # owns the name. Not an error; the existing one wins.
            _logger.info("procedure_already_registered", skill=procedure.name)
            continue
        loaded.append(procedure.name)
        _logger.info(
            "procedure_loaded", skill=procedure.name, path=str(path), risk=procedure.risk
        )
    return loaded


async def register_procedure(
    store: PostgresStore,
    markdown: str,
    *,
    tenant_id: str = DEFAULT_TENANT_ID,
    created_by_user_id: str | None = None,
    require_approval: bool = True,
) -> tuple[SkillManifest, SkillRecord, bool]:
    """Validates submitted markdown and catalogues it.

    Returns `(manifest, record, active)`. `active` is False when the
    procedure is catalogued but awaiting approval, in which case it is
    deliberately NOT registered as runnable - the same posture as a
    quarantined generated skill, for the same reason: being in the
    catalogue is bookkeeping, and being callable is a decision.

    Compilation happens first, and against the live registry, so a
    procedure naming a tool that does not exist is rejected at submit
    time with a clear message rather than accepted and then failing for
    whoever runs it.
    """
    procedure = compile_markdown(markdown)
    manifest = procedure.manifest
    code_hash = compute_code_hash(markdown)

    status = (
        SkillStatus.PENDING_APPROVAL.value if require_approval else SkillStatus.ACTIVE.value
    )

    async with store.session() as session:
        record = SkillRecord(
            tenant_id=tenant_id,
            name=manifest.name,
            risk=procedure.risk,
            # Not `synthesized`: nothing generated this. It is authored
            # configuration, and conflating the two would make the
            # approval queue unable to show a reviewer which kind of
            # thing they are looking at.
            synthesized=False,
            status=status,
            inputs_schema=dict(manifest.inputs),
            outputs_schema=dict(manifest.outputs),
            code_hash=code_hash,
            source_path=manifest.source_path or "(submitted via API)",
            # The document itself, so a procedure with no file
            # behind it survives and approval can recompile from
            # exactly what was reviewed.
            definition=markdown,
            description=manifest.description,
            owner=created_by_user_id or manifest.owner,
            # The tools this procedure composes, recorded so a
            # reviewer can see its whole blast radius from the
            # catalogue row without re-reading the markdown.
            dependencies=procedure.tool_names,
            required_permissions=list(manifest.required_permissions),
            supported_connectors=list(manifest.supported_connectors),
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)

    active = status == SkillStatus.ACTIVE.value
    if active:
        try:
            skill_registry.register_skill(procedure)
        except ValueError:
            _logger.info("procedure_already_registered", skill=procedure.name)

    _logger.info(
        "procedure_registered",
        skill=manifest.name,
        risk=procedure.risk,
        tools=procedure.tool_names,
        status=status,
        tenant_id=tenant_id,
    )
    return manifest, record, active


def activate_procedure(markdown: str) -> ProcedureSkill:
    """Makes an approved procedure runnable.

    The counterpart to moving generated code out of quarantine: an
    approved procedure is compiled fresh from the reviewed markdown and
    registered. Compiling again rather than trusting an earlier object
    is deliberate - it is the same "bind the approval to the actual
    bytes" property the code path has.
    """
    procedure = compile_markdown(markdown)
    try:
        skill_registry.register_skill(procedure)
    except ValueError:
        skill_registry._SKILLS[procedure.name] = procedure  # replace in place
    _logger.info("procedure_activated", skill=procedure.name)
    return procedure


def describe_loaded() -> list[dict[str, Any]]:
    """Every procedure currently registered, for the API's listing."""
    described: list[dict[str, Any]] = []
    for skill in skill_registry.all_skills():
        if isinstance(skill, ProcedureSkill):
            described.append(
                {
                    "name": skill.name,
                    "description": skill.description,
                    "risk": skill.risk,
                    "tools": skill.tool_names,
                    "inputs": skill.inputs,
                    "outputs": skill.outputs,
                    "source_path": skill.manifest.source_path,
                }
            )
    return described
