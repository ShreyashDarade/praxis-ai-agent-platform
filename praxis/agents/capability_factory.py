# praxis/agents/capability_factory.py
"""The Capability Factory (spec §3, §7, §9, §12, §19, §20): synthesizes a
brand-new `Skill` from an LLM, sandbox-validates it for real, and only
then registers it - this is the project's namesake capability.

**Why sandbox validation genuinely gates registration (spec §19/§20)**:
capability synthesis has no human approval gate - a deliberate decision,
not an oversight (§20 "No approval gate on capability synthesis"). The
sandbox and this bounded validation-retry are the *only* backstops, so
they have to be load-bearing, not decorative:

- A skill is written to `praxis/agents/skills/` and imported (which
  triggers its own `register_skill(...)` call - see
  `praxis.agents.skill_registry`) **only after** its generated code has
  actually run inside a real Docker sandbox and its own self-test's
  `SELF_TEST_PASS` marker actually appeared on stdout with a zero exit
  code - never merely because the code parsed or "looked" correct.
- Every attempt that fails is retried up to 2 more times (3 total),
  feeding the sandbox's *real* stdout/stderr back into the next prompt
  so the model can fix the actual problem, not guess again blind. If
  every attempt fails, `SynthesisValidationError` is raised and
  **nothing is written, imported, or registered** - no half-working
  capability, no partial file, no `SkillRecord` row (spec §12).

**Why the sandbox script injects a stub `praxis.agents.skill`/
`skill_registry` instead of mounting the real package**: the generated
module is asked to write `from praxis.agents.skill import Skill` (the
exact real contract - see `praxis/agents/skill.py`), but the sandbox
container (`praxis.sandbox.executor.DockerSandboxExecutor`) is a bare
`python:3.11-slim` image with no `praxis` package installed, no volume
mounts, and (deliberately, per spec §8/§19/§20) no network - so it can
neither `pip install praxis` nor have the real package bind-mounted in
without adding filesystem-path-translation assumptions about wherever
Docker's daemon actually runs (this project's dev machine talks to a
WSL2 dockerd over a TCP proxy, where a Windows host path is not
meaningful). Instead, `_stub_preamble()` reads the *actual*, on-disk
source of `praxis/agents/skill.py` and `praxis/agents/skill_registry.py`
at synthesis time (never a hand-duplicated copy that could drift) and
`exec()`s each into a fresh module object injected directly into
`sys.modules` before the generated code runs - Python's import system
resolves `from praxis.agents.skill import Skill` against whatever is
already in `sys.modules` under that exact dotted name, with no need for
`praxis`/`praxis.agents` to exist as real packages too (verified
directly against the real sandbox while building this). This makes the
sandboxed run genuinely enforce the real `Skill` ABC contract (an
abstract `run` that must actually be implemented, a real
`register_skill` call that really populates a registry) - byte-for-byte
the same contract, not a superficial lookalike - while staying entirely
self-contained, with zero network and zero host-filesystem dependency.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import re
from pathlib import Path
from typing import Any

from praxis.agents import skill as skill_module
from praxis.agents import skill_registry
from praxis.agents.skill import Skill
from praxis.core.exceptions import SynthesisValidationError
from praxis.core.interfaces import Connector, SandboxExecutor
from praxis.llm.catalogue import LLMCatalogue
from praxis.llm.prompt_manager import PromptManager
from praxis.memory.db import PostgresStore
from praxis.memory.graph_store import PgGraphStore
from praxis.memory.models import SkillRecord

_PROMPT_NAME = "synthesize_skill"
_PROMPT_VERSION = "v1"
_LLM_PURPOSE = "code_synthesis"

# 3 attempts total: the first, real attempt, plus up to 2 retries - "a
# bounded retry (2 attempts)" per spec §12.
_MAX_ATTEMPTS = 3
_DEFAULT_SANDBOX_TIMEOUT_SECONDS = 30

# A full skill module + a self-test with several real assertions
# routinely runs to 100+ lines combined; on a retry, the prompt also
# asks the model to reproduce a corrected version of both in the same
# response. Observed directly while building this: a too-small budget
# occasionally truncates the response mid-SELF_TEST, before its closing
# fence - which reads as a malformed response (spec-honest: fed back as
# a real parse-error attempt, exactly like any other failure - never
# silently retried with no explanation), but is really just running out
# of room, not the model failing the task. Generous on purpose.
_MAX_RESPONSE_TOKENS = 8192

# The one signal (spec §12/§19/§20: exit code alone is not enough - "a
# script that prints nothing meaningful" must never pass) that a
# synthesis attempt's own self-test genuinely asserted success.
_SUCCESS_MARKER = "SELF_TEST_PASS"

_DESCRIBES_RELATION = "describes"
_USED_SKILL_RELATION = "used_skill"
_BUILT_AGAINST_RELATION = "built_against"

# Tolerant of "MODULE" / "MODULE:", "SELF_TEST" / "SELF TEST" / "SELF_TEST:",
# and either bare ``` or ```python fences - real model responses vary in
# exactly this kind of formatting even when explicitly instructed (mirrors
# `praxis.agents.planner._strip_code_fence`'s same tolerance).
_RESPONSE_RE = re.compile(
    r"MODULE\s*:?\s*\n*```(?:python)?\s*\n(?P<module>.*?)\n?```"
    r".*?"
    r"SELF[_ ]TEST\s*:?\s*\n*```(?:python)?\s*\n(?P<self_test>.*?)\n?```",
    re.DOTALL | re.IGNORECASE,
)

# Matches `name = "..."` or `name: str = "..."` at any indentation - the
# skill's declared `name` class attribute, which becomes both the
# registry key and the `synthesized_<slug>.py` filename.
_NAME_ATTR_RE = re.compile(r"^\s*name\s*(?::[^=]+)?=\s*(['\"])(?P<name>[^'\"]+)\1", re.MULTILINE)

_SLUG_INVALID_RE = re.compile(r"[^a-z0-9_]+")


def _read_source(module: Any) -> str:
    return Path(inspect.getfile(module)).read_text(encoding="utf-8")


def _stub_preamble() -> str:
    """Builds the sandbox script's preamble: injects real, on-disk
    `praxis.agents.skill` / `praxis.agents.skill_registry` source into
    `sys.modules` under their real dotted names (see module docstring).
    """
    skill_source = _read_source(skill_module)
    registry_source = _read_source(skill_registry)
    return (
        "import sys as _praxis_stub_sys\n"
        "import types as _praxis_stub_types\n"
        f"_praxis_stub_skill_src = {skill_source!r}\n"
        f"_praxis_stub_registry_src = {registry_source!r}\n"
        "_praxis_stub_skill_mod = _praxis_stub_types.ModuleType('praxis.agents.skill')\n"
        "exec(compile(_praxis_stub_skill_src, 'praxis/agents/skill.py', 'exec'), "
        "_praxis_stub_skill_mod.__dict__)\n"
        "_praxis_stub_sys.modules['praxis.agents.skill'] = _praxis_stub_skill_mod\n"
        "_praxis_stub_registry_mod = _praxis_stub_types.ModuleType('praxis.agents.skill_registry')\n"
        "exec(compile(_praxis_stub_registry_src, 'praxis/agents/skill_registry.py', 'exec'), "
        "_praxis_stub_registry_mod.__dict__)\n"
        "_praxis_stub_sys.modules['praxis.agents.skill_registry'] = _praxis_stub_registry_mod\n"
    )


def _build_sandbox_script(module_code: str, self_test_code: str) -> str:
    """One script, three independently-compiled segments sharing one
    namespace (spec §12's "run the generated module code + the self-test
    snippet together, as one script"):

    Each segment is `exec(compile(source, label, "exec"), ns)`'d
    separately (not string-concatenated into one literal file) so each
    keeps its own valid Python syntax - critically, so the generated
    module's own `from __future__ import annotations` (every real Praxis
    skill has one) is legal as *that segment's* first statement, which it
    would not be if pasted after the preamble in one flat file. All three
    segments share one namespace dict so the self-test can see whatever
    the module segment defined.
    """
    segments = (_stub_preamble(), module_code, self_test_code)
    lines = ["_praxis_sandbox_ns = {'__name__': '__main__'}"]
    for position, segment in enumerate(segments):
        lines.append(
            f"exec(compile({segment!r}, 'synthesis_segment_{position}', 'exec'), _praxis_sandbox_ns)"
        )
    return "\n".join(lines)


def _parse_response(response: str) -> tuple[str, str]:
    """Splits a synthesis response into (module_code, self_test_code).

    Raises `ValueError` with a clear message - fed straight back into
    the next attempt's prompt as `error_detail`, exactly like a real
    sandbox failure - when the model's response doesn't contain both
    fenced sections in the requested shape.
    """
    match = _RESPONSE_RE.search(response)
    if not match:
        raise ValueError(
            "model response did not contain both a MODULE and a SELF_TEST fenced "
            f"python code block in the requested shape; raw response: {response!r}"
        )
    module_code = match.group("module").strip() + "\n"
    self_test_code = match.group("self_test").strip() + "\n"
    return module_code, self_test_code


def _extract_skill_name(module_code: str) -> str:
    match = _NAME_ATTR_RE.search(module_code)
    if not match:
        raise ValueError(
            "generated module does not declare a string literal `name` class attribute"
        )
    return match.group("name")


def _slugify(name: str) -> str:
    slug = _SLUG_INVALID_RE.sub("_", name.strip().lower()).strip("_")
    if not slug:
        raise ValueError(f"could not derive a valid module filename slug from skill name {name!r}")
    if slug[0].isdigit():
        slug = f"s_{slug}"
    return slug


class CapabilityFactory:
    """Synthesizes, sandbox-validates, and registers new `Skill`s (spec §3/§7)."""

    def __init__(
        self,
        catalogue: LLMCatalogue,
        prompt_manager: PromptManager,
        sandbox: SandboxExecutor,
        graph_store: PgGraphStore,
        store: PostgresStore,
        skills_dir: Path | str,
        *,
        sandbox_timeout_seconds: int = _DEFAULT_SANDBOX_TIMEOUT_SECONDS,
    ) -> None:
        self._catalogue = catalogue
        self._prompt_manager = prompt_manager
        self._sandbox = sandbox
        self._graph_store = graph_store
        self._store = store
        self._skills_dir = Path(skills_dir)
        self._sandbox_timeout_seconds = sandbox_timeout_seconds

    async def synthesize(
        self,
        need_description: str,
        *,
        connector: Connector | None = None,
        task_id: str | None = None,
    ) -> Skill:
        connector_schema: dict[str, Any] | None = None
        if connector is not None:
            connector_schema = await self._connector_schema(connector)
        connector_schema_text = (
            json.dumps(connector_schema, indent=2, default=str) if connector_schema is not None else None
        )

        previous_code: str | None = None
        previous_self_test: str | None = None
        error_detail: str | None = None
        last_code = ""
        last_detail = ""

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            prompt = self._prompt_manager.render(
                _PROMPT_NAME,
                _PROMPT_VERSION,
                need_description=need_description,
                connector_schema=connector_schema_text,
                previous_code=previous_code,
                previous_self_test=previous_self_test,
                error_detail=error_detail,
            )
            response = await self._catalogue.complete(
                _LLM_PURPOSE, prompt, max_tokens=_MAX_RESPONSE_TOKENS
            )

            try:
                module_code, self_test_code = _parse_response(response)
            except ValueError as exc:
                last_code = response
                last_detail = f"attempt {attempt}/{_MAX_ATTEMPTS}: {exc}"
                previous_code, previous_self_test, error_detail = response, "", last_detail
                continue

            script = _build_sandbox_script(module_code, self_test_code)
            result = await self._sandbox.run(script, timeout_seconds=self._sandbox_timeout_seconds)

            if result.exit_code == 0 and _SUCCESS_MARKER in result.stdout:
                return await self._register(module_code, connector=connector, task_id=task_id)

            last_code = module_code
            last_detail = (
                f"attempt {attempt}/{_MAX_ATTEMPTS}: sandbox exit_code={result.exit_code}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
            previous_code, previous_self_test, error_detail = module_code, self_test_code, last_detail

        raise SynthesisValidationError(
            f"synthesis of capability '{need_description}' failed sandbox validation after "
            f"{_MAX_ATTEMPTS} attempts",
            code=last_code,
            detail=last_detail,
        )

    # ------------------------------------------------------------------ #
    # Connector schema caching (spec §9: "GraphStore - connector schemas
    # already introspected, avoids re-introspecting the same DB every ask")
    # ------------------------------------------------------------------ #

    async def _connector_schema(self, connector: Connector) -> dict[str, Any]:
        cached = await self._graph_store.neighbors(connector.name, relation=_DESCRIBES_RELATION)
        if cached:
            return cached[0]["metadata"]["schema"]

        description = await connector.describe()
        digest = hashlib.sha256(
            json.dumps(description.schema, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        await self._graph_store.add_edge(
            source=connector.name,
            relation=_DESCRIBES_RELATION,
            target=f"schema:{digest}",
            metadata={"kind": description.kind, "schema": description.schema},
        )
        return description.schema

    # ------------------------------------------------------------------ #
    # Registration (only ever reached after a real sandbox pass)
    # ------------------------------------------------------------------ #

    async def _register(
        self, module_code: str, *, connector: Connector | None, task_id: str | None
    ) -> Skill:
        skill_name = _extract_skill_name(module_code)
        module_name = f"synthesized_{_slugify(skill_name)}"
        file_path = self._skills_dir / f"{module_name}.py"
        file_path.write_text(module_code, encoding="utf-8")

        # `self._skills_dir` is expected to be the real
        # `praxis/agents/skills/` package directory (spec: "write the
        # generated module to praxis/agents/skills/synthesized_<slug>.py")
        # - the dotted import path below assumes exactly that, which is
        # also what makes `discover_skills()`'s future package scan pick
        # this file up on a cold restart with zero code changes.
        dotted_module = f"praxis.agents.skills.{module_name}"
        importlib.import_module(dotted_module)

        registered_skill = skill_registry.get_skill(skill_name)

        async with self._store.session() as session:
            session.add(
                SkillRecord(
                    name=skill_name,
                    risk=registered_skill.risk,
                    synthesized=True,
                    inputs_schema=dict(registered_skill.inputs),
                    outputs_schema=dict(registered_skill.outputs),
                )
            )
            await session.commit()

        # Lineage graph (spec §9): every synthesized skill is a node
        # linked to the task that needed it and the connector schema it
        # was built against.
        if task_id is not None:
            await self._graph_store.add_edge(
                source=task_id, relation=_USED_SKILL_RELATION, target=skill_name
            )
        if connector is not None:
            await self._graph_store.add_edge(
                source=skill_name, relation=_BUILT_AGAINST_RELATION, target=connector.name
            )

        return registered_skill
