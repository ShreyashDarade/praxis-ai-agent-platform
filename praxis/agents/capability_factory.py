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

import ast
import hashlib
import importlib
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from praxis.agents import skill as skill_module
from praxis.agents import skill_registry
from praxis.agents.manifest import compute_code_hash
from praxis.agents.publication import SkillStatus
from praxis.agents.skill import Skill
from praxis.cache import scopes
from praxis.cache.keys import CacheKey
from praxis.cache.memory_cache import InMemoryCache
from praxis.connectors.retry import call_with_retry
from praxis.core.exceptions import (
    SkillPendingApprovalError,
    SynthesisValidationError,
)
from praxis.core.interfaces import Cache, Connector, SandboxExecutor
from praxis.llm.catalogue import LLMCatalogue
from praxis.llm.prompt_manager import PromptManager
from praxis.memory.db import PostgresStore
from praxis.memory.graph_store import PgGraphStore
from praxis.memory.models import DEFAULT_TENANT_ID, SkillRecord
from praxis.observability.tracing import start_span
from praxis.security.principal import Principal

_logger = structlog.get_logger(__name__)

_PROMPT_NAME = "synthesize_skill"
_PROMPT_VERSION = "v1"
_LLM_PURPOSE = "code_synthesis"

# 3 attempts total: the first, real attempt, plus up to 2 retries - "a
# bounded retry (2 attempts)" per spec §12.
_MAX_ATTEMPTS = 3
_DEFAULT_SANDBOX_TIMEOUT_SECONDS = 30

# Short TTL (spec §11: "Connector/schema-introspection cache ... fronts
# the GraphStore schema records in §9, short TTL") - long enough to
# skip re-introspection across the several syntheses one task typically
# triggers back-to-back, short enough that a connector's schema
# actually changing is noticed again well within a deployment's
# lifetime, unlike the GraphStore record itself (durable, unbounded).
# Read from `praxis.cache.scopes` rather than restated here, so the TTL
# and the reasoning for it stay in one place.
_SCHEMA_CACHE_TTL_SECONDS = scopes.ttl_for(scopes.CONNECTOR_SCHEMA)

# A full skill module + a self-test with several real assertions
# routinely runs to 100+ lines combined; on a retry, the prompt also
# asks the model to reproduce a corrected version of both in the same
# response. Observed directly while building this: a too-small budget
# occasionally truncates the response mid-SELF_TEST, before its closing
# fence - which reads as a malformed response (spec-honest: fed back as
# a real parse-error attempt, exactly like any other failure - never
# silently retried with no explanation), but is really just running out
# of room, not the model failing the task.
#
# `None` means uncapped, which is now the honest setting rather than a
# generous guess. Synthesis is the most reasoning-heavy call this
# system makes, and on a reasoning model (gpt-5, o-series) the cap
# covers thinking as well as the emitted code - so any fixed number
# here is a number that truncates some legitimate synthesis, and the
# symptom is a malformed-response retry loop that never converges.
_MAX_RESPONSE_TOKENS = None

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


@dataclass(frozen=True)
class _StaticSkillMetadata:
    """A generated skill's declared shape, read WITHOUT importing it.

    Importing a module executes every statement at its top level, so
    reading metadata by import would run generated code in the API
    process before any human approved it - which is exactly the control
    the approval gate exists to provide. `ast.parse` builds the tree
    without executing anything, and `literal_eval` refuses anything that
    is not a plain literal, so a `name` computed by calling out to the
    network simply fails to parse rather than running.
    """

    name: str
    risk: str
    inputs: dict[str, str]
    outputs: dict[str, str]
    docstring: str


def _extract_skill_metadata(module_code: str) -> _StaticSkillMetadata:
    """Reads the generated `Skill` subclass's declared attributes statically."""
    tree = ast.parse(module_code)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        attrs: dict[str, Any] = {}
        for stmt in node.body:
            target: ast.expr | None = None
            value: ast.expr | None = None
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target, value = stmt.targets[0], stmt.value
            elif isinstance(stmt, ast.AnnAssign):
                target, value = stmt.target, stmt.value
            if not isinstance(target, ast.Name) or value is None:
                continue
            if target.id not in ("name", "risk", "inputs", "outputs"):
                continue
            try:
                attrs[target.id] = ast.literal_eval(value)
            except ValueError:
                # Not a literal. Treated as absent rather than trusted -
                # a computed attribute is precisely what must not run.
                continue
        if "name" in attrs:
            return _StaticSkillMetadata(
                name=str(attrs["name"]),
                risk=str(attrs.get("risk", "mutating")),
                inputs={str(k): str(v) for k, v in dict(attrs.get("inputs", {})).items()},
                outputs={str(k): str(v) for k, v in dict(attrs.get("outputs", {})).items()},
                docstring=next(iter((ast.get_docstring(node) or '').splitlines()), ''),
            )
    raise SynthesisValidationError(
        "the generated module declares no Skill subclass with a `name` attribute",
        code=module_code,
        detail="static analysis found no class with a literal `name` class attribute",
    )


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
        "_praxis_stub_registry_mod = _praxis_stub_types.ModuleType("
        "'praxis.agents.skill_registry')\n"
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
            f"exec(compile({segment!r}, 'synthesis_segment_{position}', 'exec'), "
            "_praxis_sandbox_ns)"
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


# Phase 11 (spec §16.2's dashboard walkthrough, adapted per that phase's
# scope decision): matches a SQLAlchemy SQLite DSN in either shape
# (`sqlite:///path` or `sqlite+aiosqlite:///path`) - the one SQL dialect
# whose driver (`sqlite3`) is part of the Python standard library, and
# therefore the one dialect a *synthesized* skill can actually be
# sandbox-validated against for real (see `_connector_access_hint`
# below).
_SQLITE_DSN_RE = re.compile(r"^sqlite(?:\+\w+)?:///(?P<path>.+)$")


def _sqlite_path_from_dsn(dsn: str) -> str | None:
    """Extracts the raw filesystem path from a SQLite DSN, or `None` if
    `dsn` isn't one - see `_connector_access_hint`."""
    match = _SQLITE_DSN_RE.match(dsn)
    return match.group("path") if match else None


def _connector_access_hint(connector: Connector) -> str | None:
    """A hint appended to the synthesis prompt telling the model exactly
    how its generated code can reach `connector`'s real underlying data
    using *only* the Python standard library.

    Why this exists (see this module's own docstring for the sandbox's
    "stdlib only, no network" constraint): the introspected
    `connector_schema` alone tells the model *what shape* the data is,
    but says nothing about *how to actually reach it* from inside a
    network-isolated sandbox with no third-party packages installed -
    a real, structural gap for any capability that needs to genuinely
    query an external database, not just describe one. This is
    currently only resolvable for a `SQLConnector` backed by a local
    SQLite file (the one SQL dialect with a Python-stdlib driver,
    `sqlite3`) - returns `None` for every other connector (a real,
    honest limit, not a guess papered over: the model is left with only
    the schema for those, exactly as before this phase).
    """
    dsn = getattr(connector, "dsn", None)
    if not isinstance(dsn, str):
        return None
    sqlite_path = _sqlite_path_from_dsn(dsn)
    if sqlite_path is None:
        return None
    return (
        "This connector's underlying data is a local SQLite database file, "
        "reachable directly with Python's standard library `sqlite3` module "
        f"(no third-party driver, no network required) at the literal path "
        f"{sqlite_path!r}. Hardcode this exact literal path as a plain "
        "internal constant inside your generated module (e.g. a "
        "module-level `_DB_PATH = " + repr(str(sqlite_path)) + "`) and connect "
        "to it directly with `sqlite3.connect(_DB_PATH)` inside `run()`. "
        "DEFAULT RULE: do NOT declare a connection/path/database-location "
        "keyword argument in `inputs` or accept one as a `run(**kwargs)` "
        "parameter at all, UNLESS the capability description above "
        "explicitly asks you to accept the database path itself as a "
        "named, caller-supplied argument (e.g. it explicitly names a "
        "keyword argument for the path, such as saying the path is given "
        "via keyword argument `db_path`) - only in that explicit case "
        "should you declare such a parameter, and even then default it to "
        "this literal path so the skill still works when the caller omits "
        "it. Why the default rule matters: a real failure was traced "
        "exactly to this - when a skill declares a connection/path "
        "parameter that the capability description never actually asked "
        "for, Praxis's Planner (which fills in every declared input from "
        "the user's own casual wording) supplies a nonsense literal value "
        "for it (e.g. the string \"this Postgres DB\", lifted straight "
        "from a sentence like \"connect to this Postgres DB\"), silently "
        "overriding the correct hardcoded path with a bogus one, and the "
        "skill then fails trying to open a database file that doesn't "
        "exist. So: only declare `inputs` for values that genuinely vary "
        "per call (e.g. the query itself, a table name, a date range) or "
        "that the description explicitly asked to be parameterized - "
        "never silently add a path/connection parameter of your own "
        "accord. Connect with `sqlite3.connect(...)` directly - do not "
        "use SQLAlchemy, asyncpg, or any other third-party database "
        "library, since only the Python standard library is importable "
        "inside the validation sandbox."
    )


def _approval_required_by_settings() -> bool:
    """Whether synthesized skills need approval before they may run.

    Reads `Settings` lazily (constructing it at import time would make
    importing this module fail on an unconfigured environment) and
    fails **closed**: if configuration cannot be read at all, approval
    is required. An unreadable config must never be the reason
    unreviewed generated code becomes executable.
    """
    try:
        from praxis.config import Settings as _Settings

        return bool(_Settings().require_skill_approval)
    except Exception:  # noqa: BLE001 - see docstring
        return True


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
        schema_cache: Cache | None = None,
        schema_cache_ttl_seconds: int | None = _SCHEMA_CACHE_TTL_SECONDS,
        require_approval: bool | None = None,
    ) -> None:
        self._catalogue = catalogue
        self._prompt_manager = prompt_manager
        self._sandbox = sandbox
        self._graph_store = graph_store
        self._store = store
        self._skills_dir = Path(skills_dir)
        # Deliberately a sibling of the skills package, not inside
        # it: anything under `praxis/agents/skills/` is importable
        # and would be picked up by `discover_skills()` on the next
        # restart, which is precisely what quarantine must prevent.
        self._quarantine_dir = Path(skills_dir).parent / "_quarantine"
        self._sandbox_timeout_seconds = sandbox_timeout_seconds
        self._schema_cache: Cache = schema_cache if schema_cache is not None else InMemoryCache()
        self._schema_cache_ttl_seconds = schema_cache_ttl_seconds
        self._require_approval = (
            require_approval
            if require_approval is not None
            else _approval_required_by_settings()
        )

    async def synthesize(
        self,
        need_description: str,
        *,
        connector: Connector | None = None,
        task_id: str | None = None,
        principal: Principal | None = None,
    ) -> Skill:
        """`principal`, when supplied, scopes the connector-schema cache
        to that principal's tenant and effective permission set.

        It is optional and defaults to `None` so every existing caller
        is unchanged. What it buys when present: an introspected schema
        is a description of a tenant's own data source, and two tenants
        that happen to register a connector with the same name and DSN
        would otherwise share one cache entry (see
        `_connector_identity`, which already had to learn this lesson
        once for DSNs). It also keeps a principal whose permissions do
        not let it read a connector from being handed a schema a
        broader principal warmed the cache with.
        """
        with start_span(
            "capability_factory.synthesize",
            task_id=task_id or "",
            connector=connector.name if connector else "",
        ):
            _logger.info(
                "synthesis_started", need_description=need_description, task_id=task_id,
                connector=connector.name if connector else None,
            )
            connector_schema: dict[str, Any] | None = None
            connector_access_hint: str | None = None
            if connector is not None:
                connector_schema = await self._connector_schema(connector, principal=principal)
                connector_access_hint = _connector_access_hint(connector)
            connector_schema_text = (
                json.dumps(connector_schema, indent=2, default=str)
                if connector_schema is not None
                else None
            )

            previous_code: str | None = None
            previous_self_test: str | None = None
            error_detail: str | None = None
            last_code = ""
            last_detail = ""

            for attempt in range(1, _MAX_ATTEMPTS + 1):
                _logger.info(
                    "synthesis_attempt_started", attempt=attempt, max_attempts=_MAX_ATTEMPTS,
                    task_id=task_id,
                )
                prompt = self._prompt_manager.render(
                    _PROMPT_NAME,
                    _PROMPT_VERSION,
                    need_description=need_description,
                    connector_schema=connector_schema_text,
                    connector_access_hint=connector_access_hint,
                    previous_code=previous_code,
                    previous_self_test=previous_self_test,
                    error_detail=error_detail,
                )
                # The synthesis prompt embeds the tenant's own
                # introspected schema, so its cached completion is
                # tenant data too and is keyed accordingly.
                response = await self._catalogue.complete(
                    _LLM_PURPOSE,
                    prompt,
                    max_tokens=_MAX_RESPONSE_TOKENS,
                    principal=principal,
                )

                try:
                    module_code, self_test_code = _parse_response(response)
                except ValueError as exc:
                    last_code = response
                    last_detail = f"attempt {attempt}/{_MAX_ATTEMPTS}: {exc}"
                    previous_code, previous_self_test, error_detail = response, "", last_detail
                    _logger.warning(
                        "synthesis_attempt_failed", attempt=attempt, max_attempts=_MAX_ATTEMPTS,
                        reason="response_did_not_parse", detail=last_detail,
                    )
                    continue

                script = _build_sandbox_script(module_code, self_test_code)
                result = await self._sandbox.run(
                    script, timeout_seconds=self._sandbox_timeout_seconds
                )

                if result.exit_code == 0 and _SUCCESS_MARKER in result.stdout:
                    _logger.info(
                        "synthesis_attempt_succeeded", attempt=attempt, max_attempts=_MAX_ATTEMPTS,
                        task_id=task_id,
                    )
                    return await self._register(
                        module_code,
                        connector=connector,
                        task_id=task_id,
                        tenant_id=(
                            principal.tenant_id if principal is not None else DEFAULT_TENANT_ID
                        ),
                    )

                last_code = module_code
                last_detail = (
                    f"attempt {attempt}/{_MAX_ATTEMPTS}: sandbox exit_code={result.exit_code}\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
                previous_code = module_code
                previous_self_test = self_test_code
                error_detail = last_detail
                _logger.warning(
                    "synthesis_attempt_failed", attempt=attempt, max_attempts=_MAX_ATTEMPTS,
                    reason="sandbox_validation_failed", exit_code=result.exit_code,
                )

            _logger.error(
                "synthesis_exhausted", need_description=need_description, task_id=task_id,
                max_attempts=_MAX_ATTEMPTS,
            )
            raise SynthesisValidationError(
                f"synthesis of capability '{need_description}' failed sandbox validation after "
                f"{_MAX_ATTEMPTS} attempts",
                code=last_code,
                detail=last_detail,
            )

    # ------------------------------------------------------------------ #
    # Connector schema caching (spec §9/§11): a short-TTL InMemoryCache
    # fronts the GraphStore lookup - on a cache hit, neither the
    # GraphStore nor connector.describe() is ever touched; on a cache
    # miss, falls through to Phase 6's existing GraphStore check, then
    # connector.describe() (bounded-retried, spec §12) if that also
    # misses, populating BOTH the GraphStore edge and this cache before
    # returning.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _connector_identity(connector: Connector) -> str:
        """`connector.name` alone is the right cache/lineage key for a
        connector whose underlying data source is stable for the
        connector's whole life (Prometheus, GitHub, Slack, MCP servers -
        one name, one real endpoint, forever). It is the WRONG key for a
        `SQLConnector`, which is routinely registered under the *same*
        name against a *different* DSN per environment, per tenant, or
        per run. Keying purely by name meant a schema cached (or a
        GraphStore `describes` edge written) for one database got
        silently reused by a later run pointing the same connector name
        at different data - traced directly to a real, intermittent
        full-suite failure, where a walkthrough passed every time in
        isolation and failed only when it ran after an earlier suite
        member had already cached a schema under that name. Folding in a
        DSN digest (when the
        connector exposes one - `SQLConnector`/`PostgresConnector`'s own
        `.dsn` property) makes the identity - and therefore the cache
        key and the GraphStore lineage node - change whenever the actual
        data source does, while connectors with no such notion of a
        changeable backing DSN keep exactly their previous, simpler
        name-only identity.
        """
        dsn = getattr(connector, "dsn", None)
        if isinstance(dsn, str) and dsn:
            dsn_digest = hashlib.sha256(dsn.encode("utf-8")).hexdigest()[:12]
            return f"{connector.name}:{dsn_digest}"
        return connector.name

    async def _connector_schema(
        self, connector: Connector, *, principal: Principal | None = None
    ) -> dict[str, Any]:
        identity = self._connector_identity(connector)
        # The key composes the connector identity (name + DSN digest)
        # with the caller's tenant and effective permissions when there
        # is one; with no principal it is the same untenanted keyspace
        # every pre-tenancy caller already used.
        cache_key = CacheKey.build(scopes.CONNECTOR_SCHEMA, principal, identity=identity)

        cached = await self._schema_cache.get(cache_key.value)
        if cached is not None:
            cache_key.authorize(principal)
            return cached

        graph_hit = await self._graph_store.neighbors(identity, relation=_DESCRIBES_RELATION)
        if graph_hit:
            schema = graph_hit[0]["metadata"]["schema"]
            await self._schema_cache.set(
                cache_key.value, schema, ttl_seconds=self._schema_cache_ttl_seconds
            )
            return schema

        with start_span("connector.call", connector=connector.name, operation="describe"):
            description = await call_with_retry(
                connector.describe, connector_name=connector.name, operation="describe"
            )
        digest = hashlib.sha256(
            json.dumps(description.schema, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        await self._graph_store.add_edge(
            source=identity,
            relation=_DESCRIBES_RELATION,
            target=f"schema:{digest}",
            metadata={"kind": description.kind, "schema": description.schema},
        )
        await self._schema_cache.set(
            cache_key.value, description.schema, ttl_seconds=self._schema_cache_ttl_seconds
        )
        return description.schema

    # ------------------------------------------------------------------ #
    # Registration (only ever reached after a real sandbox pass)
    # ------------------------------------------------------------------ #

    async def _register(
        self,
        module_code: str,
        *,
        connector: Connector | None,
        task_id: str | None,
        tenant_id: str = DEFAULT_TENANT_ID,
    ) -> Skill:
        skill_name = _extract_skill_name(module_code)
        module_name = f"synthesized_{_slugify(skill_name)}"

        # The generated module's declared shape, read WITHOUT importing
        # it. See `_extract_skill_metadata`: importing would execute the
        # generated code in this process before anyone approved it.
        metadata = _extract_skill_metadata(module_code)

        # Where the code lands depends entirely on whether it may run.
        #
        # Under the approval gate it goes to a quarantine directory that
        # is deliberately NOT an importable package, so neither this
        # call nor `discover_skills()` on the next restart can import
        # it. Only an approval moves it into the skills package.
        quarantined = self._require_approval
        target_dir = self._quarantine_dir if quarantined else self._skills_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{module_name}.py"
        file_path.write_text(module_code, encoding="utf-8")

        # `self._skills_dir` is expected to be the real
        # `praxis/agents/skills/` package directory (spec: "write the
        # generated module to praxis/agents/skills/synthesized_<slug>.py")
        # - the dotted import path below assumes exactly that, which is
        # also what makes `discover_skills()`'s future package scan pick
        # this file up on a cold restart with zero code changes.
        registered_skill: Skill | None = None
        if not quarantined:
            # Only reached when this deployment has deliberately turned
            # the approval gate off, which is the one case where running
            # generated code without review is a configured choice
            # rather than an accident.
            dotted_module = f"praxis.agents.skills.{module_name}"
            importlib.import_module(dotted_module)
            registered_skill = skill_registry.get_skill(skill_name)

        # Phase 21 (Prompt §7: "require approval before
        # publishing/enabling it"). The catalogue row's status decides
        # whether this skill may actually be *invoked*:
        #
        #   pending_approval -> sandbox-validated, recorded, and
        #                       visible in the approval queue, but the
        #                       Orchestrator refuses to run it;
        #   active           -> a human with `skill:approve` has
        #                       signed off on this exact code hash.
        #
        # The gate defaults ON, because "generated code is trusted
        # unless someone remembers to switch on review" is exactly the
        # posture the brief forbids. A deployment that genuinely wants
        # autonomous activation sets `PRAXIS_REQUIRE_SKILL_APPROVAL=false`
        # deliberately, and that choice is then visible in its config
        # rather than implicit in the code.
        status = (
            SkillStatus.PENDING_APPROVAL.value
            if self._require_approval
            else SkillStatus.ACTIVE.value
        )
        async with self._store.session() as session:
            session.add(
                SkillRecord(
                    # Bound to the tenant that asked for it. Without
                    # this the row defaulted to the default tenant, and
                    # `_approval_gate_blocks` - which filters by the
                    # caller's tenant - found no row for any OTHER
                    # tenant and allowed the call through. A pending
                    # skill was therefore ungated for every tenant
                    # except the one it was created in.
                    tenant_id=tenant_id,
                    name=skill_name,
                    risk=metadata.risk,
                    synthesized=True,
                    status=status,
                    inputs_schema=dict(metadata.inputs),
                    outputs_schema=dict(metadata.outputs),
                    code_hash=compute_code_hash(module_code),
                    source_path=str(file_path),
                    description=metadata.docstring,
                )
            )
            await session.commit()

        if self._require_approval:
            _logger.info(
                "synthesized_skill_pending_approval",
                skill=skill_name,
                code_hash=compute_code_hash(module_code),
            )

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

        if registered_skill is None:
            # Quarantined: there is deliberately nothing runnable to
            # return. The caller surfaces this as a pending-approval
            # step failure rather than executing anything.
            raise SkillPendingApprovalError(
                f"skill '{skill_name}' is quarantined pending approval; an approver "
                f"holding 'skill:approve' must publish it before it can run",
                skill_name=skill_name,
                code_hash=compute_code_hash(module_code),
            )
        return registered_skill
