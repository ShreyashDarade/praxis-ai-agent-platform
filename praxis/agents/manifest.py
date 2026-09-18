# praxis/agents/manifest.py
"""Skill manifests and the SKILL.md format (Prompt §7).

Prompt §7 requires every skill/agent/tool to carry: *"name,
description, version, owner, input/output schema, required
permissions, supported connectors, risk level, execution budget, model
requirements, test cases, health status, dependencies, audit trail"*
and to be definable via *"skills/<skill-name>/SKILL.md with descriptive
name and description frontmatter"*.

Praxis's `Skill` ABC carries five of those fields. This module adds
the rest as a separate `SkillManifest`, deliberately **beside** the
`Skill` class rather than inside it:

- A manifest must be readable **without importing or executing** the
  skill's code. Routing, permission checks, and the approval queue all
  need to know what a skill claims *before* anything of it runs -
  which is impossible if the metadata only exists as class attributes
  on an imported module. For a synthesized skill awaiting approval,
  importing it is exactly what must not happen yet.
- Progressive disclosure (the prompt's *"Load metadata first and
  detailed procedures only when relevant"*): the frontmatter is small
  and always parsed; the body is the long-form procedure, read only
  when the skill is actually selected.

`SKILL.md` is YAML frontmatter plus a Markdown body, parsed with
PyYAML's `safe_load` - see `_parse_frontmatter` for the two residual
risks that are handled explicitly rather than by reimplementing a
parser.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Frontmatter is delimited by `---` lines, the universal convention.
_FRONTMATTER_RE = re.compile(r"^---\s*\n(?P<frontmatter>.*?)\n---\s*\n?(?P<body>.*)\Z", re.DOTALL)

# A manifest is partly LLM-authored content, so the parser matters.
# `yaml.safe_load` is the right tool and is already available in this
# venv: it constructs only plain Python scalars/lists/dicts and will
# NOT instantiate arbitrary objects the way `yaml.load` does. Writing
# a bespoke parser here (the first attempt) was the wrong instinct -
# PyYAML is vastly better tested than anything hand-rolled, and the
# real residual risks are handled explicitly below rather than by
# reimplementing the parser:
#
#  - Entity-expansion / alias bombs: bounded by refusing frontmatter
#    above `_MAX_FRONTMATTER_BYTES` before parsing, and by rejecting
#    the `&anchor`/`*alias` syntax outright, since a skill manifest
#    has no legitimate use for it.
#  - Surprising scalar coercion (`version: 1.0` becoming a float, or
#    `no` becoming False): handled by coercing the fields whose type
#    actually matters back to `str` after parsing, rather than by
#    refusing to use a real YAML parser.
_MAX_FRONTMATTER_BYTES = 64 * 1024

# Fields whose value must stay a string even when YAML would happily
# read it as a number or bool. `version: 1.0` silently becoming the
# float 1.0 corrupts a version number.
_FORCE_STRING_FIELDS = ("name", "version", "owner", "risk", "description")


class ManifestError(Exception):
    """A SKILL.md could not be parsed, or is missing required fields."""


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """Parses the frontmatter block with `yaml.safe_load`.

    Raises `ManifestError` (never a raw YAML error) so every caller
    sees one exception type with a message naming the real problem.
    """
    if len(text.encode("utf-8")) > _MAX_FRONTMATTER_BYTES:
        raise ManifestError(
            f"frontmatter exceeds {_MAX_FRONTMATTER_BYTES} bytes; refusing to parse"
        )
    if re.search(r"(^|\s)[&*][A-Za-z0-9_-]+", text):
        raise ManifestError(
            "YAML anchors/aliases are not permitted in a skill manifest"
        )

    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ManifestError(f"frontmatter is not valid YAML: {exc}") from exc

    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ManifestError(
            f"frontmatter must be a mapping, got {type(parsed).__name__}"
        )

    for field_name in _FORCE_STRING_FIELDS:
        if field_name in parsed and parsed[field_name] is not None:
            parsed[field_name] = str(parsed[field_name])
    return parsed


@dataclass
class SkillManifest:
    """Everything declared about a skill, readable without running it."""

    name: str
    description: str
    version: str = "1.0.0"
    owner: str = ""
    risk: str = "read_only"
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    required_permissions: list[str] = field(default_factory=list)
    supported_connectors: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    model_requirements: dict[str, Any] = field(default_factory=dict)
    execution_budget: dict[str, Any] = field(default_factory=dict)
    test_cases: list[Any] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    # The long-form procedure. Loaded with the file but only *used*
    # when the skill is actually selected - the progressive-disclosure
    # half of the prompt's requirement.
    body: str = ""
    source_path: str = ""
    code_hash: str = ""
    synthesized: bool = False

    @classmethod
    def from_markdown(cls, text: str, *, source_path: str = "") -> "SkillManifest":
        """Parses a SKILL.md document."""
        match = _FRONTMATTER_RE.match(text.lstrip("﻿"))
        if match is None:
            raise ManifestError(
                "SKILL.md must begin with a '---' delimited frontmatter block"
            )

        data = _parse_frontmatter(match.group("frontmatter"))
        body = match.group("body").strip()

        for required in ("name", "description"):
            if not data.get(required):
                raise ManifestError(f"SKILL.md frontmatter is missing required field '{required}'")

        risk = str(data.get("risk", "read_only"))
        if risk not in ("read_only", "mutating"):
            raise ManifestError(
                f"risk must be 'read_only' or 'mutating', got {risk!r}"
            )

        return cls(
            name=str(data["name"]),
            description=str(data["description"]),
            version=str(data.get("version", "1.0.0")),
            owner=str(data.get("owner", "")),
            risk=risk,
            inputs=dict(data.get("inputs") or {}),
            outputs=dict(data.get("outputs") or {}),
            required_permissions=list(data.get("required_permissions") or []),
            supported_connectors=list(data.get("supported_connectors") or []),
            dependencies=list(data.get("dependencies") or []),
            model_requirements=dict(data.get("model_requirements") or {}),
            execution_budget=dict(data.get("execution_budget") or {}),
            test_cases=list(data.get("test_cases") or []),
            keywords=list(data.get("keywords") or []),
            body=body,
            source_path=source_path,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "SkillManifest":
        file_path = Path(path)
        return cls.from_markdown(
            file_path.read_text(encoding="utf-8"), source_path=str(file_path)
        )

    @classmethod
    def from_skill(cls, skill: Any, **overrides: Any) -> "SkillManifest":
        """Derives a manifest from an already-imported `Skill`.

        The bridge for the hand-written skills that predate manifests:
        they keep working unchanged and still appear in the catalogue
        with real metadata, rather than needing a SKILL.md written for
        each before the catalogue is usable.
        """
        payload: dict[str, Any] = {
            "name": skill.name,
            "description": (getattr(skill, "__doc__", "") or "").strip().split("\n")[0],
            "risk": getattr(skill, "risk", "read_only"),
            "inputs": dict(getattr(skill, "inputs", {}) or {}),
            "outputs": dict(getattr(skill, "outputs", {}) or {}),
        }
        payload.update(overrides)
        return cls(**payload)

    def summary(self) -> dict[str, Any]:
        """The small, always-loaded half (progressive disclosure).

        This is what a router or catalogue listing sees - deliberately
        without `body`, so selecting among 50 skills does not mean
        loading 50 long-form procedures into context.
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "risk": self.risk,
            "keywords": self.keywords,
            "supported_connectors": self.supported_connectors,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "owner": self.owner,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "required_permissions": self.required_permissions,
            "dependencies": self.dependencies,
            "model_requirements": self.model_requirements,
            "execution_budget": self.execution_budget,
            "test_cases": self.test_cases,
            "source_path": self.source_path,
            "code_hash": self.code_hash,
            "synthesized": self.synthesized,
            "body": self.body,
        }


def compute_code_hash(code: str) -> str:
    """The identity of a specific published implementation.

    Part of the prompt's required audit trail (*"Audit origin, code
    hash, test evidence, approver, and version"*): an approval is
    granted for exactly this code, and a hash mismatch later means
    what is about to run is not what was reviewed.
    """
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
