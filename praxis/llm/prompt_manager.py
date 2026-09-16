# praxis/llm/prompt_manager.py
"""Prompt Manager: versioned, named Jinja2 prompt templates (spec §7).

"Every LLM-calling tool/skill pulls its prompt by `name@version`; the
version used is logged per execution ... so prompt changes are
auditable and A/B-able without code edits." This module is the loader:
callers ask for a prompt by name+version and get rendered text back -
no inline prompt string is written at any LLM-calling call site.

Layout: `praxis/llm/prompts/<name>/<version>.jinja2`, e.g.
`praxis/llm/prompts/summarize_document/v1.jinja2`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

DEFAULT_PROMPTS_DIR = Path(__file__).parent / "prompts"


class PromptManager:
    """Loads and renders a named, versioned Jinja2 prompt template."""

    def __init__(self, prompts_dir: Path | str | None = None) -> None:
        self._prompts_dir = Path(prompts_dir) if prompts_dir is not None else DEFAULT_PROMPTS_DIR
        self._env = Environment(
            loader=FileSystemLoader(str(self._prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
            keep_trailing_newline=False,
        )

    def render(self, name: str, version: str, **context: Any) -> str:
        """Renders `<name>/<version>.jinja2` with `context`.

        Raises `ValueError` for an unknown name/version combination -
        callers must not silently get an empty/default prompt.
        """
        template_path = f"{name}/{version}.jinja2"
        try:
            template = self._env.get_template(template_path)
        except TemplateNotFound:
            expected = self._prompts_dir / name / f"{version}.jinja2"
            raise ValueError(
                f"unknown prompt template '{name}@{version}' (expected file at {expected})"
            ) from None
        return template.render(**context)
