# praxis/agents/skills/web_read.py
"""`web_read`: a read-only skill wrapping `WebConnector.read_page`
(spec §6.1) - guarded-fetch -> robots check -> extract -> distill a
single URL, returned as untrusted-wrapped text.

**`known_urls` is deliberately absent from `inputs` below.** `inputs`
describes what the *Planner* should supply as plan-step arguments
(see `praxis.agents.skill.Skill`'s own docstring) - `known_urls` is
never something the Planner chooses or generates. It is injected
automatically, as a plain extra kwarg, by
`praxis.core.orchestrator.Orchestrator` on every skill call (seeded
from the task's intent text, grown from every completed step's
result - see that module) - this skill's `run()` just reads it out of
`**kwargs` (defaulting to an empty set if somehow absent, e.g. a
direct test call) and hands it straight to `WebConnector.read_page`,
which is what actually enforces spec §6.1's prior-context-only rule.
Do not add `known_urls` to `inputs` merely to "document" it there -
that would wrongly suggest the Planner should decide it.

**Registration is itself gated on `PRAXIS_WEB_TOOLS_ENABLED`** - see
`web_search.py`'s module docstring for the full rationale (spec §7's
explicit, off-by-default deployment toggle for the web tools as a
group, checked via the raw env var rather than a constructed
`Settings()` to avoid an unrelated hard dependency on
`PRAXIS_DATABASE_URL` at import time).
"""
from __future__ import annotations

import os
from typing import Any

from praxis.agents.skill import Skill
from praxis.agents.skill_registry import register_skill
from praxis.connectors.web.web_connector import WebConnector

_WEB_TOOLS_ENABLED = os.environ.get("PRAXIS_WEB_TOOLS_ENABLED", "").strip().lower() in (
    "1", "true", "yes", "on",
)


class WebReadSkill(Skill):
    name = "web_read"
    risk = "read_only"
    inputs = {
        "url": "the URL to read - must already appear in the task's prior context",
        "question": "the task's question/objective, used to focus the distillation",
    }
    outputs = {"content": "untrusted-wrapped, distilled page text"}

    async def run(self, **kwargs: Any) -> Any:
        url = kwargs["url"]
        question = kwargs["question"]
        known_urls = kwargs.get("known_urls") or set()

        connector = WebConnector()
        content = await connector.read_page(url, question=question, known_urls=known_urls)
        return {"content": content}


if _WEB_TOOLS_ENABLED:
    register_skill(WebReadSkill())

