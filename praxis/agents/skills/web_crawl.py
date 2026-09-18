# praxis/agents/skills/web_crawl.py
"""`web_crawl`: a read-only skill wrapping `WebConnector.crawl`
(spec §6.1) - a bounded, same-host BFS crawl starting from a seed URL,
distilled once over the combined collected text.

**`known_urls` is deliberately absent from `inputs` below**, for the
exact same reason documented in `web_read.py`'s module docstring - it
is an Orchestrator-injected kwarg, never a Planner-supplied argument.
Only `start_url` is checked against it (spec §6.1 explicitly scopes the
prior-context rule to the seed URL; links discovered mid-crawl are not
required to already be known - see `WebConnector.crawl`).

**Registration is itself gated on `PRAXIS_WEB_TOOLS_ENABLED`** - see
`web_search.py`'s module docstring for the full rationale.
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


class WebCrawlSkill(Skill):
    name = "web_crawl"
    risk = "read_only"
    inputs = {
        "start_url": "the seed URL to crawl from - must already appear in the task's prior context",
        "question": "the task's question/objective, used to focus the distillation",
        "max_pages": "max pages to fetch across the whole crawl (optional, default 5)",
        "max_depth": "max link-following depth from start_url (optional, default 2)",
    }
    outputs = {"content": "untrusted-wrapped, distilled text combined across every page fetched"}

    async def run(self, **kwargs: Any) -> Any:
        start_url = kwargs["start_url"]
        question = kwargs["question"]
        known_urls = kwargs.get("known_urls") or set()

        crawl_kwargs: dict[str, Any] = {}
        if "max_pages" in kwargs:
            crawl_kwargs["max_pages"] = kwargs["max_pages"]
        if "max_depth" in kwargs:
            crawl_kwargs["max_depth"] = kwargs["max_depth"]

        connector = WebConnector()
        content = await connector.crawl(start_url, question=question, known_urls=known_urls, **crawl_kwargs)
        return {"content": content}


if _WEB_TOOLS_ENABLED:
    register_skill(WebCrawlSkill())
