# praxis/agents/skills/web_search.py
"""`web_search`: a read-only skill wrapping `WebConnector.search`
(spec §6.1, §7's "Web agent").

Mirrors `post_slack_message.py`'s own "genuinely optional, degrades
clearly" pattern: no live Tavily API key is configured in this
environment, so this checks `Settings().tavily_api_key` up front and
raises `SkillConfigurationError` - a clear, typed failure, never a
silent empty-results no-op that could be mistaken for a genuine "no
results found" (spec §12 keeps those two outcomes distinct: `blocked`/
`error` vs. `barren`) - when it's unset, exactly like
`post_slack_message` does for `PRAXIS_SLACK_BOT_TOKEN`.

**Registration is itself gated on `PRAXIS_WEB_TOOLS_ENABLED`** (spec
§7: "Enablement is a deployment-level toggle (off by default, on by
config) rather than always-on ... the web connector is the exception
that needs an explicit switch precisely because 'the open web' isn't a
single system someone deliberately connected" - unlike every other
connector/skill in this codebase, which an operator already explicitly
opted into by registering it at all). Concretely: a real `Planner` LLM
call sees whatever `all_skills()` returns and is perfectly entitled to
pick `web_search` for an intent that superficially looks like it could
use it, even when Tavily isn't configured - `WebSearchSkill.run`'s own
`SkillConfigurationError` guard above only fires *after* the Planner
already picked it, which would fail an otherwise-satisfiable task
(e.g. one whose data already lives in `retrieve_documents`) for no
reason. Gating registration itself is what keeps this tool invisible
to the Planner until an operator deliberately opts in - this checks the
raw env var directly, not a constructed `Settings()`, so importing this
module never has a hard, unrelated dependency on `PRAXIS_DATABASE_URL`
being set purely to decide whether to self-register.
"""
from __future__ import annotations

import os
from typing import Any

from praxis.agents.skill import Skill, SkillConfigurationError
from praxis.agents.skill_registry import register_skill
from praxis.config import Settings
from praxis.connectors.web.web_connector import WebConnector
from praxis.connectors.web.search_provider import TavilySearchProvider

_DEFAULT_MAX_RESULTS = 5
_WEB_TOOLS_ENABLED = os.environ.get("PRAXIS_WEB_TOOLS_ENABLED", "").strip().lower() in (
    "1", "true", "yes", "on",
)


class WebSearchSkill(Skill):
    name = "web_search"
    risk = "read_only"
    inputs = {
        "objective": "why this search is being run, to steer relevance",
        "queries": "list of search query strings",
        "max_results": "max results per query (optional, default 5)",
    }
    outputs = {"results": "list of {title, url, content} search hits, deduped by url"}

    async def run(self, **kwargs: Any) -> Any:
        objective = kwargs["objective"]
        queries = kwargs["queries"]
        max_results = kwargs.get("max_results", _DEFAULT_MAX_RESULTS)

        settings = Settings()
        if not settings.tavily_api_key:
            raise SkillConfigurationError(
                "web_search requires PRAXIS_TAVILY_API_KEY to be configured; refusing to "
                "return an empty result set disguised as a real search"
            )

        connector = WebConnector(search_provider=TavilySearchProvider(api_key=settings.tavily_api_key))
        results = await connector.search(objective=objective, queries=queries, max_results=max_results)
        return {"results": results}


if _WEB_TOOLS_ENABLED:
    register_skill(WebSearchSkill())
