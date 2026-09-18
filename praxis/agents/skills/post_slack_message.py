# praxis/agents/skills/post_slack_message.py
"""`post_slack_message`: a `mutating`-risk example skill, exercising the
Orchestrator's approval-pause path end-to-end (spec §8).

No live Slack workspace/token is configured in this dev environment, so
this mirrors `praxis.connectors.bootstrap`'s own "genuinely optional,
degrades gracefully" pattern: it constructs a real `SlackConnector`
only when `Settings().slack_bot_token` is actually set, and posts for
real via `write("post_message", ...)` when it is. When it isn't, it
raises `SkillConfigurationError` - a clear, typed failure, never a
silent "pretend it worked" no-op (spec §12).

The pause/approve mechanism itself (`should_pause_for_approval`) fires
on `skill.risk == "mutating"` *before* the Orchestrator ever calls
`run()` - this skill's body only matters once a test explicitly
approves the paused step and execution actually reaches here.
"""
from __future__ import annotations

from typing import Any

from praxis.agents.skill import Skill, SkillConfigurationError
from praxis.agents.skill_registry import register_skill
from praxis.config import Settings
from praxis.connectors.slack.slack_connector import SlackConnector


class PostSlackMessageSkill(Skill):
    name = "post_slack_message"
    risk = "mutating"
    inputs = {"channel": "Slack channel", "message": "text to post"}
    outputs = {"posted": "whether the message was sent"}

    async def run(self, **kwargs: Any) -> Any:
        channel = kwargs["channel"]
        message = kwargs["message"]

        settings = Settings()
        if not settings.slack_bot_token:
            raise SkillConfigurationError(
                "post_slack_message requires PRAXIS_SLACK_BOT_TOKEN to be configured; "
                "refusing to pretend the message was sent"
            )

        connector = SlackConnector(bot_token=settings.slack_bot_token, read_only=False)
        await connector.write("post_message", channel=channel, text=message)
        return {"posted": True}


register_skill(PostSlackMessageSkill())
