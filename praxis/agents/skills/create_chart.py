# praxis/agents/skills/create_chart.py
"""`create_chart`: a read-only skill wrapping the real `PlotlyVisualizer`
(spec §13) - the hand-written example a *future* synthesized
chart-producing skill (spec §16.2's dashboard-ask, Phase 11's job to
actually exercise) can call directly, or that the Planner can slot
straight into a plan today (e.g. right after `retrieve_documents`/a
connector `read()` step) without any synthesis at all.

`data` (the tabular result to chart) is exactly the shape a real
connector already produces - `SQLConnector.read(query)` returns
`list[dict]` - so a plan step feeds this skill's real query results in
directly, unmodified.

Dependencies are constructed the same way `retrieve_documents.py`
constructs its own: `Settings()` fresh per call (so
`PRAXIS_BLOB_STORE_ROOT` can differ per test/deployment without
re-importing this module), `PlotlyVisualizer`/`LocalBlobStore` are both
cheap to construct and hold no state worth sharing across calls.

The rendered artifact's bytes are never returned inline - only the blob
store key + mime type are (spec: "not the raw bytes inline (keeps the
checklist/result JSON small; the artifact lives in blob storage,
matching how attachments already work)"). `GET /artifacts/{key}` (see
`praxis/api/main.py`) is how a caller actually fetches the rendered
chart afterward.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

from praxis.agents.skill import Skill
from praxis.agents.skill_registry import register_skill
from praxis.analytics.visualize import PlotlyVisualizer
from praxis.config import Settings
from praxis.memory.blob_store import LocalBlobStore

# One key prefix for every chart this skill ever stores, under the
# shared blob store root - mirrors how attachment blobs get their own
# namespace, keeping a chart artifact trivially distinguishable (and
# separately prunable) from any other kind of blob in the same store.
_ARTIFACT_KEY_PREFIX = "charts"

_EXTENSION_BY_MIME_TYPE = {"image/png": "png", "text/html": "html"}


class CreateChartSkill(Skill):
    name = "create_chart"
    risk = "read_only"
    inputs = {
        "data": "list of row dicts to chart",
        "chart_type": "bar|line|scatter",
        "encoding": "dict mapping chart roles to column names",
    }
    outputs = {
        "artifact_key": "blob store key where the rendered chart is stored",
        "mime_type": "MIME type of the stored chart artifact",
    }

    async def run(self, **kwargs: Any) -> Any:
        data = kwargs["data"]
        chart_type = kwargs["chart_type"]
        encoding = kwargs["encoding"]
        as_html = kwargs.get("as_html", False)

        visualizer = PlotlyVisualizer()
        # `render()` is a deliberately synchronous ABC method (spec
        # §13's own docstring in `praxis.core.interfaces.Visualizer`) -
        # kaleido's real PNG export shells out to a headless browser,
        # which can take a genuinely noticeable amount of wall-clock
        # time; dispatching it through `asyncio.to_thread` keeps this
        # coroutine from blocking the whole event loop while it runs,
        # exactly like `SentenceTransformerEmbedder.embed()` already
        # does around its own blocking `model.encode()` call.
        artifact = await asyncio.to_thread(
            visualizer.render, data, chart_type, encoding, as_html=as_html
        )

        settings = Settings()
        blob_store = LocalBlobStore(settings.blob_store_root)
        extension = _EXTENSION_BY_MIME_TYPE.get(artifact.mime_type, "bin")
        artifact_key = f"{_ARTIFACT_KEY_PREFIX}/{uuid.uuid4().hex}.{extension}"
        await blob_store.put(artifact_key, artifact.content)

        return {"artifact_key": artifact_key, "mime_type": artifact.mime_type}


register_skill(CreateChartSkill())
