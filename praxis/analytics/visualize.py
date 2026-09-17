# praxis/analytics/visualize.py
"""Plotly-backed `Visualizer` (spec §13): the MVP chart backend.

"A `Visualizer` interface (Plotly-backed for MVP) accepting a tabular
result + chart type + encoding, returning a chart artifact
(image/HTML). This is what the Factory calls when synthesizing any
chart-producing skill (the dashboard demo case) - chart backend is
swappable behind the interface like everything else."

`data` is exactly the shape a real tabular result already has
elsewhere in this codebase (`SQLConnector.read()` returns `list[dict]`)
- no separate "load this into a DataFrame yourself" step is asked of a
caller; this class does that conversion internally, once, right before
handing it to Plotly Express.

Static PNG (via `kaleido`, plotly's headless-render dependency for
`Figure.to_image()`) is the primary output - the common case for a
chart artifact that gets stored once and served as a plain image later
(`GET /artifacts/{key}`). Interactive HTML (`as_html=True`) is a cheap
secondary option built on the exact same `Figure`, for a caller that
specifically wants hover/zoom - Plotly's own JS is pulled from a CDN
(`include_plotlyjs="cdn"`) rather than embedded inline, which is what
keeps that HTML in the tens of KB instead of several MB.
"""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from praxis.core.interfaces import ChartArtifact, Visualizer

# One Plotly Express builder per supported `chart_type` (spec §13:
# "support at minimum bar, line, scatter"). A plain dict, not an
# if/elif chain, so the set of supported chart types is legible at a
# glance and `sorted(_CHART_BUILDERS)` (used in the error message below)
# always reflects reality.
_CHART_BUILDERS: dict[str, Callable[..., go.Figure]] = {
    "bar": px.bar,
    "line": px.line,
    "scatter": px.scatter,
}

# Chart roles this MVP encoding understands, mapped to the matching
# Plotly Express keyword argument. "x"/"y" are required or there is no
# way to build any of the three supported chart types below; "color"
# is the one optional role worth wiring through (real dashboard asks
# routinely group/color a series, e.g. §16.2's "chart weekly signup
# counts" broken out by department) without trying to reproduce a full
# Vega-Lite grammar's every channel.
_REQUIRED_ENCODING_ROLES = ("x", "y")
_OPTIONAL_ENCODING_ROLES = ("color",)


class PlotlyVisualizer(Visualizer):
    """The one concrete `Visualizer` this MVP ships (spec §13)."""

    def render(
        self,
        data: list[dict[str, Any]],
        chart_type: str,
        encoding: dict[str, str],
        *,
        as_html: bool = False,
    ) -> ChartArtifact:
        """Renders `data` as `chart_type`, using `encoding` to pick which
        columns play which role.

        Raises `ValueError` - never returns a placeholder/empty artifact
        - for anything that would make the result meaningless: no rows,
        an unsupported `chart_type`, an `encoding` missing a required
        role, or an `encoding` naming a column `data` doesn't actually
        have (spec §12: a failure is never silently papered over).

        `as_html=False` (the default) returns a static PNG
        (`mime_type="image/png"`) - the common case, and the only shape
        `GET /artifacts/{key}` needs to serve a chart as a plain image.
        `as_html=True` returns a self-contained interactive HTML
        document (`mime_type="text/html"`) built from the exact same
        Plotly figure, for a caller that specifically wants hover/zoom.
        """
        if not data:
            raise ValueError("cannot render a chart from empty data")

        builder = _CHART_BUILDERS.get(chart_type)
        if builder is None:
            raise ValueError(
                f"unsupported chart_type {chart_type!r}; supported: {sorted(_CHART_BUILDERS)}"
            )

        missing_roles = [role for role in _REQUIRED_ENCODING_ROLES if role not in encoding]
        if missing_roles:
            raise ValueError(
                f"encoding is missing required role(s) {missing_roles} for chart_type {chart_type!r}"
            )

        frame = pd.DataFrame(data)
        plot_kwargs: dict[str, str] = {
            role: encoding[role]
            for role in (*_REQUIRED_ENCODING_ROLES, *_OPTIONAL_ENCODING_ROLES)
            if role in encoding
        }
        missing_columns = [column for column in plot_kwargs.values() if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"encoding references column(s) not present in data: {missing_columns}"
            )

        figure = builder(frame, **plot_kwargs)

        if as_html:
            html = figure.to_html(include_plotlyjs="cdn", full_html=True)
            return ChartArtifact(content=html.encode("utf-8"), mime_type="text/html")

        png_bytes = figure.to_image(format="png")
        return ChartArtifact(content=png_bytes, mime_type="image/png")
