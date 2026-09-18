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
handing it to Plotly.

Static PNG (via `kaleido`, plotly's headless-render dependency for
`Figure.to_image()`) is the primary output - the common case for a
chart artifact that gets stored once and served as a plain image later
(`GET /artifacts/{key}`). Interactive HTML (`as_html=True`) is a cheap
secondary option built on the exact same `Figure`, for a caller that
specifically wants hover/zoom - Plotly's own JS is pulled from a CDN
(`include_plotlyjs="cdn"`) rather than embedded inline, which is what
keeps that HTML in the tens of KB instead of several MB.

Not every chart type the product asks for is a Plotly Express
one-liner: a KPI card, a table, a Sankey and a waterfall have no
`px.*` equivalent at all and are built from `plotly.graph_objects`
primitives, and a cohort grid is a heatmap over a pivot the caller's
rows do not arrive already shaped as. Those live behind the same
`_ChartSpec.build` callable as the express-backed ones, so `render()`
stays a single validate-then-build path rather than growing a branch
per chart family.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from praxis.core.interfaces import ChartArtifact, Visualizer

# A builder turns the caller's rows plus an already-validated
# role -> column-name mapping into a figure. Taking the mapping (rather
# than the raw `encoding`) means a builder never has to re-check that a
# required role is present or that a named column exists - `render()`
# has done both before any builder runs.
_ChartBuilder = Callable[[pd.DataFrame, dict[str, str]], go.Figure]

# How much of a pie to punch out for `donut`. Large enough that the
# centre reads as deliberate whitespace (the thing that makes a donut
# worth having over a pie: somewhere to put a total) rather than as a
# rendering artifact.
_DONUT_HOLE = 0.45

# Synthetic column names for the pivot below. Prefixed because they are
# injected next to the caller's own columns and a collision would
# silently pivot the wrong data.
_PIVOT_ROW = "_praxis_pivot_row"
_PIVOT_COLUMN = "_praxis_pivot_column"


@dataclass(frozen=True)
class _ChartSpec:
    """One chart type: how to build it, and which roles it actually needs.

    The required/optional roles live here rather than as one global
    `("x", "y")` pair because the roles genuinely differ per chart type
    - a pie has no axes, a KPI card has no second dimension, a Sankey's
    three roles are not positions at all. Validating every chart type
    against x+y would both reject legitimate encodings and let a pie
    through with an encoding that says nothing about the pie.
    """

    build: _ChartBuilder
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()


def _express(builder: Callable[..., go.Figure], **fixed: Any) -> _ChartBuilder:
    """Adapts a Plotly Express function to the `_ChartBuilder` shape.

    Every express-backed role below is deliberately spelled the same as
    the Plotly Express keyword it feeds (`names`/`values` for pie,
    `lat`/`lon` for map, `parents` for treemap), so the adaptation is a
    `**` splat instead of a per-chart-type translation table that would
    be one more thing to keep in sync with the roles declared right
    next to it.
    """

    def build(frame: pd.DataFrame, columns: dict[str, str]) -> go.Figure:
        return builder(frame, **columns, **fixed)

    return build


def _build_kpi_card(frame: pd.DataFrame, columns: dict[str, str]) -> go.Figure:
    """A single big number, as `go.Indicator`.

    Reads the *last* row: a KPI query usually returns exactly one row
    (where both ends agree), and when it returns an ordered series
    instead, "the KPI" means where the series has got to, not where it
    started.

    `delta` names a column holding the comparison baseline rather than
    the change itself, because that is what `go.Indicator` computes
    against - handing it a pre-computed difference would render the
    difference between the value and the difference.
    """
    current = frame.iloc[-1]
    indicator: dict[str, Any] = {"mode": "number", "value": current[columns["value"]]}
    if "label" in columns:
        indicator["title"] = {"text": str(current[columns["label"]])}
    if "delta" in columns:
        indicator["mode"] = "number+delta"
        indicator["delta"] = {"reference": current[columns["delta"]]}
    return go.Figure(go.Indicator(**indicator))


def _build_table(frame: pd.DataFrame, _columns: dict[str, str]) -> go.Figure:
    """Every column of the result, as `go.Table`.

    The only chart type here that takes no encoding at all: "show me
    the rows" has no roles to assign, and silently dropping columns the
    caller did not think to name would make the table a worse answer
    than the raw result it was built from.
    """
    return go.Figure(
        go.Table(
            header={"values": [str(column) for column in frame.columns]},
            cells={"values": [frame[column].tolist() for column in frame.columns]},
        )
    )


def _build_sankey(frame: pd.DataFrame, columns: dict[str, str]) -> go.Figure:
    """Flows between nodes, as `go.Sankey`.

    `go.Sankey` addresses nodes by position in its own `node.label`
    list, so the edge list the caller has (source name, target name,
    value) has to be turned into one deduplicated node list plus
    integer indices. Deduplicating across *both* columns at once is the
    point: a stage that is a target of one flow and the source of the
    next has to come out as one node, or the diagram shows two
    disconnected halves instead of a chain.
    """
    sources = frame[columns["source"]].astype(str)
    targets = frame[columns["target"]].astype(str)
    nodes = list(dict.fromkeys([*sources, *targets]))
    position = {node: index for index, node in enumerate(nodes)}
    return go.Figure(
        go.Sankey(
            node={"label": nodes},
            link={
                "source": [position[source] for source in sources],
                "target": [position[target] for target in targets],
                "value": frame[columns["value"]].tolist(),
            },
        )
    )


def _build_waterfall(frame: pd.DataFrame, columns: dict[str, str]) -> go.Figure:
    """Running-total bars, as `go.Waterfall`.

    `measure` is optional because plotly's own default - every bar a
    relative step - is the right reading of a plain list of deltas. A
    caller that wants a subtotal or an absolute reset bar names a
    column of plotly's own measure keywords rather than having those
    rows guessed at from their values.
    """
    waterfall: dict[str, Any] = {
        "x": frame[columns["x"]].tolist(),
        "y": frame[columns["y"]].tolist(),
    }
    if "measure" in columns:
        waterfall["measure"] = frame[columns["measure"]].astype(str).tolist()
    return go.Figure(go.Waterfall(**waterfall))


def _build_treemap(frame: pd.DataFrame, columns: dict[str, str]) -> go.Figure:
    """Nested rectangles, built through Plotly Express's `path` rather
    than its `names`/`parents` pair.

    `names` + `values` on their own render an empty canvas: a treemap
    addresses every node by its parent, so a label list with no parent
    list describes a forest of roots and draws nothing. Passing the
    grouping column as `parents` is no better - the groups in a real
    grouped result (four channels tagged with two regions) are not
    themselves rows, so every node's parent is unknown and plotly drops
    the lot, still silently. `path` takes the grouping columns in the
    shape the query already returns them and synthesizes the
    intermediate nodes and the root.
    """
    path = [columns["parents"], columns["names"]] if "parents" in columns else [columns["names"]]
    treemap_kwargs: dict[str, Any] = {"path": path, "values": columns["values"]}
    if "color" in columns:
        treemap_kwargs["color"] = columns["color"]
    return px.treemap(frame, **treemap_kwargs)


def _pivot_grid(
    frame: pd.DataFrame, row_column: str, column_column: str, value_column: str
) -> pd.DataFrame:
    """Reshapes long rows into the dense grid `go.Heatmap` wants.

    Axis order comes from first appearance in the caller's rows, not
    from sorting: the labels are frequently strings whose lexicographic
    order is wrong ("W10" before "W2", "period 10" before "period 2"),
    and the one ordering that is actually meaningful is the one the
    query already put them in.

    Duplicate (row, column) pairs are averaged rather than summed - `z`
    is the value *at* a coordinate, typically already aggregated by the
    query, so a repeated cell should not silently double.
    """
    ordered = frame.assign(
        **{
            _PIVOT_ROW: _in_first_appearance_order(frame[row_column]),
            _PIVOT_COLUMN: _in_first_appearance_order(frame[column_column]),
        }
    )
    return ordered.pivot_table(
        index=_PIVOT_ROW,
        columns=_PIVOT_COLUMN,
        values=value_column,
        aggfunc="mean",
        observed=False,
    )


def _in_first_appearance_order(values: pd.Series) -> pd.Categorical:
    return pd.Categorical(values, categories=list(dict.fromkeys(values)), ordered=True)


def _heatmap_figure(grid: pd.DataFrame) -> go.Figure:
    return go.Figure(
        go.Heatmap(z=grid.to_numpy(), x=grid.columns.tolist(), y=grid.index.tolist())
    )


def _build_heatmap(frame: pd.DataFrame, columns: dict[str, str]) -> go.Figure:
    figure = _heatmap_figure(_pivot_grid(frame, columns["y"], columns["x"], columns["z"]))
    figure.update_layout(xaxis_title=columns["x"], yaxis_title=columns["y"])
    return figure


def _build_cohort(frame: pd.DataFrame, columns: dict[str, str]) -> go.Figure:
    """A cohort grid: the same heatmap machinery, read the way cohort
    tables are conventionally read.

    Distinct from `heatmap` in its axes rather than its marks - one
    axis is always "which cohort", the other always "how long since
    that cohort started" - and in the reversed y-axis, which puts the
    oldest cohort at the top so the triangle of populated cells falls
    the way every retention table draws it. Plotly's default would put
    the oldest cohort at the bottom and invert that triangle.
    """
    figure = _heatmap_figure(
        _pivot_grid(frame, columns["cohort"], columns["period"], columns["value"])
    )
    figure.update_layout(xaxis_title=columns["period"], yaxis_title=columns["cohort"])
    figure.update_yaxes(autorange="reversed")
    return figure


# One spec per supported `chart_type`. A plain dict, not an if/elif
# chain, so the set of supported chart types (and what each one
# actually requires of an encoding) is legible at a glance and
# `sorted(_CHART_SPECS)` - used in the error message below - always
# reflects reality.
#
# `color` is wired in wherever the underlying mark supports it: real
# dashboard asks routinely group a series (e.g. §16.2's "chart weekly
# signup counts" broken out by department). Beyond that the roles stay
# deliberately close to the minimum each chart type cannot be drawn
# without, rather than reproducing a full Vega-Lite grammar's every
# channel.
_CHART_SPECS: dict[str, _ChartSpec] = {
    "kpi_card": _ChartSpec(_build_kpi_card, required=("value",), optional=("label", "delta")),
    "table": _ChartSpec(_build_table),
    "line": _ChartSpec(_express(px.line), required=("x", "y"), optional=("color",)),
    "bar": _ChartSpec(_express(px.bar), required=("x", "y"), optional=("color",)),
    "area": _ChartSpec(_express(px.area), required=("x", "y"), optional=("color",)),
    "pie": _ChartSpec(_express(px.pie), required=("names", "values"), optional=("color",)),
    "donut": _ChartSpec(
        _express(px.pie, hole=_DONUT_HOLE), required=("names", "values"), optional=("color",)
    ),
    "funnel": _ChartSpec(_express(px.funnel), required=("x", "y"), optional=("color",)),
    "cohort": _ChartSpec(_build_cohort, required=("cohort", "period", "value")),
    "heatmap": _ChartSpec(_build_heatmap, required=("x", "y", "z")),
    "scatter": _ChartSpec(_express(px.scatter), required=("x", "y"), optional=("color", "size")),
    # `px.scatter_geo`, not `px.scatter_map`: the MapLibre-backed
    # renderer fetches its base layer from a remote style/tile server,
    # so a PNG exported from a sandboxed or offline worker comes back
    # with the points floating over nothing. `scatter_geo` draws
    # plotly's own bundled vector geometry - no token, no network, and
    # the artifact is byte-reproducible.
    "map": _ChartSpec(
        _express(px.scatter_geo), required=("lat", "lon"), optional=("size", "color")
    ),
    "waterfall": _ChartSpec(_build_waterfall, required=("x", "y"), optional=("measure",)),
    "box": _ChartSpec(_express(px.box), required=("y",), optional=("x", "color")),
    "treemap": _ChartSpec(
        _build_treemap, required=("names", "values"), optional=("parents", "color")
    ),
    "sankey": _ChartSpec(_build_sankey, required=("source", "target", "value")),
}

# Exported so a caller that has to *describe* the supported set (the
# `create_chart` skill's input schema, which the Planner reads) can
# derive it from the one definition above instead of hand-maintaining a
# second list that silently goes stale.
SUPPORTED_CHART_TYPES: tuple[str, ...] = tuple(_CHART_SPECS)


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

        Which roles `encoding` must carry depends on `chart_type` (see
        `_CHART_SPECS`): `bar` needs x+y, `pie` needs names+values,
        `sankey` needs source+target+value, `kpi_card` needs just
        `value`, `table` needs nothing at all. Roles the chart type does
        not recognise are ignored rather than rejected, so a caller can
        pass one encoding dict across several chart types.

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

        spec = _CHART_SPECS.get(chart_type)
        if spec is None:
            raise ValueError(
                f"unsupported chart_type {chart_type!r}; supported: {sorted(_CHART_SPECS)}"
            )

        missing_roles = [role for role in spec.required if role not in encoding]
        if missing_roles:
            raise ValueError(
                f"encoding is missing required role(s) {missing_roles} for chart_type {chart_type!r}"
            )

        frame = pd.DataFrame(data)
        columns: dict[str, str] = {
            role: encoding[role]
            for role in (*spec.required, *spec.optional)
            if role in encoding
        }
        missing_columns = [column for column in columns.values() if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"encoding references column(s) not present in data: {missing_columns}"
            )

        figure = spec.build(frame, columns)

        if as_html:
            html = figure.to_html(include_plotlyjs="cdn", full_html=True)
            return ChartArtifact(content=html.encode("utf-8"), mime_type="text/html")

        png_bytes = figure.to_image(format="png")
        return ChartArtifact(content=png_bytes, mime_type="image/png")
