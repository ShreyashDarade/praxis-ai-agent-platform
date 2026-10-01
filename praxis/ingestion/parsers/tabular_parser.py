# praxis/ingestion/parsers/tabular_parser.py
"""pandas-backed textual summary for CSV/XLSX (spec §5 step 2, Tabular Data Handler).

Per spec §5: tabular data "kept structured (not prose-chunked) so it
stays queryable" - this parser is NOT the queryable path (that's a
future phase's job, operating on the dataframe directly). Its job in
*this* text-ingestion pipeline is narrower: produce a textual summary
(shape, columns, dtypes, numeric describe()) so semantic search can
answer "what tables do we have" - the raw rows are deliberately never
returned as parse() output.
"""
from __future__ import annotations

import asyncio
import io

import pandas as pd

from praxis.core.interfaces import Parser
from praxis.ingestion.parsers.registry import register_parser


class TabularParser(Parser):
    """CSV / XLSX -> a textual shape/schema/stats summary, not the raw data."""

    supported_mime_types = (
        "text/csv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    async def parse(self, data: bytes, mime_type: str) -> str:
        # pandas' read_csv/read_excel and describe() are sync/CPU-bound.
        return await asyncio.to_thread(self._summarize, data, mime_type)

    @staticmethod
    def _summarize(data: bytes, mime_type: str) -> str:
        buffer = io.BytesIO(data)
        frame = pd.read_csv(buffer) if mime_type == "text/csv" else pd.read_excel(buffer)

        # Blank lines separate the logical sections (shape / columns /
        # numeric summary) on purpose - TableAwareChunker splits on
        # exactly these blank-line boundaries, so this format and that
        # chunker are a matched pair.
        sections = [f"Shape: {frame.shape[0]} rows x {frame.shape[1]} columns"]

        column_lines = ["Columns:"]
        for column, dtype in frame.dtypes.items():
            column_lines.append(f"  - {column}: {dtype}")
        sections.append("\n".join(column_lines))

        numeric = frame.select_dtypes(include="number")
        if not numeric.empty:
            sections.append("Numeric summary:\n" + numeric.describe().to_string())

        return "\n\n".join(sections)


register_parser(TabularParser())
