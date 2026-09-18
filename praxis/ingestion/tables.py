# praxis/ingestion/tables.py
"""Uploaded spreadsheets as typed tables, queried with real SQL (brief §4).

The brief is unusually direct here: *"Preserve tabular data as typed
tables for computation; do not rely on embeddings for exact
aggregations."*

That warning is about a specific, common failure. The text-ingestion
path summarizes a CSV to prose, embeds the prose, and retrieves it by
similarity. Ask "what was total EMEA revenue last quarter" and a
similarity search returns *text that looks like an answer* - a chunk
mentioning EMEA and revenue - which a model then reads a number out of.
The number is frequently wrong, and nothing in that path can tell you
so. Embeddings retrieve what is *similar*; `SUM()` computes what is
*true*, and for a financial total only the second is acceptable.

So this module is the exact path, running beside the semantic one
rather than replacing it:

- `TabularParser` still produces its prose summary, and that is still
  embedded. It answers **"what tables do we have?"** - a genuinely
  semantic question that similarity search is good at.
- This module answers **"what is the number?"** by loading the original
  bytes into a typed frame and running real SQL over it.

**Why DuckDB.** The job is "SQL with correct types and exact aggregation
over a file", which is precisely what DuckDB is. Hand-rolling
aggregation over pandas would mean re-implementing GROUP BY, JOIN and
NULL semantics; `pandasql` would mean SQLite, whose dynamic typing is
the opposite of what "typed tables" asks for. DuckDB also uses decimal
arithmetic where SQLite would use float, which is the difference
between a revenue total that reconciles and one that is off by a cent.

**Reads only, always.** Every query goes through the same `SqlGuard`
the connectors use, with the `duckdb` dialect. An uploaded file must
not become a way to run DDL, and the guard's row bound applies here for
the same reason it applies to a warehouse: an unbounded result set from
a large upload is a memory problem, not an answer.

The in-memory DuckDB connection is created per query and closed after,
holding nothing between calls: two tenants' uploads are never resident
in the same connection, so cross-attachment leakage is not something
this has to be careful about - it is structurally absent.
"""
from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from praxis.safety.sql_guard import SqlGuard

# The one table name a query addresses. Fixed rather than derived from
# the file name: a user-supplied name would have to be escaped into the
# SQL text, and a fixed identifier cannot be injected through.
TABLE_NAME = "data"

CSV_MIME_TYPE = "text/csv"
XLSX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
SUPPORTED_MIME_TYPES = (CSV_MIME_TYPE, XLSX_MIME_TYPE)


class TableError(Exception):
    """An uploaded file could not be read as a typed table."""


@dataclass
class ColumnSchema:
    name: str
    dtype: str
    nullable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "nullable": self.nullable}


@dataclass
class TableSchema:
    """What a caller needs in order to write a correct query.

    Handed to the Planner instead of a sample of rows: column names and
    real dtypes are what decide whether `SUM(revenue)` is even legal,
    and they are far cheaper than rows to put in a prompt.
    """

    columns: list[ColumnSchema] = field(default_factory=list)
    row_count: int = 0
    table_name: str = TABLE_NAME

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_name": self.table_name,
            "row_count": self.row_count,
            "columns": [c.to_dict() for c in self.columns],
        }

    def describe(self) -> str:
        """A compact, promptable rendering."""
        lines = [f"Table `{self.table_name}` ({self.row_count} rows):"]
        lines.extend(f"  - {c.name}: {c.dtype}" for c in self.columns)
        return "\n".join(lines)


def _read_frame(data: bytes, mime_type: str) -> pd.DataFrame:
    buffer = io.BytesIO(data)
    if mime_type == CSV_MIME_TYPE:
        # Default inference, deliberately: pandas already maps integer
        # columns to int64 and decimal ones to float64, which is the
        # typing this module exists to preserve. Forcing everything to
        # str here would discard it at the door.
        return pd.read_csv(buffer)
    if mime_type == XLSX_MIME_TYPE:
        return pd.read_excel(buffer)
    raise TableError(
        f"'{mime_type}' is not a tabular type - supported: {', '.join(SUPPORTED_MIME_TYPES)}"
    )


def load_table(data: bytes, mime_type: str) -> pd.DataFrame:
    """Parses `data` into a typed frame, or raises `TableError`."""
    try:
        frame = _read_frame(data, mime_type)
    except TableError:
        raise
    except Exception as exc:  # noqa: BLE001 - pandas raises a wide family here
        raise TableError(f"could not read {mime_type} as a table: {exc}") from exc

    if frame.empty and not list(frame.columns):
        raise TableError("the file contains no columns")
    return frame


def schema_of(frame: pd.DataFrame) -> TableSchema:
    """The typed schema of an already-loaded frame."""
    return TableSchema(
        columns=[
            ColumnSchema(
                name=str(name),
                dtype=str(dtype),
                nullable=bool(frame[name].isna().any()),
            )
            for name, dtype in frame.dtypes.items()
        ],
        row_count=int(len(frame)),
    )


def _run_sql(frame: pd.DataFrame, sql: str) -> list[dict[str, Any]]:
    """Executes `sql` against `frame` in a throwaway DuckDB connection."""
    import duckdb

    connection = duckdb.connect()
    try:
        # Registered rather than inserted: DuckDB reads the frame's
        # memory directly, so a large upload is not copied a second time
        # just to be queried once.
        connection.register(TABLE_NAME, frame)
        result = connection.execute(sql).fetchdf()
    finally:
        connection.close()

    # `to_dict("records")` keeps numpy scalar types, which are not JSON
    # serializable and would fail at the API boundary rather than here.
    return result.astype(object).where(pd.notna(result), None).to_dict("records")


async def query_table(
    data: bytes,
    mime_type: str,
    sql: str,
    *,
    guard: SqlGuard | None = None,
) -> dict[str, Any]:
    """Runs one read-only SQL query over an uploaded table.

    The query addresses the table as `data` (`TABLE_NAME`). Returns the
    rows plus the schema and the query that actually ran, so a caller
    can show its working - the point of the exact path is that the
    number is checkable.

    Raises `TableError` for an unreadable file and `SqlGuardError` for a
    query that is not a bounded read.
    """
    guard = guard or SqlGuard()
    # Validated before the file is even parsed: a DDL attempt should be
    # refused on its own terms, not after doing the work of loading.
    safe_sql = guard.validate_read(sql, dialect="duckdb")

    frame = await asyncio.to_thread(load_table, data, mime_type)
    rows = await asyncio.to_thread(_run_sql, frame, safe_sql)

    return {
        "rows": rows,
        "row_count": len(rows),
        "query": safe_sql,
        "schema": schema_of(frame).to_dict(),
    }


async def describe_uploaded_table(data: bytes, mime_type: str) -> TableSchema:
    """The typed schema of an uploaded file, without running a query."""
    frame = await asyncio.to_thread(load_table, data, mime_type)
    return schema_of(frame)
