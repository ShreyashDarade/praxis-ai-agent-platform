---
name: chart_uploaded_spreadsheet
description: >
  Query an uploaded CSV/XLSX with exact SQL and chart the result, in one
  step. Composes existing approved tools; contains no new code.
version: 1.0.0
owner: platform
risk: read_only
keywords: [spreadsheet, chart, upload, csv, xlsx, visualise]
dependencies: [query_table, create_chart]
inputs:
  attachment_id: id of the uploaded CSV/XLSX to chart
  sql: read-only SQL over the uploaded table, which is named `data`
  chart_type: one of the supported chart types, e.g. bar or line
  encoding: mapping of chart roles to column names
outputs:
  outputs: the per-step outputs, including the chart's artifact_key
steps:
  - name: query
    tool: query_table
    from:
      attachment_id: "$.attachment_id"
      sql: "$.sql"
  - name: chart
    tool: create_chart
    from:
      data: query.rows
      chart_type: "$.chart_type"
      encoding: "$.encoding"
---

# Chart an uploaded spreadsheet

Use this when someone uploads a spreadsheet and asks to see it as a
chart. It does the two things that request always needs, in the order
they have to happen:

1. **Query the file with real SQL.** `query_table` reads the upload as a
   typed table and computes the answer exactly. This matters: retrieving
   similar-looking text and reading a number out of it is not the same
   as computing the number, and for a figure someone will act on only
   the second is acceptable.
2. **Chart what the query returned.** The rows flow straight into
   `create_chart`, so the picture is of the computed result rather than
   of a second, separately-derived view of the data.

Both tools are read-only, so this procedure is read-only. Adding a
mutating tool to it would make it mutating, and the loader will refuse
the file unless the frontmatter says so.
