"""Web search/read/crawl connector (spec §6.1) - the most safety-dense
connector in this codebase. See `connector.py`'s module docstring for
the full safety contract (prior-context-only fetch, SSRF-safe pinned
fetching, robots.txt + content-type enforcement, byte caps, untrusted-
content wrapping, distillation before context).
"""
