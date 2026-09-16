# praxis/ingestion/enrichment/document_enrichment.py
"""DocumentEnrichment: summarization/topic-tagging, and opt-in
entity extraction into the `GraphStore` (spec §5 step 4).

Per spec: "summarization, entity/topic tagging, and - only when
relations are meaningful ... - entity/relation extraction written to
the `GraphStore` as a lightweight knowledge graph." `enrich()` is the
default the ingestion pipeline uses (summary + topics, no graph writes
- "meaningful" is read as "the caller explicitly asked for it, not
every ingest"). `enrich_and_link()` is the opt-in path for callers that
want the graph side effect.

Both methods route every LLM call through the `LLMCatalogue` by
purpose (routing/cheap tier - summarization and a short entity list
don't need the strongest model) and pull every prompt from the
`PromptManager` by name@version - no hardcoded model name or inline
prompt string, per spec §7/§22 item 4.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from praxis.core.interfaces import GraphStore
from praxis.llm.catalogue import LLMCatalogue
from praxis.llm.prompt_manager import PromptManager

_SUMMARIZE_PROMPT_NAME = "summarize_document"
_SUMMARIZE_PROMPT_VERSION = "v1"
_EXTRACT_ENTITIES_PROMPT_NAME = "extract_entities"
_EXTRACT_ENTITIES_PROMPT_VERSION = "v1"

# Case-insensitive, DOTALL: real model output may vary casing/whitespace
# despite the template's explicit instruction. "Summary:" captures
# everything up to (not including) "Topics:"; "Topics:" captures the
# rest of the response.
_SUMMARY_RE = re.compile(r"summary\s*:\s*(.*?)(?=topics\s*:|\Z)", re.IGNORECASE | re.DOTALL)
_TOPICS_RE = re.compile(r"topics\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)


@dataclass
class EnrichmentResult:
    summary: str
    topics: list[str] = field(default_factory=list)


def _parse_enrichment_response(response: str) -> EnrichmentResult:
    summary_match = _SUMMARY_RE.search(response)
    topics_match = _TOPICS_RE.search(response)

    # Falls back to the whole response as the summary if the model ever
    # drops the "Summary:" label - better than returning an empty
    # summary outright (spec §12: never silently produce an emptier
    # result than what's actually available).
    summary = summary_match.group(1).strip() if summary_match else response.strip()
    topics_raw = topics_match.group(1).strip() if topics_match else ""
    topics = _split_comma_list(topics_raw)

    return EnrichmentResult(summary=summary, topics=topics)


def _split_comma_list(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",")]
    items = [item for item in items if item and item.lower() != "none"]
    return items


class DocumentEnrichment:
    """Summarization/topic-tagging (+ optional entity linking) over ingested text."""

    def __init__(self, catalogue: LLMCatalogue, prompt_manager: PromptManager) -> None:
        self._catalogue = catalogue
        self._prompt_manager = prompt_manager

    async def enrich(self, text: str) -> EnrichmentResult:
        """Summary + topics only - no graph writes (default ingestion path)."""
        prompt = self._prompt_manager.render(_SUMMARIZE_PROMPT_NAME, _SUMMARIZE_PROMPT_VERSION, text=text)
        response = await self._catalogue.complete("routing", prompt)
        return _parse_enrichment_response(response)

    async def enrich_and_link(self, text: str, doc_id: str, graph_store: GraphStore) -> EnrichmentResult:
        """Same summarization, plus entity extraction written to `graph_store`.

        One `mentions` edge is written per extracted entity:
        `add_edge(source=doc_id, relation="mentions", target=entity)`.
        Only reached when a caller explicitly opts in - never run as
        part of `enrich()`/the default ingestion path.
        """
        result = await self.enrich(text)

        entity_prompt = self._prompt_manager.render(
            _EXTRACT_ENTITIES_PROMPT_NAME, _EXTRACT_ENTITIES_PROMPT_VERSION, text=text
        )
        entity_response = await self._catalogue.complete("routing", entity_prompt)
        entities = _split_comma_list(entity_response)

        for entity in entities:
            await graph_store.add_edge(source=doc_id, relation="mentions", target=entity)

        return result
