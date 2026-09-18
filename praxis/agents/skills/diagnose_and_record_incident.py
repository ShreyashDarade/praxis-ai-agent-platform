# praxis/agents/skills/diagnose_and_record_incident.py
"""`diagnose_and_record_incident`: a hand-written, read-only skill
combining spec §16.1's "Diagnosis subagent" step (a real LLM call - LLM
Catalogue purpose `"diagnosis"`, Prompt Manager template
`diagnose_incident@v1`) with its postmortem-writeback step (spec §16.1
step 8: "Postmortem text is written back through the Ingestion pipeline
... so future incidents retrieve it as context") in one skill.

**Why one skill, not two Planner-sequenced steps**: an SRE incident's
diagnosis text and its postmortem write-back have no independent value
to a plan that splits them across two steps needing a
`"$<index>.<key>"` cross-reference - combining them removes one more
place a single, non-interactive real Planner completion (spec §7's
Planner is one JSON-producing call, not an iterative agent loop) would
have to get an exact key name right, without weakening any "real,
end-to-end" property (spec §17): the LLM call is real (`LLMCatalogue`,
purpose `"diagnosis"`), and the ingestion write-back is the real
`praxis.ingestion.pipeline.ingest()` (real parse -> chunk -> embed ->
index), never a stand-in.

Declared `read_only`, not `mutating`: it writes only to Praxis's own
internal knowledge store (blob + vector store), never to an external
system - the same posture already established by `create_chart` (writes
to blob storage, declared `read_only`). "mutating" is reserved for
irreversible action on an *external* system (spec §8), which this skill
never takes.
"""
from __future__ import annotations

import re
from typing import Any

from praxis.agents.skill import Skill
from praxis.agents.skill_registry import register_skill
from praxis.config import Settings
from praxis.ingestion.embedders.sentence_transformer_embedder import get_default_embedder
from praxis.ingestion.parsers import registry as parser_registry
from praxis.ingestion.pipeline import ingest
from praxis.llm.catalogue import LLMCatalogue
from praxis.llm.prompt_manager import PromptManager
from praxis.memory.blob_store import LocalBlobStore
from praxis.memory.db import PostgresStore
from praxis.memory.vector_store import PgVectorStore

_PROMPT_NAME = "diagnose_incident"
_PROMPT_VERSION = "v1"
_LLM_PURPOSE = "diagnosis"

# Same tolerant, case-insensitive/DOTALL split as
# `praxis.ingestion.enrichment.document_enrichment`'s summary/topics
# parse - real model output routinely varies casing/whitespace despite
# an explicit requested format.
_DIAGNOSIS_RE = re.compile(
    r"diagnosis\s*:\s*(.*?)(?=remediation\s*:|\Z)", re.IGNORECASE | re.DOTALL
)
_REMEDIATION_RE = re.compile(r"remediation\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)


def _parse_diagnosis_response(response: str) -> tuple[str, str]:
    diagnosis_match = _DIAGNOSIS_RE.search(response)
    remediation_match = _REMEDIATION_RE.search(response)
    # Falls back to the whole response as the diagnosis if the model
    # ever drops the "DIAGNOSIS:" label - better than an empty diagnosis
    # (spec §12: never silently produce an emptier result than what's
    # actually available).
    diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else response.strip()
    remediation = remediation_match.group(1).strip() if remediation_match else ""
    return diagnosis, remediation


class DiagnoseAndRecordIncidentSkill(Skill):
    name = "diagnose_and_record_incident"
    risk = "read_only"
    inputs = {
        "service": "name of the affected service",
        "observation": "what was observed (e.g. the metric name/value that triggered the alert)",
    }
    outputs = {
        "diagnosis": "a short structured diagnosis of the likely root cause",
        "remediation": "a recommended remediation action",
        "attachment_id": "id of the stored, indexed postmortem document",
    }

    async def run(self, **kwargs: Any) -> Any:
        service = kwargs["service"]
        observation = kwargs["observation"]

        catalogue = LLMCatalogue()
        prompt_manager = PromptManager()
        prompt = prompt_manager.render(
            _PROMPT_NAME, _PROMPT_VERSION, service=service, observation=observation
        )
        response = await catalogue.complete(_LLM_PURPOSE, prompt)
        diagnosis, remediation = _parse_diagnosis_response(response)

        postmortem_text = (
            f"Incident postmortem for {service}\n\n"
            f"Observation: {observation}\n\n"
            f"Diagnosis: {diagnosis}\n\n"
            f"Remediation: {remediation}\n"
        )

        # Settings/PostgresStore/blob_store/vector_store constructed
        # fresh per call - same posture as every other hand-written
        # skill in this package (e.g. `create_chart.py`).
        settings = Settings()
        db = PostgresStore(settings)
        try:
            blob_store = LocalBlobStore(settings.blob_store_root)
            vector_store = PgVectorStore(db)
            attachment_id = await ingest(
                postmortem_text.encode("utf-8"),
                "text/plain",
                source=f"postmortem:{service}",
                blob_store=blob_store,
                parser_registry=parser_registry,
                embedder=get_default_embedder(),
                vector_store=vector_store,
                db=db,
            )
        finally:
            await db.dispose()

        return {
            "diagnosis": diagnosis,
            "remediation": remediation,
            "attachment_id": str(attachment_id),
        }


register_skill(DiagnoseAndRecordIncidentSkill())
