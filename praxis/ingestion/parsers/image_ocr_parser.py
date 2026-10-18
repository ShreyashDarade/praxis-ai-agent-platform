# praxis/ingestion/parsers/image_ocr_parser.py
"""OCR-backed parser for scanned/photographed text images (spec §5 step 2).

Delegates the actual OCR work to an injected `OcrEngine` (dependency
inversion, spec §4 DIP) rather than hardcoding `pytesseract` calls
inline - swapping the OCR backend (e.g. a vision-LLM captioning path,
per spec §5) is providing a different `OcrEngine`, never editing this
class.
"""
from __future__ import annotations

from praxis.core.interfaces import OcrEngine, Parser
from praxis.ingestion.ocr import TesseractOcrEngine
from praxis.ingestion.parsers.registry import register_parser


class ImageOcrParser(Parser):
    """PNG/JPEG -> plain text, via whichever `OcrEngine` it's constructed with."""

    supported_mime_types = ("image/png", "image/jpeg")

    def __init__(self, ocr_engine: OcrEngine) -> None:
        self._ocr_engine = ocr_engine

    async def parse(self, data: bytes, mime_type: str) -> str:
        return await self._ocr_engine.extract_text(data)


# Self-registration needs a concrete OcrEngine to construct with; the
# default is the MVP-deployed backend (spec §2), same as every other
# "default backend" registration in this phase (SentenceTransformerEmbedder,
# etc). A caller wanting a different OcrEngine constructs its own
# ImageOcrParser directly rather than going through the registry.
register_parser(ImageOcrParser(TesseractOcrEngine()))
