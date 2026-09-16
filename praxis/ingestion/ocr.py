# praxis/ingestion/ocr.py
"""Tesseract-backed `OcrEngine` (spec §2, §5 step 2).

The MVP-deployed OCR backend per spec §2's table: "Tesseract -
`pytesseract` + `pypdfium2` (page rasterization), both open-source,
fully local". `pypdfium2` isn't needed here directly - page
rasterization for scanned *PDFs* is `DocumentParser`/`unstructured`'s
concern (it depends on `pypdfium2` transitively via its `pdf` extra);
this module handles the already-an-image case (`ImageOcrParser`'s
`image/png`, `image/jpeg`), where there's no page to rasterize first.
"""
from __future__ import annotations

import asyncio
import io

import pytesseract
from PIL import Image

from praxis.core.interfaces import OcrEngine


class TesseractOcrEngine(OcrEngine):
    """Runs the local Tesseract binary via `pytesseract` (spec §2's `OcrEngine` default)."""

    async def extract_text(self, image_bytes: bytes) -> str:
        # pytesseract.image_to_string shells out to the Tesseract binary
        # and blocks until it returns - run it off the event loop thread
        # so a slow OCR call doesn't stall other async work.
        return await asyncio.to_thread(self._extract_sync, image_bytes)

    @staticmethod
    def _extract_sync(image_bytes: bytes) -> str:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return pytesseract.image_to_string(image)
