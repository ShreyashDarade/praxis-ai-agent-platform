# praxis/ingestion/parsers/registry.py
"""Self-registering parser registry (spec §5 step 2, mirrors §6's connector pattern).

`praxis/connectors/factory.py` fixed a real bug in Phase 2: a hardcoded
`if/elif` dispatch chain over connector types required editing core
code to add connector #5. That pattern is explicitly forbidden here too
- adding support for a new file type must be "one new parser class
registered in `ParserRegistry`; nothing else changes" (spec §5).

Parsers are simpler than connectors in one respect: they don't need a
"configured?" check (no credentials involved - a parser is always
available). So this registry skips the ConnectorFactory-style
name -> `is_configured`/`build` indirection and just keeps parser
*instances*, keyed by every mime type each one declares support for via
`Parser.supported_mime_types`.

Each parser module (`text_parser.py`, `document_parser.py`, ...) calls
`register_parser(SomeParser(...))` once, at import time - that's the
self-registration. `discover_parsers()` below imports every flat module
in this package (not subpackages - parsers here are single files, unlike
connectors' `<name>/connector.py` subpackage layout) to trigger it.
"""
from __future__ import annotations

import importlib
import pkgutil

from praxis.core.interfaces import Parser

_PARSERS_BY_MIME_TYPE: dict[str, Parser] = {}
_DISCOVERED = False


def register_parser(parser: Parser) -> None:
    """Called by a parser module at import time to register itself.

    Registers the parser under every mime type it declares support for.
    Two parsers claiming the same mime type is a programming error, not
    a runtime condition to degrade gracefully from - fail loudly, same
    posture as `register_connector_factory`.
    """
    for mime_type in parser.supported_mime_types:
        if mime_type in _PARSERS_BY_MIME_TYPE:
            existing = _PARSERS_BY_MIME_TYPE[mime_type]
            raise ValueError(
                f"mime type '{mime_type}' is already handled by "
                f"{type(existing).__name__}; {type(parser).__name__} cannot also claim it"
            )
        _PARSERS_BY_MIME_TYPE[mime_type] = parser


def get_parser_for(mime_type: str) -> Parser:
    try:
        return _PARSERS_BY_MIME_TYPE[mime_type]
    except KeyError:
        raise ValueError(f"no parser registered for mime type '{mime_type}'") from None


def all_parsers() -> list[Parser]:
    """Every distinct registered parser instance (de-duplicated across mime types)."""
    seen: dict[int, Parser] = {id(p): p for p in _PARSERS_BY_MIME_TYPE.values()}
    return list(seen.values())


def discover_parsers() -> None:
    """Import every flat module in `praxis.ingestion.parsers` once.

    Idempotent and safe to call repeatedly (e.g. once per test) - the
    module-level `_DISCOVERED` flag plus Python's own import cache mean
    a second call is a no-op, so no parser is ever registered twice.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    import praxis.ingestion.parsers as _parsers_pkg

    for module_info in pkgutil.iter_modules(_parsers_pkg.__path__):
        if module_info.ispkg or module_info.name in ("registry",):
            continue
        importlib.import_module(f"praxis.ingestion.parsers.{module_info.name}")
    _DISCOVERED = True
