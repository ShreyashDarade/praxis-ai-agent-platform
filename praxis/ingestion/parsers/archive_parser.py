# praxis/ingestion/parsers/archive_parser.py
"""ZIP archive parser (spec §5 step 2) - expands members and delegates
each one back to the parser registry.

This is the one parser whose *job* is to hand attacker-controlled bytes
to other parsers, so the bounds it enforces are the point of the module,
not a footnote. They are all hard refusals: exceeding any of them raises
`ArchiveSafetyError` and abandons the whole archive. Nothing here ever
truncates quietly and returns a partial result, because a partial result
that looks complete is how a bomb succeeds at being ignored.

The bounds, and why each number
-------------------------------
- `MAX_MEMBER_COUNT` (512). A "file count bomb" needs no compression
  ratio at all: a million empty entries costs almost nothing to build
  and forces a million registry lookups and parser invocations. 512 is
  far above any archive a human assembles to be read.
- `MAX_MEMBER_UNCOMPRESSED_BYTES` (8 MiB). Matches `XmlParser`'s own
  input cap deliberately, so a member cannot arrive at a downstream
  parser larger than that parser was designed to accept.
- `MAX_TOTAL_UNCOMPRESSED_BYTES` (32 MiB). Every member's bytes are held
  in memory at once (members are parsed after extraction, not streamed),
  so this number *is* the peak memory this parser can be made to
  allocate for payloads. Enforced twice: once as a cheap pre-flight over
  the central directory's declared sizes, and then - authoritatively -
  as a running total of bytes actually read. The pre-flight alone would
  be worthless, since declared sizes are attacker-controlled fields in a
  header the attacker wrote.
- `MAX_COMPRESSION_RATIO` (100:1), applied only to members that expand
  past `RATIO_CHECK_MIN_BYTES` (64 KiB). DEFLATE tops out near 1032:1,
  so 100:1 is comfortably inside the danger zone while leaving ordinary
  text (which compresses maybe 4:1) untouched. The 64 KiB floor exists
  because small, highly repetitive files legitimately exceed 100:1 and
  cannot possibly be a bomb - the absolute caps above already make a
  64 KiB output harmless.
- **No recursion.** Expansion is exactly one level deep: the archive
  handed to `parse()` is expanded; an archive *inside* it is listed by
  name and left alone. Recursive nesting is the classic amplification
  trick (42.zip is 42 KB and six levels deep), and refusing depth
  entirely removes the multiplication rather than trying to bound it.
- **Member names.** Absolute paths and any `..` component are refused.

What these bounds do NOT protect against
----------------------------------------
- The path-traversal refusal is *defensive, not load-bearing here*.
  This parser never writes a member to disk - it reads into memory - so
  a `../../etc/passwd` entry is not directly exploitable by this code.
  It is refused because member names are reproduced in the output, and
  any downstream consumer that did write them out would be exploitable.
  Nothing else about names is sanitised: Windows reserved device names
  (`CON`, `NUL`), NTFS alternate-data-stream syntax, unicode homoglyphs
  and absurdly long names all pass through.
- Once a member's bytes reach its own parser, that parser's limits (or
  absence of them) govern. This module bounds how much data can be
  *handed over*, not what happens next.
- A member whose parse raises is reported inline and the archive still
  succeeds. That is intentional - one corrupt file should not void a
  readable archive - but it means a caller reading the output must not
  treat "parse succeeded" as "every member was understood".
- Encrypted members are detected and skipped, not decrypted; their
  contents are simply not in the output.
- Symlink members (a POSIX zip stores the target path as the member's
  content) are treated as ordinary files. Harmless here, since nothing
  is written to disk, but it means a symlink shows up as a one-line text
  member rather than as a link.
- `zipfile`'s own tolerance sets the floor: disagreements between local
  headers and the central directory are handled however CPython handles
  them, and this module adds nothing there.
"""
from __future__ import annotations

import asyncio
import io
import mimetypes
import posixpath
import zipfile
from dataclasses import dataclass
from typing import Protocol

import filetype

from praxis.core.interfaces import Parser
from praxis.ingestion.parsers import registry

MAX_MEMBER_COUNT = 512
MAX_MEMBER_UNCOMPRESSED_BYTES = 8 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
MAX_COMPRESSION_RATIO = 100.0
RATIO_CHECK_MIN_BYTES = 64 * 1024

_ZIP_MIME_TYPE = "application/zip"
# Sentinel mime type for a member that exists but cannot be read; kept
# distinct from `None` ("type unknown") so the output can say *why* a
# member produced no text. Not a registered mime type - nothing will
# ever claim it, and `_parse_member` short-circuits before lookup.
_ENCRYPTED_MIME_TYPE = "application/x-encrypted-zip-member"

# An explicit table rather than `mimetypes.guess_type` alone, because on
# Windows `mimetypes` reads HKEY_CLASSES_ROOT: the same archive would
# resolve `.md` or `.csv` differently depending on what the developer
# happens to have installed, and a parser whose output feeds a search
# index cannot be machine-dependent. `mimetypes` is still consulted as a
# fallback for extensions Praxis has no parser for, purely so the output
# can name the type it declined to parse.
_MIME_TYPE_BY_EXTENSION = {
    ".txt": "text/plain",
    ".log": "text/x-log",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".py": "text/x-python",
    ".json": "application/json",
    ".xml": "application/xml",
    ".html": "text/html",
    ".htm": "text/html",
    ".csv": "text/csv",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".eml": "message/rfc822",
    ".zip": _ZIP_MIME_TYPE,
}


class ArchiveSafetyError(ValueError):
    """Raised when an archive trips one of this module's hard bounds.

    A `ValueError` subclass so it fits the failure vocabulary the rest
    of the parser package already uses, while `bound` names the specific
    rule that fired (`"member_count"`, `"member_bytes"`,
    `"total_bytes"`, `"compression_ratio"`, `"member_name"`) so callers
    and tests can assert on the defence rather than on message wording.
    """

    def __init__(self, message: str, *, bound: str) -> None:
        super().__init__(message)
        self.bound = bound


class ParserRegistryLike(Protocol):
    """The one function this parser needs from a registry.

    Declared locally rather than imported from `praxis.ingestion.pipeline`
    (which defines the same shape) so a parser module never has to import
    the pipeline that drives it.
    """

    def get_parser_for(self, mime_type: str) -> Parser: ...


@dataclass(frozen=True)
class _Member:
    name: str
    mime_type: str | None
    payload: bytes


class ArchiveParser(Parser):
    """ZIP -> each member parsed by its own registered parser, bounded."""

    supported_mime_types = (_ZIP_MIME_TYPE,)

    def __init__(self, parser_registry: ParserRegistryLike | None = None) -> None:
        # Injected rather than hardcoded (spec §4 DIP, same shape as
        # `ImageOcrParser`'s `OcrEngine`): the default is the real module
        # registry, and a caller wanting a restricted set of parsers for
        # archive members supplies its own without editing this class.
        self._registry: ParserRegistryLike = (
            registry if parser_registry is None else parser_registry
        )

    async def parse(self, data: bytes, mime_type: str) -> str:
        # Extraction is sync/CPU-bound and is where every bound is
        # enforced, so it goes to a worker thread as one unit; member
        # parsing then happens back on the event loop because the
        # delegated parsers are themselves async.
        members = await asyncio.to_thread(self._read_members, data)

        total_bytes = sum(len(member.payload) for member in members)
        sections = [f"Archive: {len(members)} files, {total_bytes} bytes extracted"]
        for member in members:
            body = await self._parse_member(member)
            sections.append(f"--- {member.name} ({member.mime_type or 'unknown'}) ---\n{body}")

        return "\n\n".join(sections)

    async def _parse_member(self, member: _Member) -> str:
        if member.mime_type == _ZIP_MIME_TYPE:
            return "[nested archive; not expanded]"
        if member.mime_type == _ENCRYPTED_MIME_TYPE:
            return "[encrypted member; not parsed]"
        if member.mime_type is None:
            return "[unrecognised file type; not parsed]"
        try:
            parser = self._registry.get_parser_for(member.mime_type)
        except ValueError as exc:
            return f"[{exc}]"
        try:
            return await parser.parse(member.payload, member.mime_type)
        except Exception as exc:
            # Deliberately broad: a member parser can fail in as many
            # ways as there are formats, and one bad member must not
            # discard the rest of a readable archive. The failure is
            # reported in the output rather than swallowed, so it is
            # visible to whoever reads the extracted text. Note this
            # also catches a member parser's *own* safety refusal (e.g.
            # `XmlSafetyError` on a bomb-shaped member) - correct, since
            # that member was already refused; this archive's own bounds
            # are enforced before any of this and cannot be caught here.
            return f"[parse failed: {type(exc).__name__}: {exc}]"

    @staticmethod
    def _read_members(data: bytes) -> list[_Member]:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()

            if len(infos) > MAX_MEMBER_COUNT:
                raise ArchiveSafetyError(
                    f"archive declares {len(infos)} members, above the {MAX_MEMBER_COUNT} "
                    "member limit; refusing to expand",
                    bound="member_count",
                )

            # Cheap pre-flight so an obviously oversized archive is
            # refused before a single byte is decompressed. Not
            # trustworthy on its own - `file_size` is a field the
            # archive's author wrote - which is why the running total
            # below re-checks against bytes actually read.
            declared_total = sum(info.file_size for info in infos)
            if declared_total > MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise ArchiveSafetyError(
                    f"archive declares {declared_total} uncompressed bytes, above the "
                    f"{MAX_TOTAL_UNCOMPRESSED_BYTES} byte limit; refusing to expand",
                    bound="total_bytes",
                )

            members: list[_Member] = []
            extracted_total = 0
            for info in infos:
                _reject_unsafe_name(info.filename)
                if info.is_dir():
                    continue
                if info.flag_bits & 0x1:
                    members.append(_Member(info.filename, _ENCRYPTED_MIME_TYPE, b""))
                    continue

                payload = _read_bounded(archive, info, extracted_total)
                extracted_total += len(payload)
                members.append(
                    _Member(info.filename, _mime_type_for(info.filename, payload), payload)
                )

            return members


def _read_bounded(archive: zipfile.ZipFile, info: zipfile.ZipInfo, extracted_total: int) -> bytes:
    """Decompress one member, never reading past what is still allowed.

    Reads `limit + 1` bytes rather than `limit`: the extra byte is how
    "this member is exactly at the limit" is told apart from "this
    member is larger than the limit and was truncated", without ever
    materialising more than one byte of overrun.
    """
    limit = min(
        MAX_MEMBER_UNCOMPRESSED_BYTES,
        MAX_TOTAL_UNCOMPRESSED_BYTES - extracted_total,
    )
    with archive.open(info) as handle:
        payload = handle.read(limit + 1)

    if len(payload) > limit:
        if limit == MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise ArchiveSafetyError(
                f"archive member '{info.filename}' expands past the "
                f"{MAX_MEMBER_UNCOMPRESSED_BYTES} byte per-member limit; refusing to expand",
                bound="member_bytes",
            )
        raise ArchiveSafetyError(
            f"archive expands past the {MAX_TOTAL_UNCOMPRESSED_BYTES} byte total limit at "
            f"member '{info.filename}'; refusing to expand",
            bound="total_bytes",
        )

    if len(payload) > RATIO_CHECK_MIN_BYTES and info.compress_size > 0:
        ratio = len(payload) / info.compress_size
        if ratio > MAX_COMPRESSION_RATIO:
            raise ArchiveSafetyError(
                f"archive member '{info.filename}' expands at {ratio:.0f}:1, above the "
                f"{MAX_COMPRESSION_RATIO:.0f}:1 compression ratio limit; refusing to expand",
                bound="compression_ratio",
            )

    return payload


def _reject_unsafe_name(name: str) -> None:
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        raise ArchiveSafetyError(
            f"archive member '{name}' is an absolute path; refusing to expand",
            bound="member_name",
        )
    # `C:` / `C:dir` - a Windows drive-qualified name, absolute or
    # drive-relative, neither of which belongs inside an archive.
    if len(normalized) >= 2 and normalized[1] == ":":
        raise ArchiveSafetyError(
            f"archive member '{name}' is a drive-qualified path; refusing to expand",
            bound="member_name",
        )
    if ".." in posixpath.normpath(normalized).split("/"):
        raise ArchiveSafetyError(
            f"archive member '{name}' escapes the archive root via '..'; refusing to expand",
            bound="member_name",
        )


def _mime_type_for(name: str, payload: bytes) -> str | None:
    extension = posixpath.splitext(name.replace("\\", "/"))[1].lower()
    if extension in _MIME_TYPE_BY_EXTENSION:
        return _MIME_TYPE_BY_EXTENSION[extension]

    # Content sniffing for members with no (or an unknown) extension.
    # `filetype` is the pure-Python sniffer; `python-magic` is
    # deliberately never used here - it segfaults on this platform, as
    # pyproject.toml's Windows caveat documents at length.
    sniffed = filetype.guess(payload)
    if sniffed is not None:
        return sniffed.mime

    if extension:
        return mimetypes.guess_type(name)[0]
    return None


registry.register_parser(ArchiveParser())
