# praxis/ingestion/parsers/email_parser.py
"""RFC 822 / MIME email parser (spec §5 step 2).

Everything here is stdlib `email` with `policy.default`, which is the
modern `EmailMessage` API rather than the legacy `compat32` one. That
choice does real work: it decodes RFC 2047 encoded-words in headers
(so a `=?utf-8?B?...?=` Subject comes out as the words the sender
wrote, not as base64), and it decodes each text part's
Content-Transfer-Encoding and charset for us, so quoted-printable and
base64 bodies arrive as ordinary `str`.

Headers come first and body second on purpose. An email's routing
metadata is frequently the most searchable thing about it ("what did
Alice send about the outage in March"), and putting From/To/Subject/Date
in the first chunk means that metadata survives chunking instead of
being diluted across the whole message.

Attachments are *listed by name and type, never parsed*. That is a
security boundary, not laziness: recursively parsing attacker-supplied
attachment payloads is exactly the amplification problem
`archive_parser.py` exists to bound, and an email is a strictly more
hostile container than a zip (it arrives unsolicited). A caller that
genuinely wants an attachment's contents should extract those bytes
deliberately and route them through the registry itself, with whatever
size limits that caller's context justifies.

Other limitations, stated rather than implied:
- Header values are reproduced verbatim. A spoofed `From`, a display
  name containing a lookalike address, or header-injected newlines are
  not detected or normalised here; this parser extracts text, it does
  not authenticate mail.
- No DKIM/SPF/ARC verification is attempted, so nothing in the output
  should be read as evidence the sender is who the header claims.
- Only the *first* text body part is used (`get_body`'s plain-then-html
  preference). A message whose content is spread across several
  alternative or inline parts will lose the later ones.
- `email.message_from_bytes` is tolerant by design: malformed MIME
  yields defects rather than an exception, so a badly broken message
  can parse to a thin, mostly-empty result instead of failing loudly.
- No input size bound is imposed here. The stdlib parser allocates
  proportionally to the message, so an unbounded upload path needs its
  ceiling at the upload boundary.
"""
from __future__ import annotations

import asyncio
import email
from email import policy
from email.message import EmailMessage

from bs4 import BeautifulSoup

from praxis.core.interfaces import Parser
from praxis.ingestion.parsers.registry import register_parser

_HEADERS = ("From", "To", "Cc", "Subject", "Date")


class EmailParser(Parser):
    """message/rfc822 -> headers, body text, and an attachment manifest."""

    supported_mime_types = ("message/rfc822",)

    async def parse(self, data: bytes, mime_type: str) -> str:
        return await asyncio.to_thread(self._extract, data)

    @staticmethod
    def _extract(data: bytes) -> str:
        message = email.message_from_bytes(data, policy=policy.default)

        sections = []

        header_lines = [
            f"{name}: {str(message[name]).strip()}"
            for name in _HEADERS
            if message[name] is not None
        ]
        if header_lines:
            sections.append("\n".join(header_lines))

        body = EmailParser._body_text(message)
        if body:
            sections.append(body)

        attachments = EmailParser._attachment_lines(message)
        if attachments:
            sections.append("\n".join(["Attachments:", *attachments]))

        # Blank-line separated sections: the same boundary the chunkers
        # split on, so headers, body and manifest stay coherent units
        # rather than being cut mid-way.
        return "\n\n".join(sections)

    @staticmethod
    def _body_text(message: EmailMessage) -> str:
        body = message.get_body(preferencelist=("plain", "html"))
        if body is None:
            return ""
        content = body.get_content()
        if body.get_content_type() == "text/html":
            # HTML-only mail (marketing, most ticketing systems) is
            # common enough that falling back to it beats returning
            # nothing; reduced with the same extractor DocumentParser
            # uses for text/html so the two paths agree.
            return BeautifulSoup(content, "html.parser").get_text(separator="\n", strip=True)
        return content.strip()

    @staticmethod
    def _attachment_lines(message: EmailMessage) -> list[str]:
        lines = []
        for attachment in message.iter_attachments():
            # An attachment with no filename is legal MIME (inline
            # images referenced by Content-ID often have none); naming
            # it explicitly keeps the count honest rather than dropping
            # it from the manifest.
            filename = attachment.get_filename() or "(unnamed)"
            lines.append(f"  - {filename} ({attachment.get_content_type()})")
        return lines


register_parser(EmailParser())
