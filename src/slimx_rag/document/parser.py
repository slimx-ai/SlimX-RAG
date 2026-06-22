"""The ``DocumentParser`` interface, a registry, and the default dispatch.

A parser turns a :class:`DocumentSource` into a :class:`ParsedDocument`. New parser
providers implement this Protocol and register with a :class:`ParserRegistry`; nothing
downstream (chunking, indexing, retrieval, UI contracts) needs to change. The native
parsers live under ``slimx_rag.document.parsers`` and are wired into the default registry
by :func:`get_default_registry`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .model import DocumentSource, ParsedDocument
from .structure import detect_source_type


class DocumentError(Exception):
    """Base class for document parsing errors (safe to surface; never carries content)."""


class UnsupportedDocumentError(DocumentError):
    """No registered parser can handle the given source."""


class DocumentParseError(DocumentError):
    """A parser failed on a structurally invalid or unreadable document."""


@runtime_checkable
class DocumentParser(Protocol):
    """Provider interface: emit the canonical representation for a source it supports."""

    name: str
    version: str

    def supports(self, source: DocumentSource) -> bool:
        """True when this parser can parse ``source`` (by detected type/MIME/extension)."""
        ...

    def parse(self, source: DocumentSource) -> ParsedDocument:
        """Parse ``source`` into a :class:`ParsedDocument`. Raises on unreadable input."""
        ...


class ParserRegistry:
    """Ordered collection of parsers; the first that ``supports`` a source wins."""

    def __init__(self) -> None:
        self._parsers: list[DocumentParser] = []

    def register(self, parser: DocumentParser) -> None:
        self._parsers.append(parser)

    def parsers(self) -> tuple[DocumentParser, ...]:
        return tuple(self._parsers)

    def for_source(self, source: DocumentSource) -> DocumentParser:
        for parser in self._parsers:
            if parser.supports(source):
                return parser
        detected = source.source_type or detect_source_type(source.filename, source.mime_type)
        raise UnsupportedDocumentError(
            f"No registered parser supports source_type={detected!r} "
            f"(filename suffix / mime={source.mime_type!r})"
        )

    def parse(self, source: DocumentSource) -> ParsedDocument:
        return self.for_source(source).parse(source)


def get_default_registry() -> ParserRegistry:
    """Build a registry wired with the native parsers (PDF, DOCX, Markdown, code, text).

    Order matters: more specific parsers are registered before the catch-all text parser.
    Imports are local so importing this module never forces optional deps (pypdf, docx).
    """
    from .parsers.code import CodeParser
    from .parsers.docx import DocxParser
    from .parsers.markdown import MarkdownParser
    from .parsers.pdf import PdfParser
    from .parsers.text import TextParser

    registry = ParserRegistry()
    registry.register(PdfParser())
    registry.register(DocxParser())
    registry.register(MarkdownParser())
    registry.register(CodeParser())
    registry.register(TextParser())  # catch-all; must be last
    return registry


def parse_document(source: DocumentSource, *, registry: ParserRegistry | None = None) -> ParsedDocument:
    """Parse ``source`` using ``registry`` (or the default native registry)."""
    return (registry or get_default_registry()).parse(source)
