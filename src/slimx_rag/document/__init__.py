"""Canonical document representation and native structured parsing for SlimX-RAG.

Public API:

- model: :class:`DocumentSource`, :class:`ParsedDocument`, :class:`ParsedPage`,
  :class:`ParsedElement`, :class:`RetrievalChunk`, :class:`ElementType`, :class:`PageType`;
- parsing: :class:`DocumentParser` (Protocol), :class:`ParserRegistry`,
  :func:`get_default_registry`, :func:`parse_document`;
- helpers: :func:`detect_source_type`.
"""

from __future__ import annotations

from .model import (
    DocumentSource,
    ElementType,
    PageType,
    ParsedDocument,
    ParsedElement,
    ParsedPage,
    RetrievalChunk,
)
from .parser import (
    DocumentError,
    DocumentParseError,
    DocumentParser,
    ParserRegistry,
    UnsupportedDocumentError,
    get_default_registry,
    parse_document,
)
from .structure import detect_source_type

__all__ = [
    "DocumentSource",
    "ParsedDocument",
    "ParsedPage",
    "ParsedElement",
    "RetrievalChunk",
    "ElementType",
    "PageType",
    "DocumentParser",
    "ParserRegistry",
    "UnsupportedDocumentError",
    "DocumentParseError",
    "DocumentError",
    "get_default_registry",
    "parse_document",
    "detect_source_type",
]
