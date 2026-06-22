"""Canonical, parser-neutral document representation owned by SlimX-RAG.

Every parser (PDF, DOCX, Markdown, text, code) emits the *same* structures defined
here, so chunking, indexing, retrieval and the HTTP/UI contracts never depend on a
specific parser provider. The representation deliberately separates:

- structure (``ParsedDocument`` -> ``ParsedPage`` -> ``ParsedElement``) produced by parsing;
- retrieval units (``RetrievalChunk``) produced by chunking.

Display text is kept separate from embedding text: a child chunk may *display* a short
passage while *embedding* a version prefixed with its document/page/section identity so
it remains understandable when retrieved alone (a chunk must never look like an anonymous
"68.6 KiB ..." fragment).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ElementType(str, Enum):
    """Kind of a parsed structural element. ``str`` mixin keeps it JSON-friendly."""

    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE = "code"
    FIELD = "field"  # a "LABEL: value" pair kept together as one unit
    CAPTION = "caption"
    OTHER = "other"


class PageType(str, Enum):
    """Broad, inferred page/entry classification used as a retrieval signal.

    Intentionally coarse and never a brittle per-document switch — it is derived from
    generic structure (field density, dates, keywords), not from any model name.
    """

    FACT_SHEET = "fact_sheet"
    TIMELINE = "timeline"  # timeline or index page
    GLOSSARY = "glossary"
    APPENDIX = "appendix"  # appendix or license
    TABLE = "table"
    NARRATIVE = "narrative"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DocumentSource:
    """The original document handed to a parser.

    ``content`` carries raw bytes (binary formats: PDF/DOCX) or text (already-decoded
    text/markdown/code). Identity/scoping fields are optional so the model can be used
    both from the HTTP service (workspace/document scoped) and the CLI.
    """

    document_id: str
    filename: str
    mime_type: str | None = None
    source_type: str | None = None  # pdf|docx|markdown|text|code — filled by detection
    content: bytes | str | None = None
    workspace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedElement:
    """A single structural unit (paragraph, heading, field block, table, code, …)."""

    element_id: str  # stable within the parsed document
    ordinal: int  # global order across the document
    element_type: ElementType
    text: str  # display text
    page_number: int | None = None
    section: str | None = None
    section_path: tuple[str, ...] = ()
    parent_element_id: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedPage:
    """One page (PDF) or one synthetic page (non-paginated formats use page 1)."""

    page_number: int  # 1-based human page number
    elements: tuple[ParsedElement, ...] = ()
    title: str | None = None
    page_type: PageType = PageType.UNKNOWN
    text: str = ""  # normalized page text
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip() and not self.elements


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """A fully parsed document: ordered pages, each with ordered elements."""

    document_id: str
    title: str
    source_type: str
    parser_name: str
    parser_version: str
    page_count: int | None = None  # None => not paginated (md/docx/text/code)
    pages: tuple[ParsedPage, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def elements(self) -> list[ParsedElement]:
        """Flattened, ordered elements across all pages."""
        return [el for page in self.pages for el in page.elements]

    @property
    def element_count(self) -> int:
        return sum(len(p.elements) for p in self.pages)


@dataclass(frozen=True, slots=True)
class RetrievalChunk:
    """A retrieval unit produced by chunking a ``ParsedDocument``.

    ``display_text`` is what a reader sees; ``embedding_text`` carries the identity
    prefix (document / page / section) so the vector remains self-describing. Both the
    embedding text and the metadata preserve parent identity even when the displayed
    passage is short.
    """

    chunk_id: str
    document_id: str
    parent_id: str
    display_text: str
    embedding_text: str
    token_count: int = 0
    page_number: int | None = None
    section: str | None = None
    section_path: tuple[str, ...] = ()
    page_type: PageType = PageType.UNKNOWN
    element_types: tuple[ElementType, ...] = ()
    source_title: str = ""
    ordinal: int = 0  # ordinal within the parent
    forced_split: bool = False  # True when an oversized element was hard-split
    metadata: dict[str, Any] = field(default_factory=dict)
