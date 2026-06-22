"""Native, page-preserving PDF parser (pypdf).

Pages are parsed one at a time and never concatenated before structural processing, so
one-based human page numbers survive into every element and chunk. Repeated headers and
footers are detected conservatively (a short line recurring on most pages) and removed;
unique content is never dropped based on position. OCR is intentionally not used.
"""

from __future__ import annotations

import math
from io import BytesIO

from ..model import DocumentSource, ParsedDocument, ParsedPage
from ..parser import DocumentParseError
from ..structure import detect_source_type, structure_block

try:  # optional dependency (the `doc` extra); import at module scope so tests can patch it
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - exercised only without the extra installed
    PdfReader = None  # type: ignore[assignment, misc]

PARSER_NAME = "native-pdf"
PARSER_VERSION = "1"

_HEADER_FOOTER_MAX_LEN = 80  # only short lines are candidate headers/footers
_HEADER_FOOTER_MIN_PAGES = 3  # don't bother de-noising tiny documents


def _as_bytes(source: DocumentSource) -> bytes:
    content = source.content
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        # A PDF source should be bytes; tolerate latin-1 round-tripped bytes defensively.
        return content.encode("latin-1", errors="ignore")
    raise DocumentParseError("PDF source has no byte content")


def _edge_lines(text: str) -> tuple[str | None, str | None]:
    """First and last non-empty stripped lines of a page (header/footer candidates)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, None
    return lines[0], lines[-1]


def _detect_repeated_edges(page_texts: list[str]) -> set[str]:
    """Lines that recur as a page edge on most pages — treated as headers/footers."""
    if len(page_texts) < _HEADER_FOOTER_MIN_PAGES:
        return set()
    counts: dict[str, int] = {}
    for text in page_texts:
        top, bottom = _edge_lines(text)
        for line in {top, bottom}:
            if line and len(line) <= _HEADER_FOOTER_MAX_LEN:
                counts[line] = counts.get(line, 0) + 1
    threshold = max(_HEADER_FOOTER_MIN_PAGES, math.ceil(0.6 * len(page_texts)))
    return {line for line, c in counts.items() if c >= threshold}


def _strip_edges(text: str, repeated: set[str]) -> str:
    if not repeated:
        return text
    kept = [ln for ln in text.splitlines() if ln.strip() not in repeated]
    return "\n".join(kept)


class PdfParser:
    name = PARSER_NAME
    version = PARSER_VERSION

    def __init__(self, *, max_pages: int = 2000) -> None:
        self.max_pages = max_pages

    def supports(self, source: DocumentSource) -> bool:
        return detect_source_type(source.filename, source.mime_type) == "pdf"

    def parse(self, source: DocumentSource) -> ParsedDocument:
        if PdfReader is None:  # pragma: no cover - import guard
            raise DocumentParseError(
                "PDF parsing requires the optional dependency 'pypdf' (install the `doc` extra)."
            )
        data = _as_bytes(source)
        try:
            reader = PdfReader(BytesIO(data))
            raw_pages = list(reader.pages)
        except Exception as exc:  # pypdf raises a variety of types on malformed input
            raise DocumentParseError(f"Could not read PDF: {type(exc).__name__}") from exc

        if len(raw_pages) > self.max_pages:
            raise DocumentParseError(
                f"PDF has {len(raw_pages)} pages; the maximum supported is {self.max_pages}."
            )

        # Extract per-page text first (no concatenation), then de-noise repeated edges.
        page_texts: list[str] = []
        for page in raw_pages:
            try:
                page_texts.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 — one bad page shouldn't sink the whole doc
                page_texts.append("")
        repeated = _detect_repeated_edges(page_texts)

        doc_id = source.document_id
        title = str(source.metadata.get("title") or "") or _stem(source.filename)
        warnings: list[str] = []
        pages: list[ParsedPage] = []

        for i, raw in enumerate(page_texts, start=1):
            cleaned = _strip_edges(raw, repeated).strip()
            if not cleaned:
                warnings.append(f"page {i} produced no extractable text")
                pages.append(ParsedPage(page_number=i, elements=(), title=None, text=""))
                continue
            elements, page_title, page_type = structure_block(
                cleaned, id_prefix=f"{doc_id}#p{i}", page_number=i
            )
            pages.append(
                ParsedPage(
                    page_number=i,
                    elements=tuple(elements),
                    title=page_title,
                    page_type=page_type,
                    text=cleaned,
                )
            )

        if repeated:
            warnings.append(f"removed {len(repeated)} repeated header/footer line(s)")

        return ParsedDocument(
            document_id=doc_id,
            title=title,
            source_type="pdf",
            parser_name=self.name,
            parser_version=self.version,
            page_count=len(pages),
            pages=tuple(pages),
            warnings=tuple(warnings),
            metadata={"repeated_edges": sorted(repeated)} if repeated else {},
        )


def _stem(filename: str) -> str:
    base = (filename or "document").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] or base
