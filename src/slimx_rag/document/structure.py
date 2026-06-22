"""Generic, format-neutral structure detection.

These helpers turn a block of extracted text (e.g. one PDF page) into ordered
``ParsedElement``s, infer a page title, and classify a broad ``PageType``. All rules are
generic — uppercase field labels, date density, section keywords — and contain **no**
document- or model-specific names. The same logic serves PDF pages and plain-text blocks.
"""

from __future__ import annotations

import re

from .model import ElementType, PageType, ParsedElement

# A standalone field label line: short, mostly uppercase, e.g. "KEY DETAIL", "CONTEXT
# TOKENS", "SOURCES", "LICENSE". Generic — recognises the *shape* of a label, not any
# specific label text. Allows digits, spaces and a few separators; rejects long lines and
# sentence-like text (trailing punctuation, too many words).
_LABEL_CHARS = r"A-Z0-9 &/\-\.\(\)\+"
_STANDALONE_LABEL = re.compile(rf"^[A-Z][{_LABEL_CHARS}]{{1,38}}$")
_INLINE_LABEL = re.compile(rf"^(?P<label>[A-Z][{_LABEL_CHARS}]{{1,38}}?):\s+(?P<value>\S.*)$")

_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MAX_LABEL_WORDS = 6
_MAX_TITLE_LEN = 90


def looks_like_label(line: str) -> bool:
    """True when ``line`` is a standalone uppercase field label (not a sentence)."""
    s = line.strip()
    if not s or len(s.split()) > _MAX_LABEL_WORDS:
        return False
    # Require a real mix of letters; reject all-digits or a single short token like "OK".
    if not any(c.isalpha() for c in s):
        return False
    return bool(_STANDALONE_LABEL.match(s))


def _split_inline_label(line: str) -> tuple[str, str] | None:
    m = _INLINE_LABEL.match(line.strip())
    if not m:
        return None
    label = m.group("label").strip()
    if len(label.split()) > _MAX_LABEL_WORDS:
        return None
    return label, m.group("value").strip()


def _norm_paragraphs(text: str) -> list[str]:
    """Split a block into paragraphs on blank lines, preserving intra-paragraph newlines."""
    blocks = re.split(r"\n[ \t]*\n", text)
    return [b.strip() for b in blocks if b.strip()]


def structure_block(
    text: str,
    *,
    id_prefix: str,
    page_number: int | None,
    start_ordinal: int = 0,
) -> tuple[list[ParsedElement], str | None, PageType]:
    """Structure a text block into elements; return (elements, inferred_title, page_type).

    Field-block rule: a standalone uppercase label line absorbs the following non-empty,
    non-label lines as its value, so a label such as ``KEY DETAIL`` is never separated
    from the sentence that follows it. ``LABEL: value`` on one line is likewise kept whole.
    """
    raw_lines = text.splitlines()
    elements: list[ParsedElement] = []
    title: str | None = None
    ordinal = start_ordinal
    field_count = 0

    def _emit(el_type: ElementType, body: str, *, metadata: dict[str, object] | None = None) -> None:
        nonlocal ordinal
        body = body.strip()
        if not body:
            return
        elements.append(
            ParsedElement(
                element_id=f"{id_prefix}#e{ordinal}",
                ordinal=ordinal,
                element_type=el_type,
                text=body,
                page_number=page_number,
                metadata=metadata or {},
            )
        )
        ordinal += 1

    # First meaningful, non-label line becomes the page/section title.
    for ln in raw_lines:
        s = ln.strip()
        if s and not looks_like_label(s) and len(s) <= _MAX_TITLE_LEN and _split_inline_label(s) is None:
            title = s
            break

    i = 0
    n = len(raw_lines)
    pending: list[str] = []  # buffered narrative lines -> paragraph(s)

    def _flush_pending() -> None:
        if not pending:
            return
        block = "\n".join(pending).strip()
        pending.clear()
        for para in _norm_paragraphs(block):
            _emit(ElementType.PARAGRAPH, para)

    while i < n:
        line = raw_lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue

        inline = _split_inline_label(s)
        if inline is not None:
            _flush_pending()
            label, value = inline
            _emit(ElementType.FIELD, f"{label}: {value}", metadata={"label": label})
            field_count += 1
            i += 1
            continue

        if looks_like_label(s):
            _flush_pending()
            # Absorb following non-empty, non-label lines as this field's value.
            value_lines: list[str] = []
            j = i + 1
            while j < n:
                t = raw_lines[j].strip()
                if not t:
                    break
                if looks_like_label(t) or _split_inline_label(t) is not None:
                    break
                value_lines.append(t)
                j += 1
            if value_lines:
                _emit(
                    ElementType.FIELD,
                    f"{s}: {' '.join(value_lines)}",
                    metadata={"label": s},
                )
                field_count += 1
                i = j
            else:
                # Bare label with no value nearby — treat as a heading rather than dropping.
                _emit(ElementType.HEADING, s)
                i += 1
            continue

        pending.append(line)
        i += 1

    _flush_pending()
    page_type = infer_page_type(text, elements, title, field_count=field_count)
    return elements, title, page_type


def infer_page_type(
    text: str,
    elements: list[ParsedElement],
    title: str | None,
    *,
    field_count: int,
) -> PageType:
    """Classify a page/block from generic signals (field density, dates, keywords)."""
    lowered = text.lower()
    headline = (title or "").lower()
    dates = len(_DATE.findall(text))

    if "glossary" in headline or "glossary of terms" in lowered:
        return PageType.GLOSSARY
    if "license" in headline or "appendix" in headline:
        return PageType.APPENDIX
    # An index/timeline page: titled as such, or dominated by dated rows.
    if any(k in headline for k in ("timeline", "index", "release history", "changelog")):
        return PageType.TIMELINE
    if dates >= 4 and field_count <= 1:
        return PageType.TIMELINE
    # A fact sheet: a title plus several distinct field blocks.
    if field_count >= 3 and title:
        return PageType.FACT_SHEET
    if field_count >= 5:
        return PageType.FACT_SHEET
    if elements and all(e.element_type == ElementType.PARAGRAPH for e in elements):
        return PageType.NARRATIVE
    return PageType.UNKNOWN


# --- source-type detection (SlimX-RAG owned; independent of any caller) ----------------
_CODE_EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".c", ".cpp", ".rb"}
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def detect_source_type(filename: str, mime_type: str | None) -> str:
    """Return one of: pdf | docx | markdown | code | text."""
    lower = (filename or "").lower()
    if lower.endswith(".pdf") or mime_type == "application/pdf":
        return "pdf"
    if lower.endswith(".docx") or mime_type == _DOCX_MIME:
        return "docx"
    if lower.endswith((".md", ".markdown")) or mime_type == "text/markdown":
        return "markdown"
    if any(lower.endswith(ext) for ext in _CODE_EXTS):
        return "code"
    return "text"
