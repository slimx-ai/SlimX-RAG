"""Citation fidelity: does every returned chunk's attribution match where its text lives?

Independent of ranking quality. For each result the verifier checks, against the
original corpus bytes, that the chunk text is contained in the document it is attributed
to (``metadata.document_id``), that a PDF ``page`` is the page that actually holds the
text, that a Markdown/DOCX ``section`` is the heading the text sits under, and that the
human citation label agrees with the page metadata. Token-sequence containment (ordered,
alphanumeric tokens) is used because parsers legitimately re-join field labels and strip
repeated headers/footers.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from .corpus import Corpus, CorpusDoc

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[.,\-/_][A-Za-z0-9]+)*")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_MIN_COVERAGE = 0.95


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN.finditer(text or "")]


def _coverage(needle: list[str], haystack: list[str]) -> float:
    """Fraction of ``needle`` tokens matched as an ordered subsequence of ``haystack``."""
    if not needle:
        return 1.0
    i = 0
    matched = 0
    for token in needle:
        while i < len(haystack) and haystack[i] != token:
            i += 1
        if i < len(haystack):
            matched += 1
            i += 1
        else:
            break
    return matched / len(needle)


@dataclass(slots=True)
class DocumentText:
    name: str
    kind: str  # pdf | markdown | docx | text
    whole: list[str]
    pages: dict[int, list[str]] = field(default_factory=dict)  # pdf only
    sections: dict[str, list[str]] = field(default_factory=dict)  # markdown/docx only


def _pdf_pages(content: bytes) -> dict[int, list[str]]:
    from pypdf import PdfReader  # dev dependency; the doc extra in production

    reader = PdfReader(BytesIO(content))
    return {number: _tokens(page.extract_text() or "") for number, page in enumerate(reader.pages, start=1)}


def _markdown_sections(text: str) -> tuple[list[str], dict[str, list[str]]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        match = _MD_HEADING.match(line)
        if match:
            if current is not None:
                sections[current] = _tokens("\n".join(buffer))
            current = match.group(2).strip()
            buffer = [current]
            continue
        buffer.append(line)
    if current is not None:
        sections[current] = _tokens("\n".join(buffer))
    return _tokens(text), sections


def _docx_rows(content: bytes) -> list[tuple[bool, str]]:
    """(is_heading, text) rows in document order, from the OOXML body."""
    with zipfile.ZipFile(BytesIO(content)) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    body = root.find(f"{{{_W}}}body")
    rows: list[tuple[bool, str]] = []
    if body is None:
        return rows
    for child in list(body):
        tag = child.tag.split("}")[-1]
        if tag == "p":
            style = None
            ppr = child.find(f"{{{_W}}}pPr")
            if ppr is not None:
                pstyle = ppr.find(f"{{{_W}}}pStyle")
                if pstyle is not None:
                    style = pstyle.get(f"{{{_W}}}val")
            text = "".join(t.text or "" for t in child.iter(f"{{{_W}}}t")).strip()
            if text:
                heading = bool(style and (style.lower().startswith("heading") or style.lower() == "title"))
                rows.append((heading, text))
        elif tag == "tbl":
            cells: list[str] = []
            for tr in child.findall(f"{{{_W}}}tr"):
                for tc in tr.findall(f"{{{_W}}}tc"):
                    cells.append("".join(t.text or "" for t in tc.iter(f"{{{_W}}}t")).strip())
            rows.append((False, " | ".join(cells)))
    return rows


def _docx_sections(content: bytes) -> tuple[list[str], dict[str, list[str]]]:
    rows = _docx_rows(content)
    sections: dict[str, list[str]] = {}
    current: str | None = None
    buffer: list[str] = []
    for heading, text in rows:
        if heading:
            if current is not None:
                sections[current] = _tokens("\n".join(buffer))
            current = text
            buffer = [text]
        else:
            buffer.append(text)
    if current is not None:
        sections[current] = _tokens("\n".join(buffer))
    return _tokens("\n".join(text for _h, text in rows)), sections


def build_document_text(doc: CorpusDoc) -> DocumentText:
    lower = doc.filename.lower()
    if lower.endswith(".pdf"):
        pages = _pdf_pages(doc.content)
        whole = [token for _n, tokens in sorted(pages.items()) for token in tokens]
        return DocumentText(doc.name, "pdf", whole, pages=pages)
    if lower.endswith(".docx"):
        whole, sections = _docx_sections(doc.content)
        return DocumentText(doc.name, "docx", whole, sections=sections)
    if lower.endswith((".md", ".markdown")):
        whole, sections = _markdown_sections(doc.content.decode("utf-8"))
        return DocumentText(doc.name, "markdown", whole, sections=sections)
    return DocumentText(doc.name, "text", _tokens(doc.content.decode("utf-8")))


class FidelityVerifier:
    def __init__(self, corpus: Corpus) -> None:
        self._docs: dict[str, DocumentText] = {doc.name: build_document_text(doc) for doc in corpus.docs}
        self._name_by_id = corpus.name_by_document_id()

    def replace(self, doc: CorpusDoc) -> None:
        self._docs[doc.name] = build_document_text(doc)

    def verify(self, row: dict[str, Any]) -> dict[str, Any]:
        """Return {ok, problems: [...]} for one result row (document_id/page/section/text/citation)."""
        problems: list[str] = []
        name = self._name_by_id.get(str(row.get("document_id")))
        text_tokens = _tokens(row.get("text") or "")
        if name is None or name not in self._docs:
            return {"ok": False, "problems": ["unattributed_document"]}
        doc = self._docs[name]
        if _coverage(text_tokens, doc.whole) < _MIN_COVERAGE:
            problems.append("wrong_source")
        page = row.get("page")
        citation = str(row.get("citation") or "")
        if doc.kind == "pdf":
            if not isinstance(page, int):
                problems.append("missing_page")
            else:
                scores = {number: _coverage(text_tokens, tokens) for number, tokens in doc.pages.items()}
                best = max(scores.values()) if scores else 0.0
                if scores.get(page, 0.0) < _MIN_COVERAGE or scores.get(page, 0.0) < best:
                    problems.append("wrong_page")
                if f"p. {page}" not in citation:
                    problems.append("label_page_mismatch")
        elif page is not None:
            problems.append("page_on_unpaginated_document")
        elif "p. " in citation:
            problems.append("label_page_mismatch")
        section = row.get("section")
        if doc.kind in ("markdown", "docx"):
            if section is None:
                problems.append("missing_section")
            elif str(section) not in doc.sections:
                problems.append("unknown_section")
            else:
                section_scores = {heading: _coverage(text_tokens, tokens) for heading, tokens in doc.sections.items()}
                best_section = max(section_scores.values()) if section_scores else 0.0
                claimed = section_scores.get(str(section), 0.0)
                if claimed < _MIN_COVERAGE or claimed < best_section:
                    problems.append("wrong_section")
        return {"ok": not problems, "problems": problems}


def summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    counters = {
        "checked": 0,
        "unattributed_document": 0,
        "wrong_source": 0,
        "wrong_page": 0,
        "missing_page": 0,
        "page_on_unpaginated_document": 0,
        "label_page_mismatch": 0,
        "wrong_section": 0,
        "missing_section": 0,
        "unknown_section": 0,
        "citation_wrong_locator": 0,
    }
    for row in rows:
        counters["checked"] += 1
        for problem in row.get("problems", []):
            counters[problem] = counters.get(problem, 0) + 1
    counters["citation_wrong_locator"] = counters["wrong_page"] + counters["wrong_section"]
    return counters
