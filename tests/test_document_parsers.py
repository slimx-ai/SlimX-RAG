"""Native structured parsing: page preservation, structure, and field cohesion.

The PDF test uses a faked ``PdfReader`` so it is deterministic and needs no binary
fixture; it still exercises the real page-loop, structuring, title inference and
header/footer removal logic.
"""

from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import pytest

from slimx_rag.document import (
    DocumentSource,
    ElementType,
    PageType,
    ParserRegistry,
    UnsupportedDocumentError,
    detect_source_type,
    parse_document,
)
from slimx_rag.document.parsers import pdf as pdf_module
from slimx_rag.document.parsers.docx import DocxParser
from slimx_rag.document.parsers.pdf import PdfParser

# --- a deterministic, page-structured gallery (header + footer repeat on every page) ---
_PAGE_1 = "\n".join(
    [
        "LLM ARCHITECTURE GALLERY",
        "Model Index",
        "Kimi K2.5 2026-01-10",
        "Kimi K2.6 2026-02-20",
        "Kimi K2.7 2026-03-30",
        "GLM-5.1 2026-04-20",
        "CONFIDENTIAL",
    ]
)
_PAGE_2 = "\n".join(
    [
        "LLM ARCHITECTURE GALLERY",
        "Kimi K2.6",
        "SCALE",
        "1T total, 32B active",
        "CONTEXT TOKENS",
        "256,000",
        "ATTENTION",
        "61-MLA",
        "KEY DETAIL",
        "Uses the same text architecture as Kimi K2.5 with refined routing.",
        "CONFIDENTIAL",
    ]
)
_PAGE_3 = "\n".join(["LLM ARCHITECTURE GALLERY", "CONFIDENTIAL"])  # only header/footer


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakeReader:
    def __init__(self, _stream: object) -> None:
        self.pages = [_FakePage(_PAGE_1), _FakePage(_PAGE_2), _FakePage(_PAGE_3)]


def _parse_pdf(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setattr(pdf_module, "PdfReader", _FakeReader)
    src = DocumentSource(
        document_id="doc1",
        filename="gallery.pdf",
        mime_type="application/pdf",
        content=b"%PDF-1.4 fake",
    )
    return PdfParser().parse(src)


def test_pdf_preserves_pages_and_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = _parse_pdf(monkeypatch)
    assert doc.source_type == "pdf"
    assert doc.parser_name == "native-pdf"
    assert doc.page_count == 3
    assert [p.page_number for p in doc.pages] == [1, 2, 3]


def test_pdf_infers_page_title_and_type(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = _parse_pdf(monkeypatch)
    page1, page2, page3 = doc.pages
    assert page1.page_type == PageType.TIMELINE  # index/timeline page
    assert page2.title == "Kimi K2.6"
    assert page2.page_type == PageType.FACT_SHEET
    assert page3.is_empty


def test_pdf_keeps_key_detail_label_with_value(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = _parse_pdf(monkeypatch)
    page2 = doc.pages[1]
    fields = [e for e in page2.elements if e.element_type == ElementType.FIELD]
    key_detail = [f for f in fields if f.text.startswith("KEY DETAIL")]
    assert len(key_detail) == 1
    # The label and its actual sentence must live in ONE element (the reported failure).
    assert "refined routing" in key_detail[0].text
    # And every field element knows which page it belongs to.
    assert all(f.page_number == 2 for f in fields)


def test_pdf_strips_repeated_header_and_footer(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = _parse_pdf(monkeypatch)
    all_text = "\n".join(e.text for e in doc.elements)
    assert "LLM ARCHITECTURE GALLERY" not in all_text
    assert "CONFIDENTIAL" not in all_text
    assert any("repeated header/footer" in w for w in doc.warnings)
    assert any("page 3" in w for w in doc.warnings)


def test_markdown_preserves_hierarchy_and_code() -> None:
    md = (
        "# Title\n\nIntro paragraph.\n\n## Section A\n\nText under A.\n\n"
        "```python\nprint('hi')\n```\n\n- item 1\n- item 2\n"
    )
    doc = parse_document(DocumentSource(document_id="d", filename="n.md", content=md))
    assert doc.title == "Title"
    headings = [e.text for e in doc.elements if e.element_type == ElementType.HEADING]
    assert headings == ["Title", "Section A"]
    code = [e for e in doc.elements if e.element_type == ElementType.CODE]
    assert code and "print('hi')" in code[0].text
    under_a = [e for e in doc.elements if e.text == "Text under A."][0]
    assert under_a.section_path == ("Title", "Section A")
    assert sum(1 for e in doc.elements if e.element_type == ElementType.LIST_ITEM) == 2


def test_text_parser_keeps_field_blocks() -> None:
    txt = "Some intro.\n\nKEY DETAIL\nThe answer is forty-two.\n"
    doc = parse_document(DocumentSource(document_id="d", filename="n.txt", content=txt))
    assert doc.source_type == "text"
    fields = [e for e in doc.elements if e.element_type == ElementType.FIELD]
    assert any("KEY DETAIL" in f.text and "forty-two" in f.text for f in fields)


def test_code_parser_splits_into_blocks() -> None:
    code = "import os\n\ndef foo():\n    return 1\n\ndef bar():\n    return 2\n"
    doc = parse_document(DocumentSource(document_id="d", filename="m.py", content=code))
    assert doc.source_type == "code"
    blocks = [e for e in doc.elements if e.element_type == ElementType.CODE]
    assert len(blocks) == 3
    assert blocks[1].text.startswith("def foo")
    assert blocks[0].metadata.get("language") == "python"


def _minimal_docx_bytes() -> bytes:
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        '<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t>My Title</w:t></w:r></w:p>'
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Section</w:t></w:r></w:p>'
        "<w:p><w:r><w:t>Body text.</w:t></w:r></w:p>"
        "<w:tbl><w:tr>"
        "<w:tc><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>B</w:t></w:r></w:p></w:tc>"
        "</w:tr></w:tbl>"
        "</w:body></w:document>"
    )
    buf = BytesIO()
    with ZipFile(buf, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return buf.getvalue()


def test_docx_zip_fallback_preserves_headings_and_tables() -> None:
    parser = DocxParser()
    rows = parser._read_with_zip(_minimal_docx_bytes())
    doc = parser._build(DocumentSource(document_id="d", filename="f.docx"), rows)
    assert doc.title == "My Title"
    heads = [e.text for e in doc.elements if e.element_type in (ElementType.HEADING, ElementType.TITLE)]
    assert "Section" in heads
    tables = [e for e in doc.elements if e.element_type == ElementType.TABLE]
    assert tables and "A | B" in tables[0].text
    body = [e for e in doc.elements if e.text == "Body text."][0]
    assert body.section_path[-1] == "Section"


def test_detect_source_type_and_dispatch() -> None:
    assert detect_source_type("a.pdf", None) == "pdf"
    assert detect_source_type("a.docx", None) == "docx"
    assert detect_source_type("a.md", None) == "markdown"
    assert detect_source_type("a.py", None) == "code"
    assert detect_source_type("a.bin", "application/octet-stream") == "text"


def test_registry_raises_on_unsupported() -> None:
    registry = ParserRegistry()
    registry.register(PdfParser())  # only PDF; no catch-all text parser
    with pytest.raises(UnsupportedDocumentError):
        registry.parse(DocumentSource(document_id="d", filename="x.bin", content="hi"))
