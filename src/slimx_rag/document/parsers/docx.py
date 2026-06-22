"""Native DOCX parser: preserves paragraph order, heading styles, and tables.

Uses ``python-docx`` (the ``doc`` extra) when available, iterating the document body in
order so headings, paragraphs and tables keep their original sequence. Falls back to a
dependency-free zip + XML reader that still preserves paragraph order, heading styles and
table text, so the parser works even without the optional dependency.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from io import BytesIO
from zipfile import BadZipFile, ZipFile

from ..model import DocumentSource, ElementType, PageType, ParsedDocument, ParsedElement, ParsedPage
from ..parser import DocumentParseError
from ..structure import detect_source_type

PARSER_NAME = "native-docx"
PARSER_VERSION = "1"

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_HEADING_LEVEL = re.compile(r"heading\s*(\d)", re.IGNORECASE)


def _as_bytes(source: DocumentSource) -> bytes:
    content = source.content
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf-8", errors="ignore")
    raise DocumentParseError("DOCX source has no byte content")


def _style_to_element(style_name: str | None) -> tuple[ElementType, int | None]:
    if not style_name:
        return ElementType.PARAGRAPH, None
    if style_name.strip().lower() == "title":
        return ElementType.TITLE, 0
    m = _HEADING_LEVEL.search(style_name)
    if m:
        return ElementType.HEADING, int(m.group(1))
    return ElementType.PARAGRAPH, None


class DocxParser:
    name = PARSER_NAME
    version = PARSER_VERSION

    def supports(self, source: DocumentSource) -> bool:
        return detect_source_type(source.filename, source.mime_type) == "docx"

    def parse(self, source: DocumentSource) -> ParsedDocument:
        data = _as_bytes(source)
        try:
            rows = self._read_with_python_docx(data)
        except ImportError:
            rows = self._read_with_zip(data)
        return self._build(source, rows)

    # rows: list of (ElementType, level|None, text)
    def _read_with_python_docx(self, data: bytes) -> list[tuple[ElementType, int | None, str]]:
        import docx  # type: ignore
        from docx.table import Table  # type: ignore
        from docx.text.paragraph import Paragraph  # type: ignore

        document = docx.Document(BytesIO(data))
        body = document.element.body
        out: list[tuple[ElementType, int | None, str]] = []
        for child in body.iterchildren():
            tag = child.tag.split("}")[-1]
            if tag == "p":
                para = Paragraph(child, document)
                text = para.text.strip()
                if not text:
                    continue
                style = para.style.name if para.style is not None else None
                el_type, level = _style_to_element(style)
                out.append((el_type, level, text))
            elif tag == "tbl":
                table = Table(child, document)
                rows_text = [
                    " | ".join(cell.text.strip() for cell in row.cells) for row in table.rows
                ]
                text = "\n".join(r for r in rows_text if r.strip())
                if text:
                    out.append((ElementType.TABLE, None, text))
        return out

    def _read_with_zip(self, data: bytes) -> list[tuple[ElementType, int | None, str]]:
        try:
            with ZipFile(BytesIO(data)) as archive:
                document_xml = archive.read("word/document.xml")
        except (BadZipFile, KeyError) as exc:
            raise DocumentParseError("Invalid DOCX file") from exc
        try:
            root = ET.fromstring(document_xml)
        except ET.ParseError as exc:
            raise DocumentParseError("Malformed DOCX document XML") from exc

        body = root.find(f"{{{_W}}}body")
        out: list[tuple[ElementType, int | None, str]] = []
        if body is None:
            return out
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
                if not text:
                    continue
                el_type, level = _style_to_element(style)
                out.append((el_type, level, text))
            elif tag == "tbl":
                row_texts: list[str] = []
                for tr in child.findall(f"{{{_W}}}tr"):
                    cells = []
                    for tc in tr.findall(f"{{{_W}}}tc"):
                        cells.append("".join(t.text or "" for t in tc.iter(f"{{{_W}}}t")).strip())
                    row_texts.append(" | ".join(cells))
                text = "\n".join(r for r in row_texts if r.strip())
                if text:
                    out.append((ElementType.TABLE, None, text))
        return out

    def _build(
        self, source: DocumentSource, rows: list[tuple[ElementType, int | None, str]]
    ) -> ParsedDocument:
        doc_id = source.document_id
        section_stack: list[tuple[int, str]] = []
        elements: list[ParsedElement] = []
        doc_title: str | None = None
        full_text: list[str] = []

        for ordinal, (el_type, level, text) in enumerate(rows):
            full_text.append(text)
            if el_type in (ElementType.HEADING, ElementType.TITLE):
                lvl = level or 1
                while section_stack and section_stack[-1][0] >= lvl:
                    section_stack.pop()
                section_stack.append((lvl, text))
                if doc_title is None and el_type == ElementType.TITLE:
                    doc_title = text
            path = tuple(h for _lvl, h in section_stack)
            elements.append(
                ParsedElement(
                    element_id=f"{doc_id}#e{ordinal}",
                    ordinal=ordinal,
                    element_type=el_type,
                    text=text,
                    page_number=1,
                    section=path[-1] if path else None,
                    section_path=path,
                    metadata={"heading_level": level} if level is not None else {},
                )
            )

        title = doc_title or str(source.metadata.get("title") or "") or _stem(source.filename)
        page = ParsedPage(
            page_number=1,
            elements=tuple(elements),
            title=title,
            page_type=PageType.NARRATIVE,
            text="\n\n".join(full_text),
        )
        return ParsedDocument(
            document_id=doc_id,
            title=title,
            source_type="docx",
            parser_name=self.name,
            parser_version=self.version,
            page_count=None,
            pages=(page,),
        )


def _stem(filename: str) -> str:
    base = (filename or "document").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] or base
