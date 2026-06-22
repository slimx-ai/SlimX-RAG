"""Native Markdown parser: preserves heading hierarchy, fenced code, lists, paragraphs."""

from __future__ import annotations

import re

from ..model import DocumentSource, ElementType, PageType, ParsedDocument, ParsedElement, ParsedPage
from ..structure import detect_source_type

PARSER_NAME = "native-markdown"
PARSER_VERSION = "1"

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+\.)\s+\S")
_FENCE = re.compile(r"^\s*(```|~~~)(.*)$")


def _as_text(source: DocumentSource) -> str:
    content = source.content
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return ""


class MarkdownParser:
    name = PARSER_NAME
    version = PARSER_VERSION

    def supports(self, source: DocumentSource) -> bool:
        return detect_source_type(source.filename, source.mime_type) == "markdown"

    def parse(self, source: DocumentSource) -> ParsedDocument:
        text = _as_text(source)
        doc_id = source.document_id
        elements: list[ParsedElement] = []
        section_stack: list[tuple[int, str]] = []  # (level, heading text)
        ordinal = 0
        doc_title: str | None = None

        def section_path() -> tuple[str, ...]:
            return tuple(h for _lvl, h in section_stack)

        def emit(el_type: ElementType, body: str, **meta: object) -> None:
            nonlocal ordinal
            body = body.rstrip()
            if not body.strip():
                return
            path = section_path()
            elements.append(
                ParsedElement(
                    element_id=f"{doc_id}#e{ordinal}",
                    ordinal=ordinal,
                    element_type=el_type,
                    text=body,
                    page_number=1,
                    section=path[-1] if path else None,
                    section_path=path,
                    metadata=dict(meta),
                )
            )
            ordinal += 1

        lines = text.splitlines()
        i, n = 0, len(lines)
        para: list[str] = []

        def flush_para() -> None:
            if para:
                emit(ElementType.PARAGRAPH, "\n".join(para))
                para.clear()

        while i < n:
            line = lines[i]
            fence = _FENCE.match(line)
            if fence:
                flush_para()
                lang = fence.group(2).strip()
                marker = fence.group(1)
                body: list[str] = []
                i += 1
                while i < n and not lines[i].strip().startswith(marker):
                    body.append(lines[i])
                    i += 1
                i += 1  # consume closing fence
                emit(ElementType.CODE, "\n".join(body), language=lang or None)
                continue

            heading = _HEADING.match(line)
            if heading:
                flush_para()
                level = len(heading.group(1))
                htext = heading.group(2).strip()
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                section_stack.append((level, htext))
                if doc_title is None and level == 1:
                    doc_title = htext
                emit(ElementType.HEADING, htext, level=level)
                i += 1
                continue

            if _LIST_ITEM.match(line):
                flush_para()
                emit(ElementType.LIST_ITEM, line.strip())
                i += 1
                continue

            if not line.strip():
                flush_para()
                i += 1
                continue

            para.append(line)
            i += 1

        flush_para()

        title = doc_title or str(source.metadata.get("title") or "") or _stem(source.filename)
        page = ParsedPage(
            page_number=1,
            elements=tuple(elements),
            title=title,
            page_type=PageType.NARRATIVE,
            text=text,
        )
        return ParsedDocument(
            document_id=doc_id,
            title=title,
            source_type="markdown",
            parser_name=self.name,
            parser_version=self.version,
            page_count=None,
            pages=(page,),
        )


def _stem(filename: str) -> str:
    base = (filename or "document").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] or base
