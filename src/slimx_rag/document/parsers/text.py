"""Native plain-text parser: paragraph/field-aware via the shared structure rules.

Acts as the registry catch-all. It reuses :func:`structure_block` so a pasted fact sheet
keeps its ``LABEL: value`` blocks coherent, while ordinary prose becomes paragraphs.
"""

from __future__ import annotations

from ..model import DocumentSource, ParsedDocument, ParsedPage
from ..structure import structure_block

PARSER_NAME = "native-text"
PARSER_VERSION = "1"


def _as_text(source: DocumentSource) -> str:
    content = source.content
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return ""


class TextParser:
    name = PARSER_NAME
    version = PARSER_VERSION

    def supports(self, source: DocumentSource) -> bool:
        return True  # catch-all; registered last

    def parse(self, source: DocumentSource) -> ParsedDocument:
        text = _as_text(source)
        doc_id = source.document_id
        elements, page_title, page_type = structure_block(
            text, id_prefix=f"{doc_id}#p1", page_number=1
        )
        title = page_title or str(source.metadata.get("title") or "") or _stem(source.filename)
        page = ParsedPage(
            page_number=1,
            elements=tuple(elements),
            title=title,
            page_type=page_type,
            text=text,
        )
        return ParsedDocument(
            document_id=doc_id,
            title=title,
            source_type=source.source_type or "text",
            parser_name=self.name,
            parser_version=self.version,
            page_count=None,
            pages=(page,),
        )


def _stem(filename: str) -> str:
    base = (filename or "document").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] or base
