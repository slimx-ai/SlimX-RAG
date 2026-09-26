"""Native plain-text parser: paragraph/field-aware via the shared structure rules.

Acts as the registry catch-all. It reuses :func:`structure_block` so a pasted fact sheet
keeps its ``LABEL: value`` blocks coherent, while ordinary prose becomes paragraphs.
"""

from __future__ import annotations

from ..model import DocumentSource, ParsedDocument, ParsedPage
from ..structure import decode_text, detect_source_type, looks_binary, structure_block

PARSER_NAME = "native-text"
PARSER_VERSION = "1"


def _as_text(source: DocumentSource) -> str:
    content = source.content
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        return decode_text(content)
    return ""


class TextParser:
    name = PARSER_NAME
    version = PARSER_VERSION

    def supports(self, source: DocumentSource) -> bool:
        # Catch-all for genuine text only: markup the service has no parser for (HTML) and
        # binary payloads (images, spreadsheets, archives) must fail closed as unsupported so a
        # host can fall back to its own extraction instead of indexing garbage as text.
        if detect_source_type(source.filename, source.mime_type) == "html":
            return False
        return not looks_binary(source.content)

    def parse(self, source: DocumentSource) -> ParsedDocument:
        text = _as_text(source)
        doc_id = source.document_id
        elements, page_title, page_type = structure_block(text, id_prefix=f"{doc_id}#p1", page_number=1)
        # The caller's title is the document's product identity; the inferred first line stays
        # the page/entry title used for grouping.
        title = str(source.metadata.get("title") or "") or page_title or _stem(source.filename)
        page = ParsedPage(
            page_number=1,
            elements=tuple(elements),
            # The entry (grouping/identity) title is the document's own first line; the caller's
            # title stays the document's product identity (citations, listings).
            title=page_title or title,
            page_type=page_type,
            text=text,
            inferred_title=page_title,
        )
        return ParsedDocument(
            document_id=doc_id,
            title=title,
            own_title=page_title if page_title and page_title != title else None,
            source_type=source.source_type or "text",
            parser_name=self.name,
            parser_version=self.version,
            page_count=None,
            pages=(page,),
        )


def _stem(filename: str) -> str:
    base = (filename or "document").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] or base
