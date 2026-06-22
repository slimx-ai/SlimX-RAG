"""Native source-code parser: block-aware splitting, no heavy parser framework.

Blocks are separated on blank lines and at top-level definition boundaries
(``def``/``class``/``function``/``export``/``func``/``fn``) so a function or class stays
together as one element. The language is inferred from the file extension only — never
guessed unsafely.
"""

from __future__ import annotations

import re

from ..model import DocumentSource, ElementType, PageType, ParsedDocument, ParsedElement, ParsedPage

PARSER_NAME = "native-code"
PARSER_VERSION = "1"

_LANG_BY_EXT = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".rb": "ruby",
}
_DEF_START = re.compile(
    r"^(async\s+def|def|class|function|export\s+(default\s+)?(function|class|const)|"
    r"func|public|private|protected|fn)\b"
)
_CODE_EXTS = tuple(_LANG_BY_EXT)


def _as_text(source: DocumentSource) -> str:
    content = source.content
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return ""


def _language(filename: str) -> str | None:
    lower = (filename or "").lower()
    for ext, lang in _LANG_BY_EXT.items():
        if lower.endswith(ext):
            return lang
    return None


class CodeParser:
    name = PARSER_NAME
    version = PARSER_VERSION

    def supports(self, source: DocumentSource) -> bool:
        lower = (source.filename or "").lower()
        return lower.endswith(_CODE_EXTS)

    def parse(self, source: DocumentSource) -> ParsedDocument:
        text = _as_text(source)
        doc_id = source.document_id
        language = _language(source.filename)

        blocks: list[str] = []
        current: list[str] = []

        def flush() -> None:
            if current and "".join(current).strip():
                blocks.append("\n".join(current).rstrip())
            current.clear()

        for line in text.splitlines():
            top_level_def = bool(_DEF_START.match(line)) and not line[:1].isspace()
            if not line.strip():
                flush()
                continue
            if top_level_def and current:
                flush()
            current.append(line)
        flush()

        elements: list[ParsedElement] = []
        for ordinal, block in enumerate(blocks):
            elements.append(
                ParsedElement(
                    element_id=f"{doc_id}#e{ordinal}",
                    ordinal=ordinal,
                    element_type=ElementType.CODE,
                    text=block,
                    page_number=1,
                    metadata={"language": language} if language else {},
                )
            )

        title = str(source.metadata.get("title") or "") or _stem(source.filename)
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
            source_type="code",
            parser_name=self.name,
            parser_version=self.version,
            page_count=None,
            pages=(page,),
            metadata={"language": language} if language else {},
        )


def _stem(filename: str) -> str:
    base = (filename or "document").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] or base
