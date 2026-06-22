"""Structure- and token-aware chunking over the canonical document model.

Replaces blind character windows with structure-aware chunks:

- Parents are PDF pages (or sections, split at heading/title boundaries). A small,
  coherent fact-sheet page stays a single self-contained chunk.
- Children are created only when a parent exceeds the embedding-safe token cap, by
  packing WHOLE elements (a field's label+value is never split; tables stay isolated).
- Normal chunks use no sliding overlap; a small token overlap is used ONLY when an
  oversized, indivisible element must be force-split.
- Every child carries document/page/title/section identity in its embedding text, so a
  retrieved chunk is never an anonymous fragment. The stored embedding token count can
  never silently exceed the model's max sequence length.

The legacy ``chunk_documents`` (recursive character splitter) remains as the plain-text
fallback / evaluation baseline.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from slimx_rag.core.hashing import chunk_config_fingerprint, content_hash, make_chunk_id
from slimx_rag.document.model import (
    ElementType,
    PageType,
    ParsedDocument,
    ParsedElement,
    RetrievalChunk,
)
from slimx_rag.settings import StructuredChunkSettings

from .tokenizer import HeuristicTokenCounter, TokenCounter

_HEADING_TYPES = (ElementType.HEADING, ElementType.TITLE)


@dataclass(frozen=True, slots=True)
class _Parent:
    parent_id: str
    title: str
    section: str | None
    section_path: tuple[str, ...]
    page_number: int | None
    page_type: PageType
    elements: tuple[ParsedElement, ...]


def _iter_parents(doc: ParsedDocument) -> Iterator[_Parent]:
    """Yield parent units: each page, subdivided at heading/title boundaries."""
    paginated = doc.page_count is not None
    for page in doc.pages:
        groups: list[list[ParsedElement]] = []
        current: list[ParsedElement] = []
        for el in page.elements:
            if el.element_type in _HEADING_TYPES and current:
                groups.append(current)
                current = []
            current.append(el)
        if current:
            groups.append(current)
        for gi, group in enumerate(groups):
            head = next((e for e in group if e.element_type in _HEADING_TYPES), None)
            title = (head.text if head else None) or page.title or doc.title
            section = (head.text if head else None) or page.title
            yield _Parent(
                parent_id=f"{doc.document_id}#p{page.page_number}#g{gi}",
                title=title,
                section=section,
                section_path=group[0].section_path or (),
                page_number=page.page_number if paginated else None,
                page_type=page.page_type,
                elements=tuple(group),
            )


def _identity_prefix(
    *, source_title: str, page_number: int | None, entry: str | None, section: str | None
) -> str:
    lines = [f"Document: {source_title}"]
    if page_number is not None:
        lines.append(f"Page: {page_number}")
    if entry:
        lines.append(f"Entry: {entry}")
    if section and section != entry:
        lines.append(f"Section: {section}")
    return "\n".join(lines)


def _child_section(parent: _Parent, els: list[ParsedElement]) -> str | None:
    """A child made of one labelled FIELD is cited by that label; else the parent section."""
    if len(els) == 1 and els[0].element_type == ElementType.FIELD:
        label = els[0].metadata.get("label")
        if label:
            return str(label)
    return parent.section


def _overlap_tail_words(words: list[str], counter: TokenCounter, overlap_tokens: int) -> int:
    """How many trailing words approximate ``overlap_tokens`` tokens."""
    if overlap_tokens <= 0:
        return 0
    for back in range(1, len(words) + 1):
        if counter.count(" ".join(words[len(words) - back :])) >= overlap_tokens:
            return back
    return len(words)


def _split_text_by_tokens(
    text: str, counter: TokenCounter, max_tokens: int, overlap_tokens: int
) -> list[str]:
    """Greedily split text into <= max_tokens pieces, with a small word overlap.

    Used only for an oversized, indivisible element. Each piece is measured so it never
    exceeds the cap (no silent truncation); a single token longer than the cap is emitted
    whole (degenerate input).
    """
    words = text.split()
    if not words:
        return []
    pieces: list[str] = []
    i = 0
    n = len(words)
    while i < n:
        cur = ""
        j = i
        while j < n:
            trial = f"{cur} {words[j]}".strip() if cur else words[j]
            if cur and counter.count(trial) > max_tokens:
                break
            cur = trial
            j += 1
            if counter.count(cur) >= max_tokens:
                break
        pieces.append(cur)
        if j >= n:
            break
        back = _overlap_tail_words(words[i:j], counter, overlap_tokens)
        i = max(i + 1, j - back)
    return pieces


def chunk_parsed_document(
    doc: ParsedDocument,
    *,
    settings: StructuredChunkSettings | None = None,
    token_counter: TokenCounter | None = None,
) -> list[RetrievalChunk]:
    """Chunk a parsed document into retrieval units. Deterministic for fixed inputs."""
    settings = settings or StructuredChunkSettings()
    settings.validate()
    counter = token_counter or HeuristicTokenCounter()
    hard_cap = max(8, min(settings.max_tokens, counter.max_tokens))
    cfg_hash = chunk_config_fingerprint(
        chunk_size=settings.max_tokens,
        chunk_overlap=settings.force_split_overlap_tokens,
        separators=(
            "structured-v1",
            str(settings.target_tokens),
            str(int(settings.include_identity_prefix)),
        ),
    )
    out: list[RetrievalChunk] = []
    for parent in _iter_parents(doc):
        out.extend(_chunk_parent(doc, parent, settings, counter, hard_cap, cfg_hash))
    return out


def _chunk_parent(
    doc: ParsedDocument,
    parent: _Parent,
    settings: StructuredChunkSettings,
    counter: TokenCounter,
    hard_cap: int,
    cfg_hash: str,
) -> list[RetrievalChunk]:
    use_prefix = settings.include_identity_prefix

    def prefix_for(section: str | None) -> str:
        if not use_prefix:
            return ""
        return _identity_prefix(
            source_title=doc.title,
            page_number=parent.page_number,
            entry=parent.title,
            section=section,
        )

    # Budget so prefix + content never exceeds the hard cap. The whole-parent check uses
    # the parent-section prefix; splitting reserves the LONGEST candidate prefix (a child
    # may be cited by a field label), so every emitted child stays within the cap.
    base_prefix_tokens = counter.count(prefix_for(parent.section)) if use_prefix else 0
    candidate_sections: set[str | None] = {parent.section}
    for el in parent.elements:
        if el.element_type == ElementType.FIELD and el.metadata.get("label"):
            candidate_sections.add(str(el.metadata["label"]))
    max_prefix_tokens = (
        max(counter.count(prefix_for(s)) for s in candidate_sections) if use_prefix else 0
    )
    whole_parent_budget = max(8, hard_cap - base_prefix_tokens)
    content_budget = max(8, hard_cap - max_prefix_tokens)
    effective_target = min(settings.target_tokens, content_budget)

    chunks: list[RetrievalChunk] = []
    ordinal = 0

    def emit(display_text: str, els: list[ParsedElement], *, forced: bool) -> None:
        nonlocal ordinal
        display_text = display_text.strip()
        if not display_text:
            return
        section = _child_section(parent, els)
        prefix = prefix_for(section)
        embedding_text = f"{prefix}\n\n{display_text}" if prefix else display_text
        chunk_id = make_chunk_id(
            parent_id=parent.parent_id,
            content_hash_value=content_hash(embedding_text),
            chunk_index=ordinal,
            chunk_cfg_hash=cfg_hash,
        )
        chunks.append(
            RetrievalChunk(
                chunk_id=chunk_id,
                document_id=doc.document_id,
                parent_id=parent.parent_id,
                display_text=display_text,
                embedding_text=embedding_text,
                token_count=counter.count(embedding_text),
                page_number=parent.page_number,
                section=section,
                section_path=parent.section_path,
                page_type=parent.page_type,
                element_types=tuple(e.element_type for e in els),
                source_title=doc.title,
                ordinal=ordinal,
                forced_split=forced,
                metadata={
                    "parser": doc.parser_name,
                    "parser_version": doc.parser_version,
                    "entry": parent.title,
                    "page_type": parent.page_type.value,
                },
            )
        )
        ordinal += 1

    parent_text = "\n".join(e.text for e in parent.elements).strip()
    if not parent_text:
        return chunks

    # Whole parent fits the embedding-safe budget -> one self-contained chunk.
    if counter.count(parent_text) <= whole_parent_budget:
        emit(parent_text, list(parent.elements), forced=False)
        return chunks

    # Otherwise pack WHOLE elements (no overlap), isolating tables and force-splitting any
    # single element that alone exceeds the content budget.
    current: list[ParsedElement] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            emit("\n".join(e.text for e in current), current, forced=False)
            current = []
            current_tokens = 0

    for el in parent.elements:
        el_tokens = counter.count(el.text)
        if el_tokens > content_budget:
            flush()
            for piece in _split_text_by_tokens(
                el.text, counter, content_budget, settings.force_split_overlap_tokens
            ):
                emit(piece, [el], forced=True)
            continue
        if el.element_type == ElementType.TABLE:
            flush()
            emit(el.text, [el], forced=False)
            continue
        if current and current_tokens + el_tokens > effective_target:
            flush()
        current.append(el)
        current_tokens += el_tokens
    flush()
    return chunks
