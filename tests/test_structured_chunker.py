"""Structure- and token-aware chunking: parents, identity, cohesion, hard token caps."""

from __future__ import annotations

import pytest

from slimx_rag.chunk import chunk_parsed_document
from slimx_rag.chunk.tokenizer import HeuristicTokenCounter
from slimx_rag.document import DocumentSource, PageType, parse_document
from slimx_rag.document.parsers import pdf as pdf_module
from slimx_rag.document.parsers.pdf import PdfParser

_PAGE_TIMELINE = "\n".join(
    ["Model Index", "Kimi K2.5 2026-01-10", "Kimi K2.6 2026-02-20", "Kimi K2.7 2026-03-30"]
)
_PAGE_FACT = "\n".join(
    [
        "Kimi K2.6",
        "SCALE",
        "1T total, 32B active",
        "CONTEXT TOKENS",
        "256,000",
        "ATTENTION",
        "61-MLA",
        "KEY DETAIL",
        "Uses the same text architecture as Kimi K2.5 with refined routing.",
    ]
)


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakeReader:
    def __init__(self, _stream: object) -> None:
        self.pages = [_FakePage(_PAGE_TIMELINE), _FakePage(_PAGE_FACT)]


def _parse(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setattr(pdf_module, "PdfReader", _FakeReader)
    src = DocumentSource(
        document_id="doc1", filename="gallery.pdf", mime_type="application/pdf", content=b"%PDF"
    )
    return PdfParser().parse(src)


def test_factsheet_page_stays_one_self_contained_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = _parse(monkeypatch)
    chunks = chunk_parsed_document(doc, token_counter=HeuristicTokenCounter(max_tokens=1000))
    fact = [c for c in chunks if c.page_number == 2]
    assert len(fact) == 1
    c = fact[0]
    assert "Document: gallery" in c.embedding_text
    assert "Page: 2" in c.embedding_text
    assert "Entry: Kimi K2.6" in c.embedding_text
    # The reported failure: KEY DETAIL label and its value must stay together.
    assert "KEY DETAIL" in c.display_text and "refined routing" in c.display_text
    assert c.page_type == PageType.FACT_SHEET
    assert c.token_count > 0


def test_no_chunk_exceeds_token_cap_and_keeps_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from slimx_rag.settings import StructuredChunkSettings

    doc = _parse(monkeypatch)
    counter = HeuristicTokenCounter(max_tokens=24)  # tiny cap forces child splitting
    settings = StructuredChunkSettings(target_tokens=24, max_tokens=24, force_split_overlap_tokens=4)
    chunks = chunk_parsed_document(doc, settings=settings, token_counter=counter)
    assert chunks
    for c in chunks:
        assert c.token_count <= 24  # never silently exceeds the model cap
        assert c.embedding_text.startswith("Document:")  # identity preserved on every child
        assert c.display_text.strip() != "KEY DETAIL"  # label never orphaned from its value


def test_field_label_value_cohesion_under_split(monkeypatch: pytest.MonkeyPatch) -> None:
    from slimx_rag.settings import StructuredChunkSettings

    doc = _parse(monkeypatch)
    counter = HeuristicTokenCounter(max_tokens=40)
    settings = StructuredChunkSettings(target_tokens=8, max_tokens=40, force_split_overlap_tokens=2)
    chunks = chunk_parsed_document(doc, settings=settings, token_counter=counter)
    key = [c for c in chunks if "KEY DETAIL" in c.display_text]
    assert key
    assert all("refined routing" in c.display_text for c in key)
    # A single-field child is cited by its label.
    assert any(c.section == "KEY DETAIL" for c in chunks)


def test_force_split_oversized_element_sets_flag_and_overlaps() -> None:
    from slimx_rag.settings import StructuredChunkSettings

    big = " ".join(f"word{i}" for i in range(200))
    doc = parse_document(DocumentSource(document_id="d", filename="big.txt", content=big))
    counter = HeuristicTokenCounter(max_tokens=30)
    settings = StructuredChunkSettings(target_tokens=30, max_tokens=30, force_split_overlap_tokens=5)
    chunks = chunk_parsed_document(doc, settings=settings, token_counter=counter)
    assert len(chunks) > 1
    assert all(c.token_count <= 30 for c in chunks)
    assert any(c.forced_split for c in chunks)
    forced = [c for c in chunks if c.forced_split]
    if len(forced) >= 2:
        assert set(forced[0].display_text.split()) & set(forced[1].display_text.split())


def test_deterministic_chunk_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = _parse(monkeypatch)
    a = chunk_parsed_document(doc, token_counter=HeuristicTokenCounter(max_tokens=1000))
    b = chunk_parsed_document(doc, token_counter=HeuristicTokenCounter(max_tokens=1000))
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)  # unique ids
