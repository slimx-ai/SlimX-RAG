"""Heading-less fact sheets are addressed by their fields (index-shaping-v3).

A packed sheet embeds one mean-pooled vector over several unrelated fields, so a question about
one field ranks it far below documents that merely share its vocabulary. Each labelled field is
therefore its own retrieval unit (identity prefix + that field), every unit displays the whole
sheet, and the grouping stage never shows the same passage twice. A fact sheet under a heading
keeps the heading as its citation and stays one chunk.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from slimx_rag.chunk import chunk_parsed_document
from slimx_rag.chunk.tokenizer import HeuristicTokenCounter
from slimx_rag.document import DocumentSource, PageType, parse_document
from slimx_rag.document.parsers import pdf as pdf_module
from slimx_rag.document.parsers.pdf import PdfParser
from slimx_rag.retrieval.hybrid import ChunkRecord, HybridRetriever
from slimx_rag.settings import RetrievalSettings

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
        self.pages = [_FakePage(_PAGE_FACT)]


def test_headingless_factsheet_is_addressed_by_field_and_shows_the_whole_sheet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pdf_module, "PdfReader", _FakeReader)
    src = DocumentSource(document_id="doc1", filename="gallery.pdf", mime_type="application/pdf", content=b"%PDF")
    chunks = chunk_parsed_document(PdfParser().parse(src), token_counter=HeuristicTokenCounter(max_tokens=1000))

    assert [c.section for c in chunks] == ["SCALE", "CONTEXT TOKENS", "ATTENTION", "KEY DETAIL"]
    whole = chunks[0].display_text
    assert "KEY DETAIL" in whole and "refined routing" in whole and "256,000" in whole and "61-MLA" in whole
    assert all(c.display_text == whole for c in chunks)  # every unit shows the whole sheet
    assert len({c.chunk_id for c in chunks}) == 4 and len({c.parent_id for c in chunks}) == 1
    key = chunks[-1]
    assert "Document: gallery" in key.embedding_text and "Page: 1" in key.embedding_text
    assert "Entry: Kimi K2.6" in key.embedding_text and "Section: KEY DETAIL" in key.embedding_text
    assert "refined routing" in key.embedding_text
    assert "256,000" not in key.embedding_text and "61-MLA" not in key.embedding_text  # one field per unit
    assert all(c.page_type == PageType.FACT_SHEET for c in chunks)
    assert all(c.token_count <= 1000 for c in chunks)


def test_factsheet_under_a_heading_stays_one_chunk_cited_by_its_heading() -> None:
    src = DocumentSource(
        document_id="cat",
        filename="catalog.md",
        mime_type="text/markdown",
        content=(
            b"# Parts Catalog\n\n## GX-210 servo gearbox\nPart number: GX-210\nRatio: 25:1\n"
            b"Backlash: 3 arcmin\nMounting torque: 45 N-m\n"
        ),
    )
    chunks = chunk_parsed_document(parse_document(src), token_counter=HeuristicTokenCounter(max_tokens=1000))
    entry = [c for c in chunks if c.section == "GX-210 servo gearbox"]
    assert len(entry) == 1
    assert "Ratio: 25:1" in entry[0].display_text and "45 N-m" in entry[0].display_text
    assert not any(c.section in {"Part number", "Ratio", "Backlash", "Mounting torque"} for c in chunks)


def test_field_units_of_one_sheet_never_show_the_same_passage_twice() -> None:
    sheet = "Atlas Maintenance Log\nLAST SERVICE: 2026-04-01\nTECHNICIAN: Jonas Berg"
    records = {
        "last": ChunkRecord(
            chunk_id="last",
            text="Document: Atlas Maintenance Log\nSection: LAST SERVICE\n\nLAST SERVICE: 2026-04-01",
            parent_id="log",
            section="LAST SERVICE",
            source_title="Atlas Maintenance Log",
            entry="Atlas Maintenance Log",
            display_text=sheet,
        ),
        "tech": ChunkRecord(
            chunk_id="tech",
            text="Document: Atlas Maintenance Log\nSection: TECHNICIAN\n\nTECHNICIAN: Jonas Berg",
            parent_id="log",
            section="TECHNICIAN",
            source_title="Atlas Maintenance Log",
            entry="Atlas Maintenance Log",
            display_text=sheet,
        ),
        "other": ChunkRecord(
            chunk_id="other",
            text="Document: Safety Manual\nSection: Visitors\n\nVisitors need an escort.",
            parent_id="manual",
            section="Visitors",
            source_title="Safety Manual",
            entry="Safety Manual",
        ),
    }
    retriever = HybridRetriever(
        dense_search=lambda _q, _k: [("last", 0.9), ("tech", 0.8), ("other", 0.5)],
        get_record=records.get,
    )
    results, trace = retriever.retrieve("When was the gantry last serviced?", settings=RetrievalSettings())

    assert trace["strategy"] == "dense"
    assert [r.chunk_id for r in results] == ["last", "other"]  # the second field unit is not shown again
    assert [r.passage() for r in results].count(sheet) == 1
    assert results[0].citation() == "[Atlas Maintenance Log, LAST SERVICE]"
    assert results[0].passage() == sheet and results[0].parent_reason == "primary"


server = importlib.import_module("slimx_rag.server.app")


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("RAG_INDEX_PATH", str(out / "index.jsonl"))
    monkeypatch.setenv("RAG_STATE_PATH", str(out / "index_state.json"))
    monkeypatch.setenv("RAG_EMBED_DIM", "32")
    for name in ("DEMO_AUTH_TOKEN", "RAG_AUTH_TOKEN", "RAG_BACKEND_CONFIG", "RAG_INDEX_BACKEND", "RAG_FINAL_PARENTS"):
        monkeypatch.delenv(name, raising=False)
    server._reset_index_cache()
    return TestClient(server.app)


_SHEET = "Atlas Maintenance Log\n\nLAST SERVICE\n2026-04-01\n\nTECHNICIAN\nJonas Berg\n\nNOTES\nReplaced CC-88.\n"


def test_posted_text_factsheet_is_listed_per_field_and_retrieved_once(client: TestClient) -> None:
    res = client.post(
        "/api/index",
        json={
            "workspace_id": "ws",
            "document_id": "log",
            "text": _SHEET,
            "metadata": {"title": "Atlas Maintenance Log"},
        },
    )
    assert res.status_code == 200, res.text
    assert res.json()["chunk_count"] == 3

    listed = client.get("/api/documents/log/chunks", params={"workspace_id": "ws"}).json()["chunks"]
    assert [c["section"] for c in listed] == ["LAST SERVICE", "TECHNICIAN", "NOTES"]
    assert [c["ordinal"] for c in listed] == [0, 1, 2]
    assert len({c["text"] for c in listed}) == 1  # every unit shows the whole sheet
    assert "Jonas Berg" in listed[0]["text"] and "2026-04-01" in listed[0]["text"]

    data = client.post(
        "/api/retrieve",
        json={"question": "LAST SERVICE TECHNICIAN", "workspace_id": "ws", "document_ids": ["log"], "top_k": 8},
    ).json()
    passages = [c["text"] for c in data["chunks"]]
    assert len(passages) == 1  # two field units matched; the sheet is shown once
    assert data["chunks"][0]["citation"] in {
        "[Atlas Maintenance Log, LAST SERVICE]",
        "[Atlas Maintenance Log, TECHNICIAN]",
    }
    assert data["chunks"][0]["metadata"]["parent_reason"] == "primary"
