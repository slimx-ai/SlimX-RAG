"""Measured retrieval-quality corrections from the ControlRoom qualification (2026-09-25).

Each test pins one semantic that the qualification benchmark showed was broken or missing:
the requested top_k is the parent budget (RAG-AUD-007), heading-only parents emit no chunk
(RAG-AUD-008), code identifiers stay whole (RAG-AUD-016), flat-text documents contribute more
than one passage and carry parent identity (RAG-AUD-001), the lexical budget is spent inside
the scope (RAG-AUD-006), and equal fused scores are broken by the dense rank.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from slimx_rag.chunk import HeuristicTokenCounter, chunk_parsed_document
from slimx_rag.document import DocumentSource, parse_document
from slimx_rag.retrieval.hybrid import ChunkRecord, HybridRetriever
from slimx_rag.retrieval.lexical import Bm25Index
from slimx_rag.retrieval.tokenize import lexical_tokens, query_identifiers
from slimx_rag.settings import RetrievalSettings, StructuredChunkSettings

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


def _index_md(client: TestClient, ws: str, doc: str, body: str) -> None:
    res = client.post(
        "/api/index/file",
        data={"workspace_id": ws, "document_id": doc, "filename": f"{doc}.md", "mime_type": "text/markdown"},
        files={"file": (f"{doc}.md", body, "text/markdown")},
    )
    assert res.status_code == 200, res.text


def _retrieve(client: TestClient, question: str, **body: object) -> dict:
    res = client.post("/api/retrieve", json={"question": question, **body})
    assert res.status_code == 200, res.text
    return res.json()


# --- RAG-AUD-007 ------------------------------------------------------------------------------


def test_requested_top_k_is_the_parent_budget(client: TestClient) -> None:
    for i in range(10):
        _index_md(client, "w", f"doc{i}", f"# Note {i}\n\nThe heron count for pond {i} was {i * 3} birds today.")
    body = _retrieve(client, "heron count for the pond", workspace_id="w", top_k=8)
    assert len(body["chunks"]) == 8  # previously capped at final_parents=6
    assert body["trace"]["final_parents"] == 8
    body = _retrieve(client, "heron count for the pond", workspace_id="w", top_k=3)
    assert len(body["chunks"]) == 3


# --- RAG-AUD-008 ------------------------------------------------------------------------------


def _chunk_md(text: str):
    parsed = parse_document(DocumentSource(document_id="d", filename="d.md", content=text))
    return chunk_parsed_document(parsed, settings=StructuredChunkSettings(), token_counter=HeuristicTokenCounter())


def test_heading_only_parents_emit_no_chunk_when_the_document_has_body() -> None:
    chunks = _chunk_md("# Supply Agreement HR-1\n\n## Term\nExpires 2026-07-31.\n\n## Parties\nBuyer and seller.")
    texts = [c.display_text for c in chunks]
    assert "Supply Agreement HR-1" not in texts  # no content-less title chunk
    assert len(chunks) == 2 and all(c.section_path[0] == "Supply Agreement HR-1" for c in chunks)
    assert {c.parent_id for c in chunks} == {"d#p1#g1", "d#p1#g2"}  # group ordinals are preserved
    only_heading = _chunk_md("# Just a title")
    assert len(only_heading) == 1 and only_heading[0].display_text == "Just a title"


# --- RAG-AUD-016 ------------------------------------------------------------------------------


def test_underscore_identifiers_stay_whole() -> None:
    assert lexical_tokens("MAX_PAYLOAD_KG = 120") == ["max_payload_kg", "120"]
    assert query_identifiers("What is MAX_PAYLOAD_KG?") == {"max_payload_kg"}
    assert lexical_tokens("snake_case_name and K2.6 and 2026-04-20") == [
        "snake_case_name",
        "and",
        "k2.6",
        "and",
        "2026-04-20",
    ]
    assert lexical_tokens("_leading and trailing_") == ["leading", "and", "trailing"]


# --- RAG-AUD-001 ------------------------------------------------------------------------------


def test_text_documents_contribute_several_passages_with_parent_identity(client: TestClient) -> None:
    paragraphs = [
        f"Paragraph {i}: the pelican colony on island {i} counted {i * 7} nests this spring." for i in range(12)
    ]
    text = "\n\n".join(" ".join([p] * 6) for p in paragraphs)  # ~6 KB -> many recursive chunks
    res = client.post(
        "/api/index",
        json={"workspace_id": "w", "document_id": "notes", "text": text, "metadata": {"title": "Pelican Survey"}},
    )
    assert res.status_code == 200, res.text
    assert res.json()["chunk_count"] > 3
    body = _retrieve(client, "pelican colony nests", workspace_id="w", document_ids=["notes"], top_k=8)
    chunks = body["chunks"]
    assert len(chunks) > 1  # previously exactly one: every chunk shared the whole-document parent
    parents = {c["metadata"]["parent_id"] for c in chunks}
    assert len(parents) == len(chunks)  # every passage is its own parent (structured pipeline)
    for chunk in chunks:
        assert not chunk["text"].startswith("Document:")  # readers get the clean text
        assert chunk["metadata"]["source_title"] == "Pelican Survey"
        assert chunk["citation"] == "[Pelican Survey]"


# --- RAG-AUD-006 ------------------------------------------------------------------------------


def test_lexical_budget_is_spent_inside_the_scope(client: TestClient) -> None:
    for i in range(40):
        client.post(
            "/api/index",
            json={"workspace_id": "loud", "document_id": f"n{i}", "text": f"vault code note {i} about the vault"},
        )
    client.post(
        "/api/index", json={"workspace_id": "quiet", "document_id": "only", "text": "The quiet workspace vault code."}
    )
    body = _retrieve(client, "vault code", workspace_id="quiet")
    assert [c["metadata"]["workspace_id"] for c in body["chunks"]] == ["quiet"]
    assert body["trace"]["lexical_candidates"] == 1  # in-scope only; previously 30 from the other tenant
    assert body["chunks"][0]["metadata"]["lexical_rank"] == 0
    unscoped = _retrieve(client, "vault code")
    assert unscoped["trace"]["lexical_candidates"] == 30


def test_bm25_allow_filters_before_the_cut() -> None:
    index = Bm25Index().build([(f"c{i}", "vault code" if i % 2 else "other words") for i in range(10)])
    hits = index.search("vault", top_k=3, allow=lambda cid: cid in {"c1", "c9"})
    assert [cid for cid, _s in hits] == ["c1", "c9"]


# --- dense-first tie-break --------------------------------------------------------------------


def test_equal_fused_scores_are_broken_by_dense_rank() -> None:
    records = {
        "zz-lexical-first": ChunkRecord(chunk_id="zz-lexical-first", text="parties", parent_id="p1"),
        "aa-dense-first": ChunkRecord(chunk_id="aa-dense-first", text="term", parent_id="p2"),
    }

    class Lex:
        def search(self, query, *, top_k, allow=None):
            return [("zz-lexical-first", 2.0), ("aa-dense-first", 1.0)]

        def __len__(self):
            return 2

    retriever = HybridRetriever(
        dense_search=lambda q, k: [("aa-dense-first", 0.9), ("zz-lexical-first", 0.8)],
        get_record=records.get,
        lexical=Lex(),
    )
    results, trace = retriever.retrieve("q", settings=RetrievalSettings())
    assert [r.chunk_id for r in results] == ["aa-dense-first", "zz-lexical-first"]
    assert results[0].fusion_score == results[1].fusion_score
    assert trace["strategy"] == "hybrid"
