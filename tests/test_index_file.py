"""End-to-end file indexing + hybrid retrieval through the HTTP server (hash embedder).

Exercises the real pipeline — parse -> structured chunk -> embed -> index -> hybrid
retrieve -> parent grouping -> page/section-aware result — with deterministic offline
embeddings. PDF page preservation is covered at the parser level (test_document_parsers).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from slimx_rag.chunk import HeuristicTokenCounter
from slimx_rag.server.app import app

_GALLERY_MD = """# Kimi K2.6

## Architecture
SCALE: 1T total, 32B active
ATTENTION: 61-MLA

## Key detail
Focuses on coding-agent workflows; uses the same text architecture as Kimi K2.5.

## Sources
Internal model card; vendor documentation.
"""


@pytest.fixture
def file_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("RAG_INDEX_PATH", str(out / "index.jsonl"))
    monkeypatch.setenv("RAG_STATE_PATH", str(out / "index_state.json"))
    monkeypatch.setenv("RAG_EMBED_DIM", "32")
    monkeypatch.delenv("DEMO_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("RAG_BACKEND_CONFIG", raising=False)
    return TestClient(app)


def _index_md(client: TestClient, *, ws: str = "ws1", doc: str = "gallery") -> dict:
    res = client.post(
        "/api/index/file",
        data={"workspace_id": ws, "document_id": doc, "filename": "gallery.md", "mime_type": "text/markdown"},
        files={"file": ("gallery.md", _GALLERY_MD, "text/markdown")},
    )
    assert res.status_code == 200, res.text
    return res.json()


def test_index_file_returns_structured_diagnostics(file_client: TestClient) -> None:
    configured = file_client.get("/api/config").json()
    assert configured["index_signature_source"] == "configured_partial"
    assert configured["index_signature"]["signature_complete"] is False
    body = _index_md(file_client)
    assert body["status"] == "ready"
    assert body["parser"] == "native-markdown"
    assert body["source_type"] == "markdown"
    assert body["chunk_count"] >= 1
    assert body["parent_count"] >= 2  # sections became parents
    assert body["embedding_provider"] == "hash"
    assert body["embedding_dim"] == 32
    assert body["index_signature"]["embedding_dimension"] == 32
    assert body["index_signature"]["parser_config_version"] == "parser-registry-v1"
    assert body["index_signature"]["file_chunk_effective_max_tokens"] == 256
    assert body["index_signature"]["file_chunk_token_counter_version"] == "heuristic-counter-v1"
    assert body["document_pipeline"] == {
        "ingest_mode": "file",
        "chunker": "structured-token",
        "chunk_config_fingerprint": body["index_signature"]["file_chunk_config_fingerprint"],
        "parser": {
            "name": "native-markdown",
            "version": body["parser_version"],
            "extraction_backend": "builtin",
            "extraction_backend_version": body["parser_version"],
        },
        "source_type": "markdown",
    }
    assert body["index_signature_source"] == "persisted_build"
    assert body["index_signature"]["signature_complete"] is True
    persisted = file_client.get("/api/config").json()
    assert persisted["index_signature_source"] == "persisted_build"
    assert (
        persisted["index_signature"]["compatibility_fingerprint"]
        == body["index_signature"]["compatibility_fingerprint"]
    )
    assert body["lexical_retrieval"] is True
    assert set(body["timings_ms"]) == {"parse_ms", "chunk_ms", "embed_ms", "index_ms"}


def test_hybrid_retrieve_is_page_section_aware(file_client: TestClient) -> None:
    _index_md(file_client)
    res = file_client.post("/api/retrieve", json={"question": "What is the key detail for Kimi K2.6?", "top_k": 5})
    assert res.status_code == 200
    body = res.json()
    assert body["retrieval_strategy"] == "hybrid"  # not a false hybrid claim
    assert body["chunks"]
    top = body["chunks"][0]
    # Rich, inspectable trace fields are present.
    assert top["metadata"]["fusion_rank"] is not None
    assert "dense_rank" in top["metadata"] and "lexical_rank" in top["metadata"]
    # The key-detail section is retrieved and the entity is matched + cited.
    sections = {c["metadata"].get("section") for c in body["chunks"]}
    assert "Key detail" in sections
    assert any(c["metadata"].get("exact_match") for c in body["chunks"])
    assert any("Kimi K2.6" in (c["citation"] or "") for c in body["chunks"])


def test_file_retrieval_scoped_by_workspace(file_client: TestClient) -> None:
    _index_md(file_client, ws="wsA", doc="d1")
    scoped_other = file_client.post(
        "/api/retrieve", json={"question": "key detail", "top_k": 5, "workspace_id": "wsB"}
    ).json()
    assert scoped_other["chunks"] == []  # other workspace sees nothing
    scoped_self = file_client.post(
        "/api/retrieve", json={"question": "key detail", "top_k": 5, "workspace_id": "wsA"}
    ).json()
    assert scoped_self["chunks"]


def test_index_file_rejects_empty(file_client: TestClient) -> None:
    res = file_client.post(
        "/api/index/file",
        data={"workspace_id": "ws1", "document_id": "empty"},
        files={"file": ("empty.md", b"", "text/markdown")},
    )
    assert res.status_code == 422


def test_index_file_signature_prefers_emitted_vector_dimension(
    file_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    appmod = importlib.import_module("slimx_rag.server.app")

    class _DeclaredNineEmitsThree:
        dim = 9
        max_seq_length = 128

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0, 0.0] for _text in texts]

        def token_counter(self) -> HeuristicTokenCounter:
            return HeuristicTokenCounter(max_tokens=128)

    fake = _DeclaredNineEmitsThree()
    monkeypatch.setenv("RAG_EMBED_PROVIDER", "hf")
    monkeypatch.setattr(appmod, "get_cached_embedder", lambda _settings: fake)
    monkeypatch.setattr(appmod, "make_token_counter", lambda _settings: fake.token_counter())

    body = _index_md(file_client, doc="actual-dimension")

    assert body["embedding_dim"] == 9  # legacy declared-dimension field
    assert body["index_signature"]["embedding_dimension"] == 3
    assert file_client.get("/api/config").json()["index_signature"]["embedding_dimension"] == 3
