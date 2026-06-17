from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from slimx_rag.cli import main
from slimx_rag.server.app import app


@pytest.fixture(scope="module")
def built_index(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a tiny local index once for all server tests."""
    base = tmp_path_factory.mktemp("server_kb")
    kb = base / "kb"
    out = base / "out"
    kb.mkdir()
    (kb / "overview.md").write_text(
        "SlimX builds explicit inspectable research AI systems.", encoding="utf-8"
    )
    assert main(["run", "--kb-dir", str(kb), "--out-dir", str(out), "--embed-dim", "16"]) == 0
    return out


@pytest.fixture
def client(built_index: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("RAG_INDEX_PATH", str(built_index / "index.jsonl"))
    monkeypatch.setenv("RAG_STATE_PATH", str(built_index / "index_state.json"))
    monkeypatch.setenv("RAG_EMBED_DIM", "16")
    monkeypatch.delenv("DEMO_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("RAG_BACKEND_CONFIG", raising=False)
    return TestClient(app)


def test_health_returns_config_summary(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["index_backend"] == "local"
    assert body["embed_provider"] == "hash"


def test_config_endpoint(client: TestClient) -> None:
    res = client.get("/api/config")
    assert res.status_code == 200
    body = res.json()
    assert body["index"]["backend"] == "local"
    assert body["embed"]["provider"] == "hash"


def test_retrieve_returns_citations(client: TestClient) -> None:
    res = client.post("/api/retrieve", json={"question": "What does SlimX build?", "top_k": 1})
    assert res.status_code == 200
    body = res.json()
    assert body["chunks"]
    assert body["chunks"][0]["citation"]


def test_ask_returns_grounded_answer(client: TestClient) -> None:
    res = client.post("/api/ask", json={"question": "What does SlimX build?", "top_k": 1})
    assert res.status_code == 200
    body = res.json()
    assert body["citations"]
    assert body["answer"]
    assert body["model_trace"]["provider"] == "fake"


def test_invalid_top_k_is_rejected_with_422(client: TestClient) -> None:
    for top_k in (0, -1):
        res = client.post("/api/retrieve", json={"question": "x", "top_k": top_k})
        assert res.status_code == 422


def test_auth_token_enforced_when_configured(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_AUTH_TOKEN", "secret")

    assert client.get("/api/config").status_code == 401
    assert client.get("/api/config", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/api/ask", json={"question": "x"}).status_code == 401

    ok = client.get("/api/config", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
    # /health stays open for liveness probes
    assert client.get("/health").status_code == 200


def test_bad_backend_config_env_returns_clean_500(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_BACKEND_CONFIG", "{not json")
    res = client.post("/api/retrieve", json={"question": "x"})
    assert res.status_code == 500
    assert "Invalid RAG_BACKEND_CONFIG" in res.json()["detail"]

    monkeypatch.setenv("RAG_BACKEND_CONFIG", json.dumps(["not", "an", "object"]))
    res = client.post("/api/retrieve", json={"question": "x"})
    assert res.status_code == 500
    assert "must be a JSON object" in res.json()["detail"]


def test_root_serves_demo_ui(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "<html" in res.text.lower()


# --- HTTP ingest (/api/index) -----------------------------------------------------
@pytest.fixture
def ingest_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client with its own empty index so ingest writes don't touch the shared one."""
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("RAG_INDEX_PATH", str(out / "index.jsonl"))
    monkeypatch.setenv("RAG_STATE_PATH", str(out / "index_state.json"))
    monkeypatch.setenv("RAG_EMBED_DIM", "16")
    monkeypatch.delenv("DEMO_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("RAG_BACKEND_CONFIG", raising=False)
    return TestClient(app)


def _texts(retrieve_body: dict) -> str:
    return " ".join(c["text"] for c in retrieve_body.get("chunks", []))


def test_index_ingests_and_is_retrievable(ingest_client: TestClient) -> None:
    res = ingest_client.post(
        "/api/index",
        json={"workspace_id": "ws1", "document_id": "doc1", "text": "alpha beta gamma delta epsilon."},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ready"
    assert body["chunk_count"] >= 1
    assert body["doc_id"]

    # New chunks are visible to retrieve immediately.
    r = ingest_client.post("/api/retrieve", json={"question": "beta gamma", "top_k": 5})
    assert r.status_code == 200
    assert "beta" in _texts(r.json())


def test_index_is_idempotent_for_same_document(ingest_client: TestClient) -> None:
    payload = {"workspace_id": "ws1", "document_id": "doc1", "text": "alpha beta gamma delta."}
    first = ingest_client.post("/api/index", json=payload).json()
    index_bytes = Path(os.environ["RAG_INDEX_PATH"]).read_bytes()

    second = ingest_client.post("/api/index", json=payload).json()
    assert second["doc_id"] == first["doc_id"]
    # Same content -> deterministic chunk ids -> identical index file.
    assert Path(os.environ["RAG_INDEX_PATH"]).read_bytes() == index_bytes


def test_index_changed_content_replaces_and_isolates_other_docs(ingest_client: TestClient) -> None:
    ingest_client.post("/api/index", json={"workspace_id": "ws1", "document_id": "docA", "text": "alpha alpha alpha."})
    ingest_client.post("/api/index", json={"workspace_id": "ws1", "document_id": "docB", "text": "beta beta beta."})
    # Re-index docA with new content.
    ingest_client.post("/api/index", json={"workspace_id": "ws1", "document_id": "docA", "text": "omega omega omega."})

    all_chunks = _texts(ingest_client.post("/api/retrieve", json={"question": "x", "top_k": 50}).json())
    assert "omega" in all_chunks   # docA new content present
    assert "beta" in all_chunks    # docB untouched
    assert "alpha" not in all_chunks  # docA old content replaced
