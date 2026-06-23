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
    # device defaults to None (auto-select) when RAG_EMBED_DEVICE is unset
    assert body["embed_device"] is None


def test_ready_reports_deep_readiness(client: TestClient) -> None:
    res = client.get("/ready")
    assert res.status_code == 200
    body = res.json()
    assert body["ready"] is True
    assert body["index_backend"] == "local"
    assert body["embed_dim"] == 16  # matches the built index's embedder


def test_ready_flags_embedding_dim_mismatch(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # The built index stored dim 16; configuring a different embedder dim must be caught as
    # not-ready (querying a dim-16 index with a dim-8 embedder would silently misbehave).
    monkeypatch.setenv("RAG_EMBED_DIM", "8")
    res = client.get("/ready")
    assert res.status_code == 503
    body = res.json()
    assert body["ready"] is False
    assert body["reason"] == "embedding_dim_mismatch"
    assert body["stored_dim"] == 16 and body["embed_dim"] == 8


def test_ready_reports_backend_failure_as_not_ready(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    appmod = importlib.import_module("slimx_rag.server.app")

    def boom() -> object:
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(appmod, "_current_backend", boom)
    res = client.get("/ready")
    assert res.status_code == 503
    assert res.json()["reason"] == "backend_load_failed"


def test_config_endpoint(client: TestClient) -> None:
    res = client.get("/api/config")
    assert res.status_code == 200
    body = res.json()
    assert body["index"]["backend"] == "local"
    assert body["embed"]["provider"] == "hash"
    assert body["embed"]["device"] is None


def test_health_reports_configured_embed_device(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_EMBED_DEVICE", "cuda")
    assert client.get("/health").json()["embed_device"] == "cuda"


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


def _ws_set(body: dict) -> set:
    return {c["metadata"].get("workspace_id") for c in body.get("chunks", [])}


def test_retrieve_scopes_by_workspace_and_document(ingest_client: TestClient) -> None:
    ingest_client.post("/api/index", json={"workspace_id": "wsA", "document_id": "dA", "text": "shared keyword apple"})
    ingest_client.post("/api/index", json={"workspace_id": "wsB", "document_id": "dB", "text": "shared keyword banana"})

    # Unscoped: both workspaces' chunks are candidates.
    everything = ingest_client.post("/api/retrieve", json={"question": "shared keyword", "top_k": 10}).json()
    assert {"wsA", "wsB"} <= _ws_set(everything)

    # Scoped to wsA: only wsA chunks come back.
    scoped = ingest_client.post(
        "/api/retrieve", json={"question": "shared keyword", "top_k": 10, "workspace_id": "wsA"}
    ).json()
    assert scoped["chunks"]
    assert _ws_set(scoped) == {"wsA"}

    # Further scoped to a non-matching document within the workspace: empty.
    none_match = ingest_client.post(
        "/api/retrieve",
        json={"question": "shared keyword", "top_k": 10, "workspace_id": "wsA", "document_ids": ["dZ"]},
    ).json()
    assert none_match["chunks"] == []


# --- chunk listing (/api/documents/{id}/chunks) -----------------------------------
def test_document_chunks_lists_indexed_chunks_in_order(ingest_client: TestClient) -> None:
    text = "alpha beta gamma. " * 80  # long enough to split into several chunks
    ingest = ingest_client.post(
        "/api/index", json={"workspace_id": "ws1", "document_id": "doc1", "text": text}
    ).json()

    res = ingest_client.get("/api/documents/doc1/chunks", params={"workspace_id": "ws1"})
    assert res.status_code == 200
    body = res.json()
    # One entry per indexed chunk, contiguous ordinals starting at 0, text present.
    assert body["chunk_count"] == ingest["chunk_count"]
    assert [c["ordinal"] for c in body["chunks"]] == list(range(body["chunk_count"]))
    assert all(c["text"] for c in body["chunks"])
    assert all(c["chunk_id"] for c in body["chunks"])


def test_document_chunks_unknown_document_is_empty_not_404(ingest_client: TestClient) -> None:
    res = ingest_client.get("/api/documents/missing/chunks", params={"workspace_id": "ws1"})
    assert res.status_code == 200
    assert res.json() == {
        "document_id": "missing",
        "doc_id": res.json()["doc_id"],
        "chunk_count": 0,
        "chunks": [],
    }


def test_document_chunks_scoped_by_workspace(ingest_client: TestClient) -> None:
    ingest_client.post(
        "/api/index", json={"workspace_id": "wsA", "document_id": "dA", "text": "alpha beta gamma delta."}
    )
    # Same document_id under a different workspace resolves to a different doc identity,
    # so listing it under the wrong workspace returns nothing.
    wrong_ws = ingest_client.get("/api/documents/dA/chunks", params={"workspace_id": "wsB"})
    assert wrong_ws.json()["chunk_count"] == 0
    right_ws = ingest_client.get("/api/documents/dA/chunks", params={"workspace_id": "wsA"})
    assert right_ws.json()["chunk_count"] >= 1


# --- index caching + scoping guard ------------------------------------------------
def test_retrieve_reuses_cached_index_until_file_changes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    import slimx_rag.index.local as local_mod

    appmod = importlib.import_module("slimx_rag.server.app")
    appmod._reset_index_cache()
    loads = {"n": 0}
    orig_load = local_mod.LocalJsonlIndexBackend.load

    def counting_load(self: local_mod.LocalJsonlIndexBackend) -> None:
        loads["n"] += 1
        orig_load(self)

    monkeypatch.setattr(local_mod.LocalJsonlIndexBackend, "load", counting_load)

    client.post("/api/retrieve", json={"question": "build", "top_k": 1})
    client.post("/api/retrieve", json={"question": "build", "top_k": 1})
    assert loads["n"] == 1  # second retrieve reuses the hot backend, no reload

    # Simulate an out-of-band rebuild by bumping the index file's mtime.
    p = Path(os.environ["RAG_INDEX_PATH"])
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    client.post("/api/retrieve", json={"question": "build", "top_k": 1})
    assert loads["n"] == 2  # a changed file triggers exactly one reload


def test_scoped_retrieve_on_unsupported_backend_returns_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    appmod = importlib.import_module("slimx_rag.server.app")

    monkeypatch.setenv("RAG_INDEX_BACKEND", "qdrant")

    class _FakeRemoteBackend:
        supports_inmemory_scope_filter = False

        def __len__(self) -> int:
            return 0

        def query(self, vector: list[float], *, top_k: int | None = None) -> list:
            return []

    monkeypatch.setattr(appmod, "_current_backend", lambda: _FakeRemoteBackend())

    scoped = client.post("/api/retrieve", json={"question": "x", "workspace_id": "wsA"})
    assert scoped.status_code == 400
    detail = scoped.json()["detail"].lower()
    assert "scoping" in detail and "qdrant" in detail

    # Unscoped retrieval still works against the same backend (guard not triggered).
    assert client.post("/api/retrieve", json={"question": "x"}).status_code == 200


# --- admin: set embedding + reset index -------------------------------------------
def test_set_embedding_resets_index_and_persists(ingest_client: TestClient) -> None:
    ingest_client.post(
        "/api/index", json={"workspace_id": "ws1", "document_id": "d1", "text": "alpha beta gamma"}
    )
    assert "alpha" in _texts(
        ingest_client.post("/api/retrieve", json={"question": "alpha", "top_k": 5}).json()
    )

    res = ingest_client.post(
        "/api/admin/embedding", json={"provider": "hash", "dim": 16, "device": "cuda"}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["index_reset"] is True
    assert body["embed"]["device"] == "cuda"

    # The choice persists (read back via /api/config) and the index was discarded: a fresh
    # ingest rebuilds from scratch, so the old document is gone.
    assert ingest_client.get("/api/config").json()["embed"]["device"] == "cuda"
    ingest_client.post(
        "/api/index", json={"workspace_id": "ws1", "document_id": "d2", "text": "omega omega"}
    )
    after = _texts(ingest_client.post("/api/retrieve", json={"question": "x", "top_k": 50}).json())
    assert "omega" in after and "alpha" not in after


def test_set_embedding_requires_token_when_configured(
    ingest_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEMO_AUTH_TOKEN", "secret")
    assert ingest_client.post("/api/admin/embedding", json={"device": "cuda"}).status_code == 401
    ok = ingest_client.post(
        "/api/admin/embedding", json={"device": "cuda"}, headers={"Authorization": "Bearer secret"}
    )
    assert ok.status_code == 200


# --- document deletion (DELETE /api/documents/{id}) -------------------------------
def test_delete_document_removes_chunks_from_index(ingest_client: TestClient) -> None:
    ingest_client.post(
        "/api/index", json={"workspace_id": "ws1", "document_id": "docA", "text": "alpha alpha alpha."}
    )
    ingest_client.post(
        "/api/index", json={"workspace_id": "ws1", "document_id": "docB", "text": "beta beta beta."}
    )

    res = ingest_client.delete("/api/documents/docA", params={"workspace_id": "ws1"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "deleted"
    assert body["deleted_chunks"] >= 1

    # docA's content is gone from retrieval; docB is untouched.
    remaining = _texts(ingest_client.post("/api/retrieve", json={"question": "x", "top_k": 50}).json())
    assert "alpha" not in remaining
    assert "beta" in remaining
    # Its chunk listing is now empty too (state entry forgotten, not just vectors dropped).
    after = ingest_client.get("/api/documents/docA/chunks", params={"workspace_id": "ws1"})
    assert after.json()["chunk_count"] == 0


def test_delete_unknown_document_is_noop(ingest_client: TestClient) -> None:
    res = ingest_client.delete("/api/documents/missing", params={"workspace_id": "ws1"})
    assert res.status_code == 200
    assert res.json()["deleted_chunks"] == 0


def test_delete_document_is_workspace_scoped(ingest_client: TestClient) -> None:
    ingest_client.post(
        "/api/index", json={"workspace_id": "wsA", "document_id": "dA", "text": "alpha beta gamma."}
    )
    # Same document_id under a different workspace is a different doc identity -> no-op.
    wrong = ingest_client.delete("/api/documents/dA", params={"workspace_id": "wsB"})
    assert wrong.json()["deleted_chunks"] == 0
    assert ingest_client.get("/api/documents/dA/chunks", params={"workspace_id": "wsA"}).json()["chunk_count"] >= 1
    # Deleting under the right workspace removes it.
    right = ingest_client.delete("/api/documents/dA", params={"workspace_id": "wsA"})
    assert right.json()["deleted_chunks"] >= 1
    assert ingest_client.get("/api/documents/dA/chunks", params={"workspace_id": "wsA"}).json()["chunk_count"] == 0


def test_delete_document_is_idempotent(ingest_client: TestClient) -> None:
    ingest_client.post(
        "/api/index", json={"workspace_id": "ws1", "document_id": "doc1", "text": "alpha beta."}
    )
    first = ingest_client.delete("/api/documents/doc1", params={"workspace_id": "ws1"})
    assert first.json()["deleted_chunks"] >= 1
    second = ingest_client.delete("/api/documents/doc1", params={"workspace_id": "ws1"})
    assert second.status_code == 200
    assert second.json()["deleted_chunks"] == 0
