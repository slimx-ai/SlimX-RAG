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
    (kb / "overview.md").write_text("SlimX builds explicit inspectable research AI systems.", encoding="utf-8")
    assert main(["run", "--kb-dir", str(kb), "--out-dir", str(out), "--embed-dim", "16"]) == 0
    return out


@pytest.fixture
def client(built_index: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("RAG_INDEX_PATH", str(built_index / "index.jsonl"))
    monkeypatch.setenv("RAG_STATE_PATH", str(built_index / "index_state.json"))
    monkeypatch.setenv("RAG_EMBED_DIM", "16")
    monkeypatch.delenv("DEMO_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("RAG_AUTH_TOKEN", raising=False)
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
    assert body["engine_version"]


def test_ready_reports_deep_readiness(client: TestClient) -> None:
    res = client.get("/ready")
    assert res.status_code == 200
    body = res.json()
    assert body["ready"] is True
    assert body["index_backend"] == "local"
    assert body["embed_dim"] == 16  # matches the built index's embedder
    assert body["index_signature"]["embedding_dimension"] == 16
    assert body["index_signature"]["index_schema_version"] == 1
    assert body["index_signature"]["index_instance_id"].startswith("idx_")
    config_signature = client.get("/api/config").json()["index_signature"]
    assert config_signature["compatibility_fingerprint"] == body["index_signature"]["compatibility_fingerprint"]


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
    # The signature describes the existing corpus, so backend evidence wins over the
    # newly configured incompatible embedder (reported separately as embed_dim=8).
    assert body["index_signature"]["embedding_dimension"] == 16


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
    assert body["index_signature"]["vector_backend"] == "local"
    assert body["index_signature"]["embedding_dimension"] == 16


def test_config_is_offline_and_does_not_initialize_backend_or_embedder(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    appmod = importlib.import_module("slimx_rag.server.app")

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("config inspection must not initialize runtime dependencies")

    monkeypatch.setattr(appmod, "get_cached_embedder", forbidden)
    monkeypatch.setattr(appmod, "make_token_counter", forbidden)
    monkeypatch.setattr(appmod, "_current_backend", forbidden)

    response = client.get("/api/config")

    assert response.status_code == 200
    assert response.json()["index_signature_source"] == "persisted_build"


def test_config_returns_persisted_receipt_when_current_settings_change(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = client.get("/api/config").json()["index_signature"]
    monkeypatch.setenv("RAG_CHUNK_SIZE", "401")

    configured = client.get("/api/config").json()

    assert configured["index_signature_source"] == "persisted_build"
    assert configured["index_signature"]["compatibility_fingerprint"] == baseline["compatibility_fingerprint"]
    assert configured["configured_index_signature"]["signature_complete"] is False
    assert (
        configured["configured_index_signature"]["compatibility_fingerprint"] != baseline["compatibility_fingerprint"]
    )
    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["reason"] == "index_signature_mismatch"
    assert ready.json()["index_signature"]["compatibility_fingerprint"] == baseline["compatibility_fingerprint"]


def test_config_snapshot_holds_instance_lease_against_concurrent_reset(
    ingest_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib
    import threading
    from concurrent.futures import ThreadPoolExecutor

    appmod = importlib.import_module("slimx_rag.server.app")
    indexed = ingest_client.post(
        "/api/index", json={"workspace_id": "ws1", "document_id": "d1", "text": "alpha"}
    ).json()
    old_instance_id = indexed["index_signature"]["index_instance_id"]
    state_read_entered = threading.Event()
    allow_state_read = threading.Event()
    preflight_complete = threading.Event()
    reset_staged = threading.Event()
    original_state_load = appmod.IndexState.load
    original_get_embedder = appmod.get_cached_embedder
    original_stage = appmod._stage_reset_artifact

    def blocking_state_load(path: Path):
        state_read_entered.set()
        if not allow_state_read.wait(timeout=5):
            raise TimeoutError("test did not release config state read")
        return original_state_load(path)

    def observed_preflight(settings: object):
        embedder = original_get_embedder(settings)
        preflight_complete.set()
        return embedder

    def observed_stage(path: Path, backup: Path) -> None:
        reset_staged.set()
        original_stage(path, backup)

    monkeypatch.setattr(appmod.IndexState, "load", staticmethod(blocking_state_load))
    monkeypatch.setattr(appmod, "get_cached_embedder", observed_preflight)
    monkeypatch.setattr(appmod, "_stage_reset_artifact", observed_stage)

    with ThreadPoolExecutor(max_workers=2) as executor:
        config_future = executor.submit(appmod.config, None)
        assert state_read_entered.wait(timeout=5)
        reset_future = executor.submit(
            appmod.set_embedding,
            appmod.EmbeddingConfigRequest(dim=8),
            None,
        )
        assert preflight_complete.wait(timeout=5)
        assert not reset_staged.wait(timeout=0.1)
        allow_state_read.set()
        config_body = config_future.result(timeout=5)
        reset_body = reset_future.result(timeout=5)

    assert config_body["index_signature"]["index_instance_id"] == old_instance_id
    assert reset_body["index_signature"]["index_instance_id"] != old_instance_id
    assert reset_staged.is_set()


def test_index_rejects_pre_reset_embedding_snapshot(ingest_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import threading
    from concurrent.futures import ThreadPoolExecutor

    appmod = importlib.import_module("slimx_rag.server.app")
    embed_started = threading.Event()
    allow_embed = threading.Event()
    original_embed_chunks = appmod.embed_chunks

    def blocking_embed_chunks(*args: object, **kwargs: object):
        embed_started.set()
        if not allow_embed.wait(timeout=5):
            raise TimeoutError("test did not release embedding")
        yield from original_embed_chunks(*args, **kwargs)

    monkeypatch.setattr(appmod, "embed_chunks", blocking_embed_chunks)
    request = appmod.IndexRequest(
        workspace_id="ws1",
        document_id="stale",
        text="must not enter the replacement corpus",
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        index_future = executor.submit(appmod.index_endpoint, request, None)
        assert embed_started.wait(timeout=5)
        reset_body = appmod.set_embedding(appmod.EmbeddingConfigRequest(dim=8), None)
        allow_embed.set()
        with pytest.raises(appmod.HTTPException) as exc_info:
            index_future.result(timeout=5)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "index_configuration_changed_retry"
    assert reset_body["index_signature"]["embedding_configured_dimension"] == 8
    artifact_dir = Path(os.environ["RAG_INDEX_PATH"]).parent
    assert (artifact_dir / "index_instance_id").exists()
    assert not (artifact_dir / "index_build_receipt.json").exists()


def test_cli_and_http_share_identity_when_state_is_on_another_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    appmod = importlib.import_module("slimx_rag.server.app")
    kb = tmp_path / "kb"
    index_dir = tmp_path / "index-volume"
    state_path = tmp_path / "state-volume" / "index_state.json"
    kb.mkdir()
    (kb / "doc.md").write_text("shared artifact location", encoding="utf-8")
    assert (
        main(
            [
                "run",
                "--kb-dir",
                str(kb),
                "--out-dir",
                str(index_dir),
                "--state",
                str(state_path),
                "--embed-dim",
                "8",
            ]
        )
        == 0
    )
    identity_path = index_dir / "index_instance_id"
    receipt_path = index_dir / "index_build_receipt.json"
    instance_id = identity_path.read_text(encoding="utf-8").strip()
    assert receipt_path.exists()
    assert not (state_path.parent / "index_instance_id").exists()
    assert not (state_path.parent / "index_build_receipt.json").exists()

    monkeypatch.setenv("RAG_INDEX_PATH", str(index_dir / "index.jsonl"))
    monkeypatch.setenv("RAG_STATE_PATH", str(state_path))
    monkeypatch.setenv("RAG_EMBED_DIM", "8")
    monkeypatch.delenv("RAG_BACKEND_CONFIG", raising=False)
    monkeypatch.delenv("RAG_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("DEMO_AUTH_TOKEN", raising=False)
    appmod._reset_index_cache()
    separate_client = TestClient(app)

    configured = separate_client.get("/api/config")
    ready = separate_client.get("/ready")

    assert configured.status_code == 200
    assert configured.json()["index_signature_source"] == "persisted_build"
    assert configured.json()["index_signature"]["index_instance_id"] == instance_id
    assert ready.status_code == 200
    assert ready.json()["index_signature"]["index_instance_id"] == instance_id


def test_health_reports_configured_embed_device(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_rag_auth_token_is_the_canonical_env_and_wins_over_the_legacy_alias(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # RAG_AUTH_TOKEN is canonical; DEMO_AUTH_TOKEN remains a working legacy alias.
    monkeypatch.setenv("RAG_AUTH_TOKEN", "newtoken")
    monkeypatch.setenv("DEMO_AUTH_TOKEN", "oldtoken")

    assert client.get("/api/config", headers={"Authorization": "Bearer oldtoken"}).status_code == 401
    assert client.get("/api/config", headers={"Authorization": "Bearer newtoken"}).status_code == 200


def test_health_and_ready_report_auth_enabled(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Off by default (no token configured).
    assert client.get("/health").json()["auth_enabled"] is False

    monkeypatch.setenv("RAG_AUTH_TOKEN", "secret")
    assert client.get("/health").json()["auth_enabled"] is True
    ready = client.get("/ready", headers={"Authorization": "Bearer secret"})
    assert ready.status_code in (200, 503)
    if ready.status_code == 200:
        assert ready.json()["auth_enabled"] is True


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


def test_empty_config_is_partial_and_does_not_create_instance_identity(ingest_client: TestClient) -> None:
    identity_path = Path(os.environ["RAG_INDEX_PATH"]).parent / "index_instance_id"
    assert not identity_path.exists()

    body = ingest_client.get("/api/config").json()

    assert body["index_signature_source"] == "configured_partial"
    assert body["index_signature"]["signature_complete"] is False
    assert body["index_signature"]["index_instance_id"] is None
    assert not identity_path.exists()


def test_index_ingests_and_is_retrievable(ingest_client: TestClient) -> None:
    initial_config = ingest_client.get("/api/config").json()
    configured_signature = initial_config["index_signature"]
    assert initial_config["index_signature_source"] == "configured_partial"
    assert configured_signature["signature_complete"] is False
    res = ingest_client.post(
        "/api/index",
        json={"workspace_id": "ws1", "document_id": "doc1", "text": "alpha beta gamma delta epsilon."},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ready"
    assert body["chunk_count"] >= 1
    assert body["doc_id"]
    assert body["index_signature"]["compatibility_fingerprint"]
    assert body["index_signature"]["embedding_dimension"] == 16
    assert body["document_pipeline"]["ingest_mode"] == "text"
    assert body["document_pipeline"]["parser"] is None
    assert (
        body["document_pipeline"]["chunk_config_fingerprint"]
        == body["index_signature"]["text_chunk_config_fingerprint"]
    )
    assert body["index_signature_source"] == "persisted_build"
    assert body["index_signature"]["signature_complete"] is True
    persisted_config = ingest_client.get("/api/config").json()
    assert persisted_config["index_signature_source"] == "persisted_build"
    assert (
        persisted_config["index_signature"]["compatibility_fingerprint"]
        == body["index_signature"]["compatibility_fingerprint"]
    )
    ready = ingest_client.get("/ready").json()
    assert ready["index_signature_source"] == "persisted_build"
    assert ready["index_signature"]["compatibility_fingerprint"] == body["index_signature"]["compatibility_fingerprint"]

    # New chunks are visible to retrieve immediately.
    r = ingest_client.post("/api/retrieve", json={"question": "beta gamma", "top_k": 5})
    assert r.status_code == 200
    assert "beta" in _texts(r.json())


def test_index_signature_uses_actual_external_vector_dimension(
    ingest_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    appmod = importlib.import_module("slimx_rag.server.app")

    class _ThreeDimensionalEmbedder:
        dim = None

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0, 0.0] for _text in texts]

        def token_counter(self):
            from slimx_rag.chunk import HeuristicTokenCounter

            return HeuristicTokenCounter()

    monkeypatch.setenv("RAG_EMBED_PROVIDER", "hf")
    monkeypatch.setenv("RAG_EMBED_DIM", "384")
    monkeypatch.setattr(appmod, "get_cached_embedder", lambda _settings: _ThreeDimensionalEmbedder())

    body = ingest_client.post(
        "/api/index",
        json={"workspace_id": "ws1", "document_id": "external", "text": "actual vectors win"},
    ).json()

    assert body["embed"]["dim"] == 384  # legacy configured field stays unchanged
    assert body["index_signature"]["embedding_dimension"] == 3
    ready = ingest_client.get("/ready").json()
    assert ready["ready"] is True
    assert ready["index_signature"]["embedding_dimension"] == 3
    assert ingest_client.get("/api/config").json()["index_signature"]["embedding_dimension"] == 3


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
    assert "omega" in all_chunks  # docA new content present
    assert "beta" in all_chunks  # docB untouched
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
    ingest = ingest_client.post("/api/index", json={"workspace_id": "ws1", "document_id": "doc1", "text": text}).json()

    res = ingest_client.get("/api/documents/doc1/chunks", params={"workspace_id": "ws1"})
    assert res.status_code == 200
    body = res.json()
    # One entry per indexed chunk, contiguous ordinals starting at 0, text present.
    assert body["chunk_count"] == ingest["chunk_count"]
    assert [c["ordinal"] for c in body["chunks"]] == list(range(body["chunk_count"]))
    assert all(c["text"] for c in body["chunks"])
    assert all(c["chunk_id"] for c in body["chunks"])
    # Inspection contract: the richer structure keys are always present (None for flat text ingest;
    # populated for page-aware ingest) so ControlRoom's Document Reader can render them uniformly.
    for key in ("section_path", "parent_id", "page_type", "token_count"):
        assert all(key in c for c in body["chunks"])


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
def test_retrieve_reuses_cached_index_until_file_changes(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
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
    indexed = ingest_client.post(
        "/api/index", json={"workspace_id": "ws1", "document_id": "d1", "text": "alpha beta gamma"}
    ).json()
    previous_instance_id = indexed["index_signature"]["index_instance_id"]
    assert "alpha" in _texts(ingest_client.post("/api/retrieve", json={"question": "alpha", "top_k": 5}).json())

    res = ingest_client.post("/api/admin/embedding", json={"provider": "hash", "dim": 16, "device": "cuda"})
    assert res.status_code == 200
    body = res.json()
    assert body["index_reset"] is True
    assert body["embed"]["device"] == "cuda"
    assert body["index_signature"]["index_instance_id"] != previous_instance_id

    # The choice persists (read back via /api/config) and the index was discarded: a fresh
    # ingest rebuilds from scratch, so the old document is gone.
    assert ingest_client.get("/api/config").json()["embed"]["device"] == "cuda"
    ingest_client.post("/api/index", json={"workspace_id": "ws1", "document_id": "d2", "text": "omega omega"})
    after = _texts(ingest_client.post("/api/retrieve", json={"question": "x", "top_k": 50}).json())
    assert "omega" in after and "alpha" not in after


def test_remote_embedding_reset_is_rejected_without_mutating_corpus_or_identity(
    ingest_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexed = ingest_client.post(
        "/api/index", json={"workspace_id": "ws1", "document_id": "d1", "text": "alpha"}
    ).json()
    instance_id = indexed["index_signature"]["index_instance_id"]
    index_path = Path(os.environ["RAG_INDEX_PATH"])
    state_path = Path(os.environ["RAG_STATE_PATH"])
    index_before = index_path.read_bytes()
    state_before = state_path.read_bytes()

    monkeypatch.setenv("RAG_INDEX_BACKEND", "qdrant")
    monkeypatch.setenv(
        "RAG_BACKEND_CONFIG",
        json.dumps(
            {
                "collection": "remote-corpus",
                "url": "https://qdrant.internal",
                "api_key": "secret",
            }
        ),
    )
    response = ingest_client.post("/api/admin/embedding", json={"dim": 8})

    assert response.status_code == 409
    assert "unsupported" in response.json()["detail"]
    assert index_path.read_bytes() == index_before
    assert state_path.read_bytes() == state_before
    assert (state_path.parent / "index_instance_id").read_text(encoding="utf-8").strip() == instance_id
    assert not (state_path.parent / "embed_override.json").exists()


def test_embedding_preflight_failure_preserves_corpus_identity_and_receipt(
    ingest_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    appmod = importlib.import_module("slimx_rag.server.app")
    indexed = ingest_client.post(
        "/api/index", json={"workspace_id": "ws1", "document_id": "d1", "text": "alpha"}
    ).json()
    index_path = Path(os.environ["RAG_INDEX_PATH"])
    artifact_dir = index_path.parent
    identity_path = artifact_dir / "index_instance_id"
    receipt_path = artifact_dir / "index_build_receipt.json"
    before = {
        "index": index_path.read_bytes(),
        "identity": identity_path.read_bytes(),
        "receipt": receipt_path.read_bytes(),
    }

    def fail_preflight(_settings: object) -> object:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(appmod, "get_cached_embedder", fail_preflight)
    response = ingest_client.post("/api/admin/embedding", json={"provider": "hf", "hf_model": "missing/model"})

    assert response.status_code == 422
    assert response.json()["detail"] == "embedder_preflight_failed: RuntimeError"
    assert index_path.read_bytes() == before["index"]
    assert identity_path.read_bytes() == before["identity"]
    assert receipt_path.read_bytes() == before["receipt"]
    assert indexed["index_signature"]["index_instance_id"] == identity_path.read_text(encoding="utf-8").strip()
    assert not (artifact_dir / "embed_override.json").exists()


def test_failed_local_reset_does_not_claim_success_or_rotate_identity(
    ingest_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    appmod = importlib.import_module("slimx_rag.server.app")
    indexed = ingest_client.post(
        "/api/index", json={"workspace_id": "ws1", "document_id": "d1", "text": "alpha"}
    ).json()
    instance_id = indexed["index_signature"]["index_instance_id"]
    index_path = Path(os.environ["RAG_INDEX_PATH"])
    state_path = Path(os.environ["RAG_STATE_PATH"])
    original_stage = appmod._stage_reset_artifact

    def fail_index_stage(path: Path, backup: Path) -> None:
        if path == index_path:
            raise PermissionError("simulated reset failure")
        original_stage(path, backup)

    monkeypatch.setattr(appmod, "_stage_reset_artifact", fail_index_stage)
    response = ingest_client.post("/api/admin/embedding", json={"dim": 8})

    assert response.status_code == 500
    assert response.json()["detail"] == "index_reset_failed: PermissionError"
    assert index_path.exists()
    assert (state_path.parent / "index_instance_id").read_text(encoding="utf-8").strip() == instance_id
    assert (state_path.parent / "index_build_receipt.json").exists()
    assert not (state_path.parent / "embed_override.json").exists()


def test_partial_reset_rollback_invalidates_identity_and_receipt(
    ingest_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    appmod = importlib.import_module("slimx_rag.server.app")
    ingest_client.post("/api/index", json={"workspace_id": "ws1", "document_id": "d1", "text": "alpha"})
    index_path = Path(os.environ["RAG_INDEX_PATH"])
    artifact_dir = index_path.parent
    receipt_path = artifact_dir / "index_build_receipt.json"
    original_stage = appmod._stage_reset_artifact
    original_restore = appmod._restore_reset_artifact

    def fail_late_stage(path: Path, backup: Path) -> None:
        if path == index_path:
            raise PermissionError("simulated late reset failure")
        original_stage(path, backup)

    def fail_receipt_restore(backup: Path, path: Path) -> None:
        if path == receipt_path:
            raise PermissionError("simulated rollback failure")
        original_restore(backup, path)

    monkeypatch.setattr(appmod, "_stage_reset_artifact", fail_late_stage)
    monkeypatch.setattr(appmod, "_restore_reset_artifact", fail_receipt_restore)
    response = ingest_client.post("/api/admin/embedding", json={"dim": 8})

    assert response.status_code == 500
    assert response.json()["detail"] == "index_reset_partial_failure: rollback=PermissionError"
    assert index_path.exists()
    assert not (artifact_dir / "index_instance_id").exists()
    assert not receipt_path.exists()
    assert not (artifact_dir / "embed_override.json").exists()


def test_set_embedding_requires_token_when_configured(
    ingest_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEMO_AUTH_TOKEN", "secret")
    assert ingest_client.post("/api/admin/embedding", json={"device": "cuda"}).status_code == 401
    ok = ingest_client.post("/api/admin/embedding", json={"device": "cuda"}, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


# --- document deletion (DELETE /api/documents/{id}) -------------------------------
def test_delete_document_removes_chunks_from_index(ingest_client: TestClient) -> None:
    ingest_client.post("/api/index", json={"workspace_id": "ws1", "document_id": "docA", "text": "alpha alpha alpha."})
    ingest_client.post("/api/index", json={"workspace_id": "ws1", "document_id": "docB", "text": "beta beta beta."})

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
    ingest_client.post("/api/index", json={"workspace_id": "wsA", "document_id": "dA", "text": "alpha beta gamma."})
    # Same document_id under a different workspace is a different doc identity -> no-op.
    wrong = ingest_client.delete("/api/documents/dA", params={"workspace_id": "wsB"})
    assert wrong.json()["deleted_chunks"] == 0
    assert ingest_client.get("/api/documents/dA/chunks", params={"workspace_id": "wsA"}).json()["chunk_count"] >= 1
    # Deleting under the right workspace removes it.
    right = ingest_client.delete("/api/documents/dA", params={"workspace_id": "wsA"})
    assert right.json()["deleted_chunks"] >= 1
    assert ingest_client.get("/api/documents/dA/chunks", params={"workspace_id": "wsA"}).json()["chunk_count"] == 0


def test_delete_document_is_idempotent(ingest_client: TestClient) -> None:
    ingest_client.post("/api/index", json={"workspace_id": "ws1", "document_id": "doc1", "text": "alpha beta."})
    first = ingest_client.delete("/api/documents/doc1", params={"workspace_id": "ws1"})
    assert first.json()["deleted_chunks"] >= 1
    second = ingest_client.delete("/api/documents/doc1", params={"workspace_id": "ws1"})
    assert second.status_code == 200
    assert second.json()["deleted_chunks"] == 0
