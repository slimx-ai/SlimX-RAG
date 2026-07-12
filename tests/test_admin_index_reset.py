from __future__ import annotations

import importlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from slimx_rag.server.app import app
from slimx_rag.settings import EmbedSettings, IndexSettings

AUTH_HEADERS = {"Authorization": "Bearer reset-secret"}


@pytest.fixture
def reset_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("RAG_INDEX_PATH", str(out / "index.jsonl"))
    monkeypatch.setenv("RAG_STATE_PATH", str(out / "index_state.json"))
    monkeypatch.setenv("RAG_EMBED_DIM", "16")
    monkeypatch.delenv("RAG_INDEX_BACKEND", raising=False)
    monkeypatch.delenv("RAG_BACKEND_CONFIG", raising=False)
    monkeypatch.delenv("RAG_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("DEMO_AUTH_TOKEN", raising=False)
    appmod = importlib.import_module("slimx_rag.server.app")
    appmod._reset_index_cache()
    return TestClient(app)


def _index(client: TestClient, *, document_id: str = "d1", text: str = "alpha beta gamma") -> dict:
    response = client.post(
        "/api/index",
        json={"workspace_id": "ws1", "document_id": document_id, "text": text},
    )
    assert response.status_code == 200
    return response.json()


def _reset_payload(
    signature: dict,
    *,
    instance_id: str | None | object = ...,
    fingerprint: str | None | object = ...,
) -> dict:
    return {
        "confirmation": "RESET INDEX",
        "expected_index_instance_id": (signature["index_instance_id"] if instance_id is ... else instance_id),
        "expected_compatibility_fingerprint": (
            signature["compatibility_fingerprint"] if fingerprint is ... else fingerprint
        ),
    }


def _enable_reset_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_AUTH_TOKEN", "reset-secret")


def _artifact_dir() -> Path:
    return Path(os.environ["RAG_INDEX_PATH"]).parent


def _artifact_snapshot() -> dict[str, bytes | None]:
    out = _artifact_dir()
    names = (
        "index.jsonl",
        "index.faiss",
        "index.faiss.meta.json",
        "index_state.json",
        "index_instance_id",
        "index_build_receipt.json",
        "embed_override.json",
    )
    return {name: ((out / name).read_bytes() if (out / name).exists() else None) for name in names}


def test_reset_contract_requires_canonical_auth_confirmation_and_both_preconditions(
    reset_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexed = _index(reset_client)
    payload = _reset_payload(indexed["index_signature"])
    before = _artifact_snapshot()

    no_auth = reset_client.post("/api/admin/index/reset", json=payload)
    assert no_auth.status_code == 503
    assert no_auth.json()["detail"]["code"] == "index_reset_auth_not_configured"

    # The deprecated demo token is intentionally insufficient for this destructive route.
    monkeypatch.setenv("DEMO_AUTH_TOKEN", "legacy-only")
    legacy_only = reset_client.post(
        "/api/admin/index/reset",
        json=payload,
        headers={"Authorization": "Bearer legacy-only"},
    )
    assert legacy_only.status_code == 503

    _enable_reset_auth(monkeypatch)
    assert reset_client.post("/api/admin/index/reset", json=payload).status_code == 401
    assert (
        reset_client.post(
            "/api/admin/index/reset",
            json=payload,
            headers={"Authorization": "Bearer wrong"},
        ).status_code
        == 401
    )

    invalid_bodies = [
        {},
        {"confirmation": "RESET INDEX"},
        {
            "confirmation": "RESET INDEX",
            "expected_index_instance_id": indexed["index_signature"]["index_instance_id"],
        },
        {
            **payload,
            "confirmation": "reset index",
        },
        {
            **payload,
            "expected_compatibility_fingerprint": "not-a-fingerprint",
        },
    ]
    for body in invalid_bodies:
        assert reset_client.post("/api/admin/index/reset", json=body, headers=AUTH_HEADERS).status_code == 422

    assert _artifact_snapshot() == before


def test_reset_recovers_signature_mismatch_preserves_embedding_and_is_ready(
    reset_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexed = _index(reset_client)
    previous_signature = indexed["index_signature"]
    assert not (_artifact_dir() / "embed_override.json").exists()

    # A global shaping change makes the existing receipt incompatible without changing
    # embedding settings. This is the recovery path the dedicated endpoint exists for.
    monkeypatch.setenv("RAG_CHUNK_SIZE", "401")
    _enable_reset_auth(monkeypatch)
    mismatch = reset_client.get("/ready", headers=AUTH_HEADERS)
    assert mismatch.status_code == 503
    assert mismatch.json()["reason"] == "index_signature_mismatch"
    assert mismatch.json()["auth_enabled"] is True
    assert mismatch.json()["engine_version"]

    response = reset_client.post(
        "/api/admin/index/reset",
        json=_reset_payload(previous_signature),
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["index_reset"] is True
    assert body["previous_index_instance_id"] == previous_signature["index_instance_id"]
    assert (
        body["previous_index_signature"]["compatibility_fingerprint"] == previous_signature["compatibility_fingerprint"]
    )
    assert body["new_index_instance_id"] != body["previous_index_instance_id"]
    assert body["index_signature"]["index_instance_id"] == body["new_index_instance_id"]
    assert body["index_signature"]["signature_complete"] is False
    assert body["index_signature_source"] == "configured_partial"
    assert body["engine_version"] == body["index_signature"]["engine_version"]
    assert not (_artifact_dir() / "embed_override.json").exists()

    ready = reset_client.get("/ready", headers=AUTH_HEADERS)
    assert ready.status_code == 200
    ready_body = ready.json()
    assert ready_body["ready"] is True
    assert ready_body["index_count"] == 0
    assert (
        ready_body["index_signature"]["compatibility_fingerprint"]
        == body["index_signature"]["compatibility_fingerprint"]
    )

    # A second reset accepts the runtime-partial fingerprint returned by both the first
    # reset and /ready; consumers are not forced through the offline /api/config variant.
    repeated = reset_client.post(
        "/api/admin/index/reset",
        json=_reset_payload(body["index_signature"]),
        headers=AUTH_HEADERS,
    )
    assert repeated.status_code == 200
    assert repeated.json()["previous_index_instance_id"] == body["new_index_instance_id"]
    assert repeated.json()["new_index_instance_id"] != body["new_index_instance_id"]


def test_reset_accepts_offline_configured_partial_fingerprint(
    reset_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_reset_auth(monkeypatch)
    configured = reset_client.get("/api/config", headers=AUTH_HEADERS).json()["index_signature"]
    assert configured["index_instance_id"] is None

    first = reset_client.post(
        "/api/admin/index/reset",
        json=_reset_payload(configured),
        headers=AUTH_HEADERS,
    )
    assert first.status_code == 200

    configured_after = reset_client.get("/api/config", headers=AUTH_HEADERS).json()["index_signature"]
    second = reset_client.post(
        "/api/admin/index/reset",
        json=_reset_payload(configured_after),
        headers=AUTH_HEADERS,
    )
    assert second.status_code == 200


def test_pure_reset_preserves_existing_embedding_override_bytes(
    reset_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = reset_client.post("/api/admin/embedding", json={"device": "cuda"})
    assert changed.status_code == 200
    indexed = _index(reset_client)
    override_path = _artifact_dir() / "embed_override.json"
    override_before = override_path.read_bytes()
    _enable_reset_auth(monkeypatch)

    response = reset_client.post(
        "/api/admin/index/reset",
        json=_reset_payload(indexed["index_signature"]),
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    assert override_path.read_bytes() == override_before
    assert reset_client.get("/api/config", headers=AUTH_HEADERS).json()["embed"]["device"] == "cuda"


def test_reset_rejects_stale_or_bypassed_preconditions_without_mutation(
    reset_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexed = _index(reset_client)
    signature = indexed["index_signature"]
    before = _artifact_snapshot()
    _enable_reset_auth(monkeypatch)

    cases = [
        (_reset_payload(signature, instance_id="idx_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"), "index_instance_id"),
        (_reset_payload(signature, instance_id=None), "index_instance_id"),
        (_reset_payload(signature, fingerprint="0" * 16), "compatibility_fingerprint"),
        (_reset_payload(signature, fingerprint=None), "compatibility_fingerprint"),
    ]
    for payload, field in cases:
        response = reset_client.post("/api/admin/index/reset", json=payload, headers=AUTH_HEADERS)
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "index_reset_precondition_failed"
        assert response.json()["detail"]["field"] == field
        assert _artifact_snapshot() == before


def test_reset_recovers_unreadable_receipt_and_state_with_explicit_null_fingerprint(
    reset_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexed = _index(reset_client)
    signature = indexed["index_signature"]
    (_artifact_dir() / "index_build_receipt.json").write_text("{", encoding="utf-8")
    (_artifact_dir() / "index_state.json").write_text("{", encoding="utf-8")
    _enable_reset_auth(monkeypatch)

    diagnosed = reset_client.get("/ready", headers=AUTH_HEADERS)
    assert diagnosed.status_code == 503
    assert diagnosed.json()["reason"] == "index_build_receipt_invalid"

    response = reset_client.post(
        "/api/admin/index/reset",
        json=_reset_payload(signature, fingerprint=None),
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["previous_index_signature_source"] == "configured_partial"
    assert reset_client.get("/ready", headers=AUTH_HEADERS).status_code == 200


def test_ready_distinguishes_invalid_state_and_reset_recovers_it(
    reset_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexed = _index(reset_client)
    signature = indexed["index_signature"]
    (_artifact_dir() / "index_state.json").write_text("{", encoding="utf-8")
    _enable_reset_auth(monkeypatch)

    diagnosed = reset_client.get("/ready", headers=AUTH_HEADERS)
    assert diagnosed.status_code == 503
    assert diagnosed.json()["reason"] == "index_state_invalid"
    assert diagnosed.json()["auth_enabled"] is True
    assert diagnosed.json()["engine_version"]

    response = reset_client.post(
        "/api/admin/index/reset",
        json=_reset_payload(signature),
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert reset_client.get("/ready", headers=AUTH_HEADERS).status_code == 200


def test_reset_recovers_receipt_instance_mismatch_using_active_signature(
    reset_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _index(reset_client)
    active_instance = "idx_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    (_artifact_dir() / "index_instance_id").write_text(active_instance + "\n", encoding="utf-8")
    _enable_reset_auth(monkeypatch)

    diagnosed = reset_client.get("/ready", headers=AUTH_HEADERS)
    assert diagnosed.status_code == 503
    diagnosis = diagnosed.json()
    assert diagnosis["reason"] == "index_build_receipt_instance_mismatch"
    active_signature = diagnosis["active_index_signature"]
    assert active_signature["index_instance_id"] == active_instance

    response = reset_client.post(
        "/api/admin/index/reset",
        json=_reset_payload(active_signature),
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["previous_index_instance_id"] == active_instance
    assert response.json()["previous_index_signature_source"] == "configured_partial"


def test_reset_preflight_failure_preserves_all_artifacts(
    reset_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    appmod = importlib.import_module("slimx_rag.server.app")
    indexed = _index(reset_client)
    before = _artifact_snapshot()
    _enable_reset_auth(monkeypatch)

    def fail_preflight(_settings: object) -> object:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(appmod, "get_cached_embedder", fail_preflight)
    response = reset_client.post(
        "/api/admin/index/reset",
        json=_reset_payload(indexed["index_signature"]),
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "embedder_preflight_failed",
        "error_type": "RuntimeError",
        "owner_action": "Restore the configured embedder/tokenizer before retrying the reset.",
    }
    assert _artifact_snapshot() == before


def test_failed_reset_rolls_back_without_touching_embedding_override(
    reset_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    appmod = importlib.import_module("slimx_rag.server.app")
    reset_client.post("/api/admin/embedding", json={"device": "cuda"})
    indexed = _index(reset_client)
    before = _artifact_snapshot()
    index_path = Path(os.environ["RAG_INDEX_PATH"])
    original_stage = appmod._stage_reset_artifact

    def fail_index_stage(path: Path, backup: Path) -> None:
        if path == index_path:
            raise PermissionError("simulated reset failure")
        original_stage(path, backup)

    monkeypatch.setattr(appmod, "_stage_reset_artifact", fail_index_stage)
    _enable_reset_auth(monkeypatch)
    response = reset_client.post(
        "/api/admin/index/reset",
        json=_reset_payload(indexed["index_signature"]),
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "index_reset_failed"
    assert response.json()["detail"]["active_identity_invalidated"] is False
    assert _artifact_snapshot() == before
    assert reset_client.get("/ready", headers=AUTH_HEADERS).status_code == 200


@pytest.mark.parametrize(
    ("backend", "backend_config"),
    [
        (
            "qdrant",
            {"url": "https://qdrant.internal", "collection": "active", "api_key": "secret"},
        ),
        (
            "pgvector",
            {
                "dsn": "postgresql://user:secret@postgres.internal/rag",
                "schema": "public",
                "table": "active",
            },
        ),
    ],
)
def test_remote_reset_is_structured_409_and_does_not_mutate_local_artifacts(
    reset_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    backend_config: dict[str, str],
) -> None:
    indexed = _index(reset_client)
    before = _artifact_snapshot()
    _enable_reset_auth(monkeypatch)
    monkeypatch.setenv("RAG_INDEX_BACKEND", backend)
    monkeypatch.setenv("RAG_BACKEND_CONFIG", json.dumps(backend_config))

    response = reset_client.post(
        "/api/admin/index/reset",
        json=_reset_payload(indexed["index_signature"]),
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "index_reset_backend_unsupported"
    assert detail["backend"] == backend
    assert detail["retryable"] is False
    assert "owner_action" in detail
    assert _artifact_snapshot() == before


def _block_embedding(monkeypatch: pytest.MonkeyPatch) -> tuple[threading.Event, threading.Event]:
    appmod = importlib.import_module("slimx_rag.server.app")
    started = threading.Event()
    release = threading.Event()
    original = appmod.embed_chunks

    def blocked(*args: object, **kwargs: object):
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release embedding")
        yield from original(*args, **kwargs)

    monkeypatch.setattr(appmod, "embed_chunks", blocked)
    return started, release


def test_concurrent_first_writers_publish_exactly_one_generation(
    reset_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    appmod = importlib.import_module("slimx_rag.server.app")
    original_get_embedder = appmod.get_cached_embedder
    both_snapshotted = threading.Barrier(2, timeout=5)

    def synchronized_get_embedder(settings: object):
        embedder = original_get_embedder(settings)
        # Both requests have already snapshotted instance=None before reaching this call.
        both_snapshotted.wait()
        return embedder

    monkeypatch.setattr(appmod, "get_cached_embedder", synchronized_get_embedder)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                reset_client.post,
                "/api/index",
                json={
                    "workspace_id": "ws1",
                    "document_id": f"first-{ordinal}",
                    "text": f"concurrent first writer {ordinal}",
                },
            )
            for ordinal in range(2)
        ]
        responses = [future.result(timeout=5) for future in futures]
    monkeypatch.setattr(appmod, "get_cached_embedder", original_get_embedder)

    assert sorted(response.status_code for response in responses) == [200, 409]
    rejected = next(response for response in responses if response.status_code == 409)
    assert rejected.json()["detail"]["code"] == "index_generation_changed_retry"
    assert rejected.json()["detail"]["retryable"] is True

    state = appmod.IndexState.load(_artifact_dir() / "index_state.json")
    assert len(state.docs) == 1
    receipt = appmod.load_index_build_receipt(_artifact_dir() / "index_build_receipt.json")
    assert receipt is not None
    assert (
        receipt.index_signature.index_instance_id
        == (_artifact_dir() / "index_instance_id").read_text(encoding="utf-8").strip()
    )
    assert reset_client.get("/ready").status_code == 200


def test_text_index_started_before_reset_cannot_write_new_generation(
    reset_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexed = _index(reset_client)
    _enable_reset_auth(monkeypatch)
    started, release = _block_embedding(monkeypatch)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            reset_client.post,
            "/api/index",
            json={"workspace_id": "ws1", "document_id": "stale", "text": "stale generation text"},
            headers=AUTH_HEADERS,
        )
        assert started.wait(timeout=5)
        reset = reset_client.post(
            "/api/admin/index/reset",
            json=_reset_payload(indexed["index_signature"]),
            headers=AUTH_HEADERS,
        )
        release.set()
        stale = future.result(timeout=5)

    assert reset.status_code == 200
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "index_generation_changed_retry"
    assert stale.json()["detail"]["retryable"] is True
    assert not (_artifact_dir() / "index.jsonl").exists()
    assert not (_artifact_dir() / "index_state.json").exists()
    assert not (_artifact_dir() / "index_build_receipt.json").exists()


def test_file_index_started_before_reset_cannot_write_new_generation(
    reset_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexed = _index(reset_client)
    _enable_reset_auth(monkeypatch)
    started, release = _block_embedding(monkeypatch)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            reset_client.post,
            "/api/index/file",
            files={"file": ("stale.md", b"# Stale\n\nold generation", "text/markdown")},
            data={"workspace_id": "ws1", "document_id": "stale-file"},
            headers=AUTH_HEADERS,
        )
        assert started.wait(timeout=5)
        reset = reset_client.post(
            "/api/admin/index/reset",
            json=_reset_payload(indexed["index_signature"]),
            headers=AUTH_HEADERS,
        )
        release.set()
        stale = future.result(timeout=5)

    assert reset.status_code == 200
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "index_generation_changed_retry"
    assert stale.json()["detail"]["retryable"] is True
    assert not (_artifact_dir() / "index.jsonl").exists()
    assert not (_artifact_dir() / "index_state.json").exists()
    assert not (_artifact_dir() / "index_build_receipt.json").exists()


def test_pure_faiss_reset_removes_companion_artifacts_and_preserves_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    appmod = importlib.import_module("slimx_rag.server.app")
    out = tmp_path / "faiss"
    out.mkdir()
    index_path = out / "index.faiss"
    monkeypatch.setenv("RAG_INDEX_PATH", str(index_path))
    monkeypatch.setenv("RAG_STATE_PATH", str(out / "index_state.json"))
    index_path.write_bytes(b"faiss-index")
    index_path.with_suffix(".faiss.meta.json").write_text("{}", encoding="utf-8")
    (out / "index_state.json").write_text("{}", encoding="utf-8")
    (out / "index_build_receipt.json").write_text("invalid but staged", encoding="utf-8")
    override_path = out / "embed_override.json"
    override_path.write_text('{"provider":"hash","dim":16}', encoding="utf-8")
    override_before = override_path.read_bytes()

    with appmod.locked_index_instance(appmod._index_instance_id_path(), create=True) as lease:
        previous_instance = lease.instance_id
        new_instance = appmod._reset_index(
            IndexSettings(backend="faiss"),
            lease=lease,
            new_embed_settings=EmbedSettings(dim=16),
            persist_embed_override=False,
        )

    assert previous_instance is not None
    assert new_instance != previous_instance
    assert not index_path.exists()
    assert not index_path.with_suffix(".faiss.meta.json").exists()
    assert not (out / "index_state.json").exists()
    assert not (out / "index_build_receipt.json").exists()
    assert override_path.read_bytes() == override_before
