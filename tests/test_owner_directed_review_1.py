"""Owner-directed Claude review 1 of `1dd6736` (2026-09-26; same model as the author, not independent).

RAG-REV-001 / RAG-AUD-059: the dense-only retrieval path of the non-local backends (FAISS, Qdrant,
pgvector) built its embedder through ``make_embedder``, which did not validate the settings, so a
mutable ``RAG_HF_REVISION`` from the environment was accepted by unscoped ``/api/retrieve`` and
``/api/ask`` while ``/ready`` already reported ``embedder_config_invalid``. Every construction now
validates and the retrieval routes report the same reason.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("faiss")

import slimx_rag.embed.embedder as embedder_module  # noqa: E402
from slimx_rag.embed import reset_embedder_cache  # noqa: E402
from slimx_rag.settings import EmbedConfigError, EmbedSettings  # noqa: E402

server = importlib.import_module("slimx_rag.server.app")
_PINNED = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


@pytest.fixture
def faiss_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("RAG_INDEX_BACKEND", "faiss")
    monkeypatch.setenv("RAG_INDEX_PATH", str(out / "index.faiss"))
    monkeypatch.setenv("RAG_STATE_PATH", str(out / "index_state.json"))
    monkeypatch.setenv("RAG_EMBED_PROVIDER", "hash")
    monkeypatch.setenv("RAG_EMBED_DIM", "384")  # the dimension of the real model, so only the provider changes
    for name in (
        "DEMO_AUTH_TOKEN",
        "RAG_AUTH_TOKEN",
        "RAG_BACKEND_CONFIG",
        "RAG_HF_REVISION",
        "RAG_REQUIRE_WORKSPACE_SCOPE",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_embedder_cache()
    server._reset_index_cache()
    return TestClient(server.app)


def test_make_embedder_validates_every_construction() -> None:
    with pytest.raises(EmbedConfigError):
        embedder_module.make_embedder(EmbedSettings(provider="hf", revision="main"))


def test_non_local_dense_path_fails_closed_on_a_mutable_env_revision(
    faiss_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    res = faiss_client.post(
        "/api/index",
        json={"workspace_id": "w", "document_id": "d", "text": "Pump P-101\n\nThe pump was serviced in February."},
    )
    assert res.status_code == 200, res.text
    # The environment now names the real provider with a mutable revision.
    monkeypatch.setenv("RAG_EMBED_PROVIDER", "hf")
    monkeypatch.setenv("RAG_HF_REVISION", "main")
    reset_embedder_cache()
    server._reset_index_cache()
    seen: list[str | None] = []
    original_init = embedder_module.HuggingFaceEmbedder.__init__

    def spy(self: object, *args: object, **kwargs: object) -> None:  # pragma: no cover - must never run
        seen.append(kwargs.get("revision"))  # type: ignore[arg-type]
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(embedder_module.HuggingFaceEmbedder, "__init__", spy)
    ready = faiss_client.get("/ready")
    assert ready.status_code == 503 and "embedder_config_invalid" in ready.text
    for route in ("/api/retrieve", "/api/ask"):
        res = faiss_client.post(route, json={"question": "When was the pump serviced?"})  # unscoped: legacy dense path
        assert res.status_code == 503, f"{route}: {res.status_code} {res.text}"
        assert res.json()["detail"]["code"] == "embedder_config_invalid", res.text
        assert "mutable" in res.json()["detail"]["detail"]
    assert seen == [], "no embedder may be constructed with a mutable revision"


def test_non_local_dense_path_still_serves_a_valid_pinned_configuration(faiss_client: TestClient) -> None:
    res = faiss_client.post(
        "/api/index", json={"workspace_id": "w", "document_id": "d", "text": "Pump P-101\n\nServiced in February."}
    )
    assert res.status_code == 200, res.text
    res = faiss_client.post("/api/retrieve", json={"question": "When was the pump serviced?"})
    assert res.status_code == 200, res.text
    assert res.json()["chunks"], "the FAISS dense path returns the indexed chunk"
    assert EmbedSettings(provider="hf", revision=_PINNED).validate() is None
