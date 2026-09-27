"""Corrections from the fourth same-model review (2026-09-26, at c7148a5; not independent).

1. A model switch that supplied ``hf_revision`` followed by a device change must keep the switched
   model's revision on the volume (before: the rewrite dropped it, the image's ``RAG_HF_REVISION``
   applied to the other model and every embed path returned 503).
2. A mutable ``RAG_HF_REVISION`` from the environment is rejected by the service itself
   (``embedder_config_invalid``), not only by the admin route and the CLI.
3. A short UTF-8 note with a few accented characters and one stray byte stays UTF-8; sparse
   cp1252 accents still decode as cp1252 (the two cases the byte ratio could not separate).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from slimx_rag.document.structure import decode_text
from slimx_rag.settings import EmbedConfigError, EmbedSettings

server = importlib.import_module("slimx_rag.server.app")

_ENV_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
_OTHER_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("RAG_INDEX_PATH", str(out / "index.jsonl"))
    monkeypatch.setenv("RAG_STATE_PATH", str(out / "index_state.json"))
    monkeypatch.setenv("RAG_EMBED_DIM", "32")
    for name in ("DEMO_AUTH_TOKEN", "RAG_AUTH_TOKEN", "RAG_BACKEND_CONFIG", "RAG_INDEX_BACKEND", "RAG_HF_REVISION"):
        monkeypatch.delenv(name, raising=False)
    server._reset_index_cache()
    return TestClient(server.app)


def _override() -> dict:
    return json.loads(server._embed_override_path().read_text("utf-8"))


# --- 1. a switched model keeps its revision across a later device change ------------------------


def test_device_change_keeps_the_switched_models_persisted_revision(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_HF_REVISION", _ENV_REVISION)
    server._reset_index_cache()
    # Step 1 (the README flow): switch the model and supply its exact revision.
    res = client.post(
        "/api/admin/embedding", json={"provider": "hash", "hf_model": "org/other-model", "hf_revision": _OTHER_REVISION}
    )
    assert res.status_code == 200, res.text
    assert _override()["revision"] == _OTHER_REVISION
    # Step 2 (ControlRoom's flow, which cannot send hf_revision): a device change.
    res = client.post("/api/admin/embedding", json={"device": "cpu"})
    assert res.status_code == 200, res.text
    assert _override()["revision"] == _OTHER_REVISION, "the switched model's revision must survive the rewrite"
    settings = server._embed_settings()
    assert settings.hf_model == "org/other-model" and settings.revision == _OTHER_REVISION
    assert settings.device == "cpu"
    assert client.get("/ready").status_code == 200
    # A prefix-only change keeps it as well.
    res = client.post("/api/admin/embedding", json={"device": "cpu", "dim": 32})
    assert res.status_code == 200 and _override()["revision"] == _OTHER_REVISION


def test_env_revision_is_still_not_adopted_into_the_override(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_HF_REVISION", _ENV_REVISION)
    server._reset_index_cache()
    res = client.post("/api/admin/embedding", json={"device": "cpu"})
    assert res.status_code == 200, res.text
    assert "revision" not in _override()  # the image's RAG_HF_REVISION keeps governing
    assert server._embed_settings().revision == _ENV_REVISION


# --- 2. a mutable env revision fails closed in the service ---------------------------------------


def test_mutable_env_revision_is_rejected_by_the_service(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_EMBED_PROVIDER", "hf")
    monkeypatch.setenv("RAG_HF_REVISION", "main")
    server._reset_index_cache()
    ready = client.get("/ready")
    assert ready.status_code == 503, ready.text
    assert "embedder_config_invalid" in ready.text and "mutable" in ready.text
    res = client.post("/api/index", json={"workspace_id": "ws", "document_id": "d1", "text": "some text"})
    assert res.status_code == 503, res.text
    assert res.json()["detail"]["code"] == "embedder_config_invalid"


def test_validate_raises_the_dedicated_configuration_error() -> None:
    with pytest.raises(EmbedConfigError):
        EmbedSettings(provider="hf", revision="main").validate()
    EmbedSettings(provider="hf", revision=_ENV_REVISION).validate()
    EmbedSettings(provider="hash", revision="main").validate()  # only the hf provider pins commits


# --- 3. the decoder separates both sparse cases --------------------------------------------------


def test_short_utf8_note_with_accents_and_one_stray_byte_stays_utf8() -> None:
    note = "Contact José Müller about the café’s heater. " + "Plain ASCII follow-up text. " * 10
    content = note.encode("utf-8") + b"\xff" + b" trailing"
    decoded = decode_text(content)
    assert "José Müller" in decoded and "café’s" in decoded, decoded[:80]
    assert decoded.count("�") == 1 and "Ã" not in decoded and "â€™" not in decoded


def test_sparse_cp1252_accents_still_decode_as_cp1252() -> None:
    note = ("Operating notes for the line. " * 90) + "Range 20–25 °C, contact José Müller."
    decoded = decode_text(note.encode("cp1252"))
    assert "José Müller" in decoded and "�" not in decoded
    assert decode_text("Résumé".encode("cp1252")) == "Résumé"
