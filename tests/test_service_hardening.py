"""Service-boundary corrections from the ControlRoom qualification audit (2026-09-25).

Covers: authoritative per-document deletion and replacement after a lost state commit
(RAG-AUD-003), workspace/document identifier validation (RAG-AUD-004), fail-closed scope
validation and the RAG_REQUIRE_WORKSPACE_SCOPE mode (RAG-AUD-005), binary/HTML rejection by
the catch-all parsers (RAG-AUD-009), structured failure codes instead of bare 500s
(RAG-AUD-010), demo-endpoint restrictions (RAG-AUD-012), reserved caller metadata
(RAG-AUD-015) and request bounds (RAG-AUD-017).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from slimx_rag.document import DocumentSource, parse_document
from slimx_rag.document.parser import UnsupportedDocumentError
from slimx_rag.document.structure import detect_source_type, looks_binary
from slimx_rag.index import make_index_backend
from slimx_rag.settings import IndexSettings

server = importlib.import_module("slimx_rag.server.app")


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("RAG_INDEX_PATH", str(out / "index.jsonl"))
    monkeypatch.setenv("RAG_STATE_PATH", str(out / "index_state.json"))
    monkeypatch.setenv("RAG_EMBED_DIM", "16")
    for name in (
        "DEMO_AUTH_TOKEN",
        "RAG_AUTH_TOKEN",
        "RAG_BACKEND_CONFIG",
        "RAG_INDEX_BACKEND",
        "RAG_REQUIRE_WORKSPACE_SCOPE",
        "RAG_ALLOW_MODEL_OVERRIDE",
        "RAG_EVAL_DATASET_DIR",
        "SLIMX_LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    server._reset_index_cache()
    return TestClient(server.app, raise_server_exceptions=False)


def _index(client: TestClient, ws: str, doc: str, text: str, metadata: dict | None = None):
    return client.post(
        "/api/index", json={"workspace_id": ws, "document_id": doc, "text": text, "metadata": metadata or {}}
    )


def _retrieve(client: TestClient, question: str, **scope: object) -> list[tuple[str, str]]:
    res = client.post("/api/retrieve", json={"question": question, "top_k": 10, **scope})
    assert res.status_code == 200, res.text
    return [(c["metadata"]["workspace_id"], c["metadata"]["document_id"]) for c in res.json()["chunks"]]


# --- RAG-AUD-003: authoritative delete / replace ------------------------------------------


def _lose_state_commit(state_path: Path) -> None:
    """Simulate a crash between index save and state commit: the index keeps the chunks,
    the state forgets the document. Then force the service to reload from disk."""
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["docs"] = {}
    state_path.write_text(json.dumps(state), encoding="utf-8")
    server._reset_index_cache()


def test_delete_sweeps_chunks_after_a_lost_state_commit(client: TestClient, tmp_path: Path) -> None:
    assert _index(client, "w", "doomed", "The zebrafish enclosure is in building nine.").status_code == 200
    _lose_state_commit(tmp_path / "out" / "index_state.json")
    assert _retrieve(client, "zebrafish enclosure", workspace_id="w") == [("w", "doomed")]

    res = client.delete("/api/documents/doomed", params={"workspace_id": "w"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["deleted_chunks"] == 1 and body["swept_chunks"] == 1 and body["total"] == 0
    assert _retrieve(client, "zebrafish enclosure", workspace_id="w") == []
    # Survives a restart: the sweep was persisted, not just applied to the hot backend.
    server._reset_index_cache()
    assert _retrieve(client, "zebrafish enclosure", workspace_id="w") == []


def test_reindex_after_a_lost_state_commit_leaves_no_stale_version(client: TestClient, tmp_path: Path) -> None:
    assert _index(client, "w", "live", "Version one mentions the axolotl tank.").status_code == 200
    _lose_state_commit(tmp_path / "out" / "index_state.json")
    res = _index(client, "w", "live", "Version two mentions the newt pond.")
    assert res.status_code == 200, res.text
    assert res.json()["total"] == 1
    assert _retrieve(client, "axolotl tank", workspace_id="w") == [("w", "live")]  # only the new chunk exists
    texts = [
        c["text"]
        for c in client.post("/api/retrieve", json={"question": "axolotl", "workspace_id": "w"}).json()["chunks"]
    ]
    assert texts == ["Version two mentions the newt pond."]


def test_delete_detailed_sweeps_on_faiss_backend_too(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    from slimx_rag.embed import EmbeddedChunk

    backend = make_index_backend(tmp_path / "index.faiss", settings=IndexSettings(backend="faiss"))
    backend.load()
    backend.upsert(
        [
            EmbeddedChunk(chunk_id="c1", text="a", vector=[1.0, 0.0], metadata={"doc_id": "d"}),
            EmbeddedChunk(chunk_id="c2", text="b", vector=[0.0, 1.0], metadata={"doc_id": "d"}),
            EmbeddedChunk(chunk_id="c3", text="c", vector=[0.5, 0.5], metadata={"doc_id": "other"}),
        ],
        skip_existing=False,
    )
    backend.commit_doc_state("d", "hash", ["c1"])  # bookkeeping knows only c1
    assert backend.delete_doc_detailed("d") == (1, 1)
    assert len(backend) == 1 and [cid for cid, _t, _m in backend.iter_chunks()] == ["c3"]


# --- RAG-AUD-004 / 005 / 017: identifiers, scope, bounds ------------------------------------


@pytest.mark.parametrize(
    "workspace_id,document_id", [("a/b", "c"), ("a", "b/c"), ("", "c"), ("  ", "c"), ("a", ""), ("a\x00", "c")]
)
def test_index_rejects_ambiguous_or_empty_identifiers(client: TestClient, workspace_id: str, document_id: str) -> None:
    res = _index(client, workspace_id, document_id, "text")
    assert res.status_code == 422, res.text
    file_res = client.post(
        "/api/index/file",
        data={"workspace_id": workspace_id, "document_id": document_id, "filename": "n.md"},
        files={"file": ("n.md", b"# hi", "text/markdown")},
    )
    assert file_res.status_code == 422, file_res.text
    detail = file_res.json()["detail"]
    # An empty multipart field is rejected by FastAPI's own validation (list-shaped detail);
    # every other bad identifier reaches the service's structured code.
    assert isinstance(detail, list) or detail["code"] == "invalid_identifier"


def test_chunks_and_delete_reject_invalid_identifiers(client: TestClient) -> None:
    assert client.get("/api/documents/x/chunks", params={"workspace_id": ""}).status_code == 422
    assert client.delete("/api/documents/x", params={"workspace_id": " "}).status_code == 422


def test_retrieve_rejects_empty_scope_values_instead_of_widening(client: TestClient) -> None:
    _index(client, "tenantA", "secret", "Tenant A secret vault code 4411.")
    _index(client, "tenantB", "public", "Tenant B public notes.")
    assert client.post("/api/retrieve", json={"question": "vault", "workspace_id": ""}).status_code == 422
    assert (
        client.post(
            "/api/retrieve", json={"question": "vault", "workspace_id": "tenantB", "document_ids": []}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/retrieve", json={"question": "vault", "workspace_id": "tenantB", "document_ids": [""]}
        ).status_code
        == 422
    )
    assert client.post("/api/retrieve", json={"question": "   ", "workspace_id": "tenantB"}).status_code == 422
    assert (
        client.post("/api/retrieve", json={"question": "vault", "workspace_id": "tenantB", "top_k": 10**9}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/retrieve", json={"question": "x" * (server.MAX_QUESTION_CHARS + 1), "workspace_id": "tenantB"}
        ).status_code
        == 422
    )
    # A valid scope still works and never crosses the workspace; document_ids alone is rejected
    # in every mode because the same document_id may exist in several workspaces.
    assert _retrieve(client, "vault code", workspace_id="tenantB") == [("tenantB", "public")]
    assert _retrieve(client, "vault code", workspace_id="tenantB", document_ids=["secret"]) == []
    assert client.post("/api/retrieve", json={"question": "vault", "document_ids": ["secret"]}).status_code == 422


def test_require_workspace_scope_mode_fails_closed(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _index(client, "tenantA", "secret", "Tenant A secret vault code 4411.")
    monkeypatch.setenv("RAG_REQUIRE_WORKSPACE_SCOPE", "1")
    for path in ("/api/retrieve", "/api/ask"):
        res = client.post(path, json={"question": "vault code"})
        assert res.status_code == 400, res.text
        assert res.json()["detail"]["code"] == "workspace_scope_required"
        # document_ids never replaces the workspace: rejected by validation in every mode.
        res = client.post(path, json={"question": "vault code", "document_ids": ["secret"]})
        assert res.status_code == 422
    res = client.post("/api/eval/run", json={"dataset": "examples/research_demo/eval/questions.jsonl"})
    assert res.status_code == 400 and res.json()["detail"]["code"] == "workspace_scope_required"
    assert _retrieve(client, "vault code", workspace_id="tenantA") == [("tenantA", "secret")]
    monkeypatch.delenv("RAG_REQUIRE_WORKSPACE_SCOPE")
    assert client.post("/api/retrieve", json={"question": "vault code"}).status_code == 200  # default stays permissive


# --- RAG-AUD-015: reserved metadata -----------------------------------------------------------


def test_caller_metadata_cannot_forge_identity_or_locator_fields(client: TestClient) -> None:
    res = _index(
        client,
        "wsX",
        "d1",
        "Falcon facts.",
        {
            "workspace_id": "wsY",
            "document_id": "other",
            "page": 9,
            "section": "Forged",
            "parent_id": "p#1",
            "source_title": "Forged Title",
            "chunk_id": "forged",
            "title": "Falcon Notes",
            "author": "ok",
        },
    )
    assert res.status_code == 200, res.text
    assert _retrieve(client, "falcon facts", workspace_id="wsY") == []
    chunk = client.post("/api/retrieve", json={"question": "falcon facts", "workspace_id": "wsX"}).json()["chunks"][0]
    assert chunk["chunk_id"] != "forged"
    assert chunk["metadata"]["page"] is None and chunk["metadata"]["parent_id"] != "p#1"
    assert chunk["metadata"]["section"] != "Forged"
    assert chunk["metadata"]["source_title"] == "Falcon Notes"
    assert chunk["citation"] == "[Falcon Notes]"


# --- RAG-AUD-009: binary and HTML originals --------------------------------------------------


def test_text_catch_all_rejects_binary_and_html() -> None:
    assert looks_binary(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4)
    assert looks_binary(b"PK\x03\x04\x00\x00zipzip")
    assert not looks_binary(b"plain text")
    assert not looks_binary("café — unicode is fine".encode())
    assert detect_source_type("page.html", None) == "html"
    assert detect_source_type("page", "text/html; charset=utf-8") == "html"
    with pytest.raises(UnsupportedDocumentError):
        parse_document(
            DocumentSource(document_id="x", filename="photo.png", content=b"\x89PNG\x00\x00" + bytes(range(256)))
        )
    with pytest.raises(UnsupportedDocumentError):
        parse_document(DocumentSource(document_id="x", filename="page.html", content=b"<html><body>hi</body></html>"))
    with pytest.raises(UnsupportedDocumentError):
        parse_document(DocumentSource(document_id="x", filename="notes.md", content=b"\x00\x01\x02binary"))
    parsed = parse_document(DocumentSource(document_id="x", filename="notes.txt", content=b"Plain notes"))
    assert parsed.parser_name == "native-text"


def test_index_file_rejects_binary_with_parse_failed_so_hosts_fall_back(client: TestClient) -> None:
    res = client.post(
        "/api/index/file",
        data={"workspace_id": "w", "document_id": "sheet", "filename": "budget.xlsx"},
        files={"file": ("budget.xlsx", b"PK\x03\x04\x00\x00" + bytes(range(256)), "application/octet-stream")},
    )
    assert res.status_code == 422, res.text
    assert res.json()["detail"].startswith("parse_failed: UnsupportedDocumentError")
    res = client.post(
        "/api/index/file",
        data={"workspace_id": "w", "document_id": "page", "filename": "page.html", "mime_type": "text/html"},
        files={"file": ("page.html", b"<html><body><script>x()</script>hello</body></html>", "text/html")},
    )
    assert res.status_code == 422 and "parse_failed" in res.text


# --- RAG-AUD-010: structured failures ------------------------------------------------------------


def test_invalid_persisted_state_yields_structured_reasons_not_500(client: TestClient, tmp_path: Path) -> None:
    assert _index(client, "w", "ok", "A healthy document about pelicans.").status_code == 200
    out = tmp_path / "out"
    state = (out / "index_state.json").read_text(encoding="utf-8")
    (out / "index_state.json").write_text("{not json", encoding="utf-8")
    server._reset_index_cache()
    res = client.post("/api/retrieve", json={"question": "pelicans", "workspace_id": "w"})
    assert res.status_code == 503 and res.json()["detail"]["code"] == "index_state_invalid"
    assert client.get("/api/documents/ok/chunks", params={"workspace_id": "w"}).status_code == 503
    assert client.delete("/api/documents/ok", params={"workspace_id": "w"}).status_code == 503
    (out / "index_state.json").write_text(state, encoding="utf-8")

    receipt = out / "index_build_receipt.json"
    good = receipt.read_text(encoding="utf-8")
    receipt.write_text("{}", encoding="utf-8")
    server._reset_index_cache()
    res = _index(client, "w", "ok2", "x")
    assert res.status_code == 409 and res.json()["detail"]["code"] == "index_build_receipt_invalid"
    receipt.write_text(good, encoding="utf-8")

    index_file = out / "index.jsonl"
    good_index = index_file.read_text(encoding="utf-8")
    index_file.write_text(good_index + "{truncated", encoding="utf-8")
    server._reset_index_cache()
    res = client.post("/api/retrieve", json={"question": "pelicans", "workspace_id": "w"})
    assert res.status_code == 503 and res.json()["detail"]["code"] == "backend_load_failed"
    index_file.write_text(good_index, encoding="utf-8")
    server._reset_index_cache()
    assert client.get("/ready").status_code == 200


def test_embedder_init_failure_is_a_structured_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_EMBED_PROVIDER", "hf")
    monkeypatch.setenv("RAG_HF_MODEL", "sentence-transformers/does-not-exist-anywhere")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from slimx_rag.embed import reset_embedder_cache

    reset_embedder_cache()
    for res in (
        _index(client, "w", "d", "text"),
        client.post("/api/retrieve", json={"question": "q", "workspace_id": "w"}),
        client.post(
            "/api/index/file",
            data={"workspace_id": "w", "document_id": "f", "filename": "n.md"},
            files={"file": ("n.md", b"# hi", "text/markdown")},
        ),
    ):
        assert res.status_code == 503, res.text
        assert res.json()["detail"]["code"] == "embedder_init_failed"
    reset_embedder_cache()


# --- RAG-AUD-012: demo endpoints ----------------------------------------------------------------


def test_eval_dataset_is_confined_and_model_override_is_opt_in(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _index(client, "w", "d", "SlimX builds explicit inspectable research AI systems.")
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"question": "what?"}\n', encoding="utf-8")
    allowed = tmp_path / "datasets"
    allowed.mkdir()
    inside = allowed / "q.jsonl"
    inside.write_text('{"question": "What does SlimX build?", "expected_sources": []}\n', encoding="utf-8")
    monkeypatch.setenv("RAG_EVAL_DATASET_DIR", str(allowed))
    res = client.post("/api/eval/run", json={"dataset": str(outside)})
    assert res.status_code == 400 and res.json()["detail"]["code"] == "eval_dataset_outside_allowed_dir"
    res = client.post("/api/eval/run", json={"dataset": str(allowed / "missing.jsonl")})
    assert res.status_code == 404
    res = client.post("/api/eval/run", json={"dataset": str(inside), "model": "openai:gpt-4.1"})
    assert res.status_code == 200, res.text  # the caller's model is ignored: fake:grounded ran offline
    res = client.post(
        "/api/ask", json={"question": "What does SlimX build?", "workspace_id": "w", "model": "openai:gpt-4.1"}
    )
    assert res.status_code == 200 and res.json()["model_trace"]["provider"] == "fake"
    monkeypatch.setenv("RAG_ALLOW_MODEL_OVERRIDE", "1")
    res = client.post(
        "/api/ask", json={"question": "What does SlimX build?", "workspace_id": "w", "model": "fake:override"}
    )
    assert res.status_code == 200 and res.json()["model_trace"]["model"] == "fake:override"


# --- author-review corrections (2026-09-25) ---------------------------------------------------


def test_paragraph_boundaries_survive_in_plain_text() -> None:
    from slimx_rag.document.model import ElementType

    text = "\n\n".join(f"Paragraph {i}: the pelican colony on island {i} counted nests." for i in range(4))
    parsed = parse_document(DocumentSource(document_id="x", filename="notes.txt", content=text.encode()))
    paragraphs = [el for el in parsed.elements if el.element_type == ElementType.PARAGRAPH]
    assert len(paragraphs) == 4  # previously one block, force-split at arbitrary word positions
    assert paragraphs[0].text.startswith("Paragraph 0") and paragraphs[3].text.startswith("Paragraph 3")


def test_cp1252_and_utf8_bom_text_are_text_not_binary() -> None:
    from slimx_rag.document.structure import decode_text

    french = "Le r\u00e9glage de l'\u00e9cart de la t\u00eate laser est fix\u00e9 \u00e0 0,42 mm. " * 20
    cp1252 = french.encode("cp1252")
    assert not looks_binary(cp1252)
    assert decode_text(cp1252) == french
    bom = "\ufeffPlain notes".encode()
    assert not looks_binary(bom) and decode_text(bom) == "Plain notes"
    parsed = parse_document(DocumentSource(document_id="x", filename="notes.txt", content=cp1252))
    assert "t\u00eate laser" in parsed.pages[0].text


def test_eval_run_reports_hits_by_source_and_honours_scope(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _index(client, "w", "d", "SlimX builds explicit inspectable research AI systems.")
    _index(client, "other", "d", "Unrelated other-workspace text about pelicans.")
    allowed = tmp_path / "datasets"
    allowed.mkdir()
    dataset = allowed / "q.jsonl"
    dataset.write_text('{"question": "What does SlimX build?", "expected_sources": ["w/d"]}\n', encoding="utf-8")
    monkeypatch.setenv("RAG_EVAL_DATASET_DIR", str(allowed))
    res = client.post("/api/eval/run", json={"dataset": str(dataset), "workspace_id": "w"})
    assert res.status_code == 200, res.text
    case = res.json()["cases"][0]
    assert case["hit"] is True and case["retrieved_sources"][0] == "w/d"  # kb_relpath travels on the hybrid path
    res = client.post("/api/eval/run", json={"dataset": str(dataset), "workspace_id": "other"})
    assert res.json()["cases"][0]["retrieved_sources"] == ["other/d"]
    assert client.post("/api/eval/run", json={"dataset": str(allowed)}).status_code == 422  # a directory
    (allowed / "blank.jsonl").write_text('{"question": "   "}\n', encoding="utf-8")
    res = client.post("/api/eval/run", json={"dataset": str(allowed / "blank.jsonl")})
    assert res.status_code == 422 and res.json()["detail"]["code"] == "eval_dataset_malformed"


def test_embedding_failure_is_a_structured_retryable_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from slimx_rag.embed.embedder import HashEmbedder

    def boom(self, texts):  # noqa: ANN001, ANN202
        raise RuntimeError("provider timed out")

    monkeypatch.setattr(HashEmbedder, "embed_documents", boom)
    monkeypatch.setattr(HashEmbedder, "embed_texts", boom)
    res = _index(client, "w", "d", "text")
    assert res.status_code == 503, res.text
    assert res.json()["detail"]["code"] == "embedding_failed" and res.json()["detail"]["retryable"] is True
    res = client.post(
        "/api/index/file",
        data={"workspace_id": "w", "document_id": "f", "filename": "n.md"},
        files={"file": ("n.md", b"# hi\n\nbody", "text/markdown")},
    )
    assert res.status_code == 503 and res.json()["detail"]["code"] == "embedding_failed"


def test_query_dimension_mismatch_reports_embedding_dim_mismatch(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _index(client, "w", "d", "A healthy document about pelicans.").status_code == 200
    monkeypatch.setenv("RAG_EMBED_DIM", "32")
    from slimx_rag.embed import reset_embedder_cache

    reset_embedder_cache()
    res = client.post("/api/retrieve", json={"question": "pelicans", "workspace_id": "w"})
    assert res.status_code == 503 and res.json()["detail"]["code"] == "embedding_dim_mismatch"


def test_chunk_listing_is_authoritative_after_a_lost_state_commit(client: TestClient, tmp_path: Path) -> None:
    assert (
        _index(client, "w", "doomed", "First paragraph about zebrafish.\n\nSecond paragraph about newts.").status_code
        == 200
    )
    before = client.get("/api/documents/doomed/chunks", params={"workspace_id": "w"}).json()
    _lose_state_commit(tmp_path / "out" / "index_state.json")
    after = client.get("/api/documents/doomed/chunks", params={"workspace_id": "w"}).json()
    assert after["chunk_count"] == before["chunk_count"] > 0
    assert [c["chunk_id"] for c in after["chunks"]] == [c["chunk_id"] for c in before["chunks"]]


def test_identifier_that_lives_only_in_an_empty_heading_stays_reachable(client: TestClient) -> None:
    body = (
        "# Spec\n\n## Module MAX_PAYLOAD_KG\n\n### Overview\nThe module limits the carriage payload.\n\n"
        "## Other\n\n### Overview\nUnrelated overview text.\n"
    )
    res = client.post(
        "/api/index/file",
        data={"workspace_id": "w", "document_id": "spec", "filename": "spec.md", "mime_type": "text/markdown"},
        files={"file": ("spec.md", body, "text/markdown")},
    )
    assert res.status_code == 200, res.text
    chunks = client.post(
        "/api/retrieve", json={"question": "What does MAX_PAYLOAD_KG do?", "workspace_id": "w"}
    ).json()["chunks"]
    top = chunks[0]
    assert top["metadata"]["section_path"] == ["Spec", "Module MAX_PAYLOAD_KG", "Overview"]
    assert top["metadata"]["exact_match"] is True  # matched through the ancestor heading
    listing = client.get("/api/documents/spec/chunks", params={"workspace_id": "w"}).json()["chunks"]
    assert all("Module MAX_PAYLOAD_KG" not in (c["text"] or "") or "Path:" not in c["text"] for c in listing)
