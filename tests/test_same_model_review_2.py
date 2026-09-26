"""Corrections from the same-model review 2 of the qualification candidate (2026-09-26).

Each test pins one semantic: the document's own title survives a filename title in retrieval
identity (RAG-AUD-039), the admin embedding change keeps the pinned model revision and prefixes
(RAG-AUD-040), a missing workspace never matches the literal "None" scope (RAG-AUD-042), posted
text is bounded and never blank (RAG-AUD-046), and a UTF-8 document with a stray byte stays UTF-8
(RAG-AUD-048).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from slimx_rag.chunk import chunk_parsed_document
from slimx_rag.chunk.tokenizer import HeuristicTokenCounter
from slimx_rag.document import DocumentSource, parse_document
from slimx_rag.document.structure import decode_text
from slimx_rag.retrieval.hybrid import ChunkRecord, HybridRetriever
from slimx_rag.retrieval.tokenize import filename_words, looks_like_filename
from slimx_rag.settings import RetrievalSettings

server = importlib.import_module("slimx_rag.server.app")

_INCIDENT_MD = b"# Incident Report IR-2026-031\n\n## Summary\nOn 2026-03-14 the Atlas gantry stalled in cell 4.\n"


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


# --- RAG-AUD-039: a filename title must not erase the document's own identity ------------------


def test_filename_title_keeps_the_documents_own_title_in_the_identity() -> None:
    src = DocumentSource(
        document_id="inc",
        filename="atlas-incident-2026-03-14.md",
        mime_type="text/markdown",
        content=_INCIDENT_MD,
        metadata={"title": "atlas-incident-2026-03-14.md"},  # ControlRoom sends the upload filename
    )
    doc = parse_document(src)
    assert doc.title == "atlas-incident-2026-03-14.md"
    assert doc.own_title == "Incident Report IR-2026-031"
    chunk = chunk_parsed_document(doc, token_counter=HeuristicTokenCounter(max_tokens=1000))[0]
    assert "Document: atlas-incident-2026-03-14.md" in chunk.embedding_text
    assert "Title: Incident Report IR-2026-031" in chunk.embedding_text
    assert "Name:" not in chunk.embedding_text  # filename words are identity tokens, not prose
    assert chunk.own_title == "Incident Report IR-2026-031"
    assert "Summary" in chunk.display_text and "Document:" not in chunk.display_text
    # A human title (the benchmark's regime) adds no Title/Name lines: nothing changes there.
    human = parse_document(
        DocumentSource(document_id="inc2", filename="x.md", mime_type="text/markdown", content=_INCIDENT_MD)
    )
    assert human.title == "Incident Report IR-2026-031" and human.own_title is None
    assert looks_like_filename("atlas-incident-2026-03-14.docx") and not looks_like_filename("Incident Report")
    # Version-like titles are not filenames: their stems must never become exact identities.
    assert not looks_like_filename("GLM-5.1") and not looks_like_filename("K2.6") and not looks_like_filename("v1.2")
    assert filename_words("atlas-safety-manual.pdf") == ["atlas", "safety", "manual"]
    from slimx_rag.retrieval.tokenize import filename_identity_tokens

    assert "2026" not in filename_identity_tokens("atlas-incident-2026-03-14.docx")
    assert {"atlas-incident-2026-03-14", "incident"} <= filename_identity_tokens("atlas-incident-2026-03-14.docx")


def test_exact_identifier_in_the_documents_own_title_still_boosts_under_a_filename_title() -> None:
    records = {
        "a": ChunkRecord(
            chunk_id="a",
            text="Document: atlas-incident-2026-03-14.docx\nTitle: Incident Report IR-2026-031\n\nSummary text",
            parent_id="A",
            section="Summary",
            source_title="atlas-incident-2026-03-14.docx",
            entry="Summary",
            own_title="Incident Report IR-2026-031",
        ),
        "b": ChunkRecord(
            chunk_id="b",
            text="Document: atlas-safety-manual.pdf\n\nSection text about incidents",
            parent_id="B",
            section="Section 26",
            source_title="atlas-safety-manual.pdf",
            entry="Section 26",
        ),
    }
    retriever = HybridRetriever(dense_search=lambda _q, _k: [("b", 0.9), ("a", 0.8)], get_record=records.get)
    results, _ = retriever.retrieve("What happened in IR-2026-031?", settings=RetrievalSettings())
    assert results[0].chunk_id == "a" and results[0].exact_match is True
    results, _ = retriever.retrieve("What does the atlas-safety-manual say?", settings=RetrievalSettings())
    assert results[0].chunk_id == "b" and results[0].exact_match is True  # filename identifier


# --- RAG-AUD-040: the admin embedding change keeps the pinned revision and prefixes -------------


def test_admin_embedding_change_keeps_the_pinned_revision(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_HF_REVISION", "c9745ed1d9f207416be6d2e6f8de32d1f16199bf")
    monkeypatch.setenv("RAG_EMBED_DOCUMENT_PREFIX", "passage: ")
    server._reset_index_cache()
    res = client.post("/api/admin/embedding", json={"device": "cpu"})
    assert res.status_code == 200, res.text
    override = json.loads(server._embed_override_path().read_text("utf-8"))
    assert "revision" not in override  # the image's RAG_HF_REVISION keeps governing
    assert override["document_prefix"] == "passage: "
    settings = server._embed_settings()
    assert settings.revision == "c9745ed1d9f207416be6d2e6f8de32d1f16199bf" and settings.device == "cpu"
    assert settings.document_prefix == "passage: "
    # Changing the model on a revision-pinned deployment requires the new model's revision.
    res = client.post("/api/admin/embedding", json={"provider": "hf", "hf_model": "org/other-model"})
    assert res.status_code == 422 and "hf_revision" in res.text
    # Mutable refs are rejected at the request boundary; a revision is persisted only when supplied.
    res = client.post("/api/admin/embedding", json={"hf_revision": "main"})
    assert res.status_code == 422
    assert "revision" not in json.loads(server._embed_override_path().read_text("utf-8"))


# --- RAG-AUD-042: a chunk without workspace metadata never matches the literal "None" -----------


def test_missing_workspace_metadata_never_matches_a_literal_none_scope(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    res = client.post("/api/index", json={"workspace_id": "ws", "document_id": "d", "text": "Title\n\nA body line."})
    assert res.status_code == 200
    # Strip the workspace from the stored rows, as a CLI / RAG_KB_DIR build would leave them.
    index_path = Path(os.environ["RAG_INDEX_PATH"])
    rows = [json.loads(line) for line in index_path.read_text("utf-8").splitlines() if line.strip()]
    assert rows
    for row in rows:
        row.get("metadata", {}).pop("workspace_id", None)
    index_path.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")
    server._reset_index_cache()
    monkeypatch.setenv("RAG_REQUIRE_WORKSPACE_SCOPE", "1")
    data = client.post("/api/retrieve", json={"question": "body line", "workspace_id": "None", "top_k": 5}).json()
    assert data["chunks"] == []
    # The same rows are still retrievable without a scope when scope enforcement is off.
    monkeypatch.delenv("RAG_REQUIRE_WORKSPACE_SCOPE", raising=False)
    server._reset_index_cache()
    assert client.post("/api/retrieve", json={"question": "body line", "top_k": 5}).json()["chunks"]


# --- RAG-AUD-046: posted text is bounded and never blank -------------------------------------


def test_posted_text_is_bounded_and_never_blank(client: TestClient) -> None:
    assert (
        client.post("/api/index", json={"workspace_id": "ws", "document_id": "d", "text": "Title\n\nBody."}).status_code
        == 200
    )
    for blank in ("", "   \n\t"):
        res = client.post("/api/index", json={"workspace_id": "ws", "document_id": "d", "text": blank})
        assert res.status_code == 422, blank
    listed = client.get("/api/documents/d/chunks", params={"workspace_id": "ws"}).json()
    assert listed["chunk_count"] >= 1  # the blank posts changed nothing
    assert server.MAX_TEXT_CHARS > 0


# --- RAG-AUD-048: one stray byte does not turn a UTF-8 document into mojibake -------------------


def test_utf8_with_a_stray_byte_stays_utf8() -> None:
    text = "Ünïcödé prose with accents é à ü. " * 20
    content = text.encode("utf-8") + b"\xff" + b" more text"
    decoded = decode_text(content)
    assert "Ünïcödé" in decoded and "é à ü" in decoded
    assert decoded.count("�") == 1
    legacy = "Résumé café".encode("cp1252")
    assert decode_text(legacy) == "Résumé café"


def test_text_document_entry_is_its_own_first_line_under_a_filename_title() -> None:
    src = DocumentSource(
        document_id="log",
        filename="atlas-maintenance-log.txt",
        mime_type="text/plain",
        content=(
            b"Atlas Maintenance Log\n\nLAST SERVICE\n2026-04-01\n\nTECHNICIAN\nJonas Berg\n\nNOTES\nReplaced CC-88.\n"
        ),
        metadata={"title": "atlas-maintenance-log.txt"},
    )
    doc = parse_document(src)
    assert doc.title == "atlas-maintenance-log.txt" and doc.pages[0].title == "Atlas Maintenance Log"
    units = chunk_parsed_document(doc, token_counter=HeuristicTokenCounter(max_tokens=1000))
    assert [u.section for u in units] == ["LAST SERVICE", "TECHNICIAN", "NOTES"]
    assert "Entry: Atlas Maintenance Log" in units[0].embedding_text
    assert "Title:" not in units[0].embedding_text  # the own title already is the entry
    assert units[0].source_title == "atlas-maintenance-log.txt"


def test_sparse_legacy_accents_still_decode_as_cp1252() -> None:
    # A long English cp1252 note with a few accented characters: the accents must survive.
    note = (
        "Operating notes for the line. " * 90
    ) + "Range 20\u201325 \u00b0C, filter 5 \u00b5m, contact Jos\u00e9 M\u00fcller."
    decoded = decode_text(note.encode("cp1252"))
    assert "20\u201325 \u00b0C" in decoded and "Jos\u00e9 M\u00fcller" in decoded and "\ufffd" not in decoded


def test_supplied_revision_is_persisted_and_env_revision_is_not(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_HF_REVISION", "c9745ed1d9f207416be6d2e6f8de32d1f16199bf")
    server._reset_index_cache()
    res = client.post("/api/admin/embedding", json={"hf_revision": "0123456789abcdef0123456789abcdef01234567"})
    assert res.status_code == 200, res.text
    override = json.loads(server._embed_override_path().read_text("utf-8"))
    assert override["revision"] == "0123456789abcdef0123456789abcdef01234567"
    assert server._embed_settings().revision == "0123456789abcdef0123456789abcdef01234567"
