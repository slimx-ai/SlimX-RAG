from __future__ import annotations

import json
from pathlib import Path

from slimx_rag.cli import main
from slimx_rag.diff import build_diff, format_diff_text


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _doc(doc_id: str, content_hash: str, relpath: str) -> dict:
    return {"page_content": relpath, "metadata": {"doc_id": doc_id, "content_hash": content_hash, "kb_relpath": relpath}}


def _chunk(chunk_id: str, doc_id: str) -> dict:
    return {"page_content": chunk_id, "metadata": {"chunk_id": chunk_id, "doc_id": doc_id}}


def test_diff_detects_added_deleted_changed_documents_and_chunks(tmp_path: Path) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    _write_jsonl(old / "docs.jsonl", [_doc("same", "h1", "same.md"), _doc("deleted", "hd", "deleted.md"), _doc("changed", "old", "changed.md")])
    _write_jsonl(new / "docs.jsonl", [_doc("same", "h1", "same.md"), _doc("added", "ha", "added.md"), _doc("changed", "new", "changed.md")])
    _write_jsonl(old / "chunks.jsonl", [_chunk("c-same", "same"), _chunk("c-deleted", "deleted")])
    _write_jsonl(new / "chunks.jsonl", [_chunk("c-same", "same"), _chunk("c-added", "added")])

    result = build_diff(old, new)

    assert result["documents"]["added"] == ["added.md"]
    assert result["documents"]["deleted"] == ["deleted.md"]
    assert result["documents"]["changed"] == ["changed.md"]
    assert result["documents"]["unchanged_count"] == 1
    assert result["chunks"]["added"] == ["c-added"]
    assert result["chunks"]["deleted"] == ["c-deleted"]
    assert result["chunks"]["unchanged_count"] == 1
    assert any("manifest.json missing" in warning for warning in result["warnings"])


def test_diff_uses_manifest_for_config_changes(tmp_path: Path) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    _write_jsonl(old / "docs.jsonl", [])
    _write_jsonl(new / "docs.jsonl", [])
    _write_jsonl(old / "chunks.jsonl", [])
    _write_jsonl(new / "chunks.jsonl", [])
    base = {"chunk_config": {"chunk_size": 800}, "embed_provider": "hash", "embed_model": "m", "embed_dim": 384, "index_backend": "local", "hash_policy": {"algorithm": "blake2b"}}
    old.mkdir(exist_ok=True)
    new.mkdir(exist_ok=True)
    (old / "manifest.json").write_text(json.dumps(base), encoding="utf-8")
    changed = dict(base)
    changed["chunk_config"] = {"chunk_size": 400}
    changed["index_backend"] = "faiss"
    (new / "manifest.json").write_text(json.dumps(changed), encoding="utf-8")

    result = build_diff(old, new)

    assert result["config_changes"]["chunk_config_changed"] is True
    assert result["config_changes"]["index_backend_changed"] is True
    assert result["config_changes"]["embedding_config_changed"] is False


def test_diff_cli_text_and_json_output(tmp_path: Path, capsys) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    _write_jsonl(old / "docs.jsonl", [_doc("d", "old", "doc.md")])
    _write_jsonl(new / "docs.jsonl", [_doc("d", "new", "doc.md")])
    _write_jsonl(old / "chunks.jsonl", [_chunk("c1", "d")])
    _write_jsonl(new / "chunks.jsonl", [_chunk("c1", "d"), _chunk("c2", "d")])

    assert main(["diff", str(old), str(new)]) == 0
    text = capsys.readouterr().out
    assert "Documents changed: 1" in text
    assert "Chunks added: 1" in text
    assert "- doc.md" in text

    assert main(["diff", str(old), str(new), "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema_version"] == "diff-v1"
    assert data["documents"]["changed"] == ["doc.md"]
