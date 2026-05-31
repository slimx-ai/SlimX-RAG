from __future__ import annotations

import json
from pathlib import Path

from slimx_rag.cli import main
from slimx_rag.manifest import build_manifest, write_manifest


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_manifest_is_written_with_counts_and_hash_policy(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "docs.jsonl", [{"page_content": "a", "metadata": {"doc_id": "d1"}}])
    _write_jsonl(
        tmp_path / "chunks.jsonl",
        [
            {"page_content": "a", "metadata": {"chunk_id": "c1", "doc_id": "d1"}},
            {"page_content": "b", "metadata": {"chunk_id": "c2", "doc_id": "d1"}},
        ],
    )

    path = write_manifest(tmp_path, created_at="2026-01-01T00:00:00Z")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / "manifest.json"
    assert data["schema_version"] == "manifest-v1"
    assert data["doc_count"] == 1
    assert data["chunk_count"] == 2
    assert data["files"]["docs_jsonl"]["record_count"] == 1
    assert data["files"]["chunks_jsonl"]["record_count"] == 2
    assert data["hash_policy"]["algorithm"] == "blake2b"


def test_manifest_missing_optional_files_warns_without_crashing(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "docs.jsonl", [{"page_content": "a", "metadata": {}}])

    data = build_manifest(tmp_path)

    assert data["files"]["chunks_jsonl"]["exists"] is False
    assert data["files"]["chunks_jsonl"]["record_count"] == 0
    assert any("chunks.jsonl" in warning for warning in data["warnings"])


def test_manifest_cli_writes_manifest_for_minimal_output_dir(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "docs.jsonl", [{"page_content": "a", "metadata": {}}])

    assert main(["manifest", "--out-dir", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert data["doc_count"] == 1
