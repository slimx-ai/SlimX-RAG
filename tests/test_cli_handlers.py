from __future__ import annotations

import json
from pathlib import Path

import pytest

from slimx_rag.cli import _parse_backend_config, _parse_meta_keep, _resolve_state_path, main
from slimx_rag.settings import IndexSettings


def _write_kb(kb: Path) -> None:
    kb.mkdir(parents=True, exist_ok=True)
    (kb / "doc.md").write_text("Some knowledge base content for testing.", encoding="utf-8")


# ---------------------------------------------------------------------------
# Exit-code contract: user-input errors exit 2, never with a traceback
# ---------------------------------------------------------------------------


def test_missing_kb_dir_exits_2(tmp_path: Path) -> None:
    rc = main(["ingest", "--kb-dir", str(tmp_path / "nope"), "--out-dir", str(tmp_path)])
    assert rc == 2


def test_bad_backend_config_json_exits_2(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    _write_kb(kb)
    rc = main([
        "run", "--kb-dir", str(kb), "--out-dir", str(tmp_path / "out"),
        "--backend-config", "{not json",
    ])
    assert rc == 2


def test_backend_config_non_object_exits_2(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    _write_kb(kb)
    rc = main([
        "run", "--kb-dir", str(kb), "--out-dir", str(tmp_path / "out"),
        "--backend-config", '["a", "list"]',
    ])
    assert rc == 2


def test_index_missing_chunks_file_exits_2(tmp_path: Path) -> None:
    rc = main(["index", "--in", str(tmp_path / "missing.jsonl"), "--out-dir", str(tmp_path)])
    assert rc == 2


def test_query_missing_index_exits_2(tmp_path: Path) -> None:
    rc = main(["query", "--out-dir", str(tmp_path), "--q", "anything"])
    assert rc == 2


def test_eval_missing_dataset_exits_2(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    out = tmp_path / "out"
    _write_kb(kb)
    assert main(["run", "--kb-dir", str(kb), "--out-dir", str(out), "--embed-dim", "16"]) == 0
    rc = main(["eval", "--out-dir", str(out), "--dataset", str(tmp_path / "missing.jsonl")])
    assert rc == 2


def test_verbose_flag_is_accepted(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    _write_kb(kb)
    assert main(["--verbose", "run", "--kb-dir", str(kb), "--out-dir", str(tmp_path / "out"), "--embed-dim", "16"]) == 0


# ---------------------------------------------------------------------------
# Individual handlers (happy paths beyond the monolithic demo test)
# ---------------------------------------------------------------------------


def test_ingest_then_chunk_individually(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    out = tmp_path / "out"
    _write_kb(kb)

    assert main(["ingest", "--kb-dir", str(kb), "--out-dir", str(out)]) == 0
    docs_path = out / "docs.jsonl"
    assert docs_path.exists()
    rec = json.loads(docs_path.read_text(encoding="utf-8").splitlines()[0])
    assert rec["metadata"]["kb_relpath"] == "doc.md"

    assert main(["chunk", "--in", str(docs_path), "--out-dir", str(out)]) == 0
    chunks_path = out / "chunks.jsonl"
    assert chunks_path.exists()
    chunk = json.loads(chunks_path.read_text(encoding="utf-8").splitlines()[0])
    assert chunk["metadata"]["chunk_id"]


def test_manifest_diff_and_report_commands(tmp_path: Path, capsys) -> None:
    kb = tmp_path / "kb"
    out = tmp_path / "out"
    _write_kb(kb)
    assert main(["run", "--kb-dir", str(kb), "--out-dir", str(out), "--embed-dim", "16", "--write-manifest"]) == 0
    assert (out / "manifest.json").exists()

    assert main(["manifest", "--out-dir", str(out)]) == 0
    capsys.readouterr()

    assert main(["diff", str(out), str(out)]) == 0
    diff_out = capsys.readouterr().out
    assert "added" in diff_out.lower() or "unchanged" in diff_out.lower()

    assert main(["report", "--out-dir", str(out), "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["summary"]["doc_count"] == 1


# ---------------------------------------------------------------------------
# Helper units
# ---------------------------------------------------------------------------


def test_parse_backend_config() -> None:
    assert _parse_backend_config("") == {}
    assert _parse_backend_config('{"dim": 8}') == {"dim": 8}
    with pytest.raises(ValueError, match="Invalid JSON"):
        _parse_backend_config("{nope")
    with pytest.raises(ValueError, match="JSON object"):
        _parse_backend_config('["list"]')


def test_parse_meta_keep() -> None:
    assert _parse_meta_keep("") is None
    assert _parse_meta_keep(" a, b ,,c ") == ["a", "b", "c"]


def test_resolve_state_path(tmp_path: Path) -> None:
    index_path = tmp_path / "out" / "index.jsonl"

    explicit = tmp_path / "elsewhere" / "state.json"
    assert _resolve_state_path(args_state=explicit, index_path=index_path, index_settings=IndexSettings()) == explicit

    assert _resolve_state_path(
        args_state=None, index_path=index_path, index_settings=IndexSettings(write_state=False)
    ) is None

    resolved = _resolve_state_path(args_state=None, index_path=index_path, index_settings=IndexSettings())
    assert resolved == index_path.parent / "index_state.json"
