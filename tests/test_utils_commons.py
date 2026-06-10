from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from slimx_rag import cli
from slimx_rag.index.types import IndexState
from slimx_rag.utils.commons import (
    _atomic_write_lines,
    _atomic_write_text,
    _read_jsonl_docs,
    _write_jsonl,
)


def test_jsonl_round_trip(tmp_path: Path) -> None:
    docs = [
        Document(page_content="hello", metadata={"doc_id": "a", "kb_relpath": "a.md"}),
        Document(page_content="wörld ünïcode", metadata={"doc_id": "b", "kb_relpath": "b.md"}),
    ]
    out = tmp_path / "docs.jsonl"
    _write_jsonl(docs, out)
    loaded = list(_read_jsonl_docs(out))
    assert [d.page_content for d in loaded] == ["hello", "wörld ünïcode"]
    assert [d.metadata["doc_id"] for d in loaded] == ["a", "b"]


def test_read_jsonl_missing_file_raises_with_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(FileNotFoundError, match="nope.jsonl"):
        list(_read_jsonl_docs(missing))


def test_read_jsonl_malformed_line_names_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"page_content": "ok", "metadata": {}}\n{not json}\n', encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        list(_read_jsonl_docs(path))
    assert "bad.jsonl" in str(exc_info.value)
    assert "line 2" in str(exc_info.value)


def test_read_jsonl_non_object_record_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('["a", "list"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        list(_read_jsonl_docs(path))


def test_atomic_write_replaces_and_leaves_no_tmp_files(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    _atomic_write_text(out, "first")
    _atomic_write_text(out, "second")
    assert out.read_text(encoding="utf-8") == "second"
    leftovers = [p for p in tmp_path.iterdir() if p != out]
    assert leftovers == []


def test_atomic_write_preserves_original_when_writer_raises(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    _atomic_write_text(out, "original")

    def exploding_lines():
        yield "partial\n"
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _atomic_write_lines(out, exploding_lines())

    assert out.read_text(encoding="utf-8") == "original"
    leftovers = [p for p in tmp_path.iterdir() if p != out]
    assert leftovers == []


def test_corrupt_index_state_raises_clear_error(tmp_path: Path) -> None:
    state_path = tmp_path / "index_state.json"
    state_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError) as exc_info:
        IndexState.load(state_path)
    msg = str(exc_info.value)
    assert "index_state.json" in msg
    assert "delet" in msg.lower()


def test_cli_missing_input_file_exits_2(tmp_path: Path) -> None:
    rc = cli.main(["chunk", "--in", str(tmp_path / "missing.jsonl"), "--out-dir", str(tmp_path)])
    assert rc == 2
