from __future__ import annotations

import json
from pathlib import Path

import pytest

from slimx_rag import cli
from slimx_rag.index.local import LocalJsonlIndexBackend
from slimx_rag.settings import EmbedSettings, IndexSettings


def _write_chunks(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _chunk(doc_id: str, chunk_id: str, content_hash: str, text: str) -> dict:
    return {
        "page_content": text,
        "metadata": {"doc_id": doc_id, "chunk_id": chunk_id, "content_hash": content_hash},
    }


def _run_index(tmp_path: Path, chunks_path: Path) -> tuple[int, int, int]:
    return cli._index_chunks_file(
        in_chunks_path=chunks_path,
        embed_settings=EmbedSettings(provider="hash", dim=8),
        index_settings=IndexSettings(backend="local"),
        index_path=tmp_path / "index.jsonl",
        state_path=tmp_path / "index_state.json",
        reindex=False,
    )


def _index_chunk_ids(tmp_path: Path) -> set[str]:
    ids = set()
    for line in (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["chunk_id"])
    return ids


def _state_docs(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "index_state.json").read_text(encoding="utf-8"))["docs"]


def test_unchanged_corpus_is_idempotent(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    _write_chunks(chunks, [_chunk("d1", "c1", "h1", "alpha"), _chunk("d2", "c2", "h2", "beta")])

    deleted, written, total = _run_index(tmp_path, chunks)
    assert (deleted, written, total) == (0, 2, 2)
    first_index = (tmp_path / "index.jsonl").read_bytes()
    first_state = (tmp_path / "index_state.json").read_bytes()

    deleted, written, total = _run_index(tmp_path, chunks)
    assert (deleted, written, total) == (0, 0, 2)
    assert (tmp_path / "index.jsonl").read_bytes() == first_index
    assert (tmp_path / "index_state.json").read_bytes() == first_state


def test_changed_document_replaces_old_chunks(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    _write_chunks(chunks, [_chunk("d1", "c1", "h1", "alpha"), _chunk("d2", "c2", "h2", "beta")])
    _run_index(tmp_path, chunks)

    # d1's content changed: new content_hash, new chunk_id
    _write_chunks(chunks, [_chunk("d1", "c1b", "h1b", "alpha v2"), _chunk("d2", "c2", "h2", "beta")])
    deleted, written, total = _run_index(tmp_path, chunks)

    assert deleted == 1
    assert written == 1
    assert total == 2
    assert _index_chunk_ids(tmp_path) == {"c1b", "c2"}
    assert _state_docs(tmp_path)["d1"] == {"content_hash": "h1b", "chunk_ids": ["c1b"]}


def test_deleted_document_removes_chunks_and_state(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    _write_chunks(chunks, [_chunk("d1", "c1", "h1", "alpha"), _chunk("d2", "c2", "h2", "beta")])
    _run_index(tmp_path, chunks)

    _write_chunks(chunks, [_chunk("d1", "c1", "h1", "alpha")])
    deleted, written, total = _run_index(tmp_path, chunks)

    assert deleted == 1
    assert written == 0
    assert total == 1
    assert _index_chunk_ids(tmp_path) == {"c1"}
    assert set(_state_docs(tmp_path).keys()) == {"d1"}


def test_failed_upsert_leaves_state_untouched_and_rerun_converges(tmp_path: Path, monkeypatch) -> None:
    chunks = tmp_path / "chunks.jsonl"
    _write_chunks(chunks, [_chunk("d1", "c1", "h1", "alpha")])
    _run_index(tmp_path, chunks)
    state_before = (tmp_path / "index_state.json").read_bytes()

    # d1 changes, but the upsert blows up mid-run.
    _write_chunks(chunks, [_chunk("d1", "c1b", "h1b", "alpha v2")])

    def exploding_upsert(self, items, *, skip_existing=True):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(LocalJsonlIndexBackend, "upsert", exploding_upsert)
    with pytest.raises(RuntimeError, match="backend exploded"):
        _run_index(tmp_path, chunks)

    # State was never committed: it still describes the previous successful build.
    assert (tmp_path / "index_state.json").read_bytes() == state_before

    # A re-run with a working backend converges to the correct final state.
    monkeypatch.undo()
    deleted, written, total = _run_index(tmp_path, chunks)
    assert total == 1
    assert _index_chunk_ids(tmp_path) == {"c1b"}
    assert _state_docs(tmp_path) == {"d1": {"content_hash": "h1b", "chunk_ids": ["c1b"]}}


def test_scan_chunks_skips_incomplete_records_and_last_hash_wins(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    _write_chunks(
        chunks,
        [
            {"page_content": "no doc id", "metadata": {"chunk_id": "c-orphan"}},
            {"page_content": "no chunk id", "metadata": {"doc_id": "d-orphan"}},
            _chunk("d1", "c1", "h1", "alpha"),
            _chunk("d1", "c2", "h2", "beta"),
        ],
    )
    out = cli._scan_chunks_for_state(chunks)
    assert out == {"d1": ("h2", ["c1", "c2"])}


def test_scan_chunks_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cli._scan_chunks_for_state(tmp_path / "missing.jsonl")
