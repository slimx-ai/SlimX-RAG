from __future__ import annotations

import json
from pathlib import Path

import pytest

from slimx_rag.chunk import HeuristicTokenCounter
from slimx_rag.cli import (
    _index_chunks_file,
    _parse_backend_config,
    _parse_meta_keep,
    _resolve_state_path,
    main,
)
from slimx_rag.embed import EmbeddedChunk
from slimx_rag.index import INDEX_BUILD_RECEIPT_FILENAME, load_index_build_receipt
from slimx_rag.settings import EmbedSettings, IndexSettings


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
    rc = main(
        [
            "run",
            "--kb-dir",
            str(kb),
            "--out-dir",
            str(tmp_path / "out"),
            "--backend-config",
            "{not json",
        ]
    )
    assert rc == 2


def test_backend_config_non_object_exits_2(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    _write_kb(kb)
    rc = main(
        [
            "run",
            "--kb-dir",
            str(kb),
            "--out-dir",
            str(tmp_path / "out"),
            "--backend-config",
            '["a", "list"]',
        ]
    )
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

    assert (
        _resolve_state_path(args_state=None, index_path=index_path, index_settings=IndexSettings(write_state=False))
        is None
    )

    resolved = _resolve_state_path(args_state=None, index_path=index_path, index_settings=IndexSettings())
    assert resolved == index_path.parent / "index_state.json"


def test_cli_index_persists_emitted_dimension_over_configured_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import slimx_rag.cli as cli

    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        json.dumps(
            {
                "page_content": "text",
                "metadata": {
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "content_hash": "h1",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_embed_chunks(*_args, **_kwargs):
        yield EmbeddedChunk(chunk_id="c1", vector=[1.0, 0.0, 0.0], text="text", metadata={})

    class _FakeHfEmbedder:
        dim = None

        def token_counter(self) -> HeuristicTokenCounter:
            return HeuristicTokenCounter()

    monkeypatch.setattr(cli, "make_embedder", lambda _settings: _FakeHfEmbedder())
    monkeypatch.setattr(cli, "embed_chunks", fake_embed_chunks)
    index_path = tmp_path / "index-volume" / "index.jsonl"
    state_path = tmp_path / "state-volume" / "index_state.json"
    _index_chunks_file(
        in_chunks_path=chunks_path,
        embed_settings=EmbedSettings(provider="hf", dim=384),
        index_settings=IndexSettings(),
        index_path=index_path,
        state_path=state_path,
        reindex=False,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["embed"]["actual_dim"] == 3
    assert state["embed"]["dim"] == 3
    receipt = load_index_build_receipt(index_path.parent / INDEX_BUILD_RECEIPT_FILENAME)
    assert receipt is not None
    assert receipt.index_signature.embedding_dimension == 3
    assert receipt.index_signature.signature_complete is True
    assert receipt.document_pipeline["source"] == "cli"
    assert not (state_path.parent / INDEX_BUILD_RECEIPT_FILENAME).exists()


def test_cli_reindex_rotates_instance_and_replaces_build_receipt(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    out = tmp_path / "out"
    _write_kb(kb)
    args = ["run", "--kb-dir", str(kb), "--out-dir", str(out), "--embed-dim", "8"]

    assert main(args) == 0
    first_identity = (out / "index_instance_id").read_text(encoding="utf-8").strip()
    first_receipt = load_index_build_receipt(out / INDEX_BUILD_RECEIPT_FILENAME)
    assert first_receipt is not None
    assert first_receipt.index_signature.index_instance_id == first_identity
    first_ids = {
        json.loads(line)["chunk_id"]
        for line in (out / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    second_args = [
        *args[:-1],
        "3",
        "--chunk-size",
        "200",
        "--chunk-overlap",
        "20",
        "--reindex",
    ]
    assert main(second_args) == 0
    second_identity = (out / "index_instance_id").read_text(encoding="utf-8").strip()
    second_receipt = load_index_build_receipt(out / INDEX_BUILD_RECEIPT_FILENAME)
    second_ids = {
        json.loads(line)["chunk_id"]
        for line in (out / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    assert second_receipt is not None
    assert second_identity != first_identity
    assert first_ids.isdisjoint(second_ids)
    assert second_ids
    assert second_receipt.index_signature.embedding_dimension == 3
    assert second_receipt.index_signature.index_instance_id == second_identity
    assert second_receipt.index_signature.signature_complete is True
    assert second_receipt.receipt_fingerprint != first_receipt.receipt_fingerprint


def test_cli_remote_reindex_refuses_before_mutating_identity_receipt_or_state(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        json.dumps(
            {
                "page_content": "text",
                "metadata": {"chunk_id": "c1", "doc_id": "d1", "content_hash": "h1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "index.jsonl"
    state_path = tmp_path / "index_state.json"
    _index_chunks_file(
        in_chunks_path=chunks_path,
        embed_settings=EmbedSettings(provider="hash", dim=8),
        index_settings=IndexSettings(backend="local"),
        index_path=index_path,
        state_path=state_path,
        reindex=False,
    )
    identity_path = tmp_path / "index_instance_id"
    receipt_path = tmp_path / INDEX_BUILD_RECEIPT_FILENAME
    before = {
        "index": index_path.read_bytes(),
        "state": state_path.read_bytes(),
        "identity": identity_path.read_bytes(),
        "receipt": receipt_path.read_bytes(),
    }

    with pytest.raises(RuntimeError, match="cannot verify a full reset"):
        _index_chunks_file(
            in_chunks_path=chunks_path,
            embed_settings=EmbedSettings(provider="hash", dim=3),
            index_settings=IndexSettings(
                backend="qdrant",
                backend_config={"url": "https://qdrant.invalid", "collection": "remote"},
            ),
            index_path=index_path,
            state_path=state_path,
            reindex=True,
        )

    assert index_path.read_bytes() == before["index"]
    assert state_path.read_bytes() == before["state"]
    assert identity_path.read_bytes() == before["identity"]
    assert receipt_path.read_bytes() == before["receipt"]
