from __future__ import annotations

from pathlib import Path

import pytest

from slimx_rag.settings import (
    ChunkSettings,
    EmbedSettings,
    IndexingPipelineSettings,
    IndexSettings,
    IngestSettings,
)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"glob": "   "}, "glob"),
        ({"doc_type_mode": "bogus"}, "doc_type_mode"),
        ({"doc_type_depth": 0}, "doc_type_depth"),
    ],
)
def test_ingest_settings_invalid(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        IngestSettings(**kwargs).validate()


def test_ingest_settings_defaults_valid() -> None:
    IngestSettings().validate()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"chunk_size": 0}, "chunk_size"),
        ({"chunk_overlap": -1}, "chunk_overlap"),
        ({"chunk_size": 100, "chunk_overlap": 100}, "chunk_overlap"),
        ({"separators": ()}, "separators"),
    ],
)
def test_chunk_settings_invalid(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ChunkSettings(**kwargs).validate()


def test_chunk_settings_defaults_valid() -> None:
    ChunkSettings().validate()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"provider": "bogus"}, "provider"),
        ({"batch_size": 0}, "batch_size"),
        ({"retries": 0}, "retries"),
        ({"retry_backoff_s": 0.0}, "retry_backoff_s"),
        ({"provider": "hash", "dim": 0}, "dim"),
        ({"provider": "openai", "model": "  "}, "model"),
        ({"max_chars": 0}, "max_chars"),
        ({"device": "  "}, "device"),
    ],
)
def test_embed_settings_invalid(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        EmbedSettings(**kwargs).validate()


def test_embed_settings_defaults_valid() -> None:
    EmbedSettings().validate()


def test_embed_settings_accepts_device() -> None:
    # A device string is valid; None (default) is also valid and auto-selects.
    EmbedSettings(provider="hf", device="cuda").validate()
    assert EmbedSettings().device is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"backend": "bogus"}, "backend"),
        ({"top_k": 0}, "top_k"),
        ({"write_state": True, "state_filename": "  "}, "state_filename"),
        ({"metadata_whitelist": ["ok", ""]}, "metadata_whitelist"),
        ({"metadata_whitelist": ["  "]}, "metadata_whitelist"),
    ],
)
def test_index_settings_invalid(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        IndexSettings(**kwargs).validate()


def test_index_settings_defaults_valid() -> None:
    IndexSettings().validate()


def test_pipeline_settings_missing_kb_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="kb_dir"):
        IndexingPipelineSettings(kb_dir=tmp_path / "missing", out_dir=tmp_path / "out").validate()


def test_pipeline_settings_out_dir_is_a_file(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    kb.mkdir()
    out_file = tmp_path / "out"
    out_file.write_text("not a dir", encoding="utf-8")
    with pytest.raises(ValueError, match="out_dir"):
        IndexingPipelineSettings(kb_dir=kb, out_dir=out_file).validate()


def test_pipeline_settings_valid(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    kb.mkdir()
    settings = IndexingPipelineSettings(kb_dir=kb, out_dir=tmp_path / "out")
    settings.validate()
    assert settings.docs_path == tmp_path / "out" / "docs.jsonl"
    assert settings.index_state_path == tmp_path / "out" / "index_state.json"
