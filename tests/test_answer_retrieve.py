from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from slimx_rag.answer import answer, build_grounded_prompt
from slimx_rag.embed import EmbeddedChunk
from slimx_rag.index import make_index_backend
from slimx_rag.retrieve import retrieve
from slimx_rag.settings import EmbedSettings, IndexSettings


def test_retrieve_returns_citation_metadata(tmp_path: Path):
    index_path = tmp_path / "index.jsonl"
    idx = make_index_backend(index_path, settings=IndexSettings(backend="local"))
    idx.load()
    idx.set_embed_config(EmbedSettings(provider="hash", dim=8))
    idx.upsert([
        EmbeddedChunk(
            chunk_id="c1",
            vector=[1.0] * 8,
            text="SlimX is explicit and inspectable.",
            metadata={"kb_relpath": "docs/slimx.md", "chunk_index": 0},
        )
    ])
    idx.save()

    result = retrieve(
        "SlimX",
        index_path=index_path,
        embed_settings=EmbedSettings(provider="hash", dim=8),
        index_settings=IndexSettings(backend="local"),
    )

    assert result.query == "SlimX"
    assert result.chunks[0].citation == "[docs/slimx.md:0]"
    assert result.elapsed_ms >= 0


def test_answer_fake_model_warns_when_missing_citation(tmp_path: Path):
    index_path = tmp_path / "index.jsonl"
    idx = make_index_backend(index_path, settings=IndexSettings(backend="local"))
    idx.load()
    idx.set_embed_config(EmbedSettings(provider="hash", dim=4))
    idx.upsert([
        EmbeddedChunk(
            chunk_id="c1",
            vector=[1.0, 0.0, 0.0, 0.0],
            text="Research assistants need citations.",
            metadata={"kb_relpath": "research.md", "chunk_index": 1},
        )
    ])
    idx.save()
    retrieval = retrieve(
        "citations",
        index_path=index_path,
        embed_settings=EmbedSettings(provider="hash", dim=4),
        index_settings=IndexSettings(backend="local"),
    )

    result = answer("citations", retrieval, model="fake:grounded")

    assert "[research.md:1]" in result.answer
    assert result.citations == ["[research.md:1]"]
    assert result.retrieval["query"] == "citations"


def test_grounded_prompt_contains_context_and_instruction():
    retrieval = type("R", (), {
        "chunks": [
            type("C", (), {"citation": "[a.md:0]", "score": 0.9, "text": "Alpha"})()
        ]
    })()

    prompt = build_grounded_prompt("What?", retrieval)  # type: ignore[arg-type]

    assert "Answer only from the retrieved context" in prompt
    assert "[a.md:0]" in prompt
    assert "Alpha" in prompt
