"""Embedder refactor: document/query split, prefixes, dim/seq exposure, cache, tokens."""

from __future__ import annotations

import sys
import types

import pytest
from langchain_core.documents import Document

from slimx_rag.embed import (
    HashEmbedder,
    embed_chunks,
    get_cached_embedder,
    make_embedder,
    reset_embedder_cache,
)
from slimx_rag.settings import EmbedSettings


class _FakeTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()


class _FakeST:
    """SentenceTransformers stand-in WITHOUT encode_query/encode_document (prefix path)."""

    last_inputs: list[str] | None = None
    last_normalize: bool | None = None

    def __init__(self, model: str, device: str | None = None, **kwargs: object) -> None:
        self.tokenizer = _FakeTokenizer()

    def encode(self, texts, normalize_embeddings=False, show_progress_bar=False):  # noqa: ANN001
        _FakeST.last_inputs = list(texts)
        _FakeST.last_normalize = normalize_embeddings
        return [[float(len(t)), 0.0, 1.0] for t in texts]

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def get_max_seq_length(self) -> int:
        return 128


class _FakeST2:
    """Stand-in WITH encode_query/encode_document so we can prove they're preferred."""

    def __init__(self, model: str, device: str | None = None, **kwargs: object) -> None:
        self.tokenizer = _FakeTokenizer()

    def encode(self, texts, **_k):  # noqa: ANN001
        return [[1.0] for _ in texts]

    def encode_document(self, texts, **_k):  # noqa: ANN001
        return [[2.0] for _ in texts]

    def encode_query(self, texts, **_k):  # noqa: ANN001
        return [[3.0] for _ in texts]

    def get_sentence_embedding_dimension(self) -> int:
        return 1

    def get_max_seq_length(self) -> int:
        return 64


def _install(monkeypatch: pytest.MonkeyPatch, st_cls: type) -> None:
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=st_cls)
    )
    reset_embedder_cache()


def test_document_and_query_prefixes_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _FakeST)
    emb = make_embedder(
        EmbedSettings(
            provider="hf", hf_model="m", query_prefix="query: ", document_prefix="passage: "
        )
    )
    emb.embed_documents(["doc text"])
    assert _FakeST.last_inputs == ["passage: doc text"]
    assert _FakeST.last_normalize is True  # normalize_embeddings default True
    emb.embed_query("q")
    assert _FakeST.last_inputs == ["query: q"]


def test_dim_and_max_seq_length_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _FakeST)
    emb = make_embedder(EmbedSettings(provider="hf", hf_model="m"))
    assert emb.dim == 3
    assert emb.max_seq_length == 128


def test_encode_query_document_preferred_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _FakeST2)
    emb = make_embedder(EmbedSettings(provider="hf", hf_model="m"))
    assert emb.embed_documents(["d"]) == [[2.0]]  # encode_document path
    assert emb.embed_query("q") == [3.0]  # encode_query path


def test_token_counter_uses_embedding_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _FakeST)
    counter = make_embedder(EmbedSettings(provider="hf", hf_model="m")).token_counter()
    assert counter.count("a b c d") == 4
    assert counter.max_tokens == 128


def test_hash_token_counter_is_heuristic() -> None:
    counter = HashEmbedder(dim=8).token_counter()
    assert counter.count("one two three") == 3


def test_cache_reuses_instance_and_keys_on_vector_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _FakeST)
    s = EmbedSettings(provider="hf", hf_model="m")
    a = get_cached_embedder(s)
    assert get_cached_embedder(s) is a  # same key -> reused, not reconstructed
    other = get_cached_embedder(EmbedSettings(provider="hf", hf_model="m", normalize_embeddings=False))
    assert other is not a  # a vector-affecting change -> distinct instance
    reset_embedder_cache()
    assert get_cached_embedder(s) is not a  # cleared


def test_embed_chunks_uses_document_path_hash() -> None:
    chunks = [Document(page_content="hello", metadata={"chunk_id": "c1"})]
    out = list(embed_chunks(iter(chunks), settings=EmbedSettings(provider="hash", dim=8)))
    assert len(out) == 1
    assert len(out[0].vector) == 8
    assert out[0].chunk_id == "c1"
