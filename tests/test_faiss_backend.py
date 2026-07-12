from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from slimx_rag.embed import EmbeddedChunk
from slimx_rag.settings import EmbedSettings, IndexSettings


class FakeIndexFlatIP:
    def __init__(self, dim: int):
        self.d = dim


class FakeIndexIDMap2:
    def __init__(self, base: FakeIndexFlatIP):
        self.d = base.d
        self.ntotal = 0
        self.rows: dict[int, np.ndarray] = {}

    def add_with_ids(self, vectors, ids):
        for vec, fid in zip(vectors, ids, strict=False):
            self.rows[int(fid)] = np.array(vec, dtype="float32")
            self.ntotal += 1

    def remove_ids(self, ids):
        for fid in ids:
            if int(fid) in self.rows:
                self.rows.pop(int(fid))
                self.ntotal -= 1

    def search(self, q, k: int):
        scored = []
        for fid, vec in self.rows.items():
            scored.append((float(np.dot(q[0], vec)), fid))
        # Intentionally sort ties by ID descending to prove backend code
        # re-sorts final SearchResult objects by chunk_id ascending.
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        scored = scored[:k]
        while len(scored) < k:
            scored.append((0.0, -1))
        return (
            np.array([[score for score, _fid in scored]], dtype="float32"),
            np.array([[_fid for _score, _fid in scored]], dtype="int64"),
        )


def install_fake_faiss(monkeypatch):
    stored: dict[str, FakeIndexIDMap2] = {}

    def write_index(index: FakeIndexIDMap2, path: str) -> None:
        stored[path] = index
        Path(path).write_bytes(b"fake-faiss")

    fake = types.SimpleNamespace(
        IndexFlatIP=FakeIndexFlatIP,
        IndexIDMap2=FakeIndexIDMap2,
        read_index=lambda path: stored[path],
        write_index=write_index,
    )
    monkeypatch.setitem(sys.modules, "faiss", fake)


def test_faiss_first_upsert_creates_index_after_embed_config(monkeypatch, tmp_path):
    install_fake_faiss(monkeypatch)

    from slimx_rag.index.faiss_backend import FaissIndexBackend

    idx = FaissIndexBackend(
        tmp_path / "index.faiss",
        settings=IndexSettings(backend="faiss"),
        state_path=tmp_path / "index_state.json",
    )
    idx.load()
    assert idx.dim is None
    assert len(idx) == 0

    idx.set_embed_config(EmbedSettings(provider="hash", dim=2))
    assert idx.dim is None

    written = idx.upsert(
        [
            EmbeddedChunk(chunk_id="c1", vector=[1.0, 0.0], text="A", metadata={}),
        ]
    )

    assert written == 1
    assert idx.dim == 2
    assert len(idx) == 1
    assert idx.query([1.0, 0.0], top_k=1)[0].chunk_id == "c1"


def test_faiss_empty_corpus_can_rebuild_at_a_new_dimension(monkeypatch, tmp_path):
    install_fake_faiss(monkeypatch)

    from slimx_rag.index.faiss_backend import FaissIndexBackend

    idx = FaissIndexBackend(
        tmp_path / "index.faiss",
        settings=IndexSettings(backend="faiss"),
        state_path=tmp_path / "index_state.json",
    )
    idx.load()
    idx.upsert(
        [
            EmbeddedChunk(chunk_id="old", vector=[1.0, 0.0], text="old", metadata={}),
        ]
    )

    assert idx.delete(["old"]) == 1
    assert idx.dim is None
    idx.upsert(
        [
            EmbeddedChunk(chunk_id="new", vector=[1.0, 0.0, 0.0], text="new", metadata={}),
        ]
    )

    assert idx.dim == 3
    assert idx.query([1.0, 0.0, 0.0], top_k=1)[0].chunk_id == "new"


def test_faiss_single_item_overwrite_survives_restart(monkeypatch, tmp_path):
    install_fake_faiss(monkeypatch)

    from slimx_rag.index.faiss_backend import FaissIndexBackend

    index_path = tmp_path / "index.faiss"
    settings = IndexSettings(backend="faiss")
    state_path = tmp_path / "index_state.json"
    idx = FaissIndexBackend(index_path, settings=settings, state_path=state_path)
    idx.load()
    idx.upsert(
        [
            EmbeddedChunk(chunk_id="only", vector=[1.0, 0.0], text="old", metadata={}),
        ]
    )

    assert (
        idx.upsert(
            [EmbeddedChunk(chunk_id="only", vector=[0.0, 1.0], text="new", metadata={})],
            skip_existing=False,
        )
        == 1
    )
    assert len(idx) == 1
    assert idx.query([0.0, 1.0], top_k=1)[0].text == "new"
    idx.save()

    restarted = FaissIndexBackend(index_path, settings=settings, state_path=state_path)
    restarted.load()

    assert len(restarted) == 1
    result = restarted.query([0.0, 1.0], top_k=1)[0]
    assert result.chunk_id == "only"
    assert result.text == "new"


def test_faiss_applies_metadata_whitelist(monkeypatch, tmp_path):
    install_fake_faiss(monkeypatch)

    from slimx_rag.index.faiss_backend import FaissIndexBackend

    idx = FaissIndexBackend(
        tmp_path / "index.faiss",
        settings=IndexSettings(backend="faiss", metadata_whitelist=["keep"]),
        state_path=tmp_path / "index_state.json",
    )
    idx.load()

    idx.upsert(
        [
            EmbeddedChunk(chunk_id="c1", vector=[1.0, 0.0], text="A", metadata={"keep": 1, "drop": 2}),
        ]
    )

    assert idx.query([1.0, 0.0], top_k=1)[0].metadata == {"keep": 1}


def test_faiss_query_orders_equal_scores_by_chunk_id(monkeypatch, tmp_path):
    install_fake_faiss(monkeypatch)

    from slimx_rag.index.faiss_backend import FaissIndexBackend

    idx = FaissIndexBackend(
        tmp_path / "index.faiss",
        settings=IndexSettings(backend="faiss", top_k=3),
        state_path=tmp_path / "index_state.json",
    )
    idx.load()

    idx.upsert(
        [
            EmbeddedChunk(chunk_id="c3", vector=[1.0, 0.0], text="C", metadata={}),
            EmbeddedChunk(chunk_id="c1", vector=[1.0, 0.0], text="A", metadata={}),
            EmbeddedChunk(chunk_id="c2", vector=[1.0, 0.0], text="B", metadata={}),
        ]
    )

    assert [r.chunk_id for r in idx.query([1.0, 0.0], top_k=3)] == ["c1", "c2", "c3"]
