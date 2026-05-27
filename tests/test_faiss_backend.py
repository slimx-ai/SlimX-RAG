from __future__ import annotations

import sys
import types

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
        for vec, fid in zip(vectors, ids):
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
        scored.sort(reverse=True)
        scored = scored[:k]
        while len(scored) < k:
            scored.append((0.0, -1))
        return (
            np.array([[score for score, _fid in scored]], dtype="float32"),
            np.array([[_fid for _score, _fid in scored]], dtype="int64"),
        )


def install_fake_faiss(monkeypatch):
    fake = types.SimpleNamespace(
        IndexFlatIP=FakeIndexFlatIP,
        IndexIDMap2=FakeIndexIDMap2,
        read_index=lambda path: FakeIndexIDMap2(FakeIndexFlatIP(2)),
        write_index=lambda index, path: None,
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

    written = idx.upsert([
        EmbeddedChunk(chunk_id="c1", vector=[1.0, 0.0], text="A", metadata={}),
    ])

    assert written == 1
    assert idx.dim == 2
    assert len(idx) == 1
    assert idx.query([1.0, 0.0], top_k=1)[0].chunk_id == "c1"


def test_faiss_applies_metadata_whitelist(monkeypatch, tmp_path):
    install_fake_faiss(monkeypatch)

    from slimx_rag.index.faiss_backend import FaissIndexBackend

    idx = FaissIndexBackend(
        tmp_path / "index.faiss",
        settings=IndexSettings(backend="faiss", metadata_whitelist=["keep"]),
        state_path=tmp_path / "index_state.json",
    )
    idx.load()

    idx.upsert([
        EmbeddedChunk(chunk_id="c1", vector=[1.0, 0.0], text="A", metadata={"keep": 1, "drop": 2}),
    ])

    assert idx.query([1.0, 0.0], top_k=1)[0].metadata == {"keep": 1}
