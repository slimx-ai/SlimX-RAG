from __future__ import annotations

import sys
import types

from slimx_rag.embed import EmbeddedChunk
from slimx_rag.settings import EmbedSettings, IndexSettings


class FakeDistance:
    COSINE = "Cosine"


class FakeVectorParams:
    def __init__(self, *, size: int, distance: str):
        self.size = size
        self.distance = distance


class FakePointStruct:
    def __init__(self, *, id: str, vector: list[float], payload: dict):
        self.id = id
        self.vector = vector
        self.payload = payload


class FakePointIdsList:
    def __init__(self, *, points: list[str]):
        self.points = points


class FakeQdrantClient:
    instances: list["FakeQdrantClient"] = []

    def __init__(self, *, url: str, api_key=None, prefer_grpc: bool = False):
        self.url = url
        self.api_key = api_key
        self.prefer_grpc = prefer_grpc
        self.collections: dict[str, int] = {}
        self.points: dict[str, dict[str, FakePointStruct]] = {}
        self.retrieve_calls: list[list[str]] = []
        FakeQdrantClient.instances.append(self)

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, *, collection_name: str, vectors_config: FakeVectorParams):
        self.collections[collection_name] = int(vectors_config.size)
        self.points.setdefault(collection_name, {})

    def get_collection(self, collection_name: str):
        size = self.collections[collection_name]
        return types.SimpleNamespace(
            config=types.SimpleNamespace(
                params=types.SimpleNamespace(
                    vectors=types.SimpleNamespace(size=size),
                )
            )
        )

    def retrieve(self, *, collection_name: str, ids: list[str], with_payload: bool, with_vectors: bool):
        self.retrieve_calls.append(list(ids))
        store = self.points.get(collection_name, {})
        return [types.SimpleNamespace(id=pid) for pid in ids if str(pid) in store]

    def upsert(self, *, collection_name: str, points: list[FakePointStruct]):
        if collection_name not in self.collections:
            raise RuntimeError("collection was not created")
        store = self.points.setdefault(collection_name, {})
        for p in points:
            store[str(p.id)] = p

    def delete(self, *, collection_name: str, points_selector: FakePointIdsList):
        store = self.points.setdefault(collection_name, {})
        for p in points_selector.points:
            store.pop(str(p), None)

    def search(self, *, collection_name: str, query_vector: list[float], limit: int, with_payload: bool):
        store = self.points.get(collection_name, {})
        out = []
        for p in store.values():
            score = sum(a * b for a, b in zip(query_vector, p.vector))
            out.append(types.SimpleNamespace(id=p.id, score=score, payload=p.payload))
        out.sort(key=lambda p: p.score, reverse=True)
        return out[:limit]


def install_fake_qdrant(monkeypatch):
    FakeQdrantClient.instances.clear()

    qdrant_client = types.ModuleType("qdrant_client")
    qdrant_client.QdrantClient = FakeQdrantClient

    http = types.ModuleType("qdrant_client.http")
    models = types.ModuleType("qdrant_client.http.models")
    models.Distance = FakeDistance
    models.VectorParams = FakeVectorParams
    models.PointStruct = FakePointStruct
    models.PointIdsList = FakePointIdsList

    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_client)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", http)
    monkeypatch.setitem(sys.modules, "qdrant_client.http.models", models)


def test_qdrant_first_upsert_creates_collection_from_vector_dim(monkeypatch, tmp_path):
    install_fake_qdrant(monkeypatch)

    from slimx_rag.index.qdrant_backend import QdrantIndexBackend

    idx = QdrantIndexBackend(
        tmp_path / "unused.index",
        settings=IndexSettings(backend="qdrant", backend_config={"collection": "slimx"}),
        state_path=tmp_path / "index_state.json",
    )
    idx.load()
    idx.set_embed_config(EmbedSettings(provider="hash", dim=384))

    assert idx.dim is None
    client = FakeQdrantClient.instances[-1]
    assert client.collections == {}

    written = idx.upsert([
        EmbeddedChunk(chunk_id="c1", vector=[1.0, 0.0, 0.0], text="A", metadata={"keep": 1}),
    ])

    assert written == 1
    assert idx.dim == 3
    assert client.collections == {"slimx": 3}
    assert idx.query([1.0, 0.0, 0.0], top_k=1)[0].chunk_id == "c1"


def test_qdrant_configured_dim_is_enforced_on_first_upsert(monkeypatch, tmp_path):
    install_fake_qdrant(monkeypatch)

    from slimx_rag.index.qdrant_backend import QdrantIndexBackend

    idx = QdrantIndexBackend(
        tmp_path / "unused.index",
        settings=IndexSettings(backend="qdrant", backend_config={"collection": "slimx", "dim": 2}),
        state_path=tmp_path / "index_state.json",
    )
    idx.load()

    client = FakeQdrantClient.instances[-1]
    assert idx.dim == 2
    assert client.collections == {"slimx": 2}

    try:
        idx.upsert([
            EmbeddedChunk(chunk_id="bad", vector=[1.0, 0.0, 0.0], text="bad", metadata={}),
        ])
        assert False, "expected dimension mismatch"
    except RuntimeError as e:
        assert "Vector dim mismatch" in str(e)


def test_qdrant_applies_metadata_whitelist(monkeypatch, tmp_path):
    install_fake_qdrant(monkeypatch)

    from slimx_rag.index.qdrant_backend import QdrantIndexBackend

    idx = QdrantIndexBackend(
        tmp_path / "unused.index",
        settings=IndexSettings(
            backend="qdrant",
            backend_config={"collection": "slimx"},
            metadata_whitelist=["keep"],
        ),
        state_path=tmp_path / "index_state.json",
    )
    idx.load()

    idx.upsert([
        EmbeddedChunk(chunk_id="c1", vector=[1.0, 0.0], text="A", metadata={"keep": 1, "drop": 2}),
    ])

    assert idx.query([1.0, 0.0], top_k=1)[0].metadata == {"keep": 1}


def test_qdrant_skip_existing_does_not_overwrite_existing_points(monkeypatch, tmp_path):
    install_fake_qdrant(monkeypatch)

    from slimx_rag.index.qdrant_backend import QdrantIndexBackend

    idx = QdrantIndexBackend(
        tmp_path / "unused.index",
        settings=IndexSettings(backend="qdrant", backend_config={"collection": "slimx"}),
        state_path=tmp_path / "index_state.json",
    )
    idx.load()

    assert idx.upsert([
        EmbeddedChunk(chunk_id="c1", vector=[1.0, 0.0], text="old", metadata={"version": "old"}),
    ]) == 1

    client = FakeQdrantClient.instances[-1]
    assert client.points["slimx"]["c1"].payload["text"] == "old"

    written = idx.upsert([
        EmbeddedChunk(chunk_id="c1", vector=[0.0, 1.0], text="new", metadata={"version": "new"}),
        EmbeddedChunk(chunk_id="c2", vector=[0.0, 1.0], text="second", metadata={"version": "second"}),
    ], skip_existing=True)

    assert written == 1
    assert client.retrieve_calls[-1] == ["c1", "c2"]
    assert client.points["slimx"]["c1"].payload["text"] == "old"
    assert client.points["slimx"]["c2"].payload["text"] == "second"


def test_qdrant_skip_existing_false_overwrites_existing_points(monkeypatch, tmp_path):
    install_fake_qdrant(monkeypatch)

    from slimx_rag.index.qdrant_backend import QdrantIndexBackend

    idx = QdrantIndexBackend(
        tmp_path / "unused.index",
        settings=IndexSettings(backend="qdrant", backend_config={"collection": "slimx"}),
        state_path=tmp_path / "index_state.json",
    )
    idx.load()

    idx.upsert([
        EmbeddedChunk(chunk_id="c1", vector=[1.0, 0.0], text="old", metadata={}),
    ])

    written = idx.upsert([
        EmbeddedChunk(chunk_id="c1", vector=[0.0, 1.0], text="new", metadata={}),
    ], skip_existing=False)

    client = FakeQdrantClient.instances[-1]
    assert written == 1
    assert client.points["slimx"]["c1"].payload["text"] == "new"
