from __future__ import annotations

import sys
import types

import pytest

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
    instances: list[FakeQdrantClient] = []

    def __init__(self, *, url: str, api_key=None, prefer_grpc: bool = False):
        self.url = url
        self.api_key = api_key
        self.prefer_grpc = prefer_grpc
        self.collections: dict[str, int] = {}
        self.points: dict[str, dict[str, FakePointStruct]] = {}
        self.retrieve_calls: list[list[str]] = []
        self.upsert_calls: list[list[FakePointStruct]] = []
        self.fail_connect = False
        FakeQdrantClient.instances.append(self)

    def collection_exists(self, collection_name: str) -> bool:
        if self.fail_connect:
            raise ConnectionError("connection refused")
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
        for pid in ids:
            self._require_point_id(pid)
        store = self.points.get(collection_name, {})
        by_point = {str(p.id): p for p in store.values()}
        return [types.SimpleNamespace(id=pid) for pid in ids if str(pid) in by_point]

    @staticmethod
    def _require_point_id(value: object) -> None:
        # The real client accepts only UUIDs or unsigned integers as point ids.
        import uuid

        if isinstance(value, int) and not isinstance(value, bool):
            return
        try:
            uuid.UUID(str(value))
        except ValueError as exc:
            raise ValueError(f"Point id {value} is not a valid UUID") from exc

    def upsert(self, *, collection_name: str, points: list[FakePointStruct]):
        if collection_name not in self.collections:
            raise RuntimeError("collection was not created")
        self.upsert_calls.append(list(points))
        store = self.points.setdefault(collection_name, {})
        for p in points:
            self._require_point_id(p.id)
            # Keyed by the chunk id carried in the payload (what the tests and callers reason about).
            store[str(p.payload.get("chunk_id") or p.id)] = p

    def delete(self, *, collection_name: str, points_selector: FakePointIdsList):
        store = self.points.setdefault(collection_name, {})
        for pid in points_selector.points:
            self._require_point_id(pid)
            for cid, p in list(store.items()):
                if str(p.id) == str(pid):
                    store.pop(cid, None)

    # qdrant-client removed ``QdrantClient.search`` in 1.15; the fake deliberately offers only the
    # Universal Query API so a regression to ``search`` fails here instead of against a live server.
    def query_points(self, *, collection_name: str, query: list[float], limit: int, with_payload: bool):
        store = self.points.get(collection_name, {})
        out = []
        for p in store.values():
            score = sum(a * b for a, b in zip(query, p.vector, strict=False))
            out.append(types.SimpleNamespace(id=p.id, score=score, payload=p.payload))
        # Intentionally reverse tie order to prove backend code applies chunk_id ascending.
        out.sort(key=lambda p: (p.score, p.id), reverse=True)
        return types.SimpleNamespace(points=out[:limit])


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

    with pytest.raises(RuntimeError, match="Vector dim mismatch"):
        idx.upsert([
            EmbeddedChunk(chunk_id="bad", vector=[1.0, 0.0, 0.0], text="bad", metadata={}),
        ])


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


def test_qdrant_query_orders_equal_scores_by_chunk_id(monkeypatch, tmp_path):
    install_fake_qdrant(monkeypatch)

    from slimx_rag.index.qdrant_backend import QdrantIndexBackend

    idx = QdrantIndexBackend(
        tmp_path / "unused.index",
        settings=IndexSettings(backend="qdrant", backend_config={"collection": "slimx"}, top_k=3),
        state_path=tmp_path / "index_state.json",
    )
    idx.load()

    idx.upsert([
        EmbeddedChunk(chunk_id="c3", vector=[1.0, 0.0], text="C", metadata={}),
        EmbeddedChunk(chunk_id="c1", vector=[1.0, 0.0], text="A", metadata={}),
        EmbeddedChunk(chunk_id="c2", vector=[1.0, 0.0], text="B", metadata={}),
    ])

    assert [r.chunk_id for r in idx.query([1.0, 0.0], top_k=3)] == ["c1", "c2", "c3"]


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
    from slimx_rag.index.qdrant_backend import point_id_for

    assert client.retrieve_calls[-1] == [point_id_for("c1"), point_id_for("c2")]
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


def test_qdrant_upsert_and_retrieve_are_batched(monkeypatch, tmp_path):
    install_fake_qdrant(monkeypatch)

    from slimx_rag.index.qdrant_backend import QdrantIndexBackend

    idx = QdrantIndexBackend(
        tmp_path / "unused.index",
        settings=IndexSettings(
            backend="qdrant",
            backend_config={"collection": "slimx", "dim": 2, "batch_size": 2},
        ),
        state_path=tmp_path / "index_state.json",
    )
    idx.load()

    written = idx.upsert([
        EmbeddedChunk(chunk_id=f"c{i}", vector=[1.0, 0.0], text=f"t{i}", metadata={}) for i in range(5)
    ])
    client = FakeQdrantClient.instances[-1]

    assert written == 5
    assert [len(points) for points in client.upsert_calls] == [2, 2, 1]
    assert [len(ids) for ids in client.retrieve_calls] == [2, 2, 1]
    assert len(client.points["slimx"]) == 5


def test_qdrant_connection_failure_raises_friendly_error(monkeypatch, tmp_path):
    install_fake_qdrant(monkeypatch)

    from slimx_rag.index.qdrant_backend import QdrantIndexBackend

    idx = QdrantIndexBackend(
        tmp_path / "unused.index",
        settings=IndexSettings(backend="qdrant", backend_config={"collection": "slimx"}),
        state_path=tmp_path / "index_state.json",
    )
    FakeQdrantClient.instances[-1].fail_connect = True

    with pytest.raises(RuntimeError, match="Could not reach Qdrant"):
        idx.load()


def test_qdrant_api_key_is_passed_as_a_string_or_omitted(monkeypatch, tmp_path):
    install_fake_qdrant(monkeypatch)

    from slimx_rag.index.qdrant_backend import QdrantIndexBackend

    QdrantIndexBackend(
        tmp_path / "unused.index",
        settings=IndexSettings(backend="qdrant", backend_config={"collection": "slimx"}),
        state_path=tmp_path / "index_state.json",
    )
    assert FakeQdrantClient.instances[-1].api_key is None

    QdrantIndexBackend(
        tmp_path / "unused.index",
        settings=IndexSettings(backend="qdrant", backend_config={"collection": "slimx", "api" + "_key": "k"}),
        state_path=tmp_path / "index_state.json",
    )
    assert FakeQdrantClient.instances[-1].api_key == "k"


def test_qdrant_maps_chunk_ids_to_uuid_point_ids_against_the_real_client(monkeypatch, tmp_path):
    """Qdrant accepts only UUID or integer point ids; 64-hex chunk ids must be mapped (RAG-AUD-041)."""
    qdrant_client = pytest.importorskip("qdrant_client")

    real = qdrant_client.QdrantClient

    class MemoryClient(real):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            super().__init__(":memory:")

    monkeypatch.setattr(qdrant_client, "QdrantClient", MemoryClient)

    from slimx_rag.index.qdrant_backend import QdrantIndexBackend, point_id_for

    idx = QdrantIndexBackend(
        tmp_path / "unused.index",
        settings=IndexSettings(backend="qdrant", backend_config={"collection": "real"}),
        state_path=tmp_path / "index_state.json",
    )
    idx.load()
    idx.set_embed_config(EmbedSettings(provider="hash", dim=3))
    cid = "ef" * 32
    assert idx.upsert([EmbeddedChunk(chunk_id=cid, vector=[1.0, 0.0, 0.0], text="A", metadata={"k": 1})]) == 1
    # skip_existing recognises the mapped point; the query returns the chunk id, not the point id.
    assert idx.upsert([EmbeddedChunk(chunk_id=cid, vector=[1.0, 0.0, 0.0], text="A2", metadata={})]) == 0
    hit = idx.query([1.0, 0.0, 0.0], top_k=1)[0]
    assert hit.chunk_id == cid and hit.text == "A" and hit.metadata == {"k": 1}
    assert point_id_for(cid) != cid
    assert idx.delete([cid]) == 1
    assert idx.query([1.0, 0.0, 0.0], top_k=1) == []
