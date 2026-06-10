from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from slimx_rag.embed import EmbeddedChunk
from slimx_rag.settings import IndexSettings

from .base import IndexBackend, config_int
from .types import IndexState, SearchResult


class QdrantIndexBackend(IndexBackend):
    """Qdrant backend plugin."""

    def __init__(self, index_path: Path, *, settings: IndexSettings | None = None, state_path: Path | None = None):
        super().__init__(index_path, settings=settings, state_path=state_path)
        cfg = self.settings.backend_config or {}
        self.collection = str(cfg.get("collection") or "").strip()
        if not self.collection:
            raise ValueError("Qdrant backend requires settings.backend_config['collection']")

        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.http import models as qm  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Qdrant backend requires optional dependency. Install with: uv sync --extra qdrant"
            ) from e

        self._qm = qm
        url = str(cfg.get("url") or "http://localhost:6333")
        prefer_grpc = bool(cfg.get("prefer_grpc") or False)
        self.client = QdrantClient(url=url, api_key=cfg.get("api" + "_key"), prefer_grpc=prefer_grpc)
        self.state = IndexState.load(self.state_path)

    def _configured_dim(self) -> int:
        return config_int(self.settings.backend_config, "dim", 0)

    @staticmethod
    def _collection_dim(info: object) -> int | None:
        try:
            return int(info.config.params.vectors.size)  # type: ignore[attr-defined]
        except Exception:
            return None

    def _ensure_collection(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("Qdrant collection dimension must be > 0")
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=self._qm.VectorParams(size=dim, distance=self._qm.Distance.COSINE),
            )
            self._dim = dim
            return
        info = self.client.get_collection(self.collection)
        existing_dim = self._collection_dim(info)
        if existing_dim is not None:
            if existing_dim != dim:
                raise RuntimeError(f"Qdrant collection dim {existing_dim} does not match expected dim {dim}")
            self._dim = existing_dim
        elif self._dim is None:
            self._dim = dim

    def _existing_ids(self, ids: list[str]) -> set[str]:
        if not ids or not self.client.collection_exists(self.collection):
            return set()
        points = self.client.retrieve(
            collection_name=self.collection,
            ids=ids,
            with_payload=False,
            with_vectors=False,
        )
        return {str(p.id) for p in points or []}

    def load(self) -> None:
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            existing_dim = self._collection_dim(info)
            if existing_dim is not None:
                cfg_dim = self._configured_dim()
                if cfg_dim > 0 and cfg_dim != existing_dim:
                    raise RuntimeError(f"Qdrant collection dim {existing_dim} does not match configured dim {cfg_dim}")
                self._dim = existing_dim
        else:
            cfg_dim = self._configured_dim()
            if cfg_dim > 0:
                self._ensure_collection(cfg_dim)
        self.state = IndexState.load(self.state_path)

    def save(self) -> None:
        self._save_state_if_enabled()

    def delete(self, chunk_ids: Iterable[str]) -> int:
        ids = [str(x) for x in chunk_ids if str(x)]
        if not ids:
            return 0
        self.client.delete(collection_name=self.collection, points_selector=self._qm.PointIdsList(points=ids))
        return len(ids)

    def upsert(self, items: Iterable[EmbeddedChunk], *, skip_existing: bool = True) -> int:
        item_list = list(items)
        existing = self._existing_ids([str(it.chunk_id) for it in item_list]) if skip_existing else set()

        written = 0
        points = []
        cfg_dim = self._configured_dim()
        for it in item_list:
            cid = str(it.chunk_id)
            if cid in existing:
                continue

            vector = list(map(float, it.vector))
            actual_dim = len(vector)
            expected_dim = self._dim or cfg_dim or actual_dim
            if self._dim is None:
                self._ensure_collection(expected_dim)
            if actual_dim != int(self._dim or expected_dim):
                raise RuntimeError(f"Vector dim mismatch: expected {self._dim or expected_dim}, got {actual_dim}")
            payload = {"text": it.text, "metadata": self._apply_metadata_whitelist(dict(it.metadata))}
            points.append(self._qm.PointStruct(id=cid, vector=vector, payload=payload))
            written += 1
        if points:
            self.client.upsert(collection_name=self.collection, points=points)
        return written

    def query(self, query_vector: list[float], *, top_k: int | None = None) -> list[SearchResult]:
        if self._dim is not None and len(query_vector) != int(self._dim):
            raise RuntimeError(f"Query vector dim {len(query_vector)} does not match index dim {self._dim}")
        k = int(top_k or self.settings.top_k)
        candidate_k = max(k, k * 4)
        res = self.client.search(
            collection_name=self.collection,
            query_vector=list(map(float, query_vector)),
            limit=candidate_k,
            with_payload=True,
        )
        out: list[SearchResult] = []
        for p in res:
            payload = p.payload or {}
            out.append(SearchResult(
                chunk_id=str(p.id),
                score=float(p.score),
                text=str(payload.get("text") or ""),
                metadata=dict(payload.get("metadata") or {}),
            ))
        return self._sort_results(out, top_k=k)
