from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from slimx_rag.embed import EmbeddedChunk
from slimx_rag.settings import IndexSettings
from .base import IndexBackend
from .types import SearchResult, IndexState

class QdrantIndexBackend(IndexBackend):
    """Qdrant backend plugin (optional dependency).

    Required backend_config keys:
      - collection: str
    Optional keys:
      - url: str (default http://localhost:6333)
      - api_key: str
      - prefer_grpc: bool
    """

    def __init__(self, index_path: Path, *, settings: Optional[IndexSettings] = None, state_path: Optional[Path] = None):
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
        api_key = cfg.get("api_key")
        prefer_grpc = bool(cfg.get("prefer_grpc") or False)

        self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)

        self.state = IndexState.load(self.state_path)

    def load(self) -> None:
        # Ensure collection exists with correct vector size
        dim = self._dim or int((self.state.embed or {}).get("dim") or 0) or int(self.settings.backend_config.get("dim", 0) or 0)
        if dim <= 0:
            # Defer until set_embed_config / first upsert
            return
        self._dim = dim

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=self._qm.VectorParams(size=dim, distance=self._qm.Distance.COSINE),
            )
        else:
            info = self.client.get_collection(self.collection)
            try:
                size = int(info.config.params.vectors.size)  # type: ignore
                self._dim = size
            except Exception:
                pass

        self.state = IndexState.load(self.state_path)

    def save(self) -> None:
        # Remote backend: nothing to persist except local state
        self._save_state_if_enabled()

    def delete(self, chunk_ids: Iterable[str]) -> int:
        ids = [str(x) for x in chunk_ids if str(x)]
        if not ids:
            return 0
        self.client.delete(
            collection_name=self.collection,
            points_selector=self._qm.PointIdsList(points=ids),
        )
        return len(ids)

    def upsert(self, items: Iterable[EmbeddedChunk], *, skip_existing: bool = True) -> int:
        # Qdrant upsert overwrites; emulate skip_existing by checking existing IDs (costly).
        if skip_existing:
            # Best-effort: we can still upsert; Qdrant will overwrite.
            # For strict behavior, you'd batch scroll/retrieve to detect existence.
            pass

        written = 0
        points = []
        dim = self._dim or int((self.state.embed or {}).get("dim") or 0) or int(self.settings.backend_config.get("dim", 0) or 0)
        if dim <= 0:
            raise ValueError("Qdrant backend needs a known embedding dim. Call set_embed_config() or set backend_config['dim'].")
        self._dim = dim

        for it in items:
            if len(it.vector) != dim:
                raise RuntimeError(f"Vector dim mismatch: expected {dim}, got {len(it.vector)}")
            payload = {"text": it.text, "metadata": dict(it.metadata)}
            points.append(self._qm.PointStruct(id=str(it.chunk_id), vector=list(map(float, it.vector)), payload=payload))
            written += 1

        if points:
            self.client.upsert(collection_name=self.collection, points=points)
        return written

    def query(self, query_vector: List[float], *, top_k: Optional[int] = None) -> List[SearchResult]:
        k = int(top_k or self.settings.top_k)
        res = self.client.search(
            collection_name=self.collection,
            query_vector=list(map(float, query_vector)),
            limit=k,
            with_payload=True,
        )
        out: List[SearchResult] = []
        for p in res:
            payload = p.payload or {}
            out.append(
                SearchResult(
                    chunk_id=str(p.id),
                    score=float(p.score),
                    text=str(payload.get("text") or ""),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
        return out
