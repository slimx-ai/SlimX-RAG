from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from slimx_rag.embed import EmbeddedChunk
from slimx_rag.settings import IndexSettings

from .base import IndexBackend
from .types import SearchResult, IndexState


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    # arr: (n, d) or (d,)
    denom = np.linalg.norm(arr, axis=-1, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    return arr / denom


class FaissIndexBackend(IndexBackend):
    """FAISS backend plugin (optional dependency).

    Notes:
      - Uses cosine similarity by normalizing vectors and searching via inner product (IndexFlatIP).
      - Stores payload (text+metadata) in a JSONL sidecar next to the FAISS index file.
      - Supports delete via IndexIDMap2.remove_ids (works for this index type).
    """

    def __init__(self, index_path: Path, *, settings: Optional[IndexSettings] = None, state_path: Optional[Path] = None):
        super().__init__(index_path, settings=settings, state_path=state_path)
        try:
            import faiss  # type: ignore
        except ImportError as e:
            raise ImportError(
                "FAISS backend requires optional dependency. Install with: uv sync --extra faiss"
            ) from e

        self._faiss = faiss  # module
        self._index = None  # faiss index
        self._chunk_to_id: Dict[str, int] = {}
        self._id_to_chunk: Dict[int, str] = {}
        self._payload: Dict[str, Tuple[str, dict[str, object]]] = {}
        self._next_id: int = 1

        # reload state using shared type
        self.state = IndexState.load(self.state_path)

    @property
    def _meta_path(self) -> Path:
        return self.index_path.with_suffix(self.index_path.suffix + ".meta.json")

    def __len__(self) -> int:
        if self._index is None:
            return 0
        return int(self._index.ntotal)

    def load(self) -> None:
        # Load FAISS index
        if self.index_path.exists():
            self._index = self._faiss.read_index(str(self.index_path))
            # infer dim
            try:
                self._dim = int(self._index.d)
            except Exception:
                pass
        else:
            # create empty if possible
            dim = self._dim or int((self.state.embed or {}).get("dim") or 0) or int(self.settings.backend_config.get("dim", 0) or 0)
            if dim <= 0:
                # Defer creation until first upsert/set_embed_config
                self._index = None
                return
            self._dim = dim
            base = self._faiss.IndexFlatIP(dim)
            self._index = self._faiss.IndexIDMap2(base)

        # Load sidecar metadata
        self._chunk_to_id.clear()
        self._id_to_chunk.clear()
        self._payload.clear()
        self._next_id = 1

        if self._meta_path.exists():
            data = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._next_id = int(data.get("next_id", 1))
            self._chunk_to_id = {str(k): int(v) for k, v in (data.get("chunk_to_id") or {}).items()}
            self._id_to_chunk = {int(k): str(v) for k, v in (data.get("id_to_chunk") or {}).items()}
            self._payload = {str(k): (v.get("text", "") or "", dict(v.get("metadata") or {})) for k, v in (data.get("payload") or {}).items()}

        # Reload state
        self.state = IndexState.load(self.state_path)

    def save(self) -> None:
        if self._index is None:
            # nothing to save yet
            self._save_state_if_enabled()
            return

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(self.index_path))

        meta = {
            "next_id": self._next_id,
            "chunk_to_id": self._chunk_to_id,
            "id_to_chunk": self._id_to_chunk,
            "payload": {k: {"text": v[0], "metadata": v[1]} for k, v in self._payload.items()},
        }
        self._meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        self._save_state_if_enabled()

    def _ensure_index(self, dim: int) -> None:
        if self._index is not None:
            return
        base = self._faiss.IndexFlatIP(dim)
        self._index = self._faiss.IndexIDMap2(base)
        self._dim = dim

    def delete(self, chunk_ids: Iterable[str]) -> int:
        if self._index is None:
            return 0

        ids: List[int] = []
        for cid in chunk_ids:
            i = self._chunk_to_id.get(cid)
            if i is None:
                continue
            ids.append(int(i))

        if not ids:
            return 0

        id_arr = np.array(ids, dtype="int64")
        self._index.remove_ids(id_arr)

        deleted = 0
        for i in ids:
            cid = self._id_to_chunk.pop(int(i), None)
            if cid is None:
                continue
            self._chunk_to_id.pop(cid, None)
            self._payload.pop(cid, None)
            deleted += 1
        return deleted

    def upsert(self, items: Iterable[EmbeddedChunk], *, skip_existing: bool = True) -> int:
        written = 0
        for it in items:
            cid = it.chunk_id
            vec = np.array(it.vector, dtype="float32")
            if vec.ndim != 1:
                raise ValueError("vector must be 1D")

            if self._dim is None:
                self._ensure_index(int(vec.shape[0]))
            elif int(vec.shape[0]) != int(self._dim):
                raise RuntimeError(f"Vector dim mismatch: index dim {self._dim} vs item dim {int(vec.shape[0])}")

            assert self._index is not None
            vec_n = _l2_normalize(vec).reshape(1, -1)

            existing_id = self._chunk_to_id.get(cid)
            if existing_id is not None:
                if skip_existing:
                    continue
                # overwrite: delete then re-add
                self.delete([cid])

            faiss_id = self._next_id
            self._next_id += 1

            id_arr = np.array([faiss_id], dtype="int64")
            self._index.add_with_ids(vec_n, id_arr)

            self._chunk_to_id[cid] = faiss_id
            self._id_to_chunk[faiss_id] = cid
            self._payload[cid] = (it.text, dict(it.metadata))
            written += 1

        return written

    def query(self, query_vector: List[float], *, top_k: Optional[int] = None) -> List[SearchResult]:
        if self._index is None or self._dim is None:
            return []

        if len(query_vector) != int(self._dim):
            raise RuntimeError(f"Query vector dim {len(query_vector)} does not match index dim {self._dim}")

        k = int(top_k or self.settings.top_k)
        q = _l2_normalize(np.array(query_vector, dtype="float32")).reshape(1, -1)

        scores, ids = self._index.search(q, k)
        results: List[SearchResult] = []

        for score, fid in zip(scores[0].tolist(), ids[0].tolist()):
            if fid == -1:
                continue
            cid = self._id_to_chunk.get(int(fid))
            if not cid:
                continue
            text, md = self._payload.get(cid, ("", {}))
            results.append(SearchResult(chunk_id=cid, score=float(score), text=text, metadata=md))
        return results
