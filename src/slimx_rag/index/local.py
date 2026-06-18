from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from slimx_rag.embed import EmbeddedChunk
from slimx_rag.settings import IndexSettings
from slimx_rag.utils.commons import _atomic_write_lines

from .base import IndexBackend
from .types import IndexState, SearchResult


def _norm(vec: list[float]) -> float:
    s = 0.0
    for x in vec:
        s += float(x) * float(x)
    return math.sqrt(s)


class LocalJsonlIndexBackend(IndexBackend):
    """Simple local JSONL vector index (MVP backend plugin).

    - stores vectors + text + metadata in a JSONL file
    - loads into memory for query
    - supports incremental updates via IndexState (doc_id/content_hash tracking)

    Production note: this won't scale to huge corpora; it's a good stepping stone before FAISS/Qdrant/pgvector.
    """

    # The local backend holds every vector in memory, so retrieval-time metadata scoping
    # (workspace_id/document_ids) can be served by over-fetch + Python filter (see retriever).
    supports_inmemory_scope_filter = True

    def __init__(
        self,
        index_path: Path,
        *,
        settings: IndexSettings | None = None,
        state_path: Path | None = None,
    ):
        super().__init__(index_path, settings=settings, state_path=state_path)
        self._items: dict[str, tuple[list[float], float, str, dict[str, object]]] = {}
        # Lazily-built query cache: a stacked vector matrix + row norms, parallel to _cids.
        # Rebuilt on demand whenever _items changes (load/upsert/delete set _dirty).
        self._matrix: np.ndarray | None = None
        self._mnorms: np.ndarray | None = None
        self._cids: list[str] = []
        self._dirty = True
        # Override state load to use the shared IndexState type (for clarity)
        self.state = IndexState.load(self.state_path)

    def __len__(self) -> int:
        return len(self._items)

    def _rebuild_matrix(self) -> None:
        """Stack stored vectors into a numpy matrix for vectorized cosine query."""
        self._cids = list(self._items.keys())
        if self._cids:
            self._matrix = np.array([self._items[c][0] for c in self._cids], dtype=np.float64)
            self._mnorms = np.linalg.norm(self._matrix, axis=1)
        else:
            self._matrix = np.zeros((0, self._dim or 0), dtype=np.float64)
            self._mnorms = np.zeros((0,), dtype=np.float64)
        self._dirty = False

    def load(self) -> None:
        self._items.clear()
        self._dim = None
        self._dirty = True
        if not self.index_path.exists():
            # still load state if present
            self.state = IndexState.load(self.state_path)
            return

        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cid = str(rec.get("chunk_id") or "")
                vec = [float(x) for x in (rec.get("vector") or [])]
                txt = str(rec.get("text") or "")
                md = dict(rec.get("metadata") or {})
                if not cid:
                    continue

                if self._dim is None:
                    self._dim = len(vec)
                elif len(vec) != self._dim:
                    raise RuntimeError("Index contains mixed vector dimensions")

                n = _norm(vec)
                self._items[cid] = (vec, n, txt, md)

        # state is loaded in __init__, but reload if file exists and changed
        self.state = IndexState.load(self.state_path)

    def save(self) -> None:
        # deterministic order for reproducible builds
        items_sorted = sorted(self._items.items(), key=lambda kv: kv[0])
        _atomic_write_lines(
            self.index_path,
            (
                json.dumps({"chunk_id": cid, "vector": vec, "text": txt, "metadata": md}, ensure_ascii=False) + "\n"
                for cid, (vec, _n, txt, md) in items_sorted
            ),
        )
        self._save_state_if_enabled()

    def delete(self, chunk_ids: Iterable[str]) -> int:
        deleted = 0
        for cid in chunk_ids:
            if cid in self._items:
                del self._items[cid]
                deleted += 1
        if deleted:
            self._dirty = True
        return deleted

    def upsert(self, items: Iterable[EmbeddedChunk], *, skip_existing: bool = True) -> int:
        written = 0
        for it in items:
            cid = it.chunk_id
            if skip_existing and cid in self._items:
                continue

            vec = [float(x) for x in it.vector]
            if self._dim is None:
                self._dim = len(vec)
            elif len(vec) != self._dim:
                raise RuntimeError(f"Vector dim mismatch: index dim {self._dim} vs item dim {len(vec)}")

            md = self._apply_metadata_whitelist(dict(it.metadata))
            n = _norm(vec)
            self._items[cid] = (vec, n, it.text, md)
            written += 1
        if written:
            self._dirty = True
        return written

    def query(self, query_vector: list[float], *, top_k: int | None = None) -> list[SearchResult]:
        if self._dim is not None and len(query_vector) != self._dim:
            raise RuntimeError(f"Query vector dim {len(query_vector)} does not match index dim {self._dim}")

        k = int(top_k or self.settings.top_k)
        if self._dirty:
            self._rebuild_matrix()
        if not self._cids:
            return []

        assert self._matrix is not None and self._mnorms is not None
        q = np.asarray(query_vector, dtype=np.float64)
        qn = float(np.linalg.norm(q))
        if qn <= 0.0:
            scores = np.zeros(len(self._cids), dtype=np.float64)
        else:
            denom = self._mnorms * qn
            dots = self._matrix @ q
            scores = np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0.0)

        # Rank by score descending, then chunk_id ascending — identical to the shared
        # _sort_results contract, but via lexsort so we only materialize the top-k results
        # instead of one SearchResult per indexed chunk.
        cids = np.asarray(self._cids)
        order = np.lexsort((cids, -scores))[:k]
        scores_list = scores.tolist()
        return [
            SearchResult(
                chunk_id=self._cids[i],
                score=float(scores_list[i]),
                text=self._items[self._cids[i]][2],
                metadata=self._items[self._cids[i]][3],
            )
            for i in order.tolist()
        ]
