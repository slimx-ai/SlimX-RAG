from __future__ import annotations

import heapq
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from slimx_rag.embed import EmbeddedChunk
from slimx_rag.settings import IndexSettings

from .base import IndexBackend
from .types import SearchResult, IndexState


def _cosine(query_vec: List[float], query_norm: float, vec: List[float], vec_norm: float) -> float:
    if query_norm <= 0.0 or vec_norm <= 0.0:
        return 0.0
    dot = 0.0
    # assume same length validated upstream
    for a, b in zip(query_vec, vec):
        dot += float(a) * float(b)
    return dot / (query_norm * vec_norm)


def _norm(vec: List[float]) -> float:
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

    def __init__(
        self,
        index_path: Path,
        *,
        settings: Optional[IndexSettings] = None,
        state_path: Optional[Path] = None,
    ):
        super().__init__(index_path, settings=settings, state_path=state_path)
        self._items: Dict[str, Tuple[List[float], float, str, dict[str, object]]] = {}
        # Override state load to use the shared IndexState type (for clarity)
        self.state = IndexState.load(self.state_path)

    def __len__(self) -> int:
        return len(self._items)

    def load(self) -> None:
        self._items.clear()
        self._dim = None
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
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # deterministic order for reproducible builds
        items_sorted = sorted(self._items.items(), key=lambda kv: kv[0])

        fd, tmp_path = tempfile.mkstemp(prefix="index_", suffix=".jsonl", dir=str(self.index_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for cid, (vec, _n, txt, md) in items_sorted:
                    rec = {"chunk_id": cid, "vector": vec, "text": txt, "metadata": md}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(tmp_path, self.index_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        self._save_state_if_enabled()

    def _apply_metadata_whitelist(self, md: dict[str, object]) -> dict[str, object]:
        wl = self.settings.metadata_whitelist
        if not wl:
            return md
        keep = set(wl)
        return {k: v for k, v in md.items() if k in keep}

    def delete(self, chunk_ids: Iterable[str]) -> int:
        deleted = 0
        for cid in chunk_ids:
            if cid in self._items:
                del self._items[cid]
                deleted += 1
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
        return written

    def query(self, query_vector: List[float], *, top_k: Optional[int] = None) -> List[SearchResult]:
        if self._dim is not None and len(query_vector) != self._dim:
            raise RuntimeError(f"Query vector dim {len(query_vector)} does not match index dim {self._dim}")

        k = top_k or self.settings.top_k
        qn = _norm(query_vector)

        # Use heap for top-k without sorting all (better for mid-size)
        heap: List[Tuple[float, str]] = []
        for cid, (vec, vn, _text, _md) in self._items.items():
            score = _cosine(query_vector, qn, vec, vn)
            if len(heap) < k:
                heapq.heappush(heap, (score, cid))
            else:
                if score > heap[0][0]:
                    heapq.heapreplace(heap, (score, cid))

        heap.sort(reverse=True, key=lambda t: t[0])
        results: List[SearchResult] = []
        for score, cid in heap:
            _vec, _vn, text, md = self._items[cid]
            results.append(SearchResult(chunk_id=cid, score=float(score), text=text, metadata=md))
        return results


# Backwards-compatible alias (older code imports LocalJsonlIndex)
LocalJsonlIndex = LocalJsonlIndexBackend
