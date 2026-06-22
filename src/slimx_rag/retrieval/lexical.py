"""Backend-independent lexical retrieval (Okapi BM25) over the chunk corpus.

This is a small, dependency-free sidecar so hybrid retrieval works regardless of the
vector backend. It is built from chunk text the local backend already holds in memory;
remote/ANN backends that cannot expose their corpus simply run dense-only and the trace
reports ``strategy="dense"`` (never a false claim of hybrid).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from .tokenize import lexical_tokens


@runtime_checkable
class LexicalIndex(Protocol):
    def search(self, query: str, *, top_k: int) -> list[tuple[str, float]]:
        """Return ``(chunk_id, score)`` ranked by lexical relevance, best first."""
        ...

    def __len__(self) -> int:
        ...


class Bm25Index:
    """In-memory Okapi BM25 index keyed by chunk id."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._tf: list[Counter[str]] = []
        self._len: list[int] = []
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._n: int = 0

    def build(self, docs: Iterable[tuple[str, str]]) -> Bm25Index:
        self._ids, self._tf, self._len, self._df = [], [], [], {}
        total = 0
        for chunk_id, text in docs:
            tokens = lexical_tokens(text)
            tf = Counter(tokens)
            self._ids.append(chunk_id)
            self._tf.append(tf)
            self._len.append(len(tokens))
            total += len(tokens)
            for term in tf:
                self._df[term] = self._df.get(term, 0) + 1
        self._n = len(self._ids)
        self._avgdl = (total / self._n) if self._n else 0.0
        return self

    def __len__(self) -> int:
        return self._n

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    def search(self, query: str, *, top_k: int) -> list[tuple[str, float]]:
        if self._n == 0:
            return []
        terms = [t for t in lexical_tokens(query) if t in self._df]
        if not terms:
            return []
        avgdl = self._avgdl or 1.0
        scored: list[tuple[str, float]] = []
        for i, chunk_id in enumerate(self._ids):
            tf = self._tf[i]
            dl = self._len[i] or 1
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                score += self._idf(term) * (f * (self.k1 + 1.0)) / denom
            if score > 0.0:
                scored.append((chunk_id, score))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:top_k]
