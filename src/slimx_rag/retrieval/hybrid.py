"""Multi-stage hybrid retrieval: dense + lexical -> RRF -> exact boost -> parent grouping.

Stages (each value is recorded for inspection, never collapsed into one opaque score):

1. normalize query + extract identifiers + detect intent
2. dense candidate retrieval (injected ``dense_search``)
3. lexical candidate retrieval (BM25 sidecar)
4. reciprocal-rank fusion of the dense and lexical rankings (rank-based, not score-added)
5. exact title/entity boost + intent-aware timeline demotion
6. parent grouping + diversity (strongest child per parent first; a sibling only for a
   materially different field; cap per parent; preserve distinct parents)

Reranking is an optional hook (off by default). The result carries dense/lexical/exact/
fusion/rerank/final ranks and the parent-selection reason so Developer Mode can explain
every decision. Scores from different stages are never presented as one calibrated scale.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from slimx_rag.settings import RetrievalSettings

from .lexical import LexicalIndex
from .tokenize import lexical_tokens, normalize_query, query_identifiers, query_intent

# (query, top_k) -> [(chunk_id, cosine_score)] ranked best-first.
DenseSearch = Callable[[str, int], list[tuple[str, float]]]
# (query, [chunk_ids]) -> {chunk_id: rerank_score}; optional.
Reranker = Callable[[str, list[str]], dict[str, float]]


@dataclass(slots=True)
class ChunkRecord:
    """The corpus row hybrid retrieval reasons over (text + parent/page identity)."""

    chunk_id: str
    text: str
    parent_id: str
    page_number: int | None = None
    section: str | None = None
    page_type: str = "unknown"
    source_title: str = ""
    entry: str = ""  # the parent/page title, e.g. "Kimi K2.6"
    token_count: int = 0


@dataclass(slots=True)
class HybridResult:
    chunk_id: str
    parent_id: str
    page_number: int | None
    section: str | None
    page_type: str
    source_title: str
    entry: str
    text: str
    token_count: int = 0
    dense_score: float = 0.0
    dense_rank: int | None = None
    lexical_score: float = 0.0
    lexical_rank: int | None = None
    exact_match: bool = False
    exact_score: float = 0.0
    fusion_score: float = 0.0
    fusion_rank: int | None = None
    rerank_score: float | None = None
    final_rank: int | None = None
    parent_reason: str = ""
    sibling_expanded: bool = False

    def citation(self) -> str:
        """Human-meaningful label, e.g. ``[LLM Architecture Gallery, p. 67, Key detail]``."""
        parts = [self.source_title or self.entry or "document"]
        if self.page_number is not None:
            parts.append(f"p. {self.page_number}")
        if self.section and self.section != self.entry:
            parts.append(self.section)
        return "[" + ", ".join(parts) + "]"


def reciprocal_rank_fusion(rankings: list[list[str]], *, k: int) -> dict[str, float]:
    """RRF: score(d) = sum 1/(k + rank_i(d)). Rank-based, scale-free."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


@dataclass(slots=True)
class HybridRetriever:
    dense_search: DenseSearch
    get_record: Callable[[str], ChunkRecord | None]
    lexical: LexicalIndex | None = None
    reranker: Reranker | None = None

    def retrieve(
        self, query: str, *, settings: RetrievalSettings
    ) -> tuple[list[HybridResult], dict[str, Any]]:
        nq = normalize_query(query)
        intent = query_intent(nq)
        q_ids = query_identifiers(nq)

        dense = self.dense_search(nq, settings.dense_candidates)
        dense_rank = {cid: i for i, (cid, _) in enumerate(dense)}
        dense_score = {cid: s for cid, s in dense}

        lexical_used = bool(
            settings.enable_lexical and self.lexical is not None and len(self.lexical)
        )
        lexical = (
            self.lexical.search(nq, top_k=settings.lexical_candidates)
            if lexical_used and self.lexical is not None
            else []
        )
        lexical_rank = {cid: i for i, (cid, _) in enumerate(lexical)}
        lexical_score = {cid: s for cid, s in lexical}

        rankings = [[cid for cid, _ in dense]]
        if lexical_used:
            rankings.append([cid for cid, _ in lexical])
        fused = reciprocal_rank_fusion(rankings, k=settings.rrf_k)

        results: list[HybridResult] = []
        for cid, fscore in fused.items():
            rec = self.get_record(cid)
            if rec is None:
                continue
            identity_tokens = (
                set(lexical_tokens(rec.source_title))
                | set(lexical_tokens(rec.entry))
                | set(lexical_tokens(rec.section or ""))
            )
            exact = bool(q_ids & identity_tokens)
            text_exact = bool(q_ids & set(lexical_tokens(rec.text)))
            exact_score = (
                settings.exact_match_boost
                if exact
                else (0.3 * settings.exact_match_boost if text_exact else 0.0)
            )
            adjusted = fscore + exact_score
            # Keep timeline/index pages from outranking fact sheets on factual questions.
            if intent == "factual" and rec.page_type == "timeline":
                adjusted *= settings.timeline_penalty
            results.append(
                HybridResult(
                    chunk_id=cid,
                    parent_id=rec.parent_id,
                    page_number=rec.page_number,
                    section=rec.section,
                    page_type=rec.page_type,
                    source_title=rec.source_title,
                    entry=rec.entry,
                    text=rec.text,
                    token_count=rec.token_count,
                    dense_score=dense_score.get(cid, 0.0),
                    dense_rank=dense_rank.get(cid),
                    lexical_score=lexical_score.get(cid, 0.0),
                    lexical_rank=lexical_rank.get(cid),
                    exact_match=exact,
                    exact_score=exact_score,
                    fusion_score=adjusted,
                )
            )

        # Optional reranking over the top fused candidates (off by default).
        if settings.enable_rerank and self.reranker is not None and results:
            results.sort(key=lambda r: (-r.fusion_score, r.chunk_id))
            head = results[: max(settings.final_parents * 3, 15)]
            rr = self.reranker(nq, [r.chunk_id for r in head])
            for r in results:
                r.rerank_score = rr.get(r.chunk_id)
            results.sort(
                key=lambda r: (-(r.rerank_score if r.rerank_score is not None else -1e9), r.chunk_id)
                if r.rerank_score is not None
                else (-r.fusion_score, r.chunk_id)
            )
        else:
            results.sort(key=lambda r: (-r.fusion_score, r.chunk_id))

        for i, r in enumerate(results):
            r.fusion_rank = i

        selected = _group_by_parent(results, settings)
        trace = {
            "strategy": "hybrid" if lexical_used else "dense",
            "intent": intent,
            "query_identifiers": sorted(q_ids),
            "dense_candidates": len(dense),
            "lexical_candidates": len(lexical),
            "fused_candidates": len(results),
            "final_count": len(selected),
            "final_parents": len({r.parent_id for r in selected}),
            "reranked": bool(settings.enable_rerank and self.reranker is not None),
        }
        return selected, trace


def _group_by_parent(
    ordered: list[HybridResult], settings: RetrievalSettings
) -> list[HybridResult]:
    """Strongest child per parent first; then sibling expansion for new fields, capped."""
    per_parent: dict[str, list[str | None]] = {}
    distinct: list[str] = []
    selected: list[HybridResult] = []

    # Pass 1: one strongest child per distinct parent, up to final_parents.
    for r in ordered:
        if r.parent_id in per_parent:
            continue
        if len(distinct) >= settings.final_parents:
            continue
        per_parent[r.parent_id] = [r.section]
        distinct.append(r.parent_id)
        r.parent_reason = "primary"
        selected.append(r)

    # Pass 2: a second child of an already-selected parent only if it adds a new field.
    for r in ordered:
        if r in selected or r.parent_id not in per_parent:
            continue
        chosen = per_parent[r.parent_id]
        if len(chosen) >= settings.max_children_per_parent:
            r.parent_reason = "dropped_parent_cap"
            continue
        if r.section in chosen:
            r.parent_reason = "dropped_duplicate_section"
            continue
        chosen.append(r.section)
        r.parent_reason = "sibling_expansion"
        r.sibling_expanded = True
        selected.append(r)

    selected.sort(key=lambda r: r.fusion_rank if r.fusion_rank is not None else 0)
    for i, r in enumerate(selected):
        r.final_rank = i
    return selected
