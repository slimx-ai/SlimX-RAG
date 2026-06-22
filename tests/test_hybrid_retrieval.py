"""Hybrid retrieval: identifier tokenization, BM25, RRF, exact boost, parent grouping.

The fixture mimics the reported failure: a deliberately poor dense ranking that puts the
timeline page and anonymous key-detail fragments on top. The hybrid layer (lexical +
exact-match + intent-aware demotion + parent grouping) must recover the correct pages.
"""

from __future__ import annotations

from slimx_rag.retrieval import (
    Bm25Index,
    ChunkRecord,
    HybridRetriever,
    lexical_tokens,
    query_identifiers,
    reciprocal_rank_fusion,
)
from slimx_rag.settings import RetrievalSettings

_TITLE = "LLM Architecture Gallery"
_ARCH = "SCALE 1T total, 32B active. CONTEXT TOKENS 256,000. ATTENTION 61-MLA."

_RECORDS: dict[str, ChunkRecord] = {
    "t1": ChunkRecord(
        "t1",
        "Model Index. Kimi K2.5 2026-01-10. Kimi K2.6 2026-02-20. Kimi K2.7 2026-03-30.",
        parent_id="T",
        page_number=2,
        section="Model Index",
        page_type="timeline",
        source_title=_TITLE,
        entry="Model Index",
    ),
    "p5a": ChunkRecord("p5a", _ARCH, "P5", 65, "Kimi K2.5", "fact_sheet", _TITLE, "Kimi K2.5"),
    "p5b": ChunkRecord(
        "p5b",
        "Uses the same 1T 32B-active 61-MLA architecture as the K2 line.",
        "P5", 65, "KEY DETAIL", "fact_sheet", _TITLE, "Kimi K2.5",
    ),
    "p6a": ChunkRecord("p6a", _ARCH, "P6", 67, "Kimi K2.6", "fact_sheet", _TITLE, "Kimi K2.6"),
    "p6b": ChunkRecord(
        "p6b",
        "Focuses on coding-agent workflows; uses the same text architecture as Kimi K2.5.",
        "P6", 67, "KEY DETAIL", "fact_sheet", _TITLE, "Kimi K2.6",
    ),
    "p7a": ChunkRecord("p7a", _ARCH, "P7", 69, "Kimi K2.7", "fact_sheet", _TITLE, "Kimi K2.7"),
    "p7b": ChunkRecord(
        "p7b",
        "Extends context; uses the same 1T 32B-active 61-MLA architecture.",
        "P7", 69, "KEY DETAIL", "fact_sheet", _TITLE, "Kimi K2.7",
    ),
    "g1": ChunkRecord(
        "g1",
        "MLA means multi-head latent attention. Glossary of terms.",
        "G", 90, "Glossary", "glossary", _TITLE, "Glossary",
    ),
}

# Deliberately poor dense order: timeline + glossary + anonymous KEY DETAIL fragments first.
_DENSE_ORDER = ["t1", "g1", "p6b", "p5b", "p7b", "p6a", "p5a", "p7a"]


def _dense_search(query: str, top_k: int) -> list[tuple[str, float]]:
    return [(cid, 0.9 - 0.08 * i) for i, cid in enumerate(_DENSE_ORDER)][:top_k]


def _retriever() -> HybridRetriever:
    bm25 = Bm25Index().build(
        [(cid, f"{r.entry} {r.section} {r.text}") for cid, r in _RECORDS.items()]
    )
    return HybridRetriever(dense_search=_dense_search, get_record=_RECORDS.get, lexical=bm25)


_SETTINGS = RetrievalSettings()


def test_key_detail_query_puts_k26_page_at_top() -> None:
    selected, trace = _retriever().retrieve(
        "What is the key detail for Kimi K2.6?", settings=_SETTINGS
    )
    assert trace["strategy"] == "hybrid"  # never claim hybrid when only dense ran
    assert any(r.entry == "Kimi K2.6" for r in selected[:2])  # rank 1 or 2 (gate #7)


def test_comparison_query_returns_all_three_parents() -> None:
    selected, _ = _retriever().retrieve(
        "Compare the text architectures of Kimi K2.5, Kimi K2.6, and Kimi K2.7",
        settings=_SETTINGS,
    )
    entries = {r.entry for r in selected}
    assert {"Kimi K2.5", "Kimi K2.6", "Kimi K2.7"} <= entries  # gate #8
    assert selected[0].page_type == "fact_sheet"  # timeline didn't outrank fact sheets (#10)
    # Diversified by parent: no duplicate (parent, section) pair (gate #9, #13).
    seen: set[tuple[str, str | None]] = set()
    for r in selected:
        key = (r.parent_id, r.section)
        assert key not in seen
        seen.add(key)


def test_timeline_retrievable_for_date_question() -> None:
    selected, _ = _retriever().retrieve("When was Kimi K2.6 released?", settings=_SETTINGS)
    assert any(r.entry == "Model Index" for r in selected)  # gate #11


def test_exact_identifier_and_citation() -> None:
    selected, _ = _retriever().retrieve("key detail for Kimi K2.6", settings=_SETTINGS)
    k26 = next(r for r in selected if r.entry == "Kimi K2.6")
    assert k26.exact_match  # gate #12
    cite = k26.citation()
    assert "LLM Architecture Gallery" in cite and "p. 67" in cite  # gate #14


def test_identifier_tokenization_preserves_versions() -> None:
    ids = query_identifiers("Kimi K2.6 vs 35B-A3B GLM-5.1 MLA 256,000 2026-04-20")
    assert {"k2.6", "35b-a3b", "glm-5.1", "mla", "256,000", "2026-04-20"} <= ids
    assert "k2.6" in lexical_tokens("the Kimi K2.6 model")


def test_bm25_finds_exact_identifier() -> None:
    idx = Bm25Index().build([("a", "kimi k2.6 fact sheet"), ("b", "kimi k2.7 fact sheet")])
    res = idx.search("k2.6", top_k=2)
    assert res and res[0][0] == "a"


def test_rrf_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "b"]], k=60)
    assert fused["a"] > fused["b"]
