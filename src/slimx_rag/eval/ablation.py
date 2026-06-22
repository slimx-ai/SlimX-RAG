"""Controlled ablation over the gold gallery, holding the embedder constant (hash).

Configs (the embedder is the deterministic offline ``hash`` provider throughout, so any
gain is attributable to parsing/chunking/lexical/grouping — NOT the embedding model):

- A: flattened text + recursive character chunks + dense-only
- B: page parser + structured chunks + dense-only
- D: B + lexical (BM25) + reciprocal-rank fusion (hybrid)
- E: D + parent grouping + diversity

C (query/document-aware embeddings) is a no-op under the symmetric hash embedder, and
F/G (reranker, embedding-model swap) need real models; see the eval report notes. The
embedding-model comparison is intentionally excluded here so the table is not mistaken
for an embedding-model result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from langchain_core.documents import Document

from slimx_rag.chunk import chunk_documents, chunk_parsed_document
from slimx_rag.chunk.tokenizer import HeuristicTokenCounter
from slimx_rag.embed.embedder import HashEmbedder
from slimx_rag.retrieval.hybrid import ChunkRecord, HybridRetriever
from slimx_rag.retrieval.lexical import Bm25Index
from slimx_rag.settings import RetrievalSettings, StructuredChunkSettings

from . import metrics as M
from .gold import DOC_TITLE, GOLD_CASES, build_parsed_gallery, flattened_gallery_text

# Small cap so fact-sheet pages split into field-children — recreating the reported
# "multiple sibling chunks from one page" scenario that parent grouping must fix.
_TOKEN_CAP = 32


@dataclass(slots=True)
class ConfigMetrics:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    required_parent_coverage: float
    duplicate_parent_rate: float
    timeline_noise_rate: float
    self_contained_rate: float
    title_presence_rate: float
    truncation_rate: float


def _structured_records() -> list[ChunkRecord]:
    chunks = chunk_parsed_document(
        build_parsed_gallery(),
        settings=StructuredChunkSettings(
            target_tokens=_TOKEN_CAP, max_tokens=_TOKEN_CAP, force_split_overlap_tokens=8
        ),
        token_counter=HeuristicTokenCounter(max_tokens=_TOKEN_CAP),
    )
    return [
        ChunkRecord(
            chunk_id=ch.chunk_id,
            text=ch.embedding_text,
            parent_id=ch.parent_id,
            page_number=ch.page_number,
            section=ch.section,
            page_type=ch.page_type.value,
            source_title=ch.source_title,
            entry=str(ch.metadata.get("entry", "")),
            token_count=ch.token_count,
        )
        for ch in chunks
    ]


def _flat_records() -> list[ChunkRecord]:
    counter = HeuristicTokenCounter(max_tokens=_TOKEN_CAP)
    doc = Document(page_content=flattened_gallery_text(), metadata={"doc_id": "gallery", "title": DOC_TITLE})
    chunks = chunk_documents([doc], chunk_size=800, chunk_overlap=120)
    return [
        ChunkRecord(
            chunk_id=str(c.metadata.get("chunk_id") or f"flat{i}"),
            text=c.page_content,
            parent_id="gallery",
            page_number=None,
            section=None,
            page_type="unknown",
            source_title=DOC_TITLE,
            entry="",  # flat chunks have no parent/entity identity — the failure
            token_count=counter.count(c.page_content),
        )
        for i, c in enumerate(chunks)
    ]


def _hash_dense(records: list[ChunkRecord]):
    emb = HashEmbedder(dim=64)
    matrix = np.array(emb.embed_documents([r.text for r in records]), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    ids = [r.chunk_id for r in records]

    def search(query: str, k: int) -> list[tuple[str, float]]:
        qv = np.array(emb.embed_query(query), dtype=np.float32)
        denom = norms * (float(np.linalg.norm(qv)) or 1.0)
        sims = np.divide(matrix @ qv, denom, out=np.zeros(len(ids), dtype=np.float32), where=denom > 0)
        order = np.argsort(-sims)[:k]
        return [(ids[i], float(sims[i])) for i in order.tolist()]

    return search


def _retriever(records: list[ChunkRecord], *, lexical: bool) -> HybridRetriever:
    by_id = {r.chunk_id: r for r in records}
    bm = Bm25Index().build([(r.chunk_id, r.text) for r in records]) if lexical else None
    return HybridRetriever(dense_search=_hash_dense(records), get_record=by_id.get, lexical=bm)


def _items(retriever: HybridRetriever, settings: RetrievalSettings, question: str) -> list[M.EvalItem]:
    results, _ = retriever.retrieve(question, settings=settings)
    return [
        M.EvalItem(
            entity=r.entry,
            page=r.page_number,
            page_type=r.page_type,
            parent_id=r.parent_id,
            section=r.section,
            token_count=r.token_count,
            self_contained=bool(r.entry) and r.page_number is not None,
        )
        for r in results[:8]
    ]


def _aggregate(records: list[ChunkRecord], retriever: HybridRetriever, settings: RetrievalSettings) -> ConfigMetrics:
    r1: list[float] = []
    r3: list[float] = []
    r5: list[float] = []
    mr: list[float] = []
    cov: list[float] = []
    dup: list[float] = []
    tl: list[float] = []
    sc: list[float] = []
    tp: list[float] = []
    for gold in GOLD_CASES:
        items = _items(retriever, settings, gold.question)
        if gold.required:
            for bucket, k in ((r1, 1), (r3, 3), (r5, 5)):
                value = M.recall_at_k(items, gold, k)
                if value is not None:
                    bucket.append(value)
            mrr_value = M.mrr(items, gold)
            if mrr_value is not None:
                mr.append(mrr_value)
            coverage = M.required_parent_coverage(items, gold)
            if coverage is not None:
                cov.append(coverage)
        if gold.should_have_answer:
            dup.append(M.duplicate_parent_rate(items))
            sc.append(M.self_contained_rate(items))
            tp.append(M.title_presence_rate(items))
            if gold.intent == "factual":
                tl.append(M.timeline_noise_rate(items))
    trunc = sum(1 for r in records if r.token_count > _TOKEN_CAP) / len(records) if records else 0.0

    def mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    return ConfigMetrics(
        recall_at_1=mean(r1),
        recall_at_3=mean(r3),
        recall_at_5=mean(r5),
        mrr=mean(mr),
        required_parent_coverage=mean(cov),
        duplicate_parent_rate=mean(dup),
        timeline_noise_rate=mean(tl),
        self_contained_rate=mean(sc),
        title_presence_rate=mean(tp),
        truncation_rate=round(trunc, 3),
    )


def run_ablation() -> dict[str, ConfigMetrics]:
    flat = _flat_records()
    structured = _structured_records()
    # "Ungrouped" = parent grouping effectively disabled (caps far above the corpus size).
    dense_only = RetrievalSettings(enable_lexical=False, final_parents=999, max_children_per_parent=999)
    hybrid_ungrouped = RetrievalSettings(enable_lexical=True, final_parents=999, max_children_per_parent=999)
    hybrid_grouped = RetrievalSettings(enable_lexical=True, final_parents=6, max_children_per_parent=2)
    return {
        "A_flat_recursive_dense": _aggregate(flat, _retriever(flat, lexical=False), dense_only),
        "B_structured_dense": _aggregate(structured, _retriever(structured, lexical=False), dense_only),
        "D_structured_hybrid": _aggregate(
            structured, _retriever(structured, lexical=True), hybrid_ungrouped
        ),
        "E_structured_hybrid_grouped": _aggregate(
            structured, _retriever(structured, lexical=True), hybrid_grouped
        ),
    }


def format_report(report: dict[str, ConfigMetrics]) -> str:
    cols = [
        ("R@1", "recall_at_1"),
        ("R@3", "recall_at_3"),
        ("R@5", "recall_at_5"),
        ("MRR", "mrr"),
        ("ReqCov", "required_parent_coverage"),
        ("DupParent", "duplicate_parent_rate"),
        ("Timeline", "timeline_noise_rate"),
        ("SelfCont", "self_contained_rate"),
        ("Title", "title_presence_rate"),
        ("Trunc", "truncation_rate"),
    ]
    lines = ["| Config | " + " | ".join(h for h, _ in cols) + " |"]
    lines.append("| --- | " + " | ".join("---" for _ in cols) + " |")
    for name, metrics in report.items():
        row = " | ".join(f"{getattr(metrics, attr):.3f}" for _, attr in cols)
        lines.append(f"| {name} | {row} |")
    return "\n".join(lines) + "\n"
