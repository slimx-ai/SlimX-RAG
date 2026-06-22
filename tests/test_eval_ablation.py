"""Controlled ablation over the gold gallery — the measured before/after, as a test.

The embedder is the deterministic ``hash`` provider for every config, so the gains below
are attributable to parsing/chunking/lexical/grouping, NOT the embedding model. Numbers
are deterministic (hash + BM25 + chunking are all deterministic).
"""

from __future__ import annotations

from slimx_rag.eval.ablation import format_report, run_ablation


def test_ablation_quantifies_structured_pipeline_gains():
    report = run_ablation()
    flat = report["A_flat_recursive_dense"]
    structured_dense = report["B_structured_dense"]
    hybrid = report["D_structured_hybrid"]
    grouped = report["E_structured_hybrid_grouped"]

    # A: flattened text -> one anonymous chunk with no parent identity that also exceeds
    # the embedding token cap (silent-truncation risk). This is the reported failure.
    assert flat.recall_at_5 == 0.0
    assert flat.self_contained_rate == 0.0
    assert flat.title_presence_rate == 0.0
    assert flat.truncation_rate == 1.0

    # A -> B (page-preserving parser + structured chunks): the dominant fix. Chunks become
    # self-describing and parent-identifiable, and none exceeds the token cap.
    assert structured_dense.self_contained_rate == 1.0
    assert structured_dense.title_presence_rate == 1.0
    assert structured_dense.truncation_rate == 0.0
    assert structured_dense.recall_at_5 >= 0.8
    assert structured_dense.recall_at_5 > flat.recall_at_5

    # D -> E (parent grouping): fewer duplicate parents, full required-parent coverage for
    # comparison questions, recall preserved.
    assert grouped.duplicate_parent_rate < hybrid.duplicate_parent_rate
    assert grouped.required_parent_coverage == 1.0
    assert grouped.recall_at_5 >= 0.8

    # Overall before -> after.
    assert grouped.recall_at_5 > flat.recall_at_5
    assert grouped.self_contained_rate > flat.self_contained_rate
    assert grouped.truncation_rate == 0.0


def test_format_report_renders_all_configs():
    table = format_report(run_ablation())
    assert "| Config |" in table
    for name in ("A_flat_recursive_dense", "E_structured_hybrid_grouped"):
        assert name in table
