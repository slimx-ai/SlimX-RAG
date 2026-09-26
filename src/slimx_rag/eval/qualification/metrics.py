"""Per-case scoring for the qualification benchmark (document-level relevance)."""

from __future__ import annotations

import math
from typing import Any

from .gold import Case

KS = (1, 3, 5, 8)


def _first_ranks(results: list[dict[str, Any]], names: set[str]) -> dict[str, int]:
    """First rank (1-based) at which each named document appears."""
    first: dict[str, int] = {}
    for row in results:
        name = row.get("doc_name")
        if name in names and name not in first:
            first[name] = int(row["rank"])
    return first


def _locator_ok(case: Case, row: dict[str, Any]) -> bool:
    if case.expected_page is not None:
        return row.get("page") == case.expected_page
    if case.expected_section is not None:
        needle = case.expected_section.lower()
        section = str(row.get("section") or "").lower()
        path = " / ".join(str(part) for part in (row.get("section_path") or [])).lower()
        return needle in section or needle in path
    return True


def evaluate_case(
    case: Case,
    results: list[dict[str, Any]],
    *,
    scope_workspace_id: str,
    scope_document_ids: set[str],
    deleted_names: set[str],
) -> dict[str, Any]:
    """Score one case. ``results`` rows carry rank/doc_name/workspace_id/document_id/page/section/text."""
    expected = set(case.expected_docs)
    forbidden = set(case.forbidden_docs)
    row: dict[str, Any] = {
        "id": case.id,
        "phase": case.phase,
        "tags": list(case.tags),
        "question": case.question,
        "answerable": case.answerable,
        "no_answer": case.no_answer,
        "expected_docs": sorted(expected),
        "result_count": len(results),
        "top1": (
            {
                "doc": results[0].get("doc_name"),
                "score": results[0].get("score"),
                "page": results[0].get("page"),
                "section": results[0].get("section"),
                "citation": results[0].get("citation"),
            }
            if results
            else None
        ),
        "ranked_docs": [r.get("doc_name") for r in results],
    }
    # --- global isolation invariants (every case, every phase) ---------------------
    row["cross_workspace_leaks"] = sum(1 for r in results if str(r.get("workspace_id")) != scope_workspace_id)
    row["cross_scope_leaks"] = sum(1 for r in results if str(r.get("document_id")) not in scope_document_ids)
    # A forbidden document outside the scope set is a leak; one inside the scope is a distractor
    # that outranked the answer (a ranking-quality signal, never a security signal).
    forbidden_ids = {r.get("document_id") for r in results if r.get("doc_name") in forbidden}
    row["forbidden_document_leaks"] = sum(
        1 for r in results if r.get("doc_name") in forbidden and str(r.get("document_id")) not in scope_document_ids
    )
    row["distractor_hits"] = sum(
        1 for r in results if r.get("doc_name") in forbidden and str(r.get("document_id")) in scope_document_ids
    )
    del forbidden_ids
    row["forbidden_text_hits"] = sum(
        1 for r in results if any(needle in (r.get("text") or "") for needle in case.forbidden_text_any)
    )
    row["stale_deleted_hits"] = (
        sum(1 for r in results if r.get("doc_name") in deleted_names) if case.phase == "after_delete" else 0
    )
    row["expected_text_found"] = (
        any(any(needle in (r.get("text") or "") for needle in case.expected_text_any) for r in results)
        if case.expected_text_any
        else None
    )
    # --- duplicates / diversity ----------------------------------------------------
    chunk_ids = [r.get("chunk_id") for r in results]
    parents = [r.get("parent_id") for r in results]
    row["duplicate_chunks"] = len(chunk_ids) - len(set(chunk_ids))
    row["duplicate_parents"] = len(parents) - len(set(parents))
    row["distinct_docs"] = len({r.get("doc_name") for r in results})
    # --- ranking metrics (answerable cases only) -------------------------------------
    if case.answerable:
        first = _first_ranks(results, expected)
        best = min(first.values()) if first else None
        for k in KS:
            row[f"hit_at_{k}"] = 1.0 if best is not None and best <= k else 0.0
        row["rr"] = (1.0 / best) if best is not None else 0.0
        dcg = sum(1.0 / math.log2(rank + 1) for rank in first.values())
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, len(expected) + 1))
        row["ndcg_at_8"] = (dcg / idcg) if idcg else 0.0
        row["coverage_at_8"] = len(first) / len(expected)
        row["first_rank"] = best
        top_doc = results[0].get("doc_name") if results else None
        row["top1_expected_source"] = (top_doc in expected) if "citation_source" in case.tags else None
        if "citation_locator" in case.tags:
            expected_rows = [r for r in results if r.get("doc_name") in expected]
            if not expected_rows:
                row["expected_locator"] = "missing"
            else:
                row["expected_locator"] = "ok" if _locator_ok(case, expected_rows[0]) else "wrong"
                row["expected_locator_observed"] = {
                    "page": expected_rows[0].get("page"),
                    "section": expected_rows[0].get("section"),
                    "section_path": expected_rows[0].get("section_path"),
                }
        else:
            row["expected_locator"] = None
    else:
        row["top1_expected_source"] = None
        row["expected_locator"] = None
    return row


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r["answerable"]]
    out: dict[str, Any] = {"cases_total": len(rows), "cases_scored": len(scored)}
    for k in KS:
        out[f"hit_at_{k}"] = _mean([r[f"hit_at_{k}"] for r in scored])
    out["mrr"] = _mean([r["rr"] for r in scored])
    out["ndcg_at_8"] = _mean([r["ndcg_at_8"] for r in scored])
    out["coverage_at_8"] = _mean([r["coverage_at_8"] for r in scored])
    exact = [r for r in scored if "exact_identifier" in r["tags"]]
    out["exact_identifier_hit_at_1"] = _mean([r["hit_at_1"] for r in exact])
    multi = [r for r in scored if "multi_doc" in r["tags"]]
    out["multi_doc_coverage_at_8"] = _mean([r["coverage_at_8"] for r in multi])
    sources = [r for r in scored if r["top1_expected_source"] is not None]
    out["top1_expected_source_rate"] = _mean([1.0 if r["top1_expected_source"] else 0.0 for r in sources])
    locators = [r for r in scored if r["expected_locator"] is not None]
    out["expected_locator_ok_rate"] = _mean([1.0 if r["expected_locator"] == "ok" else 0.0 for r in locators])
    out["expected_locator_failures"] = sum(1 for r in scored if r["expected_locator"] in ("wrong", "missing"))
    out["top1_expected_source_failures"] = sum(1 for r in scored if r["top1_expected_source"] is False)
    out["duplicate_chunk_rate"] = _mean([r["duplicate_chunks"] / r["result_count"] for r in rows if r["result_count"]])
    out["duplicate_parent_rate"] = _mean(
        [r["duplicate_parents"] / r["result_count"] for r in rows if r["result_count"]]
    )
    out["mean_distinct_docs"] = _mean([float(r["distinct_docs"]) for r in rows if r["result_count"]])
    out["mean_result_count"] = _mean([float(r["result_count"]) for r in rows])
    out["distractor_hits_total"] = sum(r["distractor_hits"] for r in rows)
    out["cases_with_distractor_hits"] = sum(1 for r in rows if r["distractor_hits"])
    return out


def by_tag(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    tags: set[str] = set()
    for r in rows:
        tags.update(r["tags"])
    out: dict[str, dict[str, Any]] = {}
    for tag in sorted(tags):
        tagged = [r for r in rows if tag in r["tags"]]
        scored = [r for r in tagged if r["answerable"]]
        out[tag] = {
            "cases": len(tagged),
            "scored": len(scored),
            "hit_at_1": _mean([r["hit_at_1"] for r in scored]),
            "hit_at_3": _mean([r["hit_at_3"] for r in scored]),
            "hit_at_5": _mean([r["hit_at_5"] for r in scored]),
            "mrr": _mean([r["rr"] for r in scored]),
            "ndcg_at_8": _mean([r["ndcg_at_8"] for r in scored]),
            "coverage_at_8": _mean([r["coverage_at_8"] for r in scored]),
            "forbidden_document_leaks": sum(r["forbidden_document_leaks"] for r in tagged),
            "distractor_hits": sum(r["distractor_hits"] for r in tagged),
            "cross_scope_leaks": sum(r["cross_scope_leaks"] for r in tagged),
            "cross_workspace_leaks": sum(r["cross_workspace_leaks"] for r in tagged),
            "failed_case_ids": [r["id"] for r in scored if r["hit_at_5"] == 0.0],
        }
    return out


def hard_counters(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "cross_workspace_leaks": sum(r["cross_workspace_leaks"] for r in rows),
        "cross_project_leaks": sum(
            r["cross_scope_leaks"] for r in rows if any(t in r["tags"] for t in ("cross_project", "cross_workspace"))
        ),
        "cross_scope_leaks": sum(r["cross_scope_leaks"] for r in rows),
        "forbidden_document_leaks": sum(r["forbidden_document_leaks"] for r in rows),
        "forbidden_text_hits": sum(r["forbidden_text_hits"] for r in rows),
        "stale_deleted_hits": sum(r["stale_deleted_hits"] for r in rows),
        "updated_doc_stale_hits": sum(r["forbidden_text_hits"] for r in rows if "updated_doc" in r["tags"]),
        "updated_doc_missing_new_content": sum(
            1 for r in rows if "updated_doc" in r["tags"] and r["expected_text_found"] is False
        ),
        "duplicate_chunks": sum(r["duplicate_chunks"] for r in rows),
        "distractor_hits": sum(r["distractor_hits"] for r in rows),
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 2)
