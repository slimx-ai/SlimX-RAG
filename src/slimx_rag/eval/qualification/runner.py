"""Drive the real SlimX-RAG HTTP service (in-process ``TestClient``) through the benchmark.

The runner uses only the public endpoints ControlRoom uses (``/api/index/file``,
``/api/index``, ``/api/retrieve``, ``/api/documents/{id}``, ``/ready``, ``/api/config``)
with a Bearer service token, so the measured behavior is the served contract, not a
library shortcut. It indexes the corpus, runs every ``main`` case twice (determinism),
re-indexes the updated document, deletes the obsolete document, simulates a service
restart by dropping the hot backend, and records latency and memory.
"""

from __future__ import annotations

import dataclasses
import json
import os
import resource
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import fidelity as F
from . import metrics as M
from .corpus import WORKSPACES, Corpus, CorpusDoc, build_corpus, document_id_for, write_corpus
from .gold import Scope, cases_for_phase

TOKEN = "qualification-service-token"
DEFAULT_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# The exact model commit the candidate image bakes; the benchmark pins it so the recorded
# runtime identity is the qualified one, never the cache's mutable ``refs/main``.
DEFAULT_HF_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
# ``benchmark``: the corpus's own document titles; ``filename``: ControlRoom's convention
# (title = upload filename for every document). The gate file, gold and corpus are the same.
TITLE_MODES = ("benchmark", "filename")
_HEADERS = {"Authorization": f"Bearer {TOKEN}"}
_RESTART_SAMPLE = 12


@contextmanager
def _environment(values: dict[str, str | None]) -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _resolve_scope(corpus: Corpus, scope: Scope, deleted: set[str]) -> tuple[str, list[str]]:
    workspace_id = WORKSPACES[scope.workspace]
    if scope.document_ids is not None:
        names = list(scope.document_ids)
    else:
        assert scope.project is not None
        names = [doc.name for doc in corpus.project_docs(scope.workspace, scope.project)]
    if not scope.include_deleted:
        names = [name for name in names if name not in deleted]
    return workspace_id, [document_id_for(name) for name in names]


def _index_doc(client: Any, doc: CorpusDoc) -> dict[str, Any]:
    started = time.perf_counter()
    if doc.ingest == "file":
        form: dict[str, Any] = {
            "workspace_id": doc.workspace_id,
            "document_id": doc.document_id,
            "filename": doc.filename,
        }
        if doc.mime_type:
            form["mime_type"] = doc.mime_type
        if doc.title:
            form["title"] = doc.title
        response = client.post(
            "/api/index/file",
            data=form,
            files={"file": (doc.filename, doc.content, doc.mime_type or "application/octet-stream")},
            headers=_HEADERS,
        )
    else:
        response = client.post(
            "/api/index",
            json={
                "workspace_id": doc.workspace_id,
                "document_id": doc.document_id,
                "text": doc.content.decode("utf-8"),
                "metadata": {"title": doc.title or doc.filename},
            },
            headers=_HEADERS,
        )
    elapsed = (time.perf_counter() - started) * 1000
    ok = response.status_code == 200
    body = response.json() if ok else {"status_code": response.status_code, "detail": response.text[:500]}
    return {
        "name": doc.name,
        "ingest": doc.ingest,
        "status_code": response.status_code,
        "elapsed_ms": round(elapsed, 1),
        "chunk_count": body.get("chunk_count") if ok else None,
        "timings_ms": body.get("timings_ms") if ok else None,
        "parser": body.get("parser") if ok else None,
        "warnings": body.get("warnings") if ok else None,
        "index_signature": body.get("index_signature") if ok else None,
        "document_pipeline": body.get("document_pipeline") if ok else None,
        "error": None if ok else body,
    }


def _retrieve(
    client: Any, *, question: str, workspace_id: str, document_ids: list[str] | None, top_k: int
) -> tuple[list[dict[str, Any]] | None, float, dict[str, Any]]:
    body: dict[str, Any] = {"question": question, "top_k": top_k, "workspace_id": workspace_id}
    if document_ids:
        body["document_ids"] = document_ids
    started = time.perf_counter()
    response = client.post("/api/retrieve", json=body, headers=_HEADERS)
    elapsed = (time.perf_counter() - started) * 1000
    if response.status_code != 200:
        return None, elapsed, {"status_code": response.status_code, "detail": response.text[:300]}
    data = response.json()
    rows: list[dict[str, Any]] = []
    for rank, chunk in enumerate(data.get("chunks") or [], start=1):
        metadata = chunk.get("metadata") or {}
        rows.append(
            {
                "rank": rank,
                "chunk_id": chunk.get("chunk_id"),
                "score": chunk.get("score"),
                "document_id": metadata.get("document_id"),
                "workspace_id": metadata.get("workspace_id"),
                "page": metadata.get("page"),
                "section": metadata.get("section"),
                "section_path": metadata.get("section_path"),
                "parent_id": metadata.get("parent_id"),
                "citation": chunk.get("citation"),
                "text": chunk.get("text") or "",
            }
        )
    return rows, elapsed, {"trace": data.get("trace"), "strategy": data.get("retrieval_strategy")}


def _run_cases(
    client: Any,
    corpus: Corpus,
    phase: str,
    *,
    top_k: int,
    deleted: set[str],
    name_by_id: dict[str, str],
    latencies: list[float] | None = None,
    verifier: F.FidelityVerifier | None = None,
    fidelity_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    chunk_lists: dict[str, list[str]] = {}
    failures: list[dict[str, Any]] = []
    for case in cases_for_phase(phase):
        workspace_id, document_ids = _resolve_scope(corpus, case.scope, deleted)
        results, elapsed, extra = _retrieve(
            client, question=case.question, workspace_id=workspace_id, document_ids=document_ids, top_k=top_k
        )
        if latencies is not None:
            latencies.append(elapsed)
        if results is None:
            failures.append({"id": case.id, **extra})
            results = []
        for row in results:
            row["doc_name"] = name_by_id.get(str(row.get("document_id")), None)
            if verifier is not None and fidelity_rows is not None:
                verdict = verifier.verify(row)
                row["fidelity"] = verdict
                fidelity_rows.append({"case": case.id, "chunk_id": row.get("chunk_id"), **verdict})
        chunk_lists[case.id] = [str(r["chunk_id"]) for r in results]
        scored = M.evaluate_case(
            case,
            results,
            scope_workspace_id=workspace_id,
            scope_document_ids=set(document_ids),
            deleted_names=deleted,
        )
        scored["elapsed_ms"] = round(elapsed, 1)
        scored["strategy"] = extra.get("strategy")
        scored["results"] = [
            {k: r.get(k) for k in ("rank", "chunk_id", "score", "doc_name", "page", "section", "citation")}
            | {"text_head": (r.get("text") or "")[:120], "fidelity": r.get("fidelity")}
            for r in results
        ]
        scored["fidelity_problems"] = sum(1 for r in results if r.get("fidelity") and not r["fidelity"]["ok"])
        rows.append(scored)
    return rows, chunk_lists, failures


def run_qualification(
    *,
    provider: str,
    out_dir: Path,
    top_k: int = 8,
    hf_model: str = DEFAULT_HF_MODEL,
    device: str = "cpu",
    hf_revision: str | None = DEFAULT_HF_REVISION,
    title_mode: str = "benchmark",
) -> dict[str, Any]:
    if title_mode not in TITLE_MODES:
        raise ValueError(f"title_mode must be one of {TITLE_MODES}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus()
    if title_mode == "filename":
        corpus = dataclasses.replace(
            corpus,
            docs=tuple(dataclasses.replace(d, title=d.filename) for d in corpus.docs),
            updates={k: dataclasses.replace(v, title=v.filename) for k, v in corpus.updates.items()},
        )
    manifest = write_corpus(corpus, out_dir / "corpus")
    index_dir = out_dir / "index"
    index_dir.mkdir(exist_ok=True)
    env: dict[str, str | None] = {
        "RAG_INDEX_PATH": str(index_dir / "index.jsonl"),
        "RAG_STATE_PATH": str(index_dir / "index_state.json"),
        "RAG_INDEX_BACKEND": "local",
        "RAG_BACKEND_CONFIG": None,
        "RAG_EMBED_PROVIDER": provider,
        "RAG_EMBED_DIM": "384",
        "RAG_HF_MODEL": hf_model,
        "RAG_HF_REVISION": hf_revision if provider == "hf" else None,
        "RAG_EMBED_DEVICE": device if provider == "hf" else None,
        "RAG_AUTH_TOKEN": TOKEN,
        "DEMO_AUTH_TOKEN": None,
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    name_by_id = corpus.name_by_document_id()
    verifier = F.FidelityVerifier(corpus)
    fidelity_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "dataset_version": corpus.version,
        "title_mode": title_mode,
        "hf_revision": hf_revision if provider == "hf" else None,
        "provider": provider,
        "top_k": top_k,
        "corpus_manifest_files": len(dict(manifest["files"])),  # type: ignore[call-overload]
    }
    with _environment(env):
        import importlib

        from fastapi.testclient import TestClient

        server = importlib.import_module("slimx_rag.server.app")

        server._reset_index_cache()
        client = TestClient(server.app)
        unauth = client.post("/api/retrieve", json={"question": "x", "workspace_id": "w"})
        report["auth_rejects_missing_token"] = unauth.status_code == 401

        # --- index ------------------------------------------------------------------
        index_rows = [_index_doc(client, doc) for doc in corpus.docs]
        report["index"] = {
            "documents": len(index_rows),
            "failures": [r for r in index_rows if r["status_code"] != 200],
            "chunks_total": sum(int(r["chunk_count"] or 0) for r in index_rows),
            "elapsed_ms_total": round(sum(r["elapsed_ms"] for r in index_rows), 1),
            "elapsed_ms_median": round(statistics.median(r["elapsed_ms"] for r in index_rows), 1),
            "elapsed_ms_max": round(max(r["elapsed_ms"] for r in index_rows), 1),
            "per_document": [
                {
                    k: r[k]
                    for k in ("name", "ingest", "status_code", "elapsed_ms", "chunk_count", "parser", "timings_ms")
                }
                for r in index_rows
            ],
        }
        first_ok = next((r for r in index_rows if r["status_code"] == 200), None)
        report["index_signature"] = first_ok["index_signature"] if first_ok else None
        ready = client.get("/ready", headers=_HEADERS)
        report["ready"] = {"status_code": ready.status_code, "body": ready.json()}
        report["engine_version"] = (ready.json() or {}).get("engine_version")
        report["memory"] = {
            "max_rss_mb_after_index": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        }

        # --- main pass (twice: determinism) -------------------------------------------
        latencies: list[float] = []
        deleted: set[str] = set()
        rows, chunk_lists_1, failures_1 = _run_cases(
            client,
            corpus,
            "main",
            top_k=top_k,
            deleted=deleted,
            name_by_id=name_by_id,
            latencies=latencies,
            verifier=verifier,
            fidelity_rows=fidelity_rows,
        )
        _, chunk_lists_2, failures_2 = _run_cases(
            client, corpus, "main", top_k=top_k, deleted=deleted, name_by_id=name_by_id
        )
        unstable = sorted(cid for cid in chunk_lists_1 if chunk_lists_1[cid] != chunk_lists_2.get(cid))

        # --- workspace-only scope diagnostic (what a host that filters only by workspace would see)
        ws_only_hits = 0
        ws_only_cases = 0
        for case in cases_for_phase("main"):
            if "cross_project" not in case.tags or case.scope.workspace != "alpha":
                continue
            ws_only_cases += 1
            results, _elapsed, _extra = _retrieve(
                client, question=case.question, workspace_id=WORKSPACES["alpha"], document_ids=None, top_k=top_k
            )
            _ws, in_scope = _resolve_scope(corpus, case.scope, deleted)
            ws_only_hits += sum(1 for r in results or [] if str(r.get("document_id")) not in set(in_scope))
        report["workspace_only_scope_diagnostic"] = {
            "cases": ws_only_cases,
            "cross_project_hits": ws_only_hits,
            "note": "Informational: SlimX-RAG cannot know projects; ControlRoom must always pass document_ids.",
        }

        # --- no-answer separability ---------------------------------------------------
        no_answer_scores = [
            float(r["top1"]["score"]) for r in rows if r["no_answer"] and r["top1"] and r["top1"]["score"] is not None
        ]
        correct_scores = [
            float(r["top1"]["score"])
            for r in rows
            if r["answerable"] and r["top1"] and r.get("hit_at_1") == 1.0 and r["top1"]["score"] is not None
        ]
        report["no_answer"] = {
            "cases": sum(1 for r in rows if r["no_answer"]),
            "cases_returning_chunks": sum(1 for r in rows if r["no_answer"] and r["result_count"]),
            "top1_score_max": max(no_answer_scores) if no_answer_scores else None,
            "top1_score_median": statistics.median(no_answer_scores) if no_answer_scores else None,
            "answerable_correct_top1_score_min": min(correct_scores) if correct_scores else None,
            "answerable_correct_top1_score_median": statistics.median(correct_scores) if correct_scores else None,
            "separable_by_top1_score": (
                bool(no_answer_scores and correct_scores and max(no_answer_scores) < min(correct_scores))
            ),
        }

        # --- lifecycle: update ---------------------------------------------------------
        update_rows = [_index_doc(client, doc) for doc in corpus.updates.values()]
        for doc in corpus.updates.values():
            verifier.replace(doc)
        rows_update, _lists_u, failures_u = _run_cases(
            client,
            corpus,
            "after_update",
            top_k=top_k,
            deleted=deleted,
            name_by_id=name_by_id,
            verifier=verifier,
            fidelity_rows=fidelity_rows,
        )
        # --- lifecycle: delete ---------------------------------------------------------
        delete_rows = []
        for name in corpus.deleted:
            doc = corpus.by_name(name)
            response = client.delete(
                f"/api/documents/{doc.document_id}", params={"workspace_id": doc.workspace_id}, headers=_HEADERS
            )
            listing = client.get(
                f"/api/documents/{doc.document_id}/chunks", params={"workspace_id": doc.workspace_id}, headers=_HEADERS
            )
            delete_rows.append(
                {
                    "name": name,
                    "status_code": response.status_code,
                    "deleted_chunks": (response.json() or {}).get("deleted_chunks")
                    if response.status_code == 200
                    else None,
                    "chunks_listed_after_delete": (listing.json() or {}).get("chunk_count")
                    if listing.status_code == 200
                    else None,
                }
            )
            deleted.add(name)
        rows_delete, _lists_d, failures_d = _run_cases(
            client,
            corpus,
            "after_delete",
            top_k=top_k,
            deleted=deleted,
            name_by_id=name_by_id,
            verifier=verifier,
            fidelity_rows=fidelity_rows,
        )
        # --- restart simulation --------------------------------------------------------
        server._reset_index_cache()
        ready_after = client.get("/ready", headers=_HEADERS)
        restart_mismatch: list[str] = []
        sample = [case for case in cases_for_phase("main") if case.answerable][:_RESTART_SAMPLE]
        for case in sample:
            workspace_id, document_ids = _resolve_scope(corpus, case.scope, deleted)
            results, _elapsed, _extra = _retrieve(
                client, question=case.question, workspace_id=workspace_id, document_ids=document_ids, top_k=top_k
            )
            before = chunk_lists_1.get(case.id, [])
            after = [str(r["chunk_id"]) for r in results or []]
            # Deleted/updated documents legitimately change the result set; compare the rest.
            changed = {document_id_for(n) for n in deleted} | {document_id_for(n) for n in corpus.updates}
            if any(doc in changed for doc in document_ids):
                continue
            if before != after:
                restart_mismatch.append(case.id)
        report["lifecycle"] = {
            "update_index_status": [r["status_code"] for r in update_rows],
            "delete": delete_rows,
            "delete_status_ok": all(r["status_code"] == 200 for r in delete_rows),
            "deleted_chunks_total": sum(int(r["deleted_chunks"] or 0) for r in delete_rows),
            "chunks_listed_after_delete": sum(int(r["chunks_listed_after_delete"] or 0) for r in delete_rows),
            "ready_after_restart": ready_after.status_code == 200 and bool((ready_after.json() or {}).get("ready")),
            "restart_inconsistent_cases": restart_mismatch,
            "restart_inconsistent": len(restart_mismatch),
            "retrieve_failures": len(failures_1) + len(failures_2) + len(failures_u) + len(failures_d),
            "retrieve_failure_details": failures_1 + failures_2 + failures_u + failures_d,
        }
        config = client.get("/api/config", headers=_HEADERS)
        report["config_after"] = config.json() if config.status_code == 200 else {"status_code": config.status_code}

    all_rows = rows + rows_update + rows_delete
    hard: dict[str, Any] = dict(M.hard_counters(all_rows))
    hard["unstable_cases"] = len(unstable)
    hard["unstable_case_ids"] = unstable
    hard["restart_inconsistent"] = len(restart_mismatch)
    hard["index_failures"] = len(report["index"]["failures"])
    hard["retrieve_failures"] = report["lifecycle"]["retrieve_failures"]
    hard["chunks_listed_after_delete"] = report["lifecycle"]["chunks_listed_after_delete"]
    fidelity = F.summarize(fidelity_rows)
    hard["citation_chunks_checked"] = fidelity["checked"]
    hard["citation_wrong_source"] = fidelity["wrong_source"] + fidelity["unattributed_document"]
    hard["citation_wrong_locator"] = fidelity["citation_wrong_locator"]
    hard["citation_label_mismatch"] = fidelity["label_page_mismatch"]
    hard["citation_missing_locator"] = (
        fidelity["missing_page"] + fidelity["missing_section"] + fidelity["unknown_section"]
    )
    report["citation_fidelity"] = fidelity
    report["citation_fidelity_failures"] = [row for row in fidelity_rows if not row["ok"]][:200]
    report["hard"] = hard
    report["aggregate"] = M.aggregate(rows + rows_update)
    report["by_tag"] = M.by_tag(all_rows)
    report["latency_ms"] = {
        "n": len(latencies),
        "median": M.percentile(latencies, 0.5),
        "p95": M.percentile(latencies, 0.95),
        "max": round(max(latencies), 2) if latencies else None,
    }
    report["memory"]["max_rss_mb_after_run"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    report["cases"] = all_rows
    report["failing_cases"] = [
        {
            "id": r["id"],
            "phase": r["phase"],
            "tags": r["tags"],
            "expected_docs": r["expected_docs"],
            "ranked_docs": r["ranked_docs"],
            "top1": r["top1"],
            "expected_locator": r.get("expected_locator"),
            "fidelity_problems": r.get("fidelity_problems"),
            "leaks": r["forbidden_document_leaks"] + r["cross_scope_leaks"] + r["cross_workspace_leaks"],
            "distractor_hits": r["distractor_hits"],
        }
        for r in all_rows
        if (
            r["answerable"]
            and (
                r.get("hit_at_5") == 0.0
                or r.get("citation_source_ok") is False
                or r.get("citation_locator") in ("wrong", "missing")
            )
        )
        or r["forbidden_document_leaks"]
        or r["cross_scope_leaks"]
        or r["cross_workspace_leaks"]
        or r["forbidden_text_hits"]
        or r["stale_deleted_hits"]
        or r["expected_text_found"] is False
        or r.get("fidelity_problems")
    ]
    return report


def report_markdown(report: dict[str, Any]) -> str:
    agg = report["aggregate"]
    hard = report["hard"]
    lines = [
        f"# ControlRoom qualification report — provider `{report['provider']}`",
        "",
        f"- dataset: `{report['dataset_version']}`; top_k={report['top_k']}; engine {report.get('engine_version')}",
        f"- corpus: {report['index']['documents']} documents, {report['index']['chunks_total']} chunks, "
        f"index {report['index']['elapsed_ms_total']} ms total (median {report['index']['elapsed_ms_median']} ms/doc, "
        f"max {report['index']['elapsed_ms_max']} ms)",
        f"- retrieval latency: median {report['latency_ms']['median']} ms, p95 {report['latency_ms']['p95']} ms "
        f"(n={report['latency_ms']['n']}); max RSS {report['memory']['max_rss_mb_after_run']} MB",
        "",
        "## Aggregate (answerable cases)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key in (
        "cases_scored",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "hit_at_8",
        "mrr",
        "ndcg_at_8",
        "exact_identifier_hit_at_1",
        "multi_doc_coverage_at_8",
        "top1_expected_source_rate",
        "expected_locator_ok_rate",
        "duplicate_chunk_rate",
        "duplicate_parent_rate",
        "mean_distinct_docs",
    ):
        lines.append(f"| {key} | {agg.get(key)} |")
    lines += ["", "## Hard counters", "", "| Counter | Value |", "| --- | --- |"]
    for key, value in hard.items():
        if key.endswith("_ids"):
            continue
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## By tag",
        "",
        "| Tag | cases | hit@1 | hit@5 | MRR | nDCG@8 | leaks | failed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for tag, row in report["by_tag"].items():
        leaks = row["forbidden_document_leaks"] + row["cross_scope_leaks"] + row["cross_workspace_leaks"]
        lines.append(
            f"| {tag} | {row['cases']} | {row['hit_at_1']} | {row['hit_at_5']} | {row['mrr']} | {row['ndcg_at_8']} | "
            f"{leaks} | {', '.join(row['failed_case_ids'])} |"
        )
    lines += ["", "## Citation fidelity", "", f"```json\n{json.dumps(report['citation_fidelity'], indent=2)}\n```"]
    lines += ["", "## No-answer behavior", "", f"```json\n{json.dumps(report['no_answer'], indent=2)}\n```", ""]
    lines += ["## Failing cases", ""]
    if not report["failing_cases"]:
        lines.append("none")
    for row in report["failing_cases"]:
        lines.append(
            f"- **{row['id']}** ({row['phase']}; {', '.join(row['tags'])}): expected {row['expected_docs']}; "
            f"ranked {row['ranked_docs'][:5]}; top1 {row['top1']}; "
            f"locator {row['expected_locator']}; leaks {row['leaks']}; distractors {row['distractor_hits']}; "
            f"fidelity problems {row['fidelity_problems']}"
        )
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(report_markdown(report), encoding="utf-8")
