from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

SCHEMA_VERSION = "report-v1"
NEAR_EMPTY_CHARS = 10


def _read_json(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        warnings.append(f"Missing {path.name}")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Could not read {path.name}: {exc}")
        return None
    return data if isinstance(data, dict) else None


def _read_jsonl(path: Path, warnings: list[str], *, optional: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if not optional:
            warnings.append(f"Missing required artifact: {path.name}")
        else:
            warnings.append(f"Missing optional artifact: {path.name}")
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    warnings.append(f"Invalid JSONL in {path.name} line {line_no}: {exc}")
                    continue
                if isinstance(rec, dict):
                    records.append(rec)
    except OSError as exc:
        warnings.append(f"Could not read {path.name}: {exc}")
    return records


def _metadata(rec: dict[str, Any]) -> dict[str, Any]:
    md = rec.get("metadata") or {}
    return md if isinstance(md, dict) else {}


def _text(rec: dict[str, Any]) -> str:
    return str(rec.get("page_content") or rec.get("text") or "")


def _chunk_id(rec: dict[str, Any]) -> str:
    md = _metadata(rec)
    return str(md.get("chunk_id") or rec.get("chunk_id") or "")


def _doc_id(rec: dict[str, Any]) -> str:
    md = _metadata(rec)
    return str(md.get("doc_id") or md.get("parent_doc_id") or "")


def _relpath(rec: dict[str, Any]) -> str:
    md = _metadata(rec)
    return str(md.get("kb_relpath") or md.get("parent_kb_relpath") or md.get("source") or md.get("title") or "")


def _doc_type(rec: dict[str, Any]) -> str:
    md = _metadata(rec)
    return str(md.get("doc_type") or md.get("parent_doc_type") or "")


def _content_hash(rec: dict[str, Any]) -> str:
    md = _metadata(rec)
    return str(md.get("content_hash") or "")


def _chunk_stats(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"min_len": 0, "max_len": 0, "avg_len": 0.0, "median_len": 0}
    return {
        "min_len": min(lengths),
        "max_len": max(lengths),
        "avg_len": sum(lengths) / len(lengths),
        "median_len": median(lengths),
    }


def _top_records(
    records: list[dict[str, Any]], *, key_fn, limit: int = 5, reverse: bool = True
) -> list[dict[str, Any]]:
    return sorted(records, key=key_fn, reverse=reverse)[:limit]


def _coverage(chunks: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total_chunks": len(chunks),
        "chunk_id": sum(1 for c in chunks if _chunk_id(c)),
        "doc_id_or_parent_doc_id": sum(1 for c in chunks if _doc_id(c)),
        "kb_relpath_or_parent_kb_relpath": sum(1 for c in chunks if _relpath(c)),
        "content_hash": sum(1 for c in chunks if _content_hash(c)),
        "doc_type_or_parent_doc_type": sum(1 for c in chunks if _doc_type(c)),
    }


def _duplicate_values(values: list[str]) -> list[dict[str, Any]]:
    counts = Counter(v for v in values if v)
    return [{"value": value, "count": count} for value, count in sorted(counts.items()) if count > 1]


def build_report(out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    warnings: list[str] = []
    manifest_path = out_dir / "manifest.json"
    manifest = _read_json(manifest_path, warnings) if manifest_path.exists() else None
    if manifest is None:
        warnings.append("Missing manifest.json; report built from raw artifacts")

    docs = _read_jsonl(out_dir / "docs.jsonl", warnings, optional=False)
    chunks = _read_jsonl(out_dir / "chunks.jsonl", warnings, optional=False)
    embeddings = _read_jsonl(out_dir / "embeddings.jsonl", warnings, optional=True)
    index_items = _read_jsonl(out_dir / "index.jsonl", warnings, optional=True)
    index_state = _read_json(out_dir / "index_state.json", warnings)
    if index_state is None:
        warnings.append("Missing index state; backend and embedding config may be incomplete")

    chunk_lengths = [len(_text(c)) for c in chunks]
    chunk_ids = [_chunk_id(c) for c in chunks]
    duplicate_chunk_ids = _duplicate_values(chunk_ids)
    duplicate_texts = _duplicate_values([_text(c) for c in chunks])

    empty_chunks = [
        {"chunk_id": _chunk_id(c), "doc_id": _doc_id(c), "text_len": len(_text(c))}
        for c in chunks
        if len(_text(c).strip()) <= NEAR_EMPTY_CHARS
    ]

    if not docs:
        warnings.append("No documents found")
    if not chunks:
        warnings.append("No chunks found")
    if any(not cid for cid in chunk_ids):
        warnings.append("One or more chunks are missing chunk_id")
    if duplicate_chunk_ids:
        warnings.append("Duplicate chunk IDs detected")
    if duplicate_texts:
        warnings.append("Duplicate chunk text detected")

    summary = {
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "embedding_count": len(embeddings),
        "index_item_count": len(index_items),
    }
    if manifest:
        for key in ("doc_count", "chunk_count", "embedding_count", "index_item_count"):
            if manifest.get(key) is not None and manifest.get(key) != summary[key]:
                warnings.append(
                    f"Manifest count mismatch for {key}: manifest={manifest.get(key)} actual={summary[key]}"
                )

    largest_documents = [
        {
            "doc_id": str(_metadata(d).get("doc_id") or ""),
            "kb_relpath": _relpath(d),
            "content_len": int(_metadata(d).get("content_len") or len(_text(d))),
        }
        for d in _top_records(docs, key_fn=lambda d: int(_metadata(d).get("content_len") or len(_text(d))))
    ]
    smallest_chunks = [
        {"chunk_id": _chunk_id(c), "doc_id": _doc_id(c), "text_len": len(_text(c)), "kb_relpath": _relpath(c)}
        for c in _top_records(chunks, key_fn=lambda c: len(_text(c)), reverse=False)
    ]

    duplicate_text_groups: list[dict[str, Any]] = []
    text_to_ids: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        text_to_ids[_text(c)].append(_chunk_id(c))
    for text, ids in sorted(text_to_ids.items()):
        if text and len(ids) > 1:
            duplicate_text_groups.append({"text_preview": text[:80], "count": len(ids), "chunk_ids": ids})

    report = {
        "schema_version": SCHEMA_VERSION,
        "summary": summary,
        "document_inventory": [
            {
                "doc_id": str(_metadata(d).get("doc_id") or ""),
                "kb_relpath": _relpath(d),
                "content_len": int(_metadata(d).get("content_len") or len(_text(d))),
                "doc_type": _doc_type(d),
            }
            for d in docs
        ],
        "chunk_stats": _chunk_stats(chunk_lengths),
        "largest_documents": largest_documents,
        "smallest_chunks": smallest_chunks,
        "empty_or_near_empty_chunks": empty_chunks,
        "duplicates": {"duplicate_chunk_ids": duplicate_chunk_ids, "duplicate_chunk_texts": duplicate_text_groups},
        "metadata_coverage": _coverage(chunks),
        "config": {
            "embedding": (
                manifest
                and {
                    "provider": manifest.get("embed_provider"),
                    "model": manifest.get("embed_model"),
                    "dim": manifest.get("embed_dim"),
                }
            )
            or (index_state or {}).get("embed", {}),
            "backend": {"index_backend": (manifest or {}).get("index_backend", "unknown")},
            "hash_policy": (manifest or {}).get("hash_policy", {}),
        },
        "warnings": warnings,
    }
    return report


def _md_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["_None._"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return lines


def format_report_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    stats = report["chunk_stats"]
    lines = [
        "# SlimX-RAG RAGOps Report",
        "",
        "## Build summary",
        f"- Documents: {summary['doc_count']}",
        f"- Chunks: {summary['chunk_count']}",
        f"- Embeddings: {summary['embedding_count']}",
        f"- Index items: {summary['index_item_count']}",
        "",
        "## Document inventory",
        *_md_table(report.get("document_inventory", [])[:10], ["kb_relpath", "content_len", "doc_type"]),
        "",
        "## Chunk statistics",
        f"- Min length: {stats['min_len']}",
        f"- Max length: {stats['max_len']}",
        f"- Average length: {stats['avg_len']:.2f}",
        f"- Median length: {stats['median_len']}",
        "",
        "## Largest documents",
        *_md_table(report.get("largest_documents", []), ["kb_relpath", "content_len"]),
        "",
        "## Smallest chunks",
        *_md_table(report.get("smallest_chunks", []), ["chunk_id", "text_len", "kb_relpath"]),
        "",
        "## Duplicate chunks",
        f"- Duplicate chunk IDs: {len(report['duplicates']['duplicate_chunk_ids'])}",
        f"- Duplicate chunk texts: {len(report['duplicates']['duplicate_chunk_texts'])}",
        "",
        "## Empty or near-empty chunks",
        *_md_table(report.get("empty_or_near_empty_chunks", []), ["chunk_id", "doc_id", "text_len"]),
        "",
        "## Metadata coverage",
    ]
    for key, value in report.get("metadata_coverage", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Backend information"])
    for key, value in report.get("config", {}).get("backend", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Embedding configuration"])
    for key, value in report.get("config", {}).get("embedding", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Hash policy"])
    for key, value in report.get("config", {}).get("hash_policy", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings"])
    warnings = report.get("warnings", [])
    lines.extend([f"- {w}" for w in warnings] if warnings else ["_None._"])
    return "\n".join(lines) + "\n"
