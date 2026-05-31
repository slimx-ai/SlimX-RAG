from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "diff-v1"


def _read_json(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Could not read {path}: {exc}")
        return None
    return data if isinstance(data, dict) else None


def _read_jsonl(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        warnings.append(f"Missing artifact: {path}")
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
                    warnings.append(f"Invalid JSONL in {path} at line {line_no}: {exc}")
                    continue
                if isinstance(rec, dict):
                    records.append(rec)
                else:
                    warnings.append(f"Skipping non-object JSONL record in {path} at line {line_no}")
    except OSError as exc:
        warnings.append(f"Could not read {path}: {exc}")
    return records


def _metadata(rec: dict[str, Any]) -> dict[str, Any]:
    md = rec.get("metadata") or {}
    return md if isinstance(md, dict) else {}


def _doc_label(rec: dict[str, Any]) -> str:
    md = _metadata(rec)
    return str(md.get("kb_relpath") or md.get("source") or md.get("title") or md.get("doc_id") or "")


def _load_docs(path: Path, warnings: list[str]) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for rec in _read_jsonl(path, warnings):
        md = _metadata(rec)
        doc_id = str(md.get("doc_id") or "")
        if not doc_id:
            warnings.append(f"Skipping document without doc_id in {path}")
            continue
        docs[doc_id] = {
            "doc_id": doc_id,
            "content_hash": str(md.get("content_hash") or ""),
            "label": _doc_label(rec) or doc_id,
        }
    return docs


def _load_chunks(path: Path, warnings: list[str]) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    for rec in _read_jsonl(path, warnings):
        md = _metadata(rec)
        chunk_id = str(md.get("chunk_id") or rec.get("chunk_id") or "")
        if not chunk_id:
            warnings.append(f"Skipping chunk without chunk_id in {path}")
            continue
        chunks[chunk_id] = {
            "chunk_id": chunk_id,
            "doc_id": str(md.get("doc_id") or md.get("parent_doc_id") or ""),
            "content_hash": str(md.get("content_hash") or ""),
        }
    return chunks


def _manifest_config(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not manifest:
        return {}
    return {
        "chunk_config": manifest.get("chunk_config"),
        "embedding_config": {
            "provider": manifest.get("embed_provider"),
            "model": manifest.get("embed_model"),
            "dim": manifest.get("embed_dim"),
        },
        "index_backend": manifest.get("index_backend"),
        "hash_policy": manifest.get("hash_policy"),
    }


def _fallback_config(out_dir: Path, warnings: list[str]) -> dict[str, Any]:
    state = _read_json(out_dir / "index_state.json", warnings)
    return {
        "chunk_config": None,
        "embedding_config": (state or {}).get("embed"),
        "index_backend": "unknown",
        "hash_policy": None,
    }


def build_diff(old_dir: Path, new_dir: Path) -> dict[str, Any]:
    old_dir = Path(old_dir)
    new_dir = Path(new_dir)
    warnings: list[str] = []

    old_docs = _load_docs(old_dir / "docs.jsonl", warnings)
    new_docs = _load_docs(new_dir / "docs.jsonl", warnings)
    old_chunks = _load_chunks(old_dir / "chunks.jsonl", warnings)
    new_chunks = _load_chunks(new_dir / "chunks.jsonl", warnings)

    old_doc_ids = set(old_docs)
    new_doc_ids = set(new_docs)
    common_docs = old_doc_ids & new_doc_ids
    changed_doc_ids = sorted(
        doc_id for doc_id in common_docs if old_docs[doc_id].get("content_hash") != new_docs[doc_id].get("content_hash")
    )

    old_manifest = _read_json(old_dir / "manifest.json", warnings)
    new_manifest = _read_json(new_dir / "manifest.json", warnings)
    if not old_manifest or not new_manifest:
        warnings.append("manifest.json missing for one or both builds; using artifact fallback")
    old_config = _manifest_config(old_manifest) if old_manifest else _fallback_config(old_dir, warnings)
    new_config = _manifest_config(new_manifest) if new_manifest else _fallback_config(new_dir, warnings)

    result = {
        "schema_version": SCHEMA_VERSION,
        "old_dir": str(old_dir),
        "new_dir": str(new_dir),
        "documents": {
            "added": [new_docs[d]["label"] for d in sorted(new_doc_ids - old_doc_ids)],
            "deleted": [old_docs[d]["label"] for d in sorted(old_doc_ids - new_doc_ids)],
            "changed": [new_docs[d]["label"] for d in changed_doc_ids],
            "unchanged_count": len(common_docs) - len(changed_doc_ids),
        },
        "chunks": {
            "added": sorted(set(new_chunks) - set(old_chunks)),
            "deleted": sorted(set(old_chunks) - set(new_chunks)),
            "unchanged_count": len(set(old_chunks) & set(new_chunks)),
        },
        "config_changes": {
            "chunk_config_changed": old_config.get("chunk_config") != new_config.get("chunk_config"),
            "embedding_config_changed": old_config.get("embedding_config") != new_config.get("embedding_config"),
            "index_backend_changed": old_config.get("index_backend") != new_config.get("index_backend"),
            "hash_policy_changed": old_config.get("hash_policy") != new_config.get("hash_policy"),
        },
        "warnings": warnings,
    }
    return result


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def format_diff_text(diff: dict[str, Any]) -> str:
    docs = diff["documents"]
    chunks = diff["chunks"]
    cfg = diff["config_changes"]
    lines = [
        f"Documents added: {len(docs['added'])}",
        f"Documents changed: {len(docs['changed'])}",
        f"Documents deleted: {len(docs['deleted'])}",
        "",
        f"Chunks added: {len(chunks['added'])}",
        f"Chunks deleted: {len(chunks['deleted'])}",
        f"Chunks unchanged: {chunks['unchanged_count']}",
        "",
        f"Embedding config changed: {_yes_no(bool(cfg['embedding_config_changed']))}",
        f"Chunk config changed: {_yes_no(bool(cfg['chunk_config_changed']))}",
        f"Index backend changed: {_yes_no(bool(cfg['index_backend_changed']))}",
        f"Hash policy changed: {_yes_no(bool(cfg['hash_policy_changed']))}",
    ]
    if docs["changed"]:
        lines.extend(["", "Changed documents:"])
        lines.extend(f"- {label}" for label in docs["changed"])
    if diff.get("warnings"):
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in diff["warnings"])
    return "\n".join(lines) + "\n"
