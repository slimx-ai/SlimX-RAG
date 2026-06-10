from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from slimx_rag.core.hashing import DEFAULT_HASH_POLICY
from slimx_rag.settings import ChunkSettings, EmbedSettings, IndexingPipelineSettings
from slimx_rag.utils.commons import _atomic_write_text

SCHEMA_VERSION = "manifest-v1"
MANIFEST_FILENAME = "manifest.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _project_version() -> str:
    for name in ("slimx-rag", "slimx_rag"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return "unknown"


def _count_jsonl(path: Path, warnings: list[str]) -> int | None:
    if not path.exists():
        return None
    count = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    warnings.append(f"Invalid JSONL record in {path.name} at line {line_no}")
                    continue
                count += 1
    except OSError as exc:
        warnings.append(f"Could not read {path.name}: {exc}")
        return None
    return count


def _read_json(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Could not read {path.name}: {exc}")
        return None
    if not isinstance(data, dict):
        warnings.append(f"{path.name} is not a JSON object")
        return None
    return data


def _file_entry(path: Path, rel_path: str, warnings: list[str], *, jsonl: bool) -> dict[str, Any]:
    exists = path.exists()
    entry: dict[str, Any] = {"path": rel_path, "exists": exists}
    if jsonl:
        entry["record_count"] = _count_jsonl(path, warnings) if exists else 0
    if not exists:
        warnings.append(f"Missing optional artifact: {rel_path}")
    return entry


def _chunk_config(settings: ChunkSettings | None) -> dict[str, Any]:
    settings = settings or ChunkSettings()
    return {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "separators": list(settings.separators),
    }


def _embedding_config_from_state(state: dict[str, Any] | None, settings: EmbedSettings | None) -> dict[str, Any]:
    embed = dict((state or {}).get("embed") or {})
    if settings is not None:
        embed = {
            "provider": settings.provider,
            "model": settings.model,
            "hf_model": settings.hf_model,
            "dim": settings.dim,
            "batch_size": settings.batch_size,
            "normalize_text": settings.normalize_text,
            "max_chars": settings.max_chars,
            **embed,
        }
    return embed


def build_manifest(
    out_dir: Path,
    *,
    settings: IndexingPipelineSettings | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a best-effort manifest from existing SlimX-RAG output artifacts."""
    out_dir = Path(out_dir)
    defaults = IndexingPipelineSettings(out_dir=out_dir)
    settings = settings or defaults
    warnings: list[str] = []

    docs_rel = settings.docs_filename
    chunks_rel = settings.chunks_filename
    embeddings_rel = settings.embeddings_filename
    index_rel = settings.index_filename
    state_rel = settings.index.state_filename

    docs_entry = _file_entry(out_dir / docs_rel, docs_rel, warnings, jsonl=True)
    chunks_entry = _file_entry(out_dir / chunks_rel, chunks_rel, warnings, jsonl=True)
    embeddings_entry = _file_entry(out_dir / embeddings_rel, embeddings_rel, warnings, jsonl=True)
    index_entry = _file_entry(out_dir / index_rel, index_rel, warnings, jsonl=True)
    state_entry = _file_entry(out_dir / state_rel, state_rel, warnings, jsonl=False)
    state = _read_json(out_dir / state_rel, warnings)

    embed = _embedding_config_from_state(state, settings.embed if settings else None)
    if not embed:
        warnings.append("Embedding configuration unavailable; using unknown/null values")

    index_backend = settings.index.backend if settings else "unknown"
    if state and state.get("backend"):
        index_backend = str(state.get("backend"))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at or _utc_now_iso(),
        "slimx_rag_version": _project_version(),
        "out_dir": str(out_dir),
        "doc_count": docs_entry.get("record_count", 0),
        "chunk_count": chunks_entry.get("record_count", 0),
        "embedding_count": embeddings_entry.get("record_count", 0),
        "index_item_count": index_entry.get("record_count", 0),
        "embed_provider": embed.get("provider", "unknown"),
        "embed_model": embed.get("model") or embed.get("hf_model") or "unknown",
        "embed_dim": embed.get("dim"),
        "index_backend": index_backend,
        "chunk_config": _chunk_config(settings.chunk if settings else None),
        "hash_policy": DEFAULT_HASH_POLICY.as_manifest_dict(),
        "files": {
            "docs_jsonl": docs_entry,
            "chunks_jsonl": chunks_entry,
            "embeddings_jsonl": embeddings_entry,
            "index_jsonl": index_entry,
            "index_state_json": state_entry,
        },
        "warnings": warnings,
    }
    return manifest


def write_manifest(
    out_dir: Path,
    *,
    settings: IndexingPipelineSettings | None = None,
    created_at: str | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(out_dir, settings=settings, created_at=created_at)
    path = out_dir / MANIFEST_FILENAME
    _atomic_write_text(path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return path
