from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from slimx_rag.answer import answer
from slimx_rag.chunk import chunk_documents
from slimx_rag.diff import build_diff, format_diff_text
from slimx_rag.embed import embed_chunks, make_embedder
from slimx_rag.eval import load_eval_cases, run_eval, write_eval_report
from slimx_rag.index import make_index_backend
from slimx_rag.ingest.loader import fetch_documents
from slimx_rag.manifest import write_manifest
from slimx_rag.report import build_report, format_report_markdown
from slimx_rag.retrieval import retrieve
from slimx_rag.settings import (
    EMBED_PROVIDERS,
    INDEX_BACKENDS,
    ChunkSettings,
    EmbedSettings,
    IndexingPipelineSettings,
    IndexSettings,
    IngestSettings,
)
from slimx_rag.utils.commons import _read_jsonl_docs, _write_embeddings_jsonl, _write_jsonl

logger = logging.getLogger(__name__)
DEFAULTS = IndexingPipelineSettings()


# =============================================================================
# Small, reusable helpers (keep CLI readable, avoid duplication)
# =============================================================================

def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_backend_config(raw: str) -> dict[str, object]:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in --backend-config: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("--backend-config must be a JSON object (e.g. {\"collection\": \"slimx\"})")
    return obj


def _parse_meta_keep(raw: str) -> list[str] | None:
    keys = [k.strip() for k in (raw or "").split(",") if k.strip()]
    return keys or None


def _resolve_index_path(args_index: Path | None, out_dir: Path) -> Path:
    return args_index or out_dir / DEFAULTS.index_filename


def _resolve_state_path(
    *,
    args_state: Path | None,
    index_path: Path,
    index_settings: IndexSettings,
) -> Path | None:
    # explicit wins; else follow index location (more robust than out_dir-only)
    if args_state is not None:
        return args_state
    if not index_settings.write_state:
        return None
    return index_path.parent / index_settings.state_filename


def _backend_uses_local_index_file(index_settings: IndexSettings) -> bool:
    return index_settings.backend in {"local", "faiss"}


# =============================================================================
# Incremental state scan (kept explicit + well-documented)
# =============================================================================

def _scan_chunks_for_state(in_path: Path) -> dict[str, tuple[str, list[str]]]:
    """
    Build a doc-level "state view" used for incremental indexing.

    Returns:
        doc_id -> (content_hash, [chunk_id, ...])

    Rules:
    - doc_id from metadata["doc_id"] (fallback: metadata["parent_doc_id"])
    - chunk_id from metadata["chunk_id"]
    - content_hash from metadata["content_hash"] (optional)
    - records missing doc_id or chunk_id are skipped; we log one summary warning
    - if content_hash changes for same doc_id, "last non-empty wins"
    """
    if not in_path.exists():
        raise FileNotFoundError(f"Input chunks file not found: {in_path}")

    out: dict[str, tuple[str, list[str]]] = {}
    skipped = missing_doc = missing_chunk = 0

    for d in _read_jsonl_docs(in_path):
        md = d.metadata or {}
        doc_id = str(md.get("doc_id") or md.get("parent_doc_id") or "")
        content_hash = str(md.get("content_hash") or "")
        chunk_id = str(md.get("chunk_id") or "")

        if not doc_id or not chunk_id:
            skipped += 1
            if not doc_id:
                missing_doc += 1
            if not chunk_id:
                missing_chunk += 1
            continue

        h, ids = out.get(doc_id, (content_hash, []))
        if content_hash and h != content_hash:
            h = content_hash
        ids.append(chunk_id)
        out[doc_id] = (h, ids)

    if skipped:
        logger.warning(
            "Skipped %d record(s) in %s (missing doc_id/parent_doc_id: %d, missing chunk_id: %d).",
            skipped,
            in_path,
            missing_doc,
            missing_chunk,
        )
    return out


# =============================================================================
# Core operation (so `index` and `run` share logic without mutating args)
# =============================================================================

def _index_chunks_file(
    *,
    in_chunks_path: Path,
    embed_settings: EmbedSettings,
    index_settings: IndexSettings,
    index_path: Path,
    state_path: Path | None,
    reindex: bool,
    embeddings_out_path: Path | None = None,
) -> tuple[int, int, int]:
    """
    Embed + upsert chunks, with incremental cleanup.

    Returns: (deleted_stale, upserted, total_index_size)
    """
    if not in_chunks_path.exists():
        raise FileNotFoundError(f"Input chunks file not found: {in_chunks_path}")

    idx = make_index_backend(index_path, settings=index_settings, state_path=state_path)
    idx.load()
    idx.set_embed_config(embed_settings)

    current_docs = _scan_chunks_for_state(in_chunks_path)
    deleted = idx.apply_incremental_plan(current_docs=current_docs)

    chunks_iter = _read_jsonl_docs(in_chunks_path)
    items = embed_chunks(chunks_iter, settings=embed_settings)
    if embeddings_out_path is not None:
        materialized = list(items)
        _write_embeddings_jsonl(materialized, embeddings_out_path)
        items = iter(materialized)
    written = idx.upsert(items, skip_existing=not reindex)
    idx.save()
    # Commit state strictly after a successful upsert + save: a crash earlier
    # leaves state behind the backend (a re-run converges), never ahead of it.
    idx.commit_state(current_docs)

    return deleted, written, len(idx)


def _embed_settings_from_state_with_overrides(
    *,
    saved_cfg: dict[str, Any] | None,
    provider: str | None,
    model: str | None,
    hf_model: str | None,
    dim: int | None,
    batch_size: int | None,
    max_chars: int | None,
    normalize_text: bool | None,
    device: str | None = None,
) -> EmbedSettings:
    """
    Query-time embedding settings:
    - start from defaults
    - merge saved state (if present)
    - apply CLI overrides if explicitly set
    """
    base = EmbedSettings()
    kwargs: dict[str, Any] = {f.name: getattr(base, f.name) for f in fields(EmbedSettings)}

    if saved_cfg:
        for k, v in saved_cfg.items():
            if k in kwargs:
                kwargs[k] = v

    # explicit overrides
    if provider is not None:
        kwargs["provider"] = provider
    if model is not None:
        kwargs["model"] = model
    if hf_model is not None:
        kwargs["hf_model"] = hf_model
    if dim is not None:
        kwargs["dim"] = int(dim)
    if batch_size is not None:
        kwargs["batch_size"] = int(batch_size)
    if max_chars is not None:
        kwargs["max_chars"] = None if int(max_chars) == 0 else int(max_chars)
    if normalize_text is not None:
        kwargs["normalize_text"] = bool(normalize_text)
    if device is not None:
        kwargs["device"] = device

    # normalize types that may come back as strings in state
    if kwargs.get("dim") is not None:
        kwargs["dim"] = int(kwargs["dim"])
    if kwargs.get("batch_size") is not None:
        kwargs["batch_size"] = int(kwargs["batch_size"])
    if kwargs.get("max_chars") is not None:
        kwargs["max_chars"] = int(kwargs["max_chars"])

    es = EmbedSettings(**kwargs)
    es.validate()
    return es


# =============================================================================
# Handlers (simple, command-focused)
# =============================================================================

def handle_ingest(args: argparse.Namespace) -> int:
    settings = IndexingPipelineSettings(
        kb_dir=args.kb_dir,
        out_dir=args.out_dir,
        ingest=IngestSettings(glob=args.glob),
    )
    # validate only what matters for ingest + directory sanity
    settings.ingest.validate()
    if not settings.kb_dir.exists() or not settings.kb_dir.is_dir():
        raise ValueError(f"kb_dir must exist and be a directory: {settings.kb_dir}")
    if settings.out_dir.exists() and not settings.out_dir.is_dir():
        raise ValueError(f"out_dir must be a directory: {settings.out_dir}")

    out_path = args.out or settings.out_dir / DEFAULTS.docs_filename
    _ensure_parent_dir(out_path)

    docs = fetch_documents(settings=settings)
    _write_jsonl(docs, out_path)
    logger.info("Wrote %s", out_path)
    return 0


def handle_chunk(args: argparse.Namespace) -> int:
    if not args.in_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.in_path}")

    chunk_settings = ChunkSettings(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunk_settings.validate()

    out_path = args.out or args.out_dir / DEFAULTS.chunks_filename
    _ensure_parent_dir(out_path)

    docs = list(_read_jsonl_docs(args.in_path))
    chunks = chunk_documents(
        docs,
        chunk_size=chunk_settings.chunk_size,
        chunk_overlap=chunk_settings.chunk_overlap,
        separators=chunk_settings.separators,
    )
    _write_jsonl(chunks, out_path)
    logger.info("Wrote %s", out_path)
    return 0


def handle_index(args: argparse.Namespace) -> int:
    if not args.in_path.exists():
        raise FileNotFoundError(f"Input chunks file not found: {args.in_path}")

    embed_settings = EmbedSettings(
        provider=args.embed_provider,
        model=args.embed_model,
        hf_model=args.hf_model,
        dim=args.embed_dim,
        batch_size=args.embed_batch,
        max_chars=(None if args.embed_max_chars == 0 else int(args.embed_max_chars)),
        normalize_text=not args.embed_no_normalize,
        device=getattr(args, "embed_device", None),
    )
    embed_settings.validate()

    index_settings = IndexSettings(
        backend=args.index_backend,
        backend_config=_parse_backend_config(args.backend_config),
        top_k=args.top_k,
        metadata_whitelist=_parse_meta_keep(args.meta_keep),
    )
    index_settings.validate()

    index_path = _resolve_index_path(args.index, args.out_dir)
    state_path = _resolve_state_path(args_state=args.state, index_path=index_path, index_settings=index_settings)

    _ensure_parent_dir(index_path)
    if state_path:
        _ensure_parent_dir(state_path)

    embeddings_out_path = None if args.no_embeddings_out else args.out_dir / DEFAULTS.embeddings_filename
    deleted, written, total = _index_chunks_file(
        in_chunks_path=args.in_path,
        embed_settings=embed_settings,
        index_settings=index_settings,
        index_path=index_path,
        state_path=state_path,
        reindex=bool(args.reindex),
        embeddings_out_path=embeddings_out_path,
    )

    logger.info("Index: %s", index_path)
    if state_path:
        logger.info("State: %s", state_path)
    logger.info("Deleted stale: %s, Upserted: %s (Total: %s)", deleted, written, total)
    return 0


def handle_run(args: argparse.Namespace) -> int:
    embed_settings = EmbedSettings(
        provider=args.embed_provider,
        model=args.embed_model,
        hf_model=args.hf_model,
        dim=args.embed_dim,
        batch_size=args.embed_batch,
        max_chars=(None if args.embed_max_chars == 0 else int(args.embed_max_chars)),
        normalize_text=not args.embed_no_normalize,
        device=getattr(args, "embed_device", None),
    )
    index_settings = IndexSettings(
        backend=args.index_backend,
        backend_config=_parse_backend_config(args.backend_config),
        top_k=args.top_k,
        metadata_whitelist=_parse_meta_keep(args.meta_keep),
    )
    settings = IndexingPipelineSettings(
        kb_dir=args.kb_dir,
        out_dir=args.out_dir,
        ingest=IngestSettings(glob=args.glob),
        chunk=ChunkSettings(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap),
        embed=embed_settings,
        index=index_settings,
    )
    settings.validate()

    settings.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Ingest
    docs = fetch_documents(settings=settings)
    _write_jsonl(docs, settings.docs_path)
    logger.info("Wrote %s", settings.docs_path)

    # 2) Chunk
    chunks = chunk_documents(
        docs,
        chunk_size=settings.chunk.chunk_size,
        chunk_overlap=settings.chunk.chunk_overlap,
        separators=settings.chunk.separators,
    )
    _write_jsonl(chunks, settings.chunks_path)
    logger.info("Wrote %s", settings.chunks_path)

    # 3) Index
    index_path = _resolve_index_path(args.index, settings.out_dir)
    state_path = _resolve_state_path(args_state=args.state, index_path=index_path, index_settings=index_settings)

    embeddings_out_path = None if args.no_embeddings_out else settings.embeddings_path
    deleted, written, total = _index_chunks_file(
        in_chunks_path=settings.chunks_path,
        embed_settings=embed_settings,
        index_settings=index_settings,
        index_path=index_path,
        state_path=state_path,
        reindex=bool(args.reindex),
        embeddings_out_path=embeddings_out_path,
    )

    logger.info("Index: %s", index_path)
    if state_path:
        logger.info("State: %s", state_path)
    logger.info("Deleted stale: %s, Upserted: %s (Total: %s)", deleted, written, total)
    if getattr(args, "write_manifest", False):
        manifest_path = write_manifest(settings.out_dir, settings=settings)
        logger.info("Manifest: %s", manifest_path)
    return 0


def handle_manifest(args: argparse.Namespace) -> int:
    path = write_manifest(args.out_dir)
    print(str(path))
    return 0


def handle_diff(args: argparse.Namespace) -> int:
    result = build_diff(args.old_dir, args.new_dir)
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_diff_text(result), end="")
    return 0


def handle_report(args: argparse.Namespace) -> int:
    result = build_report(args.out_dir)
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_report_markdown(result), end="")
    return 0


def handle_query(args: argparse.Namespace) -> int:
    index_settings = IndexSettings(
        backend=args.index_backend,
        backend_config=_parse_backend_config(args.backend_config),
        top_k=args.top_k,
        metadata_whitelist=_parse_meta_keep(args.meta_keep),
    )
    index_settings.validate()

    index_path = _resolve_index_path(args.index, args.out_dir)
    state_path = _resolve_state_path(args_state=args.state, index_path=index_path, index_settings=index_settings)

    if _backend_uses_local_index_file(index_settings) and not index_path.exists():
        raise FileNotFoundError(f"Index not found at {index_path}")

    idx = make_index_backend(index_path, settings=index_settings, state_path=state_path)
    idx.load()

    saved_cfg = {}
    if getattr(idx, "state", None) is not None and getattr(idx.state, "embed", None) is not None:
        saved_cfg = idx.state.embed or {}

    embed_settings = _embed_settings_from_state_with_overrides(
        saved_cfg=saved_cfg,
        provider=args.embed_provider,
        model=args.embed_model,
        hf_model=args.hf_model,
        dim=args.embed_dim,
        batch_size=args.embed_batch,
        max_chars=args.embed_max_chars,
        normalize_text=args.embed_normalize,
        device=getattr(args, "embed_device", None),
    )

    emb = make_embedder(embed_settings)
    qvec = emb.embed_texts([args.q])[0]

    results = idx.query(list(map(float, qvec)), top_k=args.k)
    for r in results:
        print(
            json.dumps(
                {"chunk_id": r.chunk_id, "score": r.score, "text": r.text, "metadata": r.metadata},
                ensure_ascii=False,
            )
        )
    return 0


def _make_embed_settings_for_query(args: argparse.Namespace, idx_state_embed: dict[str, Any] | None) -> EmbedSettings:
    return _embed_settings_from_state_with_overrides(
        saved_cfg=idx_state_embed or {},
        provider=args.embed_provider,
        model=args.embed_model,
        hf_model=args.hf_model,
        dim=args.embed_dim,
        batch_size=args.embed_batch,
        max_chars=args.embed_max_chars,
        normalize_text=args.embed_normalize,
        device=getattr(args, "embed_device", None),
    )


def _load_embed_state(index_path: Path, index_settings: IndexSettings, state_path: Path | None) -> dict[str, Any]:
    idx = make_index_backend(index_path, settings=index_settings, state_path=state_path)
    idx.load()
    return dict(getattr(idx.state, "embed", None) or {})


def handle_retrieve(args: argparse.Namespace) -> int:
    index_settings = IndexSettings(
        backend=args.index_backend,
        backend_config=_parse_backend_config(args.backend_config),
        top_k=args.top_k,
        metadata_whitelist=_parse_meta_keep(args.meta_keep),
    )
    index_settings.validate()
    index_path = _resolve_index_path(args.index, args.out_dir)
    state_path = _resolve_state_path(args_state=args.state, index_path=index_path, index_settings=index_settings)
    if _backend_uses_local_index_file(index_settings) and not index_path.exists():
        raise FileNotFoundError(f"Index not found at {index_path}")

    embed_settings = _make_embed_settings_for_query(args, _load_embed_state(index_path, index_settings, state_path))
    result = retrieve(
        args.q,
        index_path=index_path,
        embed_settings=embed_settings,
        index_settings=index_settings,
        state_path=state_path,
        top_k=args.k,
    )
    for chunk in result.chunks:
        print(json.dumps(asdict(chunk), ensure_ascii=False))
    return 0


def handle_ask(args: argparse.Namespace) -> int:
    index_settings = IndexSettings(
        backend=args.index_backend,
        backend_config=_parse_backend_config(args.backend_config),
        top_k=args.top_k,
        metadata_whitelist=_parse_meta_keep(args.meta_keep),
    )
    index_settings.validate()
    index_path = _resolve_index_path(args.index, args.out_dir)
    state_path = _resolve_state_path(args_state=args.state, index_path=index_path, index_settings=index_settings)
    if _backend_uses_local_index_file(index_settings) and not index_path.exists():
        raise FileNotFoundError(f"Index not found at {index_path}")

    embed_settings = _make_embed_settings_for_query(args, _load_embed_state(index_path, index_settings, state_path))
    retrieval = retrieve(
        args.q,
        index_path=index_path,
        embed_settings=embed_settings,
        index_settings=index_settings,
        state_path=state_path,
        top_k=args.k,
    )
    result = answer(
        args.q,
        retrieval,
        model=args.model,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        max_context_chars=args.max_context_chars,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def handle_eval(args: argparse.Namespace) -> int:
    index_settings = IndexSettings(
        backend=args.index_backend,
        backend_config=_parse_backend_config(args.backend_config),
        top_k=args.top_k,
        metadata_whitelist=_parse_meta_keep(args.meta_keep),
    )
    index_settings.validate()
    index_path = _resolve_index_path(args.index, args.out_dir)
    state_path = _resolve_state_path(args_state=args.state, index_path=index_path, index_settings=index_settings)
    embed_settings = _make_embed_settings_for_query(args, _load_embed_state(index_path, index_settings, state_path))
    cases = load_eval_cases(args.dataset)
    report = run_eval(
        cases,
        index_path=index_path,
        embed_settings=embed_settings,
        index_settings=index_settings,
        model=args.model,
        state_path=state_path,
        top_k=args.k,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        max_context_chars=args.max_context_chars,
    )
    write_eval_report(report, args.out)
    logger.info("Wrote %s", args.out)
    return 0


def handle_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except Exception as e:  # pragma: no cover
        raise RuntimeError("`slimx-rag serve` requires the demo extra: uv sync --extra demo") from e
    uvicorn.run("slimx_rag.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


# =============================================================================
# Argument parsing (shared parents = low duplication, still command-focused)
# =============================================================================

def _add_out_dir(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out-dir", type=Path, default=DEFAULTS.out_dir, help="Base output directory")


def _add_ingest_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--kb-dir", type=Path, default=DEFAULTS.kb_dir, help="Knowledge base directory")
    p.add_argument("--glob", type=str, default=DEFAULTS.ingest.glob, help="File pattern to match")


def _add_chunk_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--chunk-size", type=int, default=DEFAULTS.chunk.chunk_size)
    p.add_argument("--chunk-overlap", type=int, default=DEFAULTS.chunk.chunk_overlap)


def _add_embed_args_indexing(p: argparse.ArgumentParser) -> None:
    d = DEFAULTS.embed
    g = p.add_argument_group("Embedding options")
    g.add_argument("--embed-provider", type=str, default=d.provider, choices=EMBED_PROVIDERS)
    g.add_argument("--embed-model", type=str, default=d.model, help="OpenAI embedding model")
    g.add_argument("--hf-model", type=str, default=d.hf_model, help="SentenceTransformers model id")
    g.add_argument("--embed-dim", type=int, default=d.dim, help="Hash dim (and optional validation)")
    g.add_argument("--embed-batch", type=int, default=d.batch_size)
    g.add_argument("--embed-max-chars", type=int, default=(d.max_chars or 0), help="0 disables")
    g.add_argument("--embed-no-normalize", action="store_true")
    g.add_argument(
        "--embed-device",
        type=str,
        default=d.device,
        help="Torch device for the hf embedder (cpu/cuda/cuda:0/mps); default auto-selects",
    )


def _add_embed_overrides_query(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Query embedding overrides (optional)")
    # defaults are None so we can tell if user explicitly overrode state
    g.add_argument("--embed-provider", type=str, default=None, choices=EMBED_PROVIDERS)
    g.add_argument("--embed-model", type=str, default=None)
    g.add_argument("--hf-model", type=str, default=None)
    g.add_argument("--embed-dim", type=int, default=None)
    g.add_argument("--embed-batch", type=int, default=None)
    g.add_argument("--embed-max-chars", type=int, default=None, help="0 disables; if omitted uses state/defaults")
    g.add_argument("--embed-device", type=str, default=None, help="Override hf embedder device")

    norm = g.add_mutually_exclusive_group()
    norm.add_argument("--embed-normalize", dest="embed_normalize", action="store_true", default=None)
    norm.add_argument("--embed-no-normalize", dest="embed_normalize", action="store_false", default=None)


def _add_index_settings_args(p: argparse.ArgumentParser) -> None:
    d = DEFAULTS.index
    g = p.add_argument_group("Index options")
    g.add_argument("--index-backend", type=str, default=d.backend, choices=INDEX_BACKENDS)
    g.add_argument("--backend-config", type=str, default="", help="Backend config JSON (object)")
    g.add_argument("--top-k", type=int, default=d.top_k, help="Default k used by the backend/settings")
    g.add_argument("--meta-keep", type=str, default="", help="Comma-separated metadata keys to keep")
    g.add_argument("--reindex", action="store_true", help="Overwrite/recompute existing chunk_ids")
    g.add_argument("--no-embeddings-out", action="store_true", help="Do not write embeddings.jsonl")


def _add_index_files_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--index", type=Path, default=None, help="Index path (default: {out-dir}/index.jsonl)")
    p.add_argument("--state", type=Path, default=None, help="State path (default: next to index when enabled)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="slimx-rag", description="SlimX-RAG CLI")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging (includes tracebacks on errors)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_out = argparse.ArgumentParser(add_help=False)
    _add_out_dir(p_out)

    p_ing = argparse.ArgumentParser(add_help=False)
    _add_ingest_args(p_ing)

    p_chk = argparse.ArgumentParser(add_help=False)
    _add_chunk_args(p_chk)

    p_emb = argparse.ArgumentParser(add_help=False)
    _add_embed_args_indexing(p_emb)

    p_idx = argparse.ArgumentParser(add_help=False)
    _add_index_settings_args(p_idx)
    _add_index_files_args(p_idx)

    # ingest
    pi = sub.add_parser("ingest", parents=[p_out, p_ing], help="Load KB docs -> docs.jsonl")
    pi.add_argument("--out", type=Path, default=None, help="Output file (default: {out-dir}/docs.jsonl)")
    pi.set_defaults(func=handle_ingest)

    # chunk
    pc = sub.add_parser("chunk", parents=[p_out, p_chk], help="Chunk docs.jsonl -> chunks.jsonl")
    pc.add_argument("--in", dest="in_path", type=Path, required=True, help="Input docs.jsonl")
    pc.add_argument("--out", type=Path, default=None, help="Output file (default: {out-dir}/chunks.jsonl)")
    pc.set_defaults(func=handle_chunk)

    # index
    px = sub.add_parser("index", parents=[p_out, p_emb, p_idx], help="Embed + index chunks.jsonl")
    px.add_argument("--in", dest="in_path", type=Path, required=True, help="Input chunks.jsonl")
    px.set_defaults(func=handle_index)

    # query
    pq = sub.add_parser("query", parents=[p_out, p_idx], help="Query an index")
    pq.add_argument("--q", type=str, required=True, help="Query text")
    pq.add_argument("--k", type=int, default=DEFAULTS.index.top_k, help="Top-k results")
    # query-specific embed overrides (state-first)
    _add_embed_overrides_query(pq)
    pq.set_defaults(func=handle_query)

    # retrieve
    prt = sub.add_parser("retrieve", parents=[p_out, p_idx], help="Retrieve ranked chunks from an index")
    prt.add_argument("--q", type=str, required=True, help="Query text")
    prt.add_argument("--k", type=int, default=DEFAULTS.index.top_k, help="Top-k results")
    _add_embed_overrides_query(prt)
    prt.set_defaults(func=handle_retrieve)

    # ask
    pa = sub.add_parser("ask", parents=[p_out, p_idx], help="Retrieve + generate a cited answer")
    pa.add_argument("--q", type=str, required=True, help="Question")
    pa.add_argument("--k", type=int, default=DEFAULTS.index.top_k, help="Top-k results")
    pa.add_argument("--model", type=str, default="fake:grounded", help="SlimX model id, e.g. openai:gpt-4.1-mini")
    pa.add_argument("--timeout", type=float, default=None,
                    help="LLM request timeout in seconds; Ollama defaults to 180")
    pa.add_argument("--max-tokens", type=int, default=None, help="LLM output token limit; Ollama defaults to 256")
    pa.add_argument("--max-context-chars", type=int, default=None,
                    help="Max retrieved context chars sent to the LLM; Ollama defaults to 3000")
    _add_embed_overrides_query(pa)
    pa.set_defaults(func=handle_ask)

    # eval
    pe = sub.add_parser("eval", parents=[p_out, p_idx], help="Run a demo evaluation dataset")
    pe.add_argument("--dataset", type=Path, required=True, help="JSONL evaluation cases")
    pe.add_argument("--out", type=Path, default=Path("output/eval_report.md"), help="Markdown or JSON report path")
    pe.add_argument("--k", type=int, default=DEFAULTS.index.top_k, help="Top-k results")
    pe.add_argument("--model", type=str, default="fake:grounded")
    pe.add_argument("--timeout", type=float, default=None,
                    help="LLM request timeout in seconds; Ollama defaults to 180")
    pe.add_argument("--max-tokens", type=int, default=None, help="LLM output token limit; Ollama defaults to 256")
    pe.add_argument("--max-context-chars", type=int, default=None,
                    help="Max retrieved context chars sent to the LLM; Ollama defaults to 3000")
    _add_embed_overrides_query(pe)
    pe.set_defaults(func=handle_eval)

    # serve
    ps = sub.add_parser("serve", help="Start the customer demo API/UI")
    ps.add_argument("--host", default="0.0.0.0")
    ps.add_argument("--port", type=int, default=8080)
    ps.add_argument("--reload", action="store_true")
    ps.set_defaults(func=handle_serve)

    # manifest
    pm = sub.add_parser("manifest", parents=[p_out], help="Write manifest.json for an output directory")
    pm.set_defaults(func=handle_manifest)

    # diff
    pd = sub.add_parser("diff", help="Compare two SlimX-RAG output directories")
    pd.add_argument("old_dir", type=Path, help="Older output directory")
    pd.add_argument("new_dir", type=Path, help="Newer output directory")
    pd.add_argument("--format", choices=("text", "json"), default="text", help="Output format")
    pd.set_defaults(func=handle_diff)

    # report
    prep = sub.add_parser("report", parents=[p_out], help="Summarize RAG output quality and metadata coverage")
    prep.add_argument("--format", choices=("json", "markdown"), default="markdown", help="Output format")
    prep.set_defaults(func=handle_report)

    # run
    pr = sub.add_parser("run", parents=[p_out, p_ing, p_chk, p_emb, p_idx], help="ingest -> chunk -> index")
    pr.add_argument("--write-manifest", action="store_true", help="Write manifest.json after a successful run")
    pr.set_defaults(func=handle_run)

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        args = build_parser().parse_args(argv)
        if getattr(args, "verbose", False):
            logging.getLogger().setLevel(logging.DEBUG)
        return int(args.func(args))
    except KeyboardInterrupt:
        logger.error("Interrupted.")
        return 130
    except (ValueError, FileNotFoundError, NotADirectoryError) as e:
        # User-input errors (bad paths, bad config, malformed files): exit 2 like argparse.
        logger.error("Error: %s", e)
        logger.debug("Traceback:", exc_info=True)
        return 2
    except Exception as e:
        logger.error("Error: %s", e)
        logger.debug("Traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
