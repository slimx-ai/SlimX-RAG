from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

from langchain_core.documents import Document

from slimx_rag.ingest.loader import fetch_documents
from slimx_rag.chunk import chunk_documents
from slimx_rag.embed import embed_chunks, make_embedder
from slimx_rag.index import make_index_backend
from slimx_rag.settings import ChunkSettings, EmbedSettings, IndexingPipelineSettings, IngestSettings, IndexSettings


def _write_jsonl(docs: Iterable[Document], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs:
            rec = {"page_content": d.page_content, "metadata": d.metadata}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl_docs(in_path: Path) -> Iterator[Document]:
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield Document(page_content=rec.get("page_content", "") or "", metadata=rec.get("metadata", {}) or {})


def _scan_chunks_for_state(in_path: Path) -> Dict[str, Tuple[str, List[str]]]:
    """Return doc_id -> (content_hash, [chunk_ids]) from chunks.jsonl."""
    out: Dict[str, Tuple[str, List[str]]] = {}
    for d in _read_jsonl_docs(in_path):
        md = d.metadata or {}
        doc_id = str(md.get("doc_id") or md.get("parent_doc_id") or "")
        content_hash = str(md.get("content_hash") or "")
        chunk_id = str(md.get("chunk_id") or "")
        if not doc_id or not chunk_id:
            # skip chunks that cannot participate in incremental indexing
            continue
        if doc_id not in out:
            out[doc_id] = (content_hash, [chunk_id])
        else:
            h, ids = out[doc_id]
            # Prefer first seen hash, but if inconsistent, last wins (should not happen)
            if content_hash and h != content_hash:
                h = content_hash
            ids.append(chunk_id)
            out[doc_id] = (h, ids)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="slimx-rag", description="SlimX-RAG CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ingest
    pi = sub.add_parser("ingest", help="Load KB documents and write docs.jsonl")
    pi.add_argument("--kb-dir", type=Path, default=IndexingPipelineSettings().kb_dir)
    pi.add_argument("--glob", type=str, default=IngestSettings().glob)
    pi.add_argument("--out", type=Path, default=IndexingPipelineSettings().docs_path)

    # chunk
    pc = sub.add_parser("chunk", help="Chunk docs.jsonl into chunks.jsonl")
    pc.add_argument("--in", dest="in_path", type=Path, required=True)
    pc.add_argument("--out", type=Path, default=IndexingPipelineSettings().chunks_path)
    pc.add_argument("--chunk-size", type=int, default=ChunkSettings().chunk_size)
    pc.add_argument("--chunk-overlap", type=int, default=ChunkSettings().chunk_overlap)

    # index
    px = sub.add_parser("index", help="Embed chunks.jsonl and build/update a local index.jsonl")
    px.add_argument("--in", dest="in_path", type=Path, required=True)
    px.add_argument("--index", type=Path, default=IndexingPipelineSettings().index_path)
    px.add_argument("--state", type=Path, default=None)
    px.add_argument("--index-backend", type=str, default=IndexSettings().backend, choices=["local", "faiss", "qdrant", "pgvector"])
    px.add_argument("--backend-config", type=str, default="", help="JSON string with backend-specific config (e.g., {\"collection\": \"slimx\"})")
    px.add_argument("--reindex", action="store_true", help="Re-embed and overwrite existing chunk_ids")
    px.add_argument("--top-k", type=int, default=IndexSettings().top_k)
    px.add_argument("--meta-keep", type=str, default="", help="Comma-separated metadata keys to keep (optional)")

    # embed options
    px.add_argument("--embed-provider", type=str, default=EmbedSettings().provider, choices=["hash", "openai", "hf"])
    px.add_argument("--embed-model", type=str, default=EmbedSettings().model, help="OpenAI embedding model")
    px.add_argument("--hf-model", type=str, default=EmbedSettings().hf_model, help="SentenceTransformers model id")
    px.add_argument("--embed-dim", type=int, default=EmbedSettings().dim, help="Hash dim (and optional validation)")
    px.add_argument("--embed-batch", type=int, default=EmbedSettings().batch_size)
    px.add_argument("--embed-max-chars", type=int, default=EmbedSettings().max_chars or 0, help="0 disables")
    px.add_argument("--embed-no-normalize", action="store_true")

    # query
    pq = sub.add_parser("query", help="Search the local index.jsonl")
    pq.add_argument("--index", type=Path, default=IndexingPipelineSettings().index_path)
    pq.add_argument("--state", type=Path, default=None)
    pq.add_argument("--index-backend", type=str, default=IndexSettings().backend, choices=["local", "faiss", "qdrant", "pgvector"])
    pq.add_argument("--backend-config", type=str, default="", help="JSON string with backend-specific config")
    pq.add_argument("--q", type=str, required=True)
    pq.add_argument("--k", type=int, default=IndexSettings().top_k)

    # run (ingest -> chunk -> index)
    pr = sub.add_parser("run", help="Run ingest -> chunk -> index")
    pr.add_argument("--kb-dir", type=Path, default=IndexingPipelineSettings().kb_dir)
    pr.add_argument("--glob", type=str, default=IngestSettings().glob)
    pr.add_argument("--out-dir", type=Path, default=IndexingPipelineSettings().out_dir)
    pr.add_argument("--chunk-size", type=int, default=ChunkSettings().chunk_size)
    pr.add_argument("--chunk-overlap", type=int, default=ChunkSettings().chunk_overlap)

    # embed/index options for run
    pr.add_argument("--embed-provider", type=str, default=EmbedSettings().provider, choices=["hash", "openai", "hf"])
    pr.add_argument("--embed-model", type=str, default=EmbedSettings().model)
    pr.add_argument("--hf-model", type=str, default=EmbedSettings().hf_model)
    pr.add_argument("--embed-dim", type=int, default=EmbedSettings().dim)
    pr.add_argument("--embed-batch", type=int, default=EmbedSettings().batch_size)
    pr.add_argument("--embed-max-chars", type=int, default=EmbedSettings().max_chars or 0)
    pr.add_argument("--embed-no-normalize", action="store_true")
    pr.add_argument("--reindex", action="store_true")
    pr.add_argument("--meta-keep", type=str, default="")
    pr.add_argument("--top-k", type=int, default=IndexSettings().top_k)
    pr.add_argument("--index-backend", type=str, default=IndexSettings().backend, choices=["local", "faiss", "qdrant", "pgvector"])
    pr.add_argument("--backend-config", type=str, default="", help="JSON string with backend-specific config")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "ingest":
        settings = IndexingPipelineSettings(kb_dir=args.kb_dir, ingest=IngestSettings(glob=args.glob))
        settings.validate()
        docs = fetch_documents(settings=settings)
        _write_jsonl(docs, args.out)
        print(f"Wrote {args.out}")
        return 0

    if args.cmd == "chunk":
        docs = list(_read_jsonl_docs(args.in_path))
        chunks = chunk_documents(
            docs,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        _write_jsonl(chunks, args.out)
        print(f"Wrote {args.out}")
        return 0

    if args.cmd in {"index", "run"}:
        # build settings from flags
        if args.cmd == "run":
            settings = IndexingPipelineSettings(
                kb_dir=args.kb_dir,
                out_dir=args.out_dir,
                ingest=IngestSettings(glob=args.glob),
                chunk=ChunkSettings(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap),
                embed=EmbedSettings(
                    provider=args.embed_provider,
                    model=args.embed_model,
                    hf_model=args.hf_model,
                    dim=args.embed_dim,
                    batch_size=args.embed_batch,
                    max_chars=(None if args.embed_max_chars == 0 else int(args.embed_max_chars)),
                    normalize_text=not args.embed_no_normalize,
                ),
                index=IndexSettings(
                    backend=args.index_backend,
                    backend_config=(json.loads(args.backend_config) if args.backend_config else {}),
                    top_k=args.top_k,
                    metadata_whitelist=[k.strip() for k in args.meta_keep.split(",") if k.strip()] or None,
                ),
            )
            settings.validate()

            docs = fetch_documents(settings=settings)
            _write_jsonl(docs, settings.docs_path)

            chunks = chunk_documents(
                docs,
                chunk_size=settings.chunk.chunk_size,
                chunk_overlap=settings.chunk.chunk_overlap,
                separators=settings.chunk.separators,
            )
            _write_jsonl(chunks, settings.chunks_path)

            in_chunks = settings.chunks_path
            index_path = settings.index_path
            state_path = settings.index_state_path
            embed_settings = settings.embed
            index_settings = settings.index
            reindex = bool(args.reindex)
        else:
            index_path = args.index
            state_path = args.state
            in_chunks = args.in_path
            embed_settings = EmbedSettings(
                provider=args.embed_provider,
                model=args.embed_model,
                hf_model=args.hf_model,
                dim=args.embed_dim,
                batch_size=args.embed_batch,
                max_chars=(None if args.embed_max_chars == 0 else int(args.embed_max_chars)),
                normalize_text=not args.embed_no_normalize,
            )
            index_settings = IndexSettings(
                backend=args.index_backend,
                backend_config=(json.loads(args.backend_config) if args.backend_config else {}),
                top_k=args.top_k,
                metadata_whitelist=[k.strip() for k in args.meta_keep.split(",") if k.strip()] or None,
            )
            reindex = bool(args.reindex)

        idx = make_index_backend(index_path, settings=index_settings, state_path=state_path)
        idx.load()
        idx.set_embed_config(embed_settings)

        # Incremental cleanup: scan chunks file for doc versions & ids, delete stale
        current_docs = _scan_chunks_for_state(in_chunks)
        deleted = idx.apply_incremental_plan(current_docs=current_docs)

        # Embed + upsert (second pass; streaming)
        chunks_iter = _read_jsonl_docs(in_chunks)
        items = embed_chunks(chunks_iter, settings=embed_settings)
        written = idx.upsert(items, skip_existing=not reindex)
        idx.save()

        print(f"Index: {index_path}")
        if state_path:
            print(f"State: {state_path}")
        print(f"Deleted stale chunks: {deleted}")
        print(f"Upserted chunks: {written} (total={len(idx)})")

        if args.cmd == "run":
            print(f"Wrote {settings.docs_path}")
            print(f"Wrote {settings.chunks_path}")
        return 0

    if args.cmd == "query":
        idx = make_index_backend(args.index, settings=IndexSettings(backend=args.index_backend, backend_config=(json.loads(args.backend_config) if args.backend_config else {})), state_path=args.state)
        idx.load()

        # Use embed config from state if available, otherwise default to hash
        embed_cfg = (idx.state.embed or {})
        embed_settings = EmbedSettings(
            provider=str(embed_cfg.get("provider") or "hash"),
            model=str(embed_cfg.get("model") or EmbedSettings().model),
            hf_model=str(embed_cfg.get("hf_model") or EmbedSettings().hf_model),
            dim=int(embed_cfg.get("dim") or EmbedSettings().dim),
        )
        emb = make_embedder(embed_settings)
        qvec = emb.embed_texts([args.q])[0]

        results = idx.query(list(map(float, qvec)), top_k=args.k)
        for r in results:
            print(json.dumps({"chunk_id": r.chunk_id, "score": r.score, "text": r.text, "metadata": r.metadata}, ensure_ascii=False))
        return 0

    return 2
