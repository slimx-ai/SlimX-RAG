from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document

from slimx_rag.ingest.loader import fetch_documents
from slimx_rag.chunk import chunk_documents
from slimx_rag.settings import IndexingSettings, IngestSettings, ChunkSettings


def _write_jsonl(docs: Iterable[Document], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"page_content": d.page_content, "metadata": d.metadata}, ensure_ascii=False) + "\n")


def _read_jsonl(in_path: Path) -> List[Document]:
    docs: List[Document] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            docs.append(Document(page_content=rec.get("page_content", ""), metadata=rec.get("metadata", {}) or {}))
    return docs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="slimx", description="SlimX-RAG indexing pipeline CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ingest
    pi = sub.add_parser("ingest", help="Load KB documents and write docs.jsonl")
    pi.add_argument("--kb-dir", type=Path, default=IndexingSettings().kb_dir)
    pi.add_argument("--glob", type=str, default=IngestSettings().glob)
    pi.add_argument("--out", type=Path, default=IndexingSettings().docs_path)

    # chunk
    pc = sub.add_parser("chunk", help="Chunk docs.jsonl into chunks.jsonl")
    pc.add_argument("--in", dest="in_path", type=Path, required=True)
    pc.add_argument("--out", type=Path, default=IndexingSettings().chunks_path)
    pc.add_argument("--chunk-size", type=int, default=ChunkSettings().chunk_size)
    pc.add_argument("--chunk-overlap", type=int, default=ChunkSettings().chunk_overlap)

    # run
    pr = sub.add_parser("run", help="Run ingest -> chunk")
    pr.add_argument("--kb-dir", type=Path, default=IndexingSettings().kb_dir)
    pr.add_argument("--glob", type=str, default=IngestSettings().glob)
    pr.add_argument("--out-dir", type=Path, default=IndexingSettings().out_dir)
    pr.add_argument("--chunk-size", type=int, default=ChunkSettings().chunk_size)
    pr.add_argument("--chunk-overlap", type=int, default=ChunkSettings().chunk_overlap)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "ingest":
        settings = IndexingSettings(kb_dir=args.kb_dir, ingest=IngestSettings(glob=args.glob))
        docs = fetch_documents(settings=settings)
        _write_jsonl(docs, args.out)
        print(f"Wrote {args.out}")
        return 0

    if args.cmd == "chunk":
        docs = _read_jsonl(args.in_path)
        chunks = chunk_documents(
            docs,
            settings=ChunkSettings(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap),
        )
        _write_jsonl(chunks, args.out)
        print(f"Wrote {args.out}")
        return 0

    if args.cmd == "run":
        settings = IndexingSettings(
            kb_dir=args.kb_dir,
            out_dir=args.out_dir,
            ingest=IngestSettings(glob=args.glob),
            chunk=ChunkSettings(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap),
        )
        settings.validate()

        docs = fetch_documents(settings=settings)
        _write_jsonl(docs, settings.docs_path)

        chunks = chunk_documents(docs, settings=settings.chunk)
        _write_jsonl(chunks, settings.chunks_path)

        print(f"Wrote {settings.docs_path}")
        print(f"Wrote {settings.chunks_path}")
        return 0

    return 2
