# SlimX-RAG

A slim, deterministic RAG indexing pipeline:

**ingest → chunk → embed → index → query**

This repo intentionally focuses on *mechanics* (determinism, incremental indexing, clean contracts)
before swapping in heavier backends.

## Install (uv)

```bash
uv sync
```

Optional providers:

```bash
uv sync --extra openai   # OpenAI embeddings (requires OPENAI_API_KEY)
uv sync --extra hf       # HuggingFace SentenceTransformers embeddings
```

## Quickstart

### 1) Ingest + chunk + index

```bash
slimx-rag run --kb-dir ./knowledge-base --out-dir ./output
```

This produces:

- `output/docs.jsonl`
- `output/chunks.jsonl`
- `output/index.jsonl`
- `output/index_state.json` (incremental state + embed config)

### 2) Query

```bash
slimx-rag query --index ./output/index.jsonl --q "What is this project?" --k 5
```

By default, the pipeline uses a deterministic **hash embedder** so it runs offline.
For real semantics, choose an embedding provider:

#### OpenAI embeddings
```bash
export OPENAI_API_KEY="..."
slimx-rag run --kb-dir ./knowledge-base --out-dir ./output \
  --embed-provider openai --embed-model text-embedding-3-small
```

#### HuggingFace embeddings (SentenceTransformers)
```bash
slimx-rag run --kb-dir ./knowledge-base --out-dir ./output \
  --embed-provider hf --hf-model sentence-transformers/all-MiniLM-L6-v2
```

## Notes

- Chunk IDs are deterministic and intended for caching/dedup.
- `index_state.json` tracks `doc_id → content_hash → chunk_ids` so the index can delete stale chunks on updates.
- `index.jsonl` is a simple local MVP backend; it loads in-memory for query. For larger corpora, swap to FAISS/Qdrant/pgvector.

