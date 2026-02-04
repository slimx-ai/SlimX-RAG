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

## Index backends (plugins)

For the developer-facing plugin contract and conventions, see: `src/slimx_rag/index/README.md`.

The indexing layer is implemented as a **backend plugin** so you can switch storage/search engines without changing pipeline code.

Supported backends:

- `local` — JSONL MVP backend (default): portable, deterministic, loads in-memory for query
- `faiss` — local FAISS index (binary) + JSON sidecar for payloads (optional extra)
- `qdrant` — remote Qdrant collection (optional extra)
- `pgvector` — Postgres + pgvector table (optional extra)

Developer spec for implementing new backends: see `src/slimx_rag/index/README.md`.

### Install optional backends

```bash
uv sync --extra faiss
uv sync --extra qdrant
uv sync --extra pgvector
```

### CLI usage

Local (default):

```bash
slimx-rag run --kb-dir ./knowledge-base --out-dir ./output --index-backend local
```

FAISS (recommend using an `.faiss` filename):

```bash
slimx-rag run --kb-dir ./knowledge-base --out-dir ./output \
  --index-backend faiss \
  --backend-config '{"dim": 384}'
```

Qdrant:

```bash
slimx-rag run --kb-dir ./knowledge-base --out-dir ./output \
  --index-backend qdrant \
  --backend-config '{"url":"http://localhost:6333","collection":"slimx"}'
```

pgvector:

```bash
slimx-rag run --kb-dir ./knowledge-base --out-dir ./output \
  --index-backend pgvector \
  --backend-config '{"dsn":"postgresql://user:pass@localhost:5432/db","table":"slimx_vectors"}'
```

## Notes

- Chunk IDs are deterministic and intended for caching/dedup.
- `index_state.json` tracks `doc_id → content_hash → chunk_ids` so the index can delete stale chunks on updates.
- Backends store vectors + payloads differently, but all implement the same interface (`IndexBackend`).

