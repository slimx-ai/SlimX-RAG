# SlimX-RAG

A slim, deterministic RAG *preprocessing* pipeline.

What’s implemented today:

**ingest → chunk → embed → index (+ query)**

---

## Install

This project is packaged as `slimx-rag` and ships a CLI command **`slimx-rag`**.

### With `uv` (recommended)

```bash
uv sync
```
with extra dependencies 

```bash
uv sync --extra dev
```

Optional embedding providers:

```bash
uv sync --extra openai   # OpenAI embeddings via langchain-openai
uv sync --extra hf       # SentenceTransformers embeddings
```

### With `pip`

```bash
python -m pip install -e ".[dev]"
```

---

## Run the tests

```bash
uv run pytest
# or
uv run pytest -q
```

---

## Quickstart: run the full pipeline

1) Put Markdown documents under a knowledge base folder:

```text
knowledge-base/
  company/
    about.md
  products/
    overview.md
```

2) Run:

```bash
uv run slimx-rag run --kb-dir ./knowledge-base --out-dir ./output
```

Outputs:

- `output/docs.jsonl` (raw documents + metadata)
- `output/chunks.jsonl` (deterministic chunks)
- `output/embeddings.jsonl` (vectors; default provider is offline `hash`)
- `output/index.jsonl` (simple cosine index; for small corpora)

---

## Run modules one by one

### 1) Ingest

```bash
uv run slimx-rag ingest --kb-dir ./knowledge-base --out ./output/docs.jsonl
```

Defaults:
- file pattern: `**/*.md`

Override the glob:

```bash
uv run slimx-rag ingest --kb-dir ./knowledge-base --glob "**/*.txt" --out ./output/docs.jsonl
```

### 2) Chunk

```bash
uv run slimx-rag chunk --in ./output/docs.jsonl --out ./output/chunks.jsonl \
  --chunk-size 800 --chunk-overlap 120
```

### 3) Embed

Offline deterministic embeddings (good for CI/dev; **not semantic**):

```bash
uv run slimx embed --in ./output/chunks.jsonl --out ./output/embeddings.jsonl \
  --embed-provider hash --embed-dim 384
```

OpenAI embeddings:

```bash
export OPENAI_API_KEY="..."
uv run slimx embed --in ./output/chunks.jsonl --out ./output/embeddings.jsonl \
  --embed-provider openai --embed-model text-embedding-3-small
```

HuggingFace SentenceTransformers:

```bash
uv run slimx embed --in ./output/chunks.jsonl --out ./output/embeddings.jsonl \
  --embed-provider hf --hf-model sentence-transformers/all-MiniLM-L6-v2
```

### 4) Index

The indexing layer is implemented as a **backend plugin** so you can switch storage/search engines without changing pipeline code.

Supported backends:

- `local` — JSONL MVP backend (default): portable, deterministic, loads in-memory for query
- `faiss` — local FAISS index (binary) + JSON sidecar for payloads (optional extra)
- `qdrant` — remote Qdrant collection (optional extra)
- `pgvector` — Postgres + pgvector table (optional extra)

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

---

## Use as a Python library

### Ingest

```python
from pathlib import Path
from slimx_rag.ingest.loader import fetch_documents
from slimx_rag.settings import IndexingPipelineSettings

settings = IndexingPipelineSettings(kb_dir=Path("./knowledge-base"))
docs = fetch_documents(settings=settings)
print(len(docs), docs[0].metadata)
```

### Chunk

```python
from slimx_rag.chunk import chunk_documents
from slimx_rag.settings import ChunkSettings

cset = ChunkSettings(chunk_size=800, chunk_overlap=120)
chunks = chunk_documents(
    docs,
    chunk_size=cset.chunk_size,
    chunk_overlap=cset.chunk_overlap,
    separators=cset.separators,
    extended_chunk_metadata=cset.extended_metadata,
)
print(len(chunks), chunks[0].metadata["chunk_id"])
```

### Embed

```python
from slimx_rag.embed import embed_chunks
from slimx_rag.settings import EmbedSettings

eset = EmbedSettings(provider="hash", dim=384)
embs = list(embed_chunks(chunks, settings=eset))
print(embs[0].chunk_id, len(embs[0].vector))
```

### Index + Query

```python
from pathlib import Path
from slimx_rag.index import NaiveIndex
from slimx_rag.embed import make_embedder
from slimx_rag.settings import EmbedSettings

idx = NaiveIndex.load(Path("./output/index.jsonl"))
eset = EmbedSettings(provider="hash", dim=idx.dim or 384)
qvec = make_embedder(eset).embed_texts(["what is this knowledge base about"])[0]
results = idx.search(list(map(float, qvec)), top_k=5)
print(results[0].chunk_id, results[0].score)
```

---

## Output formats

### `docs.jsonl` and `chunks.jsonl`

Each line is:

```json
{"page_content": "...", "metadata": {"doc_id": "...", "kb_relpath": "...", "content_hash": "..."}}
```

### `embeddings.jsonl`

Each line is:

```json
{"chunk_id": "...", "vector": [0.1, -0.2, ...], "text": "...", "metadata": {"kb_relpath": "..."}}
```

### `index.jsonl`

Each line is:

```json
{"chunk_id": "...", "vector": [0.1, -0.2, ...], "norm": 12.34, "text": "...", "metadata": {"kb_relpath": "..."}}
```

---

## Design notes (why it’s deterministic)

- **Ingest** assigns a stable `doc_id` primarily derived from `kb_relpath` (stable across machines).
- **Ingest** also assigns `content_hash` derived from document text (changes when content changes).
- **Chunking** sorts documents by a stable key (`kb_relpath/source/doc_id`) before splitting.
- **Chunk IDs** are stable hashes of `(parent identity, content_hash, chunk config, chunk_index)`.
- **Embedding** can be run offline (`hash`) so tests and CI don’t depend on network calls.
- **Indexing** precomputes vector norms to make cosine search fast and deterministic.

---

## Project layout

```text
src/slimx_rag/
  ingest/   # load docs + baseline metadata
  chunk/    # deterministic splitting + chunk_id
  embed/    # embedding providers + batching + retry
  index/    # naive cosine index + query
  settings.py
  cli.py
tests/
```

