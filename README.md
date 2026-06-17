# SlimX-RAG

A slim, deterministic RAG pipeline for customer-demo-ready private knowledge assistants.

What’s implemented today:

**ingest → chunk → embed → index → retrieve → answer → cite → evaluate → serve**

---

## Install

This project is packaged as `slimx-rag` and ships a CLI command **`slimx-rag`**.

### With `uv` (recommended)

```bash
uv sync
```
with extra dependencies 

```bash
uv sync --group dev
```

Optional embedding providers:

```bash
uv sync --extra openai   # OpenAI embeddings via langchain-openai
uv sync --extra hf       # SentenceTransformers embeddings
uv sync --extra demo     # FastAPI customer demo server
uv sync --extra doc      # PDF/DOCX loaders
```

### With `pip`

```bash
python -m pip install -e ".[dev]"
source venv/bin/activate  # or your venv path
```

---

## Run the tests

After installing with dev extras, activate the venv:

```bash
source .venv/bin/activate
```

Then run:

```bash
# requires uv sync --group dev
pytest
# or
pytest -q
```

(Or use `uv run pytest` without activating)

---

## Quickstart: run the full pipeline

1) Use one of the example knowledge bases, or put Markdown documents under your own folder:

```text
examples/tiny_demo/knowledge-base/
  overview.md
```

2) Activate the venv:

```bash
source .venv/bin/activate
```

Then run:

```bash
slimx-rag run --kb-dir examples/tiny_demo/knowledge-base --out-dir ./output
slimx-rag ask --out-dir ./output --q "What is this knowledge base about?"
```

(Or use `uv run slimx-rag run ...` without activating)

Outputs:

- `output/docs.jsonl` (raw documents + metadata)
- `output/chunks.jsonl` (deterministic chunks)
- `output/embeddings.jsonl` (vectors; default provider is offline `hash`)
- `output/index.jsonl` (vector index records; for small corpora with the local backend)
- `output/index_state.json` (embedding config and incremental index state)
- `output/manifest.json` (optional RAGOps build manifest when requested)

---


## Run as a Docker service

The Docker image is **turnkey build-then-serve**: point it at a knowledge-base directory
and it builds the index on first start (with the local `hf` sentence-transformers
embedder, baked into the image — no network needed at runtime), then serves
`/health`, `/api/retrieve`, and `/api/ask` on port 8080.

```bash
# Build (or pull the published image: ghcr.io/slimx-ai/slimx-rag)
docker build -t slimxai/slimx-rag:hf .

# Index a KB + serve in one container
docker run --rm -p 8080:8080 \
  -e RAG_KB_DIR=/kb \
  -v "$PWD/examples/tiny_demo/knowledge-base:/kb:ro" \
  -v slimx_rag_index:/app/output \
  slimxai/slimx-rag:hf

curl localhost:8080/health   # {"status":"ok","embed_provider":"hf",...}
```

Or use the demo compose (builds the image, indexes the bundled corpus, serves):

```bash
docker compose -f docker-compose.demo.yml up --build
```

Set `RAG_REINDEX=1` to rebuild the index on start. Override the embedder via
`RAG_EMBED_PROVIDER` (`hash` | `openai` | `hf`) and related `RAG_*` env. The image is
published to `ghcr.io/slimx-ai/slimx-rag` by `.github/workflows/publish-image.yaml` on
release; downstream apps consume that image (they do not build it).

---

## RAGOps artifacts

SlimX-RAG can generate reproducibility and inspection artifacts for deterministic, auditable RAG indexing:

- `manifest.json` records what was built, including artifact counts, embedding settings, backend information, chunk config, and the centralized hash policy.
- `slimx-rag diff` compares two builds by documents, chunks, and available config metadata.
- `slimx-rag report` summarizes index quality, metadata coverage, duplicates, near-empty chunks, backend details, and warnings.

Generate and inspect RAGOps artifacts:

```bash
slimx-rag run --kb-dir examples/tiny_demo/knowledge-base --out-dir ./output --write-manifest
slimx-rag manifest --out-dir ./output
slimx-rag diff ./output-v1 ./output-v2
slimx-rag diff ./output-v1 ./output-v2 --format json
slimx-rag report --out-dir ./output --format markdown
slimx-rag report --out-dir ./output --format json
```

Expected RAGOps output files:

- `output/manifest.json` — pretty JSON build manifest for reproducibility and audits.
- `slimx-rag diff ...` — text or JSON comparison output; it does not write files by default.
- `slimx-rag report ...` — Markdown or JSON report printed to stdout.

These modules are intended for reproducible, auditable RAGOps for private/local AI systems without changing the ingest, chunk, embed, index, or query artifacts.

---

## Run modules one by one

### 1) Ingest

```bash
slimx-rag ingest --kb-dir examples/tiny_demo/knowledge-base --out ./output/docs.jsonl
```

Defaults:
- file pattern: `**/*.md`

Override the glob:

```bash
slimx-rag ingest --kb-dir examples/tiny_demo/knowledge-base --glob "**/*.txt" --out ./output/docs.jsonl
```

### 2) Chunk

```bash
slimx-rag chunk --in ./output/docs.jsonl --out ./output/chunks.jsonl \
  --chunk-size 800 --chunk-overlap 120
```

### 3) Embed

Offline deterministic embeddings (good for CI/dev; **not semantic**):

```bash
slimx-rag index --in ./output/chunks.jsonl --embed-provider hash --embed-dim 384
```

OpenAI embeddings:

```bash
export OPENAI_API_KEY="..."
slimx-rag index --in ./output/chunks.jsonl --embed-provider openai --embed-model text-embedding-3-small
```

HuggingFace SentenceTransformers:

```bash
slimx-rag index --in ./output/chunks.jsonl --embed-provider hf --hf-model sentence-transformers/all-MiniLM-L6-v2
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
slimx-rag run --kb-dir examples/insurance_demo/knowledge-base --out-dir ./output --index-backend local
```

FAISS (dimension can be inferred from the first vector; `dim` is optional but useful for empty-index creation):

```bash
slimx-rag run --kb-dir examples/insurance_demo/knowledge-base --out-dir ./output \
  --index-backend faiss \
  --backend-config '{"dim": 384}'
```

Qdrant:

```bash
slimx-rag run --kb-dir examples/insurance_demo/knowledge-base --out-dir ./output \
  --index-backend qdrant \
  --backend-config '{"url":"http://localhost:6333","collection":"slimx"}'
```

pgvector:

```bash
slimx-rag run --kb-dir examples/insurance_demo/knowledge-base --out-dir ./output \
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

settings = IndexingPipelineSettings(kb_dir=Path("examples/tiny_demo/knowledge-base"))
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
from slimx_rag.index import make_index_backend
from slimx_rag.embed import make_embedder
from slimx_rag.settings import EmbedSettings, IndexSettings

index_path = Path("./output/index.jsonl")
index_settings = IndexSettings(backend="local")
idx = make_index_backend(index_path, settings=index_settings)
idx.load()
eset = EmbedSettings(provider="hash", dim=idx.dim or 384)
qvec = make_embedder(eset).embed_texts(["what is this knowledge base about"])[0]
results = idx.query(list(map(float, qvec)), top_k=5)
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
{"chunk_id": "...", "vector": [0.1, -0.2, ...], "text": "...", "metadata": {"kb_relpath": "..."}}
```

---

## Design notes (why it’s deterministic)

- **Ingest** assigns a stable `doc_id` primarily derived from `kb_relpath` (stable across machines).
- **Ingest** also assigns `content_hash` derived from document text (changes when content changes).
- **Chunking** sorts documents by a stable key (`kb_relpath/source/doc_id`) before splitting.
- **Chunk IDs** are stable hashes of `(parent identity, content_hash, chunk config, chunk_index)`.
- **Embedding** can be run offline (`hash`) so tests and CI don’t depend on network calls.
- **Indexing** stores vectors deterministically; the local backend computes vector norms when loading/querying for fast cosine search.

---

## Project layout

```text
src/slimx_rag/
  ingest/   # load docs + baseline metadata
  chunk/    # deterministic splitting + chunk_id
  embed/    # embedding providers + batching + retry
  index/    # backend plugins + vector query
  retrieval/ # retrieval result mapping + citations
  answer/   # grounded answer generation
  eval/     # demo evaluation runner
  server/   # FastAPI demo app
  manifest/ # RAGOps build manifests
  diff/     # RAGOps build comparisons
  report/   # RAGOps quality reports
  settings.py
  cli.py
tests/
```
Skip the embeddings artifact for remote-only deployments:

```bash
slimx-rag run --kb-dir examples/insurance_demo/knowledge-base --out-dir ./output --no-embeddings-out
```

### Retrieve and answer

```bash
slimx-rag retrieve --out-dir ./output --q "What does the corpus say about edge AI?"
slimx-rag ask --out-dir ./output --q "What does the corpus say about edge AI?"
```

Use a real SlimX model for polished demos:

```bash
export OPENAI_API_KEY="..."
slimx-rag ask --out-dir ./output --model openai:gpt-4.1-mini --q "Compare SlimX Core and SlimX-RAG."
```

### Run the research demo

```bash
uv sync --extra demo
uv pip install -e ../slimx
ollama pull llama3.2:3b
uv run slimx-rag run --kb-dir examples/research_demo/knowledge-base --out-dir output
OLLAMA_BASE_URL=http://localhost:11434 uv run slimx-rag ask --out-dir output --q "What is SlimX.ai's strongest business wedge for research organizations?" --model ollama:llama3.2:3b --timeout 300 --max-tokens 220 --max-context-chars 2500 --k 2
uv run slimx-rag eval --out-dir output --dataset examples/research_demo/eval/questions.jsonl --out output/eval_report.md
SLIMX_LLM_MODEL=ollama:llama3.2:3b SLIMX_LLM_TIMEOUT=300 SLIMX_LLM_MAX_TOKENS=220 SLIMX_MAX_CONTEXT_CHARS=2500 OLLAMA_BASE_URL=http://localhost:11434 uv run slimx-rag serve --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080`.

Use `--model fake:grounded` when you need a deterministic no-network smoke test.
If Ollama times out on the first request, warm it with `ollama run llama3.2:3b "Say ready"` and retry with a larger `--timeout`, smaller `--max-tokens`, smaller `--max-context-chars`, or lower `--k`.
