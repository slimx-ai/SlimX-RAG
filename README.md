# SlimX-RAG

A modular, testable RAG pipeline implementation.

Current modules:
- **ingest**: load documents from a local knowledge base and attach stable metadata
- **chunk**: split documents into deterministic chunks suitable for embedding/retrieval

## Install (dev)

```bash
uv sync --extra -dev
```
or 

```bash
uv venv
uv pip install -e ".[dev]"
```

## Ingest

```bash
slimx-ingest --kb-dir ./knowledge-base
```

See: `src/slimx_rag/ingest/README.md`

## Chunk

```bash
slimx-chunk --kb-dir ./knowledge-base --out ./output/chunks.jsonl
```

See: `src/slimx_rag/chunk/README.md`

## Tests

```bash
uv run pytest -q
```
