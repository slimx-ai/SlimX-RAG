# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --group dev          # install with dev deps (pytest, build)
uv run pytest -q             # run all tests
uv run pytest tests/test_chunker.py -k test_name   # run a single test
uv run python -m build       # build the package (CI does this too)
```

Optional extras enable specific providers/backends: `uv sync --extra openai|hf|doc|demo|faiss|qdrant|pgvector`.

Tests do not require installation — `tests/conftest.py` adds `src/` to `sys.path`. Backend tests (qdrant, pgvector, faiss) use fake client classes injected via `sys.modules`/monkeypatch, so no servers or optional deps are needed to run the full suite. CI runs pytest + build on Python 3.11–3.13.

Run the pipeline end-to-end:

```bash
uv run slimx-rag run --kb-dir examples/tiny_demo/knowledge-base --out-dir ./output
uv run slimx-rag ask --out-dir ./output --q "What is this knowledge base about?"
```

## Architecture

A deterministic RAG pipeline: **ingest → chunk → embed → index → retrieve → answer → cite → evaluate → serve**. Each stage is a package under `src/slimx_rag/` and stages communicate through JSONL artifacts in an output directory: `docs.jsonl` → `chunks.jsonl` → `embeddings.jsonl` → `index.jsonl` + `index_state.json` (and optionally `manifest.json`). The CLI (`cli.py`, entry point `slimx-rag`) exposes each stage as a subcommand (`ingest`, `chunk`, `index`, `query`, `retrieve`, `ask`, `eval`, `serve`, `manifest`, `diff`, `report`, `run`) using shared argparse parent parsers.

### Determinism is the core invariant

All identities flow through `core/hashing.py`. `HashPolicy` is a *versioned protocol*, not a runtime setting — changing algorithm/digest sizes breaks document IDs, chunk IDs, and `index_state.json` compatibility across builds. The identity chain:

- `doc_id` = hash of `kb_relpath` (stable across machines)
- `content_hash` = hash of document text (changes when content changes)
- `chunk_id` = hash of (parent identity, content_hash, chunk config fingerprint, chunk_index)

Chunking sorts documents by a stable key before splitting; query results are sorted deterministically (score desc, then chunk_id asc). The default embedding provider `hash` is offline and deterministic so tests/CI never hit the network (it is **not semantic** — use `openai` or `hf` for real demos).

### Settings

`settings.py` defines frozen dataclasses (`IngestSettings`, `ChunkSettings`, `EmbedSettings`, `IndexSettings`, composed into `IndexingPipelineSettings`), each with a `validate()` method. CLI defaults are derived from a `DEFAULTS = IndexingPipelineSettings()` instance — change defaults in settings, not in the CLI. Valid providers/backends are the tuples `EMBED_PROVIDERS` and `INDEX_BACKENDS` at the top of this file.

### Index backend plugin system

`index/__init__.py:make_index_backend()` is the factory; backends (`local` JSONL, `faiss`, `qdrant`, `pgvector`) subclass `index/base.py:IndexBackend` and implement `load/save/upsert/delete/query`. Shared behavior lives in the base class: metadata whitelisting, deterministic result sorting, and the incremental-indexing plan (`apply_incremental_plan` compares `doc_id`/`content_hash` against `index_state.json` to delete stale chunks before upsert). Optional backend deps are imported lazily inside the factory so the core package works without extras. New backends should be added to the factory, `INDEX_BACKENDS`, and a pyproject extra.

### Answer generation

`answer/generator.py` takes model strings of the form `provider:model` (e.g. `openai:gpt-4.1-mini`, `ollama:llama3.2:3b`). `fake:grounded` is a deterministic, no-network model for tests and smoke checks. Real models are executed via the external `slimx` package (separate repo, installed with `uv pip install -e ../slimx`) — SlimX-RAG owns retrieval/citations/evaluation, SlimX owns model execution and traces. Per-provider defaults (timeout, max_tokens, context budget) are derived from the model prefix.

### RAGOps modules

`manifest/`, `diff/`, and `report/` are read-only inspection tools over output directories (reproducibility manifests, build comparisons, quality reports). They must not change the ingest/chunk/embed/index/query artifacts.
