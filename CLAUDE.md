# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --group dev          # install with dev deps (pytest, ruff, mypy, build)
uv run pytest -q             # run all tests
uv run pytest tests/test_chunker.py -k test_name   # run a single test
uv run ruff check .          # lint (CI enforces on all Python versions)
uv run mypy                  # type check src/ (CI enforces on 3.12)
uv run python -m build       # build the package (CI does this too)
```

CLI error contract: user-input errors (bad paths, malformed JSONL, bad `--backend-config`) exit 2 with a one-line message; unexpected errors exit 1; pass `--verbose` (before the subcommand) for debug logs and tracebacks. All artifact writes are atomic (temp file + `os.replace` via `utils/commons.py` helpers). Incremental index state is committed strictly after a successful upsert + save (`IndexBackend.commit_state`) — state may lag the backend but never lead it.

Optional extras enable specific providers/backends: `uv sync --extra openai|hf|doc|demo|faiss|qdrant|pgvector`.

Tests do not require installation — `tests/conftest.py` adds `src/` to `sys.path`. Backend tests (qdrant, pgvector, faiss) use fake client classes injected via `sys.modules`/monkeypatch, so no servers or optional deps are needed to run the full suite. CI runs pytest + build on Python 3.11–3.13.

Run the pipeline end-to-end:

```bash
uv run slimx-rag run --kb-dir examples/tiny_demo/knowledge-base --out-dir ./output
uv run slimx-rag ask --out-dir ./output --q "What is this knowledge base about?"
```

Run as an HTTP service / in Docker:

```bash
# HTTP demo server (needs the `demo` extra; configured via RAG_*/SLIMX_* env, not flags)
uv run slimx-rag serve --host 127.0.0.1 --port 8080

# Turnkey Docker: build the index from a KB dir on first start, then serve
docker run --rm -p 8080:8080 -e RAG_KB_DIR=/kb \
  -v "$PWD/examples/tiny_demo/knowledge-base:/kb:ro" -v slimx_rag_index:/app/output \
  ghcr.io/slimx-ai/slimx-rag:latest
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

### HTTP service mode (`server/`)

`server/app.py` is a FastAPI app (the `serve` subcommand; needs the `demo` extra). Unlike the CLI, it is configured **entirely through environment variables**, rebuilt per request: `RAG_*` for index/embed/chunk settings (`RAG_INDEX_BACKEND`, `RAG_EMBED_PROVIDER`, `RAG_INDEX_PATH`, `RAG_STATE_PATH`, `RAG_TOP_K`, …), `SLIMX_*` for the LLM (`SLIMX_LLM_MODEL`, `SLIMX_LLM_TIMEOUT`, `SLIMX_LLM_MAX_TOKENS`, `SLIMX_MAX_CONTEXT_CHARS`), and optional `DEMO_AUTH_TOKEN` for Bearer auth. `serve` itself takes only `--host`/`--port`. See `.env.example` for the full surface.

Endpoints: `GET /health`, `GET /api/config`, `POST /api/retrieve`, `POST /api/ask`, `POST /api/eval/run`, `POST /api/index`, and `GET /` (HTML UI from `server/static/index.html`).

The server keeps **one hot in-memory backend** (`_current_backend()`), loaded once and reused across requests — `retrieve()` otherwise re-reads and re-parses the whole index file every call. The cache is refreshed only when the index file's `(mtime, size)` changes (e.g. an external CLI rebuild); pass `backend=` to `retrieve()` to inject it. A single reentrant `_index_lock` guards both the cached backend and the read-modify-write ingest path, so reads never see a half-applied write. `_reset_index_cache()` exists for tests.

Service mode is retrieval-only except `POST /api/index`, which ingests one posted document (chunk → embed → upsert) so a downstream app can index over HTTP. It reuses the hot backend under `_index_lock` (load amortized across posts) and `save()`s atomically (`os.replace`). Identity is `path_id("{workspace_id}/{document_id}")`, so re-posting is idempotent — it calls `delete_doc` + `commit_doc_state` (the single-document analogues of `apply_incremental_plan`/`commit_state`) to replace just that document's chunks.

Retrieval scoping: `/api/retrieve` and `/api/ask` accept optional `workspace_id`/`document_ids` that filter chunks by metadata tagged at index time. This works by over-fetching the whole index and post-filtering, so it is **local-backend only** — backends without `supports_inmemory_scope_filter` (qdrant/pgvector/faiss) raise `ScopeNotSupportedError` (HTTP 400) instead of silently returning under-filled results. Backend-native filter pushdown is the tracked follow-up.

Docker: a turnkey build-then-serve image (`Dockerfile`, `docker-entrypoint.sh`). The entrypoint runs `slimx-rag run` to build the index from `RAG_KB_DIR` on first start (or when `RAG_REINDEX` is set), then `slimx-rag serve`. The default embedder is `hf` with the model baked into the image (no runtime network). Published to `ghcr.io/slimx-ai/slimx-rag` on release by `.github/workflows/publish-image.yaml`; `deploy/vps/` holds a Caddy + Compose VPS deployment.

### RAGOps modules

`manifest/`, `diff/`, and `report/` are read-only inspection tools over output directories (reproducibility manifests, build comparisons, quality reports). They must not change the ingest/chunk/embed/index/query artifacts.
