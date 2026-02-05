# Index backend plugin spec

This folder contains SlimX-RAG's **index backends** (vector storage + similarity search) implemented as **plugins**.

**Design goal:** the pipeline (ingest → chunk → embed → index → query) must not care whether you use:

- Local JSONL (MVP)
- FAISS
- Qdrant
- PostgreSQL + pgvector

Switching backend should be a **settings change**, not a code change.

---

## Concepts and invariants

### Terminology

- **Document**: an input source file/record. Identified by a stable `doc_id`.
- **Content hash**: a fingerprint of the document content used for incremental updates (`content_hash`).
- **Chunk**: a piece of a document. Identified by a stable `chunk_id`.
- **Vector / embedding**: `List[float]` of fixed dimension `D`.
- **Payload**: `text` and `metadata` stored alongside the vector.

### Required invariants

- `doc_id` and `chunk_id` **MUST be deterministic**.
- All vectors inside one index **MUST share the same dimension** `D`.
- `chunk_id` **MUST be globally unique** within an index.
- `metadata` should be **JSON-serializable** (for portability across backends).

### Deterministic ordering

Backends SHOULD return query results ordered by:

1. `score` descending (higher = more similar)
2. `chunk_id` ascending (tie-break)

This makes tests stable and avoids flaky output when scores are identical.

---

## IndexBackend contract

The canonical interface is defined in `base.py` as `IndexBackend`. Every backend must implement the same surface area.

### Methods

- `load() -> None`
  - Load on-disk structures into memory (if applicable).
  - Remote backends may treat this as a light connectivity check or no-op.

- `save() -> None`
  - Persist updated index structures (if applicable).
  - Remote backends may treat this as a no-op.

- `set_embed_config(embed: EmbedSettings) -> None`
  - Store the embedding configuration in `state.embed` for traceability.

- `upsert(items: Iterable[EmbeddedChunk], *, skip_existing: bool = True) -> int`
  - Insert or update chunks.
  - If `skip_existing=True`, do **insert-only** for existing `chunk_id`.
  - Returns the number of chunks written (best-effort for remote backends).

- `delete(chunk_ids: Iterable[str]) -> int`
  - Delete chunks by `chunk_id`.
  - MUST be idempotent: deleting a missing `chunk_id` must not raise.
  - Returns the number of chunks actually removed (best-effort for remote backends).

- `query(query_vector: List[float], *, top_k: Optional[int] = None) -> List[SearchResult]`
  - Return up to `top_k` results (default is `settings.top_k`).
  - The returned `SearchResult.score` must be comparable across records **within the same backend**.

### Properties

- `dim: Optional[int]`
  - The embedding dimension. Backends must validate dimension consistency.

- `state: IndexState`
  - Tracks incremental indexing info:
    - `docs[doc_id] = {"content_hash": str, "chunk_ids": [str, ...]}`

---

## Incremental indexing semantics

The pipeline computes the current document inventory and passes it to:

`apply_incremental_plan(current_docs=...)`

Where:

`current_docs: Dict[str, Tuple[str, List[str]]]` maps:

- `doc_id -> (content_hash, [chunk_id, ...])`

Expected behavior:

- **Doc removed**: delete all previously associated chunk IDs.
- **Doc changed** (`content_hash` differs): delete all previously associated chunk IDs (stale embeddings).
- **Doc unchanged**: keep existing chunks.

To avoid duplicating this logic, `apply_incremental_plan()` is implemented once in `IndexBackend` and uses `delete()`.

### Crash-consistency guideline

Best-effort pattern:

1) apply plan (deletes)
2) upsert new chunks
3) save index structures
4) save state

Not all backends can make index + state fully transactional, but aim for this order.

---

## Settings / configuration

Backends are selected via `IndexSettings`:

- `backend`: `local` | `faiss` | `qdrant` | `pgvector`
- `backend_config`: JSON-serializable dict passed to the backend

Guidelines:

- Keep `backend_config` JSON-serializable.
- Prefer environment variables for secrets (API keys, passwords).
- If required config keys are missing, raise `ValueError` with a clear message.

### Recommended `backend_config` keys

#### local

- No required keys.
- Optional:
  - `index_filename` (if you want to enforce a filename convention)

#### faiss

- `dim` (int): required when creating a new FAISS index
- Optional:
  - `path` (str): path to `.faiss` file (defaults to `<index_path>.faiss` in the backend)
  - `metric` (str): `cosine` (default) or `l2`

#### qdrant

- `url` (str): e.g. `http://localhost:6333`
- `collection` (str)
- Optional:
  - `api_key` (str) — or prefer env var `QDRANT_API_KEY`

#### pgvector

- `dsn` (str): PostgreSQL DSN, e.g. `postgresql://user:pass@localhost:5432/db`
- `table` (str)
- Optional:
  - `schema` (str)

---

## Payload rules

Backends should treat payloads consistently:

- Apply `settings.metadata_whitelist` when non-empty.
- Store `text` and `metadata` for each `chunk_id`.
- Keep payload JSON-compatible (especially for remote backends).

---

## Error handling and validation

- **Dimension mismatch**:
  - If `query_vector` length != `dim`, raise `ValueError` (or `RuntimeError`) with a clear message.
  - If an upsert contains a vector length != `dim`, raise `ValueError`.

- **Missing configuration**:
  - Raise `ValueError` listing missing keys.

- **Connectivity failures** (remote backends):
  - Raise `ConnectionError` / `RuntimeError` including the endpoint/DSN and a short hint.

---

## Optional capabilities (allowed but not required)

Backends may optionally support:

- approximate search
- server-side filtering
- hybrid search
- batched deletes/upserts

These must not change the core pipeline contract.

---

## Adding a new backend

1) Create `src/slimx_rag/index/<name>_backend.py`.
2) Implement the `IndexBackend` contract.
3) Register it in `make_index_backend()` in `src/slimx_rag/index/__init__.py`.
4) Add optional dependency under `pyproject.toml` extras (if needed).
5) Add a unit test:
   - upsert a few known vectors
   - query a known vector
   - assert ordering and deterministic tie-break

### Minimal backend skeleton

```python
from __future__ import annotations

from typing import Iterable, List, Optional

from slimx_rag.embed import EmbeddedChunk
from slimx_rag.settings import EmbedSettings, IndexSettings

from .base import IndexBackend
from .types import IndexState, SearchResult


class MyBackend(IndexBackend):
    def __init__(self, index_path, *, settings: Optional[IndexSettings] = None, state_path=None):
        self.index_path = index_path
        self.settings = settings or IndexSettings()
        self.state_path = state_path
        self.state = IndexState.load(self.state_path) if self.state_path else IndexState()
        self._dim: Optional[int] = None

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    def load(self) -> None:
        ...

    def save(self) -> None:
        ...

    def set_embed_config(self, embed: EmbedSettings) -> None:
        self.state.embed = {
            "provider": embed.provider,
            "model": embed.model,
            "hf_model": embed.hf_model,
            "dim": embed.dim,
        }

    def upsert(self, items: Iterable[EmbeddedChunk], *, skip_existing: bool = True) -> int:
        ...

    def delete(self, chunk_ids: Iterable[str]) -> int:
        ...

    def query(self, query_vector: List[float], *, top_k: Optional[int] = None) -> List[SearchResult]:
        ...
```

---

## Backend-specific notes

- **FAISS deletes:** not all FAISS index types support deletion; for an MVP, it’s acceptable to rebuild the FAISS index after deletes.
- **Remote backends:** `save()` can be a no-op, but should exist for interface parity.
- **Payload storage:** if the vector store does not store payloads (e.g. raw FAISS), keep a sidecar mapping `chunk_id -> (text, metadata)`.
