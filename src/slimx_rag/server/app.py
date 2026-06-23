from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from slimx_rag.answer import answer
from slimx_rag.chunk import chunk_documents, chunk_parsed_document
from slimx_rag.core.hashing import content_hash, path_id
from slimx_rag.document import DocumentError, DocumentSource, parse_document
from slimx_rag.embed import EmbeddedChunk, embed_chunks, get_cached_embedder, make_token_counter
from slimx_rag.eval import load_eval_cases, run_eval
from slimx_rag.index import IndexBackend, make_index_backend
from slimx_rag.index.types import SearchResult
from slimx_rag.retrieval import Bm25Index, ChunkRecord, HybridRetriever, ScopeNotSupportedError, retrieve
from slimx_rag.settings import (
    ChunkSettings,
    EmbedSettings,
    IndexSettings,
    RetrievalSettings,
    StructuredChunkSettings,
)

# Bounded limits for the file-indexing endpoint (all tunable via env).
MAX_FILE_BYTES = int(os.getenv("RAG_MAX_FILE_BYTES", str(25 * 1024 * 1024)))
MAX_ELEMENTS = int(os.getenv("RAG_MAX_ELEMENTS", "20000"))


def _backend_config() -> dict[str, object]:
    raw = os.getenv("RAG_BACKEND_CONFIG", "")
    if raw:
        try:
            cfg = json.loads(raw)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Invalid RAG_BACKEND_CONFIG: {e}") from e
        if not isinstance(cfg, dict):
            raise HTTPException(status_code=500, detail="Invalid RAG_BACKEND_CONFIG: must be a JSON object")
        return cfg
    if os.getenv("RAG_INDEX_BACKEND", "local") == "qdrant":
        return {
            "url": os.getenv("QDRANT_URL", "http://qdrant:6333"),
            "collection": os.getenv("QDRANT_COLLECTION", "slimx_demo"),
        }
    return {}


def _index_settings() -> IndexSettings:
    return IndexSettings(
        backend=os.getenv("RAG_INDEX_BACKEND", "local"),
        backend_config=_backend_config(),
        top_k=int(os.getenv("RAG_TOP_K", "5")),
    )


_EMBED_OVERRIDE_FILENAME = "embed_override.json"


def _embed_override_path() -> Path:
    # Persist alongside the index (on the same volume) so an applied embedding choice
    # survives restarts. Written by POST /api/admin/embedding.
    return _index_path().parent / _EMBED_OVERRIDE_FILENAME


def _load_embed_override() -> dict[str, Any]:
    try:
        data = json.loads(_embed_override_path().read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _embed_settings() -> EmbedSettings:
    # Env supplies defaults; a persisted override (set at runtime via the admin endpoint)
    # wins so ControlRoom can switch the embedding model/device without an image change.
    override = _load_embed_override()
    dim = override.get("dim")
    return EmbedSettings(
        provider=str(override.get("provider") or os.getenv("RAG_EMBED_PROVIDER", "hash")),
        model=str(override.get("model") or os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")),
        hf_model=str(
            override.get("hf_model")
            or os.getenv("RAG_HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        ),
        dim=int(dim if dim is not None else os.getenv("RAG_EMBED_DIM", "384")),
        device=override["device"] if "device" in override else (os.getenv("RAG_EMBED_DEVICE") or None),
    )


def _write_embed_override(settings: EmbedSettings) -> None:
    path = _embed_override_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "provider": settings.provider,
                "model": settings.model,
                "hf_model": settings.hf_model,
                "dim": settings.dim,
                "device": settings.device,
            }
        ),
        "utf-8",
    )


def _reset_index() -> None:
    """Drop the on-disk index so the next ingest rebuilds it under the new embedding.

    Changing the embedding model/dim changes the vector space, so the existing index is
    invalid and must be rebuilt. This resets the local index + state files (the default and
    GPU-image backend); remote backends (qdrant/pgvector) expose no truncate primitive here,
    so those deployments must be rebuilt externally.
    """
    with _index_lock:
        for p in (_index_path(), _state_path()):
            try:
                p.unlink()
            except (FileNotFoundError, OSError):
                pass
        _reset_index_cache()


def _chunk_settings() -> ChunkSettings:
    defaults = ChunkSettings()
    return ChunkSettings(
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", str(defaults.chunk_size))),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", str(defaults.chunk_overlap))),
    )


def _structured_chunk_settings() -> StructuredChunkSettings:
    defaults = StructuredChunkSettings()
    return StructuredChunkSettings(
        target_tokens=int(os.getenv("RAG_TARGET_TOKENS", str(defaults.target_tokens))),
        max_tokens=int(os.getenv("RAG_MAX_TOKENS", str(defaults.max_tokens))),
    )


def _retrieval_settings() -> RetrievalSettings:
    defaults = RetrievalSettings()
    return RetrievalSettings(
        dense_candidates=int(os.getenv("RAG_DENSE_CANDIDATES", str(defaults.dense_candidates))),
        lexical_candidates=int(os.getenv("RAG_LEXICAL_CANDIDATES", str(defaults.lexical_candidates))),
        final_parents=int(os.getenv("RAG_FINAL_PARENTS", str(defaults.final_parents))),
        enable_lexical=os.getenv("RAG_ENABLE_LEXICAL", "1").lower() not in ("0", "false", "no"),
    )


def _current_lexical(backend: IndexBackend) -> Bm25Index | None:
    """Return the BM25 sidecar over the hot backend's corpus, rebuilt when it changes.

    Returns None for backends that cannot enumerate their corpus (remote/ANN), so hybrid
    retrieval truthfully falls back to dense-only rather than claiming a hybrid that did
    not run.
    """
    global _lexical, _lexical_token
    if not getattr(backend, "supports_inmemory_scope_filter", False):
        return None
    with _index_lock:
        token = _cache_key()
        if _lexical is None or _lexical_token != token:
            _lexical = Bm25Index().build((cid, text) for cid, text, _md in backend.iter_chunks())
            _lexical_token = token
        return _lexical


def _as_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _to_chunk_record(result: SearchResult) -> ChunkRecord:
    md = result.metadata or {}
    page = md.get("page")
    return ChunkRecord(
        chunk_id=result.chunk_id,
        text=result.text,
        parent_id=str(
            md.get("parent_id") or md.get("parent_doc_id") or md.get("doc_id") or result.chunk_id
        ),
        page_number=page if isinstance(page, int) and not isinstance(page, bool) else None,
        section=str(md["section"]) if md.get("section") is not None else None,
        page_type=str(md.get("page_type") or "unknown"),
        source_title=str(md.get("source_title") or md.get("title") or ""),
        entry=str(md.get("entry") or ""),
        token_count=_as_int(md.get("token_count")),
    )


def _chunks_to_documents(
    chunks: list[Any],
    *,
    workspace_id: str,
    document_id: str,
    doc_id: str,
    kb_relpath: str,
    content_hash_value: str,
) -> list[Document]:
    """Map RetrievalChunks to embeddable Documents.

    The identity-prefixed ``embedding_text`` is embedded and stored as the index text (so
    lexical/exact matching see the entity), while the clean ``display_text`` and the
    page/section/parent identity ride in metadata for citations and inspection.
    """
    docs: list[Document] = []
    for ch in chunks:
        docs.append(
            Document(
                page_content=ch.embedding_text,
                metadata={
                    "chunk_id": ch.chunk_id,
                    "doc_id": doc_id,
                    "kb_relpath": kb_relpath,
                    "content_hash": content_hash_value,
                    "workspace_id": workspace_id,
                    "document_id": document_id,
                    "parent_id": ch.parent_id,
                    "page": ch.page_number,
                    "section": ch.section,
                    "section_path": list(ch.section_path),
                    "page_type": ch.page_type.value,
                    "entry": ch.metadata.get("entry", ""),
                    "source_title": ch.source_title,
                    "display_text": ch.display_text,
                    "token_count": ch.token_count,
                    "ordinal": ch.ordinal,
                    "element_types": [t.value for t in ch.element_types],
                    "forced_split": ch.forced_split,
                },
            )
        )
    return docs


def _index_path() -> Path:
    return Path(os.getenv("RAG_INDEX_PATH", "output/index.jsonl"))


def _state_path() -> Path:
    return Path(os.getenv("RAG_STATE_PATH", "output/index_state.json"))


def _llm_timeout() -> float | None:
    raw = os.getenv("SLIMX_LLM_TIMEOUT", "")
    return float(raw) if raw else None


def _llm_max_tokens() -> int | None:
    raw = os.getenv("SLIMX_LLM_MAX_TOKENS", "")
    return int(raw) if raw else None


def _max_context_chars() -> int | None:
    raw = os.getenv("SLIMX_MAX_CONTEXT_CHARS", "")
    return int(raw) if raw else None


def _check_token(authorization: str | None) -> None:
    token = os.getenv("DEMO_AUTH_TOKEN")
    if not token:
        return
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Missing or invalid demo token")


class QuestionRequest(BaseModel):
    question: str
    model: str | None = None
    top_k: int | None = Field(default=None, gt=0)
    # Optional retrieval scope. When set, only chunks whose metadata matches are
    # returned (chunks are tagged with workspace_id/document_id at ingest time).
    workspace_id: str | None = None
    document_ids: list[str] | None = None


class EvalRequest(BaseModel):
    dataset: str = "examples/research_demo/eval/questions.jsonl"
    model: str | None = None
    top_k: int | None = Field(default=None, gt=0)


class IndexRequest(BaseModel):
    workspace_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] | None = None


class EmbeddingConfigRequest(BaseModel):
    # All optional; omitted fields keep the current value. Setting any of these changes the
    # vector space, so the index is reset and must be rebuilt by re-indexing documents.
    provider: str | None = None
    model: str | None = None
    hf_model: str | None = None
    dim: int | None = Field(default=None, gt=0)
    device: str | None = None


app = FastAPI(title="SlimX-RAG Research Demo")

# One hot, in-process index shared by all requests. retrieve() otherwise re-reads and
# re-parses the entire index file on every call (seconds per request as the corpus grows);
# here we load it once and reuse it, refreshing only when the on-disk file changes (e.g. an
# external CLI rebuild). A single reentrant lock guards both the cached backend's in-memory
# state and the read-modify-write ingest path, so reads never observe a half-applied write.
_index_lock = threading.RLock()
# A SEPARATE lock for embedding so embed work never blocks (or is blocked by) index
# reads/writes. Embedding is the expensive step; keeping it off the index lock lets a
# write's embedding run while reads proceed, while still serializing model use for safety.
_embed_lock = threading.Lock()
_backend: IndexBackend | None = None
_backend_token: tuple[object, ...] | None = None
# BM25 lexical sidecar, rebuilt when the index file changes (keyed by the same token).
_lexical: Bm25Index | None = None
_lexical_token: tuple[object, ...] | None = None


def _index_token(path: Path) -> tuple[object, ...]:
    try:
        st = path.stat()
        return (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return (str(path), -1, -1)


def _cache_key() -> tuple[object, ...]:
    settings = _index_settings()
    return (_index_token(_index_path()), settings.backend, repr(settings.backend_config), str(_state_path()))


def _current_backend() -> IndexBackend:
    """Return the shared, loaded index backend, (re)loading only when its inputs change."""
    global _backend, _backend_token
    with _index_lock:
        token = _cache_key()
        if _backend is None or _backend_token != token:
            backend = make_index_backend(_index_path(), settings=_index_settings(), state_path=_state_path())
            backend.load()
            _backend = backend
            _backend_token = token
        return _backend


def _mark_index_written() -> None:
    """Re-token the cache after our own write so the next read reuses the hot backend."""
    global _backend_token
    with _index_lock:
        _backend_token = _cache_key()


def _reset_index_cache() -> None:
    """Drop the cached backend + lexical sidecar (used by tests; safe to call anytime)."""
    global _backend, _backend_token, _lexical, _lexical_token
    with _index_lock:
        _backend = None
        _backend_token = None
        _lexical = None
        _lexical_token = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "index_backend": _index_settings().backend,
        "embed_provider": _embed_settings().provider,
        "embed_device": _embed_settings().device,
        "llm_model": os.getenv("SLIMX_LLM_MODEL", "fake:grounded"),
    }


@app.get("/ready")
def ready(authorization: str | None = Header(default=None)) -> JSONResponse:
    """Deep readiness: prove the service can actually index/retrieve, not just echo config.

    Unlike the shallow ``/health`` liveness probe, this verifies the index output directory is
    writable, the backend loads, and the embedder initializes, and it flags an embedding-
    dimension mismatch between the existing index and the configured embedder. Returns 200
    ``{"ready": true, ...}`` or 503 ``{"ready": false, "reason", ...}`` so a downstream app can
    gate indexing on real readiness instead of liveness. Keep using ``/health`` for container
    liveness.
    """
    _check_token(authorization)
    index_settings = _index_settings()
    embed_settings = _embed_settings()

    def not_ready(reason: str, **extra: Any) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "reason": reason,
                "index_backend": index_settings.backend,
                **extra,
            },
        )

    # 1. The index output directory must be writable (state + index files live here).
    out_dir = _index_path().parent
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return not_ready("index_dir_unwritable", detail=type(exc).__name__)
    if not os.access(out_dir, os.W_OK):
        return not_ready("index_dir_unwritable")

    # 2. The backend must load (report its current size + the index's stored embedding dim).
    try:
        with _index_lock:
            backend = _current_backend()
            index_count = len(backend)
            stored_dim = (backend.state.embed or {}).get("dim")
    except Exception as exc:  # noqa: BLE001 — readiness must surface a reason, never raise
        return not_ready("backend_load_failed", detail=type(exc).__name__)

    # 3. The embedder must initialize (loads + caches the model).
    try:
        embed_dim = get_cached_embedder(embed_settings).dim
    except Exception as exc:  # noqa: BLE001
        return not_ready("embedder_init_failed", detail=type(exc).__name__)

    # 4. A populated index whose vectors don't match the configured embedder can't be queried
    #    correctly — flag it so the caller reindexes rather than silently mixing dimensions.
    if stored_dim and embed_dim and int(stored_dim) != int(embed_dim):
        return not_ready(
            "embedding_dim_mismatch", stored_dim=int(stored_dim), embed_dim=int(embed_dim)
        )

    model = embed_settings.hf_model if embed_settings.provider == "hf" else embed_settings.model
    return JSONResponse(
        status_code=200,
        content={
            "ready": True,
            "index_backend": index_settings.backend,
            "index_count": index_count,
            "embed_provider": embed_settings.provider,
            "embed_model": model,
            "embed_dim": embed_dim,
        },
    )


@app.get("/api/config")
def config(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_token(authorization)
    return {
        "index_path": str(_index_path()),
        "state_path": str(_state_path()),
        "index": {
            "backend": _index_settings().backend,
            "top_k": _index_settings().top_k,
        },
        "embed": {
            "provider": _embed_settings().provider,
            "model": _embed_settings().model,
            "hf_model": _embed_settings().hf_model,
            "device": _embed_settings().device,
        },
        "llm_model": os.getenv("SLIMX_LLM_MODEL", "fake:grounded"),
    }


@app.post("/api/admin/embedding")
def set_embedding(
    payload: EmbeddingConfigRequest, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """Set the active embedding config and reset the index (guarded by the demo token).

    Switching the embedding model/device changes the vector space, so the index is reset
    here; the caller (ControlRoom) must then re-index its documents under the new embedding.
    The choice is persisted alongside the index, so it survives a restart.
    """
    _check_token(authorization)
    current = _embed_settings()
    merged = EmbedSettings(
        provider=payload.provider or current.provider,
        model=payload.model or current.model,
        hf_model=payload.hf_model or current.hf_model,
        dim=payload.dim or current.dim,
        device=payload.device if payload.device is not None else current.device,
    )
    try:
        merged.validate()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _write_embed_override(merged)
    _reset_index()
    return {
        "embed": {
            "provider": merged.provider,
            "model": merged.model,
            "hf_model": merged.hf_model,
            "dim": merged.dim,
            "device": merged.device,
        },
        "index_reset": True,
    }


def _hybrid_retrieve_response(
    payload: QuestionRequest, backend: IndexBackend, embed_settings: EmbedSettings
) -> dict[str, Any]:
    """Run multi-stage hybrid retrieval and shape the rich, inspectable response."""
    started = time.perf_counter()
    question = payload.question
    scope_ws = str(payload.workspace_id) if payload.workspace_id else None
    scope_docs = {str(d) for d in payload.document_ids} if payload.document_ids else None

    def in_scope(md: dict[str, object]) -> bool:
        if scope_ws is not None and str(md.get("workspace_id")) != scope_ws:
            return False
        if scope_docs is not None and str(md.get("document_id")) not in scope_docs:
            return False
        return True

    # Embed the query once, OFF the index lock (and off any per-request model rebuild).
    embedder = get_cached_embedder(embed_settings)
    with _embed_lock:
        qvec = [float(x) for x in embedder.embed_query(question)]

    meta_cache: dict[str, dict[str, object]] = {}
    record_cache: dict[str, ChunkRecord | None] = {}
    settings = _retrieval_settings()

    with _index_lock:
        lexical = _current_lexical(backend)

        def dense_search(_q: str, k: int) -> list[tuple[str, float]]:
            if scope_ws or scope_docs:
                raw = backend.query(qvec, top_k=len(backend) or k)
                raw = [r for r in raw if in_scope(r.metadata or {})]
            else:
                raw = backend.query(qvec, top_k=k)
            return [(r.chunk_id, float(r.score)) for r in raw[:k]]

        def get_record(cid: str) -> ChunkRecord | None:
            if cid in record_cache:
                return record_cache[cid]
            found = backend.get_chunks([cid])
            rec: ChunkRecord | None = None
            if found:
                md = found[0].metadata or {}
                if in_scope(md):  # scope enforced here: out-of-scope -> None -> dropped
                    meta_cache[cid] = md
                    rec = _to_chunk_record(found[0])
            record_cache[cid] = rec
            return rec

        retriever = HybridRetriever(dense_search=dense_search, get_record=get_record, lexical=lexical)
        results, trace = retriever.retrieve(question, settings=settings)

    top_k = payload.top_k or _index_settings().top_k
    if top_k:
        results = results[:top_k]
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    chunks_out: list[dict[str, Any]] = []
    for r in results:
        md = meta_cache.get(r.chunk_id, {})
        chunks_out.append(
            {
                "chunk_id": r.chunk_id,
                "score": r.fusion_score,
                "text": md.get("display_text") or r.text,
                "citation": r.citation(),
                "metadata": {
                    "document_id": md.get("document_id"),
                    "workspace_id": md.get("workspace_id"),
                    "parent_id": r.parent_id,
                    "page": r.page_number,
                    "section": r.section,
                    "section_path": md.get("section_path"),
                    "page_type": r.page_type,
                    "entry": r.entry,
                    "source_title": r.source_title,
                    "token_count": r.token_count,
                    "element_types": md.get("element_types"),
                    "dense_rank": r.dense_rank,
                    "dense_score": r.dense_score,
                    "lexical_rank": r.lexical_rank,
                    "lexical_score": r.lexical_score,
                    "exact_match": r.exact_match,
                    "exact_score": r.exact_score,
                    "fusion_rank": r.fusion_rank,
                    "rerank_score": r.rerank_score,
                    "final_rank": r.final_rank,
                    "parent_reason": r.parent_reason,
                    "sibling_expanded": r.sibling_expanded,
                },
            }
        )
    model = embed_settings.hf_model if embed_settings.provider == "hf" else embed_settings.model
    return {
        "query": question,
        "chunks": chunks_out,
        "embed": {"provider": embed_settings.provider, "model": model, "dim": embed_settings.dim},
        "elapsed_ms": elapsed_ms,
        "retrieval_strategy": trace["strategy"],
        "trace": trace,
    }


@app.post("/api/retrieve")
def retrieve_endpoint(payload: QuestionRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_token(authorization)
    embed_settings = _embed_settings()
    backend = _current_backend()
    # Hybrid retrieval needs to enumerate the corpus (BM25) and read chunk metadata, which
    # only the in-memory local backend supports. Remote/ANN backends use the legacy dense
    # path (still enforcing scope or raising ScopeNotSupportedError).
    if not getattr(backend, "supports_inmemory_scope_filter", False):
        try:
            with _index_lock:
                result = retrieve(
                    payload.question,
                    index_path=_index_path(),
                    state_path=_state_path(),
                    embed_settings=embed_settings,
                    index_settings=_index_settings(),
                    top_k=payload.top_k,
                    workspace_id=payload.workspace_id,
                    document_ids=payload.document_ids,
                    backend=backend,
                )
        except ScopeNotSupportedError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return result.to_dict()
    return _hybrid_retrieve_response(payload, backend, embed_settings)


@app.post("/api/index")
def index_endpoint(payload: IndexRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Ingest one posted document into the live index (chunk -> embed -> upsert).

    Service mode is otherwise retrieval-only; this lets a downstream app index uploaded
    documents over HTTP instead of via the CLI. Identity is derived from
    workspace_id/document_id so re-posting the same document is idempotent (and a content
    change replaces just that document's chunks). The next /api/retrieve sees the new
    chunks immediately — reads and writes share one hot in-memory backend under _index_lock.
    """
    _check_token(authorization)
    embed_settings = _embed_settings()
    index_settings = _index_settings()
    chunk_settings = _chunk_settings()

    kb_relpath = f"{payload.workspace_id}/{payload.document_id}"
    doc_id = path_id(kb_relpath)
    ch = content_hash(payload.text)
    metadata: dict[str, Any] = {
        "doc_id": doc_id,
        "kb_relpath": kb_relpath,
        "content_hash": ch,
        "content_len": len(payload.text),
        "source": f"api://{kb_relpath}",
        "title": (payload.metadata or {}).get("title") or payload.document_id,
        "workspace_id": payload.workspace_id,
        "document_id": payload.document_id,
    }
    # Carry through caller metadata without overriding identity fields.
    for key, value in (payload.metadata or {}).items():
        if key not in ("doc_id", "kb_relpath", "content_hash"):
            metadata.setdefault(key, value)

    doc = Document(page_content=payload.text, metadata=metadata)
    chunks = chunk_documents(
        [doc],
        chunk_size=chunk_settings.chunk_size,
        chunk_overlap=chunk_settings.chunk_overlap,
        separators=chunk_settings.separators,
    )

    with _index_lock:
        idx = _current_backend()  # hot backend; load is amortized across posts
        idx.set_embed_config(embed_settings)
        idx.delete_doc(doc_id)  # replace this document's chunks; no-op when new
        items: list[EmbeddedChunk] = list(embed_chunks(iter(chunks), settings=embed_settings))
        upserted = idx.upsert(items, skip_existing=False)
        idx.save()
        _mark_index_written()  # our own write must not trigger a reload on the next read
        chunk_ids = [item.chunk_id for item in items]
        idx.commit_doc_state(doc_id, ch, chunk_ids)
        total = len(idx)

    model = embed_settings.hf_model if embed_settings.provider == "hf" else embed_settings.model
    return {
        "status": "ready",
        "doc_id": doc_id,
        "rag_index_ref": f"slimx-rag:{doc_id}",
        "chunk_count": len(chunk_ids),
        "upserted": upserted,
        "total": total,
        "vector_backend": index_settings.backend,
        "embed": {"provider": embed_settings.provider, "model": model, "dim": embed_settings.dim},
    }


@app.post("/api/index/file")
def index_file_endpoint(
    file: UploadFile = File(...),  # noqa: B008 — FastAPI dependency-injection default
    workspace_id: str = Form(...),
    document_id: str = Form(...),
    filename: str | None = Form(default=None),
    mime_type: str | None = Form(default=None),
    title: str | None = Form(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Index an ORIGINAL file page-aware: parse -> structured chunk -> embed -> index.

    Unlike ``/api/index`` (flattened text), this preserves PDF pages and DOCX/Markdown
    structure so chunks stay self-describing. Parsing, chunking and embedding run OUTSIDE
    the index mutation lock; only the delete/upsert/save/state section holds ``_index_lock``
    (embedding holds the separate ``_embed_lock``). Errors are redacted (no document text).
    """
    _check_token(authorization)
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="empty file")
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"file exceeds {MAX_FILE_BYTES} bytes")

    embed_settings = _embed_settings()
    source = DocumentSource(
        document_id=document_id,
        filename=filename or file.filename or document_id,
        mime_type=mime_type,
        content=raw,
        workspace_id=workspace_id,
        metadata={"title": title} if title else {},
    )

    timings: dict[str, int] = {}
    t0 = time.perf_counter()
    try:
        parsed = parse_document(source)
    except DocumentError as exc:
        # Redacted: type only, never the document content.
        raise HTTPException(status_code=422, detail=f"parse_failed: {type(exc).__name__}") from exc
    timings["parse_ms"] = int((time.perf_counter() - t0) * 1000)
    if parsed.element_count > MAX_ELEMENTS:
        raise HTTPException(
            status_code=413, detail=f"document has {parsed.element_count} elements; max {MAX_ELEMENTS}"
        )

    t1 = time.perf_counter()
    token_counter = make_token_counter(embed_settings)
    chunks = chunk_parsed_document(
        parsed, settings=_structured_chunk_settings(), token_counter=token_counter
    )
    timings["chunk_ms"] = int((time.perf_counter() - t1) * 1000)

    kb_relpath = f"{workspace_id}/{document_id}"
    doc_id = path_id(kb_relpath)
    ch_hash = content_hash("\n\n".join(p.text for p in parsed.pages))
    docs = _chunks_to_documents(
        chunks,
        workspace_id=workspace_id,
        document_id=document_id,
        doc_id=doc_id,
        kb_relpath=kb_relpath,
        content_hash_value=ch_hash,
    )

    embedder = get_cached_embedder(embed_settings)
    t2 = time.perf_counter()
    with _embed_lock:  # embedding off the index lock
        items: list[EmbeddedChunk] = list(
            embed_chunks(iter(docs), settings=embed_settings, embedder=embedder)
        )
    timings["embed_ms"] = int((time.perf_counter() - t2) * 1000)

    t3 = time.perf_counter()
    with _index_lock:  # hold the lock ONLY for the protected delete/upsert/save/state section
        idx = _current_backend()
        idx.set_embed_config(embed_settings)
        idx.delete_doc(doc_id)
        upserted = idx.upsert(items, skip_existing=False)
        idx.save()
        _mark_index_written()
        idx.commit_doc_state(doc_id, ch_hash, [it.chunk_id for it in items])
        total = len(idx)
        lexical_capable = bool(getattr(idx, "supports_inmemory_scope_filter", False))
    timings["index_ms"] = int((time.perf_counter() - t3) * 1000)

    model = embed_settings.hf_model if embed_settings.provider == "hf" else embed_settings.model
    parent_count = len({str(d.metadata["parent_id"]) for d in docs})
    return {
        "status": "ready",
        "document_id": document_id,
        "doc_id": doc_id,
        "rag_index_ref": f"slimx-rag:{doc_id}",
        "parser": parsed.parser_name,
        "parser_version": parsed.parser_version,
        "source_type": parsed.source_type,
        "page_count": parsed.page_count,
        "element_count": parsed.element_count,
        "parent_count": parent_count,
        "chunk_count": len(items),
        "upserted": upserted,
        "total": total,
        "embedding_provider": embed_settings.provider,
        "embedding_model": model,
        "embedding_dim": embedder.dim,
        "embedding_max_seq_len": embedder.max_seq_length,
        "vector_backend": _index_settings().backend,
        "lexical_retrieval": lexical_capable,
        "warnings": list(parsed.warnings),
        "timings_ms": timings,
    }


@app.get("/api/documents/{document_id}/chunks")
def document_chunks_endpoint(
    document_id: str,
    workspace_id: str,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """List one document's indexed chunks in order, for inspection (not retrieval).

    The document's chunk ids are recorded in IndexState at ingest time (in chunk order);
    we read each chunk's text + metadata back from the live backend. Identity is derived
    from workspace_id/document_id exactly as ``/api/index`` derives it, so a caller that
    indexed a document can list its chunks. Unknown / not-yet-indexed documents return an
    empty list (chunk_count 0) rather than a 404 — chunk listing is a best-effort view.
    """
    _check_token(authorization)
    doc_id = path_id(f"{workspace_id}/{document_id}")
    with _index_lock:
        backend = _current_backend()
        entry = backend.state.docs.get(doc_id) or {}
        chunk_ids = [str(c) for c in (entry.get("chunk_ids") or [])]
        stored = backend.get_chunks(chunk_ids)
    chunks: list[dict[str, Any]] = []
    for ordinal, sc in enumerate(stored):
        md = sc.metadata or {}
        chunks.append(
            {
                "chunk_id": sc.chunk_id,
                "ordinal": ordinal,
                "text": sc.text,
                "page": md.get("page"),
                "section": md.get("section") or md.get("title"),
                "start_offset": md.get("start_offset"),
                "end_offset": md.get("end_offset"),
            }
        )
    return {
        "document_id": document_id,
        "doc_id": doc_id,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


@app.delete("/api/documents/{document_id}")
def delete_document_endpoint(
    document_id: str,
    workspace_id: str,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Permanently remove one document's chunks from the live index.

    Identity is derived from workspace_id/document_id exactly as ``/api/index`` derives it,
    so a caller that indexed a document can delete it. Idempotent: deleting an unknown /
    already-deleted document is a no-op (``deleted_chunks`` 0), not a 404 — mirrors the
    chunks endpoint. Reads and writes share one hot in-memory backend under ``_index_lock``,
    so the next ``/api/retrieve`` no longer sees the chunks. State is committed strictly
    after the backend save, the same crash-safety ordering as ``/api/index``.
    """
    _check_token(authorization)
    doc_id = path_id(f"{workspace_id}/{document_id}")
    with _index_lock:
        idx = _current_backend()
        deleted = idx.delete_doc(doc_id)  # drop this document's vectors; no-op when unknown
        idx.save()
        _mark_index_written()  # our own write must not trigger a reload on the next read
        idx.forget_doc_state(doc_id)  # forget the doc -> chunk_ids bookkeeping (state last)
        total = len(idx)
    return {
        "status": "deleted",
        "document_id": document_id,
        "doc_id": doc_id,
        "deleted_chunks": deleted,
        "total": total,
    }


@app.post("/api/ask")
def ask_endpoint(payload: QuestionRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_token(authorization)
    try:
        with _index_lock:
            retrieval = retrieve(
                payload.question,
                index_path=_index_path(),
                state_path=_state_path(),
                embed_settings=_embed_settings(),
                index_settings=_index_settings(),
                top_k=payload.top_k,
                workspace_id=payload.workspace_id,
                document_ids=payload.document_ids,
                backend=_current_backend(),
            )
    except ScopeNotSupportedError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    # answer() may call out to an LLM; run it outside the index lock.
    model: str = payload.model or os.getenv("SLIMX_LLM_MODEL") or "fake:grounded"
    result = answer(
        payload.question,
        retrieval,
        model=model,
        timeout=_llm_timeout(),
        max_tokens=_llm_max_tokens(),
        max_context_chars=_max_context_chars(),
    )
    return result.to_dict()


@app.post("/api/eval/run")
def eval_endpoint(payload: EvalRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_token(authorization)
    cases = load_eval_cases(Path(payload.dataset))
    model: str = payload.model or os.getenv("SLIMX_LLM_MODEL") or "fake:grounded"
    report = run_eval(
        cases,
        index_path=_index_path(),
        state_path=_state_path(),
        embed_settings=_embed_settings(),
        index_settings=_index_settings(),
        model=model,
        top_k=payload.top_k or _index_settings().top_k,
        timeout=_llm_timeout(),
        max_tokens=_llm_max_tokens(),
        max_context_chars=_max_context_chars(),
    )
    return {"markdown": report.to_markdown(), "cases": report.cases}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    static = Path(__file__).resolve().parents[1] / "static" / "index.html"
    if not static.exists():
        raise HTTPException(status_code=404, detail="Demo UI not installed")
    return static.read_text(encoding="utf-8")
