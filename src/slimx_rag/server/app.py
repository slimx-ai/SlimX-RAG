from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from slimx_rag.answer import answer
from slimx_rag.chunk import chunk_documents
from slimx_rag.core.hashing import content_hash, path_id
from slimx_rag.embed import EmbeddedChunk, embed_chunks
from slimx_rag.eval import load_eval_cases, run_eval
from slimx_rag.index import IndexBackend, make_index_backend
from slimx_rag.retrieval import ScopeNotSupportedError, retrieve
from slimx_rag.settings import ChunkSettings, EmbedSettings, IndexSettings


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
        provider=override.get("provider") or os.getenv("RAG_EMBED_PROVIDER", "hash"),
        model=override.get("model") or os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small"),
        hf_model=override.get("hf_model")
        or os.getenv("RAG_HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
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
_backend: IndexBackend | None = None
_backend_token: tuple[object, ...] | None = None


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
    """Drop the cached backend (used by tests; safe to call anytime)."""
    global _backend, _backend_token
    with _index_lock:
        _backend = None
        _backend_token = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "index_backend": _index_settings().backend,
        "embed_provider": _embed_settings().provider,
        "embed_device": _embed_settings().device,
        "llm_model": os.getenv("SLIMX_LLM_MODEL", "fake:grounded"),
    }


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


@app.post("/api/retrieve")
def retrieve_endpoint(payload: QuestionRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_token(authorization)
    try:
        with _index_lock:
            result = retrieve(
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
    return result.to_dict()


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
