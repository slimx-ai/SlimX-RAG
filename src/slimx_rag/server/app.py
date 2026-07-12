from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from slimx_rag.answer import answer
from slimx_rag.chunk import TokenCounter, chunk_documents, chunk_parsed_document
from slimx_rag.core.hashing import content_hash, path_id
from slimx_rag.document import DocumentError, DocumentSource, parse_document
from slimx_rag.embed import EmbeddedChunk, embed_chunks, get_cached_embedder, make_token_counter
from slimx_rag.eval import load_eval_cases, run_eval
from slimx_rag.index import (
    INDEX_BUILD_RECEIPT_FILENAME,
    INDEX_INSTANCE_ID_FILENAME,
    INDEX_SCHEMA_VERSION,
    IndexBackend,
    IndexInstanceLease,
    IndexSignature,
    IndexState,
    build_index_signature,
    load_index_build_receipt,
    locked_index_instance,
    make_index_backend,
    resolve_embedding_dimension,
    write_index_build_receipt,
)
from slimx_rag.index.types import SearchResult
from slimx_rag.retrieval import Bm25Index, ChunkRecord, HybridRetriever, ScopeNotSupportedError, retrieve
from slimx_rag.settings import (
    ChunkSettings,
    EmbedSettings,
    IndexSettings,
    RetrievalSettings,
    StructuredChunkSettings,
)
from slimx_rag.utils.commons import _atomic_write_text
from slimx_rag.version import get_engine_version

# Bounded limits for the file-indexing endpoint (all tunable via env).
MAX_FILE_BYTES = int(os.getenv("RAG_MAX_FILE_BYTES", str(25 * 1024 * 1024)))
MAX_ELEMENTS = int(os.getenv("RAG_MAX_ELEMENTS", "20000"))
logger = logging.getLogger(__name__)


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
        hf_model=str(override.get("hf_model") or os.getenv("RAG_HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2")),
        dim=int(dim if dim is not None else os.getenv("RAG_EMBED_DIM", "384")),
        device=override["device"] if "device" in override else (os.getenv("RAG_EMBED_DEVICE") or None),
    )


def _write_embed_override(settings: EmbedSettings) -> None:
    path = _embed_override_path()
    _atomic_write_text(
        path,
        json.dumps(
            {
                "provider": settings.provider,
                "model": settings.model,
                "hf_model": settings.hf_model,
                "dim": settings.dim,
                "device": settings.device,
            }
        ),
    )


class UnsupportedIndexResetError(RuntimeError):
    """The selected backend has no verified corpus-reset primitive."""


class IndexResetPartialFailure(RuntimeError):
    """A reset failed and the previous corpus could not be fully restored."""


def _stage_reset_artifact(path: Path, backup: Path) -> None:
    os.replace(path, backup)


def _restore_reset_artifact(backup: Path, path: Path) -> None:
    os.replace(backup, path)


def _reset_index(
    index_settings: IndexSettings,
    *,
    lease: IndexInstanceLease,
    new_embed_settings: EmbedSettings,
    persist_embed_override: bool,
) -> str:
    """Transactionally stage the active local corpus and publish a fresh identity.

    Both embedding changes and explicit maintenance resets use this one verified primitive.
    Only local backends have a verified reset here. Remote backends must be reset externally;
    rejecting them prevents a false ``index_reset`` response and preserves the old instance
    identity until the corpus is truly gone.
    """
    backend = (index_settings.backend or "local").lower().strip()
    if backend not in {"local", "faiss"}:
        raise UnsupportedIndexResetError(
            f"index reset is unsupported for backend {backend!r}; reset its corpus externally"
        )
    transaction_id = secrets.token_hex(8)
    artifacts = [
        _index_instance_id_path(),
        _index_build_receipt_path(),
        _state_path(),
        _index_path(),
    ]
    if persist_embed_override:
        artifacts.insert(2, _embed_override_path())
    if backend == "faiss":
        artifacts.append(_index_path().with_suffix(_index_path().suffix + ".meta.json"))
    staged: list[tuple[Path, Path]] = []
    wrote_new_override = False
    try:
        # Identity is staged first. Any partial mutation therefore cannot keep serving the
        # old corpus fingerprint; a fully successful rollback restores it.
        for path in artifacts:
            if not path.exists():
                continue
            backup = path.with_name(f".{path.name}.reset-{transaction_id}")
            _stage_reset_artifact(path, backup)
            staged.append((path, backup))
        if persist_embed_override:
            _write_embed_override(new_embed_settings)
            wrote_new_override = True
        new_instance_id = lease.publish_new()
    except Exception as exc:
        rollback_errors: list[str] = []
        staged_paths = {original for original, _backup in staged}
        if wrote_new_override and _embed_override_path() not in staged_paths:
            try:
                _embed_override_path().unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(type(rollback_exc).__name__)
        for original, backup in reversed(staged):
            if not backup.exists():
                rollback_errors.append("MissingBackup")
                continue
            try:
                _restore_reset_artifact(backup, original)
            except OSError as rollback_exc:
                rollback_errors.append(type(rollback_exc).__name__)
        if rollback_errors:
            # Rollback could not prove restoration. Keep identity/receipt invalidated so
            # callers cannot mistake a partial corpus for the old successful build.
            try:
                lease.remove()
            except OSError as invalidation_exc:
                rollback_errors.append(f"identity-{type(invalidation_exc).__name__}")
            try:
                _index_build_receipt_path().unlink(missing_ok=True)
            except OSError as invalidation_exc:
                rollback_errors.append(f"receipt-{type(invalidation_exc).__name__}")
            raise IndexResetPartialFailure(
                "index_reset_partial_failure: rollback=" + ",".join(rollback_errors)
            ) from exc
        raise

    _reset_index_cache()
    for _original, backup in staged:
        try:
            backup.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            logger.warning("Could not remove inactive reset backup: %s", type(cleanup_exc).__name__)
    return new_instance_id


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


def _index_signature_contract(
    *,
    index_settings: IndexSettings,
    embed_settings: EmbedSettings,
    embedding_dimension: int | None,
    index_instance_id: str | None,
    structured_token_counter: TokenCounter | None,
    text_chunk_settings: ChunkSettings | None = None,
    file_chunk_settings: StructuredChunkSettings | None = None,
    signature_complete: bool = True,
    index_schema_version: int = INDEX_SCHEMA_VERSION,
) -> IndexSignature:
    return build_index_signature(
        index_settings=index_settings,
        embed_settings=embed_settings,
        chunk_settings=text_chunk_settings or _chunk_settings(),
        structured_chunk_settings=file_chunk_settings or _structured_chunk_settings(),
        embedding_dimension=embedding_dimension,
        index_instance_id=index_instance_id,
        structured_token_counter=structured_token_counter,
        signature_complete=signature_complete,
        index_schema_version=index_schema_version,
    )


def _index_signature_payload(
    *,
    index_settings: IndexSettings,
    embed_settings: EmbedSettings,
    embedding_dimension: int | None,
    index_instance_id: str | None,
    structured_token_counter: TokenCounter | None,
    text_chunk_settings: ChunkSettings | None = None,
    file_chunk_settings: StructuredChunkSettings | None = None,
    signature_complete: bool = True,
    index_schema_version: int = INDEX_SCHEMA_VERSION,
) -> dict[str, object]:
    return _index_signature_contract(
        index_settings=index_settings,
        embed_settings=embed_settings,
        embedding_dimension=embedding_dimension,
        index_instance_id=index_instance_id,
        structured_token_counter=structured_token_counter,
        text_chunk_settings=text_chunk_settings,
        file_chunk_settings=file_chunk_settings,
        signature_complete=signature_complete,
        index_schema_version=index_schema_version,
    ).to_dict()


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
        parent_id=str(md.get("parent_id") or md.get("parent_doc_id") or md.get("doc_id") or result.chunk_id),
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


def _index_instance_id_path() -> Path:
    return _index_path().parent / INDEX_INSTANCE_ID_FILENAME


def _index_build_receipt_path() -> Path:
    return _index_path().parent / INDEX_BUILD_RECEIPT_FILENAME


def _llm_timeout() -> float | None:
    raw = os.getenv("SLIMX_LLM_TIMEOUT", "")
    return float(raw) if raw else None


def _llm_max_tokens() -> int | None:
    raw = os.getenv("SLIMX_LLM_MAX_TOKENS", "")
    return int(raw) if raw else None


def _max_context_chars() -> int | None:
    raw = os.getenv("SLIMX_MAX_CONTEXT_CHARS", "")
    return int(raw) if raw else None


def _auth_token() -> str | None:
    """The configured service token. ``RAG_AUTH_TOKEN`` is the canonical env name;
    ``DEMO_AUTH_TOKEN`` remains supported as the deprecated legacy alias."""
    return os.getenv("RAG_AUTH_TOKEN") or os.getenv("DEMO_AUTH_TOKEN")


def _auth_enabled() -> bool:
    return bool(_auth_token())


def _check_token(authorization: str | None) -> None:
    token = _auth_token()
    if not token:
        return
    # Constant-time comparison: a plain `!=` short-circuits on the first differing byte and
    # leaks token prefixes through response timing.
    expected = f"Bearer {token}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid service token")


def _check_index_reset_token(authorization: str | None) -> None:
    """Require the canonical service token for the destructive maintenance reset."""
    token = os.getenv("RAG_AUTH_TOKEN")
    if not token:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "index_reset_auth_not_configured",
                "owner_action": "Configure RAG_AUTH_TOKEN before using the index reset endpoint.",
            },
        )
    expected = f"Bearer {token}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid service token")


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


class IndexResetRequest(BaseModel):
    """Explicit destructive confirmation plus required optimistic corpus preconditions."""

    confirmation: Literal["RESET INDEX"]
    # Required but nullable: callers must explicitly assert that they observed either a
    # concrete instance or an empty/uninitialized corpus. Omitting the field is not allowed.
    expected_index_instance_id: str | None = Field(pattern=r"^idx_[0-9a-f]{32}$")
    # Also required but nullable. Null is an explicit assertion that no trustworthy
    # fingerprint exists (for example, a malformed receipt); instance CAS still applies.
    expected_compatibility_fingerprint: str | None = Field(pattern=r"^[0-9a-f]{16}$")


class IndexResetResponse(BaseModel):
    index_reset: Literal[True]
    previous_index_instance_id: str | None
    previous_index_signature: dict[str, Any]
    previous_index_signature_source: Literal["persisted_build", "configured_partial"]
    new_index_instance_id: str
    index_signature: dict[str, Any]
    index_signature_source: Literal["configured_partial"]
    engine_version: str


app = FastAPI(title="SlimX-RAG Research Demo", version=get_engine_version())

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
    index_settings = _index_settings()
    embed_settings = _embed_settings()
    return {
        "status": "ok",
        "index_backend": index_settings.backend,
        "embed_provider": embed_settings.provider,
        "embed_device": embed_settings.device,
        "llm_model": os.getenv("SLIMX_LLM_MODEL", "fake:grounded"),
        # Whether this server enforces a bearer token. Lets a caller that *sends* a token
        # detect a deployment where the server never received one (auth silently off).
        "auth_enabled": _auth_enabled(),
        "engine_version": get_engine_version(),
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
    signature_dimension: int | None = embed_settings.dim
    signature_schema_version = INDEX_SCHEMA_VERSION
    signature_instance_id: str | None = None
    signature_token_counter: TokenCounter | None = None
    authoritative_signature: dict[str, object] | None = None

    def not_ready(reason: str, **extra: Any) -> JSONResponse:
        partial = _index_signature_payload(
            index_settings=index_settings,
            embed_settings=embed_settings,
            embedding_dimension=signature_dimension,
            index_instance_id=signature_instance_id,
            structured_token_counter=signature_token_counter,
            signature_complete=False,
            index_schema_version=signature_schema_version,
        )
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "reason": reason,
                "index_backend": index_settings.backend,
                "auth_enabled": _auth_enabled(),
                "engine_version": get_engine_version(),
                "index_signature": authoritative_signature or partial,
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
    try:
        # Identity, durable receipt, backend inspection, and active signature comparison are
        # one critical section. Reset/index cannot race a success response into describing a
        # different corpus instance.
        with _index_lock, locked_index_instance(_index_instance_id_path(), create=True) as lease:
            # Reset writes the embedding override under this same lease. Re-read settings
            # here so readiness cannot validate a new corpus with a pre-reset snapshot.
            index_settings = _index_settings()
            embed_settings = _embed_settings()
            signature_dimension = embed_settings.dim
            signature_instance_id = lease.instance_id
            try:
                receipt = load_index_build_receipt(_index_build_receipt_path())
            except RuntimeError as exc:
                return not_ready("index_build_receipt_invalid", detail=type(exc).__name__)
            if receipt is not None:
                authoritative_signature = receipt.index_signature.to_dict()
                if receipt.index_signature.index_instance_id != signature_instance_id:
                    active_signature = _index_signature_payload(
                        index_settings=index_settings,
                        embed_settings=embed_settings,
                        embedding_dimension=signature_dimension,
                        index_instance_id=signature_instance_id,
                        structured_token_counter=None,
                        signature_complete=False,
                    )
                    return not_ready(
                        "index_build_receipt_instance_mismatch",
                        active_index_signature=active_signature,
                    )

            try:
                IndexState.load(_state_path())
            except RuntimeError as exc:
                return not_ready("index_state_invalid", detail=type(exc).__name__)

            # 2. The backend must load and reveal its corpus dimension when possible.
            backend = _current_backend()
            index_count = len(backend)
            has_known_corpus = index_count > 0 or bool(backend.state.docs)
            backend_dim = backend.dim
            persisted_actual_dim = (backend.state.embed or {}).get("actual_dim")
            stored_dim = resolve_embedding_dimension(
                backend_dimension=backend_dim,
                persisted_actual_dimension=persisted_actual_dim,
            )
            signature_dimension = resolve_embedding_dimension(
                backend_dimension=backend_dim,
                persisted_actual_dimension=persisted_actual_dim,
                configured_dimension=embed_settings.dim,
            )
            signature_schema_version = backend.state.version

            # 3. The embedder/tokenizer must initialize for deep readiness.
            try:
                embedder = get_cached_embedder(embed_settings)
                embed_dim = embedder.dim
                signature_token_counter = embedder.token_counter()
            except Exception as exc:  # noqa: BLE001
                return not_ready("embedder_init_failed", detail=type(exc).__name__)
            signature_dimension = resolve_embedding_dimension(
                backend_dimension=backend_dim,
                persisted_actual_dimension=persisted_actual_dim,
                configured_dimension=embed_dim or embed_settings.dim,
            )

            # 4. Dimension mismatch remains a distinct actionable readiness reason.
            if stored_dim and embed_dim and int(stored_dim) != int(embed_dim):
                return not_ready(
                    "embedding_dim_mismatch",
                    stored_dim=int(stored_dim),
                    embed_dim=int(embed_dim),
                )

            active_signature = _index_signature_payload(
                index_settings=index_settings,
                embed_settings=embed_settings,
                embedding_dimension=signature_dimension,
                index_instance_id=signature_instance_id,
                structured_token_counter=signature_token_counter,
                signature_complete=receipt is not None or has_known_corpus,
                index_schema_version=signature_schema_version,
            )
            if receipt is None and has_known_corpus:
                return not_ready(
                    "index_build_receipt_missing",
                    active_index_signature=active_signature,
                )
            if (
                receipt is not None
                and receipt.index_signature.compatibility_fingerprint != active_signature["compatibility_fingerprint"]
            ):
                return not_ready(
                    "index_signature_mismatch",
                    active_index_signature=active_signature,
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
                    "auth_enabled": _auth_enabled(),
                    "engine_version": get_engine_version(),
                    "index_signature": authoritative_signature or active_signature,
                    "index_signature_source": ("persisted_build" if receipt is not None else "configured_partial"),
                },
            )
    except TimeoutError as exc:
        return not_ready("index_instance_id_unavailable", detail=type(exc).__name__)
    except Exception as exc:  # noqa: BLE001 — readiness surfaces a reason, never raises
        return not_ready("backend_load_failed", detail=type(exc).__name__)


@app.get("/api/config")
def config(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_token(authorization)
    # Config is inspection-only: never initialize/download a model or contact a remote
    # backend. The durable receipt is authoritative; otherwise return an explicitly partial
    # configured identity from local state/settings only.
    with locked_index_instance(_index_instance_id_path(), create=False) as lease:
        # Reset writes the embedding override under this lease, so settings are part of the
        # same immutable response snapshot as identity, receipt, and state.
        index_settings = _index_settings()
        embed_settings = _embed_settings()
        receipt_status = "missing"
        try:
            receipt = load_index_build_receipt(_index_build_receipt_path())
            if receipt is not None:
                receipt_status = "valid"
        except RuntimeError:
            # The explicit admin reset can recover this state. Config remains an offline,
            # non-mutating way to obtain the configured-partial fingerprint needed for its
            # optimistic precondition.
            receipt = None
            receipt_status = "invalid"
        if receipt is not None and receipt.index_signature.index_instance_id != lease.instance_id:
            raise HTTPException(status_code=500, detail="index_build_receipt_instance_mismatch")
        instance_id = lease.instance_id
        state = IndexState.load(_state_path())
        persisted_actual_dimension = (state.embed or {}).get("actual_dim")
        embedding_dimension = resolve_embedding_dimension(
            persisted_actual_dimension=persisted_actual_dimension,
            configured_dimension=embed_settings.dim,
        )
        configured_signature = _index_signature_payload(
            index_settings=index_settings,
            embed_settings=embed_settings,
            embedding_dimension=embedding_dimension,
            index_instance_id=instance_id,
            structured_token_counter=None,
            signature_complete=False,
            index_schema_version=state.version,
        )
        index_signature = receipt.index_signature.to_dict() if receipt is not None else configured_signature
        return {
            "index_path": str(_index_path()),
            "state_path": str(_state_path()),
            "index": {
                "backend": index_settings.backend,
                "top_k": index_settings.top_k,
            },
            "embed": {
                "provider": embed_settings.provider,
                "model": embed_settings.model,
                "hf_model": embed_settings.hf_model,
                "device": embed_settings.device,
            },
            "llm_model": os.getenv("SLIMX_LLM_MODEL", "fake:grounded"),
            "index_signature": index_signature,
            "index_signature_source": "persisted_build" if receipt is not None else "configured_partial",
            "configured_index_signature": configured_signature,
            "index_build_receipt_status": receipt_status,
        }


@app.post("/api/admin/embedding")
def set_embedding(payload: EmbeddingConfigRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Set the active embedding config and reset the index (guarded by the demo token).

    Switching the embedding model/device changes the vector space and resets the index.
    ControlRoom must then re-index its documents under the new embedding. The choice is
    persisted alongside the index, so it survives a restart.
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
    index_settings = _index_settings()
    if index_settings.backend not in {"local", "faiss"}:
        raise HTTPException(
            status_code=409,
            detail=(f"index reset is unsupported for backend {index_settings.backend!r}; reset its corpus externally"),
        )
    # Preflight every resource needed to describe the new build before touching the old
    # corpus. A missing HF model/dependency/tokenizer can never destroy a working index.
    try:
        embedder = get_cached_embedder(merged)
        token_counter = embedder.token_counter()
    except Exception as exc:  # noqa: BLE001 — surface only the safe exception type
        raise HTTPException(status_code=422, detail=f"embedder_preflight_failed: {type(exc).__name__}") from exc
    try:
        with _index_lock, locked_index_instance(_index_instance_id_path(), create=True) as lease:
            new_instance_id = _reset_index(
                index_settings,
                lease=lease,
                new_embed_settings=merged,
                persist_embed_override=True,
            )
    except IndexResetPartialFailure as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=f"index_reset_failed: {type(exc).__name__}") from exc
    return {
        "embed": {
            "provider": merged.provider,
            "model": merged.model,
            "hf_model": merged.hf_model,
            "dim": merged.dim,
            "device": merged.device,
        },
        "index_reset": True,
        "index_signature": _index_signature_payload(
            index_settings=index_settings,
            embed_settings=merged,
            embedding_dimension=resolve_embedding_dimension(configured_dimension=embedder.dim or merged.dim),
            index_instance_id=new_instance_id,
            structured_token_counter=token_counter,
            signature_complete=False,
        ),
        "index_signature_source": "configured_partial",
    }


@app.post("/api/admin/index/reset", response_model=IndexResetResponse)
def reset_index_endpoint(
    payload: IndexResetRequest,
    authorization: str | None = Header(default=None),
) -> IndexResetResponse:
    """Discard the configured local corpus without changing its embedding configuration.

    This is a deliberately narrow maintenance operation. It accepts no filesystem path or
    backend namespace, requires an exact confirmation literal, and compares the caller's
    last observed instance and fingerprint while holding the corpus identity lease.
    """
    _check_index_reset_token(authorization)
    index_settings = _index_settings()
    backend_name = (index_settings.backend or "local").lower().strip()
    if backend_name not in {"local", "faiss"}:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "index_reset_backend_unsupported",
                "backend": backend_name,
                "retryable": False,
                "owner_action": (
                    "Reset the remote corpus namespace with its owning backend, then reconfigure and reindex."
                ),
            },
        )

    # Reuse the active settings exactly. Preflight both the embedder and token counter before
    # entering the reset transaction; an unavailable model/tokenizer cannot destroy the corpus.
    embed_settings = _embed_settings()
    text_chunk_settings = _chunk_settings()
    file_chunk_settings = _structured_chunk_settings()
    try:
        embedder = get_cached_embedder(embed_settings)
        token_counter = embedder.token_counter()
    except Exception as exc:  # noqa: BLE001 — expose only the safe exception type
        raise HTTPException(
            status_code=422,
            detail={
                "code": "embedder_preflight_failed",
                "error_type": type(exc).__name__,
                "owner_action": "Restore the configured embedder/tokenizer before retrying the reset.",
            },
        ) from exc
    embedding_dimension = resolve_embedding_dimension(configured_dimension=embedder.dim or embed_settings.dim)

    try:
        with _index_lock, locked_index_instance(_index_instance_id_path(), create=False) as lease:
            # Embedding configuration is mutable through the sibling admin route. Refuse a
            # preflight snapshot that changed while this request waited for the reset lock.
            if (
                _index_settings() != index_settings
                or _embed_settings() != embed_settings
                or _chunk_settings() != text_chunk_settings
                or _structured_chunk_settings() != file_chunk_settings
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "index_reset_configuration_changed_retry",
                        "retryable": True,
                        "owner_action": "Refresh the current signature and retry with new preconditions.",
                    },
                )

            previous_instance_id = lease.instance_id
            receipt_untrusted = False
            try:
                previous_receipt = load_index_build_receipt(_index_build_receipt_path())
            except RuntimeError as exc:
                # A malformed receipt is one reason this recovery endpoint exists. It cannot be
                # treated as authoritative, so preserve a configured partial snapshot instead.
                logger.warning("Ignoring unreadable build receipt during explicit reset: %s", type(exc).__name__)
                previous_receipt = None
                receipt_untrusted = True

            if (
                previous_receipt is not None
                and previous_receipt.index_signature.index_instance_id != previous_instance_id
            ):
                # The receipt describes another corpus generation. Preserve it on disk until
                # the transaction commits, but never use its foreign identity for reset CAS.
                previous_receipt = None
                receipt_untrusted = True

            if previous_receipt is not None:
                previous_signature = previous_receipt.index_signature.to_dict()
                previous_signature_source: Literal["persisted_build", "configured_partial"] = "persisted_build"
                accepted_fingerprints = {str(previous_signature["compatibility_fingerprint"])}
            else:
                try:
                    previous_state = IndexState.load(_state_path())
                except RuntimeError as exc:
                    logger.warning("Ignoring unreadable index state during explicit reset: %s", type(exc).__name__)
                    previous_state = IndexState()
                previous_dimension = resolve_embedding_dimension(
                    persisted_actual_dimension=(previous_state.embed or {}).get("actual_dim"),
                    configured_dimension=embedding_dimension,
                )
                previous_configured_signature = _index_signature_payload(
                    index_settings=index_settings,
                    embed_settings=embed_settings,
                    embedding_dimension=previous_dimension,
                    index_instance_id=previous_instance_id,
                    # Match the offline partial snapshot from /api/config. Runtime token-
                    # counter evidence belongs to the new signature produced below.
                    structured_token_counter=None,
                    text_chunk_settings=text_chunk_settings,
                    file_chunk_settings=file_chunk_settings,
                    signature_complete=False,
                    index_schema_version=previous_state.version,
                )
                previous_signature = _index_signature_payload(
                    index_settings=index_settings,
                    embed_settings=embed_settings,
                    embedding_dimension=previous_dimension,
                    index_instance_id=previous_instance_id,
                    structured_token_counter=token_counter,
                    text_chunk_settings=text_chunk_settings,
                    file_chunk_settings=file_chunk_settings,
                    signature_complete=False,
                    index_schema_version=previous_state.version,
                )
                previous_signature_source = "configured_partial"
                accepted_fingerprints = {
                    str(previous_configured_signature["compatibility_fingerprint"]),
                    str(previous_signature["compatibility_fingerprint"]),
                }

            if payload.expected_index_instance_id != previous_instance_id:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "index_reset_precondition_failed",
                        "field": "index_instance_id",
                        "retryable": True,
                        "owner_action": "Refresh the current signature and review the changed corpus before retrying.",
                    },
                )
            if payload.expected_compatibility_fingerprint is None and not receipt_untrusted:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "index_reset_precondition_failed",
                        "field": "compatibility_fingerprint",
                        "retryable": True,
                        "owner_action": (
                            "Provide the current compatibility fingerprint; null is only valid "
                            "when no trustworthy receipt exists."
                        ),
                    },
                )
            if (
                payload.expected_compatibility_fingerprint is not None
                and payload.expected_compatibility_fingerprint not in accepted_fingerprints
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "index_reset_precondition_failed",
                        "field": "compatibility_fingerprint",
                        "retryable": True,
                        "owner_action": "Refresh the current signature and review the changed corpus before retrying.",
                    },
                )

            new_instance_id = _reset_index(
                index_settings,
                lease=lease,
                new_embed_settings=embed_settings,
                persist_embed_override=False,
            )
            new_signature = _index_signature_payload(
                index_settings=index_settings,
                embed_settings=embed_settings,
                embedding_dimension=embedding_dimension,
                index_instance_id=new_instance_id,
                structured_token_counter=token_counter,
                text_chunk_settings=text_chunk_settings,
                file_chunk_settings=file_chunk_settings,
                signature_complete=False,
            )
    except HTTPException:
        raise
    except IndexResetPartialFailure as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "index_reset_partial_failure",
                "error_type": type(exc).__name__,
                "active_identity_invalidated": True,
                "owner_action": "Inspect and restore the index volume before serving retrieval or indexing.",
            },
        ) from exc
    except UnsupportedIndexResetError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "index_reset_backend_unsupported",
                "backend": backend_name,
                "retryable": False,
                "owner_action": (
                    "Reset the remote corpus namespace with its owning backend, then reconfigure and reindex."
                ),
            },
        ) from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "index_reset_failed",
                "error_type": type(exc).__name__,
                "active_identity_invalidated": False,
                "owner_action": "Resolve the local index-volume error and retry with refreshed preconditions.",
            },
        ) from exc

    return IndexResetResponse(
        index_reset=True,
        previous_index_instance_id=previous_instance_id,
        previous_index_signature=previous_signature,
        previous_index_signature_source=previous_signature_source,
        new_index_instance_id=new_instance_id,
        index_signature=new_signature,
        index_signature_source="configured_partial",
        engine_version=get_engine_version(),
    )


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
    # Snapshot the corpus generation and every signature-shaping setting before off-lock
    # chunking/embedding. The final write lease rejects work that crossed a reset.
    with locked_index_instance(_index_instance_id_path(), create=False) as snapshot_lease:
        expected_instance_id = snapshot_lease.instance_id
        embed_settings = _embed_settings()
        index_settings = _index_settings()
        chunk_settings = _chunk_settings()
        structured_chunk_settings = _structured_chunk_settings()

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
    embedder = get_cached_embedder(embed_settings)
    token_counter = embedder.token_counter()
    with _embed_lock:
        items: list[EmbeddedChunk] = list(embed_chunks(iter(chunks), settings=embed_settings, embedder=embedder))

    with _index_lock, locked_index_instance(_index_instance_id_path(), create=False) as lease:
        if lease.instance_id != expected_instance_id:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "index_generation_changed_retry",
                    "retryable": True,
                    "owner_action": "Retry indexing against the current corpus generation.",
                },
            )
        if (
            _embed_settings() != embed_settings
            or _index_settings() != index_settings
            or _chunk_settings() != chunk_settings
            or _structured_chunk_settings() != structured_chunk_settings
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "index_configuration_changed_retry",
                    "retryable": True,
                    "owner_action": "Refresh readiness and retry indexing with the current configuration.",
                },
            )
        if lease.instance_id is None:
            lease.publish_new()
        assert lease.instance_id is not None
        previous_receipt = load_index_build_receipt(_index_build_receipt_path())
        idx = _current_backend()  # hot backend; load is amortized across posts
        persisted_actual_dimension = (idx.state.embed or {}).get("actual_dim")
        actual_dimension = resolve_embedding_dimension(
            emitted_dimension=(len(items[0].vector) if items else None),
            backend_dimension=idx.dim,
            persisted_actual_dimension=persisted_actual_dimension,
        )
        embedding_dimension = resolve_embedding_dimension(
            emitted_dimension=actual_dimension,
            configured_dimension=embedder.dim or embed_settings.dim,
        )
        candidate_signature = _index_signature_contract(
            index_settings=index_settings,
            embed_settings=embed_settings,
            embedding_dimension=embedding_dimension,
            index_instance_id=lease.instance_id,
            structured_token_counter=token_counter,
            text_chunk_settings=chunk_settings,
            file_chunk_settings=structured_chunk_settings,
            index_schema_version=idx.state.version,
        )
        if previous_receipt is None and (len(idx) > 0 or bool(idx.state.docs)):
            raise HTTPException(status_code=409, detail="index_build_receipt_missing_reindex_required")
        if previous_receipt is not None and (
            previous_receipt.index_signature.index_instance_id != lease.instance_id
            or previous_receipt.index_signature.compatibility_fingerprint
            != candidate_signature.compatibility_fingerprint
        ):
            raise HTTPException(status_code=409, detail="index_signature_mismatch_reindex_required")

        idx.delete_doc(doc_id)  # replace this document's chunks; no-op when new
        upserted = idx.upsert(items, skip_existing=False)
        idx.set_embed_config(embed_settings, dimension=actual_dimension)
        idx.save()
        _mark_index_written()  # our own write must not trigger a reload on the next read
        chunk_ids = [item.chunk_id for item in items]
        idx.commit_doc_state(doc_id, ch, chunk_ids)
        total = len(idx)
        index_schema_version = idx.state.version
        signature = _index_signature_contract(
            index_settings=index_settings,
            embed_settings=embed_settings,
            embedding_dimension=embedding_dimension,
            index_instance_id=lease.instance_id,
            structured_token_counter=token_counter,
            text_chunk_settings=chunk_settings,
            file_chunk_settings=structured_chunk_settings,
            index_schema_version=index_schema_version,
        )
        pipeline = {
            "ingest_mode": "text",
            "chunker": "recursive-character",
            "chunk_config_fingerprint": signature.text_chunk_config_fingerprint,
            "parser": None,
        }
        build_receipt = write_index_build_receipt(_index_build_receipt_path(), signature, document_pipeline=pipeline)

    model = embed_settings.hf_model if embed_settings.provider == "hf" else embed_settings.model
    index_signature = signature.to_dict()
    return {
        "status": "ready",
        "doc_id": doc_id,
        "rag_index_ref": f"slimx-rag:{doc_id}",
        "chunk_count": len(chunk_ids),
        "upserted": upserted,
        "total": total,
        "vector_backend": index_settings.backend,
        "embed": {"provider": embed_settings.provider, "model": model, "dim": embed_settings.dim},
        "index_signature": index_signature,
        "index_signature_source": "persisted_build",
        "index_build_receipt": build_receipt.to_dict(),
        "document_pipeline": pipeline,
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
    # Snapshot generation/settings before parsing and embedding outside the mutation lock.
    with locked_index_instance(_index_instance_id_path(), create=False) as snapshot_lease:
        expected_instance_id = snapshot_lease.instance_id
        embed_settings = _embed_settings()
        index_settings = _index_settings()
        text_chunk_settings = _chunk_settings()
        structured_chunk_settings = _structured_chunk_settings()

    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="empty file")
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"file exceeds {MAX_FILE_BYTES} bytes")

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
        # Surface the parser's failure-mode message (e.g. "requires the optional dependency
        # 'pypdf'", "Could not read PDF: <ErrType>") so the caller can act. These messages
        # describe the failure, never the document content, so they are safe to return.
        raise HTTPException(status_code=422, detail=f"parse_failed: {type(exc).__name__}: {exc}") from exc
    timings["parse_ms"] = int((time.perf_counter() - t0) * 1000)
    if parsed.element_count > MAX_ELEMENTS:
        raise HTTPException(status_code=413, detail=f"document has {parsed.element_count} elements; max {MAX_ELEMENTS}")

    t1 = time.perf_counter()
    token_counter = make_token_counter(embed_settings)
    chunks = chunk_parsed_document(parsed, settings=structured_chunk_settings, token_counter=token_counter)
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
        items: list[EmbeddedChunk] = list(embed_chunks(iter(docs), settings=embed_settings, embedder=embedder))
    timings["embed_ms"] = int((time.perf_counter() - t2) * 1000)

    t3 = time.perf_counter()
    with _index_lock, locked_index_instance(_index_instance_id_path(), create=False) as lease:
        if lease.instance_id != expected_instance_id:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "index_generation_changed_retry",
                    "retryable": True,
                    "owner_action": "Retry indexing against the current corpus generation.",
                },
            )
        if (
            _embed_settings() != embed_settings
            or _index_settings() != index_settings
            or _chunk_settings() != text_chunk_settings
            or _structured_chunk_settings() != structured_chunk_settings
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "index_configuration_changed_retry",
                    "retryable": True,
                    "owner_action": "Refresh readiness and retry indexing with the current configuration.",
                },
            )
        if lease.instance_id is None:
            lease.publish_new()
        assert lease.instance_id is not None
        previous_receipt = load_index_build_receipt(_index_build_receipt_path())
        idx = _current_backend()
        persisted_actual_dimension = (idx.state.embed or {}).get("actual_dim")
        actual_dimension = resolve_embedding_dimension(
            emitted_dimension=(len(items[0].vector) if items else None),
            backend_dimension=idx.dim,
            persisted_actual_dimension=persisted_actual_dimension,
        )
        embedding_dimension = resolve_embedding_dimension(
            emitted_dimension=actual_dimension,
            configured_dimension=embedder.dim or embed_settings.dim,
        )
        candidate_signature = _index_signature_contract(
            index_settings=index_settings,
            embed_settings=embed_settings,
            embedding_dimension=embedding_dimension,
            index_instance_id=lease.instance_id,
            structured_token_counter=token_counter,
            text_chunk_settings=text_chunk_settings,
            file_chunk_settings=structured_chunk_settings,
            index_schema_version=idx.state.version,
        )
        if previous_receipt is None and (len(idx) > 0 or bool(idx.state.docs)):
            raise HTTPException(status_code=409, detail="index_build_receipt_missing_reindex_required")
        if previous_receipt is not None and (
            previous_receipt.index_signature.index_instance_id != lease.instance_id
            or previous_receipt.index_signature.compatibility_fingerprint
            != candidate_signature.compatibility_fingerprint
        ):
            raise HTTPException(status_code=409, detail="index_signature_mismatch_reindex_required")

        idx.delete_doc(doc_id)
        upserted = idx.upsert(items, skip_existing=False)
        idx.set_embed_config(embed_settings, dimension=actual_dimension)
        idx.save()
        _mark_index_written()
        idx.commit_doc_state(doc_id, ch_hash, [it.chunk_id for it in items])
        total = len(idx)
        lexical_capable = bool(getattr(idx, "supports_inmemory_scope_filter", False))
        index_schema_version = idx.state.version
        signature = _index_signature_contract(
            index_settings=index_settings,
            embed_settings=embed_settings,
            embedding_dimension=embedding_dimension,
            index_instance_id=lease.instance_id,
            structured_token_counter=token_counter,
            text_chunk_settings=text_chunk_settings,
            file_chunk_settings=structured_chunk_settings,
            index_schema_version=index_schema_version,
        )
        parser_pipeline: dict[str, object] = {
            "name": parsed.parser_name,
            "version": parsed.parser_version,
            "extraction_backend": parsed.metadata.get("extraction_backend") or "builtin",
            "extraction_backend_version": (parsed.metadata.get("extraction_backend_version") or parsed.parser_version),
        }
        pipeline = {
            "ingest_mode": "file",
            "chunker": "structured-token",
            "chunk_config_fingerprint": signature.file_chunk_config_fingerprint,
            "parser": parser_pipeline,
            "source_type": parsed.source_type,
        }
        build_receipt = write_index_build_receipt(_index_build_receipt_path(), signature, document_pipeline=pipeline)
    timings["index_ms"] = int((time.perf_counter() - t3) * 1000)

    model = embed_settings.hf_model if embed_settings.provider == "hf" else embed_settings.model
    index_signature = signature.to_dict()
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
        "vector_backend": index_settings.backend,
        "lexical_retrieval": lexical_capable,
        "warnings": list(parsed.warnings),
        "timings_ms": timings,
        "index_signature": index_signature,
        "index_signature_source": "persisted_build",
        "index_build_receipt": build_receipt.to_dict(),
        "document_pipeline": pipeline,
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
                # Richer inspection fields (already stored on chunk metadata) so ControlRoom's
                # Document Reader can show structure/provenance, not just page/section.
                "section_path": md.get("section_path"),
                "parent_id": md.get("parent_id"),
                "page_type": md.get("page_type"),
                "token_count": md.get("token_count"),
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
