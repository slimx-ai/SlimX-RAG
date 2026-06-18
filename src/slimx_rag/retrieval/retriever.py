from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slimx_rag.embed import make_embedder
from slimx_rag.index import IndexBackend, make_index_backend
from slimx_rag.index.types import SearchResult
from slimx_rag.settings import EmbedSettings, IndexSettings


class ScopeNotSupportedError(ValueError):
    """Raised when workspace/document scoping is requested on a backend that cannot serve it.

    Scoping is applied by over-fetching the whole index and filtering chunk metadata in
    Python, which is only correct for in-memory backends that report len() accurately
    (the local backend). Remote/ANN backends must push the filter down natively first.
    """


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, object]
    citation: str

    @classmethod
    def from_search_result(cls, result: SearchResult) -> RetrievedChunk:
        metadata = dict(result.metadata or {})
        source = (
            metadata.get("parent_kb_relpath")
            or metadata.get("kb_relpath")
            or metadata.get("source")
            or "unknown"
        )
        chunk_index = metadata.get("chunk_index", metadata.get("page", "?"))
        return cls(
            chunk_id=result.chunk_id,
            score=float(result.score),
            text=result.text,
            metadata=metadata,
            citation=f"[{source}:{chunk_index}]",
        )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    embed: dict[str, Any]
    elapsed_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "embed": self.embed,
            "elapsed_ms": self.elapsed_ms,
        }


def retrieve(
    question: str,
    *,
    index_path: Path,
    embed_settings: EmbedSettings,
    index_settings: IndexSettings,
    state_path: Path | None = None,
    top_k: int | None = None,
    workspace_id: str | None = None,
    document_ids: list[str] | None = None,
    backend: IndexBackend | None = None,
) -> RetrievalResult:
    started = time.perf_counter()
    # A caller (e.g. the server) may pass a pre-loaded backend to avoid re-reading the
    # index from disk on every request; otherwise construct + load one as before.
    if backend is not None:
        idx = backend
    else:
        idx = make_index_backend(index_path, settings=index_settings, state_path=state_path)
        idx.load()

    scope_ws = str(workspace_id) if workspace_id else None
    scope_docs = {str(d) for d in document_ids} if document_ids else None
    if (scope_ws or scope_docs) and not getattr(idx, "supports_inmemory_scope_filter", False):
        raise ScopeNotSupportedError(
            f"workspace_id/document_ids scoping is not supported by the "
            f"{index_settings.backend!r} backend; use the 'local' backend "
            f"(backend-native filtering for remote backends is not yet implemented)."
        )

    embedder = make_embedder(embed_settings)
    qvec = embedder.embed_texts([question])[0]
    k = top_k or index_settings.top_k

    if scope_ws or scope_docs:
        # Scope to one workspace (and optionally specific documents) by filtering chunk
        # metadata. Over-fetch the whole index so the post-filter is exact; this is safe
        # only because the guard above restricts scoping to backends that report len()
        # accurately (supports_inmemory_scope_filter).
        fetch_k = len(idx) or k
        candidates = idx.query(list(map(float, qvec)), top_k=fetch_k)

        def _in_scope(md: dict[str, object]) -> bool:
            if scope_ws is not None and str(md.get("workspace_id")) != scope_ws:
                return False
            if scope_docs is not None and str(md.get("document_id")) not in scope_docs:
                return False
            return True

        raw_results = [r for r in candidates if _in_scope(r.metadata or {})][:k]
    else:
        raw_results = idx.query(list(map(float, qvec)), top_k=k)
    chunks = [RetrievedChunk.from_search_result(result) for result in raw_results]
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return RetrievalResult(
        query=question,
        chunks=chunks,
        embed={
            "provider": embed_settings.provider,
            "model": embed_settings.model,
            "hf_model": embed_settings.hf_model,
            "dim": embed_settings.dim,
        },
        elapsed_ms=elapsed_ms,
    )
