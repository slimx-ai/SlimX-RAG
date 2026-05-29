from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from slimx_rag.embed import make_embedder
from slimx_rag.index import make_index_backend
from slimx_rag.index.types import SearchResult
from slimx_rag.settings import EmbedSettings, IndexSettings


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, object]
    citation: str

    @classmethod
    def from_search_result(cls, result: SearchResult) -> "RetrievedChunk":
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
    state_path: Optional[Path] = None,
    top_k: Optional[int] = None,
) -> RetrievalResult:
    started = time.perf_counter()
    idx = make_index_backend(index_path, settings=index_settings, state_path=state_path)
    idx.load()

    embedder = make_embedder(embed_settings)
    qvec = embedder.embed_texts([question])[0]
    raw_results = idx.query(list(map(float, qvec)), top_k=top_k or index_settings.top_k)
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
