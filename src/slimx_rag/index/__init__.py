from __future__ import annotations

from pathlib import Path

from slimx_rag.settings import IndexSettings

from .base import IndexBackend
from .types import IndexState, SearchResult


def make_index_backend(
    index_path: Path,
    *,
    settings: IndexSettings | None = None,
    state_path: Path | None = None,
) -> IndexBackend:
    """Factory for index backends (plugin architecture).

    Backends are selected via settings.backend:
      - local (JSONL MVP)
      - faiss (local binary)
      - qdrant (remote)
      - pgvector (Postgres)
    """
    # TODO: using registry pattern would be cleaner, but this is straightforward enough for now
    st = settings or IndexSettings()
    backend = (st.backend or "local").lower().strip()

    if backend == "local":
        from .local import LocalJsonlIndexBackend
        return LocalJsonlIndexBackend(index_path, settings=st, state_path=state_path)

    if backend == "faiss":
        from .faiss_backend import FaissIndexBackend
        return FaissIndexBackend(index_path, settings=st, state_path=state_path)

    if backend == "qdrant":
        from .qdrant_backend import QdrantIndexBackend
        return QdrantIndexBackend(index_path, settings=st, state_path=state_path)

    if backend == "pgvector":
        from .pgvector_backend import PgVectorIndexBackend
        return PgVectorIndexBackend(index_path, settings=st, state_path=state_path)

    raise ValueError(f"Unknown index backend: {backend}")


__all__ = [
    "IndexBackend",
    "make_index_backend",
    "SearchResult",
    "IndexState",
]
