from __future__ import annotations

from pathlib import Path

from slimx_rag.settings import IndexSettings

from .base import IndexBackend
from .signature import (
    BACKEND_NAMESPACE_VERSION,
    EMBEDDING_CONFIG_VERSION,
    INDEX_BUILD_RECEIPT_FILENAME,
    INDEX_BUILD_RECEIPT_VERSION,
    INDEX_INSTANCE_ID_FILENAME,
    INDEX_SHAPING_VERSION,
    INDEX_SIGNATURE_VERSION,
    PARSER_CONFIG_VERSION,
    STRUCTURED_CHUNK_CONFIG_VERSION,
    IndexBuildReceipt,
    IndexInstanceLease,
    IndexSignature,
    backend_corpus_namespace,
    build_index_build_receipt,
    build_index_signature,
    delete_index_instance_id,
    get_or_create_index_instance_id,
    load_index_build_receipt,
    locked_index_instance,
    read_index_instance_id,
    resolve_embedding_dimension,
    write_index_build_receipt,
)
from .types import INDEX_SCHEMA_VERSION, IndexState, SearchResult


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
    "INDEX_SCHEMA_VERSION",
    "INDEX_SIGNATURE_VERSION",
    "EMBEDDING_CONFIG_VERSION",
    "PARSER_CONFIG_VERSION",
    "STRUCTURED_CHUNK_CONFIG_VERSION",
    "BACKEND_NAMESPACE_VERSION",
    "INDEX_SHAPING_VERSION",
    "IndexSignature",
    "build_index_signature",
    "INDEX_INSTANCE_ID_FILENAME",
    "INDEX_BUILD_RECEIPT_FILENAME",
    "INDEX_BUILD_RECEIPT_VERSION",
    "IndexBuildReceipt",
    "IndexInstanceLease",
    "get_or_create_index_instance_id",
    "read_index_instance_id",
    "locked_index_instance",
    "build_index_build_receipt",
    "load_index_build_receipt",
    "write_index_build_receipt",
    "backend_corpus_namespace",
    "delete_index_instance_id",
    "resolve_embedding_dimension",
]
