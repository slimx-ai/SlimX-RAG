from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from slimx_rag.embed import EmbeddedChunk
from slimx_rag.settings import EmbedSettings, IndexSettings

from .types import IndexState, SearchResult


def config_int(config: dict[str, object] | None, key: str, default: int) -> int:
    """Read an integer from a backend_config dict, with a clear error for bad values."""
    raw = (config or {}).get(key, default)
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise ValueError(f"backend_config[{key!r}] must be an integer, got {raw!r}")
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"backend_config[{key!r}] must be an integer, got {raw!r}") from e


class IndexBackend(ABC):
    """Backend-agnostic index interface.

    Implementations can be local (JSONL), local-binary (FAISS), or remote (Qdrant / pgvector).
    The pipeline (CLI + future library users) should depend only on this interface.
    """

    def __init__(
        self,
        index_path: Path,
        *,
        settings: IndexSettings | None = None,
        state_path: Path | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.settings = settings or IndexSettings()
        # Canonical state path: prefer explicit path; else place next to index output directory.
        self.state_path = Path(state_path) if state_path else (self.index_path.parent / self.settings.state_filename)
        self.state: IndexState = IndexState.load(self.state_path)
        self._dim: int | None = None

    def __len__(self) -> int:
        """Optional: some backends can report number of chunks (best-effort)."""
        return 0

    @property
    def dim(self) -> int | None:
        return self._dim

    @abstractmethod
    def load(self) -> None:
        """Load backend resources into memory (if applicable)."""

    @abstractmethod
    def save(self) -> None:
        """Persist backend resources (if applicable). Remote backends may treat this as a no-op."""

    @abstractmethod
    def upsert(self, items: Iterable[EmbeddedChunk], *, skip_existing: bool = True) -> int:
        """Upsert embedded chunks. Returns number of newly written chunks."""

    @abstractmethod
    def delete(self, chunk_ids: Iterable[str]) -> int:
        """Delete chunks by chunk_id. Returns number of deleted chunks (best-effort)."""

    @abstractmethod
    def query(self, query_vector: list[float], *, top_k: int | None = None) -> list[SearchResult]:
        """Return top-k most similar chunks."""

    def set_embed_config(self, embed: EmbedSettings) -> None:
        """Persist embedding configuration used to produce vectors.

        This intentionally does not set backend dimension. The backend dimension
        is the actual stored vector dimension and should be inferred from vectors
        or explicitly constrained by backend_config['dim'] where a backend needs
        to create remote storage before upsert.
        """
        self.state.embed = {
            "provider": embed.provider,
            "model": embed.model,
            "hf_model": embed.hf_model,
            "dim": embed.dim,
            "batch_size": embed.batch_size,
            "retries": embed.retries,
            "retry_backoff_s": embed.retry_backoff_s,
            "normalize_text": embed.normalize_text,
            "max_chars": embed.max_chars,
        }

    def _apply_metadata_whitelist(self, md: dict[str, object]) -> dict[str, object]:
        """Apply the shared metadata whitelist contract for all backends."""
        wl = self.settings.metadata_whitelist
        if not wl:
            return dict(md)
        keep = {str(k) for k in wl}
        return {k: v for k, v in dict(md).items() if k in keep}

    @staticmethod
    def _sort_results(results: Iterable[SearchResult], *, top_k: int | None = None) -> list[SearchResult]:
        """Deterministically order results by score descending, then chunk_id ascending."""
        ordered = sorted(results, key=lambda r: (-float(r.score), str(r.chunk_id)))
        return ordered[:top_k] if top_k is not None else ordered

    def _save_state_if_enabled(self) -> None:
        if self.settings.write_state:
            self.state.save(self.state_path)

    def apply_incremental_plan(self, *, current_docs: dict[str, tuple[str, list[str]]]) -> int:
        """Delete stale chunks based on doc_id/content_hash comparison.

        current_docs: doc_id -> (content_hash, [chunk_ids])
        Returns number of deleted chunks.
        """
        deleted = 0
        previous_doc_ids = set(self.state.docs.keys())
        current_doc_ids = set(current_docs.keys())

        # Deleted docs: present in state, absent now
        for doc_id in (previous_doc_ids - current_doc_ids):
            old = self.state.docs.get(doc_id) or {}
            deleted += self.delete(old.get("chunk_ids", []) or [])
            self.state.docs.pop(doc_id, None)

        # Changed docs: content_hash differs
        for doc_id in current_doc_ids:
            new_hash, new_chunk_ids = current_docs[doc_id]
            old = self.state.docs.get(doc_id) or {}
            if old and str(old.get("content_hash")) != str(new_hash):
                deleted += self.delete(old.get("chunk_ids", []) or [])

            # Update state to reflect current doc version + chunk ids (even before upsert).
            # (If you need stronger crash-consistency, update state after a successful save.)
            self.state.docs[doc_id] = {"content_hash": new_hash, "chunk_ids": list(new_chunk_ids)}

        return deleted
