from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True, slots=True)
class IngestSettings:
    glob: str = "**/*.md"
    multithreading: bool = False
    show_progress: bool = False


@dataclass(frozen=True, slots=True)
class ChunkSettings:
    chunk_size: int = 800
    chunk_overlap: int = 120
    separators: Sequence[str] = ("\n\n", "\n", " ", "")


@dataclass(frozen=True, slots=True)
class EmbedSettings:
    # Providers: "hash" (deterministic testing), "openai", "hf" (SentenceTransformers)
    provider: str = "hash"
    model: str = "text-embedding-3-small"  # used by provider=openai
    hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # used by provider=hf
    dim: int = 384  # used by provider=hash and as an optional validation target for others
    batch_size: int = 32
    max_chars: Optional[int] = 6000  # safety guard for providers with limits; None disables
    normalize_text: bool = True
    retries: int = 5
    retry_backoff_s: float = 1.0


@dataclass(frozen=True, slots=True)
class IndexSettings:
    # Backend plugin: "local" (JSONL), "faiss", "qdrant", "pgvector"
    backend: str = "local"
    # Backend-specific configuration. Keep secrets in env vars rather than here.
    backend_config: dict[str, object] = field(default_factory=dict)

    top_k: int = 5
    # If set, only these metadata keys are persisted into the index (reduces size).
    metadata_whitelist: Optional[Sequence[str]] = None
    # If True, also store embedding config into state file for query consistency.
    write_state: bool = True
    state_filename: str = "index_state.json"


@dataclass(frozen=True, slots=True)
class IndexingSettings:

    kb_dir: Path = Path("knowledge-base")
    out_dir: Path = Path("output")

    docs_filename: str = "docs.jsonl"
    chunks_filename: str = "chunks.jsonl"
    index_filename: str = "index.jsonl"

    ingest: IngestSettings = field(default_factory=IngestSettings)
    chunk: ChunkSettings = field(default_factory=ChunkSettings)
    embed: EmbedSettings = field(default_factory=EmbedSettings)
    index: IndexSettings = field(default_factory=IndexSettings)

    @property
    def docs_path(self) -> Path:
        return self.out_dir / self.docs_filename

    @property
    def chunks_path(self) -> Path:
        return self.out_dir / self.chunks_filename

    @property
    def index_path(self) -> Path:
        return self.out_dir / self.index_filename

    @property
    def index_state_path(self) -> Path:
        return self.out_dir / self.index.state_filename

    def validate(self) -> None:
        # chunk
        if self.chunk.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk.chunk_overlap >= self.chunk.chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")

        # embed
        if self.embed.batch_size <= 0:
            raise ValueError("embed.batch_size must be > 0")
        if self.embed.dim <= 0:
            raise ValueError("embed.dim must be > 0")
        if self.embed.provider not in {"hash", "openai", "hf"}:
            raise ValueError("embed.provider must be one of: hash, openai, hf")
        if self.embed.retries < 1:
            raise ValueError("embed.retries must be >= 1")
        if self.embed.retry_backoff_s <= 0:
            raise ValueError("embed.retry_backoff_s must be > 0")

        # index
        if self.index.top_k <= 0:
            raise ValueError("index.top_k must be > 0")

        if self.index.backend not in {"local", "faiss", "qdrant", "pgvector"}:
            raise ValueError('index.backend must be one of: local, faiss, qdrant, pgvector')
