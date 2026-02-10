from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence 

# settings.py (top-level)
EMBED_PROVIDERS: tuple[str, ...] = ("hash", "openai", "hf")
EXTERNAL_EMBED_PROVIDERS: tuple[str, ...] = tuple(p for p in EMBED_PROVIDERS if p != "hash")
INDEX_BACKENDS: tuple[str, ...] = ("local", "faiss", "qdrant", "pgvector")


@dataclass(frozen=True, slots=True)
class IngestSettings:
    glob: str = "**/*.md"
    multithreading: bool = False
    show_progress: bool = False

    def validate(self) -> None:
        if not self.glob or not self.glob.strip():
            raise ValueError("ingest.glob must be a non-empty glob pattern")


@dataclass(frozen=True, slots=True)
class ChunkSettings:
    chunk_size: int = 800
    chunk_overlap: int = 120
    extended_metadata: bool = True
    separators: tuple[str, ...] = ("\n\n", "\n", " ", "")

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk.chunk_size must be > 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk.chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk.chunk_overlap must be < chunk.chunk_size")
        if not self.separators:
            raise ValueError("chunk.separators must not be empty")


@dataclass(frozen=True, slots=True)
class EmbedSettings:
    provider: str = "hash"  # hash | openai | hf
    dim: int = 384
    model: str = "text-embedding-3-small"
    hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    retries: int = 3
    retry_backoff_s: float = 1.0
    normalize_text: bool = True
    max_chars: int | None = None

    def validate(self) -> None:
        if self.provider not in set(EMBED_PROVIDERS):
            raise ValueError(f"embed.provider must be one of: {', '.join(EMBED_PROVIDERS)}")
        if self.batch_size <= 0:
            raise ValueError("embed.batch_size must be > 0")
        if self.retries < 1:
            raise ValueError("embed.retries must be >= 1")
        if self.retry_backoff_s <= 0:
            raise ValueError("embed.retry_backoff_s must be > 0")

        if self.provider == "hash" and self.dim <= 0:
            raise ValueError("embed.dim must be > 0 for provider='hash'")
        if self.provider == "openai" and not self.model.strip():
            raise ValueError(f"embed.model must be non-empty for provider={EXTERNAL_EMBED_PROVIDERS}")

        if self.max_chars is not None and self.max_chars <= 0:
            raise ValueError("embed.max_chars must be > 0 when set")


@dataclass(frozen=True, slots=True)
class IndexSettings:
    backend: str = "local"
    backend_config: dict[str, object] = field(default_factory=dict)
    top_k: int = 5
    metadata_whitelist: Sequence[str] | None = None
    write_state: bool = True
    state_filename: str = "index_state.json"

    def validate(self) -> None:
        if self.backend not in set(INDEX_BACKENDS):
            raise ValueError(f"index.backend must be one of: {', '.join(INDEX_BACKENDS)}")
        if self.top_k <= 0:
            raise ValueError("index.top_k must be > 0")
        if self.write_state and not self.state_filename.strip():
            raise ValueError("index.state_filename must be non-empty when write_state=True")

        if self.metadata_whitelist is not None:
            if any((not k) or (not str(k).strip()) for k in self.metadata_whitelist):
                raise ValueError("index.metadata_whitelist must contain only non-empty strings")


@dataclass(frozen=True, slots=True)
class IndexingPipelineSettings:

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
        # local / intrinsic checks
        self.ingest.validate()
        self.chunk.validate()
        self.embed.validate()
        self.index.validate()

        # context checks (belong here)
        if not self.kb_dir.exists() or not self.kb_dir.is_dir():
            raise ValueError(f"kb_dir must exist and be a directory: {self.kb_dir}")
        
        if self.out_dir.exists() and not self.out_dir.is_dir():
            raise ValueError(f"out_dir must be a directory: {self.out_dir}")


        # optional: filename sanity
        for name in (self.docs_filename, self.chunks_filename, self.index_filename):
            if not name.strip():
                raise ValueError("output filenames must be non-empty")
