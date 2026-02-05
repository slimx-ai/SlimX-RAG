from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class IngestSettings:
    glob: str = "**/*.md"
    multithreading: bool = False
    show_progress: bool = False


@dataclass(frozen=True, slots=True)
class ChunkSettings:
    chunk_size: int = 800
    chunk_overlap: int = 120
    extended_metadata: bool = True
    separators: tuple[str, ...] = ("\n\n", "\n", " ", "") # Change Sequence to tuple to emphasize immutability


@dataclass(frozen=True, slots=True)
class EmbedSettings:
    """Embedding configuration.

    The default embedder is deterministic and offline ("hash") so the pipeline
    can run in CI without network credentials.

    Providers:
      - hash: local deterministic pseudo-embeddings (for tests/dev)
      - openai: OpenAI embeddings via langchain-openai (requires OPENAI_API_KEY)
      - hf: SentenceTransformers embeddings (requires sentence-transformers)
    """

    provider: str = "hash"  # hash | openai | hf

    # hash provider
    dim: int = 384

    # openai provider
    model: str = "text-embedding-3-small"

    # hf provider
    hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # batching + robustness
    batch_size: int = 32
    retries: int = 3
    retry_backoff_s: float = 1.0

    # text hygiene (helps determinism / avoids hidden differences)
    normalize_text: bool = True
    max_chars: int | None = None


@dataclass(frozen=True, slots=True)
class IndexingSettings:
    kb_dir: Path = Path("./knowledge-base")
    out_dir: Path = Path("./output")

    ingest: IngestSettings = field(default_factory=IngestSettings)
    chunk: ChunkSettings = field(default_factory=ChunkSettings)
    embed: EmbedSettings = field(default_factory=EmbedSettings)

    docs_filename: str = "docs.jsonl"
    chunks_filename: str = "chunks.jsonl"
    embeddings_filename: str = "embeddings.jsonl"
    index_filename: str = "index.jsonl"

    @property
    def docs_path(self) -> Path:
        return self.out_dir / self.docs_filename

    @property
    def chunks_path(self) -> Path:
        return self.out_dir / self.chunks_filename

    @property
    def embeddings_path(self) -> Path:
        return self.out_dir / self.embeddings_filename

    @property
    def index_path(self) -> Path:
        return self.out_dir / self.index_filename
    
    def validate(self) -> None:
        if self.chunk.chunk_overlap >= self.chunk.chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        if self.chunk.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")

        if self.embed.provider not in {"hash", "openai", "hf"}:
            raise ValueError("embed.provider must be one of: hash, openai, hf")

        if self.embed.provider == "hash" and self.embed.dim <= 0:
            raise ValueError("embed.dim must be > 0")
