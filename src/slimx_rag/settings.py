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
class IndexingSettings:
    kb_dir: Path = Path("./knowledge-base")
    out_dir: Path = Path("./output")

    ingest: IngestSettings = field(default_factory=IngestSettings)
    chunk: ChunkSettings = field(default_factory=ChunkSettings)

    docs_filename: str = "docs.jsonl"
    chunks_filename: str = "chunks.jsonl"

    @property
    def docs_path(self) -> Path:
        return self.out_dir / self.docs_filename

    @property
    def chunks_path(self) -> Path:
        return self.out_dir / self.chunks_filename
    
    def validate(self) -> None:
        if self.chunk.chunk_overlap >= self.chunk.chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        if self.chunk.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
