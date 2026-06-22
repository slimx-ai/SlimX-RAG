from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

# settings.py (top-level)
EMBED_PROVIDERS: tuple[str, ...] = ("hash", "openai", "hf")
EXTERNAL_EMBED_PROVIDERS: tuple[str, ...] = tuple(p for p in EMBED_PROVIDERS if p != "hash")
INDEX_BACKENDS: tuple[str, ...] = ("local", "faiss", "qdrant", "pgvector")


@dataclass(frozen=True, slots=True)
class IngestSettings:
    glob: str = "**/*.md"
    multithreading: bool = False
    show_progress: bool = False
    doc_type_mode: str = "subdir"
    doc_type_depth: int = 1

    def validate(self) -> None:
        if not self.glob or not self.glob.strip():
            raise ValueError("ingest.glob must be a non-empty glob pattern")
        if self.doc_type_mode not in {"subdir", "none"}:
            raise ValueError("ingest.doc_type_mode must be 'subdir' or 'none'")
        if self.doc_type_depth <= 0:
            raise ValueError("ingest.doc_type_depth must be > 0")


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
class StructuredChunkSettings:
    """Token-based, structure-aware chunking config (page/section parents).

    Sizes are in embedding-tokenizer tokens, not characters. ``max_tokens`` is the hard
    cap; at runtime the chunker clamps it to the embedding model's max sequence length so
    a chunk can never silently exceed the model and be truncated. A small sliding overlap
    is applied ONLY when an oversized, indivisible element must be force-split — normal
    semantic chunks use no overlap.
    """

    target_tokens: int = 256
    max_tokens: int = 512
    force_split_overlap_tokens: int = 32
    include_identity_prefix: bool = True
    fallback: str = "recursive"  # strategy for unstructured plain text

    def validate(self) -> None:
        if self.target_tokens <= 0:
            raise ValueError("structured_chunk.target_tokens must be > 0")
        if self.max_tokens < self.target_tokens:
            raise ValueError("structured_chunk.max_tokens must be >= target_tokens")
        if not 0 <= self.force_split_overlap_tokens < self.target_tokens:
            raise ValueError(
                "structured_chunk.force_split_overlap_tokens must be in [0, target_tokens)"
            )
        if self.fallback not in {"recursive", "none"}:
            raise ValueError("structured_chunk.fallback must be 'recursive' or 'none'")


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """Multi-stage hybrid retrieval configuration (dense + lexical + RRF + grouping).

    Candidate counts are starting points to be tuned from measured results. Parent
    grouping is mandatory; reranking is off by default (enable only on a measured gain).
    """

    dense_candidates: int = 30
    lexical_candidates: int = 30
    final_parents: int = 6
    max_children_per_parent: int = 2
    rrf_k: int = 60
    exact_match_boost: float = 0.5
    timeline_penalty: float = 0.5  # factor applied to timeline pages for factual intents
    enable_lexical: bool = True
    enable_rerank: bool = False

    def validate(self) -> None:
        for name, value in (
            ("dense_candidates", self.dense_candidates),
            ("lexical_candidates", self.lexical_candidates),
            ("final_parents", self.final_parents),
            ("max_children_per_parent", self.max_children_per_parent),
            ("rrf_k", self.rrf_k),
        ):
            if value <= 0:
                raise ValueError(f"retrieval.{name} must be > 0")
        if not 0.0 <= self.timeline_penalty <= 1.0:
            raise ValueError("retrieval.timeline_penalty must be in [0, 1]")
        if self.exact_match_boost < 0:
            raise ValueError("retrieval.exact_match_boost must be >= 0")


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
    # Torch device for the local `hf` embedder (e.g. "cpu", "cuda", "cuda:0", "mps").
    # None lets SentenceTransformers auto-select. Ignored by `hash`/`openai` so the
    # deterministic offline default never depends on hardware.
    device: str | None = None
    # Vector-space knobs for the local `hf` embedder. These change the produced vectors,
    # so they are part of the embedder cache key (see embed.get_cached_embedder).
    # normalize_embeddings produces unit vectors (cosine == dot); query/document prefixes
    # support asymmetric models (e.g. E5/BGE "query:" / "passage:"); revision pins weights.
    normalize_embeddings: bool = True
    query_prefix: str = ""
    document_prefix: str = ""
    revision: str | None = None

    def validate(self) -> None:
        if self.provider not in set(EMBED_PROVIDERS):
            raise ValueError(f"embed.provider must be one of: {', '.join(EMBED_PROVIDERS)}")
        if self.device is not None and not self.device.strip():
            raise ValueError("embed.device must be a non-empty device string when set")
        if self.revision is not None and not self.revision.strip():
            raise ValueError("embed.revision must be a non-empty string when set")
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
    embeddings_filename: str = "embeddings.jsonl"
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
    def embeddings_path(self) -> Path:
        return self.out_dir / self.embeddings_filename

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
