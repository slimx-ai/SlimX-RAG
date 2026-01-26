from __future__ import annotations

import hashlib
import time
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence

from langchain_core.documents import Document

from slimx_rag.settings import EmbedSettings


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk_id: str
    vector: List[float]
    text: str
    metadata: dict[str, object]


class Embedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts into vectors."""


class HashEmbedder(Embedder):
    """Deterministic local embedder (no network). Good for testing pipelines.

    Turns text into a pseudo-vector by hashing. Not semantically meaningful,
    but stable and fast.
    """

    def __init__(self, dim: int = 384):
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self.dim = dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            vec: List[float] = []
            seed = (t or "").encode("utf-8", errors="ignore")
            counter = 0
            while len(vec) < self.dim:
                h = hashlib.blake2b(digest_size=64)
                h.update(seed)
                h.update(b"\n")
                h.update(str(counter).encode("utf-8"))
                digest = h.digest()  # 64 bytes
                for b in digest:
                    if len(vec) >= self.dim:
                        break
                    vec.append((b / 255.0) * 2.0 - 1.0)
                counter += 1
            out.append(vec)
        return out


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings via langchain-openai (optional dependency)."""

    def __init__(self, model: str):
        try:
            from langchain_openai import OpenAIEmbeddings  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "OpenAIEmbedder requires optional dependency 'langchain-openai'. "
                "Install extras and set OPENAI_API_KEY."
            ) from e
        self._emb = OpenAIEmbeddings(model=model)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._emb.embed_documents(texts)


class HuggingFaceEmbedder(Embedder):
    """HuggingFace/SentenceTransformers embeddings (optional dependency)."""

    def __init__(self, model: str):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "HuggingFaceEmbedder requires optional dependency 'sentence-transformers'. "
                "Install extras (e.g. `uv sync --extra hf`)."
            ) from e
        self._model = SentenceTransformer(model)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # returns numpy array; convert to list of lists
        embs = self._model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
        return [list(map(float, row)) for row in embs]


def make_embedder(settings: EmbedSettings) -> Embedder:
    if settings.provider == "hash":
        return HashEmbedder(dim=settings.dim)
    if settings.provider == "openai":
        return OpenAIEmbedder(model=settings.model)
    if settings.provider == "hf":
        return HuggingFaceEmbedder(model=settings.hf_model)
    raise ValueError(f"Unknown embed provider: {settings.provider}")


def _normalize_text(text: str, *, max_chars: Optional[int], normalize: bool) -> str:
    t = text or ""
    if normalize:
        # NFC helps stabilize unicode forms
        t = unicodedata.normalize("NFC", t)
        # normalize newlines
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        # strip trailing whitespace (keeps semantics mostly intact)
        t = "\n".join(line.rstrip() for line in t.split("\n"))
    if max_chars is not None and max_chars > 0 and len(t) > max_chars:
        t = t[:max_chars]
    return t


def _validate_vectors(
    vectors: List[List[float]],
    *,
    expected_dim: Optional[int],
    count: int,
) -> None:
    if len(vectors) != count:
        raise RuntimeError(f"Embedding provider returned {len(vectors)} vectors for {count} texts")
    dim0: Optional[int] = None
    for v in vectors:
        if dim0 is None:
            dim0 = len(v)
        elif len(v) != dim0:
            raise RuntimeError("Embedding provider returned inconsistent vector dimensions within batch")
    if expected_dim is not None and expected_dim > 0 and dim0 is not None and dim0 != expected_dim:
        raise RuntimeError(f"Embedding dim mismatch: expected {expected_dim}, got {dim0}")


def embed_chunks(
    chunks: Iterable[Document],
    *,
    settings: EmbedSettings,
    batch_size: Optional[int] = None,
) -> Iterator[EmbeddedChunk]:
    """Embed chunk Documents into vectors.

    Requires each chunk to have metadata['chunk_id'].

    Production-oriented choices:
      - batch embedding (throughput)
      - retry with exponential backoff (network providers)
      - optional text normalization + max length policy (limits)
      - validate vector count + dimensions
    """
    embedder = make_embedder(settings)
    bs = batch_size or settings.batch_size

    batch_docs: List[Document] = []
    batch_texts: List[str] = []

    def flush() -> Iterator[EmbeddedChunk]:
        if not batch_docs:
            return iter(())
        # Normalize text upfront so embedding input is stable and safe.
        texts = [
            _normalize_text(t, max_chars=settings.max_chars, normalize=settings.normalize_text)
            for t in batch_texts
        ]

        last_err: Optional[Exception] = None
        for attempt in range(settings.retries):
            try:
                vectors = embedder.embed_texts(texts)
                _validate_vectors(vectors, expected_dim=(settings.dim if settings.provider == "hash" else None), count=len(texts))
                for d, v, t_in in zip(batch_docs, vectors, texts):
                    md = dict(d.metadata)
                    cid = str(md.get("chunk_id") or "")
                    if not cid:
                        raise ValueError("Chunk is missing metadata['chunk_id']")
                    yield EmbeddedChunk(
                        chunk_id=cid,
                        vector=[float(x) for x in v],
                        text=t_in,
                        metadata=md,
                    )
                return iter(())
            except Exception as e:  # pragma: no cover (depends on provider/runtime)
                last_err = e
                # Simple exponential backoff
                if attempt < settings.retries - 1:
                    time.sleep(settings.retry_backoff_s * (2 ** attempt))
                else:
                    raise RuntimeError(f"Embedding failed after {settings.retries} attempts") from last_err
        return iter(())

    for d in chunks:
        batch_docs.append(d)
        batch_texts.append(d.page_content or "")
        if len(batch_docs) >= bs:
            yield from flush()
            batch_docs.clear()
            batch_texts.clear()

    yield from flush()
