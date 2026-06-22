"""Embedding providers, a model cache, and a tokenizer-backed token counter.

Design notes:

- ``embed_documents`` (passages) and ``embed_query`` (questions) are separated so
  asymmetric models can apply the right prompt/prefix; symmetric models share one path.
- The loaded model is cached by ``get_cached_embedder`` keyed on *every* setting that
  changes the vector representation, so service mode never reconstructs a
  ``SentenceTransformer`` per request.
- ``Embedder.token_counter()`` returns a counter aligned with the embedding tokenizer
  (the chunker uses it to bound chunks so vectors are never silently truncated).
- ``embed_texts`` is retained (it is the document path) for backward compatibility.
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from langchain_core.documents import Document

from slimx_rag.chunk.tokenizer import HeuristicTokenCounter, TokenCounter
from slimx_rag.settings import EmbedSettings
from slimx_rag.utils.commons import _normalize_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk_id: str
    vector: list[float]
    text: str
    metadata: dict[str, object]


class EmbeddingTokenCounter:
    """Token counter backed by a Hugging Face tokenizer (the embedding model's own)."""

    name = "embedding"

    def __init__(self, tokenizer: object, *, max_tokens: int = 256) -> None:
        self._tok = tokenizer
        self.max_tokens = max_tokens

    def count(self, text: str) -> int:
        text = text or ""
        encode = getattr(self._tok, "encode", None)
        if callable(encode):
            try:
                return len(encode(text))
            except Exception:  # noqa: BLE001 — fall back to the call form below
                pass
        try:
            out = self._tok(text)  # type: ignore[operator]
            ids = out["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return len(ids)
        except Exception:  # noqa: BLE001 — degenerate tokenizer; estimate by words
            return len(text.split())


class Embedder(ABC):
    """Provider-neutral embedder. Subclasses implement at least ``embed_texts``."""

    name: str = "embedder"

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (the document path by default)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing. Defaults to ``embed_texts``."""
        return self.embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed one query. Symmetric models reuse the document path."""
        return self.embed_documents([query])[0]

    @property
    def dim(self) -> int | None:
        return None

    @property
    def max_seq_length(self) -> int | None:
        return None

    @property
    def revision(self) -> str | None:
        return None

    def token_counter(self) -> TokenCounter:
        """A token counter aligned with this embedder (heuristic unless overridden)."""
        return HeuristicTokenCounter(max_tokens=self.max_seq_length or 256)


class HashEmbedder(Embedder):
    """Deterministic, offline pseudo-embedder (no network). Not semantically meaningful."""

    name = "hash"

    def __init__(self, dim: int = 384):
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self._dim = dim

    @property
    def dim(self) -> int | None:
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec: list[float] = []
            seed = (t or "").encode("utf-8", errors="ignore")
            counter = 0
            while len(vec) < self._dim:
                h = hashlib.blake2b(digest_size=64)
                h.update(seed)
                h.update(b"\n")
                h.update(str(counter).encode("utf-8"))
                for b in h.digest():
                    if len(vec) >= self._dim:
                        break
                    vec.append((b / 255.0) * 2.0 - 1.0)
                counter += 1
            out.append(vec)
        return out


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings via langchain-openai (optional dependency)."""

    name = "openai"

    def __init__(self, model: str):
        try:
            from langchain_openai import OpenAIEmbeddings  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "OpenAIEmbedder requires optional dependency 'langchain-openai'. "
                "Install extras and set OPENAI_API_KEY."
            ) from e
        self._emb = OpenAIEmbeddings(model=model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._emb.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        return self._emb.embed_query(query)


class HuggingFaceEmbedder(Embedder):
    """HuggingFace/SentenceTransformers embeddings (optional dependency).

    Uses ``encode_document``/``encode_query`` when the installed SentenceTransformers
    version provides them; otherwise applies the configured query/document prefixes and
    falls back to ``encode``. Vectors are normalized when ``normalize_embeddings`` is set.
    """

    name = "hf"

    def __init__(
        self,
        model: str,
        device: str | None = None,
        *,
        normalize_embeddings: bool = True,
        query_prefix: str = "",
        document_prefix: str = "",
        revision: str | None = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "HuggingFaceEmbedder requires optional dependency 'sentence-transformers'. "
                "Install extras (e.g. `uv sync --extra hf`)."
            ) from e
        kwargs: dict[str, object] = {}
        if revision:
            kwargs["revision"] = revision
        # device=None lets SentenceTransformers auto-select (CUDA if available, else CPU).
        self._model = SentenceTransformer(model, device=device, **kwargs)
        self._normalize = normalize_embeddings
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._revision = revision

    @property
    def dim(self) -> int | None:
        fn = getattr(self._model, "get_sentence_embedding_dimension", None)
        if callable(fn):
            try:
                return int(fn())
            except Exception:  # noqa: BLE001
                return None
        return None

    @property
    def max_seq_length(self) -> int | None:
        fn = getattr(self._model, "get_max_seq_length", None)
        if callable(fn):
            try:
                v = fn()
                return int(v) if v else None
            except Exception:  # noqa: BLE001
                pass
        v = getattr(self._model, "max_seq_length", None)
        return int(v) if isinstance(v, int) else None

    @property
    def revision(self) -> str | None:
        return self._revision

    def _encode(self, texts: list[str]) -> list[list[float]]:
        embs = self._model.encode(
            texts, normalize_embeddings=self._normalize, show_progress_bar=False
        )
        return [[float(x) for x in row] for row in embs]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        fn = getattr(self._model, "encode_document", None)
        if callable(fn):
            embs = fn(texts, normalize_embeddings=self._normalize, show_progress_bar=False)
            return [[float(x) for x in row] for row in embs]
        prepared = [self._document_prefix + t for t in texts] if self._document_prefix else texts
        return self._encode(prepared)

    def embed_query(self, query: str) -> list[float]:
        fn = getattr(self._model, "encode_query", None)
        if callable(fn):
            emb = fn([query], normalize_embeddings=self._normalize, show_progress_bar=False)[0]
            return [float(x) for x in emb]
        prepared = (self._query_prefix + query) if self._query_prefix else query
        return self._encode([prepared])[0]

    def token_counter(self) -> TokenCounter:
        tok = getattr(self._model, "tokenizer", None)
        if tok is not None:
            return EmbeddingTokenCounter(tok, max_tokens=self.max_seq_length or 256)
        return super().token_counter()


def make_embedder(settings: EmbedSettings) -> Embedder:
    """Construct a fresh embedder (no caching). See ``get_cached_embedder`` for reuse."""
    if settings.provider == "hash":
        return HashEmbedder(dim=settings.dim)
    if settings.provider == "openai":
        return OpenAIEmbedder(model=settings.model)
    if settings.provider == "hf":
        return HuggingFaceEmbedder(
            model=settings.hf_model,
            device=settings.device,
            normalize_embeddings=settings.normalize_embeddings,
            query_prefix=settings.query_prefix,
            document_prefix=settings.document_prefix,
            revision=settings.revision,
        )
    raise ValueError(f"Unknown embed provider: {settings.provider}")


# --- model cache -----------------------------------------------------------------------
# Service mode reuses one loaded model across requests. The key includes everything that
# changes the produced vectors, so switching any of them yields a distinct cached model.
_EMBEDDER_CACHE: dict[tuple[object, ...], Embedder] = {}


def _cache_key(s: EmbedSettings) -> tuple[object, ...]:
    return (
        s.provider,
        s.model,
        s.hf_model,
        s.dim,
        s.device,
        s.normalize_embeddings,
        s.query_prefix,
        s.document_prefix,
        s.revision,
    )


def get_cached_embedder(settings: EmbedSettings) -> Embedder:
    """Return a process-cached embedder for ``settings`` (constructs once per key)."""
    key = _cache_key(settings)
    embedder = _EMBEDDER_CACHE.get(key)
    if embedder is None:
        embedder = make_embedder(settings)
        _EMBEDDER_CACHE[key] = embedder
    return embedder


def reset_embedder_cache() -> None:
    """Clear the embedder cache (tests / after a model switch)."""
    _EMBEDDER_CACHE.clear()


def make_token_counter(settings: EmbedSettings) -> TokenCounter:
    """A token counter aligned with the active embedding model (cached)."""
    return get_cached_embedder(settings).token_counter()


def _validate_vectors(
    vectors: list[list[float]],
    *,
    expected_dim: int | None,
    count: int,
) -> None:
    if len(vectors) != count:
        raise RuntimeError(f"Embedding provider returned {len(vectors)} vectors for {count} texts")
    dim0: int | None = None
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
    batch_size: int | None = None,
    embedder: Embedder | None = None,
) -> Iterator[EmbeddedChunk]:
    """Embed chunk Documents into vectors using the document path.

    Requires each chunk to have metadata['chunk_id']. Pass ``embedder`` to reuse a cached
    model (service mode); otherwise a fresh one is constructed.
    """
    embedder = embedder or make_embedder(settings)
    bs = batch_size or settings.batch_size

    batch_docs: list[Document] = []
    batch_texts: list[str] = []

    def flush() -> Iterator[EmbeddedChunk]:
        nonlocal batch_docs, batch_texts
        if not batch_docs:
            return
        texts = [
            _normalize_text(t, max_chars=settings.max_chars, normalize=settings.normalize_text)
            for t in batch_texts
        ]
        try:
            last_err: Exception | None = None
            for attempt in range(settings.retries):
                try:
                    vectors = embedder.embed_documents(texts)
                    _validate_vectors(
                        vectors,
                        expected_dim=(settings.dim if settings.provider == "hash" else None),
                        count=len(texts),
                    )
                    for d, v, t_in in zip(batch_docs, vectors, texts, strict=True):
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
                    return
                except Exception as e:
                    last_err = e
                    if attempt < settings.retries - 1:
                        time.sleep(settings.retry_backoff_s * (2**attempt))
                        continue
                    raise RuntimeError(
                        f"Embedding failed after {settings.retries} attempts"
                    ) from last_err
        finally:
            batch_docs.clear()
            batch_texts.clear()

    empty_count = 0
    for d in chunks:
        batch_docs.append(d)
        batch_texts.append(d.page_content or "")
        if not (d.page_content or "").strip():
            empty_count += 1
        if len(batch_docs) >= bs:
            yield from flush()

    yield from flush()
    if empty_count:
        logger.warning("Embedded %d empty-text chunk(s); they add no retrieval value", empty_count)
