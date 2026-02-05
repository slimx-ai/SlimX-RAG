from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from slimx_rag.settings import IndexingPipelineSettings

logger = logging.getLogger(__name__)


def iter_subdirs(kb_dir: Path) -> Iterable[Path]:
    """Yield subdirectories of kb_dir representing document types."""
    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")
    if not kb_dir.is_dir():
        raise NotADirectoryError(f"Knowledge base path is not a directory: {kb_dir}")

    for p in kb_dir.iterdir():
        if p.is_dir():
            yield p


def _hash_text(text: str, *, digest_size: int) -> str:
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _hash_path(kb_relpath: str) -> str:
    # doc identity: stable across edits as long as path stays stable
    return _hash_text(kb_relpath, digest_size=32)


def _content_hash(text: str) -> str:
    # doc version fingerprint: changes when content changes (supports incremental indexing later)
    return _hash_text(text or "", digest_size=16)


def _doc_type_from_relpath(relpath: Path, depth: int) -> str:
    """Derive doc_type from relative path parts up to the given depth."""
    parts = relpath.parts[: max(1, depth)]
    return "/".join(parts) if parts else ""


def _fallback_doc_id(source: str, content_hash: str) -> str:
    # identity for docs with no stable kb_relpath: stable for same source+content
    payload = f"fallback\n{source}\n{content_hash}"
    return _hash_text(payload, digest_size=32)


def fetch_documents(settings: IndexingPipelineSettings) -> List[Document]:
    """
    Load documents from knowledge-base (subfolders) and attach stable metadata.

    Baseline metadata (always set):
      - doc_id          (stable identity; relpath-based when possible)
      - content_hash    (version fingerprint; changes when content changes)
      - content_len     (chars)
      - doc_type        (folder-derived when possible, else "unknown")
      - file_ext        (when discoverable)
      - kb_relpath      (when discoverable)

    Notes:
      - doc_id is *identity* (path-based when possible)
      - content_hash is *version* (content-based)
    """

    settings.ingest.validate()
    if not settings.kb_dir.exists():
        raise FileNotFoundError(f"kb_dir not found: {settings.kb_dir}")
    if not settings.kb_dir.is_dir():
        raise NotADirectoryError(f"kb_dir is not a directory: {settings.kb_dir}")


    kb_dir = settings.kb_dir
    glob = settings.ingest.glob
    documents: List[Document] = []
    logger.info(f"Loading documents from knowledge base at: {kb_dir}")

    for doc_type_dir in sorted(iter_subdirs(kb_dir)):  # deterministic directory order
        logger.debug(f"Scanning {doc_type_dir}")

        loader = DirectoryLoader(
            str(doc_type_dir),
            glob=glob,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=settings.ingest.show_progress,
            use_multithreading=settings.ingest.multithreading,
        )

        docs = loader.load()

        for d in docs:
            source = str(d.metadata.get("source") or "")
            relpath: Optional[Path] = None

            if source:
                source_path = Path(source)
                # Best-effort: attach a stable relpath when source is under kb_dir
                if kb_dir in source_path.parents:
                    relpath = source_path.relative_to(kb_dir)

            # Always compute version fingerprint (B)
            text = d.page_content or ""
            d.metadata["content_hash"] = _content_hash(text)
            d.metadata["content_len"] = len(text)

            # Prefer stable identity from relpath; otherwise fall back (A)
            if relpath is not None:
                d.metadata["kb_relpath"] = relpath.as_posix() # consistent slashes across OSes
                d.metadata["file_ext"] = relpath.suffix.lower().lstrip(".")
                d.metadata["doc_id"] = _hash_path(str(relpath))
                d.metadata["doc_type"] = _doc_type_from_relpath(relpath, depth=1)
            else:
                # No relpath => still guarantee minimal metadata
                d.metadata.setdefault("kb_relpath", "")
                if "file_ext" not in d.metadata:
                    try:
                        d.metadata["file_ext"] = Path(source).suffix.lower().lstrip(".") if source else ""
                    except Exception:
                        d.metadata["file_ext"] = ""
                d.metadata["doc_id"] = _fallback_doc_id(source, d.metadata["content_hash"])
                d.metadata.setdefault("doc_type", "unknown")

        documents.extend(docs)

    logger.info(f"Fetched {len(documents)} documents from knowledge base.")
    return documents
