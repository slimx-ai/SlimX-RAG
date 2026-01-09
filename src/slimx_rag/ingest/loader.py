from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from slimx_rag.settings import IndexingSettings

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


def _hash_path(kb_relpath: str) -> str:
    h = hashlib.blake2b(digest_size=32)
    h.update(kb_relpath.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _doc_type_from_relpath(relpath: Path, depth: int) -> str:
    """Derive doc_type from relative path parts up to the given depth."""
    parts = relpath.parts[: max(1, depth)]
    return "/".join(parts) if parts else ""


def fetch_documents(
    settings: Optional[PipelineSettings] = None,
    *,
    kb_dir: Optional[Path] = None,
    glob: Optional[str] = None,
) -> List[Document]:
    """
    Load documents from knowledge-base (subfolders) and attach stable metadata.

    Baseline metadata (when source is available):
      - kb_relpath
      - file_ext
      - doc_id

    Optional semantic metadata:
      - doc_type (derived from folder structure; depth=1 means top-level folder)
    """
    settings = settings or PipelineSettings()
    settings.validate()

    kb_dir = kb_dir or settings.kb_dir
    glob = glob or settings.ingest.glob

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
            source = d.metadata.get("source")
            relpath: Optional[Path] = None

            if source:
                source_path = Path(source)
                if kb_dir in source_path.parents:
                    relpath = source_path.relative_to(kb_dir)

            if relpath is not None:
                d.metadata["kb_relpath"] = str(relpath)
                d.metadata["file_ext"] = relpath.suffix.lower().lstrip(".")
                d.metadata["doc_id"] = _hash_path(str(relpath))

                # If you later add doc_type control, wire it here; for now doc_type is always derived
                d.metadata["doc_type"] = _doc_type_from_relpath(relpath, depth=1)

        documents.extend(docs)

    logger.info(f"Fetched {len(documents)} documents from knowledge base.")
    return documents
