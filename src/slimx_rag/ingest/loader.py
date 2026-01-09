from __future__ import annotations 

import logging
import hashlib
from pathlib import Path
from typing import List, Iterable, Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from slimx_rag.settings import Settings

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
    """Generate a short hash for a given relative path."""
    h = hashlib.sha256(kb_relpath.encode("utf-8")).hexdigest()
    return h[:16]

def _doc_type_from_relpath(relpath: Path, depth: int) -> str:
    """Derive doc_type from relative path parts up to the given depth."""
    parts = relpath.parts[:depth]
    if len(parts) == 0:
        return ""
    d = max(1, depth)
    return "/".join(parts[:d])

def fetch_documents(
        settings: Optional[Settings] = None,
        *, 
        kb_dir: Optional[Path] = None, 
        glob: Optional[str] = None
        ) -> List[Document]:
    
    """
    Load documents from knowledge-base (subfolders).

    Metadata added:
        if settings.doc_type_mode == "subdir":
            - doc_type: subfolder name inside kb_dir

    Args:
      settings: Settings instance. If omitted, Settings.default() is used.
      kb_dir: Direct override for the KB path (highest priority).
      glob: Glob pattern passed to DirectoryLoader.

    Returns:
      List[Document]
    """

    settings = settings or Settings.default()
    kb_dir = kb_dir or settings.kb_dir
    glob = glob or settings.glob

    documents: List[Document] = []
    logger.info(f"Loading documents from knowledge base at: {kb_dir}")

    for doc_type_dir in sorted(iter_subdirs(kb_dir)): # yield subdirs in a deterministic order
        logger.debug(f"Scanning {doc_type_dir}")

        loader = DirectoryLoader(
            str(doc_type_dir),
            glob=glob,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
            use_multithreading=True,
        )

        docs = loader.load()

        for d in docs:
            # Baseline metadata: always set
            # Many loaders already provide "source" metadata, but we normalize/enrich it
            source = d.metadata.get("source")

            if source:
                source_path = Path(source)
                relpath = source_path.relative_to(kb_dir) if kb_dir in source_path.parents else None
            else:
                relpath = None

            if relpath is not None:
                d.metadata["kb_relpath"] = str(relpath)
                d.metadata["file_ext"] = relpath.suffix.lower().lstrip(".")
                d.metadata["doc_id"] = _hash_path(str(relpath))

                if settings.doc_type_mode == "subdir":
                    d.metadata["doc_type"] = _doc_type_from_relpath(
                        relpath,
                        settings.doc_type_depth,)
            
        documents.extend(docs)

    logger.info(f"Fetched {len(documents)} documents from knowledge base.")
    return documents
