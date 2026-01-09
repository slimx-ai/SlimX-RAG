from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _stable_doc_sort_key(doc: Document) -> Tuple[str, str]:
    """
    Deterministic sort key for documents.

    Why:
    - Filesystem loaders can return docs in non-stable order (OS differences,
      multithreading, etc.).
    - Sorting ensures chunk output stays stable run-to-run.
    """
    src = str(doc.metadata.get("source", ""))
    doc_type = str(doc.metadata.get("doc_type", ""))
    return (src, doc_type)


def _make_chunk_id(parent_source: str, chunk_index: int, chunk_text: str) -> str:
    """
    Stable unique ID for a chunk.

    Uses BLAKE2b hash over: (parent_source, chunk_index, chunk_text)
    """
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(parent_source.encode("utf-8", errors="ignore"))
    hasher.update(b"\n")
    hasher.update(str(chunk_index).encode("utf-8", errors="ignore"))
    hasher.update(b"\n")
    hasher.update(chunk_text.encode("utf-8", errors="ignore"))
    return hasher.hexdigest()


def chunk_documents(
    docs: Iterable[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    separators: Sequence[str] = ("\n\n", "\n", " ", ""),
    add_position_chunk_metadata: bool = True,
) -> List[Document]:
    """
    Split documents into deterministic chunks for embedding and retrieval.

    Determinism strategy:
    1) Sort docs by stable metadata.
    2) Use deterministic splitter for same text/settings.
    3) Add stable chunk_id per chunk.
    """
    docs_list = list(docs)
    docs_list.sort(key=_stable_doc_sort_key)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators),
    )

    out: List[Document] = []

    for doc in docs_list:
        parent_source = str(doc.metadata.get("source", ""))
        splits = splitter.split_text(doc.page_content or "")

        for chunk_index, chunk_text in enumerate(splits):
            chunk_metadata = dict(doc.metadata)

            if add_position_chunk_metadata:
                chunk_metadata["chunk_index"] = chunk_index
                chunk_metadata["parent_source"] = parent_source
            
            # Add stable chunk_id
            chunk_metadata["chunk_id"] = _make_chunk_id(parent_source, chunk_index, chunk_text)

            out.append(Document(page_content=chunk_text, metadata=chunk_metadata))

    return out
