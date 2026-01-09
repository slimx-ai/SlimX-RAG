from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _stable_doc_sort_key(doc: Document) -> Tuple[str, str]:
    """
    Build a deterministic sort key for documents.

    We prefer metadata["kb_relpath"] when present because it is stable and
    independent of absolute paths. Fall back to metadata["source"].
    """
    src = str(doc.metadata.get("kb_relpath") or doc.metadata.get("source") or "")
    doc_type = str(doc.metadata.get("doc_type", ""))
    return (src, doc_type)


def _make_chunk_id(parent_id: str, chunk_index: int, chunk_text: str) -> str:
    """
    Create a stable chunk ID.

    Non-security use-case: fingerprinting / caching / deduping.
    BLAKE2b is fast and supports a short digest size.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(parent_id.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(str(chunk_index).encode("utf-8"))
    h.update(b"\n")
    h.update(chunk_text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def chunk_documents(
    docs: Iterable[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    separators: Sequence[str] = ("\n\n", "\n", " ", ""),
    extended_chunk_metadata: bool = True,
) -> List[Document]:
    """
    Split Documents into deterministic chunks for embedding/retrieval.

    Determinism strategy:
      - sort input docs by stable key (kb_relpath/source + doc_type)
      - RecursiveCharacterTextSplitter is deterministic for same text/settings
      - chunk_id is stable hash of (parent_id, chunk_index, chunk_text)

    Metadata:
      - preserves all parent metadata on each chunk
      - optionally adds: chunk_index, parent_doc_id, parent_kb_relpath,
        parent_doc_type, chunk_id
    """
    # Validate params (mirror IndexingSettings.validate logic for safety)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    docs_list = list(docs)
    docs_list.sort(key=_stable_doc_sort_key)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators),
    )

    out: List[Document] = []
    for doc in docs_list:
        parent_doc_id = str(doc.metadata.get("doc_id", ""))  # from ingest
        parent_kb_relpath = str(doc.metadata.get("kb_relpath", ""))  # from ingest
        parent_doc_type = str(doc.metadata.get("doc_type", ""))  # from ingest
        parent_id_for_hash = parent_doc_id or parent_kb_relpath or str(doc.metadata.get("source", ""))

        splits = splitter.split_text(doc.page_content or "")


        for i, text in enumerate(splits):
            md = dict(doc.metadata)  # preserve ingest metadata
            # always add chunk_id for deduping/caching purposes
            md["chunk_id"] = _make_chunk_id(parent_id_for_hash, i, text)
            if extended_chunk_metadata:
                md["chunk_index"] = i
                md["parent_doc_id"] = parent_doc_id
                md["parent_kb_relpath"] = parent_kb_relpath
                md["parent_doc_type"] = parent_doc_type

            out.append(Document(page_content=text, metadata=md))

    return out
