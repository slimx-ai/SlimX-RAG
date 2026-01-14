from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _stable_doc_sort_key(doc: Document) -> Tuple[str, str]:
    """
    Build a deterministic sort key for documents.

    Preference order:
      1) kb_relpath (stable across machines)
      2) source
      3) doc_id
    """
    src = str(
        doc.metadata.get("kb_relpath")
        or doc.metadata.get("source")
        or doc.metadata.get("doc_id")
        or ""
    )
    doc_type = str(doc.metadata.get("doc_type", ""))
    return (src, doc_type)


def _hash_text(text: str, *, digest_size: int = 16) -> str:
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _chunk_config_fingerprint(
    *,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str],
) -> str:
    """Fingerprint chunking params so chunk_id changes when config changes."""
    seps = "\u241E".join(separators)  # record separators unambiguously
    return _hash_text(f"{chunk_size}|{chunk_overlap}|{seps}", digest_size=8)


def _make_chunk_id(
    *,
    parent_id: str,
    content_hash: str,
    chunk_index: int,
    chunk_cfg_hash: str,
) -> str:
    """
    Create a stable chunk ID.

    Uses doc identity + doc version + chunk params + index.
    This avoids hashing the full chunk text (faster) while staying deterministic.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(parent_id.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(content_hash.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(chunk_cfg_hash.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(str(chunk_index).encode("utf-8"))
    return h.hexdigest()


def chunk_documents(
    docs: Iterable[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    separators: Sequence[str] = ("\n\n", "\n", " ", ""),
    extended_chunk_metadata: bool = True,
    include_chunk_cfg_hash: bool = False,
) -> List[Document]:
    """
    Split Documents into deterministic chunks for embedding/retrieval.

    Determinism strategy:
      - sort input docs by stable key (kb_relpath/source/doc_id + doc_type)
      - RecursiveCharacterTextSplitter is deterministic for same text/settings
      - chunk_id is stable hash of (parent_id, content_hash, chunk_cfg_hash, chunk_index)

    Metadata:
      - preserves all parent metadata on each chunk
      - always adds: chunk_id
      - optionally adds: chunk_index, parent_doc_id, parent_kb_relpath, parent_doc_type
      - optionally adds: chunk_cfg_hash (useful for debugging)
    """
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

    chunk_cfg_hash = _chunk_config_fingerprint(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )

    out: List[Document] = []
    for doc in docs_list:
        parent_doc_id = str(doc.metadata.get("doc_id", ""))
        parent_kb_relpath = str(doc.metadata.get("kb_relpath", ""))
        parent_doc_type = str(doc.metadata.get("doc_type", ""))

        # Ensure content_hash exists even if ingest was skipped
        if doc.metadata.get("content_hash"):
            content_hash = str(doc.metadata["content_hash"])
        else:
            content_hash = _hash_text(doc.page_content or "", digest_size=16)

        parent_id_for_hash = parent_doc_id or parent_kb_relpath or str(doc.metadata.get("source", ""))

        splits = splitter.split_text(doc.page_content or "")

        for i, text in enumerate(splits):
            md = dict(doc.metadata)

            md["chunk_id"] = _make_chunk_id(
                parent_id=parent_id_for_hash,
                content_hash=content_hash,
                chunk_index=i,
                chunk_cfg_hash=chunk_cfg_hash,
            )

            if include_chunk_cfg_hash:
                md["chunk_cfg_hash"] = chunk_cfg_hash

            if extended_chunk_metadata:
                md["chunk_index"] = i
                md["parent_doc_id"] = parent_doc_id
                md["parent_kb_relpath"] = parent_kb_relpath
                md["parent_doc_type"] = parent_doc_type

            out.append(Document(page_content=text, metadata=md))

    return out
