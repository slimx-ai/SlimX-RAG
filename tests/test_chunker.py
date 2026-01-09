from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from slimx_rag.chunk import chunk_documents, ChunkSettings


def test_chunk_documents_basic():
    docs = [
        Document(
            page_content="A\n\nB\n\nC\n\nD\n\nE",
            metadata={"doc_id": "abcd1234", "kb_relpath": "company/about.md", "doc_type": "company"},
        )
    ]

    chunks = chunk_documents(docs, settings=ChunkSettings(chunk_size=4, chunk_overlap=0))

    assert len(chunks) >= 2
    assert all("chunk_id" in c.metadata for c in chunks)
    assert all(c.metadata["parent_doc_id"] == "abcd1234" for c in chunks)
    assert sorted(set(c.metadata["chunk_index"] for c in chunks)) == list(range(len(chunks)))


def test_chunk_documents_validation():
    docs = [Document(page_content="hello", metadata={})]

    with pytest.raises(ValueError):
        chunk_documents(docs, settings=ChunkSettings(chunk_size=0))

    with pytest.raises(ValueError):
        chunk_documents(docs, settings=ChunkSettings(chunk_size=10, chunk_overlap=10))
