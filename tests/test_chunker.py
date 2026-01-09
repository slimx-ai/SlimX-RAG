from __future__ import annotations

from langchain_core.documents import Document

from slimx_rag.chunk import chunk_documents
from slimx_rag.settings import ChunkSettings


def test_chunk_is_deterministic():
    docs = [
        Document(
            page_content="A " * 2000,
            metadata={"kb_relpath": "company/about.md", "doc_type": "company", "doc_id": "docA"},
        ),
        Document(
            page_content="B " * 2000,
            metadata={"kb_relpath": "products/overview.md", "doc_type": "products", "doc_id": "docB"},
        ),
    ]

    s = ChunkSettings(chunk_size=200, chunk_overlap=20)

    c1 = chunk_documents(docs, chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap, separators=s.separators)
    c2 = chunk_documents(docs, chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap, separators=s.separators)

    assert [d.page_content for d in c1] == [d.page_content for d in c2]
    assert [d.metadata.get("chunk_id") for d in c1] == [d.metadata.get("chunk_id") for d in c2]


def test_chunk_preserves_parent_metadata_and_adds_chunk_fields():
    doc = Document(
        page_content=("Hello\n" * 1000),
        metadata={"kb_relpath": "company/overview.md", "doc_type": "company", "doc_id": "docX"},
    )

    s = ChunkSettings(chunk_size=100, chunk_overlap=10)
    chunks = chunk_documents([doc], chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap, separators=s.separators)

    assert len(chunks) > 1

    for i, c in enumerate(chunks):
        # preserved
        assert c.metadata["kb_relpath"] == "company/overview.md"
        assert c.metadata["doc_type"] == "company"
        assert c.metadata["doc_id"] == "docX"

        # added
        assert c.metadata["chunk_index"] == i
        assert c.metadata["parent_doc_id"] == "docX"
        assert c.metadata["parent_kb_relpath"] == "company/overview.md"
        assert c.metadata["parent_doc_type"] == "company"
        assert "chunk_id" in c.metadata
        assert isinstance(c.metadata["chunk_id"], str)
        assert len(c.metadata["chunk_id"]) == 32  # blake2b digest_size=16 => 32 hex chars


def test_chunk_validates_parameters():
    doc = Document(page_content="x" * 1000, metadata={"kb_relpath": "x.md", "doc_id": "x"})

    # overlap >= size
    try:
        chunk_documents([doc], chunk_size=100, chunk_overlap=100)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # size <= 0
    try:
        chunk_documents([doc], chunk_size=0, chunk_overlap=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # overlap < 0
    try:
        chunk_documents([doc], chunk_size=100, chunk_overlap=-1)
        assert False, "Expected ValueError"
    except ValueError:
        pass
