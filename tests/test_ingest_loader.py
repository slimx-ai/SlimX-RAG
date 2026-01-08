from __future__ import annotations

from pathlib import Path
import pytest

from slimx_rag.ingest.loader import fetch_documents
from slimx_rag.settings import Settings


def write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_fetch_documents_adds_doc_type_and_baseline_metadata(tmp_path: Path) -> None:
    kb_dir = tmp_path / "knowledge-base"
    write(kb_dir / "company" / "about.md", "# About\nThis is the about page.")
    write(kb_dir / "products" / "features.md", "# Features\nThese are the features.")

    settings = Settings(kb_dir=kb_dir)  # default doc_type_mode="subdir"
    documents = fetch_documents(settings=settings)

    assert len(documents) == 2

    # doc_type should be derived from the first subfolder under kb_dir
    doc_types = sorted(d.metadata.get("doc_type") for d in documents)
    assert doc_types == ["company", "products"]

    # Baseline metadata should always exist when "source" is available
    relpaths = sorted(d.metadata.get("kb_relpath") for d in documents)
    assert relpaths == ["company/about.md", "products/features.md"]

    exts = sorted(d.metadata.get("file_ext") for d in documents)
    assert exts == ["md", "md"]

    doc_ids = [d.metadata.get("doc_id") for d in documents]
    assert all(isinstance(x, str) and len(x) == 16 for x in doc_ids)


def test_fetch_documents_no_doc_type_when_disabled(tmp_path: Path) -> None:
    kb_dir = tmp_path / "knowledge-base"
    write(kb_dir / "company" / "about.md", "# About\nThis is the about page.")

    settings = Settings(kb_dir=kb_dir, doc_type_mode="none")
    documents = fetch_documents(settings=settings)

    assert len(documents) == 1
    assert "doc_type" not in documents[0].metadata  # disabled

    # Baseline metadata should still exist
    assert documents[0].metadata.get("kb_relpath") == "company/about.md"
    assert documents[0].metadata.get("file_ext") == "md"
    assert isinstance(documents[0].metadata.get("doc_id"), str)
    assert len(documents[0].metadata["doc_id"]) == 16


def test_fetch_documents_raises_if_missing_kb(tmp_path: Path) -> None:
    kb_dir = tmp_path / "nonexistent-kb"
    settings = Settings(kb_dir=kb_dir)

    with pytest.raises(FileNotFoundError):
        fetch_documents(settings=settings)


def test_fetch_documents_raises_if_kb_is_file(tmp_path: Path) -> None:
    kb_file = tmp_path / "knowledge-base"
    kb_file.write_text("not a directory", encoding="utf-8")

    settings = Settings(kb_dir=kb_file)

    with pytest.raises(NotADirectoryError):
        fetch_documents(settings=settings)
