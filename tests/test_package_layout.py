from __future__ import annotations

from pathlib import Path

from slimx_rag.answer import answer
from slimx_rag.eval import run_eval
from slimx_rag.retrieve import RetrievalResult as CompatRetrievalResult
from slimx_rag.retrieve import retrieve as compat_retrieve
from slimx_rag.retrieval import RetrievalResult, retrieve

from slimx_rag.cli import main


def test_retrieval_package_preserves_legacy_imports():
    assert compat_retrieve is retrieve
    assert CompatRetrievalResult is RetrievalResult
    assert callable(answer)
    assert callable(run_eval)


def test_tiny_demo_example_runs_with_hash_embeddings(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    kb_dir = repo_root / "examples" / "tiny_demo" / "knowledge-base"

    assert main([
        "run",
        "--kb-dir",
        str(kb_dir),
        "--out-dir",
        str(tmp_path / "out"),
        "--embed-provider",
        "hash",
    ]) == 0
