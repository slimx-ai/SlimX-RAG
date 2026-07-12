from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("reindex", [None, "1; touch must-not-run"])
def test_entrypoint_passes_exact_artifact_paths_and_safe_reindex_flag(tmp_path: Path, reindex: str | None) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    capture_path = tmp_path / "captured-args.txt"
    fake_cli = fake_bin / "slimx-rag"
    fake_cli.write_text(
        '#!/usr/bin/env sh\nset -eu\nprintf \'%s\\n\' "$@" > "$CAPTURE_PATH"\n',
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)
    kb_dir = tmp_path / "knowledge base"
    kb_dir.mkdir()
    index_path = tmp_path / "index volume" / "custom-index.jsonl"
    state_path = tmp_path / "state volume" / "custom-state.json"
    if reindex is not None:
        index_path.parent.mkdir(parents=True)
        index_path.write_text("existing corpus", encoding="utf-8")
    hostile_device = "cuda:0; touch must-not-run"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CAPTURE_PATH": str(capture_path),
        "RAG_KB_DIR": str(kb_dir),
        "RAG_INDEX_PATH": str(index_path),
        "RAG_STATE_PATH": str(state_path),
        "RAG_EMBED_PROVIDER": "hash",
        "RAG_HF_MODEL": "unused/model",
        "RAG_EMBED_DIM": "8",
        "RAG_EMBED_DEVICE": hostile_device,
        "RAG_INDEX_BACKEND": "local",
    }
    if reindex is None:
        env.pop("RAG_REINDEX", None)
    else:
        env["RAG_REINDEX"] = reindex

    completed = subprocess.run(
        ["sh", str(repo_root / "docker-entrypoint.sh"), "true"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    args = capture_path.read_text(encoding="utf-8").splitlines()
    assert args[:3] == ["run", "--kb-dir", str(kb_dir)]
    assert args[args.index("--index") + 1] == str(index_path)
    assert args[args.index("--state") + 1] == str(state_path)
    assert args[args.index("--embed-device") + 1] == hostile_device
    assert ("--reindex" in args) is (reindex is not None)
    assert not (tmp_path / "must-not-run").exists()
