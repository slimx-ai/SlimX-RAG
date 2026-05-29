from __future__ import annotations

import json
from pathlib import Path

from slimx_rag.cli import main


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_cli_run_retrieve_ask_and_eval(tmp_path: Path, capsys):
    kb = tmp_path / "kb"
    out = tmp_path / "out"
    write(kb / "overview.md", "SlimX builds explicit inspectable research AI systems.")
    assert main(["run", "--kb-dir", str(kb), "--out-dir", str(out), "--embed-dim", "16"]) == 0
    assert (out / "embeddings.jsonl").exists()

    assert main(["retrieve", "--out-dir", str(out), "--q", "research AI", "--k", "1"]) == 0
    retrieved = json.loads(capsys.readouterr().out.splitlines()[0])
    assert "citation" in retrieved

    assert main(["ask", "--out-dir", str(out), "--q", "What does SlimX build?", "--k", "1"]) == 0
    asked = json.loads(capsys.readouterr().out)
    assert asked["citations"]

    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        json.dumps({"question": "What does SlimX build?", "expected_sources": ["overview.md"]}) + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.md"
    assert main(["eval", "--out-dir", str(out), "--dataset", str(dataset), "--out", str(report), "--k", "1"]) == 0
    assert "hit@k" in report.read_text(encoding="utf-8")
