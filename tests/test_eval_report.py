from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from slimx_rag.eval.runner import EvalReport, load_eval_cases, write_eval_report


def _report() -> EvalReport:
    return EvalReport(
        cases=[
            {
                "question": "What is X?",
                "hit": True,
                "cited": True,
                "warnings": [],
                "retrieved_sources": ["doc.md"],
            }
        ],
        hit_at_k=1.0,
        citation_rate=1.0,
        insufficient_context_pass_rate=1.0,
    )


def test_write_eval_report_markdown(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "report.md"
    write_eval_report(_report(), out)
    text = out.read_text(encoding="utf-8")
    assert "hit@k: 1.00" in text
    assert "| What is X? | True | True |" in text


def test_write_eval_report_json(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    write_eval_report(_report(), out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == asdict(_report())


def test_load_eval_cases_round_trip(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps({"question": "Q1", "expected_sources": ["a.md"]})
        + "\n\n"
        + json.dumps({"question": "Q2", "should_have_answer": False})
        + "\n",
        encoding="utf-8",
    )
    cases = load_eval_cases(dataset)
    assert [c.question for c in cases] == ["Q1", "Q2"]
    assert cases[0].expected_sources == ["a.md"]
    assert cases[1].should_have_answer is False


def test_load_eval_cases_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.jsonl"):
        load_eval_cases(tmp_path / "missing.jsonl")

    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_eval_cases(bad)

    no_question = tmp_path / "noq.jsonl"
    no_question.write_text(json.dumps({"expected_sources": []}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="question"):
        load_eval_cases(no_question)
