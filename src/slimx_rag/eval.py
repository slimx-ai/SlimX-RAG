from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from slimx_rag.answer import answer
from slimx_rag.retrieve import retrieve
from slimx_rag.settings import EmbedSettings, IndexSettings


@dataclass(frozen=True, slots=True)
class EvalCase:
    question: str
    expected_sources: list[str]
    should_have_answer: bool = True


@dataclass(frozen=True, slots=True)
class EvalReport:
    cases: list[dict]
    hit_at_k: float
    citation_rate: float
    insufficient_context_pass_rate: float

    def to_markdown(self) -> str:
        lines = [
            "# SlimX-RAG Demo Evaluation",
            "",
            f"- hit@k: {self.hit_at_k:.2f}",
            f"- citation rate: {self.citation_rate:.2f}",
            f"- insufficient-context pass rate: {self.insufficient_context_pass_rate:.2f}",
            "",
            "| Question | Hit | Cited | Warnings |",
            "| --- | --- | --- | --- |",
        ]
        for case in self.cases:
            lines.append(
                f"| {case['question']} | {case['hit']} | {case['cited']} | {', '.join(case['warnings'])} |"
            )
        return "\n".join(lines) + "\n"


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cases.append(
            EvalCase(
                question=str(obj["question"]),
                expected_sources=[str(x) for x in obj.get("expected_sources", [])],
                should_have_answer=bool(obj.get("should_have_answer", True)),
            )
        )
    return cases


def run_eval(
    cases: list[EvalCase],
    *,
    index_path: Path,
    embed_settings: EmbedSettings,
    index_settings: IndexSettings,
    model: str,
    state_path: Path | None = None,
    top_k: int = 5,
    timeout: float | None = None,
    max_tokens: int | None = None,
    max_context_chars: int | None = None,
) -> EvalReport:
    rows = []
    hits = cited = insufficient_pass = insufficient_total = 0
    for case in cases:
        retrieval = retrieve(
            case.question,
            index_path=index_path,
            embed_settings=embed_settings,
            index_settings=index_settings,
            state_path=state_path,
            top_k=top_k,
        )
        result = answer(
            case.question,
            retrieval,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            max_context_chars=max_context_chars,
        )
        retrieved_sources = [
            str(chunk.metadata.get("parent_kb_relpath") or chunk.metadata.get("kb_relpath") or "")
            for chunk in retrieval.chunks
        ]
        hit = not case.expected_sources or any(
            expected in source
            for expected in case.expected_sources
            for source in retrieved_sources
        )
        has_citation = any(citation in result.answer for citation in result.citations)
        if hit:
            hits += 1
        if has_citation:
            cited += 1
        if not case.should_have_answer:
            insufficient_total += 1
            if "not contain enough information" in result.answer.lower() or "insufficient" in " ".join(result.warnings):
                insufficient_pass += 1
        rows.append({
            "question": case.question,
            "hit": hit,
            "cited": has_citation,
            "warnings": result.warnings,
            "retrieved_sources": retrieved_sources,
        })
    total = max(len(cases), 1)
    return EvalReport(
        cases=rows,
        hit_at_k=hits / total,
        citation_rate=cited / total,
        insufficient_context_pass_rate=(insufficient_pass / insufficient_total if insufficient_total else 1.0),
    )


def write_eval_report(report: EvalReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        out_path.write_text(report.to_markdown(), encoding="utf-8")
