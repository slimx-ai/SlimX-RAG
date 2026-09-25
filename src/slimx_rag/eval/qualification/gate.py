"""Frozen quality gate evaluation (thresholds live in a committed JSON file)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GateCheck:
    name: str
    kind: str  # max | min
    threshold: float
    observed: float | None
    passed: bool


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    checks: tuple[GateCheck, ...]
    skipped: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
            "skipped": list(self.skipped),
        }

    def to_markdown(self) -> str:
        lines = ["| Check | Rule | Observed | Result |", "| --- | --- | --- | --- |"]
        for check in self.checks:
            rule = f"{'<=' if check.kind == 'max' else '>='} {check.threshold}"
            observed = "n/a" if check.observed is None else f"{check.observed}"
            lines.append(f"| {check.name} | {rule} | {observed} | {'PASS' if check.passed else 'FAIL'} |")
        return "\n".join(lines) + "\n"


def load_gate(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_gate(report: dict[str, Any], gate: dict[str, Any]) -> GateResult:
    checks: list[GateCheck] = []
    skipped: list[str] = []
    if gate.get("dataset_version") != report.get("dataset_version"):
        checks.append(
            GateCheck(
                "dataset_version",
                "max",
                0,
                1,
                False,
            )
        )
    hard = report.get("hard", {})
    for name, limit in gate.get("hard", {}).items():
        observed = hard.get(name)
        passed = observed is not None and float(observed) <= float(limit)
        checks.append(
            GateCheck(f"hard.{name}", "max", float(limit), None if observed is None else float(observed), passed)
        )
    for name, limit in gate.get("lifecycle", {}).items():
        observed = report.get("lifecycle", {}).get(name)
        passed = observed is not None and float(observed) <= float(limit)
        checks.append(
            GateCheck(f"lifecycle.{name}", "max", float(limit), None if observed is None else float(observed), passed)
        )
    provider = str(report.get("provider"))
    for name, limit in gate.get("hard_by_provider", {}).get(provider, {}).items():
        observed = hard.get(name, report.get("aggregate", {}).get(name))
        passed = observed is not None and float(observed) <= float(limit)
        checks.append(
            GateCheck(
                f"hard[{provider}].{name}", "max", float(limit), None if observed is None else float(observed), passed
            )
        )
    ranking = gate.get("ranking", {}).get(provider)
    if ranking is None:
        skipped.append(f"ranking thresholds: none frozen for provider {provider!r}")
    else:
        aggregate = report.get("aggregate", {})
        for name, minimum in ranking.items():
            observed = aggregate.get(name)
            passed = observed is not None and float(observed) >= float(minimum)
            checks.append(
                GateCheck(
                    f"ranking.{name}", "min", float(minimum), None if observed is None else float(observed), passed
                )
            )
    for tag, minimums in gate.get("by_tag", {}).get(provider, {}).items():
        tag_row = report.get("by_tag", {}).get(tag, {})
        for name, minimum in minimums.items():
            observed = tag_row.get(name)
            passed = observed is not None and float(observed) >= float(minimum)
            checks.append(
                GateCheck(
                    f"by_tag.{tag}.{name}", "min", float(minimum), None if observed is None else float(observed), passed
                )
            )
    return GateResult(passed=all(c.passed for c in checks), checks=tuple(checks), skipped=tuple(skipped))
