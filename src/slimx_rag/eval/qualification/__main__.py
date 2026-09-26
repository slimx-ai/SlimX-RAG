"""``python -m slimx_rag.eval.qualification --provider hf --out DIR [--gate quality-gate.json]``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .gate import evaluate_gate, load_gate
from .runner import DEFAULT_HF_MODEL, DEFAULT_HF_REVISION, TITLE_MODES, run_qualification, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="slimx_rag.eval.qualification")
    parser.add_argument("--provider", choices=("hash", "hf"), default="hash")
    parser.add_argument("--out", required=True, help="output directory for corpus, index, report")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hf-revision", default=DEFAULT_HF_REVISION, help="exact model commit for --provider hf")
    parser.add_argument(
        "--title-mode",
        choices=TITLE_MODES,
        default="benchmark",
        help="document titles: the corpus's own (benchmark) or ControlRoom's upload filenames (filename)",
    )
    parser.add_argument("--gate", default=None, help="frozen quality-gate JSON; exit 1 on failure")
    args = parser.parse_args(argv)

    out = Path(args.out)
    report = run_qualification(
        provider=args.provider,
        out_dir=out,
        top_k=args.top_k,
        hf_model=args.hf_model,
        device=args.device,
        hf_revision=args.hf_revision,
        title_mode=args.title_mode,
    )
    write_report(report, out)
    print(json.dumps({k: report[k] for k in ("provider", "aggregate", "hard", "lifecycle", "latency_ms")}, indent=2))
    if args.gate:
        result = evaluate_gate(report, load_gate(Path(args.gate)))
        (out / "gate-result.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
        (out / "gate-result.md").write_text(result.to_markdown(), encoding="utf-8")
        print(result.to_markdown())
        for note in result.skipped:
            print(f"skipped: {note}")
        print("GATE:", "PASS" if result.passed else "FAIL")
        return 0 if result.passed else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
