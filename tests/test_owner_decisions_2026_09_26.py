"""Owner decisions of 2026-09-26 on the ControlRoom qualification benchmark.

1. The original frozen gate (``quality-gate.json``) is unchanged; ControlRoom's title convention
   (``--title-mode filename``) is a second, mandatory gate with identical thresholds.
2. Evaluator version 2: lifecycle stale text is tied to the revised document's identity, not to
   the text occurring anywhere (RAG-AUD-045).
3. A source-code document carries a ``Language:`` identity line (``index-shaping-v5``) so a
   question about "the Python function that ..." reaches the code unit lexically as well.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slimx_rag.chunk import chunk_parsed_document
from slimx_rag.chunk.tokenizer import HeuristicTokenCounter
from slimx_rag.document import DocumentSource, parse_document
from slimx_rag.eval.qualification import EVALUATOR_VERSION, evaluate_gate, load_gate, run_qualification
from slimx_rag.eval.qualification.gold import Case, Scope
from slimx_rag.eval.qualification.metrics import evaluate_case, hard_counters, stale_source_docs

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "examples" / "controlroom_qualification"

_MODULE = '''"""Atlas gantry controller helpers (AG-7)."""

LASER_OFFSET_MM = 0.42


def compute_gantry_offset(laser_mm: float) -> float:
    """Apply the 0.42 mm laser head offset loaded from calib_v3.cfg."""
    return laser_mm - LASER_OFFSET_MM
'''

_MARKDOWN_WITH_FENCE = """# Calibration notes

Load the file and run:

```python
compute_gantry_offset(1.0)
```
"""


# --- 3. Language identity line -----------------------------------------------------------------


def test_code_document_chunks_carry_a_language_identity_line_and_metadata() -> None:
    src = DocumentSource(
        document_id="ctl",
        filename="atlas_controller.py",
        mime_type="text/x-python",
        content=_MODULE.encode(),
        metadata={"title": "atlas_controller.py"},  # ControlRoom sends the upload filename
    )
    doc = parse_document(src)
    assert doc.source_type == "code" and doc.metadata.get("language") == "python"
    chunks = chunk_parsed_document(doc, token_counter=HeuristicTokenCounter(max_tokens=1000))
    assert chunks, "a module must produce at least one retrieval unit"
    for chunk in chunks:
        assert chunk.embedding_text.startswith("Document: atlas_controller.py\nLanguage: python\n")
        assert "Language:" not in chunk.display_text
        assert chunk.metadata["language"] == "python"


def test_non_code_documents_have_no_language_line_even_with_a_code_fence() -> None:
    src = DocumentSource(
        document_id="notes",
        filename="calibration-notes.md",
        mime_type="text/markdown",
        content=_MARKDOWN_WITH_FENCE.encode(),
        metadata={"title": "calibration-notes.md"},
    )
    doc = parse_document(src)
    chunks = chunk_parsed_document(doc, token_counter=HeuristicTokenCounter(max_tokens=1000))
    assert chunks
    for chunk in chunks:
        assert "Language:" not in chunk.embedding_text
        assert chunk.metadata["language"] is None


# --- 2. Evaluator version 2 ----------------------------------------------------------------------


def _row(rank: int, doc: str, text: str) -> dict:
    return {
        "rank": rank,
        "chunk_id": f"{doc}-{rank}",
        "doc_name": doc,
        "document_id": doc,
        "workspace_id": "ws",
        "text": text,
        "parent_id": f"{doc}-p",
    }


_AFTER_UPDATE = Case(
    "upd-x",
    "Which technician performed the last service?",
    Scope("alpha", project="atlas"),
    ("updated_doc", "name", "citation_source"),
    ("log",),
    expected_text_any=("Priya Nair",),
    forbidden_text_any=("Jonas Berg",),
    phase="after_update",
)
_AFTER_DELETE = Case(
    "del-x",
    "What is the ZEPHYR-9 bracket?",
    Scope("alpha", project="atlas", include_deleted=True),
    ("stale_deleted",),
    forbidden_docs=("obsolete",),
    forbidden_text_any=("ZEPHYR-9",),
    no_answer=True,
    phase="after_delete",
)
_ISOLATION = Case(
    "ws-x",
    "What force does the press apply?",
    Scope("alpha", project="atlas"),
    ("cross_workspace", "no_answer"),
    forbidden_docs=("press-spec",),
    forbidden_text_any=("950 kN",),
    no_answer=True,
)


def test_evaluator_version_is_two_and_names_the_stale_sources() -> None:
    assert EVALUATOR_VERSION == "2"
    assert stale_source_docs(_AFTER_UPDATE) == {"log"}
    assert stale_source_docs(_AFTER_DELETE) == {"obsolete"}
    assert stale_source_docs(_ISOLATION) is None


def _score(case: Case, rows: list[dict], scope_docs: set[str]) -> dict:
    return evaluate_case(case, rows, scope_workspace_id="ws", scope_document_ids=scope_docs, deleted_names={"obsolete"})


def test_updated_document_stale_text_counts_only_in_the_revised_documents_own_chunks() -> None:
    rows = [
        _row(1, "log", "TECHNICIAN\nPriya Nair"),
        _row(2, "incident", "Attendees: Ines Halvorsen, Jonas Berg (Technician)"),
    ]
    scored = _score(_AFTER_UPDATE, rows, {"log", "incident"})
    assert scored["forbidden_text_hits"] == 0, "another document's legitimate mention is not stale text"
    assert scored["forbidden_text_hits_any_document"] == 1
    assert scored["expected_text_found"] is True
    assert hard_counters([scored])["updated_doc_stale_hits"] == 0

    stale = _score(_AFTER_UPDATE, [_row(1, "log", "TECHNICIAN\nJonas Berg")], {"log"})
    assert stale["forbidden_text_hits"] == 1, "the old revision's text in the updated document is stale"
    assert hard_counters([stale])["updated_doc_stale_hits"] == 1


def test_deleted_document_stale_text_is_tied_to_the_deleted_document() -> None:
    served = _score(_AFTER_DELETE, [_row(1, "obsolete", "The ZEPHYR-9 bracket is 140 mm long")], {"obsolete"})
    assert served["forbidden_text_hits"] == 1 and served["stale_deleted_hits"] == 1
    mention = _score(_AFTER_DELETE, [_row(1, "timeline", "2026-02: ZEPHYR-9 withdrawn")], {"timeline"})
    assert mention["forbidden_text_hits"] == 0 and mention["stale_deleted_hits"] == 0
    assert mention["forbidden_text_hits_any_document"] == 1


def test_isolation_cases_keep_the_global_forbidden_text_rule() -> None:
    scored = _score(_ISOLATION, [_row(1, "atlas-notes", "the press applies 950 kN")], {"atlas-notes"})
    assert scored["forbidden_text_hits"] == 1 == scored["forbidden_text_hits_any_document"]


# --- 1. Two mandatory gates --------------------------------------------------------------------


def test_filename_gate_copies_every_threshold_of_the_frozen_gate() -> None:
    frozen = load_gate(FIXTURES / "quality-gate.json")
    filename = load_gate(FIXTURES / "quality-gate-filename.json")
    assert "title_mode" not in frozen and "evaluator_version" not in frozen, "the original gate is untouched"
    assert filename["title_mode"] == "filename"
    assert filename["evaluator_version"] == EVALUATOR_VERSION
    assert filename["dataset_version"] == frozen["dataset_version"]
    for section in ("hard", "hard_by_provider", "ranking"):
        assert filename[section] == frozen[section], f"{section}: thresholds must be identical (never lowered)"
    assert filename.get("by_tag", {}) == frozen.get("by_tag", {})


def test_gate_regime_pins_reject_a_report_from_another_regime_or_evaluator() -> None:
    gate = {"dataset_version": "d", "title_mode": "filename", "evaluator_version": "2", "hard": {}}
    report = {"dataset_version": "d", "title_mode": "benchmark", "evaluator_version": "2", "hard": {}, "provider": "hf"}
    failed = [c.name for c in evaluate_gate(report, gate).checks if not c.passed]
    assert failed == ["title_mode"]
    report["title_mode"] = "filename"
    report["evaluator_version"] = "1"
    failed = [c.name for c in evaluate_gate(report, gate).checks if not c.passed]
    assert failed == ["evaluator_version"]
    report["evaluator_version"] = "2"
    assert evaluate_gate(report, gate).passed


@pytest.fixture(scope="module")
def hash_filename_report(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return run_qualification(
        provider="hash", out_dir=tmp_path_factory.mktemp("qualification-filename"), title_mode="filename"
    )


def test_hash_run_under_filename_titles_holds_the_filename_gates_hard_section(hash_filename_report: dict) -> None:
    assert hash_filename_report["title_mode"] == "filename"
    assert hash_filename_report["evaluator_version"] == EVALUATOR_VERSION
    gate = load_gate(FIXTURES / "quality-gate-filename.json")
    result = evaluate_gate(hash_filename_report, gate)
    by_name = {c.name: c for c in result.checks}
    assert "title_mode" not in by_name and "evaluator_version" not in by_name, "pins are satisfied"
    failed = [c.name for c in result.checks if c.name.startswith("hard") and not c.passed]
    assert failed == [], f"hash run under filename titles violates hard invariants: {failed}"
    assert hash_filename_report["forbidden_text_any_document"]["hits"] >= 0


def test_report_records_the_evaluator_and_regime(hash_filename_report: dict) -> None:
    text = json.dumps({k: hash_filename_report[k] for k in ("dataset_version", "evaluator_version", "title_mode")})
    assert '"evaluator_version": "2"' in text and '"title_mode": "filename"' in text
