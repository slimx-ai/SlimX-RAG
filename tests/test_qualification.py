"""ControlRoom qualification benchmark: corpus pinning and the hard isolation/lifecycle invariants.

Runs the real HTTP service (TestClient) with the deterministic ``hash`` embedder, so this
is fast and offline. Ranking quality with the real CPU model is measured by the evidence
runs (``python -m slimx_rag.eval.qualification --provider hf``), not asserted here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slimx_rag.eval.qualification import (
    CASES,
    DATASET_VERSION,
    build_corpus,
    corpus_manifest,
    evaluate_gate,
    load_gate,
    run_qualification,
)
from slimx_rag.eval.qualification.corpus import WORKSPACES, document_id_for

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "examples" / "controlroom_qualification"


def test_corpus_matches_committed_manifest() -> None:
    committed = json.loads((FIXTURES / "MANIFEST.json").read_text(encoding="utf-8"))
    assert corpus_manifest(build_corpus()) == committed, (
        "the generated corpus differs from examples/controlroom_qualification/MANIFEST.json; "
        "bump DATASET_VERSION and regenerate the manifest deliberately"
    )
    assert committed["dataset_version"] == DATASET_VERSION


def test_gold_cases_reference_only_corpus_documents() -> None:
    names = {doc.name for doc in build_corpus().docs}
    ids = [case.id for case in CASES]
    assert len(ids) == len(set(ids))
    for case in CASES:
        for name in case.expected_docs + case.forbidden_docs + (case.scope.document_ids or ()):
            assert name in names, f"{case.id} references unknown document {name}"
        assert case.scope.workspace in WORKSPACES
        assert case.phase in {"main", "after_update", "after_delete"}
        if case.expected_page is not None or case.expected_section is not None:
            assert "citation_locator" in case.tags or case.expected_section, case.id
    assert document_id_for("atlas-glossary") != document_id_for("borealis-glossary")


@pytest.fixture(scope="module")
def hash_report(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return run_qualification(provider="hash", out_dir=tmp_path_factory.mktemp("qualification"))


def test_hash_run_holds_every_hard_invariant(hash_report: dict) -> None:
    hard = hash_report["hard"]
    assert hash_report["auth_rejects_missing_token"] is True
    assert hard["index_failures"] == 0 and hard["retrieve_failures"] == 0
    assert hard["cross_workspace_leaks"] == 0
    assert hard["cross_scope_leaks"] == 0
    assert hard["cross_project_leaks"] == 0
    assert hard["forbidden_document_leaks"] == 0
    assert hard["stale_deleted_hits"] == 0
    assert hard["updated_doc_stale_hits"] == 0
    assert hard["forbidden_text_hits"] == 0
    assert hard["unstable_cases"] == 0
    assert hard["restart_inconsistent"] == 0
    assert hard["duplicate_chunks"] == 0
    assert hard["chunks_listed_after_delete"] == 0
    assert hash_report["lifecycle"]["ready_after_restart"] is True
    assert hash_report["lifecycle"]["deleted_chunks_total"] > 0


def test_hash_run_report_shape_and_frozen_gate_hard_section(hash_report: dict) -> None:
    assert hash_report["dataset_version"] == DATASET_VERSION
    assert hash_report["index"]["documents"] == len(build_corpus().docs)
    assert hash_report["aggregate"]["cases_scored"] > 50
    assert set(hash_report["by_tag"]) >= {"cross_workspace", "cross_project", "stale_deleted", "citation_locator"}
    gate = load_gate(FIXTURES / "quality-gate.json")
    result = evaluate_gate(hash_report, gate)
    hard_checks = [check for check in result.checks if check.name.startswith("hard.")]
    assert hard_checks, "the frozen gate must carry hard invariants"
    failed = [check.name for check in hard_checks if not check.passed]
    assert failed == [], f"hash run violates frozen hard invariants: {failed}"
