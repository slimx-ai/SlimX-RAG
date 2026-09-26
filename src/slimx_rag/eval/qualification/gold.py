"""Gold cases for the ControlRoom qualification benchmark (versioned with the corpus).

Every case names documents by their corpus ``name``; the runner resolves names to the
UUID-shaped ``document_id`` values ControlRoom would send. A *scope* is expressed the way
ControlRoom expresses it: a ``workspace_id`` plus an explicit ``document_ids`` list derived
from a project (or an explicit restriction). ``phase`` places a case in the document
lifecycle: ``main`` (after the initial index), ``after_update`` (after the maintenance log
was re-indexed with new content) or ``after_delete`` (after the obsolete spec was deleted).

Tags drive the per-category report and the gate:

- ``citation_source``: the question unambiguously targets one document, so the top-1
  result MUST come from it (gate: 0 wrong sources);
- ``citation_locator``: the best result from the expected document MUST carry the
  expected page (PDF) or section (Markdown/DOCX) (gate: 0 wrong locators);
- ``exact_identifier``, ``acronym``, ``filename``, ``name``, ``date``, ``number``,
  ``paraphrase``, ``semantic``, ``timeline``, ``duplicated_terms``, ``near_duplicate``,
  ``distractor``, ``multi_doc``, ``long_doc``, ``pdf_page``, ``docx_section``,
  ``md_section``, ``code``, ``single_source``, ``conflicting``, ``ranking_tie``: quality
  categories reported separately;
- ``cross_project``, ``cross_workspace``, ``doc_restriction``, ``stale_deleted``,
  ``updated_doc``: isolation and lifecycle categories whose leak counters gate at zero;
- ``no_answer``: nothing in scope answers the question.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Scope:
    workspace: str  # corpus workspace key: alpha | beta
    project: str | None = None  # all documents of this project ...
    document_ids: tuple[str, ...] | None = None  # ... or exactly these document names
    include_deleted: bool = False  # keep deleted names in the id set (lagging host scope)


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    question: str
    scope: Scope
    tags: tuple[str, ...] = ()
    expected_docs: tuple[str, ...] = ()
    expected_page: int | None = None
    expected_section: str | None = None  # substring of section or section_path
    forbidden_docs: tuple[str, ...] = ()
    expected_text_any: tuple[str, ...] = ()  # at least one returned text must contain one of these
    forbidden_text_any: tuple[str, ...] = ()  # no returned text may contain any of these
    no_answer: bool = False
    phase: str = "main"
    notes: str = ""

    @property
    def answerable(self) -> bool:
        return bool(self.expected_docs) and not self.no_answer


_P1 = Scope("alpha", project="atlas")
_P2 = Scope("alpha", project="borealis")
_P3 = Scope("beta", project="cascade")
_P1_LAGGING = Scope("alpha", project="atlas", include_deleted=True)

CASES: list[Case] = [
    # --- exact identifiers / part numbers ------------------------------------------
    Case(
        "id-001",
        "What is the rated dynamic load of PX-4471-B?",
        _P1,
        ("exact_identifier", "md_section", "citation_source", "citation_locator", "number"),
        ("atlas-parts-catalog",),
        expected_section="PX-4471-B carriage bearing",
        forbidden_docs=("cascade-parts-catalog", "borealis-parts-catalog"),
    ),
    Case(
        "id-002",
        "What is the replacement interval of the PX-4471-C bearing?",
        _P1,
        ("exact_identifier", "md_section", "citation_source", "citation_locator", "near_duplicate"),
        ("atlas-parts-catalog",),
        expected_section="PX-4471-C",
    ),
    Case(
        "id-003",
        "What is the gear ratio of GX-210?",
        _P1,
        ("exact_identifier", "md_section", "citation_source", "citation_locator", "distractor"),
        ("atlas-parts-catalog",),
        expected_section="GX-210 servo gearbox",
    ),
    Case(
        "id-004",
        "Which line uses the GX-2100 gearbox?",
        _P1,
        ("exact_identifier", "md_section", "citation_source", "citation_locator", "distractor"),
        ("atlas-parts-catalog",),
        expected_section="GX-2100",
    ),
    Case(
        "id-005",
        "What does incident report IR-2026-031 describe?",
        _P1,
        ("exact_identifier", "docx_section", "citation_source"),
        ("atlas-incident-2026-03-14",),
        forbidden_docs=("atlas-incident-2026-05-02",),
    ),
    Case(
        "id-006",
        "When does supply agreement HR-2026-0042 expire?",
        _P1,
        ("exact_identifier", "md_section", "date", "citation_source", "citation_locator"),
        ("atlas-vendor-contract",),
        expected_section="Term",
    ),
    Case(
        "id-007",
        "Where must lockout tag LT-0917 be applied?",
        _P1,
        ("exact_identifier", "pdf_page", "long_doc", "citation_source", "citation_locator"),
        ("atlas-safety-manual",),
        expected_page=17,
        forbidden_docs=("cascade-press-spec", "cascade-notes"),
        notes="The incident report also mentions LT-0917; the locator gate accepts only the manual page.",
    ),
    Case(
        "id-008",
        "What is the code FB-77?",
        _P1,
        ("exact_identifier", "duplicated_terms", "citation_source"),
        ("atlas-shift-handover",),
    ),
    Case(
        "id-009",
        "What is part CC-88?",
        _P1,
        ("exact_identifier", "md_section", "citation_source", "citation_locator"),
        ("atlas-parts-catalog",),
        expected_section="Cable chain CC-88",
    ),
    # --- acronyms ---------------------------------------------------------------------
    Case(
        "acr-010",
        "What is the Atlas OEE target?",
        _P1,
        ("acronym", "md_section", "citation_source", "citation_locator", "cross_project"),
        ("atlas-glossary",),
        expected_section="OEE",
        forbidden_docs=("borealis-glossary", "cascade-glossary"),
    ),
    Case(
        "acr-011",
        "What is MTBF and what is the Atlas target?",
        _P1,
        ("acronym", "md_section", "citation_source", "citation_locator"),
        ("atlas-glossary",),
        expected_section="MTBF",
        forbidden_docs=("borealis-glossary",),
    ),
    Case(
        "acr-012",
        "What does LOTO stand for?",
        _P1,
        ("acronym", "md_section", "citation_source", "citation_locator"),
        ("atlas-glossary",),
        expected_section="LOTO",
    ),
    # --- filenames --------------------------------------------------------------------
    Case(
        "file-013",
        "Which calibration step loads calib_v3.cfg?",
        _P1,
        ("filename", "pdf_page", "citation_source", "citation_locator"),
        ("atlas-calibration-procedure",),
        expected_page=5,
    ),
    Case(
        "file-014",
        "Which Python function applies the laser offset from calib_v3.cfg?",
        _P1,
        ("filename", "code", "citation_source"),
        ("atlas-controller",),
    ),
    # --- names ------------------------------------------------------------------------
    Case(
        "name-015",
        "Who approved the corrective actions after the March incident?",
        _P1,
        ("name", "docx_section", "citation_source", "citation_locator"),
        ("atlas-incident-2026-03-14",),
        expected_section="Corrective actions",
        forbidden_docs=("borealis-incident-2026-04-19", "cascade-incident"),
    ),
    Case(
        "name-016",
        "Who is the supplier account manager at Nordlys Components?",
        _P1,
        ("name", "md_section", "citation_source", "citation_locator"),
        ("atlas-vendor-contract",),
        expected_section="Contacts",
    ),
    Case(
        "name-017",
        "Which technician performed the last Atlas service?",
        _P1,
        ("name", "citation_source", "updated_doc"),
        ("atlas-maintenance-log",),
        expected_text_any=("Jonas Berg",),
    ),
    Case(
        "name-063",
        "What did Ines Halvorsen approve?",
        _P1,
        ("name", "cross_project", "citation_source"),
        ("atlas-incident-2026-03-14",),
        forbidden_docs=("borealis-incident-2026-04-19", "borealis-notes", "cascade-incident"),
    ),
    # --- dates ------------------------------------------------------------------------
    Case(
        "date-018",
        "When was firmware 2.3.1 released?",
        _P1,
        ("date", "timeline", "multi_doc"),
        ("atlas-release-timeline", "atlas-firmware-notes"),
    ),
    Case(
        "date-019",
        "What happened on 2026-05-02?",
        _P1,
        ("date", "docx_section", "citation_source"),
        ("atlas-incident-2026-05-02",),
        forbidden_docs=("atlas-incident-2026-03-14",),
    ),
    Case(
        "date-020",
        "On which date does the Nordlys supply agreement end?",
        _P1,
        ("date", "paraphrase", "md_section", "citation_source", "citation_locator"),
        ("atlas-vendor-contract",),
        expected_section="Term",
    ),
    # --- numbers ----------------------------------------------------------------------
    Case(
        "num-021",
        "What press force was being applied when the gantry stalled in March?",
        _P1,
        ("number", "docx_section", "citation_source", "citation_locator", "cross_workspace"),
        ("atlas-incident-2026-03-14",),
        expected_section="Summary",
        forbidden_docs=("cascade-incident", "cascade-press-spec"),
    ),
    Case(
        "num-022",
        "What is the light curtain response time?",
        _P1,
        ("number", "pdf_page", "long_doc", "citation_source", "citation_locator"),
        ("atlas-safety-manual",),
        expected_page=23,
    ),
    Case(
        "num-023",
        "What laser head offset is set during calibration?",
        _P1,
        ("number", "pdf_page", "citation_source", "citation_locator"),
        ("atlas-calibration-procedure",),
        expected_page=3,
    ),
    Case(
        "num-024",
        "What is the annual value of the Nordlys contract?",
        _P1,
        ("number", "md_section", "citation_source", "citation_locator"),
        ("atlas-vendor-contract",),
        expected_section="Pricing",
    ),
    # --- paraphrases / semantic -------------------------------------------------------
    Case(
        "para-025",
        "How much weight can the gantry carry before the interlock stops it?",
        _P1,
        ("paraphrase", "multi_doc"),
        ("atlas-firmware-notes", "atlas-controller", "atlas-meeting-notes-w12", "atlas-safety-manual"),
    ),
    Case(
        "para-026",
        "What did Kwame propose about the interlock threshold?",
        _P1,
        ("paraphrase", "name", "citation_source"),
        ("atlas-meeting-notes-w12",),
    ),
    Case(
        "sem-027",
        "Why did the gantry stop in March?",
        _P1,
        ("semantic", "docx_section"),
        ("atlas-incident-2026-03-14",),
        expected_section="Root cause",
    ),
    Case(
        "sem-028",
        "How do I restart the controller after an interlock trip?",
        _P1,
        ("semantic", "md_section", "citation_source"),
        ("atlas-faq",),
        expected_section="reset the controller",
    ),
    Case(
        "sem-029",
        "What must be done before entering the robot cell for maintenance?",
        _P1,
        ("semantic", "long_doc"),
        ("atlas-safety-manual", "atlas-glossary"),
    ),
    # --- timeline ---------------------------------------------------------------------
    Case(
        "time-030",
        "List the Atlas firmware releases in order.",
        _P1,
        ("timeline", "citation_source"),
        ("atlas-release-timeline",),
    ),
    Case(
        "time-031",
        "Which firmware version added the payload-limit interlock?",
        _P1,
        ("timeline", "multi_doc"),
        ("atlas-release-timeline", "atlas-firmware-notes"),
    ),
    # --- duplicated terms / near duplicates / ranking ties ---------------------------
    Case(
        "dup-032",
        "Which code identifies the spare fuse box?",
        _P1,
        ("duplicated_terms", "semantic"),
        ("atlas-shift-handover",),
    ),
    Case(
        "near-033",
        "Which operators are on the distribution list for cell 9?",
        _P1,
        ("near_duplicate", "ranking_tie", "citation_source"),
        ("atlas-boilerplate-b",),
        forbidden_docs=("atlas-boilerplate-a",),
    ),
    Case(
        "near-034",
        "Which operators are on the distribution list for cell 4?",
        _P1,
        ("near_duplicate", "ranking_tie", "citation_source"),
        ("atlas-boilerplate-a",),
        forbidden_docs=("atlas-boilerplate-b",),
    ),
    # --- distractors / conflicting evidence ----------------------------------------
    Case(
        "dis-035",
        "What is the mounting torque specification for the GX-210 gearbox?",
        _P1,
        ("distractor", "number", "multi_doc"),
        ("atlas-parts-catalog", "atlas-calibration-procedure"),
    ),
    Case(
        "conf-036",
        "What gearbox mounting torque was measured in the May incident?",
        _P1,
        ("conflicting", "number", "docx_section", "citation_source", "citation_locator"),
        ("atlas-incident-2026-05-02",),
        expected_section="Root cause",
        forbidden_docs=("atlas-incident-2026-03-14",),
    ),
    Case(
        "conf-060",
        "What is the specified GX-210 mounting torque and what was actually measured?",
        _P1,
        ("conflicting", "multi_doc"),
        ("atlas-calibration-procedure", "atlas-incident-2026-05-02", "atlas-parts-catalog"),
    ),
    # --- multi-document ----------------------------------------------------------------
    Case(
        "multi-037",
        "Compare the rated dynamic loads of PX-4471-B and PX-4471-C.",
        _P1,
        ("multi_doc", "exact_identifier", "md_section"),
        ("atlas-parts-catalog",),
    ),
    Case(
        "multi-038",
        "Which documents mention the 120 kg payload limit?",
        _P1,
        ("multi_doc", "number"),
        ("atlas-firmware-notes", "atlas-controller", "atlas-meeting-notes-w12", "atlas-safety-manual"),
    ),
    Case(
        "multi-039",
        "What incidents occurred on the Atlas gantry in 2026?",
        _P1,
        ("multi_doc", "date"),
        ("atlas-incident-2026-03-14", "atlas-incident-2026-05-02"),
    ),
    # --- long document / PDF pages --------------------------------------------------
    Case(
        "long-040",
        "Where are the emergency stop buttons located in the Atlas cell?",
        _P1,
        ("long_doc", "pdf_page", "single_source", "citation_source", "citation_locator", "cross_project"),
        ("atlas-safety-manual",),
        expected_page=8,
        forbidden_docs=("borealis-parts-catalog", "borealis-conveyor-spec"),
    ),
    Case(
        "long-041",
        "At what noise level does hearing protection become mandatory?",
        _P1,
        ("long_doc", "pdf_page", "number", "citation_source", "citation_locator"),
        ("atlas-safety-manual",),
        expected_page=27,
    ),
    Case(
        "long-066",
        "What is the manual mode speed limit?",
        _P1,
        ("long_doc", "pdf_page", "number", "citation_source", "citation_locator"),
        ("atlas-safety-manual",),
        expected_page=10,
    ),
    # --- DOCX / Markdown sections ------------------------------------------------------
    Case(
        "docx-042",
        "What corrective actions were taken after IR-2026-031?",
        _P1,
        ("docx_section", "exact_identifier", "citation_source", "citation_locator"),
        ("atlas-incident-2026-03-14",),
        expected_section="Corrective actions",
    ),
    Case(
        "docx-067",
        "What was the root cause of the March bearing failure?",
        _P1,
        ("docx_section", "semantic", "citation_locator"),
        ("atlas-incident-2026-03-14",),
        expected_section="Root cause",
    ),
    # --- updated documents ------------------------------------------------------------
    Case(
        "upd-043a",
        "When was the Atlas gantry last serviced?",
        _P1,
        ("updated_doc", "date", "citation_source"),
        ("atlas-maintenance-log",),
        expected_text_any=("2026-04-01",),
        phase="main",
    ),
    Case(
        "upd-043b",
        "When was the Atlas gantry last serviced?",
        _P1,
        ("updated_doc", "date", "citation_source"),
        ("atlas-maintenance-log",),
        expected_text_any=("2026-06-15",),
        forbidden_text_any=("2026-04-01", "Jonas Berg"),
        phase="after_update",
        notes="The v1 content must be gone after re-indexing the same document id with new text.",
    ),
    Case(
        "upd-043c",
        "Which technician performed the last Atlas service?",
        _P1,
        ("updated_doc", "name", "citation_source"),
        ("atlas-maintenance-log",),
        expected_text_any=("Priya Nair",),
        forbidden_text_any=("Jonas Berg",),
        phase="after_update",
    ),
    # --- deleted documents ------------------------------------------------------------
    Case(
        "del-044a",
        "What is the ZEPHYR-9 bracket?",
        _P1,
        ("stale_deleted", "exact_identifier", "citation_source"),
        ("atlas-obsolete-spec",),
        phase="main",
        notes="Proves the document was retrievable before deletion.",
    ),
    Case(
        "del-044b",
        "What is the ZEPHYR-9 bracket?",
        _P1_LAGGING,
        ("stale_deleted",),
        forbidden_docs=("atlas-obsolete-spec",),
        forbidden_text_any=("ZEPHYR-9",),
        no_answer=True,
        phase="after_delete",
        notes="Host scope still lists the deleted id (lagging host); the service must not serve stale chunks.",
    ),
    Case(
        "del-044c",
        "How long is the ZEPHYR-9 bracket?",
        _P1,
        ("stale_deleted",),
        forbidden_docs=("atlas-obsolete-spec",),
        forbidden_text_any=("ZEPHYR-9",),
        no_answer=True,
        phase="after_delete",
    ),
    # --- same workspace, different project --------------------------------------------
    Case(
        "proj-046",
        "What is the rated dynamic load of PX-4471-B?",
        _P2,
        ("cross_project", "exact_identifier"),
        ("borealis-parts-catalog",),
        forbidden_docs=("atlas-parts-catalog", "cascade-parts-catalog"),
        notes="Borealis has its own compatibility note about PX-4471-B; the Atlas catalog must not leak.",
    ),
    Case(
        "proj-047",
        "What is the OEE target?",
        _P2,
        ("cross_project", "acronym", "citation_source"),
        ("borealis-glossary",),
        forbidden_docs=("atlas-glossary", "cascade-glossary"),
    ),
    Case(
        "proj-048",
        "What is the Borealis belt speed?",
        _P1,
        ("cross_project", "no_answer"),
        forbidden_docs=("borealis-conveyor-spec", "borealis-notes"),
        no_answer=True,
    ),
    Case(
        "proj-049",
        "Who approved the corrective actions for the belt drift incident?",
        _P2,
        ("cross_project", "name", "citation_source"),
        ("borealis-incident-2026-04-19",),
        forbidden_docs=("atlas-incident-2026-03-14", "atlas-incident-2026-05-02"),
    ),
    Case(
        "proj-059",
        "What is the head drive motor rating?",
        _P1,
        ("cross_project", "no_answer", "number"),
        forbidden_docs=("borealis-conveyor-spec", "borealis-notes"),
        no_answer=True,
    ),
    # --- different workspace -------------------------------------------------------------
    Case(
        "ws-050",
        "How many PX-4471-B units are kept in the shared spares cage?",
        _P1,
        ("cross_workspace", "exact_identifier"),
        ("atlas-parts-catalog",),
        forbidden_docs=("cascade-parts-catalog",),
        forbidden_text_any=("spares cage",),
        notes="Only the beta workspace knows the spares cage; the strongest lexical lure.",
    ),
    Case(
        "ws-051",
        "What is the rated dynamic load of PX-4471-B?",
        _P3,
        ("cross_workspace", "exact_identifier", "citation_source"),
        ("cascade-parts-catalog",),
        forbidden_docs=("atlas-parts-catalog", "borealis-parts-catalog"),
    ),
    Case(
        "ws-052",
        "Where must lockout tag LT-0917 be applied?",
        _P3,
        ("cross_workspace", "exact_identifier"),
        ("cascade-press-spec", "cascade-notes"),
        forbidden_docs=("atlas-safety-manual", "atlas-incident-2026-03-14"),
    ),
    Case(
        "ws-053",
        "What force does the Cascade press apply?",
        _P1,
        ("cross_workspace", "no_answer", "number"),
        forbidden_docs=("cascade-press-spec", "cascade-notes", "cascade-incident"),
        forbidden_text_any=("950 kN",),
        no_answer=True,
    ),
    Case(
        "ws-068",
        "What is the OEE target?",
        _P3,
        ("cross_workspace", "acronym", "citation_source"),
        ("cascade-glossary",),
        forbidden_docs=("atlas-glossary", "borealis-glossary"),
    ),
    # --- explicit document restriction -------------------------------------------------
    Case(
        "doc-054",
        "What is the Atlas OEE target?",
        Scope("alpha", document_ids=("atlas-glossary",)),
        ("doc_restriction", "acronym", "citation_source"),
        ("atlas-glossary",),
    ),
    Case(
        "doc-055",
        "What gearbox mounting torque was measured in the May incident?",
        Scope("alpha", document_ids=("atlas-incident-2026-03-14",)),
        ("doc_restriction", "no_answer"),
        forbidden_docs=("atlas-incident-2026-05-02", "atlas-calibration-procedure", "atlas-parts-catalog"),
        forbidden_text_any=("48 N-m",),
        no_answer=True,
    ),
    Case(
        "doc-056",
        "Which Python function applies the laser offset?",
        Scope("alpha", document_ids=("atlas-calibration-procedure",)),
        ("doc_restriction",),
        ("atlas-calibration-procedure",),
        forbidden_docs=("atlas-controller",),
    ),
    # --- no-answer questions ---------------------------------------------------------
    Case(
        "na-057",
        "What is the warranty period for the Atlas gantry?",
        _P1,
        ("no_answer",),
        no_answer=True,
    ),
    Case(
        "na-058",
        "Who is the chief executive of Halvorsen Robotics?",
        _P1,
        ("no_answer",),
        no_answer=True,
    ),
    # --- single-source facts / code -------------------------------------------------
    Case(
        "single-061",
        "What is the checksum of firmware 2.3.1?",
        _P1,
        ("single_source", "citation_source"),
        ("atlas-firmware-notes",),
        forbidden_docs=("atlas-release-timeline",),
    ),
    Case(
        "single-062",
        "Which bays have emergency stops?",
        _P1,
        ("single_source", "pdf_page", "long_doc", "citation_locator"),
        ("atlas-safety-manual",),
        expected_page=8,
    ),
    Case(
        "code-064",
        "What value is MAX_PAYLOAD_KG set to in the controller?",
        _P1,
        ("code", "exact_identifier", "citation_source"),
        ("atlas-controller",),
    ),
    Case(
        "code-065",
        "Which exception does the controller raise when the interlock halts motion?",
        _P1,
        ("code", "semantic", "citation_source"),
        ("atlas-controller",),
    ),
]


def cases_for_phase(phase: str) -> list[Case]:
    return [case for case in CASES if case.phase == phase]


def all_tags() -> list[str]:
    tags: set[str] = set()
    for case in CASES:
        tags.update(case.tags)
    return sorted(tags)
