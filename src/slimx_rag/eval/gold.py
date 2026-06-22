"""Deterministic gold regression dataset: a page-structured 'LLM Architecture Gallery'.

The gallery has a timeline/index page, three closely-related fact-sheet pages (Kimi
K2.5/K2.6/K2.7 with similar architecture values but distinct titles + key details), and a
glossary page that acts as retrieval noise. Gold cases mirror the reference questions.
Nothing here is hard-coded into production parsing or ranking logic — it lives only in eval.
"""

from __future__ import annotations

from dataclasses import dataclass

from slimx_rag.document.model import ParsedDocument, ParsedPage
from slimx_rag.document.structure import structure_block

_ARCH = "SCALE\n1T total, 32B active\nCONTEXT TOKENS\n256,000\nATTENTION\n61-MLA"

# Per-page text (already de-headered/footered — the parser's edge stripping is tested
# separately in test_document_parsers). The first line of each page is its title.
GALLERY_PAGES: list[str] = [
    # p1 — timeline / index
    "Model Index\nKimi K2.5 2026-01-10\nKimi K2.6 2026-02-20\nKimi K2.7 2026-03-30\n"
    "GLM-5.1 2026-04-20",
    # p2 — Kimi K2.5 fact sheet
    f"Kimi K2.5\n{_ARCH}\nCONTEXT WINDOW\n128,000\nKEY DETAIL\n"
    "Introduces the 1T 32B-active 61-MLA line for long-context reasoning.",
    # p3 — Kimi K2.6 fact sheet
    f"Kimi K2.6\n{_ARCH}\nKEY DETAIL\n"
    "Focuses on coding-agent workflows; uses the same text architecture as Kimi K2.5.",
    # p4 — Kimi K2.7 fact sheet
    f"Kimi K2.7\n{_ARCH}\nKEY DETAIL\n"
    "Extends context to 512,000 tokens using the same 1T 32B-active 61-MLA architecture.",
    # p5 — glossary (noise)
    "Glossary\nMLA means multi-head latent attention.\nMoE means mixture of experts.",
]

DOC_TITLE = "LLM Architecture Gallery"


@dataclass(frozen=True, slots=True)
class GoldCase:
    question: str
    expected_entity: str | None = None  # single-parent answer, e.g. "Kimi K2.6"
    expected_page: int | None = None
    expected_entities: tuple[str, ...] = ()  # comparison: all required parents
    expected_section: str | None = None
    should_have_answer: bool = True
    intent: str = "factual"  # factual | temporal

    @property
    def required(self) -> set[str]:
        if self.expected_entities:
            return set(self.expected_entities)
        return {self.expected_entity} if self.expected_entity else set()


GOLD_CASES: list[GoldCase] = [
    GoldCase(
        "Compare the text architectures of Kimi K2.5, Kimi K2.6, and Kimi K2.7",
        expected_entities=("Kimi K2.5", "Kimi K2.6", "Kimi K2.7"),
    ),
    GoldCase("What is the key detail for Kimi K2.6?", "Kimi K2.6", 3, expected_section="KEY DETAIL"),
    GoldCase("When was Kimi K2.6 released?", "Model Index", 1, intent="temporal"),
    GoldCase("Which Kimi version focuses on coding-agent workflows?", "Kimi K2.6", 3),
    GoldCase(
        "Which versions use the same 1T 32B-active 61-MLA architecture?",
        expected_entities=("Kimi K2.5", "Kimi K2.6", "Kimi K2.7"),
    ),
    GoldCase("What is the context window of Kimi K2.5?", "Kimi K2.5", 2),
    GoldCase("What is the parameter count of Llama 4 Maverick?", should_have_answer=False),
]


def build_parsed_gallery() -> ParsedDocument:
    """Build the page-structured ParsedDocument the structured configs index."""
    pages = []
    for i, text in enumerate(GALLERY_PAGES, start=1):
        elements, title, page_type = structure_block(
            text, id_prefix=f"gallery#p{i}", page_number=i
        )
        pages.append(
            ParsedPage(
                page_number=i,
                elements=tuple(elements),
                title=title,
                page_type=page_type,
                text=text,
            )
        )
    return ParsedDocument(
        document_id="gallery",
        title=DOC_TITLE,
        source_type="pdf",
        parser_name="native-pdf",
        parser_version="1",
        page_count=len(pages),
        pages=tuple(pages),
    )


def flattened_gallery_text() -> str:
    """Simulate ControlRoom's old behavior: all pages flattened into one blob."""
    return "\n\n".join(GALLERY_PAGES)
