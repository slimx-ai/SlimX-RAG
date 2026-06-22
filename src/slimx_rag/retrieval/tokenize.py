"""Identifier-aware tokenization, query normalization, and coarse intent detection.

Technical identifiers must survive tokenization: ``K2.6``, ``GLM-5.1``, ``35B-A3B``,
``68.6``, ``2026-04-20``, ``256,000``, ``MLA``. A token is a run of alphanumerics with
optional internal separators (``. , - /``), so version numbers and dates stay whole
instead of being shattered into ``k2`` + ``6``.
"""

from __future__ import annotations

import re

# Internal separators are kept ONLY between alphanumerics, so trailing punctuation is
# dropped while "k2.6" / "35b-a3b" / "256,000" / "2026-04-20" stay intact.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.,\-/][A-Za-z0-9]+)*")

_TEMPORAL = {
    "when", "date", "dated", "released", "release", "launched", "launch",
    "timeline", "year", "chronology", "history",
}


def lexical_tokens(text: str) -> list[str]:
    """Lowercased identifier-preserving tokens for lexical matching."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def normalize_query(query: str) -> str:
    """Collapse whitespace; preserve identifiers and punctuation-in-identifiers."""
    return " ".join((query or "").split())


def _looks_like_identifier(token_lower: str, raw: str) -> bool:
    if any(c.isdigit() for c in token_lower):
        return True
    if any(sep in token_lower for sep in (".", "-", "/", ",")):
        return True
    # Short all-caps acronym (e.g. MLA, GLM) — case checked on the raw token.
    return raw.isupper() and 2 <= len(raw) <= 5


def query_identifiers(query: str) -> set[str]:
    """Lowercased tokens that look like technical identifiers (for exact-match boosting)."""
    out: set[str] = set()
    for m in _TOKEN_RE.finditer(query or ""):
        raw = m.group(0)
        low = raw.lower()
        if _looks_like_identifier(low, raw):
            out.add(low)
    return out


def query_intent(query: str) -> str:
    """Coarse intent: ``temporal`` (dates/chronology) vs ``factual`` (everything else)."""
    return "temporal" if set(lexical_tokens(query)) & _TEMPORAL else "factual"
