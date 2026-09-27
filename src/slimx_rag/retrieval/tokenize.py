"""Identifier-aware tokenization, query normalization, and coarse intent detection.

Technical identifiers must survive tokenization: ``K2.6``, ``GLM-5.1``, ``35B-A3B``,
``68.6``, ``2026-04-20``, ``256,000``, ``MLA``, ``MAX_PAYLOAD_KG``. A token is a run of
alphanumerics with optional internal separators (``. , - / _``), so version numbers, dates and
code identifiers stay whole instead of being shattered into ``k2`` + ``6`` or ``max`` + ``kg``.
"""

from __future__ import annotations

import re

# Internal separators are kept ONLY between alphanumerics, so trailing punctuation is
# dropped while "k2.6" / "35b-a3b" / "256,000" / "2026-04-20" stay intact.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.,\-/_][A-Za-z0-9]+)*")

_TEMPORAL = {
    "when",
    "date",
    "dated",
    "released",
    "release",
    "launched",
    "launch",
    "timeline",
    "year",
    "chronology",
    "history",
}


# A caller title that is an upload filename (``atlas-incident-2026-03-14.docx``) is one lexical
# token, so its words are surfaced separately for exact-identifier matching. Only the extensions
# the parsers accept count: a version-like title such as ``GLM-5.1`` or ``K2.6`` is not a filename
# and must not contribute its stem (``glm-5``) as an exact identity.
_FILENAME_EXTENSIONS = frozenset(
    {
        "pdf",
        "docx",
        "md",
        "markdown",
        "txt",
        "text",
        "rst",
        "csv",
        "json",
        "yaml",
        "yml",
        "html",
        "htm",
        "xhtml",
        "py",
        "ts",
        "tsx",
        "js",
        "jsx",
        "go",
        "rs",
        "java",
        "c",
        "cpp",
        "rb",
    }
)
_FILENAME_RE = re.compile(r"^[^\s/\\]+\.([A-Za-z0-9]{1,8})$")


def looks_like_filename(title: str) -> bool:
    """True for a bare filename with a known document/code extension and no spaces."""
    m = _FILENAME_RE.match((title or "").strip())
    return m is not None and m.group(1).lower() in _FILENAME_EXTENSIONS


def filename_words(title: str) -> list[str]:
    """Lowercased words of a filename's stem (``atlas-safety-manual.pdf`` -> atlas safety manual)."""
    stem = (title or "").strip().rsplit(".", 1)[0]
    return [w.lower() for w in re.split(r"[-_.\s]+", stem) if w]


def filename_identity_tokens(title: str) -> set[str]:
    """Identity tokens of a filename title: its alphabetic words, its stem and the full name.

    Numeric fragments (``2026``, ``03``) are excluded: a bare number in a question must never
    exact-match a date embedded in a filename.
    """
    stripped = (title or "").strip()
    stem = stripped.rsplit(".", 1)[0]
    words = {w for w in filename_words(stripped) if not w.isdigit()}
    return words | {stem.lower(), stripped.lower()}


def lexical_tokens(text: str) -> list[str]:
    """Lowercased identifier-preserving tokens for lexical matching."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def normalize_query(query: str) -> str:
    """Collapse whitespace; preserve identifiers and punctuation-in-identifiers."""
    return " ".join((query or "").split())


def _looks_like_identifier(token_lower: str, raw: str) -> bool:
    if any(c.isdigit() for c in token_lower):
        return True
    if any(sep in token_lower for sep in (".", "-", "/", ",", "_")):
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
