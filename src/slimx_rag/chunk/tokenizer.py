"""Token counting abstraction used by the structure-aware chunker.

The chunker enforces hard limits in *tokens*, but it must stay testable and offline, so
it depends on this small ``TokenCounter`` Protocol rather than on a specific model. The
real embedding-tokenizer-backed counter is wired in at the embedder boundary (Stage 4);
``HeuristicTokenCounter`` is a deterministic, dependency-free fallback (and the default
in tests).
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens for a piece of text and reports the model's hard token limit."""

    name: str
    max_tokens: int

    def count(self, text: str) -> int:
        """Return the number of tokens ``text`` would occupy."""
        ...


class HeuristicTokenCounter:
    """Deterministic, offline token estimate (words + standalone punctuation).

    Not exact for any specific model, but stable and never silently truncating — good
    enough to bound chunk sizes when a real tokenizer is unavailable.
    """

    name = "heuristic"

    def __init__(self, max_tokens: int = 256) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self.max_tokens = max_tokens

    def count(self, text: str) -> int:
        return len(_TOKEN_RE.findall(text or ""))
