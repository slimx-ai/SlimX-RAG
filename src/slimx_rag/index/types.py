from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from slimx_rag.utils.commons import _atomic_write_text


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single vector search result."""

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, object]


@dataclass
class IndexState:
    """Tracks document versions to support incremental indexing.

    docs: doc_id -> {content_hash: str, chunk_ids: [str]}
    embed: provider/model/dim/etc used to build the index
    """

    version: int = 1
    embed: dict | None = None
    docs: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> IndexState:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("state must be a JSON object")
            st = cls()
            st.version = int(data.get("version", 1))
            st.embed = data.get("embed")
            st.docs = dict(data.get("docs", {}) or {})
            return st
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            # Never auto-reset: silently discarding state would make incremental
            # indexing skip or duplicate chunks unpredictably.
            raise RuntimeError(
                f"Corrupt index state file {path}: {e}. Fix or delete it (deleting forces a full reindex)."
            ) from e

    def save(self, path: Path) -> None:
        payload = {"version": self.version, "embed": self.embed, "docs": self.docs}
        _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
