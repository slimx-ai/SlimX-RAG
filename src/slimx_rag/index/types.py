from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


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
    embed: Optional[dict] = None
    docs: Dict[str, dict] = None

    def __post_init__(self) -> None:
        # Avoid the mutable-default pitfall: allocate a new dict per instance.
        if self.docs is None:
            self.docs = {}

    @classmethod
    def load(cls, path: Path) -> "IndexState":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        st = cls()
        st.version = int(data.get("version", 1))
        st.embed = data.get("embed")
        st.docs = dict(data.get("docs", {}) or {})
        return st

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": self.version, "embed": self.embed, "docs": self.docs}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
