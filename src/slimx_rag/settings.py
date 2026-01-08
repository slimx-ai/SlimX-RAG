from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Settings:
    """
    Application settings.

    kb_dir:
      Path to the knowledge base folder. Defaults to <repo-root>/knowledge-base
      assuming a standard "src layout":
        repo/
          knowledge-base/
          src/slimx_assistant/...
    """
        
    kb_dir: Path
    glob: str = "**/*.md"

    # How to infer doc_type:
    # - "subdir": doc_type is drived from the frst folder(s) under kb_dir
    # - "none": do not add doc_type

    doc_type_mode: Literal["subdir", "none"] = "subdir"
    doc_type_depth: int = 1  # number of path parts to use for doc_type when doc_type_mode="subdir"

    @staticmethod
    def default() -> Settings:
        root_dir = Path(__file__).resolve().parents[2] # TODO: improve detection of repo root
        kb_dir = root_dir / "knowledge-base"
        return Settings(kb_dir=kb_dir)