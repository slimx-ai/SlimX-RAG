import json
import os
import tempfile
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


def _normalize_text(text: str, *, max_chars: int | None, normalize: bool) -> str:
    t = text or ""
    if normalize:
        # NFC helps stabilize unicode forms
        t = unicodedata.normalize("NFC", t)
        # normalize newlines
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        # strip trailing whitespace (keeps semantics mostly intact)
        t = "\n".join(line.rstrip() for line in t.split("\n"))
    if max_chars is not None and max_chars > 0 and len(t) > max_chars:
        t = t[:max_chars]
    return t


def _atomic_write_lines(out_path: Path, lines: Iterable[str]) -> None:
    """Write lines to out_path atomically (temp file in target dir + os.replace).

    A crash mid-write never leaves a partial or corrupted file at out_path:
    either the previous content survives or the new content fully replaces it.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=out_path.name + ".", suffix=".tmp", dir=str(out_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line)
        os.replace(tmp_path, out_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _atomic_write_text(out_path: Path, text: str) -> None:
    """Write text to out_path atomically (temp file in target dir + os.replace)."""
    _atomic_write_lines(out_path, [text])


def _write_jsonl(docs: Iterable[Document], out_path: Path) -> None:
    _atomic_write_lines(
        out_path,
        (json.dumps({"page_content": d.page_content, "metadata": d.metadata}, ensure_ascii=False) + "\n" for d in docs),
    )


def _read_jsonl_docs(in_path: Path) -> Iterator[Document]:
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    with in_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSONL in {in_path} at line {line_no}: {e}") from e
            if not isinstance(rec, dict):
                raise ValueError(f"Malformed JSONL in {in_path} at line {line_no}: expected an object")
            yield Document(page_content=rec.get("page_content", "") or "", metadata=rec.get("metadata", {}) or {})


def _write_embeddings_jsonl(items: Iterable[Any], out_path: Path) -> None:
    _atomic_write_lines(
        out_path,
        (
            json.dumps(
                {"chunk_id": item.chunk_id, "vector": item.vector, "text": item.text, "metadata": item.metadata},
                ensure_ascii=False,
            )
            + "\n"
            for item in items
        ),
    )
