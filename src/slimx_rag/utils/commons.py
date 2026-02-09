import hashlib
import unicodedata
from typing import Optional
from pathlib import Path
import json
from langchain_core.documents import Document
from typing import Iterable, Iterator


def _hash_text(text: str, *, digest_size: int) -> str:
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _hash_path(kb_relpath: str) -> str:
    # doc identity: stable across edits as long as path stays stable
    return _hash_text(kb_relpath, digest_size=32)


# move to utils
def _content_hash(text: str) -> str:
    # doc version fingerprint: changes when content changes (supports incremental indexing later)
    return _hash_text(text or "", digest_size=16)


def _normalize_text(text: str, *, max_chars: Optional[int], normalize: bool) -> str:
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

def _write_jsonl(docs: Iterable[Document], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs:
            rec = {"page_content": d.page_content, "metadata": d.metadata}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl_docs(in_path: Path) -> Iterator[Document]:
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield Document(page_content=rec.get("page_content", "") or "", metadata=rec.get("metadata", {}) or {})
