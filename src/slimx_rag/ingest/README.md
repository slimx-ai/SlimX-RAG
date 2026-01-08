# Ingest Module

The ingest module is responsible for **loading raw documents** from a local
knowledge base and attaching **stable, minimal metadata** that later RAG stages
(chunking, embedding, indexing, retrieval) can reliably depend on.

The module is intentionally designed to be:
- deterministic
- easy to test
- dataset-agnostic
- free of side effects (no printing, no global state)

---

## Responsibilities

- Scan a knowledge base directory on disk
- Load Markdown files (`**/*.md` by default)
- Return `List[langchain_core.documents.Document]`
- Attach baseline metadata for every document:
  - `kb_relpath` – path relative to the knowledge base root
  - `file_ext` – file extension (e.g. `md`)
  - `doc_id` – stable short hash derived from `kb_relpath`
- Optionally attach semantic metadata:
  - `doc_type` – derived from directory structure (configurable)

---

## Knowledge Base Layout

By default, the module works well with a structure like:

```text
knowledge-base/
  company/
    about.md
    careers.md
  products/
    product_a.md
    product_b.md
````

With default settings:

* `doc_type = "company"` or `"products"`
* `kb_relpath = "company/about.md"`

However, the module also supports **flat or arbitrary directory layouts**
by disabling `doc_type` derivation.

---

## Public API

```python
from slimx_rag.ingest import fetch_documents

docs = fetch_documents()
```

### Function signature

```python
fetch_documents(
    settings: Settings | None = None,
    *,
    kb_dir: Path | None = None,
    glob: str | None = None,
) -> list[Document]
```

* `settings` – application settings (recommended)
* `kb_dir` – optional override for the knowledge-base path
* `glob` – file glob pattern (default: `"**/*.md"`)

---

## Configuration

Configuration is defined via the `Settings` dataclass:

```python
from slimx_rag.settings import Settings
from pathlib import Path

settings = Settings(
    kb_dir=Path("./knowledge-base"),
    glob="**/*.md",
    doc_type_mode="subdir",   # or "none"
    doc_type_depth=1,
)
```

### Settings fields

| Field            | Type      | Description                                 |                           |
| ---------------- | --------- | ------------------------------------------- | ------------------------- |
| `kb_dir`         | `Path`    | Root directory of the knowledge base        |                           |
| `glob`           | `str`     | File glob pattern                           |                           |
| `doc_type_mode`  | `"subdir" | "none"`                                     | How `doc_type` is derived |
| `doc_type_depth` | `int`     | Number of path segments used for `doc_type` |                           |

### Examples

Disable `doc_type` entirely (flat datasets):

```python
settings = Settings(
    kb_dir=Path("./knowledge-base"),
    doc_type_mode="none",
)
```

Use nested folder depth for semantic grouping:

```python
# knowledge-base/finance/contracts/...
settings = Settings(
    kb_dir=Path("./knowledge-base"),
    doc_type_mode="subdir",
    doc_type_depth=2,
)
```

---

## Metadata Contract

Every returned document guarantees the following metadata:

* `kb_relpath` – relative path inside the knowledge base
* `file_ext` – file extension
* `doc_id` – stable short hash derived from the relative path

When enabled:

* `doc_type` – derived from directory structure

Downstream modules should rely only on this contract.

---

## CLI Usage

If wired in `pyproject.toml`:

```bash
slimx-ingest --kb-dir ./knowledge-base -v
```

### Flags

| Flag             | Description                       |
| ---------------- | --------------------------------- |
| `--kb-dir PATH`  | Override knowledge base directory |
| `--glob PATTERN` | Override file glob                |
| `-v`             | INFO logging                      |
| `-vv`            | DEBUG logging                     |

---

## Logging

* Uses Python logging (no prints).
* Core logic logs via module logger.
* CLI configures log level and formatting.
* Suitable for structured logging and tracing later.

---

## Testing

Unit tests validate:

* Correct document loading
* Metadata correctness (`doc_type`, `kb_relpath`, `doc_id`)
* Error handling (missing or invalid KB paths)

Run:

```bash
uv run pytest -q
```
