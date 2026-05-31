# Ingest Module

The ingest module loads local knowledge-base files and attaches deterministic metadata for downstream RAG stages.

Supported by default:

- Markdown and text files.
- Root-level files and nested folders.
- Optional PDF and DOCX loading with the `doc` extra.

Public API:

```python
from pathlib import Path
from slimx_rag.ingest.loader import fetch_documents
from slimx_rag.settings import IndexingPipelineSettings

settings = IndexingPipelineSettings(kb_dir=Path("examples/tiny_demo/knowledge-base"))
docs = fetch_documents(settings=settings)
```

Guaranteed metadata:

- `source`
- `title`
- `kb_relpath`
- `file_ext`
- `doc_id`
- `doc_type`
- `content_hash`
- `content_len`

`doc_type` is derived from the first path segment by default. Configure `IngestSettings(doc_type_mode="none")` to disable folder-derived document types.
