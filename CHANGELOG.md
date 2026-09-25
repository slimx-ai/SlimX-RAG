# Changelog

## 0.3.0 — unreleased (ControlRoom qualification, 2026-09-25)

Audit record: `docs/reviews/controlroom-qualification-2026-09-25.md`. Indexes shaped by earlier
releases report `index_signature_mismatch` and must be rebuilt (`index-shaping-v2`).

### Changed (retrieval contract)

- `POST /api/index` parses posted text with the native text parser and chunks it with the same
  structure/token-aware chunker as `/api/index/file`; every chunk carries parent identity,
  `page_type`, `section`, `display_text` and embeds the identity prefix. The text-mode
  `document_pipeline` receipt now reports `chunker: "structured-token"`,
  `parser: {"name": "native-text", ...}` and the `file_chunk_config_fingerprint`.
- Parent grouping collapses field views of one entity only; consecutive prose passages of one
  page/section are distinct parents, so long documents contribute several passages.
- The requested `top_k` is the parent budget (`final_parents` is a floor, no longer a ceiling).
- Heading-only parents (a title directly followed by the first heading) emit no chunk.
- Lexical (BM25) candidates are selected inside the retrieval scope; `trace.lexical_candidates`
  counts in-scope candidates only.
- Identifier tokens keep `_` (`MAX_PAYLOAD_KG` stays whole); equal fused scores are broken by
  the dense rank, then chunk id.
- `/api/ask` and `/api/eval/run` retrieve through the same hybrid path (and citation labels) as
  `/api/retrieve` on enumerable backends.
- Citation labels include the section whenever it locates the passage and differs from the
  document title; a caller-supplied `title` wins over an inferred heading in every parser.
- `GET /api/documents/{id}/chunks` returns the reader text (`display_text`) and the stored
  `section` only.

### Changed (service boundary; fail closed)

- `workspace_id`/`document_id` must be non-empty, without `/` or control characters (422).
- `/api/retrieve` and `/api/ask`: empty `workspace_id`, empty `document_ids` or empty entries
  are 422; `RAG_REQUIRE_WORKSPACE_SCOPE=1` refuses retrieval without a workspace (400
  `workspace_scope_required`); `question`, `top_k` and `document_ids` are bounded.
- Binary payloads and HTML are rejected by the text/markdown/code parsers (422
  `parse_failed: UnsupportedDocumentError`) so hosts fall back to their own extraction.
- Invalid persisted state, unreadable receipts, dimension mismatches and embedder
  initialization failures return the structured reasons `/ready` reports (503/409) instead
  of bare 500s.
- `/api/ask` and `/api/eval/run` honour a caller `model` only with `RAG_ALLOW_MODEL_OVERRIDE=1`;
  eval datasets must live under `RAG_EVAL_DATASET_DIR` (default `examples`).
- Caller metadata on `/api/index` cannot set identity, locator or ranking fields.
- `DELETE /api/documents/{id}` and document replacement sweep every stored chunk tagged with the
  document's `doc_id` in addition to the bookkept ids (`swept_chunks` reported), so a lost
  state commit can never leave deleted or superseded content retrievable.

### Added

- `RAG_HF_REVISION` pins the Hugging Face model commit; `RAG_EMBED_QUERY_PREFIX` /
  `RAG_EMBED_DOCUMENT_PREFIX` support asymmetric models.
- ControlRoom qualification benchmark (`slimx_rag.eval.qualification`,
  `examples/controlroom_qualification/`) with a frozen quality gate.

## 0.2.8 — 2026-07-12

### Added

- Authenticated `POST /api/admin/index/reset` maintenance contract for explicit local/FAISS corpus reset without changing embedding settings.
- Exact destructive confirmation, required instance/fingerprint compare-and-swap preconditions, structured failure details, and previous/new signature provenance.
- Readiness diagnostics for invalid state/receipts and receipt-instance mismatch recovery.

### Fixed

- Text and file indexing now reject work that crossed an index-generation reset, including concurrent first-writer races.
- Pure index reset preserves the existing embedding override byte-for-byte and rolls back local artifacts transactionally.
- Every `/ready` response now includes `auth_enabled` and `engine_version`.

