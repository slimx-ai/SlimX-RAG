# Changelog

## 0.3.0 — 2026-09-25 (ControlRoom qualification candidate; no GitHub release yet)

Audit record: `docs/reviews/controlroom-qualification-2026-09-25.md`. Indexes shaped by earlier
releases report `index_signature_mismatch` and must be rebuilt (`index-shaping-v4`).

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

### Same-model review 2 corrections (2026-09-26, same release; `index-shaping-v4`)

- Retrieval identity keeps the document's own title next to a caller title that is an upload
  filename (ControlRoom's convention, `title = filename`): the identity prefix gains
  `Title: <own title>` (Markdown H1, DOCX title, a text file's first line), and exact-identifier
  matching sees the own title plus the filename's alphabetic words, stem and full name (a
  `Name:` prefix line was measured and rejected; only titles with a known document or code
  extension count as filenames). Before this, a filename title erased the heading identity
  ("Incident Report IR-2026-031" became `atlas-incident-2026-03-14.docx`) and the frozen gate held
  only under the benchmark's human titles (RAG-AUD-039). `index-shaping-v4`; older indexes rebuild.
- `POST /api/admin/embedding` keeps the pinned model revision and the query/document prefixes
  (they are persisted with the override); it accepts `hf_revision` and requires it when
  `hf_model` changes on a revision-pinned deployment. Before this, a device change in the offline
  image failed with `embedder_preflight_failed: OSError` because the merged settings dropped the
  revision (RAG-AUD-040).
- Qdrant backend: chunk ids (64-hex digests) map to deterministic UUIDv5 point ids and ride in
  the payload; Qdrant accepts only UUID or integer ids, so the backend could not upsert at all.
  Tested against `QdrantClient(":memory:")`; the unit-test fake now enforces the id rule
  (RAG-AUD-041; the 0.3.0 note that RAG-AUD-038 "corrected" the backend was incomplete).
- Scope filtering treats missing or non-string `workspace_id` / `document_id` metadata as out of
  scope; `str(None)` no longer matches the literal scope `"None"` (RAG-AUD-042).
- `POST /api/index` bounds `text` (`RAG_MAX_TEXT_CHARS`, default the file limit), rejects blank
  text (422) and applies the element cap (413); a blank post used to replace a document with
  nothing (RAG-AUD-046).
- Text decoding keeps UTF-8 with sparse stray bytes (each becomes U+FFFD); only text with many
  invalid sequences falls back to cp1252 / Latin-1 (RAG-AUD-048).
- Qualification benchmark: `--hf-revision` (default the image's pinned commit `c9745ed1`, recorded
  in the report; earlier reports recorded the cache's `refs/main`) and `--title-mode
  benchmark|filename` (ControlRoom's title convention as a second regime; gate, gold and corpus
  unchanged) (RAG-AUD-044).
- Same-model review 3 residuals (RAG-AUD-050..054): only titles with a known document or code
  extension count as filenames (a version-like `GLM-5.1` never contributes `glm-5` as an exact
  identity); `hf_revision` and `RAG_HF_REVISION` must be exact 40-hex commits; the admin route
  persists a revision only when the request supplied one, so the image's `RAG_HF_REVISION`
  governs otherwise (recovery from a stale persisted revision: delete `embed_override.json`); the
  UTF-8 stray-byte ratio is measured against the non-ASCII bytes so sparse legacy accents survive.
- Known limitation, documented: BM25 document-frequency statistics span every workspace, so
  another tenant's text moves in-scope lexical scores and a scoped caller can infer term frequency
  across tenants; candidates and text never cross (RAG-AUD-043).

### Field-addressed fact sheets (2026-09-25, same release; `index-shaping-v3`)

- A heading-less fact sheet (a title line plus at least two labelled fields and no heading of its
  own, e.g. a maintenance record posted as text) that fits the token budget with its longest label
  prefix is addressed by its fields: one retrieval unit per field, embedded as the identity prefix
  plus that field alone, plus one unit for the remaining elements minus the sheet's own title line
  (recognised from the text itself, so a caller title such as the upload filename does not create
  a title-only unit). Every unit displays the whole sheet, shares one retrieval parent and is cited
  by its field label (`[Atlas Maintenance Log, LAST SERVICE]`; a heading-less PDF fact page is now
  cited by page and label, `[gallery, p. 2, KEY DETAIL]`, instead of page and page title). One
  mean-pooled vector over several unrelated fields answered a question about one of them poorly
  (ControlRoom qualification: cosine 0.48 packed vs 0.62 for the matching field; the record fell
  outside the dense top-30 while BM25 ranked it first). Fact sheets under a heading, Markdown and
  DOCX sections and every heading-based locator are unchanged; a sheet that does not fit the budget
  keeps the packing path. On the hybrid path (local backend) parent grouping never shows the same
  passage twice (`parent_reason: dropped_duplicate_text`); FAISS/Qdrant/pgvector run dense-only
  without grouping and return the units individually as stored. `GET /api/documents/{id}/chunks`
  lists one entry per unit (each showing the whole sheet, differing by `section`) and
  `chunk_count` counts units. Frozen ControlRoom gate with the retained model: FAIL (2 of 26) →
  PASS (26 of 26); hit@1 0.921 → 0.936, hit@5 0.968 → 1.000, MRR 0.937 → 0.954, top-1 expected
  source 0.913 → 0.935. Indexes shaped by v1 or v2 report `index_signature_mismatch` and are
  rebuilt.

### Author-side review corrections (2026-09-25, same release)

- Plain text: blank lines are paragraph boundaries again in `structure_block`, so a multi-
  paragraph text document yields PARAGRAPH elements instead of one block force-split at word
  positions.
- Text sources decode as UTF-8 (BOM tolerated), else Windows-1252, else Latin-1; accented legacy
  text is no longer mistaken for binary.
- `document_ids` requires `workspace_id` on every endpoint (422): the same document id may exist
  in several workspaces.
- `/api/eval/run` accepts `workspace_id`/`document_ids`, honours `RAG_REQUIRE_WORKSPACE_SCOPE`,
  returns 422 for unreadable or malformed datasets (blank/oversized questions), and `/api/retrieve`
  metadata carries `kb_relpath` so `hit@k` is computed from real sources on the hybrid path.
- Embedding failures during indexing return 503 `embedding_failed` (`retryable: true`); a query
  dimension mismatch maps to `embedding_dim_mismatch`.
- `GET /api/documents/{id}/chunks` is authoritative on enumerable backends (chunks the
  bookkeeping lost are listed too, in ordinal order), matching the delete sweep.
- Ancestor headings ride in the identity prefix (`Path: A > B`) and in exact-identifier matching,
  so an identifier that lives only in a heading-only parent stays reachable.
- Build: the base-image label reflects `ARG PYTHON_IMAGE`; the BuildKit syntax directive is gone;
  `uv lock --check` guards the image build and the candidate workflow; CI pins uv 0.9.18 and
  lock-verifies the service extra set on one leg; the GPU Dockerfile reinstalls CUDA torch and
  asserts it (still unverified, not part of the CPU qualification).
- Qdrant backend: queries go through the Universal Query API (`query_points`).
  `QdrantClient.search`, which qdrant-client removed in 1.15, left the locked client (1.19.1)
  unable to query at all; the `qdrant` extra now requires qdrant-client >= 1.10. Surfaced by the
  new CI leg that lock-verifies the service image's extra set (RAG-AUD-038).

### Added

- `/api/retrieve` responses carry `vector_backend`.
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

