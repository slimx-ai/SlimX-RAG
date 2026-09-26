# SlimX-RAG ControlRoom qualification audit — 2026-09-25

**Provenance:** AUTHOR-SIDE / SAME-MODEL REVIEW — NOT INDEPENDENT REVIEW. Written by the
implementing session (Claude Fable 5.1) for the SlimX-RAG → ControlRoom qualification program.
Every finding below carries executed evidence unless marked "read-only". Independent source review
belongs to Codex or a human and has not happened at the time of writing.

## 1. Identities

| Item | Value |
| --- | --- |
| ControlRoom-pinned SlimX-RAG source (old) | `8ebcb276cc7c6e2df09ec15976a73c7173e23c1a` (= tag `v0.2.8`) |
| SlimX-RAG program base (= `origin/main` at 2026-09-25 10:22 UTC) | `eb365f7be5b6051fb75ed7253d212963e8dc820d` |
| Delta `8ebcb276..eb365f7b` | 3 commits: `b88a6f7` licensing/packaging (MIT `LICENSE`, PEP 639 `license = "MIT"`, hatchling floor `>=1.27.0`), `e64022a` website (`site/`, Pages workflow), `eb365f7` README. No runtime, test, dependency-range, release-infrastructure or Docker change. The candidate is therefore based on current `main`; the packaging change only affects wheel metadata. |
| Benchmark commits (this program, before any behavior change) | `6e8f9548` (benchmark + frozen gate), `dbc1829` (gate serializer) |
| ControlRoom base | `e634d6bb19a980566d71a62880fc06813e8102df` (post-#183 `main`) |
| ControlRoom's reviewed SlimX source | `e4b7ca30f9b528df477672cfa5911c69b722a556` |
| Audit environment | Python 3.12, torch 2.7.1+cpu, sentence-transformers 3.4.1, faiss-cpu 1.15.1, pypdf, python-docx; `HF_HUB_OFFLINE=1`; no GPU; local worktree venv |
| Baseline checks at `eb365f7b` | `pytest` 242 passed; `ruff check` clean; `ruff format --check` would reformat 25 files (format is not a CI gate); `mypy` 10 errors in `embed/embedder.py` **only when the `hf` extra is installed** (CI installs the dev group without extras, so CI mypy passes) |

## 2. Method and limits

- Complete read of `src/slimx_rag` (server, retrieval, chunking, parsers, index backends, signature,
  embedder, answer generator, eval) and of ControlRoom's `services/rag/*`, `routes_rag.py`,
  `indexing_service.py`, `document_cleanup.py`, `document_extraction.py`, Compose and release files.
- Executed experiments against the real FastAPI app (in-process `TestClient`, `hash` embedder unless
  stated): scope/tenancy (`scope_experiments.py`), deletion crash window, request bounds, binary
  ingestion, malformed persisted state, embedder device/model failures
  (`lifecycle_experiments.py`, `failure_experiments.py`), concurrency and corpus-size scaling
  (`concurrency_scale.py`). Scripts and outputs are preserved in the evidence root under `01-audit/`.
- The ControlRoom qualification benchmark (`src/slimx_rag/eval/qualification`, 28 synthetic
  documents, 71 gold cases, real PDF/DOCX) was run with both the deterministic `hash` embedder and the
  CPU release embedder `sentence-transformers/all-MiniLM-L6-v2` (resolved commit
  `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`).
- A multi-agent audit workflow was attempted twice and terminated both times by the account's
  subagent session limit; only the answer/citation finder completed. Its findings were re-verified
  by the author before inclusion (marked "G-finder").
- Not exercised: Qdrant/pgvector against real servers (fake clients only, as in the test suite),
  OpenAI embeddings, GPU, the CLI pipeline beyond `run`/`ask` smoke, the demo UI in a browser.

## 3. Summary

| Severity | Count | IDs |
| --- | --- | --- |
| Critical | 0 | — |
| High | 3 | 001, 002, 003 |
| Medium | 12 | 004–014, 038 |
| Low | 15 | 015–029 |
| Informational | 8 | 030–037 |

No cross-workspace, cross-project or forbidden-document leak was observed in any scoped retrieval
(0 of 425 returned chunks over 71 gold cases with both embedders; 0 in the adversarial scope
experiments). The High findings concern retrievability of text-ingested documents, mutable runtime
model identity, and non-authoritative deletion after a lost state commit.

Answers to the section-7 review targets are in §6.

## 4. Findings

Fields: path/function · observed · reproduction · consequence (product / security / quality) ·
smallest sound disposition · class (merge blocker / release blocker / later improvement).

### RAG-AUD-001 — High — Text-ingested documents contribute at most one chunk per retrieval
- `server/app.py::_to_chunk_record` + `retrieval/hybrid.py::_group_by_parent`.
- Observed (G-finder, re-verified): chunks produced by `/api/index` carry no `parent_id`/`section`,
  so `_to_chunk_record` falls back to `parent_id = doc_id` and `section = None`; grouping admits one
  primary per parent and drops every sibling with the same (`None`) section. An 8-chunk text document
  scoped to itself returned exactly 1 chunk for `top_k=10` (`fused_candidates: 8, final_count: 1`).
- Reproduction: `POST /api/index` with ~6 KB of text, `POST /api/retrieve` scoped to that document.
- Consequence: ControlRoom's degraded text fallback (`index_document`) and any host using the text
  path can never surface more than one passage of a document per question; multi-part answers are
  structurally unreachable. Security: none. Quality: high.
- Disposition: give flat-text chunks a per-chunk parent identity at index time
  (`parent_id = f"{doc_id}#c{chunk_index}"`, `ordinal` stored) so grouping treats them as distinct
  parents; add a server test asserting a multi-chunk text document returns more than one chunk.
- Class: release blocker.

### RAG-AUD-002 — High — Runtime embedding-model identity is mutable
- `embed/embedder.py::HuggingFaceEmbedder.__init__` (revision passed only when configured),
  `Dockerfile` (model downloaded at build without a revision), `docker-entrypoint.sh` (no
  `HF_HUB_OFFLINE`).
- Observed (read-only + local run): the service loads `sentence-transformers/all-MiniLM-L6-v2` at
  the mutable `main` revision. With hub egress available, a newer `main` would be downloaded at
  startup and the signature's `embedding_runtime_identity` would change; the signature correctly
  *detects* this (every document then needs reindex) but the published image would no longer be the
  artifact that was qualified. Offline, the cached snapshot resolves to `c9745ed1…`.
- Consequence: release identity of the image is not the runtime identity; a fleet-wide
  `needs_reindex` can be triggered by an upstream model push. Security: supply-chain integrity.
- Disposition: pin the revision (`RAG_HF_REVISION` → `EmbedSettings.revision`) to the exact commit
  in the image, set `HF_HUB_OFFLINE=1` in the image environment, and fail the build if the baked
  snapshot's commit differs from the pin.
- Class: release blocker (build hardening).

### RAG-AUD-003 — High — Deletion and replacement are not authoritative after a lost state commit
- `index/base.py::delete_doc/commit_doc_state`, `server/app.py::delete_document_endpoint`,
  `index_endpoint`, `index_file_endpoint`.
- Observed (executed): the index file is saved before `index_state.json` is committed by design.
  Simulating the crash window (state entry lost, service restarted): `DELETE` returned
  `deleted_chunks: 0, total: 1` and the document stayed retrievable while the chunk listing reported
  0; a changed-content re-index of the same document id then left the **old version retrievable next
  to the new one** ("Version one…" and "Version two…" both returned).
- Consequence: after any crash between save and commit, a later update leaves stale text of a
  *live* document retrievable under its real `document_id`, which ControlRoom's host-side
  `document_ids` filter cannot reject (the id is legitimate); a deleted document's content is
  retained on disk and remains retrievable by the service. Security: data retention of deleted
  content. Quality: stale answers.
- Disposition: on backends that can enumerate their corpus, make `delete_doc` authoritative by also
  sweeping every stored chunk whose metadata `doc_id` matches (union with the state's `chunk_ids`);
  use the same sweep for the replace step of both index endpoints; report `swept_chunks` separately
  from bookkeeping. Add tests for the crash window.
- Class: release blocker.

### RAG-AUD-004 — Medium — Ambiguous document identity across workspaces
- `server/app.py::index_endpoint/index_file_endpoint/document_chunks_endpoint/delete_document_endpoint`
  (`doc_id = path_id(f"{workspace_id}/{document_id}")`).
- Observed (executed): `(workspace_id="a/b", document_id="c")` and `("a", "b/c")` produce the same
  `doc_id`; the second index replaced the first document's chunks (its workspace tag became `a`), and
  ids containing `/` cannot be listed or deleted through the path endpoints (404).
- Consequence: with free-form ids a tenant can overwrite another tenant's document. ControlRoom sends
  UUIDs, so it is not affected today; the service contract is unsafe for arbitrary hosts.
- Disposition: reject `workspace_id`/`document_id` containing `/`, or empty/whitespace-only, with
  422 on every endpoint (backward compatible for every id that worked end to end).
- Class: merge blocker.

### RAG-AUD-005 — Medium — Empty or absent workspace scope silently widens retrieval
- `server/app.py::_hybrid_retrieve_response`, `retrieval/retriever.py::retrieve`.
- Observed (executed): `workspace_id: ""` and `workspace_id: null` return the whole corpus across
  tenants; `document_ids: []` is treated as "no document filter"; `document_ids` without
  `workspace_id` returns another workspace's document by id.
- Consequence: a host bug that sends an empty scope turns into a cross-tenant read; nothing in the
  service can be configured to refuse it. ControlRoom always sends both and post-validates every
  returned chunk against its eligible set, so the product is defended in depth, but the service
  boundary is permissive.
- Disposition: 422 for empty-string `workspace_id`, empty `document_ids` and empty entries; add
  `RAG_REQUIRE_WORKSPACE_SCOPE=1` (400 `workspace_scope_required` when `workspace_id` is absent) for
  multi-tenant deployments and set it in ControlRoom's Compose; document that `document_ids` narrows a
  workspace, never replaces it.
- Class: merge blocker (validation); release blocker (mode adoption in ControlRoom).

### RAG-AUD-006 — Medium — Lexical candidates are consumed by out-of-scope chunks
- `server/app.py::_current_lexical` (corpus-global BM25), `retrieval/hybrid.py::HybridRetriever.retrieve`.
- Observed (executed): with tenant B scoped and tenant A holding 40 matching documents, the trace
  reported `lexical_candidates: 30` while zero lexical candidates were in scope; the response still
  said `strategy: hybrid`. Dense candidates are scope-filtered before slicing; lexical ones are not.
- Consequence: in a multi-workspace index the lexical half of hybrid retrieval degrades to nothing for
  small tenants, and the trace discloses how many other tenants' chunks matched.
- Disposition: score lexically, filter to scope, then slice to `lexical_candidates`; report in-scope
  counts only.
- Class: release blocker (measured quality + trace honesty).

### RAG-AUD-007 — Medium — `final_parents` caps results below the requested `top_k`
- `settings.py::RetrievalSettings.final_parents = 6`, `server/app.py::_hybrid_retrieve_response`.
- Observed (benchmark): ControlRoom requests `top_k=8`; the mean result count was 5.99 in both
  embedder runs because at most six distinct parents are admitted. The maintenance-log answer
  (`upd-043*`) never reached the eight returned chunks.
- Consequence: the host receives fewer passages than it budgets for; three gold lifecycle cases fail.
- Disposition: derive the parent cap from the request (`final_parents = max(configured, top_k)`);
  measure before/after.
- Class: release blocker (frozen-gate failure).

### RAG-AUD-008 — Medium — Heading-only parents become content-less chunks that win exact boosts
- `chunk/structured.py::_iter_parents`, `retrieval/hybrid.py` exact boost.
- Observed (benchmark, `id-006`): a Markdown `# Supply Agreement HR-2026-0042` heading with no body
  before the next heading becomes its own chunk; the identifier match on its title gave it top-1 with
  no content, pushing the `Term` section down.
- Disposition: skip emitting a chunk for a parent that contains only heading/title elements when the
  document has other content (the heading survives in the children's `section_path` and title).
- Class: release blocker (locator gate).

### RAG-AUD-009 — Medium — The catch-all text parser accepts binary and HTML originals
- `document/parsers/text.py::TextParser.supports` (always true), `document/structure.py::detect_source_type`.
- Observed (executed): an xlsx-like zip and a PNG posted to `/api/index/file` returned 200 with
  `parser: native-text` (79 chunks of control characters for the PNG, title `�PNG`); a corrupt PDF
  correctly returned 422. ControlRoom uploads `.html/.htm` and extracts their text itself, but sends
  the original bytes, which SlimX-RAG indexes as raw markup.
- Consequence: garbage or markup enters the index as a successful document; ControlRoom's fallback
  to its own extracted text only triggers on 422 `parse_failed`.
- Disposition: fail closed for non-text content (NUL bytes / high replacement ratio) and for
  HTML/unknown binary types with 422 `parse_failed: UnsupportedDocumentError`, so ControlRoom falls
  back to its extracted text.
- Class: release blocker.

### RAG-AUD-010 — Medium — Unstructured 500 responses on invalid state or embedder failure
- `server/app.py::retrieve_endpoint/index_endpoint/index_file_endpoint`.
- Observed (executed): corrupt `index_state.json` → `/ready` 503 `index_state_invalid` but
  `/api/retrieve` 500; truncated index line → `backend_load_failed` vs retrieve 500; corrupt receipt →
  `/api/index` 500; `RAG_EMBED_DEVICE=cuda` on a CPU host or an uncached model offline →
  `embedder_init_failed` vs `/api/index` 500.
- Consequence: ControlRoom records "SlimX-RAG returned 500" as a plain failure instead of the
  maintenance state readiness already knows; its per-request readiness probe limits exposure to races.
- Disposition: map these known failures to structured 503/409 codes identical to the `/ready` reasons.
- Class: merge blocker (cheap, contract hygiene).

### RAG-AUD-011 — Medium — `/api/ask` and `/api/eval/run` use a different retrieval path than `/api/retrieve`
- `server/app.py::ask_endpoint/eval_endpoint`, `retrieval/retriever.py::retrieve`.
- Observed (G-finder, re-verified): the legacy dense-only path returns different chunks for the same
  question and labels them `[kb_relpath:index]` (`None` for non-paginated files) with identity-prefixed
  embedding text. ControlRoom uses `/api/retrieve` only and is unaffected; the demo UI uses `/api/ask`.
- Disposition: make the hybrid path the single retrieval owner (`/api/ask` and eval build their
  `RetrievalResult` from hybrid results and hybrid citation labels); keep the host boundary unchanged.
- Class: release blocker (one canonical retrieval owner), low risk.

### RAG-AUD-012 — Medium — Caller-controlled model provider and dataset path on demo endpoints
- `server/app.py::ask_endpoint/eval_endpoint` (`payload.model`, `payload.dataset`).
- Observed (read-only): any caller with the service token (or anyone when no token is configured)
  selects the LLM provider (`openai:`/`anthropic:` egress with the server's credentials) and points
  `/api/eval/run` at an arbitrary server file path (existence oracle, JSONL parse errors).
- Consequence: cost/egress and filesystem probing on a service that ControlRoom deploys without a host
  port and without cloud keys; standalone deployments are exposed.
- Disposition: restrict `dataset` to a configured directory (`RAG_EVAL_DATASET_DIR`, default
  `examples/`) and honour `model` only when `RAG_ALLOW_MODEL_OVERRIDE=1`.
- Class: merge blocker.

### RAG-AUD-013 — Medium — Whole-index rewrite per mutation on the local backend
- `index/local.py::save`, `server/app.py` mutation endpoints (all under `_index_lock`).
- Observed (executed, hash embedder, ~8 chunks/doc): see §7 for the measured table. Every `/api/index`
  and `DELETE` rewrites the complete JSONL file (384 floats per chunk as text) under the lock that
  reads also take; the first retrieval after a write rebuilds BM25 over the corpus.
- Consequence: throughput and latency degrade linearly with corpus size; acceptable for the
  workspace sizes ControlRoom targets today, unacceptable without a boundary statement.
- Disposition: document the measured boundary for the local backend in ControlRoom's knowledge-engine
  doc; no storage-format change in this program.
- Class: later improvement (boundary must be documented for release).

### RAG-AUD-014 — Medium — Mutable and unlocked image build inputs
- `Dockerfile`, `.github/workflows/publish-image.yaml`, `.gitignore` (`uv.lock` ignored),
  `pyproject.toml` ranges.
- Observed (read-only): `python:3.12-slim` by tag; `pip install uv` unversioned; CPU torch unpinned;
  no committed lock; `slimx @ git+https://github.com/slimx-ai/slimx.git` from a moving branch (only
  `/api/ask` with real models imports `slimx`); model baked without revision; git left in the image;
  no OCI labels, SBOM or provenance; the publish workflow always builds CPU and GPU and moves `latest`;
  CI type-checks without the `hf`/`doc` extras.
- Disposition: base image by digest, pinned `uv`, committed `uv.lock` honoured with `--frozen`,
  exact torch CPU version, SlimX from the reviewed archive `e4b7ca30…`, pinned model revision and
  offline runtime, OCI labels, SBOM/provenance, a CPU-only `candidate-<sha>` publication path that never
  touches GPU or `latest`.
- Class: release blocker (build hardening).

### RAG-AUD-015 — Low — Caller metadata can forge citation fields on `/api/index`
Executed: `metadata: {page: 9, section: "Forged", parent_id: "p#1", source_title: "Forged Title"}` →
citation `[Forged Title, p. 9, Forged]`. `workspace_id`/`document_id` cannot be overridden (identity
keys are set first). Disposition: strip reserved keys (`page`, `section`, `section_path`,
`parent_id`, `source_title`, `entry`, `page_type`, `chunk_id`, `workspace_id`, `document_id`) from
caller metadata. Merge blocker (trivial).

### RAG-AUD-016 — Low — Underscore identifiers are split by the lexical tokenizer
`retrieval/tokenize.py::_TOKEN_RE` keeps `.,-/` but not `_`, so `MAX_PAYLOAD_KG` becomes three
common tokens (benchmark `code-064`: a safety-manual page outranked the controller source).
Disposition: keep `_` as an internal separator; measure. Later improvement, measured in this program.

### RAG-AUD-017 — Low — Unbounded request fields
Executed: a 2 MB question, `top_k=10**9`, an empty question and 200 000 `document_ids` are all
accepted (97 ms for the 2 MB question with the hash embedder). Disposition: bound `question` length,
`top_k` and `document_ids` count with 422. Merge blocker (trivial).

### RAG-AUD-018 — Low — Trace discloses out-of-scope match counts
Resolved by RAG-AUD-006.

### RAG-AUD-019 — Low — Parsers override the caller-supplied title (G-finder)
Markdown/DOCX/text parsers prefer an inferred heading over `metadata.title`; only PDF honours the
caller. Disposition: prefer the caller title when present. Later improvement (done with the citation
hygiene change if cheap).

### RAG-AUD-020 — Low — Citation label omits the section when it equals the entry (G-finder)
Non-paginated documents get no locator in the label although `metadata.section` is correct.
Disposition: include the section whenever it differs from the source title. Later improvement.

### RAG-AUD-021 — Low — Chunk listing returns embedding text and title-as-section (G-finder)
`/api/documents/{id}/chunks` returns the identity-prefixed embedding text, reports the title as
`section` for flat-text chunks and never fills offsets. Disposition: return `display_text`, `section`
only. Later improvement.

### RAG-AUD-022 — Low — Answer generator truncates mid-sentence and lists unshown citations (G-finder)
Demo path only; ControlRoom generates its own answers. Later improvement.

### RAG-AUD-023 — Low — `fake:grounded` relevance heuristic and a dead eval branch (G-finder)
Later improvement.

### RAG-AUD-024 — Low — `score` is an RRF value while ControlRoom's `min_score` is a 0..1 threshold (G-finder)
Observed: fusion scores are ~0.016–0.53 (rank-based plus boosts); the benchmark shows no-answer
questions are not separable from answered ones by top-1 score. Disposition: document the scale in the
contract and in ControlRoom's `rag_min_score` help text; ControlRoom's default `0.0` is correct.

### RAG-AUD-025 — Low — mypy is green only without the `hf` extra
With sentence-transformers installed, `embedder.py:253` fails on the `**kwargs` typing. Disposition:
type the optional `revision` argument explicitly. Merge blocker (trivial).

### RAG-AUD-026 — Low — Misleading tokenizer warnings when measuring oversized parents
"Token indices sequence length is longer than the specified maximum (736 > 256)" is emitted by the
counter while measuring a whole parent before splitting; every stored chunk is ≤ 254 tokens (verified
over the benchmark index, stored `token_count` equals the real count). Disposition: count without the
warning. Later improvement.

### RAG-AUD-027 — Low — `deleted_chunks: 0` is ambiguous
Unknown document and lost bookkeeping look identical. Resolved by the sweep in RAG-AUD-003
(`swept_chunks` reported).

### RAG-AUD-028 — Low — No request logging or ids in service mode
Later improvement.

### RAG-AUD-029 — Low — Retrieval holds the index lock for the whole hybrid computation
Reads serialize with each other and with writes (measured in §7). Acceptable at current scale.
Later improvement.

### RAG-AUD-038 — Medium — Qdrant backend calls a removed client method (CI-surfaced)
`index/qdrant_backend.py:171` called `QdrantClient.search`, which qdrant-client removed in 1.15; with
the locked client (1.19.1, the version the service image installs) every Qdrant query raised
`AttributeError`, so the `qdrant` extra was non-functional. Not exercised by the qualification
(ControlRoom uses the local JSONL backend; the unit-test fake still offered `search`). Surfaced on
2026-09-25 by the new 3.12 CI leg that lock-verifies the service extra set and runs mypy against the
real client types (three errors: `api_key` typed `object`, `PointIdsList(points=list[str])`, missing
`search`). Disposition (applied in the same range): query through `query_points` (Universal Query
API, qdrant-client >= 1.10; the extra floor is raised), typed client inputs, and a fake that offers
only `query_points` so a regression to `search` fails in the unit tests. Merge blocker (corrected).

### Informational
- **RAG-AUD-030** The exact-match boost (0.5 identity / 0.15 text) dominates RRF scores (~1/61); this is
  intended for identifiers and produced no pathological ranking in the benchmark.
- **RAG-AUD-031** BM25 IDF statistics span all tenants (information-theoretic only; no text crosses).
- **RAG-AUD-032** Flat-text chunks have no page/section locator by construction; documented.
- **RAG-AUD-033** Scoped retrieval on FAISS/Qdrant/pgvector fails closed with HTTP 400 (verified for
  FAISS); unscoped retrieval on those backends returns every tenant. ControlRoom's declared topology
  is the local backend, so this remains the right boundary; backend-native filtering is not required
  by the topology and is not implemented here.
- **RAG-AUD-034** Reranking is off by default; the benchmark gives no evidence that it is needed.
- **RAG-AUD-035** The `(mtime_ns, size)` hot-backend cache token is practically safe on Linux.
- **RAG-AUD-036** Coverage is 89% overall; least covered: `index/__init__.py` 57%, `ingest/loader.py`
  64%, `qdrant_backend.py` 74%, `answer/generator.py` 77%. The suite has no multithreaded test and no
  test for the deletion crash window (added by this program).
- **RAG-AUD-037** Untitled text documents are cited by their id (ControlRoom always sends a title).

## 5. Verified sound (executed)

- Workspace and document scoping in `_hybrid_retrieve_response`: 0 leaks in 71 gold cases × 2
  embedders (425 chunks) and in the adversarial experiments; identity keys `workspace_id`/`document_id`
  cannot be forged through caller metadata.
- Citation fidelity: every returned chunk's text was found in the attributed document, on the
  attributed PDF page and under the attributed heading (425/425), and every label agrees with the page
  metadata.
- Determinism: identical chunk lists across two passes and after a simulated restart.
- Delete on a consistent state removes every chunk (chunk listing 0, no stale hits, lagging host scope
  included); re-indexing the same id with new content replaces the old text.
- Structured chunker honours the real tokenizer: max stored chunk 254 tokens, stored `token_count`
  exact.
- Readiness truth: corrupt state, corrupt receipt, truncated index, mixed dimensions, dimension
  mismatch, failed embedder init and a missing model each produce a structured 503 reason.
- Concurrency: 8 threads × 40 mixed index/retrieve/delete operations, 320 × HTTP 200, no exception;
  afterwards the index file parsed and its chunk ids equalled the state's chunk ids exactly.
- Auth: constant-time comparison; a missing token yields 401 when a token is configured; `/ready`
  reports `auth_enabled`.
- Path traversal: the multipart filename is metadata only; no filesystem path is derived from it.
- Instance lock: crash-releasing advisory lock plus owner metadata and stale recovery (read-only).

## 6. Section-7 review targets

1. **`/api/retrieve` vs `/api/ask`** — confirmed divergent (RAG-AUD-011); not intentional as a product
   contract; converge on the hybrid path as the single retrieval owner. ControlRoom is unaffected.
2. **Scoped retrieval on non-enumerable backends** — fails closed (HTTP 400). The declared ControlRoom
   topology is the local backend, so this is the right release boundary; no backend-native filtering
   is added.
3. **Reranking** — remains off; the frozen gate is met without it after the measured corrections
   (see §8 for the before/after record); no evidence justifies the added cost.
4. **Candidate counts** — measured: `dense_candidates=lexical_candidates=30` are adequate for the
   benchmark corpus (118 chunks) and the 2 000-document experiment; the binding limit was
   `final_parents` (RAG-AUD-007), not the candidate counts. The lexical budget was wasted across
   tenants (RAG-AUD-006).
5. **Deletion consistency** — proven consistent on a healthy state and proven non-authoritative after
   a lost state commit (RAG-AUD-003). ControlRoom's host-side `document_ids` filter hides a deleted
   document from users but cannot hide stale text of a live document.
6. **Project isolation** — SlimX-RAG knows workspace/document only; ControlRoom derives the eligible
   `document_ids` server-side (`routes_rag.py`, `compatible_document_ids`) and re-validates every
   returned chunk (`qa_service.perform_retrieval`). The benchmark's workspace-only diagnostic shows
   what a workspace-only filter would leak (22 cross-project hits over 8 cases), which is why the
   explicit set is mandatory; `RAG_REQUIRE_WORKSPACE_SCOPE` adds service-side defence in depth.

## 7. Measured scale (local JSONL backend, hash embedder, ~8 chunks per document, one process)

| documents | chunks | index.jsonl | one `/api/index` (replace) | first `/api/retrieve` after a write (BM25 rebuild) | warm `/api/retrieve` | one `DELETE` |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 794 | 7.2 MB | 187 ms | 49 ms | 8 ms | 164 ms |
| 500 | 3 994 | 36.6 MB | 818 ms | 221 ms | 20 ms | 805 ms |
| 1 000 | 7 975 | 73.2 MB | 1 643 ms | 552 ms | 38 ms | 1 613 ms |
| 2 000 | 16 081 | 147.6 MB | 2 409 ms | 747 ms | 211 ms | 2 967 ms |

Every mutation rewrites the whole JSONL file and the first read after a write rebuilds the BM25
sidecar, both linear in corpus size; warm retrieval stays fast up to ~10k chunks and reaches
~0.2 s at 16k. Concurrency: 8 threads × 40 mixed index/retrieve/delete operations completed with
320 × HTTP 200 and a consistent index/state afterwards (`01-audit/concurrency_scale.out`).

## 8. Quality baseline and frozen gate

Baseline at `dbc1829` (benchmark commits on top of `eb365f7b`, no behavior change), `top_k=8`:

| Metric | hash | hf (all-MiniLM-L6-v2, CPU) |
| --- | --- | --- |
| cases scored | 63 | 63 |
| hit@1 / hit@3 / hit@5 | 0.635 / 0.762 / 0.810 | 0.905 / 0.937 / 0.952 |
| MRR / nDCG@8 | 0.710 / 0.729 | 0.921 / 0.921 |
| exact-identifier hit@1 | 0.941 | 0.941 |
| top-1 from expected source (46 tagged) | 0.609 | 0.891 |
| expected locator ok (25 tagged) | 0.680 | 0.920 |
| multi-document coverage@8 | 0.781 | 1.000 |
| cross-workspace / cross-project / forbidden leaks | 0 / 0 / 0 | 0 / 0 / 0 |
| stale deleted hits / stale updated text | 0 / 0 | 0 / 0 |
| citation wrong source / wrong locator (fidelity, 425 chunks) | 0 / 0 | 0 / 0 |
| unstable cases / restart inconsistent | 0 / 0 | 0 / 0 |
| retrieval latency median / p95 | 4 ms / 5 ms | 12–14 ms / 13–17 ms |
| index time (28 docs, 118 chunks) | 1.2 s | 3.9 s |
| peak RSS | — | 660 MB |

The frozen gate (`examples/controlroom_qualification/quality-gate.json`, sha256
`0d506812a63b46384ce8252ceccef8b6c9fa0b15b693caf9469b09a9e66cbd84`) passes every hard invariant
for both embedders and **fails four checks for the release embedder at baseline**:
`updated_doc_missing_new_content` 3 (RAG-AUD-007), `expected_locator_failures` 2 (RAG-AUD-008 and
one paraphrase locator miss), `exact_identifier_hit_at_1` 0.941 (RAG-AUD-016), and
`top1_expected_source_rate` 0.891 (RAG-AUD-007/016 and one conflicting-evidence case). These are the
improvement targets; thresholds are not lowered.

## 9. Disposition plan

| Commit | Scope | Findings |
| --- | --- | --- |
| correctness/security corrections | authoritative delete/replace sweep; id and scope validation; require-scope mode; binary/HTML rejection; structured failure codes; demo-endpoint restrictions; reserved metadata keys; request bounds; mypy typing | 003, 004, 005, 009, 010, 012, 015, 017, 025 |
| measured retrieval-quality improvements | text-chunk parent identity; in-scope lexical candidates; top_k-derived parent cap; heading-only parents; underscore identifiers; single retrieval owner; citation/title/chunk-listing hygiene | 001, 006, 007, 008, 011, 016, 019, 020, 021 |
| build hardening | digest-pinned base, pinned uv/torch, committed lock, reviewed SlimX archive, pinned model revision + offline, labels, SBOM/provenance, CPU-only candidate publication | 002, 014 |
| docs/evidence/version | changelog, contract docs, boundary statement for 013, version bump | 013, 024, 026–037 |
| CI-surfaced correction (after the author-side review) | Qdrant Universal Query API, typed client inputs, `qdrant` extra floor >= 1.10 | 038 |
| quality-correction pass (owner decision, after the first hard stop) | field-addressed heading-less fact sheets, duplicate-passage guard, `index-shaping-v3` | §10 |

Deferred with explicit owner acceptance required: 022, 023, 028, 029, and the storage-format change
behind 013.

## 10. Quality-correction pass (owner decision, 2026-09-25 evening)

At the first hard stop the owner declined to accept the frozen gate's two failing checks with the
retained embedder and directed one bounded correction pass: same model, reranking off, thresholds
and gold untouched (a gold case may change only with evidence and owner approval).

Diagnosis (executed against the real service, hf provider; evidence root
`04-rag-corrections/quality-correction-pass/`): the failing question "When was the Atlas gantry
last serviced?" targets the plain-text maintenance log, a heading-less fact sheet (title line plus
`LAST SERVICE` / `TECHNICIAN` / `NOTES` fields) stored as one packed 49-token chunk. BM25 ranked
that chunk first, but dense retrieval left it outside the top-30 candidates (dense rank 41 of 78 in
scope): cosine 0.478 against the packed chunk versus 0.616 against the title plus the `LAST
SERVICE` line alone, because one mean-pooled vector over three unrelated fields answers a question
about any one of them poorly. With one list missing, reciprocal-rank fusion gave the chunk 1/61 and
it fell to rank 19. The remaining top-1 misses were near ties (file-014: RRF 0.1825 vs 0.1823;
conf-036: the calibration procedure's section is literally titled "Gearbox mounting torque"). The
gold case is valid and unambiguous; it was not changed.

Correction (`chunk/structured.py`, `retrieval/hybrid.py`, `server/app.py`, `document/model.py`
and the text/PDF parsers; `index-shaping-v3`): a heading-less fact sheet with at least two labelled
fields that fits the budget with its longest label prefix is addressed by its fields (one retrieval
unit per field, embedded as the identity prefix plus that field alone, plus one unit for the
remaining elements minus the sheet's own title line, recognised from the page text rather than the
caller's title); every unit displays the whole sheet, shares one retrieval parent and is cited by
its field label, and the hybrid grouping stage never shows the same passage twice
(`dropped_duplicate_text`). Fact sheets under a heading, Markdown and DOCX sections and every
heading-based locator are unchanged; a heading-less PDF fact page is now cited by page and matched
label (`[gallery, p. 2, KEY DETAIL]`) instead of page and page title, which the gate's page-only
PDF locator check cannot see and which the chunker tests pin. Remote backends (dense-only, no
grouping) return the units individually as stored. Indexes shaped by v1 or v2 rebuild.

Narrow same-model sanity review of the first version of this correction (`b5447ef8`) found and
the follow-up commit fixed: (High) a caller title that differs from the sheet's first line, which
is ControlRoom's convention (upload filename), produced a title-only unit under a second retrieval
parent so the same sheet could be shown twice; (Medium) a field unit could exceed the token cap
because the branch was gated on the shorter parent-section prefix; (Medium) the heading test was
vacuous (Markdown never yields fields); documentation over-claimed "no PDF locator moves"; the
ablation tool built records without `display_text`; the chunk-listing contract change was
undocumented.

Measurement (frozen gate, identical corpus, gold and thresholds; before = `bf306d5e`):

| Metric (hf, all-MiniLM-L6-v2, CPU, top_k 8) | before | after |
| --- | --- | --- |
| hit@1 / hit@3 / hit@5 | 0.921 / 0.952 / 0.968 | 0.936 / 0.984 / 1.000 |
| MRR / nDCG@8 | 0.937 / 0.937 | 0.954 / 0.958 |
| exact-identifier hit@1 | 1.000 | 1.000 |
| top-1 expected source (46 tagged) | 0.913 | 0.935 |
| expected locator ok / failures | 1.000 / 0 | 1.000 / 0 |
| `updated_doc_missing_new_content` | 2 | 0 |
| leaks / stale deleted / fidelity errors / unstable | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| latency median / p95 | 11.7 / 12.3 ms | 11.7 / 12.5 ms |
| frozen gate | FAIL (2 of 26) | PASS (26 of 26) |
| hash provider (deterministic) | hit@1 0.635, PASS | hit@1 0.651, PASS |

Only three cases changed rank: upd-043a (absent → 4), upd-043b (absent → 3) and file-014 (2 → 1).
The file-014 change is a tie-break outcome (both boosted candidates now share one fused score and
the dense-first tie-break selects the code file), reported as such rather than as a designed gain.
upd-043a/b are now found but still rank behind the two incident summaries the embedding model
prefers for "Atlas gantry"; the top-1 threshold is met with one case of margin (43 of 46).
conf-036 remains the single documented top-1 miss. Not done on purpose: no threshold, gold,
embedder, reranker, identity-prefix or fusion-constant change.

## 11. Same-model review 2 (Claude, 2026-09-26) and the corrections it required

A second same-model review of the complete range `eb365f7b..0a7bfb69` (NOT independent; the Codex
review of the SlimX-RAG range remains required) returned B. CORRECTION REQUIRED with eleven findings.
Every finding was re-verified by execution before correction; dispositions:

| Id | Sev | Finding | Disposition |
| --- | --- | --- | --- |
| RAG-AUD-039 | High | The gate PASS held only under the benchmark's human titles: ControlRoom sends `title = upload filename`, and this range made the caller's title win, so a document's own heading identity ("Incident Report IR-2026-031") vanished; replayed with filename titles the gate failed 6 checks (hit@1 0.905, exact-id 0.941, top-1 0.891, locator failures 1, missing new content 2, stale/forbidden text 1). | Corrected: parsers keep the document's own title (`ParsedDocument.own_title`), the identity prefix carries `Title: <own title>` when it differs from the caller's title and the entry, a text/Markdown/DOCX document's entry title is its own title, and exact-identifier identity includes the own title plus the filename's alphabetic words, stem and full name. A `Name:` prefix line spelling out filename words was tried and rejected (it put the shared upload prefix into every tiny unit and displaced four top-1 results). The benchmark gained `--title-mode benchmark|filename`; the official regime stays `benchmark` pending owner authorization. |
| RAG-AUD-040 | Medium | `POST /api/admin/embedding` rebuilt settings without `revision`/prefixes and did not persist them; in the offline image `{"device":"cpu"}` returned 422 `embedder_preflight_failed: OSError` (reproduced). | Corrected: `dataclasses.replace(current, …)`, persisted `revision`/prefixes, `hf_revision` accepted and required when `hf_model` changes on a pinned deployment; test with `RAG_HF_REVISION`; the image smoke probes the route offline. |
| RAG-AUD-041 | Medium | Qdrant accepts only UUID/integer point ids; 64-hex chunk ids failed `upsert` with the real client (reproduced in `:memory:`), so RAG-AUD-038's "corrected" claim was incomplete. | Corrected: deterministic UUIDv5 point ids with the chunk id in the payload; retrieve/delete/query map both ways; test against `QdrantClient(":memory:")`; the fake enforces the id rule. |
| RAG-AUD-042 | Medium | `str(md.get("workspace_id"))` turned a missing value into `"None"`, so `{"workspace_id":"None"}` returned unscoped chunks under `RAG_REQUIRE_WORKSPACE_SCOPE`. | Corrected in the hybrid and dense-only scope filters: missing or non-string metadata is out of scope; test. |
| RAG-AUD-043 | Low | BM25 statistics span every workspace (lexical scores move with other tenants' text; term frequency inferable). | Documented as a known limitation (README, CHANGELOG); candidates and text never cross scope (RAG-AUD-006/031). |
| RAG-AUD-044 | Low | The benchmark never set `RAG_HF_REVISION`, so reports recorded the cache's `refs/main` (`1110a243`, byte-identical weights) while the record claimed `c9745ed1`. | Corrected: the runner pins `DEFAULT_HF_REVISION = c9745ed1…` (recorded in the report, CLI `--hf-revision`); the earlier records are annotated below. |
| RAG-AUD-045 | Low | `forbidden_text_hits` / `updated_doc_stale_hits` count matches in any returned document, so a legitimately different document that mentions the forbidden name ("Jonas Berg" in the March incident's attendee table) fails two hard checks. | Owner-authorized 2026-09-26 and corrected in evaluator version 2 (§13): lifecycle stale text counts only in the revised document's own chunks; the global count stays visible as `forbidden_text_hits_any_document`. |
| RAG-AUD-046 | Low | `/api/index` had no text bound and no element cap; posting `""` for an existing document returned 200 and silently emptied it (reproduced). | Corrected: `text` bounded (`RAG_MAX_TEXT_CHARS`), blank rejected (422), element cap applied (413); test. |
| RAG-AUD-047 | Low | `b5447ef8` and `0a7bfb69` chunk differently under one shaping version. | Corrected: `index-shaping-v4` (also covers the identity change); `b5447ef8`/`0a7bfb69` volumes rebuild. |
| RAG-AUD-048 | Low | One stray byte made `decode_text` fall back to cp1252 for the whole file (mojibake). | Corrected: UTF-8 with replacement when stray bytes are sparse (<= 0.5 %); legacy encodings still fall back; test. |
| RAG-AUD-049 | Info | `test_release_inputs.py` checks strings; the fidelity verifier does not check text-document label citations; determinism runs twice in one process; field units add N+1 vectors per sheet (unmeasured on larger corpora); `Dockerfile.gpu` not reviewed in depth. | Recorded; not changed in this pass. |

Measurement after the corrections (frozen gate sha256 `0d506812…`, gold and corpus unchanged,
`all-MiniLM-L6-v2@c9745ed1` pinned and recorded, top_k 8):

| Regime | hit@1 | hit@5 | MRR | nDCG@8 | exact-id@1 | top-1 source | locator failures | missing new content | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| benchmark titles, before (0a7bfb69) | 0.937 | 1.000 | 0.954 | 0.958 | 1.000 | 0.935 | 0 | 0 | PASS 26/26 |
| benchmark titles, after | 0.937 | 1.000 | 0.954 | 0.958 | 1.000 | 0.935 | 0 | 0 | PASS 26/26 |
| filename titles (ControlRoom), before (0a7bfb69) | 0.905 | 0.968 | 0.925 | 0.928 | 0.941 | 0.891 | 1 | 2 | FAIL 6 |
| filename titles (ControlRoom), after | 0.921 | 1.000 | 0.948 | 0.954 | 1.000 | 0.913 | 0 | 0 | FAIL 3 |
| hash provider, benchmark titles, after | 0.651 | 0.810 | 0.724 | 0.737 | 1.000 | 0.609 | 9 | 3 | PASS (17 hard) |

The three remaining filename-regime failures are `updated_doc_stale_hits` and `forbidden_text_hits`
(one case, RAG-AUD-045: the forbidden name occurs in a different, legitimately retrieved document) and
`top1_expected_source_rate` 0.913 (three misses: conf-036, file-014's fused tie and name-017). Whether
the filename regime becomes the official gate regime, and whether the RAG-AUD-045 metric change is
authorized, are owner decisions; nothing in the frozen inputs was changed.

Record correction (RAG-AUD-044): the "resolved `c9745ed1`" statements in §1 and §8 describe the
image's pinned revision; the host benchmark runs recorded before this section resolved the cache's
`refs/main` `1110a243` (weights, tokenizer and configs byte-identical; only the README differs), and
a pinned rerun reproduced identical metrics.

## 12. Same-model review 3 (Claude, 2026-09-26) of `eb365f7b..1f0cd413` and the corrections it required

Again NOT independent (the Codex review of this range remains required). The reviewer reproduced
ruff/mypy clean, 300 tests, GATE PASS 26/26 in benchmark mode with `hf_revision` c9745ed1 recorded,
the three filename-mode failures exactly as stated, and the offline admin-embedding fix in the image.
It confirmed RAG-AUD-039..044 and 046..048 corrected and returned B for the following residuals:

| Id | Sev | Finding | Disposition |
| --- | --- | --- | --- |
| RAG-AUD-050 | Low | The changelog, the shaping comment and docs/index-signature.md described a `Name: <filename words>` prefix line that the code never adds (it was measured and rejected); `humanized_filename` was dead code. | Corrected: the three sentences state the truth; the helper is deleted. |
| RAG-AUD-051 | Low | Version-like titles (`GLM-5.1`, `K2.6`, `v1.2`) matched the filename pattern, so their stems (`glm-5`, `k2`) became exact identities and a `GLM-5.1` document could outrank a `GLM-5` document for "GLM-5". | Corrected: only titles with a known document or code extension count as filenames; tested. |
| RAG-AUD-052 | Low | Persisting the revision on every admin change (from RAG-AUD-040) let a stale pinned revision on the volume strand a later image baked at another commit (`/ready` 503, admin 422). | Corrected: the revision is persisted only when the request supplied `hf_revision`; otherwise the image's `RAG_HF_REVISION` governs; recovery (delete `embed_override.json`) documented in the README; tested. |
| RAG-AUD-053 | Low | `hf_revision` accepted mutable refs such as `main`. | Corrected: the request field and `EmbedSettings.validate` require an exact 40-hex commit for the hf provider; tested. |
| RAG-AUD-054 | Low | The UTF-8 replacement ratio was measured against the whole text, so a long English cp1252 note with sparse accents decoded to mojibake. | Corrected: the ratio is measured against the non-ASCII bytes; a sparse-cp1252 test added. |
| RAG-AUD-055 | Info | A tautological assertion in a test; the admin-revision test uses the hash provider (the hf preflight was verified in the image); an acronym filename boosts every chunk of its file, consistent with human titles and unmeasured. | The assertion is fixed; the rest recorded. |

Both owner decisions were taken on 2026-09-26 and are implemented in §13: RAG-AUD-045 (evaluator
version 2) and the filename regime as a second mandatory gate (under it, top-1 source was 42 of 46 =
0.913 against 0.93, the deciding case being file-014's fused tie; §13 diagnoses and corrects it).

## 13. Owner decisions on the benchmark (2026-09-26) and their implementation

Owner instruction (2026-09-26): preserve the original 26-check gate unchanged; correct only the
demonstrably unsound stale-text evaluator semantics (RAG-AUD-045); add ControlRoom's filename-title
regime as a separate, mandatory qualification gate; make the candidate pass both without lowering a
threshold or changing a gold case; a remaining top-1 shortfall under the filename regime is a real
quality failure to diagnose, not to waive; rerun the image and the cross-repository smoke if source
moves; then stop with a new final head and an updated Codex prompt. No publication or merge.

### 13.1 Evaluator version 2 (RAG-AUD-045)

`forbidden_text_hits` (and the derived `updated_doc_stale_hits`) counted a forbidden string in any
returned chunk. Under the filename regime the question of upd-043c ("Which technician performed the
last Atlas service?") legitimately retrieves the March incident report, whose attendee table lists
"Jonas Berg, Technician"; that chunk is current content of a different document, not the maintenance
log's previous revision, yet it failed two hard checks. `metrics.EVALUATOR_VERSION = "2"`: in the
lifecycle phases the forbidden text counts only inside the revised document's own chunks — the
re-indexed document (`after_update`, same document id) and the deleted documents (`after_delete`).
Isolation cases keep the global rule (their forbidden text marks another workspace's or project's
content, already tied to `forbidden_docs`). The global occurrence stays visible per case as
`forbidden_text_hits_any_document` and in the report's `forbidden_text_any_document` (informational,
not gated). Every report records `evaluator_version`; a gate file may pin it. Thresholds, gold cases,
corpus, model and reranking policy are unchanged. Unit tests cover the four semantics (updated document
with the old text = stale; another document's mention = not stale; deleted document served = stale;
isolation = global).

### 13.2 Two mandatory gates

`quality-gate.json` is byte-for-byte unchanged (the historical gate; sha256 `0d506812…`).
`quality-gate-filename.json` copies every threshold unchanged and pins `title_mode: filename` and
`evaluator_version: 2` (`evaluate_gate` adds a failing `title_mode` / `evaluator_version` check on a
mismatch, so a benchmark-title report cannot satisfy the filename gate); a test asserts the two files'
`hard`, `hard_by_provider` and `ranking` sections are identical and that the original carries no pin.
The deterministic hash run under filename titles now holds the hard invariants in the test suite as
well. `gate-result.json` names the gate, dataset, evaluator and regime.

### 13.3 The remaining filename-regime failure: file-014 (retrieval correction, measured)

With evaluator 2 alone the filename regime still failed one check, `top1_expected_source_rate` 0.913
(42 of 46; the benchmark regime tolerates its three misses conf-036, upd-043a, upd-043b at 43 of 46 =
0.935). The extra miss is file-014, "Which Python function applies the laser offset from calib_v3.cfg?"
(expected `atlas_controller.py`). Diagnosis with the full candidate ranking (evidence
`04-rag-corrections/owner-decisions-2026-09-26/file-014/`): the code unit is dense rank 1 in both
regimes; the lexical stage ranks it third behind two calibration-procedure paragraphs ("The controller
applies the laser offset from this file" shares more query terms), and both leaders carry the same
`calib_v3.cfg` text boost. Under benchmark titles the fused margin was 0.000008 in the code unit's
favour (dense 1 + lexical 3 against dense 2 + lexical 2); the filename title changed the BM25 length
normalisation of the calibration chunks and flipped it (dense 2 + lexical 1). The question's
discriminating words, "Python function", matched nothing in the code unit's identity: the parser
infers `language = python` from the extension but the chunker dropped it.

Correction (`index-shaping-v5`): a source-code document's chunks carry a `Language: <language>`
identity line and a `language` metadata key — a fact about the document like a page number, never a
guess. Measured with the real model, pinned revision, frozen inputs:

| regime | change | hit@1 | hit@3 | hit@5 | MRR | nDCG@8 | exact-id | top-1 source | locator fail | forbidden / stale | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| benchmark | before (7e802dcb) | 0.937 | 0.968 | 1.000 | 0.954 | 0.958 | 1.000 | 0.935 | 0 | 0 / 0 | PASS 26/26 |
| benchmark | evaluator 2 only | 0.937 | 0.968 | 1.000 | 0.954 | 0.958 | 1.000 | 0.935 | 0 | 0 / 0 | PASS 26/26 (every value identical) |
| benchmark | evaluator 2 + Language line | 0.937 | 0.968 | 1.000 | 0.954 | 0.958 | 1.000 | 0.935 | 0 | 0 / 0 | PASS 26/26 (every value identical) |
| filename | before (7e802dcb) | 0.921 | 0.952 | 1.000 | 0.948 | 0.954 | 1.000 | 0.913 | 0 | 1 / 1 | FAIL 3 |
| filename | evaluator 2 only | 0.921 | 0.952 | 1.000 | 0.948 | 0.954 | 1.000 | 0.913 | 0 | 0 / 0 (any-document 1: upd-043c) | FAIL 1 (top-1) |
| filename | evaluator 2 + Language line | 0.937 | 0.952 | 1.000 | 0.956 | 0.960 | 1.000 | 0.935 | 0 | 0 / 0 (any-document 1) | PASS 26/26 under `quality-gate-filename.json` |

file-014 is lexical rank 1 in both regimes after the change (fused 0.18279 against 0.18226);
no other case changed rank in either regime. A benchmark-title report evaluated against the
filename gate fails exactly the `title_mode` pin (verified). Tests: `tests/test_owner_decisions_2026_09_26.py`.

### 13.4 What was not changed

No threshold, gold case, corpus document, embedding model, revision or reranking policy. The
Qdrant/pgvector backends are untouched (the line lives in the embedding text and metadata).

## 14. Same-model review 4 (Claude, 2026-09-26) of `eb365f7b..c7148a5` and the corrections it required

Not independent (author-side evidence, never posted as a review approval). The reviewer reproduced
every claim of §13: ruff/mypy clean, 312 tests, both gates PASS 26/26 with `hf_revision` c9745ed1
recorded, the gate pins working both ways, benchmark mode rank-identical to 1f0cd413, filename mode
changed only at file-014 in the gated ranks (two lower-rank swaps: file-013 6↔7, sem-028 5↔6), the
evaluator-2 scoping sound, and the build inputs byte-unchanged since bf306d5e. Three findings remained:

| Id | Severity | Finding | Disposition |
| --- | --- | --- | --- |
| RAG-AUD-056 | Medium | A model switch that supplied `hf_revision` (the README flow) followed by a ControlRoom-style `{"device":"cpu"}` rewrote `embed_override.json` without the revision, so the switched model ran with the image's `RAG_HF_REVISION` and `/ready` / `/api/index` returned 503 `embedder_init_failed`; ControlRoom cannot send `hf_revision` and had no recovery (reproduced offline with bge-small cached; unreachable in the qualified offline image, reachable online). | Corrected: a revision the override already carries is persisted again when the request names none and the model is unchanged; the image's env revision is still never adopted into the file. Two-step test (switch with revision, then device change; then a prefix-only change). |
| RAG-AUD-057 | Low | `RAG_HF_REVISION=main` from the environment was accepted by the service (`/ready` 200, indexing on a mutable ref) although the README says mutable refs are rejected; `EmbedSettings.validate()` ran only in the admin route and the CLI. | Corrected: `get_cached_embedder` validates the settings on every construction; `/ready` and every embedding path report 503 `embedder_config_invalid` (`EmbedConfigError`, a `ValueError` subclass) with the reason. README states it. Tested. |
| RAG-AUD-058 | Low | The 5 % replacement-ratio rule (RAG-AUD-054) broke the opposite sparse case: a short UTF-8 note with four accented characters and one stray byte crossed the threshold and fell back to cp1252 (whole note mojibake, "JosÃ©", "â€™"). | Corrected: invalid sequences are compared with the valid multibyte characters (cp1252 prose decodes to essentially none; UTF-8 prose with a stray byte keeps every accent), so both sparse cases decode correctly; tested both ways. |
| — | Info | The `Language:` line is the second correction designed against a single gold case; both regimes pass top-1 with one case to spare (43 of 46) and there is no held-out set. | Recorded: the margin is thin. The line is a document fact (parser-inferred language), measured with no other gated-rank change; a held-out set is a benchmark-extension decision for the owner, not taken here. |

Nothing in these corrections touches retrieval, chunking or the benchmark; the final-quality runs are
repeated at the corrected head for the record and must be identical.

## 15. Owner-directed Claude review 1 (2026-09-26/27) of `eb365f7b..1dd6736` and the correction it required

Not independent: the same Claude session that ran the same-model passes 1–4 reviewed the head at the
owner's direction (evidence `13-review-claude-owner-directed-1dd67362-20260926T220604Z/` in the
evidence root: REVIEW.md, commands, logs, full gate reports, probes p1–p6). It executed ruff, mypy,
318 tests, both hf gates and the hash gate at the head (both gates PASS 26/26 with the recorded values),
verified the gate files (the original byte-identical to the benchmark commit, the filename gate an
exact threshold copy with honest pins), evaluator version 2, the revision rules and decoder in every
sequence, scope/delete/rebuild, the image-to-source correspondence (63 source files byte-identical to
`git archive`) and ControlRoom `815229e1`'s consuming contract. Disposition B with one Low finding:

| Id | Severity | Finding | Disposition |
| --- | --- | --- | --- |
| RAG-AUD-059 (RAG-REV-001) | Low | `retrieval/retriever.py` `retrieve()` built its embedder with `make_embedder`, which did not validate the settings, so on the non-local backends (FAISS, Qdrant, pgvector) an unscoped `/api/retrieve` or `/api/ask` still embedded with `RAG_HF_REVISION=main` (reproduced on FAISS: `/ready` 503 `embedder_config_invalid`, `/api/retrieve` 200 with an embedder built at `main`). Readiness and embedding disagreed and the RAG-AUD-057 statement "every embedding path" was false for this path. Unreachable from ControlRoom (local backend, always scoped). | Corrected: `make_embedder` validates every construction (the cache validates as well), `_service_failure` maps `EmbedConfigError` to 503 `embedder_config_invalid` with the reason, so `/api/retrieve`, `/api/ask` and `/api/eval/run` report what `/ready` reports. Regression test: FAISS index built with a valid configuration, then env `main` — unscoped retrieve and ask return 503 `embedder_config_invalid` and no embedder is constructed with `main` (spy); the test fails without the fix. Two token-counter tests that built embedders with non-commit revisions to exercise the counter's own guard now build the embedder directly, so both guards stay tested. |

The reviewer's independent diagnostic (a fresh 12-document corpus, 20 questions, filename titles; not a
gate) is recorded as supporting generality without proving it: top-1 16/20 before field addressing,
13/20 with field addressing before the own-title fix, 18/20 with the own-title fix, 19/20 at the head
(the `Language:` line fixed only the "Which Python function" question). Non-blocking observations
recorded without change: a future image that changes the default model runs a volume override's older
model at the new image's revision (documented recovery: delete `embed_override.json`); a file mixing
UTF-8 and cp1252 is inherently ambiguous; a synthetic payload of only high bytes passes the binary check
(inherited; real formats are rejected); the exact-identifier boost can let a document outrank a better
textual answer; BM25 statistics span workspaces (documented, RAG-AUD-043); ControlRoom's `.env.example`
still defaults to the moving `slimx-rag:latest` image (belongs to the ControlRoom delta review); Qdrant
is exercised in-memory and with fakes only, so it is not qualified for ControlRoom.

