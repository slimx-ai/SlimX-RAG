# ControlRoom qualification benchmark

Owner: `src/slimx_rag/eval/qualification/`. Dataset version: `cq-2026-09-25.1`.

This benchmark measures the served SlimX-RAG contract the way ControlRoom uses it: the
runner drives the FastAPI app through `/api/index/file`, `/api/index`, `/api/retrieve`,
`DELETE /api/documents/{id}`, `/ready` and `/api/config` with a Bearer service token.
Nothing is measured through a library shortcut.

## Corpus

A deterministic synthetic corpus (`corpus.py`, fictional company, parts and incidents):
28 documents across two workspaces and three projects, in Markdown, plain text, Python,
real multi-page PDF (30-page manual with repeated headers/footers) and real DOCX (Title,
Heading and table styles). `MANIFEST.json` pins the SHA-256 of every generated file;
`tests/test_qualification.py` refuses a silent corpus change. Workspace `beta` repeats
identifiers that exist in `alpha` (`PX-4471-B`, `LT-0917`, `47.5 kN`) as the strongest
possible cross-workspace lure; project `borealis` shares a workspace with `atlas`.

## Gold cases

`gold.py` holds 71 cases over exact identifiers, alphanumeric part codes, acronyms, file
names, names, dates, numbers, paraphrases, semantic questions, timelines, duplicated terms,
near-duplicate passages, distractors, multi-document questions, a long document, PDF pages,
DOCX/Markdown sections, an updated document, a deleted document, same-workspace/different-
project scope, different-workspace scope, explicit document restriction, no-answer
questions, conflicting passages, single-source facts and ranking ties. ControlRoom
expresses a project as an explicit `document_ids` set, so every scope here is a
`workspace_id` plus that set (a lagging host set keeps the deleted id on purpose).

## Metrics and gate

Per case: hit@1/3/5/8, reciprocal rank, nDCG@8, expected-document coverage, top-1
citation source, page/section locator of the best expected-document result, scope leak
counters (cross-workspace, cross-scope, forbidden document, stale deleted, stale updated
text), in-scope distractor hits, duplicates and distinct documents. The report also records
determinism across two identical passes, restart consistency, index timings, retrieval
latency (median/p95) and peak RSS.

`quality-gate.json` is the frozen acceptance gate. Its `hard` section is provider-neutral
and must be zero; its `ranking` thresholds apply to the real CPU model. Do not lower a
threshold after seeing post-change results without explicit owner authorization.

## Running

```bash
# deterministic, offline (also run by the test suite)
python -m slimx_rag.eval.qualification --provider hash --out /tmp/cq-hash

# the real CPU model (sentence-transformers/all-MiniLM-L6-v2, cached; no network, no GPU)
HF_HUB_OFFLINE=1 python -m slimx_rag.eval.qualification --provider hf --out /tmp/cq-hf \
  --gate examples/controlroom_qualification/quality-gate.json
```

Outputs: `report.json` (everything, per case), `report.md` (summary, by-tag table,
failing cases), `gate-result.{json,md}` when a gate is supplied (exit code 1 on failure).
