# Index compatibility signature

SlimX-RAG 0.2.7 introduces an additive `index_signature` contract for downstream
applications that need to know whether a previously indexed document still belongs to the
active, compatible corpus.

The signature is returned by:

- `GET /ready` (including structured not-ready responses when a partial signature is knowable)
- `GET /api/config`
- `POST /api/index`
- `POST /api/index/file`
- `POST /api/admin/embedding`
- `POST /api/admin/index/reset` (added in 0.2.8)

All pre-0.2.7 response fields remain unchanged.

## Contract

`IndexSignature` is an immutable engine DTO. Its wire form contains:

- `signature_version`, `engine`, and informational `engine_version`
- `signature_complete` (`false` means configured/partial, never a build claim)
- persistent, opaque `index_instance_id`
- `vector_backend`, backend-namespace version, and a sanitized corpus namespace
- index-shaping version/fingerprint and the normalized metadata whitelist
- effective embedding provider, model, dimension, resolved runtime identity, configuration
  version, and fingerprint
- parser-registry version and fingerprint (ordered parser names/versions plus effective
  extraction backend identity)
- `chunk_config_version` for the shared chunk-ID envelope, the combined
  `chunk_config_fingerprint`, and separate `text_chunk_config_fingerprint`,
  `file_chunk_config_version`, and `file_chunk_config_fingerprint`
- the file pipeline's effective token cap and versioned counter/tokenizer identity
- document/content identity protocol versions and persisted index-state schema version
- one canonical `compatibility_fingerprint`

The text fingerprint uses the same size, overlap, and ordered separators used by the
recursive character chunker. The file fingerprint uses the exact structured chunk-ID
formula used by the token/structure-aware chunker. Keeping those formulas in the hashing
layer prevents the receipt from drifting from real chunk identities.

`chunk_config_version="chunk-v1"` identifies the shared chunk-ID envelope made from
parent identity, content hash, configuration fingerprint, and ordinal. The structured
file configuration preimage has its own
`file_chunk_config_version="structured-v2"`. Version 2 adds the effective token cap and
counter/tokenizer identity to the earlier structured-v1 preimage, so indexes built with
the pre-0.2.7 formula intentionally receive different file fingerprints and chunk IDs.
They require a rebuild rather than being described as another structured-v1 corpus.

`index_instance_id` is stored in `index_instance_id` on the index output volume. It survives
ordinary service restarts and document additions. A new index volume, or the embedding
admin reset that deliberately discards the corpus, creates a new identity even when every
configuration value is otherwise identical. First-use creation uses an exclusive lock and
atomic publish, so concurrent workers converge on one persisted identity. The lock retains
an advisory owner lock for its full critical section and safely recovers an abandoned stale
owner file after a crash.

The CLI and HTTP service both derive identity and receipt paths from `index_path.parent`;
placing `index_state.json` on a separate directory or volume does not split the lifecycle.

Every successful CLI or HTTP build also atomically writes `index_build_receipt.json` beside
the identity. The receipt contains the complete canonical signature and document pipeline;
its own fingerprint binds both. CLI `--reindex` rotates the instance ID and writes a receipt
for the replacement build. It clears the prior tracked corpus before upserting, so changed
chunk IDs or embedding dimensions cannot leave vectors from two signatures in one index.
This verified CLI reset is limited to local JSONL and FAISS. Qdrant/pgvector `--reindex`
fails before touching identity, receipt, state, or the remote namespace until those
backends provide a verified full-namespace reset primitive.

## Compatibility semantics

Consumers should persist the `index_signature` returned by a successful index operation
and compare its `compatibility_fingerprint` with the current successful `/ready` response.
A different fingerprint means the document must be reindexed. A missing signature from an
older server or document record is unknown compatibility and should also be treated as
needing reindex, not as compatible.

The compatibility fingerprint includes the persistent instance ID, backend, all
vector-affecting embedding inputs, parser registry, both chunk profiles, sanitized corpus
namespace, metadata shaping, and identity/index schema versions. It deliberately excludes
package patch version and operational settings such as top-k, retrieval candidate counts,
device, batching, retries, auth, LLM settings, paths, and timestamps.
Compatibility-changing implementation work must bump the relevant explicit
config/protocol version.

For the deterministic hash provider, the effective model is `hash-blake2b-v1`; inactive
OpenAI/Hugging Face model settings do not affect its fingerprint. Index receipts use the
canonical dimension evidence order: emitted vector length, loaded backend dimension,
persisted `actual_dim`, then configured fallback. The same resolver is used by HTTP text/file
indexing, readiness/config inspection, and CLI indexing. Persisted `actual_dim` is written
only when emitted/backend evidence exists; a configured fallback is never relabeled actual.

Structured file signatures use the same effective cap as real chunking:
`max(8, min(configured_max_tokens, counter.max_tokens))`. The cap, counter name, counter
protocol version, and tokenizer identity all participate in the real chunk ID and public
fingerprint. Changing a tokenizer/counter contract therefore cannot silently reuse old
structured chunk identities.

For Hugging Face embeddings, a complete signature requires the loaded model's resolved
commit and also binds the tokenizer revision, class, and vocabulary fingerprint. An
unpinned model is therefore never represented by a generic `@unpinned` token or a mutable
configured label such as `main`. PDF extraction binds the effective `pypdf`
availability/version; DOCX binds either the installed `python-docx` version or the built-in
`zip-xml-v1` fallback. Dependency changes cannot silently reuse a receipt built through
another extraction path.

If SentenceTransformers cannot expose runtime commit metadata, an explicitly configured
40- or 64-hex commit digest is accepted as immutable provenance. Mutable labels (including
`main`) and local model paths without a resolved commit/content identity fail closed rather
than producing a complete receipt.

An empty/unbuilt `GET /api/config` response is explicitly `configured_partial` and has
`signature_complete=false`. Config inspection reads only settings, local state, identity,
and the persisted receipt: it does not initialize/download an embedder or contact a remote
index. Once a build receipt exists, `/api/config` returns that persisted build signature as
authoritative and includes the current configured partial signature separately. `/ready`
performs the deep runtime comparison and reports a mismatch instead of reconstructing the
old corpus identity from today's environment.

## Document pipeline receipts

Successful indexing responses also include `document_pipeline`:

- `/api/index`: `ingest_mode="text"`, recursive-character chunker, no parser
- `/api/index/file`: `ingest_mode="file"`, structured-token chunker, actual source type,
  parser name/version, and extraction backend/version

This receipt identifies which of the two chunk fingerprints applies to that document.

## Reset behavior

`POST /api/admin/embedding` can verify corpus reset for the local JSONL and FAISS backends.
It initializes the requested embedder/tokenizer before mutation, then stages the identity,
receipt, override, state, and corpus artifacts as one reset transaction. A failure restores
the prior build. If restoration itself cannot be proven, the endpoint reports
`index_reset_partial_failure` and leaves the active identity/receipt invalidated rather than
describing a partial corpus as the old successful build.

Qdrant and pgvector currently have no host-level truncate primitive, so the endpoint returns
HTTP 409 before changing the embedding override, corpus metadata, or instance identity.
Operators must reset those remote corpus namespaces explicitly and then reconfigure/reindex.

### Explicit same-embedding maintenance reset (0.2.8)

`POST /api/admin/index/reset` is the recovery contract for a global signature, receipt,
state, or instance mismatch when changing embedding settings would be false provenance.
It is intentionally narrower than a generic deletion API:

- canonical `RAG_AUTH_TOKEN` must be configured and supplied as a Bearer token;
- `confirmation` must be the exact literal `RESET INDEX`;
- `expected_index_instance_id` is required but nullable (`null` asserts that no identity
  exists), and is always compared under the corpus lease;
- `expected_compatibility_fingerprint` is also required but nullable. Null is accepted only
  when the receipt is unreadable or belongs to a different instance; otherwise it must match
  the authoritative receipt or one of the engine's configured/runtime partial signatures;
- no request field selects a path, collection, namespace, document, or arbitrary artifact.

The endpoint preflights the active embedder and token counter, then reuses the transactional
local/FAISS reset with the current embedding settings while leaving `embed_override.json`
absent or byte-for-byte unchanged. Its response contains `previous_index_instance_id`,
`previous_index_signature` and source, `new_index_instance_id`, the new incomplete
`index_signature` and source, top-level `engine_version`, and `index_reset=true`.

HTTP 409 precondition/configuration failures are structured and do not mutate artifacts.
Qdrant/pgvector return `index_reset_backend_unsupported` plus an `owner_action` before any
preflight or mutation. A failed local transaction rolls all staged corpus artifacts back;
if rollback cannot be proven, the existing partial-failure contract invalidates identity and
receipt rather than claiming the previous corpus is usable.

Every indexing request snapshots the instance generation and signature-shaping settings
before off-lock parse/chunk/embed work, then compares them under the final write lease. Work
that crossed a reset returns structured `index_generation_changed_retry`; concurrent first
writers cannot both publish into the initial generation.

`/ready` is the recovery diagnostic surface. Every ready/not-ready response includes
`auth_enabled` and top-level `engine_version`; invalid receipts and state use the distinct
reasons `index_build_receipt_invalid` and `index_state_invalid`. For a receipt-instance
mismatch it also returns `active_index_signature`, built with the observed active instance,
so the caller can satisfy instance CAS without trusting the foreign receipt. `/api/config`
remains offline; an unreadable state may still make config inspection fail, so recovery
clients should use `/ready` and may send a null fingerprint only when no trustworthy
signature is available.

## Security

The contract never serializes raw `backend_config`, URLs, connection strings, API keys,
tokens, passwords, index/storage paths, or embedding device selection. The sanitized
namespace includes the non-secret remote corpus coordinates needed to avoid false matches:
Qdrant host/port + collection, or PostgreSQL host/port + database + schema/table. An explicit
`corpus_namespace` can replace those derived coordinates. Credential and connection-option
rotation remains compatibility-neutral; changing the remote cluster/database, collection,
schema/table, backend type, metadata shaping, or persistent instance changes compatibility.
