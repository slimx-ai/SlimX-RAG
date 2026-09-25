# Deployment

Local smoke tests use the local JSONL index and hash embeddings.

Customer VPS demos should use Docker Compose, Caddy HTTPS, Qdrant, OpenAI embeddings, and an OpenAI or Anthropic generation model.

Use `.env.example` as the environment contract. Never commit provider keys. Set
`RAG_AUTH_TOKEN` when exposing the service; `DEMO_AUTH_TOKEN` remains a legacy alias for
ordinary endpoints, but cannot authorize the destructive index-reset maintenance route.

Refresh data by rebuilding the index:

```bash
slimx-rag run --kb-dir examples/research_demo/knowledge-base --out-dir output --reindex
```

## Service image (release identity)

The CPU image built by `Dockerfile` is the only artifact qualified for ControlRoom. Every build
input is immutable:

| Input | Pinned as |
| --- | --- |
| Base image | `python:3.12-slim@sha256:2f17fc04…` (`ARG PYTHON_IMAGE`) |
| Installer | `ghcr.io/astral-sh/uv:0.9.18@sha256:5713fa82…` (`ARG UV_IMAGE`, copied binary) |
| Python resolution | the committed `uv.lock`, installed with `uv sync --frozen` (a build fails if the lock cannot be honoured); torch is the CPU wheel from `download.pytorch.org/whl/cpu` (`[tool.uv.sources]`), pinned by `constraint-dependencies` |
| SlimX | the exact reviewed archive `slimx-ai/slimx@e4b7ca30…` (`answer` extra); no git in the image |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` at commit `c9745ed1…` (`ARG RAG_HF_MODEL` / `RAG_HF_REVISION`); the build verifies the baked snapshot is that commit and the runtime sets `HF_HUB_OFFLINE=1` |
| Identity | OCI labels `org.opencontainers.image.{source,revision,version}` and `ai.slimx.rag.embedding_{model,revision}`; `SOURCE_REVISION` is the build argument the publish workflow sets to the exact commit |

The container runs as user `1000:1000`; `/app/output` (index volume) is owned by that user. A
bind mount must be writable by uid 1000. `HEALTHCHECK` probes `/health` (liveness only; hosts
gate on `/ready`).

Publication: `.github/workflows/publish-image.yaml` → *Run workflow* → `candidate-cpu` builds
only the CPU image from the dispatched commit, pushes `ghcr.io/slimx-ai/slimx-rag:candidate-<sha>`
with SBOM and provenance attestations and prints the registry digest. The digest, not the tag,
is the release identity a host pins (`image@sha256:…`); `latest` never moves and the GPU image
is never built on this path. The `release` path keeps the historical semver/`latest`/GPU
publication for GitHub releases.

Multi-tenant hosts should set `RAG_REQUIRE_WORKSPACE_SCOPE=1` (retrieval without a workspace
fails closed) and pass an explicit `document_ids` set per project; the local JSONL backend
rewrites the whole index file on every mutation, so keep per-instance corpora in the tens of
thousands of chunks or below (measured: 8k chunks ≈ 73 MB, 1.6 s per index/delete, 40 ms warm
retrieval, 0.5 s first retrieval after a write).
