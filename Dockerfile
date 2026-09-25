# syntax=docker/dockerfile:1.7
# SlimX-RAG CPU service image — every build input is immutable (ControlRoom release identity):
#   - base image and the uv tool image are pinned by digest;
#   - the Python resolution is the committed uv.lock, honoured with --frozen (a build fails if
#     the lock cannot be satisfied); torch is the CPU wheel from the PyTorch index; SlimX is
#     the exact reviewed archive (no git, no moving branch);
#   - the Hugging Face embedding model is baked at an exact commit and the runtime is offline,
#     so a newer upstream revision can never be adopted silently;
#   - OCI labels carry the source revision/version; the publish workflow attaches SBOM and
#     provenance to the pushed digest.
# Build: docker build --build-arg SOURCE_REVISION=$(git rev-parse HEAD) -t slimx-rag:candidate .
ARG PYTHON_IMAGE=python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.9.18@sha256:5713fa8217f92b80223bc83aac7db36ec80a84437dbc0d04bbc659cae030d8c9

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE}

ARG SOURCE_REVISION=unknown
ARG VERSION=0.3.0
# Default embedding model identity baked into the image. RAG_HF_REVISION is an exact commit.
ARG RAG_HF_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG RAG_HF_REVISION=c9745ed1d9f207416be6d2e6f8de32d1f16199bf

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    HF_HOME=/app/hf-cache \
    RAG_HF_MODEL=${RAG_HF_MODEL} \
    RAG_HF_REVISION=${RAG_HF_REVISION}

WORKDIR /app
COPY --from=uv /uv /uvx /bin/

# Locked, non-editable install of the service with every extra the service image needs
# (demo = FastAPI server, doc = PDF/DOCX parsers, hf = sentence-transformers, answer = SlimX).
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable \
      --extra demo --extra openai --extra qdrant --extra hf --extra doc --extra answer \
  && /app/.venv/bin/python -c "import pypdf, docx, sentence_transformers, slimx, fastapi" \
  && /app/.venv/bin/python -c "import torch; assert not torch.cuda.is_available(); print('torch', torch.__version__)"
ENV PATH=/app/.venv/bin:$PATH

# Bake the embedding model at its exact commit, then verify the snapshot really is that commit.
RUN python - <<'PY'
import os, pathlib
from sentence_transformers import SentenceTransformer
model, revision = os.environ["RAG_HF_MODEL"], os.environ["RAG_HF_REVISION"]
SentenceTransformer(model, revision=revision, device="cpu")
root = pathlib.Path(os.environ["HF_HOME"]) / "hub" / ("models--" + model.replace("/", "--")) / "snapshots"
snapshots = sorted(p.name for p in root.iterdir())
assert snapshots == [revision], f"baked snapshots {snapshots} != pinned revision {revision}"
print("baked", model, "@", revision)
PY
# From here on the runtime never reaches the Hugging Face hub.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

COPY examples ./examples
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod 0755 /app/docker-entrypoint.sh \
  && useradd --uid 1000 --user-group --no-create-home --shell /usr/sbin/nologin slimx \
  && mkdir -p /app/output \
  && chown -R 1000:1000 /app/output /app/hf-cache \
  && chmod -R a+rX /app/src /app/examples /app/.venv /app/hf-cache

LABEL org.opencontainers.image.title="slimx-rag" \
      org.opencontainers.image.description="SlimX-RAG knowledge service (CPU): parse, chunk, embed, index, retrieve, cite" \
      org.opencontainers.image.source="https://github.com/slimx-ai/SlimX-RAG" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.licenses="MIT" \
      ai.slimx.rag.embedding_model="${RAG_HF_MODEL}" \
      ai.slimx.rag.embedding_revision="${RAG_HF_REVISION}" \
      ai.slimx.rag.base_image="python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9"

USER 1000:1000
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
  CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3).status == 200 else 1)"]
# Entrypoint builds the index from RAG_KB_DIR (if set and missing) then serves; with no
# args it defaults to `slimx-rag serve`. Pass a CLI command to override.
ENTRYPOINT ["/app/docker-entrypoint.sh"]
