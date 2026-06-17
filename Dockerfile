FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# git: required to install the `slimx` package from its git repo below.
RUN apt-get update && apt-get install -y --no-install-recommends git \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples

# Install CPU-only torch first so the `hf` extra (sentence-transformers) doesn't pull
# the multi-GB CUDA build, then install with the hf extra for local semantic embeddings.
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch \
  && uv pip install --system ".[demo,openai,qdrant,hf]" \
  && uv pip install --system "slimx @ git+https://github.com/slimx-ai/slimx.git"

# Bake the default hf embedding model so the served image needs no network at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8080
# Entrypoint builds the index from RAG_KB_DIR (if set and missing) then serves; with no
# args it defaults to `slimx-rag serve`. Pass a CLI command to override.
ENTRYPOINT ["/app/docker-entrypoint.sh"]
