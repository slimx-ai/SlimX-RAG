#!/usr/bin/env sh
# Turnkey build-then-serve for the SlimX-RAG service image.
#
# If RAG_KB_DIR points at a knowledge-base directory and no index exists yet (or
# RAG_REINDEX is set), build the index first, then serve it. Defaults to the `hf`
# embedder for real, local, no-egress semantic embeddings; everything is overridable
# via the RAG_* env the server already reads (see src/slimx_rag/server/app.py).
#
# With no arguments this runs `slimx-rag serve`; pass a CLI command to override
# (e.g. `docker run ... slimx-rag ask --q "..."`).
set -eu

: "${RAG_INDEX_PATH:=output/index.jsonl}"
out_dir=$(dirname "$RAG_INDEX_PATH")
: "${RAG_STATE_PATH:=$out_dir/index_state.json}"
: "${RAG_EMBED_PROVIDER:=hf}"
: "${RAG_HF_MODEL:=sentence-transformers/all-MiniLM-L6-v2}"
: "${RAG_EMBED_DIM:=384}"
# Empty = let SentenceTransformers auto-select (CUDA if the GPU image + driver are present).
: "${RAG_EMBED_DEVICE:=}"
: "${RAG_INDEX_BACKEND:=local}"
: "${PORT:=8080}"
export RAG_INDEX_PATH RAG_STATE_PATH RAG_EMBED_PROVIDER RAG_HF_MODEL RAG_EMBED_DIM RAG_EMBED_DEVICE RAG_INDEX_BACKEND

if [ -n "${RAG_KB_DIR:-}" ] && { [ ! -f "$RAG_INDEX_PATH" ] || [ -n "${RAG_REINDEX:-}" ]; }; then
  mkdir -p "$out_dir"
  echo "[slimx-rag] indexing '$RAG_KB_DIR' -> '$out_dir' (provider=$RAG_EMBED_PROVIDER)"
  slimx-rag run \
    --kb-dir "$RAG_KB_DIR" \
    --out-dir "$out_dir" \
    --embed-provider "$RAG_EMBED_PROVIDER" \
    --hf-model "$RAG_HF_MODEL" \
    --embed-dim "$RAG_EMBED_DIM" \
    ${RAG_EMBED_DEVICE:+--embed-device "$RAG_EMBED_DEVICE"} \
    --index-backend "$RAG_INDEX_BACKEND"
fi

if [ "$#" -eq 0 ]; then
  set -- slimx-rag serve --host 0.0.0.0 --port "$PORT"
fi
exec "$@"
