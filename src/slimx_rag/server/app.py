from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from slimx_rag.answer import answer
from slimx_rag.eval import load_eval_cases, run_eval
from slimx_rag.retrieval import retrieve
from slimx_rag.settings import EmbedSettings, IndexSettings


def _backend_config() -> dict[str, object]:
    raw = os.getenv("RAG_BACKEND_CONFIG", "")
    if raw:
        return json.loads(raw)
    if os.getenv("RAG_INDEX_BACKEND", "local") == "qdrant":
        return {"url": os.getenv("QDRANT_URL", "http://qdrant:6333"), "collection": os.getenv("QDRANT_COLLECTION", "slimx_demo")}
    return {}


def _index_settings() -> IndexSettings:
    return IndexSettings(
        backend=os.getenv("RAG_INDEX_BACKEND", "local"),
        backend_config=_backend_config(),
        top_k=int(os.getenv("RAG_TOP_K", "5")),
    )


def _embed_settings() -> EmbedSettings:
    return EmbedSettings(
        provider=os.getenv("RAG_EMBED_PROVIDER", "hash"),
        model=os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small"),
        hf_model=os.getenv("RAG_HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        dim=int(os.getenv("RAG_EMBED_DIM", "384")),
    )


def _index_path() -> Path:
    return Path(os.getenv("RAG_INDEX_PATH", "output/index.jsonl"))


def _state_path() -> Path:
    return Path(os.getenv("RAG_STATE_PATH", "output/index_state.json"))


def _llm_timeout() -> float | None:
    raw = os.getenv("SLIMX_LLM_TIMEOUT", "")
    return float(raw) if raw else None


def _llm_max_tokens() -> int | None:
    raw = os.getenv("SLIMX_LLM_MAX_TOKENS", "")
    return int(raw) if raw else None


def _max_context_chars() -> int | None:
    raw = os.getenv("SLIMX_MAX_CONTEXT_CHARS", "")
    return int(raw) if raw else None


def _check_token(authorization: str | None) -> None:
    token = os.getenv("DEMO_AUTH_TOKEN")
    if not token:
        return
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Missing or invalid demo token")


class QuestionRequest(BaseModel):
    question: str
    model: str | None = None
    top_k: int | None = None


class EvalRequest(BaseModel):
    dataset: str = "examples/research_demo/eval/questions.jsonl"
    model: str | None = None
    top_k: int | None = None


app = FastAPI(title="SlimX-RAG Research Demo")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "index_backend": _index_settings().backend,
        "embed_provider": _embed_settings().provider,
        "llm_model": os.getenv("SLIMX_LLM_MODEL", "fake:grounded"),
    }


@app.get("/api/config")
def config(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_token(authorization)
    return {
        "index_path": str(_index_path()),
        "state_path": str(_state_path()),
        "index": {
            "backend": _index_settings().backend,
            "top_k": _index_settings().top_k,
        },
        "embed": {
            "provider": _embed_settings().provider,
            "model": _embed_settings().model,
            "hf_model": _embed_settings().hf_model,
        },
        "llm_model": os.getenv("SLIMX_LLM_MODEL", "fake:grounded"),
    }


@app.post("/api/retrieve")
def retrieve_endpoint(payload: QuestionRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_token(authorization)
    result = retrieve(
        payload.question,
        index_path=_index_path(),
        state_path=_state_path(),
        embed_settings=_embed_settings(),
        index_settings=_index_settings(),
        top_k=payload.top_k,
    )
    return result.to_dict()


@app.post("/api/ask")
def ask_endpoint(payload: QuestionRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_token(authorization)
    retrieval = retrieve(
        payload.question,
        index_path=_index_path(),
        state_path=_state_path(),
        embed_settings=_embed_settings(),
        index_settings=_index_settings(),
        top_k=payload.top_k,
    )
    result = answer(
        payload.question,
        retrieval,
        model=payload.model or os.getenv("SLIMX_LLM_MODEL", "fake:grounded"),
        timeout=_llm_timeout(),
        max_tokens=_llm_max_tokens(),
        max_context_chars=_max_context_chars(),
    )
    return result.to_dict()


@app.post("/api/eval/run")
def eval_endpoint(payload: EvalRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_token(authorization)
    cases = load_eval_cases(Path(payload.dataset))
    report = run_eval(
        cases,
        index_path=_index_path(),
        state_path=_state_path(),
        embed_settings=_embed_settings(),
        index_settings=_index_settings(),
        model=payload.model or os.getenv("SLIMX_LLM_MODEL", "fake:grounded"),
        top_k=payload.top_k or _index_settings().top_k,
        timeout=_llm_timeout(),
        max_tokens=_llm_max_tokens(),
        max_context_chars=_max_context_chars(),
    )
    return {"markdown": report.to_markdown(), "cases": report.cases}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    static = Path(__file__).resolve().parents[1] / "static" / "index.html"
    return static.read_text(encoding="utf-8")
