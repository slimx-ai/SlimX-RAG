from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from slimx_rag.retrieval import RetrievalResult


DEFAULT_SYSTEM_PROMPT = (
    "You are a research AI assistant. Answer only from the retrieved context. "
    "Cite every factual claim using the provided citation labels. "
    "If the context is insufficient, say that the corpus does not contain enough information."
)


def default_timeout_for_model(model: str) -> Optional[float]:
    if model.startswith("ollama:"):
        return 180.0
    return None


def default_max_tokens_for_model(model: str) -> Optional[int]:
    if model.startswith("ollama:"):
        return 256
    if model.startswith("fake:"):
        return None
    return 700


def default_max_context_chars_for_model(model: str) -> Optional[int]:
    if model.startswith("ollama:"):
        return 3000
    return None


@dataclass(frozen=True, slots=True)
class AnswerResult:
    question: str
    answer: str
    citations: list[str]
    retrieval: dict[str, Any]
    model_trace: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_grounded_prompt(
    question: str,
    retrieval: RetrievalResult,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_context_chars: Optional[int] = None,
) -> str:
    context_blocks = []
    remaining_chars = max_context_chars
    for i, chunk in enumerate(retrieval.chunks, start=1):
        if remaining_chars is not None and remaining_chars <= 0:
            break
        text = chunk.text
        if remaining_chars is not None:
            text = text[:remaining_chars]
            remaining_chars -= len(text)
        context_blocks.append(
            f"Context {i} {chunk.citation}\n"
            f"Score: {chunk.score:.4f}\n"
            f"Text:\n{text}"
        )
    context = "\n\n---\n\n".join(context_blocks) or "No retrieved context."
    return (
        f"{system_prompt}\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer with concise bullets where helpful. Include citation labels inline."
    )


def answer(
    question: str,
    retrieval: RetrievalResult,
    *,
    model: str = "fake:grounded",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: Optional[float] = 0.1,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    max_context_chars: Optional[int] = None,
) -> AnswerResult:
    warnings: list[str] = []
    citations = [chunk.citation for chunk in retrieval.chunks]
    if not retrieval.chunks:
        warnings.append("no_retrieved_context")

    if model.startswith("fake:"):
        context_text = "\n".join(chunk.text.lower() for chunk in retrieval.chunks)
        question_terms = [t.strip(".,?!:;()[]").lower() for t in question.split()]
        important_terms = [t for t in question_terms if len(t) > 4]
        context_looks_relevant = any(t in context_text for t in important_terms)
        if not retrieval.chunks or not context_looks_relevant:
            text = "The corpus does not contain enough information to answer this question."
        else:
            first = retrieval.chunks[0]
            text = (
                f"Based on the retrieved corpus, the strongest evidence is in {first.citation}. "
                f"{first.text[:280].strip()}"
            )
        model_trace = {"provider": "fake", "model": model, "elapsed_ms": 0, "timeout": timeout}
    else:
        try:
            from slimx import llm
        except Exception as e:  # pragma: no cover - dependency packaging guard
            raise RuntimeError("SlimX answer generation requires the 'slimx' package.") from e

        effective_max_tokens = max_tokens if max_tokens is not None else default_max_tokens_for_model(model)
        effective_max_context_chars = (
            max_context_chars if max_context_chars is not None else default_max_context_chars_for_model(model)
        )
        prompt = build_grounded_prompt(
            question,
            retrieval,
            system_prompt=system_prompt,
            max_context_chars=effective_max_context_chars,
        )
        effective_timeout = timeout if timeout is not None else default_timeout_for_model(model)
        response = llm(model, temperature=temperature, max_tokens=effective_max_tokens, timeout=effective_timeout)(prompt)
        text = response.text
        model_trace = dict(getattr(response, "trace", {}) or {})

    if citations and not any(citation in text for citation in citations):
        warnings.append("answer_missing_citation")
    return AnswerResult(
        question=question,
        answer=text,
        citations=citations,
        retrieval=retrieval.to_dict(),
        model_trace=model_trace,
        warnings=warnings,
    )
