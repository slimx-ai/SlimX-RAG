# Customer Demo Quickstart

## Preferred local demo: Ollama

Run the research demo locally with an Ollama model:

```bash
uv sync --extra dev --extra demo
uv pip install -e ../slimx
ollama pull llama3.1
uv run slimx-rag run --kb-dir examples/research_demo/knowledge-base --out-dir output
OLLAMA_BASE_URL=http://localhost:11434 uv run slimx-rag ask --out-dir output --q "What is SlimX.ai's strongest business wedge for research organizations?" --model ollama:llama3.1 --timeout 300
uv run slimx-rag eval --out-dir output --dataset examples/research_demo/eval/questions.jsonl --out output/eval_report.md
SLIMX_LLM_MODEL=ollama:llama3.1 SLIMX_LLM_TIMEOUT=300 OLLAMA_BASE_URL=http://localhost:11434 uv run slimx-rag serve --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080`.

If the first Ollama call times out, run `ollama run llama3.1 "Say ready"` once to load the model, then retry. Slower machines may need `--timeout 600`.

## Deterministic smoke mode

Use this when Ollama is unavailable or CI needs fully offline behavior:

```bash
uv run slimx-rag ask --out-dir output --q "Compare SlimX Core and SlimX-RAG." --model fake:grounded
```

For a polished live demo, set `SLIMX_LLM_MODEL=openai:gpt-4.1-mini`, `RAG_EMBED_PROVIDER=openai`, and `OPENAI_API_KEY`.
