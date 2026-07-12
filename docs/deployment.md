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
