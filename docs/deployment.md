# Deployment

Local smoke tests use the local JSONL index and hash embeddings.

Customer VPS demos should use Docker Compose, Caddy HTTPS, Qdrant, OpenAI embeddings, and an OpenAI or Anthropic generation model.

Use `.env.example` as the environment contract. Never commit provider keys. Set `DEMO_AUTH_TOKEN` when exposing a non-public demo.

Refresh data by rebuilding the index:

```bash
slimx-rag run --kb-dir examples/research_demo/knowledge-base --out-dir output --reindex
```
