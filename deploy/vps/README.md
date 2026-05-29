# VPS Deployment

1. Build and push or copy the `slimx-rag-demo:latest` image to the VPS.
2. Copy `deploy/vps/docker-compose.yml`, `deploy/vps/Caddyfile`, and `.env.example` to the server.
3. Rename `.env.example` to `.env` and set `SLIMX_LLM_MODEL`, provider keys, and `DEMO_AUTH_TOKEN`.
4. Replace `demo.example.com` in `Caddyfile` with the real hostname.
5. Start with `docker compose up -d`.

Recommended customer-demo settings:

```env
RAG_INDEX_BACKEND=qdrant
RAG_BACKEND_CONFIG={"url":"http://qdrant:6333","collection":"slimx_demo"}
RAG_EMBED_PROVIDER=openai
SLIMX_LLM_MODEL=openai:gpt-4.1-mini
DEMO_AUTH_TOKEN=<long random token>
```

Back up the `slimx_rag_data` and `qdrant_data` Docker volumes before reindexing or upgrading.
