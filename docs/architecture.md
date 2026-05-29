# Architecture

SlimX-RAG executes:

```text
ingest -> chunk -> embed -> index -> retrieve -> answer -> cite -> evaluate
```

SlimX-RAG owns deterministic document processing, vector storage, retrieval, citations, evaluation, and the demo API.

SlimX owns provider-neutral model execution, structured responses, tool calls, streaming, retries, and trace metadata.

The customer demo exposes this through a FastAPI server and static UI so users can inspect the answer, retrieved chunks, citations, model/provider trace, and evaluation report.
