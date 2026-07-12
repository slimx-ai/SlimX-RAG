# Architecture

SlimX-RAG executes:

```text
ingest -> chunk -> embed -> index -> retrieve -> answer -> cite -> evaluate
```

SlimX-RAG owns deterministic document processing, vector storage, retrieval, citations, evaluation, and the demo API.

SlimX owns provider-neutral model execution, structured responses, tool calls, streaming, retries, and trace metadata.

The customer demo exposes this through a FastAPI server and static UI so users can inspect the answer, retrieved chunks, citations, model/provider trace, and evaluation report.

The HTTP service also publishes a canonical, engine-owned
[index compatibility signature](index-signature.md). It distinguishes the recursive text
and structured-file pipelines and lets downstream applications detect a replacement index
volume or incompatible embedding/chunk/parser/extraction/metadata-shaping change without
inspecting backend credentials or duplicating SlimX-RAG configuration logic. Successful
builds persist the full signature and pipeline as an engine-owned build receipt; shallow
config inspection reads that receipt offline, while deep readiness validates it against the
active runtime.
