# Demo Script

1. Open the demo UI and show the configured provider/model.
2. Ask: “What is SlimX.ai's strongest business wedge for research organizations?”
3. Show the answer citations and retrieved chunks.
4. Ask: “Compare SlimX Core and SlimX-RAG.”
5. Run the evaluation panel and show hit/citation metrics.
6. Ask: “What information is missing from the corpus about pricing?”
7. Explain that the assistant should refuse or flag insufficient context.

For local demos, prefer Ollama with `SLIMX_LLM_MODEL=ollama:llama3.1` and `OLLAMA_BASE_URL=http://localhost:11434`.

If Ollama or the live provider is unavailable, switch `SLIMX_LLM_MODEL=fake:grounded` and continue with the deterministic smoke demo.
