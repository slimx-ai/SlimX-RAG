FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples

RUN uv pip install --system ".[demo,openai,qdrant]" \
  && uv pip install --system "slimx @ git+https://github.com/slimx-ai/slimx.git"

EXPOSE 8080
CMD ["slimx-rag", "serve", "--host", "0.0.0.0", "--port", "8080"]
