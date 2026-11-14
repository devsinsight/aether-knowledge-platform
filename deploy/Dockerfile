# Platform image: sync service, pipeline CLI and Knowledge MCP. Linux, so native wheels (torch, grpc) load.
FROM python:3.12.15-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/cache/huggingface FASTEMBED_CACHE_PATH=/cache/fastembed \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY packages ./packages
RUN uv sync --locked
RUN apt-get update && apt-get install -y --no-install-recommends libatomic1 && rm -rf /var/lib/apt/lists/*
COPY pipelines ./pipelines
COPY golden-set ./golden-set
