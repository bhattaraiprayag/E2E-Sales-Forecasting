# Stage 1: Build Backend Environment
FROM python:3.12-slim AS backend-builder
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Stage 2: Runtime
FROM python:3.12-slim
WORKDIR /app

RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY --from=backend-builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY backend ./backend

RUN mkdir -p /app/backend/models && \
    mkdir -p /app/backend/static/generated && \
    mkdir -p /app/models && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app

COPY frontend/dist ./frontend/dist

USER root
RUN apt-get update && apt-get install -y curl libgomp1 && rm -rf /var/lib/apt/lists/*
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
