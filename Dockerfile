# Stage 1: Build Frontend
FROM node:20-slim AS frontend-builder
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack enable
WORKDIR /app/frontend

COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN --mount=type=cache,id=pnpm,target=/pnpm/store pnpm install --frozen-lockfile

COPY frontend ./
RUN pnpm run build

# Stage 2: Build Backend Environment
FROM python:3.12-slim AS backend-builder
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Stage 3: Runtime
FROM python:3.12-slim
WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy backend venv
COPY --from=backend-builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy backend code
COPY backend ./backend

# Create necessary directories and set permissions
RUN mkdir -p /app/backend/models && \
    mkdir -p /app/backend/static/generated && \
    mkdir -p /app/models && \
    mkdir -p /app/data && \
    mkdir -p /app/backend/mpl_config && \
    chown -R appuser:appuser /app

# Set Matplotlib config directory to writable location
ENV MPLCONFIGDIR=/app/backend/mpl_config

# Copy built frontend assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Install runtime dependencies (curl for healthcheck, libgomp1 for LightGBM)
USER root
RUN apt-get update && apt-get install -y curl libgomp1 && rm -rf /var/lib/apt/lists/*
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
