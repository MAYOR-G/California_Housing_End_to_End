# syntax=docker/dockerfile:1.7
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/home/appuser/.local/bin:/root/.local/bin:$PATH"

WORKDIR /app

# --- System deps (curl for healthchecks / uv installer) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Install uv (fast resolver) ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- -y

# --- Create non-root user ---
RUN useradd -m -u 10001 appuser


# ========== Dependencies layer (cacheable) ==========

COPY pyproject.toml ./
COPY uv.lock ./            

# Buildkit cache for uv to speed up installs (requires buildx)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen || uv sync


# ========== App layer ==========

COPY main.py ./main.py
COPY templates ./templates

COPY model.joblib ./model.joblib

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck (keep it simple; compose can also define this)
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Run the API
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
