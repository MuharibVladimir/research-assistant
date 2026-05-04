# syntax=docker/dockerfile:1.7
#
# Multi-stage image for Research Assistant.
# Stage 1 (builder): install deps with uv into a shared .venv
# Stage 2 (runtime): slim Python + copied .venv + app code
#
# Same image is used for both the FastAPI app and the Telegram bot —
# they differ only in the CMD specified in docker-compose.
# ------------------------------------------------------------------------

ARG PYTHON_VERSION=3.14

# ------------------------------------------------------------------------
# Builder — install dependencies
# ------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# uv from its official slim image
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app

# Install runtime/system libs needed by psycopg
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better cache)
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy project and install the package itself
COPY app ./app
COPY bot ./bot
COPY alembic.ini ./
COPY migrations ./migrations
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ------------------------------------------------------------------------
# Runtime — lean image
# ------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app && useradd --system -g app -u 1000 app

WORKDIR /app

COPY --from=builder --chown=app:app /app /app

USER app

EXPOSE 8000

# Default command runs the FastAPI app. Override in docker-compose for bot.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
