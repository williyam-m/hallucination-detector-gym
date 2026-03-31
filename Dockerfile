# ──────────────────────────────────────────────────────────────────────────────
# Hallucination Detector Gym — Production Dockerfile
#
# OpenEnv-compatible multi-stage build.
# Works for both:
#   - openenv push (pushes to HF Spaces with correct structure)
#   - standalone Docker builds (docker build -t hallucination-detector-gym .)
#
# The openenv build/push CLI handles context detection and sets build args.
# ──────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest

# ── Builder stage ─────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure git is available (required for VCS dependencies)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Build arguments for openenv build modes
ARG BUILD_MODE=in-repo
ARG ENV_NAME=hallucination_detector_gym

# Copy environment code
COPY . /app/env

WORKDIR /app/env

# Ensure uv is available (for local builds where base image may lack it)
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies using uv sync
# If uv.lock exists, use frozen install; otherwise resolve on the fly
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# ── Final runtime stage ──────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy the environment code
COPY --from=builder /app/env /app/env

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Disable Python output buffering
ENV PYTHONUNBUFFERED=1

# Enable the custom Gradio web interface
ENV ENABLE_WEB_INTERFACE=true

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
