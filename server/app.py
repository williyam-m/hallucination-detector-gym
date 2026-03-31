"""
Hallucination Detector Gym — FastAPI Application.

Creates the FastAPI app exposed by the OpenEnv server. Registers all HTTP
and WebSocket routes via openenv's create_app helper.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - GET /health: Health check
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1

    # Via OpenEnv CLI:
    uv run --project . server
"""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n"
        "    uv sync\n'"
    ) from e

try:
    from hallucination_detector_gym.models import (
        HallucinationAction,
        HallucinationObservation,
    )
    from server.hallucination_environment import HallucinationDetectorEnvironment
except ModuleNotFoundError:  # pragma: no cover
    from hallucination_detector_gym.models import (
        HallucinationAction,
        HallucinationObservation,
    )
    from server.hallucination_environment import HallucinationDetectorEnvironment

from hallucination_detector_gym.logging_config import configure_logging

# ── Configure structured logging on import ───────────────────────────────────
configure_logging(json_output=True, log_level="INFO")

# ── Create the OpenEnv-compliant FastAPI app ─────────────────────────────────
app = create_app(
    HallucinationDetectorEnvironment,
    HallucinationAction,
    HallucinationObservation,
    env_name="hallucination_detector_gym",
    max_concurrent_envs=1,  # increase to allow more concurrent WebSocket sessions
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for direct execution via `uv run --project . server`.

    Args:
        host: Host address to bind to (default: "0.0.0.0").
        port: Port number to listen on (default: 8000).
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
