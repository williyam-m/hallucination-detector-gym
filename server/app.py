"""
Hallucination Detector Gym — FastAPI Application.

Creates the FastAPI app exposed by the OpenEnv server.  When the web interface
is enabled (``ENABLE_WEB_INTERFACE=true``), the default generic Gradio UI is
**replaced** by a hallucination-specific custom UI built in
``server.gradio_builder``.  This avoids the confusing "Playground / Custom"
tab layout that the ``gradio_builder`` callback mechanism produces, and
instead presents a single, polished interface.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - GET /health: Health check
    - WS /ws: WebSocket endpoint for persistent sessions
    - /web: Gradio web UI (when ENABLE_WEB_INTERFACE=true)

Usage:
    # Development (with auto-reload):
    ENABLE_WEB_INTERFACE=true uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1

    # Via OpenEnv CLI:
    uv run --project . server
"""

from __future__ import annotations

import os

try:
    from openenv.core.env_server.http_server import create_app, create_fastapi_app
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


# ──────────────────────────────────────────────────────────────────────────────
# Build the app
# ──────────────────────────────────────────────────────────────────────────────
def _build_app():
    """Build the FastAPI application.

    When ENABLE_WEB_INTERFACE is true, we replace the default Gradio UI
    entirely with our custom hallucination-specific one so the user sees
    a single polished interface (not "Playground | Custom" tabs).
    """
    enable_web = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in (
        "true", "1", "yes",
    )

    if not enable_web:
        # No web UI — just the API server (standard path)
        return create_app(
            HallucinationDetectorEnvironment,
            HallucinationAction,
            HallucinationObservation,
            env_name="hallucination_detector_gym",
            max_concurrent_envs=1,
        )

    # ── Web UI enabled: build our own Gradio mount ───────────────────────
    from typing import Any, Dict, Optional

    import gradio as gr
    from fastapi import Body, WebSocket, WebSocketDisconnect
    from fastapi.responses import RedirectResponse

    from openenv.core.env_server.gradio_theme import (
        OPENENV_GRADIO_CSS,
        OPENENV_GRADIO_THEME,
    )
    from openenv.core.env_server.web_interface import (
        WebInterfaceManager,
        _extract_action_fields,
        _is_chat_env,
        get_quick_start_markdown,
        load_environment_metadata,
    )
    from server.gradio_builder import build_hallucination_gradio_app

    # 1. Create the base FastAPI app (all API routes: /reset, /step, etc.)
    fastapi_app = create_fastapi_app(
        HallucinationDetectorEnvironment,
        HallucinationAction,
        HallucinationObservation,
        max_concurrent_envs=1,
    )

    # 2. Load metadata & build the WebInterfaceManager
    metadata = load_environment_metadata(
        HallucinationDetectorEnvironment, "hallucination_detector_gym"
    )
    web_manager = WebInterfaceManager(
        HallucinationDetectorEnvironment,
        HallucinationAction,
        HallucinationObservation,
        metadata,
    )

    # 3. Web convenience routes
    @fastapi_app.get("/", include_in_schema=False)
    async def web_root():
        return RedirectResponse(url="/web/")

    @fastapi_app.get("/web", include_in_schema=False)
    async def web_root_no_slash():
        return RedirectResponse(url="/web/")

    @fastapi_app.get("/web/metadata")
    async def web_metadata():
        return web_manager.metadata.model_dump()

    @fastapi_app.websocket("/ws/ui")
    async def websocket_ui_endpoint(websocket: WebSocket):
        await web_manager.connect_websocket(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await web_manager.disconnect_websocket(websocket)

    @fastapi_app.post("/web/reset")
    async def web_reset(request: Optional[Dict[str, Any]] = Body(default=None)):
        return await web_manager.reset_environment(request)

    @fastapi_app.post("/web/step")
    async def web_step(request: Dict[str, Any]):
        if "message" in request:
            action_data = {"message": request["message"]}
        else:
            action_data = request.get("action", {})
        return await web_manager.step_environment(action_data)

    @fastapi_app.get("/web/state")
    async def web_state():
        return web_manager.get_state()

    # 4. Build the custom Gradio Blocks (our enhanced UI)
    action_fields = _extract_action_fields(HallucinationAction)
    is_chat_env = _is_chat_env(HallucinationAction)
    quick_start_md = get_quick_start_markdown(
        metadata, HallucinationAction, HallucinationObservation,
    )

    gradio_blocks = build_hallucination_gradio_app(
        web_manager,
        action_fields,
        metadata,
        is_chat_env,
        title=metadata.name,
        quick_start_md=quick_start_md,
    )

    # 5. Mount as the only Gradio app at /web
    fastapi_app = gr.mount_gradio_app(
        fastapi_app,
        gradio_blocks,
        path="/web",
        theme=OPENENV_GRADIO_THEME,
        css=OPENENV_GRADIO_CSS,
    )

    return fastapi_app


app = _build_app()


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
