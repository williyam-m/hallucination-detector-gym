"""
Integration tests for the Hallucination Detector Gym server.

Tests the full FastAPI server with HTTP and WebSocket endpoints.

Note: OpenEnv HTTP endpoints are stateless (factory per request).
      For stateful multi-step tests, use the WebSocket endpoint.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestResetEndpoint:
    """Tests for the /reset endpoint."""

    def test_reset_default_task(self, client: TestClient) -> None:
        """Reset without task_id should load easy task."""
        response = client.post("/reset", json={})
        assert response.status_code == 200
        data = response.json()
        assert "observation" in data
        assert data["done"] is False
        assert data["reward"] == 0.0

        obs = data["observation"]
        assert obs["passage"] is not None
        assert obs["source_context"] is not None
        assert obs["step_feedback"] is not None

    def test_reset_with_easy_task(self, client: TestClient) -> None:
        """Reset with easy task_id should load that task."""
        response = client.post("/reset", json={"task_id": "task_easy_factual"})
        assert response.status_code == 200
        data = response.json()
        obs = data["observation"]
        assert obs["task_id"] == "task_easy_factual"
        assert obs["difficulty"] == "easy"
        assert obs["num_hallucinations"] == 1

    def test_reset_with_medium_task(self, client: TestClient) -> None:
        """Reset with medium task_id should load that task."""
        response = client.post("/reset", json={"task_id": "task_medium_entity"})
        assert response.status_code == 200
        data = response.json()
        obs = data["observation"]
        assert obs["task_id"] == "task_medium_entity"
        assert obs["difficulty"] == "medium"

    def test_reset_hard_task(self, client: TestClient) -> None:
        """Reset with hard task should work with no hints."""
        response = client.post("/reset", json={"task_id": "task_hard_multi"})
        assert response.status_code == 200
        data = response.json()
        obs = data["observation"]
        assert obs["task_id"] == "task_hard_multi"
        assert obs["num_hallucinations"] is None  # No hint for hard


class TestSchemaEndpoint:
    """Tests for the /schema endpoint."""

    def test_schema_returns_all_schemas(self, client: TestClient) -> None:
        """Schema endpoint should return action, observation, and state schemas."""
        response = client.get("/schema")
        assert response.status_code == 200
        data = response.json()
        assert "action" in data
        assert "observation" in data
        assert "state" in data

    def test_action_schema_has_action_type(self, client: TestClient) -> None:
        """Action schema should define action_type field."""
        response = client.get("/schema")
        data = response.json()
        action_schema = data["action"]
        assert "properties" in action_schema
        assert "action_type" in action_schema["properties"]


class TestWebSocketEndpoint:
    """Tests for the /ws WebSocket endpoint — stateful multi-step episodes."""

    def test_websocket_reset(self, client: TestClient) -> None:
        """WebSocket reset should return initial observation."""
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({
                "type": "reset",
                "data": {"task_id": "task_easy_factual"},
            }))
            response = json.loads(ws.receive_text())
            assert response["type"] == "observation"
            obs = response["data"]
            assert obs["done"] is False
            assert "passage" in obs["observation"]

    def test_websocket_full_episode(self, client: TestClient) -> None:
        """WebSocket full detect→classify→correct→submit episode."""
        with client.websocket_connect("/ws") as ws:
            # Reset
            ws.send_text(json.dumps({
                "type": "reset",
                "data": {"task_id": "task_easy_factual"},
            }))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "observation"
            assert resp["data"]["done"] is False

            # Detect
            ws.send_text(json.dumps({
                "type": "step",
                "data": {
                    "action_type": "detect",
                    "hallucination_detected": True,
                    "hallucinated_span": "Munich, Germany",
                },
            }))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "observation"
            obs = resp["data"]
            assert obs["done"] is False

            # Classify
            ws.send_text(json.dumps({
                "type": "step",
                "data": {
                    "action_type": "classify",
                    "hallucination_type": "factual_error",
                    "hallucinated_span": "Munich, Germany",
                },
            }))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "observation"

            # Correct
            ws.send_text(json.dumps({
                "type": "step",
                "data": {
                    "action_type": "correct",
                    "hallucinated_span": "Munich, Germany",
                    "corrected_text": "Ulm, in the Kingdom of Württemberg",
                },
            }))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "observation"

            # Submit
            ws.send_text(json.dumps({
                "type": "step",
                "data": {"action_type": "submit"},
            }))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "observation"
            assert resp["data"]["done"] is True

    def test_websocket_state(self, client: TestClient) -> None:
        """WebSocket state endpoint should return current state."""
        with client.websocket_connect("/ws") as ws:
            # Reset first
            ws.send_text(json.dumps({
                "type": "reset",
                "data": {"task_id": "task_easy_factual"},
            }))
            ws.receive_text()  # consume reset response

            # Get state
            ws.send_text(json.dumps({"type": "state"}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "state"
            state = resp["data"]
            assert "step_count" in state
            assert state["step_count"] == 0

    def test_websocket_step_increments_state(self, client: TestClient) -> None:
        """Step should increment step_count in state."""
        with client.websocket_connect("/ws") as ws:
            # Reset
            ws.send_text(json.dumps({
                "type": "reset",
                "data": {"task_id": "task_easy_factual"},
            }))
            ws.receive_text()

            # Step
            ws.send_text(json.dumps({
                "type": "step",
                "data": {"action_type": "noop"},
            }))
            ws.receive_text()

            # Check state
            ws.send_text(json.dumps({"type": "state"}))
            resp = json.loads(ws.receive_text())
            assert resp["data"]["step_count"] == 1
