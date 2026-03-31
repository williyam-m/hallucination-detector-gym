"""
Hallucination Detector Gym — OpenEnv Client.

Provides a typed Python client for interacting with the Hallucination Detector
Gym environment over WebSocket. Compatible with the OpenEnv EnvClient protocol.

Example:
    >>> from hallucination_detector_gym.client import HallucinationDetectorEnv
    >>> from hallucination_detector_gym.models import HallucinationAction
    >>>
    >>> with HallucinationDetectorEnv(base_url="http://localhost:8000") as client:
    ...     result = client.reset(task_id="task_easy_factual")
    ...     print(result.observation.passage)
    ...
    ...     result = client.step(HallucinationAction(
    ...         action_type="detect",
    ...         hallucination_detected=True,
    ...         hallucinated_span="Munich, Germany",
    ...     ))
    ...     print(result.observation.step_feedback)
"""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from hallucination_detector_gym.models import (
    HallucinationAction,
    HallucinationObservation,
)


class HallucinationDetectorEnv(
    EnvClient[HallucinationAction, HallucinationObservation, State]
):
    """
    Client for the Hallucination Detector Gym Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example — local server:
        >>> with HallucinationDetectorEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task_id="task_easy_factual")
        ...     print(result.observation.passage)
        ...
        ...     action = HallucinationAction(
        ...         action_type="detect",
        ...         hallucination_detected=True,
        ...         hallucinated_span="Munich, Germany",
        ...     )
        ...     result = client.step(action)
        ...     print(result.observation.step_feedback, result.observation.reward)

    Example — Docker:
        >>> client = HallucinationDetectorEnv.from_docker_image(
        ...     "hallucination-detector-gym:latest"
        ... )
        >>> try:
        ...     result = client.reset(task_id="task_easy_factual")
        ...     result = client.step(HallucinationAction(
        ...         action_type="detect",
        ...         hallucination_detected=True,
        ...     ))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: HallucinationAction) -> Dict:
        """Convert HallucinationAction to JSON payload for step message.

        Args:
            action: HallucinationAction instance.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        payload: Dict = {"action_type": action.action_type}
        if action.hallucination_detected is not None:
            payload["hallucination_detected"] = action.hallucination_detected
        if action.hallucination_type is not None:
            payload["hallucination_type"] = action.hallucination_type
        if action.hallucinated_span is not None:
            payload["hallucinated_span"] = action.hallucinated_span
        if action.corrected_text is not None:
            payload["corrected_text"] = action.corrected_text
        if action.reasoning is not None:
            payload["reasoning"] = action.reasoning
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[HallucinationObservation]:
        """Parse server response into StepResult[HallucinationObservation].

        Args:
            payload: JSON response data from server.

        Returns:
            StepResult with HallucinationObservation.
        """
        obs_data = payload.get("observation", {})
        observation = HallucinationObservation(
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            passage=obs_data.get("passage", ""),
            source_context=obs_data.get("source_context", ""),
            num_hallucinations=obs_data.get("num_hallucinations"),
            step_feedback=obs_data.get("step_feedback", ""),
            steps_remaining=obs_data.get("steps_remaining", 0),
            cumulative_reward=obs_data.get("cumulative_reward", 0.0),
            action_history=obs_data.get("action_history", []),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object.

        Args:
            payload: JSON response from state request.

        Returns:
            State object with episode_id and step_count.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
