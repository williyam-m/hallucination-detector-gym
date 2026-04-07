"""
Hallucination Detector Gym — Core Environment Implementation.

Implements the OpenEnv Environment base class with step(), reset(), state().
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import structlog

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import (
    Action,
    Observation,
    State,
)

from hallucination_detector_gym.constants import (
    ActionType,
    MAX_STEPS_PER_EPISODE,
    TaskID,
)
from hallucination_detector_gym.models import (
    HallucinationAction,
    HallucinationObservation,
    HallucinationState,
)
from hallucination_detector_gym.rewards import RewardEngine
from hallucination_detector_gym.tasks import get_task, TaskDefinition

logger = structlog.get_logger(__name__)


class HallucinationDetectorEnvironment(Environment):
    """OpenEnv environment for hallucination detection, classification, and correction.

    The agent receives a passage with potential hallucinations and must:
    1. Detect if hallucinations are present
    2. Classify their type (factual error, entity fabrication, logical inconsistency)
    3. Propose corrections

    Rewards are dense and partial-progress, not just binary end-of-episode.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        """Initialize the environment with default state."""
        super().__init__()
        self._state = HallucinationState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
        )
        self._task: Optional[TaskDefinition] = None
        self._reward_engine: Optional[RewardEngine] = None
        self._action_history: list[str] = []
        self._current_task_id: Optional[TaskID] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> HallucinationObservation:
        """Reset the environment and return the initial observation.

        Args:
            seed: Optional random seed (unused — tasks are deterministic).
            episode_id: Optional custom episode identifier.
            task_id: Which task to load. Defaults to TASK_EASY.
            **kwargs: Additional reset parameters.

        Returns:
            Initial HallucinationObservation with the passage to analyse.
        """
        # Resolve task
        resolved_task_id = TaskID(task_id) if task_id else TaskID.TASK_EASY
        self._task = get_task(resolved_task_id)
        self._current_task_id = resolved_task_id
        self._reward_engine = RewardEngine(self._task)
        self._action_history = []

        ep_id = episode_id or str(uuid.uuid4())
        self._state = HallucinationState(
            episode_id=ep_id,
            step_count=0,
            task_id=resolved_task_id,
            difficulty=self._task.difficulty,
            cumulative_reward=0.0,
            detections_submitted=0,
            classifications_submitted=0,
            corrections_submitted=0,
            is_done=False,
        )

        logger.info(
            "environment_reset",
            episode_id=ep_id,
            task_id=resolved_task_id.value,
            difficulty=self._task.difficulty.value,
        )

        return HallucinationObservation(
            done=False,
            reward=0.0,
            task_id=resolved_task_id,
            difficulty=self._task.difficulty,
            passage=self._task.passage,
            source_context=self._task.source_context,
            num_hallucinations=self._task.hint_num_hallucinations,
            step_feedback="Environment reset. Analyse the passage for hallucinations.",
            steps_remaining=MAX_STEPS_PER_EPISODE,
            cumulative_reward=0.0,
            action_history=[],
            metadata={
                "task_title": self._task.title,
                "task_description": self._task.description,
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> HallucinationObservation:
        """Execute an agent action and return the resulting observation.

        Args:
            action: A HallucinationAction from the agent.
            timeout_s: Optional timeout (unused).
            **kwargs: Additional step parameters.

        Returns:
            HallucinationObservation with feedback and updated state.
        """
        if self._task is None or self._reward_engine is None:
            return HallucinationObservation(
                done=True,
                reward=0.0,
                step_feedback="Error: environment not reset. Call reset() first.",
                metadata={"error": "environment_not_reset"},
            )

        if self._state.is_done:
            return HallucinationObservation(
                done=True,
                reward=0.0,
                step_feedback="Episode already finished. Call reset() to start a new one.",
                cumulative_reward=self._reward_engine.cumulative_reward,
                metadata={"error": "episode_already_done"},
            )

        # Cast to typed action — handle malformed input gracefully
        try:
            if isinstance(action, HallucinationAction):
                typed_action = action
            elif isinstance(action, dict):
                typed_action = HallucinationAction(**action)
            else:
                typed_action = HallucinationAction(**action.model_dump())
        except Exception as exc:
            logger.warning("malformed_action", error=str(exc))
            typed_action = HallucinationAction(
                action_type=ActionType.NOOP,
                reasoning=f"Malformed action received: {exc}",
            )

        # Increment step
        self._state.step_count += 1
        steps_remaining = MAX_STEPS_PER_EPISODE - self._state.step_count

        # Track action type counts
        if typed_action.action_type == ActionType.DETECT:
            self._state.detections_submitted += 1
        elif typed_action.action_type == ActionType.CLASSIFY:
            self._state.classifications_submitted += 1
        elif typed_action.action_type == ActionType.CORRECT:
            self._state.corrections_submitted += 1

        # Compute reward
        reward, feedback = self._reward_engine.compute_reward(typed_action)
        self._state.cumulative_reward = self._reward_engine.cumulative_reward

        # Build action history entry
        action_summary = f"Step {self._state.step_count}: {typed_action.action_type.value}"
        if typed_action.hallucination_detected is not None:
            action_summary += f" (detected={typed_action.hallucination_detected})"
        if typed_action.hallucination_type is not None:
            action_summary += f" (type={typed_action.hallucination_type.value})"
        self._action_history.append(action_summary)

        # Check termination
        is_done = (
            typed_action.action_type == ActionType.SUBMIT
            or steps_remaining <= 0
        )
        self._state.is_done = is_done

        if is_done and typed_action.action_type != ActionType.SUBMIT:
            feedback += " Episode ended: maximum steps reached."

        logger.info(
            "step_executed",
            episode_id=self._state.episode_id,
            step=self._state.step_count,
            action_type=typed_action.action_type.value,
            reward=round(reward, 4),
            cumulative=round(self._state.cumulative_reward, 4),
            done=is_done,
        )

        # Compute grader score only when episode ends
        final_grader_score = (
            self._reward_engine.get_final_score() if is_done else None
        )

        return HallucinationObservation(
            done=is_done,
            reward=reward,
            task_id=self._current_task_id,
            difficulty=self._task.difficulty,
            passage=self._task.passage,
            source_context=self._task.source_context,
            num_hallucinations=self._task.hint_num_hallucinations,
            step_feedback=feedback,
            steps_remaining=max(0, steps_remaining),
            cumulative_reward=self._state.cumulative_reward,
            action_history=list(self._action_history),
            grader_score=final_grader_score,
            metadata={
                "grader_score": final_grader_score,
            },
        )

    @property
    def state(self) -> HallucinationState:
        """Return the current internal environment state.

        Returns:
            The current HallucinationState.
        """
        return self._state

    def close(self) -> None:
        """Clean up environment resources."""
        self._task = None
        self._reward_engine = None
        self._action_history.clear()
        logger.info("environment_closed", episode_id=self._state.episode_id)
