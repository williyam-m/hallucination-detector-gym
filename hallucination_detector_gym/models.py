"""
Hallucination Detector Gym — Pydantic Models.

Typed Action, Observation, and State models conforming to the OpenEnv spec.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import (
    Action as BaseAction,
    Observation as BaseObservation,
    State as BaseState,
)

from .constants import ActionType, Difficulty, HallucinationType, TaskID


# ──────────────────────────────────────────────────────────────────────────────
# Action
# ──────────────────────────────────────────────────────────────────────────────
class HallucinationAction(BaseAction):
    """Action the agent submits each step.

    The agent can:
      - detect: flag whether the current passage contains a hallucination
      - classify: label the hallucination type
      - correct: propose a corrected span
      - submit: finalise the episode with accumulated answers
      - noop: skip (useful if agent needs to reason before acting)
    """

    action_type: ActionType = Field(
        default=ActionType.NOOP,
        description=(
            "The type of action to perform: detect, classify, correct, submit, or noop. "
            "Defaults to 'noop' if omitted."
        ),
    )
    hallucination_detected: Optional[bool] = Field(
        default=None,
        description="Whether the agent believes a hallucination exists in the current passage.",
    )
    hallucination_type: Optional[HallucinationType] = Field(
        default=None,
        description="The category of hallucination (required for 'classify' action).",
    )
    hallucinated_span: Optional[str] = Field(
        default=None,
        description="The exact substring the agent considers hallucinated.",
    )
    corrected_text: Optional[str] = Field(
        default=None,
        description="The agent's proposed correction for the hallucinated span.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional chain-of-thought reasoning the agent provides.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Observation
# ──────────────────────────────────────────────────────────────────────────────
class HallucinationObservation(BaseObservation):
    """Observation returned to the agent after each step / reset.

    Provides the passage to analyse, context about the source, and feedback
    from previous actions.
    """

    task_id: Optional[TaskID] = Field(
        default=None,
        description="Identifier for the current task.",
    )
    difficulty: Optional[Difficulty] = Field(
        default=None,
        description="Difficulty level of the current task.",
    )
    passage: Optional[str] = Field(
        default=None,
        description="The LLM-generated text passage the agent must analyse.",
    )
    source_context: Optional[str] = Field(
        default=None,
        description="Reference context or source material the passage should be faithful to.",
    )
    num_hallucinations: Optional[int] = Field(
        default=None,
        description="Hint: number of hallucinations in the passage (provided for easy tasks).",
    )
    step_feedback: Optional[str] = Field(
        default=None,
        description="Textual feedback from the last action.",
    )
    steps_remaining: Optional[int] = Field(
        default=None,
        description="Number of steps the agent has left.",
    )
    cumulative_reward: Optional[float] = Field(
        default=None,
        description="Running total of rewards earned so far in this episode.",
    )
    action_history: List[str] = Field(
        default_factory=list,
        description="Summary of actions taken so far.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
class HallucinationState(BaseState):
    """Internal environment state exposed via state()."""

    task_id: Optional[TaskID] = Field(
        default=None,
        description="Currently active task.",
    )
    difficulty: Optional[Difficulty] = Field(
        default=None,
        description="Difficulty of the current task.",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Total reward accumulated this episode.",
    )
    detections_submitted: int = Field(
        default=0,
        description="Number of detection actions taken.",
    )
    classifications_submitted: int = Field(
        default=0,
        description="Number of classification actions taken.",
    )
    corrections_submitted: int = Field(
        default=0,
        description="Number of correction actions taken.",
    )
    is_done: bool = Field(
        default=False,
        description="Whether the episode has ended.",
    )
