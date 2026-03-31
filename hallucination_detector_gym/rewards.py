"""
Hallucination Detector Gym — Reward Engine.

Computes partial-progress rewards for each agent action. Provides dense signal
across the full trajectory (not just binary end-of-episode).
"""

from __future__ import annotations

import structlog
from typing import List, Optional

from .constants import (
    ActionType,
    HallucinationType,
    PENALTY_NOOP_WHEN_HALLUCINATION,
    PENALTY_REPEATED_ACTION,
    PENALTY_WRONG_CLASSIFICATION,
    PENALTY_WRONG_DETECTION,
    REWARD_CORRECT_CLASSIFICATION,
    REWARD_CORRECT_CORRECTION,
    REWARD_CORRECT_DETECTION,
    REWARD_CORRECT_SPAN,
)
from .models import HallucinationAction
from .tasks import HallucinationAnnotation, TaskDefinition

logger = structlog.get_logger(__name__)


def _normalize_text(text: str) -> str:
    """Lowercase and strip whitespace for fuzzy comparison."""
    return " ".join(text.lower().split())


def _span_overlap_ratio(predicted: str, ground_truth: str) -> float:
    """Compute token-level overlap ratio between predicted and ground truth spans.

    Args:
        predicted: The span predicted by the agent.
        ground_truth: The ground truth hallucinated span.

    Returns:
        A float in [0, 1] representing the Jaccard similarity of token sets.
    """
    pred_tokens = set(_normalize_text(predicted).split())
    gt_tokens = set(_normalize_text(ground_truth).split())
    if not gt_tokens:
        return 0.0
    intersection = pred_tokens & gt_tokens
    union = pred_tokens | gt_tokens
    return len(intersection) / len(union) if union else 0.0


def _find_best_matching_annotation(
    action: HallucinationAction,
    annotations: List[HallucinationAnnotation],
) -> tuple[Optional[HallucinationAnnotation], float]:
    """Find the annotation that best matches the agent's action.

    Args:
        action: The agent's action containing predicted span.
        annotations: Ground truth annotations.

    Returns:
        Tuple of (best_annotation, overlap_score). If no span provided,
        returns (first annotation, 0.0).
    """
    if not annotations:
        return None, 0.0

    if action.hallucinated_span is None:
        return annotations[0], 0.0

    best_annotation = annotations[0]
    best_overlap = 0.0
    for ann in annotations:
        overlap = _span_overlap_ratio(action.hallucinated_span, ann.hallucinated_span)
        if overlap > best_overlap:
            best_overlap = overlap
            best_annotation = ann
    return best_annotation, best_overlap


class RewardEngine:
    """Stateful reward calculator for a single episode.

    Tracks which annotations have been correctly identified to avoid
    double-counting rewards. Provides partial progress signals.
    """

    def __init__(self, task: TaskDefinition) -> None:
        """Initialize reward engine for a given task.

        Args:
            task: The task definition with ground truth annotations.
        """
        self._task = task
        self._found_annotations: set[int] = set()
        self._action_history: list[str] = []
        self._cumulative_reward: float = 0.0

    @property
    def cumulative_reward(self) -> float:
        """Total reward accumulated so far."""
        return self._cumulative_reward

    def compute_reward(self, action: HallucinationAction) -> tuple[float, str]:
        """Compute reward for a single agent action.

        Args:
            action: The agent's action.

        Returns:
            Tuple of (reward_value, feedback_message).
        """
        reward = 0.0
        feedback_parts: list[str] = []
        action_repr = f"{action.action_type.value}"

        # Penalize repeated identical actions
        if action_repr in self._action_history[-3:] if len(self._action_history) >= 3 else False:
            reward += PENALTY_REPEATED_ACTION
            feedback_parts.append("Repeated action penalty applied.")

        self._action_history.append(action_repr)

        annotations = self._task.annotations
        has_hallucinations = len(annotations) > 0

        if action.action_type == ActionType.NOOP:
            if has_hallucinations:
                reward += PENALTY_NOOP_WHEN_HALLUCINATION
                feedback_parts.append("Skipped, but hallucinations remain undetected.")
            else:
                feedback_parts.append("No action taken.")

        elif action.action_type == ActionType.DETECT:
            reward, feedback_parts = self._handle_detect(
                action, annotations, has_hallucinations, reward, feedback_parts
            )

        elif action.action_type == ActionType.CLASSIFY:
            reward, feedback_parts = self._handle_classify(
                action, annotations, reward, feedback_parts
            )

        elif action.action_type == ActionType.CORRECT:
            reward, feedback_parts = self._handle_correct(
                action, annotations, reward, feedback_parts
            )

        elif action.action_type == ActionType.SUBMIT:
            feedback_parts.append("Episode submitted.")

        self._cumulative_reward += reward
        feedback = " ".join(feedback_parts)

        logger.info(
            "reward_computed",
            action_type=action.action_type.value,
            reward=round(reward, 4),
            cumulative=round(self._cumulative_reward, 4),
            feedback=feedback,
        )

        return reward, feedback

    def _handle_detect(
        self,
        action: HallucinationAction,
        annotations: List[HallucinationAnnotation],
        has_hallucinations: bool,
        reward: float,
        feedback_parts: list[str],
    ) -> tuple[float, list[str]]:
        """Handle a 'detect' action.

        Args:
            action: The agent's action.
            annotations: Ground truth annotations.
            has_hallucinations: Whether the passage has hallucinations.
            reward: Current reward accumulator.
            feedback_parts: Feedback message parts.

        Returns:
            Updated (reward, feedback_parts).
        """
        if action.hallucination_detected is None:
            feedback_parts.append("Detection action requires 'hallucination_detected' field.")
            return reward, feedback_parts

        if action.hallucination_detected == has_hallucinations:
            reward += REWARD_CORRECT_DETECTION
            feedback_parts.append("Correct detection!")

            # Bonus for identifying the span
            if action.hallucinated_span and has_hallucinations:
                best_ann, overlap = _find_best_matching_annotation(action, annotations)
                if overlap > 0.3:
                    span_reward = REWARD_CORRECT_SPAN * overlap
                    reward += span_reward
                    ann_idx = annotations.index(best_ann) if best_ann else -1
                    if ann_idx >= 0:
                        self._found_annotations.add(ann_idx)
                    feedback_parts.append(
                        f"Span partially matched (overlap={overlap:.2f})."
                    )
                else:
                    feedback_parts.append("Span did not match any known hallucination.")
        else:
            reward += PENALTY_WRONG_DETECTION
            if has_hallucinations:
                feedback_parts.append("Missed hallucination in the passage.")
            else:
                feedback_parts.append("False positive: no hallucination present.")

        return reward, feedback_parts

    def _handle_classify(
        self,
        action: HallucinationAction,
        annotations: List[HallucinationAnnotation],
        reward: float,
        feedback_parts: list[str],
    ) -> tuple[float, list[str]]:
        """Handle a 'classify' action.

        Args:
            action: The agent's action.
            annotations: Ground truth annotations.
            reward: Current reward accumulator.
            feedback_parts: Feedback message parts.

        Returns:
            Updated (reward, feedback_parts).
        """
        if action.hallucination_type is None:
            feedback_parts.append("Classification action requires 'hallucination_type' field.")
            return reward, feedback_parts

        best_ann, overlap = _find_best_matching_annotation(action, annotations)
        if best_ann is None:
            feedback_parts.append("No annotations to classify against.")
            return reward, feedback_parts

        if action.hallucination_type == best_ann.hallucination_type:
            reward += REWARD_CORRECT_CLASSIFICATION
            feedback_parts.append(
                f"Correct classification: {action.hallucination_type.value}."
            )
        else:
            reward += PENALTY_WRONG_CLASSIFICATION
            feedback_parts.append(
                f"Incorrect classification. "
                f"You said '{action.hallucination_type.value}'."
            )

        return reward, feedback_parts

    def _handle_correct(
        self,
        action: HallucinationAction,
        annotations: List[HallucinationAnnotation],
        reward: float,
        feedback_parts: list[str],
    ) -> tuple[float, list[str]]:
        """Handle a 'correct' action.

        Args:
            action: The agent's action.
            annotations: Ground truth annotations.
            reward: Current reward accumulator.
            feedback_parts: Feedback message parts.

        Returns:
            Updated (reward, feedback_parts).
        """
        if action.corrected_text is None:
            feedback_parts.append("Correction action requires 'corrected_text' field.")
            return reward, feedback_parts

        best_ann, span_overlap = _find_best_matching_annotation(action, annotations)
        if best_ann is None:
            feedback_parts.append("No annotations to correct against.")
            return reward, feedback_parts

        correction_overlap = _span_overlap_ratio(
            action.corrected_text, best_ann.corrected_text
        )

        if correction_overlap > 0.3:
            correction_reward = REWARD_CORRECT_CORRECTION * correction_overlap
            reward += correction_reward
            feedback_parts.append(
                f"Correction partially correct (similarity={correction_overlap:.2f})."
            )
        else:
            feedback_parts.append("Correction did not match expected fix.")

        return reward, feedback_parts

    def get_final_score(self) -> float:
        """Compute a normalised final score in [0, 1] for grading.

        Returns:
            Score between 0.0 and 1.0.
        """
        num_annotations = len(self._task.annotations)
        if num_annotations == 0:
            return 1.0 if self._cumulative_reward >= 0 else 0.0

        # Max theoretical reward per annotation:
        # detect(0.3) + span(0.2) + classify(0.3) + correct(0.2) = 1.0
        max_reward = num_annotations * 1.0
        normalised = max(0.0, min(1.0, self._cumulative_reward / max_reward))
        return round(normalised, 4)
