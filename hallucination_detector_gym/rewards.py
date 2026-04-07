"""
Hallucination Detector Gym — Reward Engine.

Computes partial-progress rewards for each agent action. Provides dense signal
across the full trajectory (not just binary end-of-episode).

Key design decisions:
- Annotation deduplication: Each ground-truth annotation can only be rewarded
  once per action type (detect/classify/correct), preventing exploit via
  repeated identical actions.
- Step efficiency bonus: Fewer steps to achieve the same score earns a bonus.
- Sequence-aware span matching: Uses both Jaccard (token-set) and longest
  common subsequence (LCS) ratio, averaged for a balanced metric.
"""

from __future__ import annotations

import structlog
from typing import List, Optional

from .constants import (
    ActionType,
    GRADER_SCORE_MAX,
    GRADER_SCORE_MIN,
    HallucinationType,
    MAX_STEPS_PER_EPISODE,
    BONUS_STEP_EFFICIENCY,
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


def _lcs_ratio(a: str, b: str) -> float:
    """Compute longest common subsequence ratio between two strings.

    Uses token-level LCS to reward sequence-correct span matches,
    preventing bag-of-words exploitation.

    Args:
        a: First text string.
        b: Second text string.

    Returns:
        A float in [0, 1] representing 2*LCS_length / (len_a + len_b).
    """
    tokens_a = _normalize_text(a).split()
    tokens_b = _normalize_text(b).split()
    if not tokens_a or not tokens_b:
        return 0.0
    m, n = len(tokens_a), len(tokens_b)
    # Space-optimised LCS using two rows
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if tokens_a[i - 1] == tokens_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs_len = prev[n]
    return (2.0 * lcs_len) / (m + n)


def _span_overlap_ratio(predicted: str, ground_truth: str) -> float:
    """Compute combined token-level overlap ratio between predicted and ground truth spans.

    Uses the average of Jaccard similarity (token-set) and LCS ratio
    (sequence-aware) for a balanced metric that resists bag-of-words gaming.

    Args:
        predicted: The span predicted by the agent.
        ground_truth: The ground truth hallucinated span.

    Returns:
        A float in [0, 1] representing the combined similarity score.
    """
    pred_tokens = set(_normalize_text(predicted).split())
    gt_tokens = set(_normalize_text(ground_truth).split())
    if not gt_tokens:
        return 0.0
    intersection = pred_tokens & gt_tokens
    union = pred_tokens | gt_tokens
    jaccard = len(intersection) / len(union) if union else 0.0
    lcs = _lcs_ratio(predicted, ground_truth)
    return (jaccard + lcs) / 2.0


def _find_best_matching_annotation(
    action: HallucinationAction,
    annotations: List[HallucinationAnnotation],
    exclude_indices: Optional[set[int]] = None,
) -> tuple[Optional[HallucinationAnnotation], float, int]:
    """Find the annotation that best matches the agent's action.

    Args:
        action: The agent's action containing predicted span.
        annotations: Ground truth annotations.
        exclude_indices: Annotation indices to skip (already addressed).

    Returns:
        Tuple of (best_annotation, overlap_score, annotation_index).
        If no span provided, returns (first non-excluded annotation, 0.0, idx).
    """
    if not annotations:
        return None, 0.0, -1

    exclude = exclude_indices or set()

    if action.hallucinated_span is None:
        for idx, ann in enumerate(annotations):
            if idx not in exclude:
                return ann, 0.0, idx
        return annotations[0], 0.0, 0

    best_annotation = None
    best_overlap = 0.0
    best_idx = -1
    for idx, ann in enumerate(annotations):
        if idx in exclude:
            continue
        overlap = _span_overlap_ratio(action.hallucinated_span, ann.hallucinated_span)
        if overlap > best_overlap or best_annotation is None:
            best_overlap = overlap
            best_annotation = ann
            best_idx = idx

    if best_annotation is None:
        # All excluded; fall back to best overall match
        for idx, ann in enumerate(annotations):
            overlap = _span_overlap_ratio(action.hallucinated_span, ann.hallucinated_span)
            if overlap > best_overlap or best_annotation is None:
                best_overlap = overlap
                best_annotation = ann
                best_idx = idx

    return best_annotation, best_overlap, best_idx


class RewardEngine:
    """Stateful reward calculator for a single episode.

    Tracks which annotations have been correctly identified to avoid
    double-counting rewards. Each annotation can only be rewarded once
    per action type (detect, classify, correct).
    """

    def __init__(self, task: TaskDefinition) -> None:
        """Initialize reward engine for a given task.

        Args:
            task: The task definition with ground truth annotations.
        """
        self._task = task
        self._detected_annotations: set[int] = set()
        self._classified_annotations: set[int] = set()
        self._corrected_annotations: set[int] = set()
        self._action_history: list[str] = []
        self._cumulative_reward: float = 0.0
        self._step_count: int = 0

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
        self._step_count += 1

        # Build a content-aware action representation for repeat detection
        action_repr = action.action_type.value
        if action.hallucinated_span:
            action_repr += f":{_normalize_text(action.hallucinated_span)}"
        if action.hallucination_type:
            ht = action.hallucination_type
            type_val = ht.value if hasattr(ht, "value") else str(ht)
            action_repr += f":{type_val}"

        # Penalize repeated identical actions (same type AND content 3x in a row)
        recent = self._action_history[-3:] if len(self._action_history) >= 3 else []
        if len(recent) >= 3 and all(a == action_repr for a in recent):
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
            # Award efficiency bonus for finishing early
            if self._step_count < MAX_STEPS_PER_EPISODE:
                efficiency = 1.0 - (self._step_count / MAX_STEPS_PER_EPISODE)
                eff_bonus = BONUS_STEP_EFFICIENCY * efficiency
                reward += eff_bonus
                feedback_parts.append(
                    f"Episode submitted. Efficiency bonus: +{eff_bonus:.3f}."
                )
            else:
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
        """Handle a 'detect' action with annotation deduplication.

        Each annotation can only be detected once. Subsequent detect actions
        targeting the same annotation receive no reward.

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
                best_ann, overlap, ann_idx = _find_best_matching_annotation(
                    action, annotations, self._detected_annotations
                )
                if ann_idx in self._detected_annotations:
                    feedback_parts.append("This hallucination has already been detected.")
                elif overlap > 0.3:
                    span_reward = REWARD_CORRECT_SPAN * overlap
                    reward += span_reward
                    if ann_idx >= 0:
                        self._detected_annotations.add(ann_idx)
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
        """Handle a 'classify' action with annotation deduplication.

        Each annotation can only be classified once.

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

        best_ann, overlap, ann_idx = _find_best_matching_annotation(
            action, annotations, self._classified_annotations
        )
        if best_ann is None:
            feedback_parts.append("No annotations to classify against.")
            return reward, feedback_parts

        if ann_idx in self._classified_annotations:
            feedback_parts.append("This hallucination has already been classified.")
            return reward, feedback_parts

        if action.hallucination_type == best_ann.hallucination_type:
            reward += REWARD_CORRECT_CLASSIFICATION
            self._classified_annotations.add(ann_idx)
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
        """Handle a 'correct' action with annotation deduplication.

        Each annotation can only be corrected once.

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

        best_ann, span_overlap, ann_idx = _find_best_matching_annotation(
            action, annotations, self._corrected_annotations
        )
        if best_ann is None:
            feedback_parts.append("No annotations to correct against.")
            return reward, feedback_parts

        if ann_idx in self._corrected_annotations:
            feedback_parts.append("This hallucination has already been corrected.")
            return reward, feedback_parts

        correction_overlap = _span_overlap_ratio(
            action.corrected_text, best_ann.corrected_text
        )

        if correction_overlap > 0.3:
            correction_reward = REWARD_CORRECT_CORRECTION * correction_overlap
            reward += correction_reward
            self._corrected_annotations.add(ann_idx)
            feedback_parts.append(
                f"Correction partially correct (similarity={correction_overlap:.2f})."
            )
        else:
            feedback_parts.append("Correction did not match expected fix.")

        return reward, feedback_parts

    def get_final_score(self) -> float:
        """Compute a normalised final score in (0, 1) for grading.

        Returns:
            Score strictly between GRADER_SCORE_MIN and GRADER_SCORE_MAX.
        """
        num_annotations = len(self._task.annotations)
        if num_annotations == 0:
            return GRADER_SCORE_MAX if self._cumulative_reward >= 0 else GRADER_SCORE_MIN

        # Max theoretical reward per annotation:
        # detect(0.3) + span(0.2) + classify(0.3) + correct(0.2) = 1.0
        max_reward = num_annotations * 1.0
        normalised = max(0.0, min(1.0, self._cumulative_reward / max_reward))
        clamped = max(GRADER_SCORE_MIN, min(GRADER_SCORE_MAX, normalised))
        return round(clamped, 4)
