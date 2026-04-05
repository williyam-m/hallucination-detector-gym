"""
Tests for the Hallucination Detector Gym environment.

Covers: models, reward engine, graders, environment lifecycle, and task registry.
"""

from __future__ import annotations

import pytest

from hallucination_detector_gym.constants import (
    ActionType,
    Difficulty,
    HallucinationType,
    TaskID,
)
from hallucination_detector_gym.graders import TaskGrader
from hallucination_detector_gym.models import (
    HallucinationAction,
    HallucinationObservation,
    HallucinationState,
)
from hallucination_detector_gym.rewards import RewardEngine, _span_overlap_ratio
from hallucination_detector_gym.tasks import (
    get_task,
    list_tasks,
    TASK_EASY,
    TASK_HARD,
    TASK_MEDIUM,
    TASK_REGISTRY,
)


# ──────────────────────────────────────────────────────────────────────────────
# Task Registry Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestTaskRegistry:
    """Tests for the task registry and task definitions."""

    def test_registry_has_three_tasks(self) -> None:
        """Must have at least 3 tasks."""
        assert len(TASK_REGISTRY) >= 3

    def test_tasks_cover_all_difficulties(self) -> None:
        """Tasks must span easy, medium, and hard."""
        difficulties = {t.difficulty for t in TASK_REGISTRY.values()}
        assert Difficulty.EASY in difficulties
        assert Difficulty.MEDIUM in difficulties
        assert Difficulty.HARD in difficulties

    def test_get_task_returns_correct_task(self) -> None:
        """get_task should return the correct definition."""
        task = get_task(TaskID.TASK_EASY)
        assert task.task_id == TaskID.TASK_EASY
        assert task.difficulty == Difficulty.EASY

    def test_get_task_raises_on_invalid_id(self) -> None:
        """get_task should raise KeyError for unknown IDs."""
        with pytest.raises(KeyError):
            get_task("nonexistent_task")  # type: ignore[arg-type]

    def test_list_tasks_sorted_by_difficulty(self) -> None:
        """list_tasks should return tasks in difficulty order."""
        tasks = list_tasks()
        assert tasks[0].difficulty == Difficulty.EASY
        assert tasks[-1].difficulty == Difficulty.HARD

    def test_each_task_has_annotations(self) -> None:
        """Every task must have at least one annotation."""
        for task in TASK_REGISTRY.values():
            assert len(task.annotations) >= 1, f"Task {task.task_id} has no annotations"

    def test_easy_task_has_hint(self) -> None:
        """Easy task provides a hint about number of hallucinations."""
        assert TASK_EASY.hint_num_hallucinations is not None
        assert TASK_EASY.hint_num_hallucinations == 1

    def test_hard_task_has_no_hint(self) -> None:
        """Hard task should not provide hints."""
        assert TASK_HARD.hint_num_hallucinations is None


# ──────────────────────────────────────────────────────────────────────────────
# Model Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestModels:
    """Tests for Pydantic models."""

    def test_action_creation_detect(self) -> None:
        """Create a detect action."""
        action = HallucinationAction(
            action_type=ActionType.DETECT,
            hallucination_detected=True,
            hallucinated_span="Munich, Germany",
        )
        assert action.action_type == ActionType.DETECT
        assert action.hallucination_detected is True

    def test_action_creation_classify(self) -> None:
        """Create a classify action."""
        action = HallucinationAction(
            action_type=ActionType.CLASSIFY,
            hallucination_type=HallucinationType.FACTUAL_ERROR,
            hallucinated_span="Munich, Germany",
        )
        assert action.hallucination_type == HallucinationType.FACTUAL_ERROR

    def test_action_creation_noop(self) -> None:
        """Create a noop action."""
        action = HallucinationAction(action_type=ActionType.NOOP)
        assert action.action_type == ActionType.NOOP
        assert action.hallucination_detected is None

    def test_observation_defaults(self) -> None:
        """Observation should have sensible defaults."""
        obs = HallucinationObservation()
        assert obs.done is False
        assert obs.reward is None
        assert obs.action_history == []

    def test_state_defaults(self) -> None:
        """State should have sensible defaults."""
        state = HallucinationState()
        assert state.step_count == 0
        assert state.cumulative_reward == 0.0
        assert state.is_done is False


# ──────────────────────────────────────────────────────────────────────────────
# Reward Engine Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestRewardEngine:
    """Tests for the reward computation engine."""

    def test_correct_detection_positive_reward(self) -> None:
        """Correct detection should yield positive reward."""
        engine = RewardEngine(TASK_EASY)
        action = HallucinationAction(
            action_type=ActionType.DETECT,
            hallucination_detected=True,
            hallucinated_span="Munich, Germany",
        )
        reward, feedback = engine.compute_reward(action)
        assert reward > 0.0
        assert "Correct" in feedback

    def test_wrong_detection_negative_reward(self) -> None:
        """Wrong detection should yield negative reward."""
        engine = RewardEngine(TASK_EASY)
        action = HallucinationAction(
            action_type=ActionType.DETECT,
            hallucination_detected=False,
        )
        reward, feedback = engine.compute_reward(action)
        assert reward < 0.0

    def test_correct_classification_positive_reward(self) -> None:
        """Correct classification should yield positive reward."""
        engine = RewardEngine(TASK_EASY)
        action = HallucinationAction(
            action_type=ActionType.CLASSIFY,
            hallucination_type=HallucinationType.FACTUAL_ERROR,
            hallucinated_span="Munich, Germany",
        )
        reward, feedback = engine.compute_reward(action)
        assert reward > 0.0

    def test_wrong_classification_negative_reward(self) -> None:
        """Wrong classification should yield negative reward."""
        engine = RewardEngine(TASK_EASY)
        action = HallucinationAction(
            action_type=ActionType.CLASSIFY,
            hallucination_type=HallucinationType.ENTITY_FABRICATION,
            hallucinated_span="Munich, Germany",
        )
        reward, feedback = engine.compute_reward(action)
        assert reward < 0.0

    def test_noop_penalty_when_hallucinations_exist(self) -> None:
        """Noop should be penalised when hallucinations are present."""
        engine = RewardEngine(TASK_EASY)
        action = HallucinationAction(action_type=ActionType.NOOP)
        reward, _ = engine.compute_reward(action)
        assert reward < 0.0

    def test_cumulative_reward_tracks(self) -> None:
        """Cumulative reward should accumulate across actions."""
        engine = RewardEngine(TASK_EASY)

        action1 = HallucinationAction(
            action_type=ActionType.DETECT,
            hallucination_detected=True,
            hallucinated_span="Munich, Germany",
        )
        r1, _ = engine.compute_reward(action1)

        action2 = HallucinationAction(
            action_type=ActionType.CLASSIFY,
            hallucination_type=HallucinationType.FACTUAL_ERROR,
            hallucinated_span="Munich, Germany",
        )
        r2, _ = engine.compute_reward(action2)

        assert abs(engine.cumulative_reward - (r1 + r2)) < 1e-6

    def test_final_score_in_range(self) -> None:
        """Final score must be in [0, 1]."""
        engine = RewardEngine(TASK_EASY)
        score = engine.get_final_score()
        assert 0.0 <= score <= 1.0

    def test_span_overlap_ratio_identical(self) -> None:
        """Identical spans should have overlap 1.0."""
        assert _span_overlap_ratio("Munich, Germany", "Munich, Germany") == 1.0

    def test_span_overlap_ratio_disjoint(self) -> None:
        """Completely different spans should have overlap ~0."""
        overlap = _span_overlap_ratio("hello world", "foo bar baz")
        assert overlap == 0.0

    def test_span_overlap_ratio_partial(self) -> None:
        """Partially overlapping spans should have overlap between 0 and 1."""
        overlap = _span_overlap_ratio("Munich Germany city", "Munich Germany")
        assert 0.0 < overlap < 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Grader Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestGrader:
    """Tests for the task grader."""

    def test_grader_score_in_range(self) -> None:
        """Grader must return score in [0.0, 1.0]."""
        grader = TaskGrader(TaskID.TASK_EASY)
        actions = [
            HallucinationAction(
                action_type=ActionType.DETECT,
                hallucination_detected=True,
            ),
            HallucinationAction(action_type=ActionType.SUBMIT),
        ]
        score = grader.grade(actions)
        assert 0.0 <= score <= 1.0

    def test_perfect_easy_task_high_score(self) -> None:
        """Perfect actions on easy task should yield high score."""
        grader = TaskGrader(TaskID.TASK_EASY)
        actions = [
            HallucinationAction(
                action_type=ActionType.DETECT,
                hallucination_detected=True,
                hallucinated_span="Munich, Germany",
            ),
            HallucinationAction(
                action_type=ActionType.CLASSIFY,
                hallucination_type=HallucinationType.FACTUAL_ERROR,
                hallucinated_span="Munich, Germany",
            ),
            HallucinationAction(
                action_type=ActionType.CORRECT,
                hallucinated_span="Munich, Germany",
                corrected_text="Ulm, in the Kingdom of Württemberg in the German Empire",
            ),
            HallucinationAction(action_type=ActionType.SUBMIT),
        ]
        score = grader.grade(actions)
        assert score > 0.7

    def test_empty_actions_score_zero(self) -> None:
        """No actions should produce score 0."""
        grader = TaskGrader(TaskID.TASK_EASY)
        score = grader.grade([])
        assert score == 0.0

    def test_all_wrong_low_score(self) -> None:
        """All wrong actions should yield low score."""
        grader = TaskGrader(TaskID.TASK_EASY)
        actions = [
            HallucinationAction(
                action_type=ActionType.DETECT,
                hallucination_detected=False,
            ),
            HallucinationAction(
                action_type=ActionType.CLASSIFY,
                hallucination_type=HallucinationType.LOGICAL_INCONSISTENCY,
            ),
            HallucinationAction(action_type=ActionType.SUBMIT),
        ]
        score = grader.grade(actions)
        assert score < 0.3

    def test_grader_deterministic(self) -> None:
        """Grading same actions twice should produce identical scores."""
        grader = TaskGrader(TaskID.TASK_MEDIUM)
        actions = [
            HallucinationAction(
                action_type=ActionType.DETECT,
                hallucination_detected=True,
                hallucinated_span="Berlin Institute of Genomic Sciences",
            ),
            HallucinationAction(action_type=ActionType.SUBMIT),
        ]
        score1 = grader.grade(actions)
        score2 = grader.grade(actions)
        assert score1 == score2

    def test_medium_task_grading(self) -> None:
        """Medium task should be gradable."""
        grader = TaskGrader(TaskID.TASK_MEDIUM)
        score = grader.grade([
            HallucinationAction(
                action_type=ActionType.DETECT,
                hallucination_detected=True,
                hallucinated_span="Berlin Institute of Genomic Sciences",
            ),
            HallucinationAction(
                action_type=ActionType.CLASSIFY,
                hallucination_type=HallucinationType.ENTITY_FABRICATION,
                hallucinated_span="Berlin Institute of Genomic Sciences",
            ),
            HallucinationAction(action_type=ActionType.SUBMIT),
        ])
        assert 0.0 <= score <= 1.0

    def test_hard_task_grading(self) -> None:
        """Hard task should be gradable."""
        grader = TaskGrader(TaskID.TASK_HARD)
        score = grader.grade([
            HallucinationAction(
                action_type=ActionType.DETECT,
                hallucination_detected=True,
                hallucinated_span="Cape Canaveral Space Center",
            ),
            HallucinationAction(action_type=ActionType.SUBMIT),
        ])
        assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Anti-Exploitation Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestAntiExploitation:
    """Tests for reward deduplication and anti-gaming measures."""

    def test_detect_double_counting_blocked(self) -> None:
        """Same detection action repeated should not award span bonus twice."""
        engine = RewardEngine(TASK_EASY)
        action = HallucinationAction(
            action_type=ActionType.DETECT,
            hallucination_detected=True,
            hallucinated_span="Munich, Germany",
        )
        r1, f1 = engine.compute_reward(action)
        r2, f2 = engine.compute_reward(action)
        # First detection gets full reward, second gets base detection but no span bonus
        assert r1 > r2
        assert "already been detected" in f2.lower()

    def test_classify_double_counting_blocked(self) -> None:
        """Same classification repeated should not award twice."""
        engine = RewardEngine(TASK_EASY)
        action = HallucinationAction(
            action_type=ActionType.CLASSIFY,
            hallucination_type=HallucinationType.FACTUAL_ERROR,
            hallucinated_span="Munich, Germany",
        )
        r1, _ = engine.compute_reward(action)
        r2, f2 = engine.compute_reward(action)
        assert r1 > 0
        assert r2 == 0.0 or "already been classified" in f2.lower()

    def test_correct_double_counting_blocked(self) -> None:
        """Same correction repeated should not award twice."""
        engine = RewardEngine(TASK_EASY)
        action = HallucinationAction(
            action_type=ActionType.CORRECT,
            hallucinated_span="Munich, Germany",
            corrected_text="Ulm, in the Kingdom of Württemberg in the German Empire",
        )
        r1, _ = engine.compute_reward(action)
        r2, f2 = engine.compute_reward(action)
        assert r1 > 0
        assert r2 == 0.0 or "already been corrected" in f2.lower()

    def test_step_efficiency_bonus(self) -> None:
        """Submit action should award efficiency bonus for early finish."""
        engine = RewardEngine(TASK_EASY)
        submit = HallucinationAction(action_type=ActionType.SUBMIT)
        reward, feedback = engine.compute_reward(submit)
        assert reward > 0
        assert "efficiency" in feedback.lower() or "bonus" in feedback.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Environment Lifecycle Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestEnvironmentLifecycle:
    """Tests for the HallucinationDetectorEnvironment class."""

    def test_reset_produces_clean_state(self) -> None:
        """Reset should produce a fresh environment state."""
        from server.hallucination_environment import HallucinationDetectorEnvironment
        env = HallucinationDetectorEnvironment()
        obs = env.reset(task_id="task_easy_factual")
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.passage is not None
        assert obs.source_context is not None
        assert env.state.step_count == 0

    def test_step_without_reset_returns_error(self) -> None:
        """Step before reset should return done=True with error."""
        from server.hallucination_environment import HallucinationDetectorEnvironment
        env = HallucinationDetectorEnvironment()
        obs = env.step(HallucinationAction(action_type=ActionType.NOOP))
        assert obs.done is True
        assert "not reset" in (obs.step_feedback or "").lower()

    def test_full_episode_lifecycle(self) -> None:
        """Full detect-classify-correct-submit episode should work."""
        from server.hallucination_environment import HallucinationDetectorEnvironment
        env = HallucinationDetectorEnvironment()
        env.reset(task_id="task_easy_factual")

        obs = env.step(HallucinationAction(
            action_type=ActionType.DETECT,
            hallucination_detected=True,
            hallucinated_span="Munich, Germany",
        ))
        assert obs.done is False
        assert obs.reward > 0

        obs = env.step(HallucinationAction(
            action_type=ActionType.CLASSIFY,
            hallucination_type=HallucinationType.FACTUAL_ERROR,
            hallucinated_span="Munich, Germany",
        ))
        assert obs.done is False

        obs = env.step(HallucinationAction(
            action_type=ActionType.CORRECT,
            hallucinated_span="Munich, Germany",
            corrected_text="Ulm, in the Kingdom of Württemberg",
        ))
        assert obs.done is False

        obs = env.step(HallucinationAction(action_type=ActionType.SUBMIT))
        assert obs.done is True
