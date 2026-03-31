"""
Hallucination Detector Gym — Task Graders.

Programmatic graders that score agent performance on each task.
Scores range from 0.0 to 1.0 with deterministic, reproducible criteria.
"""

from __future__ import annotations

import structlog
from typing import List

from .constants import GRADER_SCORE_MAX, GRADER_SCORE_MIN, TaskID
from .models import HallucinationAction
from .rewards import RewardEngine
from .tasks import TaskDefinition, get_task

logger = structlog.get_logger(__name__)


class TaskGrader:
    """Deterministic grader for a single task.

    Runs through a sequence of agent actions and produces a normalised
    score in [0.0, 1.0].
    """

    def __init__(self, task_id: TaskID) -> None:
        """Initialize grader for a specific task.

        Args:
            task_id: The task to grade.
        """
        self._task: TaskDefinition = get_task(task_id)
        self._task_id = task_id

    @property
    def task(self) -> TaskDefinition:
        """The task definition being graded."""
        return self._task

    def grade(self, actions: List[HallucinationAction]) -> float:
        """Grade a sequence of agent actions.

        Args:
            actions: Ordered list of actions the agent took during the episode.

        Returns:
            A score in [0.0, 1.0].
        """
        engine = RewardEngine(self._task)

        for action in actions:
            engine.compute_reward(action)

        score = engine.get_final_score()
        clamped = max(GRADER_SCORE_MIN, min(GRADER_SCORE_MAX, score))

        logger.info(
            "task_graded",
            task_id=self._task_id.value,
            num_actions=len(actions),
            raw_score=score,
            clamped_score=clamped,
        )

        return clamped


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: grade all tasks at once
# ──────────────────────────────────────────────────────────────────────────────
def grade_all_tasks(
    actions_per_task: dict[TaskID, List[HallucinationAction]],
) -> dict[TaskID, float]:
    """Grade all tasks and return scores.

    Args:
        actions_per_task: Mapping from TaskID to the list of actions taken.

    Returns:
        Mapping from TaskID to score in [0.0, 1.0].
    """
    results: dict[TaskID, float] = {}
    for task_id, actions in actions_per_task.items():
        grader = TaskGrader(task_id)
        results[task_id] = grader.grade(actions)
    return results
