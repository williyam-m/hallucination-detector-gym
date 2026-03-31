"""
Hallucination Detector Gym — Package Exports.
"""

from .constants import (
    ActionType,
    Difficulty,
    HallucinationType,
    TaskID,
)
from .models import (
    HallucinationAction,
    HallucinationObservation,
    HallucinationState,
)
from .tasks import get_task, list_tasks, TaskDefinition
from .graders import TaskGrader, grade_all_tasks
from .rewards import RewardEngine

__all__ = [
    # Constants / Enums
    "ActionType",
    "Difficulty",
    "HallucinationType",
    "TaskID",
    # Models
    "HallucinationAction",
    "HallucinationObservation",
    "HallucinationState",
    # Tasks
    "TaskDefinition",
    "get_task",
    "list_tasks",
    # Grading
    "TaskGrader",
    "grade_all_tasks",
    # Rewards
    "RewardEngine",
]
