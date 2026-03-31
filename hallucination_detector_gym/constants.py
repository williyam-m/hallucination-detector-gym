"""
Hallucination Detector Gym — Constants and Configuration.

All configurable values are centralized here. No hardcoded magic numbers elsewhere.
"""

from enum import Enum, unique
from typing import Final


# ──────────────────────────────────────────────────────────────────────────────
# Environment Metadata
# ──────────────────────────────────────────────────────────────────────────────
ENV_NAME: Final[str] = "hallucination_detector_gym"
ENV_VERSION: Final[str] = "1.0.0"
ENV_DESCRIPTION: Final[str] = (
    "An OpenEnv environment where AI agents detect, classify, and correct "
    "hallucinations in LLM-generated text. Covers factual errors, entity "
    "fabrication, and logical inconsistencies."
)

# ──────────────────────────────────────────────────────────────────────────────
# Episode Configuration
# ──────────────────────────────────────────────────────────────────────────────
MAX_STEPS_PER_EPISODE: Final[int] = 10
DEFAULT_SEED: Final[int] = 42

# ──────────────────────────────────────────────────────────────────────────────
# Reward Configuration
# ──────────────────────────────────────────────────────────────────────────────
REWARD_CORRECT_DETECTION: Final[float] = 0.3
REWARD_CORRECT_CLASSIFICATION: Final[float] = 0.3
REWARD_CORRECT_SPAN: Final[float] = 0.2
REWARD_CORRECT_CORRECTION: Final[float] = 0.2
PENALTY_WRONG_DETECTION: Final[float] = -0.15
PENALTY_WRONG_CLASSIFICATION: Final[float] = -0.10
PENALTY_NOOP_WHEN_HALLUCINATION: Final[float] = -0.05
PENALTY_REPEATED_ACTION: Final[float] = -0.05

# Grader score bounds
GRADER_SCORE_MIN: Final[float] = 0.0
GRADER_SCORE_MAX: Final[float] = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────
@unique
class HallucinationType(str, Enum):
    """Categories of hallucinations an agent can identify."""

    FACTUAL_ERROR = "factual_error"
    ENTITY_FABRICATION = "entity_fabrication"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    NONE = "none"


@unique
class Difficulty(str, Enum):
    """Task difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@unique
class ActionType(str, Enum):
    """Available agent actions."""

    DETECT = "detect"
    CLASSIFY = "classify"
    CORRECT = "correct"
    SUBMIT = "submit"
    NOOP = "noop"


@unique
class TaskID(str, Enum):
    """Identifiers for the three required tasks."""

    TASK_EASY = "task_easy_factual"
    TASK_MEDIUM = "task_medium_entity"
    TASK_HARD = "task_hard_multi"
