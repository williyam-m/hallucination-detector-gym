"""
Hallucination Detector Gym — Task Definitions and Dataset.

Contains the curated passages with ground-truth hallucination annotations.
Each task has a difficulty level (easy / medium / hard) with increasing complexity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .constants import Difficulty, HallucinationType, TaskID


@dataclass(frozen=True)
class HallucinationAnnotation:
    """Ground-truth annotation for a single hallucination in a passage."""

    hallucination_type: HallucinationType
    hallucinated_span: str
    corrected_text: str
    explanation: str


@dataclass(frozen=True)
class TaskDefinition:
    """Full definition of one task (passage + ground truth)."""

    task_id: TaskID
    difficulty: Difficulty
    title: str
    description: str
    source_context: str
    passage: str
    annotations: List[HallucinationAnnotation]
    hint_num_hallucinations: Optional[int] = None


# ──────────────────────────────────────────────────────────────────────────────
# TASK 1 — EASY: Single Factual Error
# ──────────────────────────────────────────────────────────────────────────────
TASK_EASY = TaskDefinition(
    task_id=TaskID.TASK_EASY,
    difficulty=Difficulty.EASY,
    title="Simple Factual Error Detection",
    description=(
        "The agent must identify a single factual error in a short biographical "
        "passage. The source context provides the correct facts."
    ),
    source_context=(
        "Albert Einstein (1879–1955) was a German-born theoretical physicist. "
        "He developed the theory of relativity, one of the two pillars of modern "
        "physics. He received the Nobel Prize in Physics in 1921 for his "
        "explanation of the photoelectric effect. Einstein was born in Ulm, "
        "in the Kingdom of Württemberg in the German Empire."
    ),
    passage=(
        "Albert Einstein was a renowned theoretical physicist born in 1879 in "
        "Munich, Germany. He is best known for developing the theory of relativity. "
        "Einstein received the Nobel Prize in Physics in 1921 for his explanation "
        "of the photoelectric effect. He passed away in 1955."
    ),
    annotations=[
        HallucinationAnnotation(
            hallucination_type=HallucinationType.FACTUAL_ERROR,
            hallucinated_span="Munich, Germany",
            corrected_text="Ulm, in the Kingdom of Württemberg in the German Empire",
            explanation=(
                "Einstein was born in Ulm, not Munich. The passage incorrectly "
                "states his birthplace."
            ),
        ),
    ],
    hint_num_hallucinations=1,
)


# ──────────────────────────────────────────────────────────────────────────────
# TASK 2 — MEDIUM: Entity Fabrication + Factual Error
# ──────────────────────────────────────────────────────────────────────────────
TASK_MEDIUM = TaskDefinition(
    task_id=TaskID.TASK_MEDIUM,
    difficulty=Difficulty.MEDIUM,
    title="Entity Fabrication and Factual Error Detection",
    description=(
        "The agent must identify two hallucinations: a fabricated entity "
        "(non-existent institution) and a factual error in a research summary."
    ),
    source_context=(
        "CRISPR-Cas9 is a genome editing technology adapted from a natural "
        "defense mechanism in bacteria. Jennifer Doudna and Emmanuelle "
        "Charpentier published their landmark paper in Science in June 2012, "
        "demonstrating that Cas9 could be programmed with RNA to edit DNA. "
        "They were awarded the Nobel Prize in Chemistry in 2020. The technology "
        "has applications in treating genetic diseases like sickle cell disease "
        "and beta-thalassemia."
    ),
    passage=(
        "CRISPR-Cas9 is a revolutionary genome editing tool. Jennifer Doudna "
        "and Emmanuelle Charpentier, researchers at the Berlin Institute of "
        "Genomic Sciences, published their groundbreaking paper in Science in "
        "2012 showing that Cas9 can be guided by RNA to edit specific DNA "
        "sequences. They won the Nobel Prize in Physiology or Medicine in 2020 "
        "for this work. CRISPR is now being used to treat genetic conditions "
        "such as sickle cell disease."
    ),
    annotations=[
        HallucinationAnnotation(
            hallucination_type=HallucinationType.ENTITY_FABRICATION,
            hallucinated_span="Berlin Institute of Genomic Sciences",
            corrected_text=(
                "University of California, Berkeley (Doudna) and Umeå "
                "University / Max Planck Institute (Charpentier)"
            ),
            explanation=(
                "The 'Berlin Institute of Genomic Sciences' does not exist. "
                "Doudna was at UC Berkeley and Charpentier at Umeå University."
            ),
        ),
        HallucinationAnnotation(
            hallucination_type=HallucinationType.FACTUAL_ERROR,
            hallucinated_span="Nobel Prize in Physiology or Medicine",
            corrected_text="Nobel Prize in Chemistry",
            explanation=(
                "Doudna and Charpentier won the Nobel Prize in Chemistry, "
                "not Physiology or Medicine."
            ),
        ),
    ],
    hint_num_hallucinations=None,  # No hint for medium
)


# ──────────────────────────────────────────────────────────────────────────────
# TASK 3 — HARD: Multi-type Hallucinations with Logical Inconsistency
# ──────────────────────────────────────────────────────────────────────────────
TASK_HARD = TaskDefinition(
    task_id=TaskID.TASK_HARD,
    difficulty=Difficulty.HARD,
    title="Complex Multi-type Hallucination Detection",
    description=(
        "The agent must identify three hallucinations of different types: "
        "a factual error, an entity fabrication, and a logical inconsistency. "
        "No hints are provided."
    ),
    source_context=(
        "The Apollo 11 mission launched on July 16, 1969, from Kennedy Space "
        "Center. The crew consisted of Commander Neil Armstrong, Command Module "
        "Pilot Michael Collins, and Lunar Module Pilot Buzz Aldrin. Armstrong "
        "and Aldrin landed on the Moon on July 20, 1969, in the lunar module "
        "Eagle, while Collins orbited above in the command module Columbia. "
        "Armstrong was the first person to walk on the Moon. The mission "
        "returned to Earth on July 24, 1969, splashing down in the Pacific "
        "Ocean. The total mission duration was approximately 8 days, 3 hours. "
        "The Saturn V rocket that launched the mission was 363 feet tall."
    ),
    passage=(
        "The Apollo 11 mission, humanity's first Moon landing, launched on "
        "July 16, 1969, from Cape Canaveral Space Center. The crew included "
        "Commander Neil Armstrong, Command Module Pilot Michael Collins, and "
        "Lunar Module Pilot Buzz Aldrin. Armstrong and Aldrin descended to the "
        "lunar surface on July 20, 1969, aboard the Eagle module, while Collins "
        "remained in orbit. After spending approximately 21 hours on the surface, "
        "Armstrong became the first person to set foot on the Moon. Remarkably, "
        "Collins also briefly walked on the lunar surface during a secondary EVA "
        "before the crew reunited for the journey home. The mission concluded "
        "with a splashdown in the Atlantic Ocean on July 24, 1969. The Saturn V "
        "rocket used for the mission stood 363 feet tall."
    ),
    annotations=[
        HallucinationAnnotation(
            hallucination_type=HallucinationType.FACTUAL_ERROR,
            hallucinated_span="Cape Canaveral Space Center",
            corrected_text="Kennedy Space Center",
            explanation=(
                "While Cape Canaveral and Kennedy Space Center are adjacent, "
                "Apollo 11 launched from Kennedy Space Center (Launch Complex 39A), "
                "not Cape Canaveral Space Center."
            ),
        ),
        HallucinationAnnotation(
            hallucination_type=HallucinationType.ENTITY_FABRICATION,
            hallucinated_span=(
                "Collins also briefly walked on the lunar surface during a "
                "secondary EVA"
            ),
            corrected_text=(
                "Collins never walked on the Moon; he remained in lunar orbit "
                "aboard the command module Columbia for the entire mission."
            ),
            explanation=(
                "This is a complete fabrication. Michael Collins stayed in "
                "the command module and never set foot on the Moon."
            ),
        ),
        HallucinationAnnotation(
            hallucination_type=HallucinationType.LOGICAL_INCONSISTENCY,
            hallucinated_span="splashdown in the Atlantic Ocean",
            corrected_text="splashdown in the Pacific Ocean",
            explanation=(
                "The passage states the splashdown was in the Atlantic Ocean, "
                "contradicting the source which says Pacific Ocean. This is "
                "also logically inconsistent with the mission trajectory."
            ),
        ),
    ],
    hint_num_hallucinations=None,  # No hint for hard
)


# ──────────────────────────────────────────────────────────────────────────────
# Task Registry
# ──────────────────────────────────────────────────────────────────────────────
TASK_REGISTRY: dict[TaskID, TaskDefinition] = {
    TaskID.TASK_EASY: TASK_EASY,
    TaskID.TASK_MEDIUM: TASK_MEDIUM,
    TaskID.TASK_HARD: TASK_HARD,
}


def get_task(task_id: TaskID) -> TaskDefinition:
    """Retrieve a task definition by its ID.

    Args:
        task_id: The unique task identifier.

    Returns:
        The corresponding TaskDefinition.

    Raises:
        KeyError: If task_id is not found in the registry.
    """
    if task_id not in TASK_REGISTRY:
        raise KeyError(f"Unknown task_id: {task_id!r}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_id]


def list_tasks() -> list[TaskDefinition]:
    """Return all registered tasks sorted by difficulty."""
    order = {Difficulty.EASY: 0, Difficulty.MEDIUM: 1, Difficulty.HARD: 2}
    return sorted(TASK_REGISTRY.values(), key=lambda t: order[t.difficulty])
