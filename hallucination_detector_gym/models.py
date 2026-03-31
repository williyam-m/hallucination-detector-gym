"""
Hallucination Detector Gym — Pydantic Models.

Typed Action, Observation, and State models conforming to the OpenEnv spec.
Uses Literal types for enum fields so the OpenEnv Gradio web interface
renders them as dropdown selectors instead of free-text inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import ConfigDict, Field, model_validator

from openenv.core.env_server.types import (
    Action as BaseAction,
    Observation as BaseObservation,
    State as BaseState,
)

from .constants import ActionType, Difficulty, HallucinationType, TaskID

# ──────────────────────────────────────────────────────────────────────────────
# Literal type aliases (mirrors Enum values for JSON schema dropdown support)
# ──────────────────────────────────────────────────────────────────────────────
ActionTypeLiteral = Literal["detect", "classify", "correct", "submit", "noop"]
HallucinationTypeLiteral = Literal[
    "factual_error", "entity_fabrication", "logical_inconsistency", "none"
]


def _flatten_enum_from_anyof(schema: Dict[str, Any]) -> None:
    """Post-process JSON schema: lift ``enum`` out of ``anyOf`` wrappers.

    The OpenEnv Gradio UI reads ``field_info.get("enum")`` at the top level
    of each property to decide whether to render a dropdown.  Pydantic v2
    puts ``enum`` inside ``anyOf`` for ``Optional[Literal[...]]`` fields,
    so this helper copies it up so the UI can discover it.

    Also resolves ``$ref`` to ``$defs`` entries so enum values are always
    available inline on each property.
    """
    defs = schema.get("$defs", {})

    for prop in schema.get("properties", {}).values():
        # Resolve $ref → inline enum
        if "$ref" in prop and "enum" not in prop:
            ref_path = prop["$ref"]  # e.g. "#/$defs/ActionType"
            ref_name = ref_path.rsplit("/", 1)[-1]
            ref_def = defs.get(ref_name, {})
            if "enum" in ref_def:
                prop["enum"] = ref_def["enum"]

        # Lift enum from anyOf (for Optional[Literal[...]])
        if "anyOf" in prop and "enum" not in prop:
            for variant in prop["anyOf"]:
                # Direct enum in variant
                if "enum" in variant:
                    prop["enum"] = variant["enum"]
                    break
                # $ref inside anyOf
                if "$ref" in variant:
                    ref_name = variant["$ref"].rsplit("/", 1)[-1]
                    ref_def = defs.get(ref_name, {})
                    if "enum" in ref_def:
                        prop["enum"] = ref_def["enum"]
                        break


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

    model_config = ConfigDict(json_schema_extra=_flatten_enum_from_anyof)

    action_type: ActionTypeLiteral = Field(
        default="noop",
        description=(
            "The type of action to perform: detect, classify, correct, submit, or noop. "
            "Defaults to 'noop' if omitted."
        ),
    )
    hallucination_detected: Optional[bool] = Field(
        default=None,
        description="Whether the agent believes a hallucination exists in the current passage.",
    )
    hallucination_type: Optional[HallucinationTypeLiteral] = Field(
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

    # ── Validators: coerce Literal strings → Enum instances ──────────────────
    @model_validator(mode="after")
    def _coerce_to_enums(self) -> "HallucinationAction":
        """Convert literal string values to their Enum counterparts.

        This lets the rest of the codebase use ``action.action_type == ActionType.DETECT``
        while the JSON schema still emits inline ``enum`` arrays for the UI.
        """
        # action_type: str → ActionType
        if isinstance(self.action_type, str):
            object.__setattr__(self, "action_type", ActionType(self.action_type))

        # hallucination_type: str → HallucinationType
        if isinstance(self.hallucination_type, str):
            object.__setattr__(
                self, "hallucination_type", HallucinationType(self.hallucination_type)
            )

        return self


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
