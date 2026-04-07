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
    """Post-process JSON schema for OpenEnv Gradio UI compatibility.

    The OpenEnv Gradio UI inspects top-level property keys to decide widget
    types:
      • ``"enum"``            → ``gr.Dropdown``
      • ``"type": "boolean"`` → ``gr.Checkbox``
      • ``maxLength > 100``   → ``gr.Textbox(lines=3)``  (textarea)

    Pydantic v2 wraps ``Optional[...]`` fields in ``anyOf`` and uses
    ``$ref`` for enum types, hiding both signals.  This helper promotes
    them to the top level so the UI renders the correct widgets.

    It also injects a ``"x-ui-widget"`` hint (ignored by OpenEnv core
    but available for custom UIs) and reorders properties for a logical
    action-building flow.
    """
    defs = schema.get("$defs", {})

    for _name, prop in schema.get("properties", {}).items():
        # ── Resolve $ref → inline enum ───────────────────────────────────
        if "$ref" in prop and "enum" not in prop:
            ref_path = prop["$ref"]  # e.g. "#/$defs/ActionType"
            ref_name = ref_path.rsplit("/", 1)[-1]
            ref_def = defs.get(ref_name, {})
            if "enum" in ref_def:
                prop["enum"] = ref_def["enum"]

        # ── Lift enum + type from anyOf (for Optional[Literal[...]] etc) ─
        if "anyOf" in prop:
            for variant in prop["anyOf"]:
                # Skip the null variant
                if variant.get("type") == "null":
                    continue

                # Promote enum
                if "enum" not in prop and "enum" in variant:
                    prop["enum"] = variant["enum"]

                # Promote type (e.g. "boolean", "integer", "number", "string")
                if "type" not in prop and "type" in variant:
                    prop["type"] = variant["type"]

                # Promote maxLength (triggers textarea in Gradio UI)
                if "maxLength" not in prop and "maxLength" in variant:
                    prop["maxLength"] = variant["maxLength"]

                # Promote minLength
                if "minLength" not in prop and "minLength" in variant:
                    prop["minLength"] = variant["minLength"]

                # Resolve $ref inside anyOf
                if "enum" not in prop and "$ref" in variant:
                    ref_name = variant["$ref"].rsplit("/", 1)[-1]
                    ref_def = defs.get(ref_name, {})
                    if "enum" in ref_def:
                        prop["enum"] = ref_def["enum"]

    # ── Reorder properties for a logical action-building flow ────────────
    # The Gradio UI renders fields in iteration order — put the most
    # important fields first so the user fills them top-to-bottom.
    desired_order = [
        "action_type",
        "hallucination_detected",
        "hallucination_type",
        "hallucinated_span",
        "corrected_text",
        "reasoning",
        "metadata",
    ]
    props = schema.get("properties", {})
    ordered: Dict[str, Any] = {}
    for key in desired_order:
        if key in props:
            ordered[key] = props[key]
    # Append any remaining fields not in the desired order
    for key, val in props.items():
        if key not in ordered:
            ordered[key] = val
    schema["properties"] = ordered


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
        title="Action Type",
        description=(
            "Choose: detect (flag hallucinations), classify (label type), "
            "correct (propose fix), submit (finalise), noop (skip)."
        ),
    )
    hallucination_detected: Optional[bool] = Field(
        default=None,
        title="Hallucination Detected",
        description=(
            "Check if the passage contains a hallucination. "
            "Required for 'detect' action."
        ),
    )
    hallucination_type: Optional[HallucinationTypeLiteral] = Field(
        default=None,
        title="Hallucination Type",
        description=(
            "Hallucination category. Required for 'classify' action."
        ),
    )
    hallucinated_span: Optional[str] = Field(
        default=None,
        title="Hallucinated Span",
        max_length=500,
        description=(
            "Exact hallucinated text from the passage. Better overlap = higher reward."
        ),
    )
    corrected_text: Optional[str] = Field(
        default=None,
        title="Corrected Text",
        max_length=500,
        description=(
            "Factually-correct replacement. Required for 'correct' action."
        ),
    )
    reasoning: Optional[str] = Field(
        default=None,
        title="Reasoning (optional)",
        max_length=2000,
        description=(
            "Optional chain-of-thought explanation. Not scored."
        ),
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
    grader_score: Optional[float] = Field(
        default=None,
        description="Normalised task score strictly in (0, 1). Set when done=True.",
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
