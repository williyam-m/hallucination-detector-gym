"""
Hallucination Detector Gym — Custom Gradio UI Builder.

Provides a richer observation display tailored to hallucination-detection
tasks. Passed as ``gradio_builder`` to ``create_app()`` so the OpenEnv
web interface renders environment-specific fields (passage, source context,
step feedback, action history, etc.) in formatted Markdown instead of
relying solely on the generic raw-JSON view.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import gradio as gr

from openenv.core.env_server.types import EnvironmentMetadata


# ── Helpers ──────────────────────────────────────────────────────────────────

def _escape_md(text: str) -> str:
    """Escape Markdown special characters in user-controlled content."""
    return re.sub(r"([\\`*_\{\}\[\]()#+\-.!|~>])", r"\\\1", str(text))


def _format_hallucination_observation(data: Dict[str, Any]) -> str:
    """Format a hallucination-environment response as rich Markdown."""
    lines: List[str] = []
    obs = data.get("observation", data)  # support both wrapped and flat

    # ── Status bar ──────────────────────────────────────────────────────
    done = data.get("done", obs.get("done"))
    reward = data.get("reward", obs.get("reward"))
    cumulative = obs.get("cumulative_reward")
    steps_remaining = obs.get("steps_remaining")

    status_parts: List[str] = []
    if reward is not None:
        indicator = "+" if reward > 0 else ("-" if reward < 0 else " ")
        status_parts.append(f"**Reward:** `{indicator}{abs(reward):.4f}`")
    if cumulative is not None:
        status_parts.append(f"**Cumulative:** `{cumulative:.4f}`")
    if steps_remaining is not None:
        status_parts.append(f"**Steps left:** `{steps_remaining}`")
    if done is not None:
        status_parts.append(f"**Done:** `{done}`")
    if status_parts:
        lines.append(" | ".join(status_parts))
        lines.append("")

    # ── Step feedback ───────────────────────────────────────────────────
    feedback = obs.get("step_feedback")
    if feedback:
        lines.append(f"> **Feedback:** {_escape_md(feedback)}")
        lines.append("")

    # ── Task info ───────────────────────────────────────────────────────
    task_title = (data.get("metadata") or {}).get("task_title")
    difficulty = obs.get("difficulty")
    num_hall = obs.get("num_hallucinations")

    info_parts: List[str] = []
    if task_title:
        info_parts.append(f"**Task:** {_escape_md(task_title)}")
    if difficulty:
        info_parts.append(f"**Difficulty:** `{difficulty}`")
    if num_hall is not None:
        info_parts.append(f"**Hallucinations in passage:** `{num_hall}`")
    if info_parts:
        lines.append(" | ".join(info_parts))
        lines.append("")

    # ── Passage ─────────────────────────────────────────────────────────
    passage = obs.get("passage")
    if passage:
        lines.append("### Passage to Analyse")
        lines.append("")
        lines.append(f"```\n{passage}\n```")
        lines.append("")

    # ── Source context (open by default) ────────────────────────────────
    source = obs.get("source_context")
    if source:
        lines.append("<details open><summary><b>Source Context</b> (reference material)</summary>\n")
        lines.append(f"```\n{source}\n```")
        lines.append("</details>\n")

    # ── Action history ──────────────────────────────────────────────────
    history = obs.get("action_history", [])
    if history:
        lines.append("<details open><summary><b>Action History</b></summary>\n")
        for entry in history:
            lines.append(f"- {_escape_md(entry)}")
        lines.append("</details>\n")

    return "\n".join(lines) if lines else "*Click **Reset** to start a new episode.*"


def _readme_section(metadata: Optional[EnvironmentMetadata]) -> str:
    """Extract README content from environment metadata."""
    if not metadata or not metadata.readme_content:
        return (
            "# Hallucination Detector Gym\n\n"
            "An OpenEnv RL environment where AI agents learn to detect, classify, "
            "and correct hallucinations in LLM-generated text.\n\n"
            "**Tasks:** 3 difficulty levels (easy, medium, hard)\n\n"
            "**Actions:** detect, classify, correct, submit, noop\n\n"
            "**Rewards:** Dense per-step signal (not binary end-of-episode)\n\n"
            "[GitHub](https://github.com/williyam/hallucination-detector-gym) | "
            "[Model](https://huggingface.co/williyam/hallucination-detector-agent-qwen3-0.6b)"
        )
    return metadata.readme_content


# ── Builder ──────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; margin: auto; }
.col-left { border-right: 1px solid var(--border-color-primary); padding-right: 16px; }
.col-right { padding-left: 16px; }
.status-bar { background: var(--background-fill-secondary); padding: 8px 12px; border-radius: 6px; margin-bottom: 8px; }
"""


def build_hallucination_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[EnvironmentMetadata],
    is_chat_env: bool,
    title: str = "OpenEnv Environment",
    quick_start_md: Optional[str] = None,
) -> gr.Blocks:
    """Custom Gradio Blocks app with hallucination-specific observation display.

    Args:
        web_manager: The WebInterfaceManager instance.
        action_fields: List of action field definitions.
        metadata: Environment metadata.
        is_chat_env: Whether this is a chat-style environment.
        title: Display title.
        quick_start_md: Quick start markdown content.

    Returns:
        gr.Blocks: Configured Gradio application.
    """
    from openenv.core.env_server.gradio_ui import get_gradio_display_title

    readme_content = _readme_section(metadata)
    display_title = get_gradio_display_title(metadata, fallback=title)

    # ── Callbacks ────────────────────────────────────────────────────────

    async def reset_env():
        """Reset the environment and return formatted observation."""
        try:
            data = await web_manager.reset_environment()
            obs_md = _format_hallucination_observation(data)
            return obs_md, json.dumps(data, indent=2), "Environment reset. Analyse the passage."
        except Exception as e:
            return "", "", f"Error: {e}"

    def _step_with_action(action_data: Dict[str, Any]):
        async def _run():
            try:
                data = await web_manager.step_environment(action_data)
                obs_md = _format_hallucination_observation(data)
                done = data.get("done", False)
                status_msg = "Episode complete." if done else "Step executed."
                return obs_md, json.dumps(data, indent=2), status_msg
            except Exception as e:
                return "", "", f"Error: {e}"
        return _run

    async def step_form(*values):
        """Process form values and execute a step."""
        if not action_fields:
            return await _step_with_action({})()
        action_data: Dict[str, Any] = {}
        has_user_input = False
        for i, field in enumerate(action_fields):
            if i >= len(values):
                break
            name = field["name"]
            val = values[i]
            if field.get("type") == "checkbox":
                action_data[name] = bool(val)
            elif val is not None and val != "":
                action_data[name] = val
                has_user_input = True
        if not has_user_input and action_data.get("action_type") in (None, "", "noop"):
            return (
                "",
                "",
                "No input provided. Using default values. Please enter action details.",
            )
        return await _step_with_action(action_data)()

    def get_state_sync():
        """Get current environment state."""
        try:
            data = web_manager.get_state()
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"Error: {e}"

    # ── Layout ───────────────────────────────────────────────────────────

    with gr.Blocks(title=display_title, css=CUSTOM_CSS) as demo:
        with gr.Row():
            # ── Left column: README / Quick Start ────────────────────────
            with gr.Column(scale=1, elem_classes="col-left"):
                if quick_start_md:
                    with gr.Accordion("Quick Start", open=True):
                        gr.Markdown(quick_start_md)
                with gr.Accordion("README", open=False):
                    gr.Markdown(readme_content)

                # ── Workflow guide ───────────────────────────────────────
                with gr.Accordion("Workflow Guide", open=True):
                    gr.Markdown(
                        "### Recommended action sequence\n\n"
                        "1. **Reset** the environment to get a passage.\n"
                        "2. **detect** - check Hallucination Detected if the "
                        "passage has errors.\n"
                        "3. **classify** - pick the hallucination type.\n"
                        "4. **correct** - paste the bad span and write a fix.\n"
                        "5. **submit** - finalise your answers.\n\n"
                        "*Each correct action earns partial reward. "
                        "Maximize your cumulative score.*"
                    )

            # ── Right column: Playground ─────────────────────────────────
            with gr.Column(scale=2, elem_classes="col-right"):
                obs_display = gr.Markdown(
                    value=(
                        "# Hallucination Detector Playground\n\n"
                        "Click **Reset** to start a new episode."
                    ),
                )

                with gr.Group():
                    gr.Markdown("### Action")
                    step_inputs: List[Any] = []
                    for field in action_fields:
                        name = field["name"]
                        field_type = field.get("type", "text")
                        label = field.get("name", "").replace("_", " ").title()
                        desc = field.get("description", "")
                        # Shorten hints: remove emoji, truncate
                        desc = re.sub(r"[\U0001f300-\U0001fAFF]", "", desc).strip()
                        if len(desc) > 120:
                            desc = desc[:117] + "..."
                        placeholder = field.get("placeholder", "")

                        if field_type == "checkbox":
                            inp = gr.Checkbox(
                                label=label,
                                info=desc,
                            )
                        elif field_type == "number":
                            inp = gr.Number(label=label, info=desc)
                        elif field_type == "select":
                            choices = field.get("choices") or []
                            inp = gr.Dropdown(
                                choices=choices,
                                label=label,
                                info=desc,
                                allow_custom_value=False,
                            )
                        elif field_type in ("textarea", "tensor"):
                            inp = gr.Textbox(
                                label=label,
                                placeholder=placeholder or "Enter value...",
                                info=desc,
                                lines=3,
                            )
                        else:
                            inp = gr.Textbox(
                                label=label,
                                placeholder=placeholder or "Enter value...",
                                info=desc,
                            )
                        step_inputs.append(inp)

                with gr.Row():
                    step_btn = gr.Button("Step", variant="primary")
                    reset_btn = gr.Button("Reset", variant="secondary")
                    state_btn = gr.Button("State", variant="secondary")

                status = gr.Textbox(label="Status", interactive=False)

                with gr.Accordion("Raw JSON Response", open=True):
                    raw_json = gr.Code(
                        label="JSON",
                        language="json",
                        interactive=False,
                    )

        # ── Event bindings ───────────────────────────────────────────────
        reset_btn.click(
            fn=reset_env,
            outputs=[obs_display, raw_json, status],
        )
        step_btn.click(
            fn=step_form,
            inputs=step_inputs,
            outputs=[obs_display, raw_json, status],
        )
        state_btn.click(
            fn=get_state_sync,
            outputs=[raw_json],
        )

    return demo
