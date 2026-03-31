"""
Hallucination Detector Gym — Custom Gradio UI Builder.

Provides a richer observation display tailored to hallucination-detection
tasks.  Passed as ``gradio_builder`` to ``create_app()`` so the OpenEnv
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
        emoji = "🟢" if reward > 0 else ("🔴" if reward < 0 else "⚪")
        status_parts.append(f"{emoji} **Reward:** `{reward:+.4f}`")
    if cumulative is not None:
        status_parts.append(f"📊 **Cumulative:** `{cumulative:.4f}`")
    if steps_remaining is not None:
        status_parts.append(f"⏱️ **Steps left:** `{steps_remaining}`")
    if done is not None:
        status_parts.append(f"{'🏁' if done else '▶️'} **Done:** `{done}`")
    if status_parts:
        lines.append(" &nbsp;|&nbsp; ".join(status_parts))
        lines.append("")

    # ── Step feedback ───────────────────────────────────────────────────
    feedback = obs.get("step_feedback")
    if feedback:
        lines.append(f"> 💡 **Feedback:** {_escape_md(feedback)}")
        lines.append("")

    # ── Task info ───────────────────────────────────────────────────────
    task_title = (data.get("metadata") or {}).get("task_title")
    task_desc = (data.get("metadata") or {}).get("task_description")
    difficulty = obs.get("difficulty")
    task_id = obs.get("task_id")
    num_hall = obs.get("num_hallucinations")

    info_parts: List[str] = []
    if task_title:
        info_parts.append(f"📝 **Task:** {_escape_md(task_title)}")
    if difficulty:
        diff_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(str(difficulty), "⚪")
        info_parts.append(f"{diff_emoji} **Difficulty:** `{difficulty}`")
    if num_hall is not None:
        info_parts.append(f"🔍 **Hallucinations in passage:** `{num_hall}`")
    if info_parts:
        lines.append(" &nbsp;|&nbsp; ".join(info_parts))
        lines.append("")

    # ── Passage ─────────────────────────────────────────────────────────
    passage = obs.get("passage")
    if passage:
        lines.append("### 📄 Passage to Analyse")
        lines.append("")
        lines.append(f"```\n{passage}\n```")
        lines.append("")

    # ── Source context ──────────────────────────────────────────────────
    source = obs.get("source_context")
    if source:
        lines.append("<details><summary><b>📚 Source Context</b> (click to expand)</summary>\n")
        lines.append(f"```\n{source}\n```")
        lines.append("</details>\n")

    # ── Action history ──────────────────────────────────────────────────
    history = obs.get("action_history", [])
    if history:
        lines.append("<details><summary><b>📋 Action History</b> (click to expand)</summary>\n")
        for entry in history:
            lines.append(f"- {_escape_md(entry)}")
        lines.append("</details>\n")

    return "\n".join(lines) if lines else "*Click **Reset** to start a new episode.*"


def _readme_section(metadata: Optional[EnvironmentMetadata]) -> str:
    if not metadata or not metadata.readme_content:
        return "*No README available.*"
    return metadata.readme_content


# ── Builder ──────────────────────────────────────────────────────────────────

def build_hallucination_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[EnvironmentMetadata],
    is_chat_env: bool,
    title: str = "OpenEnv Environment",
    quick_start_md: Optional[str] = None,
) -> gr.Blocks:
    """Custom Gradio Blocks app with hallucination-specific observation display.

    Signature matches ``gradio_builder`` expected by ``create_web_interface_app``.
    """
    from openenv.core.env_server.gradio_ui import get_gradio_display_title

    readme_content = _readme_section(metadata)
    display_title = get_gradio_display_title(metadata, fallback=title)

    # ── Callbacks ────────────────────────────────────────────────────────

    async def reset_env():
        try:
            data = await web_manager.reset_environment()
            obs_md = _format_hallucination_observation(data)
            return obs_md, json.dumps(data, indent=2), "✅ Environment reset."
        except Exception as e:
            return "", "", f"❌ Error: {e}"

    def _step_with_action(action_data: Dict[str, Any]):
        async def _run():
            try:
                data = await web_manager.step_environment(action_data)
                obs_md = _format_hallucination_observation(data)
                done = data.get("done", False)
                status_msg = "🏁 Episode complete!" if done else "✅ Step executed."
                return obs_md, json.dumps(data, indent=2), status_msg
            except Exception as e:
                return "", "", f"❌ Error: {e}"
        return _run

    async def step_form(*values):
        if not action_fields:
            return await _step_with_action({})()
        action_data: Dict[str, Any] = {}
        for i, field in enumerate(action_fields):
            if i >= len(values):
                break
            name = field["name"]
            val = values[i]
            if field.get("type") == "checkbox":
                # Only send boolean if user is doing a detect action;
                # otherwise leave it out so the model gets None.
                action_data[name] = bool(val)
            elif val is not None and val != "":
                action_data[name] = val
        return await _step_with_action(action_data)()

    def get_state_sync():
        try:
            data = web_manager.get_state()
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"Error: {e}"

    # ── Layout ───────────────────────────────────────────────────────────

    with gr.Blocks(title=display_title) as demo:
        with gr.Row():
            # ── Left column: README / Quick Start ────────────────────────
            with gr.Column(scale=1, elem_classes="col-left"):
                if quick_start_md:
                    with gr.Accordion("🚀 Quick Start", open=True):
                        gr.Markdown(quick_start_md)
                with gr.Accordion("📖 README", open=False):
                    gr.Markdown(readme_content)

                # ── Workflow guide ───────────────────────────────────────
                with gr.Accordion("🗺️ Workflow Guide", open=True):
                    gr.Markdown(
                        "### Recommended action sequence\n\n"
                        "1. **Reset** the environment to get a passage.\n"
                        "2. **detect** — check *Hallucination Detected* if the "
                        "passage has errors.\n"
                        "3. **classify** — pick the hallucination type from the "
                        "dropdown.\n"
                        "4. **correct** — paste the bad span and write a fix.\n"
                        "5. **submit** — finalise your answers.\n\n"
                        "💡 *Each correct action earns partial reward. "
                        "Maximize your cumulative score!*"
                    )

            # ── Right column: Playground ─────────────────────────────────
            with gr.Column(scale=2, elem_classes="col-right"):
                obs_display = gr.Markdown(
                    value=(
                        "# 🔬 Hallucination Detector Playground\n\n"
                        "Click **Reset** to start a new episode."
                    ),
                )

                with gr.Group():
                    gr.Markdown("### ⚡ Action")
                    step_inputs: List[Any] = []
                    for field in action_fields:
                        name = field["name"]
                        field_type = field.get("type", "text")
                        label = field.get("name", "").replace("_", " ").title()
                        desc = field.get("description", "")
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
                                placeholder=placeholder,
                                info=desc,
                                lines=3,
                            )
                        else:
                            inp = gr.Textbox(
                                label=label,
                                placeholder=placeholder,
                                info=desc,
                            )
                        step_inputs.append(inp)

                with gr.Row():
                    step_btn = gr.Button("⚡ Step", variant="primary")
                    reset_btn = gr.Button("🔄 Reset", variant="secondary")
                    state_btn = gr.Button("📊 State", variant="secondary")

                status = gr.Textbox(label="Status", interactive=False)

                with gr.Accordion("🔎 Raw JSON Response", open=False):
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
