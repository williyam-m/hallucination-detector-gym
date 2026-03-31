"""
Inference Script — Hallucination Detector Gym
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your
  environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root
  directory of the project.
- Participants must use OpenAI Client for all LLM calls using above variables.

This script runs a baseline LLM agent against all 3 tasks in the Hallucination
Detector Gym environment. It communicates with the environment via WebSocket
for stateful multi-step episodes and uses the OpenAI-compatible chat completion
API for the agent's decisions.

Runtime target: < 20 min on vcpu=2, memory=8gb.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Configuration from environment variables (no hardcoded secrets)
# ──────────────────────────────────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

# Environment server URL (local Docker or HF Space)
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:8000")

# Agent configuration
MAX_STEPS: int = 8
TEMPERATURE: float = 0.1
MAX_TOKENS: int = 800

# Task IDs matching the environment
TASK_IDS: List[str] = [
    "task_easy_factual",
    "task_medium_entity",
    "task_hard_multi",
]

# ──────────────────────────────────────────────────────────────────────────────
# System prompt for the hallucination detector agent
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = textwrap.dedent("""\
    You are an expert hallucination detector. You analyse LLM-generated passages
    by comparing them against source context to identify factual errors, entity
    fabrications, and logical inconsistencies.

    You interact with an environment by sending JSON actions. Each action must be
    a valid JSON object with the following schema:

    {
        "action_type": "detect" | "classify" | "correct" | "submit" | "noop",
        "hallucination_detected": true | false,    // required for "detect"
        "hallucination_type": "factual_error" | "entity_fabrication" | "logical_inconsistency" | "none",  // for "classify"
        "hallucinated_span": "exact text from passage",   // the hallucinated substring
        "corrected_text": "what it should say",            // for "correct"
        "reasoning": "your chain-of-thought"               // optional
    }

    Strategy:
    1. First, carefully read the passage and source context.
    2. Use "detect" to flag each hallucination you find, providing the span.
    3. Use "classify" to label the type of each hallucination.
    4. Use "correct" to propose the right text.
    5. Use "submit" when done.

    Always respond with ONLY a valid JSON object. No markdown, no explanation outside JSON.
""").strip()


def env_reset_ws(ws: Any, task_id: str) -> Dict[str, Any]:
    """Reset the environment via WebSocket for stateful interaction.

    Args:
        ws: WebSocket connection object.
        task_id: The task identifier to load.

    Returns:
        The observation data from reset.
    """
    ws.send(json.dumps({
        "type": "reset",
        "data": {"task_id": task_id},
    }))
    response = json.loads(ws.recv())
    if response.get("type") == "error":
        raise RuntimeError(f"Reset failed: {response.get('data', {}).get('message', 'unknown')}")
    return response.get("data", response)


def env_step_ws(ws: Any, action: Dict[str, Any]) -> Dict[str, Any]:
    """Send an action to the environment via WebSocket.

    Args:
        ws: WebSocket connection object.
        action: The action dictionary to send.

    Returns:
        The observation data from step.
    """
    ws.send(json.dumps({
        "type": "step",
        "data": action,
    }))
    response = json.loads(ws.recv())
    if response.get("type") == "error":
        raise RuntimeError(f"Step failed: {response.get('data', {}).get('message', 'unknown')}")
    return response.get("data", response)


def parse_action_from_response(response_text: str) -> Dict[str, Any]:
    """Extract a JSON action from the LLM's response text.

    Args:
        response_text: Raw text from the LLM.

    Returns:
        Parsed action dictionary.
    """
    if not response_text:
        return {"action_type": "noop", "reasoning": "Empty response from model."}

    # Try direct JSON parse
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object in the response
    json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return {"action_type": "noop", "reasoning": f"Could not parse action from: {response_text[:200]}"}


def build_user_prompt(observation: Dict[str, Any], step_num: int) -> str:
    """Build the user prompt from the current observation.

    Args:
        observation: The observation dictionary from the environment.
        step_num: Current step number.

    Returns:
        Formatted user prompt string.
    """
    obs = observation.get("observation", observation)

    passage = obs.get("passage", "(no passage)")
    source_context = obs.get("source_context", "(no source)")
    feedback = obs.get("step_feedback", "")
    steps_remaining = obs.get("steps_remaining", "?")
    cumulative_reward = obs.get("cumulative_reward", 0.0)
    action_history = obs.get("action_history", [])
    num_hallucinations = obs.get("num_hallucinations")
    difficulty = obs.get("difficulty", "unknown")

    hint_line = ""
    if num_hallucinations is not None:
        hint_line = f"\nHint: This passage contains exactly {num_hallucinations} hallucination(s)."

    history_str = "\n".join(action_history[-5:]) if action_history else "None"

    prompt = textwrap.dedent(f"""\
        Step: {step_num}
        Difficulty: {difficulty}
        Steps remaining: {steps_remaining}
        Cumulative reward: {cumulative_reward:.3f}

        === SOURCE CONTEXT (ground truth) ===
        {source_context}

        === PASSAGE TO ANALYSE ===
        {passage}
        {hint_line}
        === FEEDBACK FROM LAST ACTION ===
        {feedback}

        === ACTION HISTORY ===
        {history_str}

        Respond with a single JSON action object.
        Focus on finding and addressing hallucinations systematically.
        If you have identified and processed all hallucinations, use "submit".
    """).strip()

    return prompt


def run_task(client: OpenAI, task_id: str) -> Dict[str, Any]:
    """Run the agent on a single task via WebSocket for stateful interaction.

    Args:
        client: OpenAI client instance.
        task_id: The task to run.

    Returns:
        Dictionary with task results including score and trajectory.
    """
    import websocket

    print(f"\n{'='*60}")
    print(f"  Task: {task_id}")
    print(f"{'='*60}")

    # Build WebSocket URL from HTTP URL
    ws_url = ENV_BASE_URL.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws"

    ws = websocket.create_connection(ws_url, timeout=30)

    trajectory: List[Dict[str, Any]] = []
    final_score: Optional[float] = None

    try:
        # Reset environment
        obs_data = env_reset_ws(ws, task_id)
        done = obs_data.get("done", False)
        obs = obs_data.get("observation", obs_data)

        for step_num in range(1, MAX_STEPS + 1):
            if done:
                print(f"  Episode done at step {step_num - 1}.")
                break

            user_prompt = build_user_prompt(obs_data, step_num)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                print(f"  Model request failed: {exc}")
                response_text = '{"action_type": "noop", "reasoning": "Model request failed"}'

            action = parse_action_from_response(response_text)
            action_type = action.get("action_type", "noop")

            print(f"  Step {step_num}: {action_type}", end="")
            if action.get("hallucinated_span"):
                span_preview = action["hallucinated_span"][:50]
                print(f" | span='{span_preview}...'", end="")
            print()

            # Send action to environment via WebSocket
            try:
                obs_data = env_step_ws(ws, action)
            except Exception as exc:
                print(f"  Step failed: {exc}")
                break

            obs = obs_data.get("observation", obs_data)
            reward = obs_data.get("reward", obs.get("reward", 0.0))
            done = obs_data.get("done", obs.get("done", False))
            feedback = obs.get("step_feedback", "")

            trajectory.append({
                "step": step_num,
                "action": action,
                "reward": reward,
                "done": done,
                "feedback": feedback,
            })

            print(f"         reward={reward:+.3f} | done={done}")
            if feedback:
                print(f"         feedback: {feedback[:100]}")

            # Extract grader score if episode is done
            metadata = obs.get("metadata", {})
            if done and metadata.get("grader_score") is not None:
                final_score = metadata["grader_score"]

        if not done:
            print(f"  Reached max steps ({MAX_STEPS}). Submitting...")
            try:
                submit_data = env_step_ws(ws, {"action_type": "submit"})
                submit_obs = submit_data.get("observation", submit_data)
                metadata = submit_obs.get("metadata", {})
                if metadata.get("grader_score") is not None:
                    final_score = metadata["grader_score"]
            except Exception as exc:
                print(f"  Submit failed: {exc}")

    finally:
        ws.close()

    # Fallback: compute approximate score from cumulative reward
    if final_score is None:
        cumulative = obs.get("cumulative_reward", 0.0)
        if cumulative is not None:
            final_score = max(0.0, min(1.0, cumulative))
        else:
            final_score = 0.0

    print(f"\n  Final Score: {final_score:.4f}")

    return {
        "task_id": task_id,
        "score": final_score,
        "num_steps": len(trajectory),
        "trajectory": trajectory,
    }


def main() -> None:
    """Run the baseline inference across all 3 tasks."""
    print("=" * 60)
    print("  Hallucination Detector Gym — Baseline Inference")
    print("=" * 60)
    print(f"  API Base URL:  {API_BASE_URL}")
    print(f"  Model:         {MODEL_NAME}")
    print(f"  Env URL:       {ENV_BASE_URL}")
    print(f"  Max Steps:     {MAX_STEPS}")
    print()

    if not API_KEY:
        print("WARNING: No API key found. Set HF_TOKEN or API_KEY env var.")
        print("Continuing — the model endpoint may reject requests.\n")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results: List[Dict[str, Any]] = []
    start_time = time.time()

    for task_id in TASK_IDS:
        task_result = run_task(client, task_id)
        results.append(task_result)

    elapsed = time.time() - start_time

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BASELINE RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Task':<30} {'Score':>8} {'Steps':>8}")
    print("-" * 50)

    total_score = 0.0
    for r in results:
        print(f"{r['task_id']:<30} {r['score']:>8.4f} {r['num_steps']:>8d}")
        total_score += r["score"]

    avg_score = total_score / len(results) if results else 0.0
    print("-" * 50)
    print(f"{'Average':<30} {avg_score:>8.4f}")
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 60)

    # Write results to file for reproducibility
    results_output = {
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "env_base_url": ENV_BASE_URL,
        "elapsed_seconds": round(elapsed, 2),
        "average_score": round(avg_score, 4),
        "task_results": [
            {
                "task_id": r["task_id"],
                "score": round(r["score"], 4),
                "num_steps": r["num_steps"],
            }
            for r in results
        ],
    }

    output_path = "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(results_output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
