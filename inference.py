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

STDOUT FORMAT
- The script emits exactly three line types to stdout:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

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
MAX_STEPS: int = 10
TEMPERATURE: float = 0.1
MAX_TOKENS: int = 800

# Benchmark name for structured logging
BENCHMARK: str = "hallucination_detector_gym"

# Task IDs matching the environment
TASK_IDS: List[str] = [
    "task_easy_factual",
    "task_medium_entity",
    "task_hard_multi",
]


# ──────────────────────────────────────────────────────────────────────────────
# Mandatory structured stdout logging: [START], [STEP], [END]
# ──────────────────────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line at episode begin."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit [STEP] line immediately after env.step() returns."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] line after episode completion (always emitted, even on error)."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

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

    Emits mandatory [START], [STEP], [END] structured logs to stdout.

    Args:
        client: OpenAI client instance.
        task_id: The task to run.

    Returns:
        Dictionary with task results including score and trajectory.
    """
    import websocket

    # Build WebSocket URL from HTTP URL
    ws_url = ENV_BASE_URL.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws"

    ws = websocket.create_connection(ws_url, timeout=30)

    trajectory: List[Dict[str, Any]] = []
    rewards: List[float] = []
    final_score: Optional[float] = None
    steps_taken = 0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        obs_data = env_reset_ws(ws, task_id)
        done = obs_data.get("done", False)
        obs = obs_data.get("observation", obs_data)

        for step_num in range(1, MAX_STEPS + 1):
            if done:
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
                response_text = '{"action_type": "noop", "reasoning": "Model request failed"}'

            action = parse_action_from_response(response_text)
            action_type = action.get("action_type", "noop")

            # Send action to environment via WebSocket
            step_error: Optional[str] = None
            try:
                obs_data = env_step_ws(ws, action)
            except Exception as exc:
                step_error = str(exc)
                log_step(step=step_num, action=action_type, reward=0.0, done=False, error=step_error)
                rewards.append(0.0)
                steps_taken = step_num
                break

            obs = obs_data.get("observation", obs_data)
            reward = obs_data.get("reward", obs.get("reward", 0.0))
            done = obs_data.get("done", obs.get("done", False))
            feedback = obs.get("step_feedback", "")

            rewards.append(reward)
            steps_taken = step_num

            # Emit mandatory [STEP] log
            log_step(
                step=step_num,
                action=action_type,
                reward=reward,
                done=done,
                error=step_error,
            )

            trajectory.append({
                "step": step_num,
                "action": action,
                "reward": reward,
                "done": done,
                "feedback": feedback,
            })

            # Extract grader score if episode is done
            metadata = obs.get("metadata", {})
            if done and metadata.get("grader_score") is not None:
                final_score = metadata["grader_score"]

        if not done:
            try:
                submit_data = env_step_ws(ws, {"action_type": "submit"})
                submit_obs = submit_data.get("observation", submit_data)
                submit_reward = submit_data.get("reward", submit_obs.get("reward", 0.0))
                rewards.append(submit_reward)
                steps_taken += 1
                metadata = submit_obs.get("metadata", {})
                if metadata.get("grader_score") is not None:
                    final_score = metadata["grader_score"]
                log_step(
                    step=steps_taken, action="submit",
                    reward=submit_reward, done=True, error=None,
                )
            except Exception as exc:
                log_step(
                    step=steps_taken + 1, action="submit",
                    reward=0.0, done=True, error=str(exc),
                )

    except Exception as exc:
        # Ensure [END] is always emitted even on unexpected errors
        if final_score is None:
            final_score = 0.0
        log_end(success=False, steps=steps_taken, score=final_score, rewards=rewards)
        ws.close()
        return {
            "task_id": task_id,
            "score": final_score,
            "num_steps": steps_taken,
            "trajectory": trajectory,
        }
    finally:
        ws.close()

    # Fallback: compute approximate score from cumulative reward
    if final_score is None:
        cumulative = obs.get("cumulative_reward", 0.0)
        if cumulative is not None:
            final_score = max(0.0, min(1.0, cumulative))
        else:
            final_score = 0.0

    final_score = max(0.0, min(1.0, final_score))
    success = final_score > 0.0

    # Emit mandatory [END] log
    log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    return {
        "task_id": task_id,
        "score": final_score,
        "num_steps": len(trajectory),
        "trajectory": trajectory,
    }


def main() -> None:
    """Run the baseline inference across all 3 tasks."""
    if not API_KEY:
        print("WARNING: No API key found. Set HF_TOKEN or API_KEY env var.", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results: List[Dict[str, Any]] = []
    start_time = time.time()

    for task_id in TASK_IDS:
        task_result = run_task(client, task_id)
        results.append(task_result)

    elapsed = time.time() - start_time

    # Write results to file for reproducibility
    total_score = sum(r["score"] for r in results)
    avg_score = total_score / len(results) if results else 0.0

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

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_results.json")
    with open(output_path, "w") as f:
        json.dump(results_output, f, indent=2)


if __name__ == "__main__":
    main()
