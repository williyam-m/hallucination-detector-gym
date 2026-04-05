# API Reference — Hallucination Detector Gym

## Overview

The Hallucination Detector Gym exposes a REST + WebSocket API conforming to the OpenEnv specification. All endpoints accept and return JSON.

**Base URL:** `http://localhost:8000` (local) or `https://williyam-hallucination-detector-gym.hf.space` (deployed)

---

## REST Endpoints

### `POST /reset`

Reset the environment and start a new episode.

**Request Body:**
```json
{
  "task_id": "task_easy_factual",
  "seed": 42,
  "episode_id": "optional-custom-id"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `task_id` | `string` | No | `"task_easy_factual"` | One of: `task_easy_factual`, `task_medium_entity`, `task_hard_multi` |
| `seed` | `int` | No | `42` | Random seed for reproducibility |
| `episode_id` | `string` | No | auto-generated UUID | Custom episode identifier |

**Response (200):**
```json
{
  "done": false,
  "reward": 0.0,
  "observation": {
    "task_id": "task_easy_factual",
    "difficulty": "easy",
    "passage": "Albert Einstein was a renowned theoretical physicist...",
    "source_context": "Albert Einstein (1879–1955) was a German-born...",
    "num_hallucinations": 1,
    "step_feedback": "Environment reset. Analyse the passage for hallucinations.",
    "steps_remaining": 12,
    "cumulative_reward": 0.0,
    "action_history": []
  },
  "metadata": {
    "task_title": "Simple Factual Error Detection",
    "task_description": "The agent must identify a single factual error..."
  }
}
```

---

### `POST /step`

Execute an agent action.

**Request Body:**
```json
{
  "action": {
    "action_type": "detect",
    "hallucination_detected": true,
    "hallucinated_span": "Munich, Germany",
    "hallucination_type": null,
    "corrected_text": null,
    "reasoning": "The source says Ulm, not Munich."
  }
}
```

**Action Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `action_type` | `string` | Yes | One of: `detect`, `classify`, `correct`, `submit`, `noop` |
| `hallucination_detected` | `bool` | For `detect` | Whether a hallucination was found |
| `hallucination_type` | `string` | For `classify` | One of: `factual_error`, `entity_fabrication`, `logical_inconsistency`, `none` |
| `hallucinated_span` | `string` | Recommended | Exact text from passage that is hallucinated (max 500 chars) |
| `corrected_text` | `string` | For `correct` | Proposed factually-correct replacement (max 500 chars) |
| `reasoning` | `string` | No | Chain-of-thought explanation (max 2000 chars, not scored) |

**Response (200):** Same structure as `/reset` response, with updated reward, feedback, and state.

---

### `GET /state`

Return the current internal environment state.

**Response (200):**
```json
{
  "episode_id": "abc-123",
  "step_count": 3,
  "task_id": "task_easy_factual",
  "difficulty": "easy",
  "cumulative_reward": 0.65,
  "detections_submitted": 1,
  "classifications_submitted": 1,
  "corrections_submitted": 1,
  "is_done": false
}
```

---

### `GET /health`

Health check endpoint.

**Response (200):**
```json
{"status": "healthy"}
```

---

### `GET /schema`

Return JSON schemas for action, observation, and state models.

**Response (200):**
```json
{
  "action": { "properties": { ... } },
  "observation": { "properties": { ... } },
  "state": { "properties": { ... } }
}
```

---

## WebSocket Endpoint

### `WS /ws`

Persistent WebSocket connection for stateful multi-step episodes. Preferred for inference scripts.

**Message format (client -> server):**
```json
{"type": "reset", "data": {"task_id": "task_easy_factual"}}
{"type": "step", "data": {"action_type": "detect", "hallucination_detected": true}}
{"type": "state"}
```

**Response format (server -> client):**
```json
{"type": "observation", "data": { ... }}
{"type": "state", "data": { ... }}
{"type": "error", "data": {"message": "error description"}}
```

---

## Reward Structure

| Action | Correct Reward | Incorrect Penalty |
|--------|---------------|-------------------|
| `detect` | +0.30 | -0.15 |
| `detect` + matching span | +0.20 * overlap | — |
| `classify` | +0.30 | -0.10 |
| `correct` | +0.20 * similarity | — |
| `submit` (early) | +0.10 * efficiency | — |
| `noop` (hallucinations exist) | — | -0.05 |
| Repeated action (3x identical) | — | -0.05 |

**Final score:** `clamp(cumulative_reward / num_annotations, 0.0, 1.0)`

---

## Error Handling

- Invalid `task_id` in `/reset`: 400 Bad Request with error message
- Malformed action in `/step`: Action coerced to `noop` with feedback
- Step before reset: Returns `done=True` with error in metadata
- Step after episode done: Returns `done=True` with "already done" feedback
