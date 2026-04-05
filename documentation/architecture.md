# Architecture Documentation — Hallucination Detector Gym

## High-Level System Design

The Hallucination Detector Gym is an OpenEnv-compatible reinforcement learning environment where AI agents learn to detect, classify, and correct hallucinations in LLM-generated text. The system follows a client-server architecture deployed as a containerised Hugging Face Space.

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
│                                                                 │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐ │
│  │ inference.py  │  │  client.py    │  │  Gradio Web UI       │ │
│  │ (OpenAI API)  │  │  (EnvClient)  │  │  (browser-based)     │ │
│  └──────┬───────┘  └───────┬───────┘  └──────────┬───────────┘ │
│         │ WebSocket        │ WebSocket            │ HTTP        │
└─────────┼──────────────────┼──────────────────────┼─────────────┘
          │                  │                      │
          ▼                  ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Server Layer (FastAPI)                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    server/app.py                          │   │
│  │  POST /reset  POST /step  GET /state  GET /health        │   │
│  │  GET /schema  WS /ws  WS /ws/ui  GET /web/*              │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │        server/hallucination_environment.py                │   │
│  │        HallucinationDetectorEnvironment                   │   │
│  │        implements: reset() step() state() close()         │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │                Core Library Layer                         │   │
│  │                                                           │   │
│  │  ┌──────────┐  ┌──────────────┐  ┌──────────┐           │   │
│  │  │  tasks   │  │ RewardEngine │  │  graders  │           │   │
│  │  │ Registry │  │ (per-step)   │  │ (0→1.0)   │           │   │
│  │  └──────────┘  └──────────────┘  └──────────┘           │   │
│  │                                                           │   │
│  │  ┌──────────┐  ┌──────────────┐  ┌──────────┐           │   │
│  │  │  models  │  │  constants   │  │ logging  │           │   │
│  │  │ Pydantic │  │ Enums/Config │  │ structlog│           │   │
│  │  └──────────┘  └──────────────┘  └──────────┘           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Diagram

### Core Library (`hallucination_detector_gym/`)

| Component | File | Responsibility |
|-----------|------|----------------|
| **Constants** | `constants.py` | All enums (ActionType, HallucinationType, Difficulty, TaskID), reward weights, episode config. Single source of truth for magic numbers. |
| **Models** | `models.py` | Typed Pydantic models: `HallucinationAction` (agent input), `HallucinationObservation` (env output), `HallucinationState` (internal state). All inherit from OpenEnv base types. |
| **Tasks** | `tasks.py` | Frozen dataclass task definitions with ground-truth `HallucinationAnnotation`. Task registry mapping `TaskID` to `TaskDefinition`. |
| **Rewards** | `rewards.py` | Stateful `RewardEngine` computing dense per-step rewards. Handles detect/classify/correct with annotation deduplication, span overlap (Jaccard + LCS), and step efficiency bonus. |
| **Graders** | `graders.py` | Deterministic `TaskGrader` that replays action sequences through `RewardEngine` and produces normalised scores in [0.0, 1.0]. |
| **Logging** | `logging_config.py` | Structured logging via structlog. JSON output in production, coloured console in development. |

### Server Layer (`server/`)

| Component | File | Responsibility |
|-----------|------|----------------|
| **App** | `app.py` | FastAPI application factory. Creates OpenEnv-compatible endpoints. Optionally mounts custom Gradio UI. |
| **Environment** | `hallucination_environment.py` | OpenEnv `Environment` subclass implementing `reset()`, `step()`, `state()`, `close()`. Manages episode lifecycle, step counting, and action validation. |
| **Gradio UI** | `gradio_builder.py` | Custom Gradio Blocks application for browser-based interaction. Rich Markdown observation display, action form, workflow guide. |

### Client Layer

| Component | File | Responsibility |
|-----------|------|----------------|
| **EnvClient** | `client.py` | Typed Python client wrapping OpenEnv's `EnvClient` for WebSocket-based interaction. |
| **Inference** | `inference.py` | Baseline LLM agent using OpenAI API. Emits mandatory `[START]`, `[STEP]`, `[END]` structured logs. |

## Data Flow

### Episode Lifecycle

```
1. Agent calls POST /reset (task_id="task_easy_factual")
   └─> Environment.reset()
       ├─> Resolve TaskDefinition from TASK_REGISTRY
       ├─> Create fresh RewardEngine(task)
       ├─> Initialize HallucinationState (step=0, reward=0)
       └─> Return HallucinationObservation (passage, source_context, hints)

2. Agent calls POST /step (action={detect, span="Munich, Germany"})
   └─> Environment.step(action)
       ├─> Validate/coerce action to HallucinationAction
       ├─> Increment step_count
       ├─> RewardEngine.compute_reward(action)
       │   ├─> Check repeated action penalty
       │   ├─> Match action to best annotation (Jaccard+LCS)
       │   ├─> Check deduplication (already detected/classified/corrected?)
       │   ├─> Award/penalise based on correctness
       │   └─> Return (reward, feedback)
       ├─> Update HallucinationState
       ├─> Check termination (submit action or max steps)
       └─> Return HallucinationObservation (feedback, reward, done)

3. Agent calls POST /step (action={submit})
   └─> Environment.step(action)
       ├─> Award step efficiency bonus
       ├─> Set done=True
       ├─> Compute final grader score via RewardEngine.get_final_score()
       └─> Return HallucinationObservation (done=True, grader_score in metadata)
```

### Reward Computation Flow

```
action
  │
  ├─> Repeated action check ──────> PENALTY_REPEATED_ACTION (-0.05)
  │
  ├─> DETECT handler
  │   ├─> hallucination_detected matches? ──> REWARD_CORRECT_DETECTION (+0.30)
  │   │   └─> span provided?
  │   │       ├─> Best matching annotation (exclude already detected)
  │   │       ├─> Overlap > 0.3? ──> REWARD_CORRECT_SPAN * overlap
  │   │       └─> Mark annotation as detected
  │   └─> mismatch ──> PENALTY_WRONG_DETECTION (-0.15)
  │
  ├─> CLASSIFY handler
  │   ├─> Already classified? ──> "Already classified" (0 reward)
  │   ├─> Type matches annotation? ──> REWARD_CORRECT_CLASSIFICATION (+0.30)
  │   └─> mismatch ──> PENALTY_WRONG_CLASSIFICATION (-0.10)
  │
  ├─> CORRECT handler
  │   ├─> Already corrected? ──> "Already corrected" (0 reward)
  │   ├─> Correction overlap > 0.3? ──> REWARD_CORRECT_CORRECTION * overlap
  │   └─> mismatch ──> 0 reward
  │
  ├─> SUBMIT handler
  │   └─> Step efficiency bonus ──> BONUS_STEP_EFFICIENCY * (1 - step/max_steps)
  │
  └─> NOOP handler
      └─> Hallucinations exist? ──> PENALTY_NOOP_WHEN_HALLUCINATION (-0.05)
```

### Span Matching Algorithm

The span matching uses a combined metric of Jaccard similarity (bag-of-words) and Longest Common Subsequence ratio (sequence-aware):

```
score = (jaccard(pred, truth) + lcs_ratio(pred, truth)) / 2.0
```

This prevents gaming via word reordering while still allowing fuzzy matching for partial spans.

## Deployment Architecture

```
┌─────────────────────────────────────────────────────┐
│              Hugging Face Spaces                     │
│                                                      │
│  ┌───────────────────────────────────────────────┐  │
│  │           Docker Container                     │  │
│  │                                                │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │  uvicorn (ASGI server)                  │  │  │
│  │  │  server.app:app                         │  │  │
│  │  │  host=0.0.0.0  port=8000               │  │  │
│  │  └─────────────────────┬───────────────────┘  │  │
│  │                        │                       │  │
│  │  ┌─────────────────────▼───────────────────┐  │  │
│  │  │  FastAPI Application                    │  │  │
│  │  │  + Gradio UI at /web                    │  │  │
│  │  │  + API at /reset /step /state /ws       │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  │                                                │  │
│  │  HEALTHCHECK: curl -f localhost:8000/health   │  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
│  Resources: vcpu=2, memory=8gb                       │
└─────────────────────────────────────────────────────┘
```

## Security Considerations

- No hardcoded API keys; all secrets via environment variables
- Input validation via Pydantic models with strict typing
- Invalid `task_id` in `reset()` raises explicit `ValueError` (not unhandled 500)
- Markdown user content escaped via `_escape_md()` to prevent XSS
- WebSocket endpoints are unauthenticated (acceptable for HF Spaces)
- Container runs with minimal privileges
