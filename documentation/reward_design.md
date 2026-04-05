# Reward Design — Hallucination Detector Gym

## Philosophy

The reward function is designed to provide **dense, partial-progress signals** across the full agent trajectory. Unlike binary end-of-episode rewards common in many RL environments, every agent action receives immediate feedback proportional to its quality. This is critical for effective RL training because:

1. **Credit assignment**: Dense rewards make it easier for the agent to learn which specific actions lead to positive outcomes.
2. **Exploration guidance**: Partial rewards for partially correct actions (e.g., detecting a hallucination but with imperfect span) guide the agent toward the correct behaviour.
3. **Anti-exploitation**: Deduplication tracking and repeated-action penalties prevent degenerate policies from farming rewards.

## Reward Components

### 1. Detection Reward (+0.30 / -0.15)

Awarded when the agent correctly identifies whether a passage contains a hallucination.

- **Correct detection** (`hallucination_detected == True` when passage has hallucinations): +0.30
- **Wrong detection**: -0.15
- Each annotation can only be detected **once**. Subsequent detections of the same hallucination earn 0 reward.

### 2. Span Overlap Bonus (+0.20 * overlap)

When the agent provides a `hallucinated_span` during detection, bonus reward is awarded based on how well the span matches the ground truth.

The overlap metric combines:
- **Jaccard similarity** (bag-of-words): `|intersection| / |union|`
- **LCS ratio** (sequence-aware): `2 * LCS_length / (len_a + len_b)`
- **Final score**: `(jaccard + lcs) / 2.0`

Threshold: overlap must exceed **0.3** to earn any bonus. This prevents random guessing from yielding signal.

### 3. Classification Reward (+0.30 / -0.10)

Awarded when the agent correctly classifies the hallucination type:
- `factual_error`: An incorrect fact
- `entity_fabrication`: A non-existent entity presented as real
- `logical_inconsistency`: Contradicts the source or self-contradicts

Each annotation can only be classified once.

### 4. Correction Reward (+0.20 * similarity)

Awarded when the agent proposes a correction that matches the expected fix. Uses the same overlap metric as span matching, with a 0.3 threshold.

Each annotation can only be corrected once.

### 5. Step Efficiency Bonus (+0.10 * efficiency)

Awarded at episode submission. Agents that solve the task in fewer steps receive a bonus:

```
efficiency = 1.0 - (steps_taken / max_steps)
bonus = 0.10 * efficiency
```

This encourages agents to be decisive rather than exploratory.

### 6. Penalties

| Penalty | Amount | Condition |
|---------|--------|-----------|
| Wrong detection | -0.15 | `hallucination_detected` does not match ground truth |
| Wrong classification | -0.10 | Incorrect hallucination type |
| Noop when hallucinations exist | -0.05 | Agent skips when there are undetected hallucinations |
| Repeated action | -0.05 | Same action (type + content) submitted 3 times in a row |

## Anti-Exploitation Measures

### Annotation Deduplication

Each ground-truth annotation tracks three independent sets:
- `_detected_annotations`: Annotations that have been correctly detected
- `_classified_annotations`: Annotations that have been correctly classified
- `_corrected_annotations`: Annotations that have been correctly corrected

An agent cannot earn detection reward for the same hallucination twice.

### Content-Aware Repeat Detection

The repeated action penalty uses a content-aware action representation:
```
action_repr = "detect:munich, germany:factual_error"
```

Two different detect actions (different spans) are NOT considered repeated. Only truly identical actions are penalised.

### Span Matching Robustness

The combined Jaccard + LCS metric prevents:
- **Bag-of-words gaming**: Submitting jumbled words from the passage
- **Keyword stuffing**: Including many extra words to inflate intersection
- **Order manipulation**: Reordering words to match Jaccard without understanding

## Theoretical Score Bounds

**Per annotation:** detect (0.30) + span (0.20) + classify (0.30) + correct (0.20) = **1.0**

**Final grader score:** `clamp(cumulative / num_annotations, 0.0, 1.0)`

| Task | Annotations | Max Raw Reward | Max Score |
|------|------------|----------------|-----------|
| Easy | 1 | 1.0 + efficiency bonus | 1.0 |
| Medium | 2 | 2.0 + efficiency bonus | 1.0 |
| Hard | 3 | 3.0 + efficiency bonus | 1.0 |
