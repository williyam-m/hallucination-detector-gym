---
base_model: Qwen/Qwen3-0.6B
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
language:
  - en
tags:
  - base_model:adapter:Qwen/Qwen3-0.6B
  - grpo
  - lora
  - transformers
  - trl
  - hallucination-detection
  - ai-safety
  - reinforcement-learning
  - openenv
datasets:
  - custom (hallucination-detector-gym tasks)
---

# Hallucination Detector Agent — Qwen3-0.6B (GRPO LoRA)

A GRPO-fine-tuned LoRA adapter for hallucination detection, classification, and correction in LLM-generated text.

## Model Details

- **Base Model**: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache 2.0, ungated)
- **Fine-tuning Method**: GRPO (Group Relative Policy Optimization) via [TRL](https://github.com/huggingface/trl)
- **Adapter**: LoRA (rank=16, ~1.5M trainable parameters)
- **Training Data**: 3 hallucination detection tasks from the Hallucination Detector Gym environment
- **Developed by**: Hallucination Detector Gym Team
- **License**: Apache 2.0

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen3-0.6B |
| Method | GRPO + LoRA |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Trainable Parameters | ~1.5M |
| GRPO Generations | 2 completions per prompt |
| KL Beta | 0.04 |
| Learning Rate | 5e-6 (cosine schedule) |
| Epochs | 3 |
| Thinking Mode | Disabled (structured JSON output) |

## Reward Functions

The GRPO trainer uses 3 weighted reward signals from the Hallucination Detector Gym:

| Reward | Weight | Description |
|--------|--------|-------------|
| `reward_format` | 1.0 | Valid JSON output with correct action fields |
| `reward_detection` | 2.0 | Hallucination detection accuracy + span overlap + type match |
| `reward_correction` | 1.5 | Correction quality via span similarity |

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = PeftModel.from_pretrained(base_model, "williyam/hallucination-detector-agent-qwen3-0.6b")
tokenizer = AutoTokenizer.from_pretrained("williyam/hallucination-detector-agent-qwen3-0.6b")

prompt = """You are a hallucination detector. Analyse the passage and output a JSON action.

Passage: Albert Einstein was born in Munich, Germany in 1879.
Source: Albert Einstein was born in Ulm, in the Kingdom of Württemberg.

Output a JSON action:"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Environment

This model was trained using the [Hallucination Detector Gym](https://huggingface.co/spaces/williyam/hallucination-detector-gym) OpenEnv environment. The full training pipeline is documented in the `training_hallucination_detector.ipynb` notebook.

## Links

- **Environment**: [HF Space](https://huggingface.co/spaces/williyam/hallucination-detector-gym)
- **Repository**: [GitHub](https://github.com/williyam/hallucination-detector-gym)
- **Training Notebook**: `training_hallucination_detector.ipynb`

### Framework Versions

- PEFT 0.18.1
- Transformers (latest)
- TRL (latest)