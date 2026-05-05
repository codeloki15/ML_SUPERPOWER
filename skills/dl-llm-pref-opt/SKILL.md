---
name: dl-llm-pref-opt
description: Use to run preference optimization on an LLM after SFT — DPO (pairwise), KTO (binary thumbs), ORPO (single-pass SFT+pref), or GRPO (verifiable reward). Selector based on data shape; not a per-method skill. Do NOT use for SFT (use dl-llm-instruction-tune), pretraining, or non-LLM tasks.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# LLM Preference Optimization

Pick DPO / KTO / ORPO / GRPO based on the user's preference data shape and constraints. Hand off to TRL's corresponding trainer. Always run after SFT (except ORPO, which combines SFT + preference in one pass).

## When to invoke

- After `dl-llm-instruction-tune` has produced an SFT'd model (or skip if using ORPO).
- User has preference data: pairs (chosen/rejected), binary signals (thumbs-up/down), or verifiable rewards.

## When NOT to invoke

- Pure SFT (use `dl-llm-instruction-tune`).
- No preference data and no reward function — preference tuning needs one of these.
- Encoder NLP / VLM / non-LLM.

## Decision rules

Pick by data shape:

- **Pairwise (chosen + rejected per prompt) → DPO.** Standard, most common. Default if data has the shape.
- **Binary (thumbs-up or thumbs-down per prompt+response) → KTO.** Each example is independent — no pair needed.
- **Single-GPU, memory-tight, want SFT+pref in one pass → ORPO.** Combines SFT and preference learning; skips the explicit SFT step. Best when SFT step is expensive.
- **Verifiable reward (math, code, structured output where correctness is checkable programmatically) → GRPO.** RLHF with auto-grader; no separate reward model needed.

Library: TRL (all 4 methods supported via dedicated trainers). Axolotl wraps TRL for multi-GPU.

## Process

### Step 1 — Inspect preference data

Read first 5 examples; print to `<workdir>/samples/pref_data_inspection.txt`. Determine shape (pairwise / binary / reward).

### Step 2 — Apply decision rules

Pick method. Surface choice + rationale + ask user to confirm.

### Step 3 — Build the trainer

TRL `DPOTrainer` / `KTOTrainer` / `ORPOTrainer` / `GRPOTrainer`. Wire `dl-experiment-track` + `dl-checkpoint`.

### Step 4 — Smoke test (10 steps)

Verify loss is non-NaN, decreased or stable.

### Step 5 — Train + generation sanity check

After training, generate 5 completions on held-out prompts; compare to pre-pref-opt model. Save both to `<workdir>/samples/pref_comparison.txt`.

### Step 6 — Hand off to eval

`dl-llm-eval` for benchmarks.

## Recipe template

### `<workdir>/src/_pref_opt_trl.py`

```python
"""TRL preference-optimization trainers (DPO / KTO / ORPO / GRPO selector)."""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def make_dpo_trainer(model, ref_model, tokenizer, train_dataset, beta: float = 0.1, **kw):
    from trl import DPOConfig, DPOTrainer
    args = DPOConfig(
        output_dir=str(WORKDIR / "checkpoints"),
        beta=beta,
        per_device_train_batch_size=kw.pop("per_device_train_batch_size", 2),
        gradient_accumulation_steps=kw.pop("gradient_accumulation_steps", 4),
        learning_rate=kw.pop("learning_rate", 5e-6),
        num_train_epochs=kw.pop("num_train_epochs", 1),
        bf16=kw.pop("bf16", True),
        report_to=kw.pop("report_to", ["wandb"]),
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        **kw,
    )
    return DPOTrainer(model=model, ref_model=ref_model, args=args,
                      train_dataset=train_dataset, tokenizer=tokenizer)


def make_kto_trainer(model, ref_model, tokenizer, train_dataset, beta: float = 0.1, **kw):
    from trl import KTOConfig, KTOTrainer
    args = KTOConfig(
        output_dir=str(WORKDIR / "checkpoints"),
        beta=beta,
        per_device_train_batch_size=kw.pop("per_device_train_batch_size", 2),
        learning_rate=kw.pop("learning_rate", 5e-6),
        num_train_epochs=kw.pop("num_train_epochs", 1),
        bf16=kw.pop("bf16", True),
        report_to=kw.pop("report_to", ["wandb"]),
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        **kw,
    )
    return KTOTrainer(model=model, ref_model=ref_model, args=args,
                      train_dataset=train_dataset, tokenizer=tokenizer)


def make_orpo_trainer(model, tokenizer, train_dataset, beta: float = 0.1, **kw):
    """ORPO: single-pass SFT + preference. No ref_model needed."""
    from trl import ORPOConfig, ORPOTrainer
    args = ORPOConfig(
        output_dir=str(WORKDIR / "checkpoints"),
        beta=beta,
        per_device_train_batch_size=kw.pop("per_device_train_batch_size", 2),
        learning_rate=kw.pop("learning_rate", 8e-6),
        num_train_epochs=kw.pop("num_train_epochs", 3),
        bf16=kw.pop("bf16", True),
        report_to=kw.pop("report_to", ["wandb"]),
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        **kw,
    )
    return ORPOTrainer(model=model, args=args,
                      train_dataset=train_dataset, tokenizer=tokenizer)


def make_grpo_trainer(model, tokenizer, train_dataset, reward_fn, **kw):
    """GRPO: verifiable reward — reward_fn(prompt, completion) -> float."""
    from trl import GRPOConfig, GRPOTrainer
    args = GRPOConfig(
        output_dir=str(WORKDIR / "checkpoints"),
        per_device_train_batch_size=kw.pop("per_device_train_batch_size", 2),
        learning_rate=kw.pop("learning_rate", 1e-6),
        num_train_epochs=kw.pop("num_train_epochs", 1),
        bf16=kw.pop("bf16", True),
        report_to=kw.pop("report_to", ["wandb"]),
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        **kw,
    )
    return GRPOTrainer(model=model, reward_funcs=[reward_fn], args=args,
                       train_dataset=train_dataset, tokenizer=tokenizer)
```

## Hard constraints

- NEVER skip the SFT step before DPO/KTO/GRPO unless using ORPO. The reference model used for KL divergence must be a competent base; pure-base models without SFT diverge.
- NEVER use a reward model trained on a different prompt template than the policy. Template mismatch silently inverts the reward signal.
- NEVER mix preference data formats. Pick chosen/rejected OR binary OR reward — and structure the entire dataset uniformly.
- NEVER use DPO with `beta` outside [0.01, 0.5] without research backing. Standard range is 0.1-0.2.
- NEVER train preference optimization without a generation sanity check. The model often becomes more verbose / repetitive after pref-tuning even when loss looks fine.

## Research hooks

- **Current TRL trainer API.** Query: *"Has TRL renamed `DPOTrainer` arguments (e.g., `tokenizer` -> `processing_class`) as of {today}? Current `DPOConfig` / `KTOConfig` / `ORPOConfig` / `GRPOConfig` defaults."*
- **DPO vs ORPO quality on `{task_type}`.** Query: *"Latest comparison of DPO vs ORPO vs simpo / IPO on `{task_type}` as of {today}."*
- **GRPO reward-function patterns.** Query: *"Current best practices for `reward_funcs` in TRL GRPOTrainer (single fn vs list, verifiable vs LLM-as-judge) as of {today}."*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- Method (DPO/KTO/ORPO/GRPO) saved to `<workdir>/pref_opt_method.json`.
- For DPO/KTO: `ref_model` is the SFT'd model from `dl-llm-instruction-tune` (NOT the raw base).
- For GRPO: `reward_fn` is callable and returns finite floats on test prompts.
- 5 pre-vs-post generation comparisons saved to `<workdir>/samples/pref_comparison.txt`.
- 10-step smoke test loss is non-NaN.
- Tracker + checkpoint wired.

## Output checklist

- [ ] Preference data shape inspected (5 samples saved)
- [ ] Method picked per decision rules; user confirmed
- [ ] Trainer built with correct ref_model / reward_fn / beta
- [ ] Smoke test (10 steps) clean
- [ ] Full training ran
- [ ] Pre-vs-post generation comparison saved
- [ ] Adapter saved (not merged)
- [ ] Handed off to `dl-llm-eval`
