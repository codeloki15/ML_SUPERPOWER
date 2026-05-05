---
name: dl-finetune-loop
description: Use to construct the training loop for any DL task (CV, NLP, LLM, VLM) once data is loaded, augmentation is wired, env is detected, tracking + checkpointing are configured. Picks HF Trainer (default for standard finetune) vs Accelerate (custom loop with bespoke loss / multi-task / manual gradient handling) based on task complexity. Do NOT use for inference-only scripts or for evaluation harnesses (use the relevant dl-{cv,nlp,llm}-eval-* skill).
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# Finetune Loop

Construct the training loop. Decide between HF Trainer (high-level, handles 90% of cases) and Accelerate (low-level, for custom loops). Wire mixed precision, gradient accumulation, lr scheduling, and the callbacks for tracking and checkpointing — but DO NOT do those things itself; those are owned by `dl-experiment-track` and `dl-checkpoint` already.

## When to invoke

- After data is loaded (`dl-load-data`), augmentation is wired (`dl-augment`), env is known (`dl-detect-env`), tracking is wired (`dl-experiment-track`), checkpointing is wired (`dl-checkpoint`).
- The task is a "standard" finetune of a HF-compatible model (CV, NLP, encoder/decoder LLM).
- User asks for a "training loop" or "finetune".

## When NOT to invoke

- Inference-only scripts (use `dl-llm-serve` or write inference directly).
- Evaluation harnesses (use `dl-cv-eval-*` or `dl-nlp-eval-*`).
- LLM finetune that already uses Unsloth (Unsloth ships its own loop; `dl-llm-lora` handles).

## Decision rules

Pick Trainer vs Accelerate based on the task's needs:

- **HF Trainer** is the default. Use it WHEN:
  - Standard supervised loss (CrossEntropy, MSE, etc.) handled by the model's `forward(labels=...)`.
  - Standard eval loop (predict on val set, compute metric, log).
  - Standard callbacks (logging, eval, checkpointing).
  - FSDP via `TrainingArguments(fsdp=...)` is sufficient for distributed.

- **Accelerate** (custom loop with `Accelerator.prepare()`). Use it WHEN:
  - Loss combines multiple terms with custom weighting.
  - Multi-task training with separate forward passes.
  - Manual gradient manipulation (e.g., GradCache, gradient surgery).
  - Custom batch construction outside the dataloader.
  - User explicitly requests "custom training loop" or "raw PyTorch".

- IF the task is genuinely simple (single loss, single eval, single dataloader): default to **Trainer**. Do NOT introduce Accelerate complexity unless the task needs it.

- IF Unsloth is the chosen LoRA path (`dl-llm-lora` decided so): do NOT invoke this skill. Unsloth ships its own loop.

## Process

### Step 1 — Determine task complexity

Read `<workdir>/data_policy.json` and `<workdir>/env.json`. Determine:
- Task type (single-task vs multi-task).
- Loss shape (single vs combined).
- Distributed (single-GPU vs FSDP / DeepSpeed — read from `dl-distributed`'s output).
- Whether a custom loop is needed (user request, multi-task, custom loss).

### Step 2 — Apply decision rules

Pick Trainer or Accelerate. Report the choice with one-sentence rationale.

### Step 3 — Generate the training loop

For **HF Trainer** path: generate a `train.py` that:
- Loads model via `AutoModelFor{TaskType}.from_pretrained(model_id)`.
- Constructs `TrainingArguments` with lr, batch size, grad_accum, scheduler, mixed precision.
- Wires the tracker callback (already configured by `dl-experiment-track`).
- Wires checkpointing (`save_strategy="steps"`, `save_steps`, `save_total_limit` — already configured by `dl-checkpoint`).
- Constructs `Trainer(model, args, train_dataset, eval_dataset, tokenizer/data_collator, compute_metrics)`.
- Calls `trainer.train()` and `trainer.save_model()`.

For **Accelerate** path: generate a `train.py` that:
- `accelerator = Accelerator(mixed_precision="bf16")`.
- `model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)`.
- Manual training loop with `accelerator.backward(loss)`, `accelerator.log({...})`.
- Manual checkpoint save via `accelerator.save_state(...)`.

### Step 4 — Wire mixed precision

Pick the precision based on hardware (read from `<workdir>/env.json`):
- A100, H100, RTX 30xx/40xx → `bf16=True`.
- T4, V100 → `fp16=True`.
- CPU/MPS → `fp16=False, bf16=False` (or use bf16 on MPS if available; otherwise stay fp32).

For Trainer: set in `TrainingArguments(bf16=True)` or `fp16=True`.
For Accelerate: pass to `Accelerator(mixed_precision="bf16")`.

### Step 5 — Verify

Run a 10-step smoke test on the training data. Confirm:
- Loss decreases (or at least changes — flat loss = broken).
- No OOM, no NaN.
- Tracker shows step count + metrics.
- Checkpoint dir starts to populate.

If anything fails, hand off to `dl-debug-training` with the failure context.

## Recipe template

### `<workdir>/src/_train_trainer.py` (HF Trainer path)

```python
"""HF Trainer training loop. Adapt model/dataset/metric per task."""
import json
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()
ENV = json.loads((WORKDIR / "env.json").read_text())
POLICY = json.loads((WORKDIR / "data_policy.json").read_text())

ACTIVE_ENV = ENV["environments"][ENV["active"]]
DEVICE = ACTIVE_ENV.get("device", "cpu")


def pick_precision():
    """Pick bf16 vs fp16 based on the active device."""
    if DEVICE == "cuda":
        return {"bf16": True}  # safe default for A100/H100/30xx/40xx
    if DEVICE == "mps":
        return {"bf16": False, "fp16": False}  # MPS bf16 support is uneven; default to fp32
    return {"bf16": False, "fp16": False}


def make_training_args(output_dir: str | None = None, **base_kwargs) -> "TrainingArguments":
    from transformers import TrainingArguments
    output_dir = output_dir or str(WORKDIR / "checkpoints")
    # Caller-supplied precision overrides device defaults; pop before merging to avoid duplicate-kwarg TypeError.
    precision = pick_precision()
    for key in ("bf16", "fp16", "fp16_full_eval", "bf16_full_eval"):
        if key in base_kwargs:
            precision[key] = base_kwargs.pop(key)
    return TrainingArguments(
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=base_kwargs.pop("save_steps", 100),
        save_total_limit=3,
        save_safetensors=True,
        logging_steps=base_kwargs.pop("logging_steps", 10),
        eval_strategy=base_kwargs.pop("eval_strategy", "steps"),
        eval_steps=base_kwargs.pop("eval_steps", 100),
        report_to=base_kwargs.pop("report_to", ["wandb"]),
        run_name=base_kwargs.pop("run_name", WORKDIR.name),
        **precision,
        **base_kwargs,
    )


def run_smoke_test(trainer, max_steps: int = 10):
    """Run a tiny number of steps to confirm the loop works before committing to a full run."""
    original_max_steps = trainer.args.max_steps
    trainer.args.max_steps = max_steps
    trainer.train()
    trainer.args.max_steps = original_max_steps
    print(f"Smoke test: {max_steps} steps completed without error.")
```

### `<workdir>/src/_train_accelerate.py` (Accelerate custom loop)

```python
"""Accelerate custom training loop. Use when standard Trainer can't express the task."""
import json
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def make_accelerator():
    from accelerate import Accelerator
    env = json.loads((WORKDIR / "env.json").read_text())
    device = env["environments"][env["active"]].get("device", "cpu")
    precision = "bf16" if device == "cuda" else "no"
    return Accelerator(mixed_precision=precision, log_with="wandb")


def train_loop(accelerator, model, optimizer, scheduler, train_loader, eval_loader, num_epochs, compute_loss):
    """Generic Accelerate loop. compute_loss(model, batch) -> scalar tensor."""
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )
    accelerator.init_trackers(WORKDIR.name)

    step = 0
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                loss = compute_loss(model, batch)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step % 10 == 0:
                accelerator.log({"train/loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=step)
            step += 1

    accelerator.end_training()
```

## Hard constraints

- NEVER use `Trainer` AND `Accelerator.prepare(model)` together. They're mutually exclusive code paths; combining will produce subtle distributed bugs.
- NEVER skip the smoke test (Step 5). A 10-step smoke is cheap; a 10-hour broken training run is not.
- NEVER set `bf16=True` on hardware that does not support bf16 (T4, V100, pre-Volta GPUs). Use `fp16=True` instead.
- NEVER mix `bf16` and `fp16` flags. Pick one.
- NEVER re-implement what `dl-experiment-track` and `dl-checkpoint` already provide. Wire their outputs in; do not duplicate.
- NEVER set `report_to=[]` silently. If user declined tracking, set `report_to=[]` AND surface a `[no tracking]` banner.

## Research hooks

Trainer/Accelerate APIs evolve with each HF release. Before generating the loop for an unfamiliar combination, invoke `ml-engineer-research`:

- **TrainingArguments deprecations.** Query: *"Recently deprecated `TrainingArguments` parameters in HF Transformers as of {today} (e.g., `evaluation_strategy` → `eval_strategy`)."*
- **Accelerate prepare semantics.** Query: *"Current `Accelerator.prepare()` ordering and side-effects for distributed training as of {today}."*
- **Mixed precision recommendations per GPU.** Query: *"Current bf16 vs fp16 recommendation for `{gpu_class}` as of {today}."*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- A training script exists at `<workdir>/src/train.py` (or wherever the orchestrator placed it).
- The script imports the right Trainer/Accelerator class per the chosen path.
- The script wires `dl-experiment-track`'s init AND finish calls.
- The script wires `dl-checkpoint`'s `save_strategy` / `save_steps` / `save_total_limit`.
- A 10-step smoke test ran without OOM / NaN / Inf.
- For Trainer path: `report_to` is set (either to a real tracker OR `[]` with a `[no tracking]` banner).

## Output checklist

- [ ] Task complexity assessed; Trainer or Accelerate chosen
- [ ] Training script generated in `<workdir>/src/train.py`
- [ ] Mixed precision picked per active device
- [ ] Tracker + checkpoint wired (not duplicated)
- [ ] 10-step smoke test ran clean
- [ ] No mutual-exclusion violations (Trainer + Accelerate, bf16 + fp16, Unsloth + this skill)
