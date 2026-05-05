---
name: dl-distillation
description: Use to compress a teacher model into a smaller student via logit distillation (KL divergence on softmax), feature distillation (MSE on hidden states), or CoT distillation (teacher generates reasoning chains, student trained on Q→reasoning→A). Recurring Kaggle finisher and 2025 LLM trend. Do NOT use when teacher is worse than student baseline, when latency isn't a concern (just deploy the teacher), or for non-trained models.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# Distillation

Compress a larger teacher model into a smaller student. Pick logit / feature / CoT distillation by goal. Student must be evaluated against teacher AND against student-trained-from-scratch baselines — distillation should beat both.

## When to invoke

- Teacher model is too large for serving (latency, memory, cost).
- User wants a smaller model with similar quality.
- LLM domain: CoT distillation to capture teacher's reasoning chains in a smaller model (2025 trend per spec).

## When NOT to invoke

- Teacher is worse than the student-trained-from-scratch baseline (you'd transfer worse priors).
- Latency / size isn't a concern — just use the teacher.
- For non-trained models (you need a teacher with logits or generations to distill from).

## Decision rules

### Distillation type

- **Logit distillation (KL divergence on teacher's softmax)**: classification / token classification. Standard, fast.
- **Feature distillation (MSE on hidden states)**: representation transfer; useful when student architecture differs significantly from teacher.
- **CoT (Chain-of-Thought) distillation**: LLM-specific. Teacher generates reasoning + answer; student trained on (Q → reasoning → A). 2025 trend; transfers reasoning capability into smaller models.

### Loss balance

Standard combo: `loss = alpha * teacher_loss + (1 - alpha) * student_loss`.
- alpha=0.7 (default) — heavy teacher signal.
- Lower if student converges to teacher mistakes; raise if student under-fits.

### Temperature (logit distillation)

- Standard: T=2-5. Higher temperature = softer distribution = more information to distill.
- T=1: identical to hard-label cross-entropy; no distillation benefit.
- T>10: distribution flattens too much; signal lost.

### Student size

- Logit distillation: student should be ~10-50% of teacher params.
- Feature distillation: student can be smaller; align hidden dims with projection layers if needed.
- CoT: student typically 1B-7B distilling from 70B+ teacher.

## Process

### Step 1 — Verify teacher quality

Compare teacher's metric to student-baseline metric (student trained on same data without distillation). Teacher MUST be better; if not, halt — distillation can't help.

### Step 2 — Pick distillation type

Apply decision rules. Surface choice.

### Step 3 — Pre-compute teacher outputs (if static)

For logit distillation on a fixed dataset: pre-compute teacher logits once and cache. Saves teacher forward passes during student training (often 5-10x speedup).

For CoT: use `dl-llm-serve` to spin up teacher, generate reasoning chains for each training prompt, save to JSONL.

### Step 4 — Train student with distillation loss

Custom Trainer subclass with combined loss:
- `student_logits = student(input)`
- `teacher_logits = cached or teacher(input)`
- `kl = KL(softmax(student_logits/T), softmax(teacher_logits/T)) * T**2`
- `ce = CE(student_logits, true_labels)` (if labels exist)
- `loss = alpha * kl + (1 - alpha) * ce`

### Step 5 — Verify

Eval student against:
- Teacher (target — student should approach but not match teacher).
- Student-baseline (student trained without distillation — student-with-distillation should beat).

If student does NOT beat student-baseline, distillation failed. Common causes: temperature too low, alpha too low, teacher logits stale, student too small.

## Recipe template

### `<workdir>/src/_distill_logit.py`

```python
"""Logit distillation via custom Trainer subclass."""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def make_distillation_trainer(student, teacher, train_dataset, eval_dataset=None,
                               temperature: float = 4.0, alpha: float = 0.7, **kw):
    """Returns a Trainer subclass that combines KL(student||teacher) + CE(student, labels)."""
    import torch
    import torch.nn.functional as F
    from transformers import Trainer, TrainingArguments

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    class DistillationTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            student_outputs = model(**inputs)
            student_logits = student_outputs.logits

            with torch.no_grad():
                # remove labels for teacher forward to avoid CE computation
                teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
                teacher_logits = teacher(**teacher_inputs).logits

            # KL distillation loss
            kl = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)

            # Standard CE on hard labels (if present)
            ce = student_outputs.loss if labels is not None else 0.0
            loss = alpha * kl + (1 - alpha) * ce
            return (loss, student_outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(WORKDIR / "checkpoints"),
        num_train_epochs=kw.pop("num_train_epochs", 3),
        per_device_train_batch_size=kw.pop("per_device_train_batch_size", 8),
        learning_rate=kw.pop("learning_rate", 1e-4),
        bf16=kw.pop("bf16", True),
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        report_to=kw.pop("report_to", ["wandb"]),
        run_name=kw.pop("run_name", WORKDIR.name + "_distill"),
        **kw,
    )

    return DistillationTrainer(
        model=student, args=args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
    )
```

### `<workdir>/src/_distill_cot.py` (LLM Chain-of-Thought distillation)

```python
"""CoT distillation: teacher generates reasoning + answer; student trained on (Q -> reasoning -> A)."""
import json
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def generate_cot_dataset(teacher_serve_endpoint: str, prompts: list[str],
                         system_prompt: str = "Reason step by step before answering.",
                         max_tokens: int = 512) -> list[dict]:
    """Use dl-llm-serve to generate (prompt, reasoning, answer) triples for CoT distillation."""
    import requests
    out = []
    for prompt in prompts:
        payload = {
            "model": "teacher",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens, "temperature": 0.0,
        }
        r = requests.post(f"{teacher_serve_endpoint}/v1/chat/completions", json=payload, timeout=120)
        r.raise_for_status()
        completion = r.json()["choices"][0]["message"]["content"]
        out.append({"prompt": prompt, "completion": completion})

    out_path = WORKDIR / "cot_distillation_dataset.jsonl"
    with out_path.open("w") as f:
        for record in out:
            f.write(json.dumps(record) + "\n")
    print(f"CoT dataset saved to {out_path}")
    return out
```

## Hard constraints

- NEVER distill from a teacher that is WORSE than the student-trained-from-scratch baseline. You'd transfer the teacher's worse priors.
- NEVER use temperature outside [1, 10] without research backing. Standard is 2-5.
- ALWAYS evaluate the distilled student against BOTH the teacher AND the student-baseline. Distillation should beat student-baseline; if not, it failed.
- NEVER cache teacher logits across model versions. If the teacher retrained, regenerate teacher outputs.
- NEVER distill at temperature=1 — that's equivalent to hard-label cross-entropy; no distillation benefit.
- NEVER skip the alpha balance check. alpha=1 (pure teacher) ignores hard labels; alpha=0 (pure CE) is just normal training.

## Research hooks

- **CoT distillation effectiveness on `{benchmark}`.** Query: *"Latest results for CoT distillation from `{teacher_size}` to `{student_size}` on `{benchmark}` as of {today}."*
- **Temperature / alpha defaults per task.** Query: *"Recommended distillation temperature and alpha for `{task_type}` (classification / NER / generation) as of {today}."*
- **Feature distillation alignment.** Query: *"Current best practice for hidden-dim alignment in feature distillation when student.hidden_size != teacher.hidden_size as of {today}."*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- Teacher metric is better than student-baseline metric.
- Distilled student metric > student-baseline metric (if not, distillation failed).
- Distillation type recorded in `<workdir>/distillation_config.json`.
- Temperature in [1, 10]; alpha in [0, 1].
- For CoT: cached generation dataset (jsonl) exists; teacher generations have non-trivial length (avg > 50 tokens).
- Teacher set to eval mode AND requires_grad=False (no teacher gradient updates).

## Output checklist

- [ ] Teacher quality verified > student-baseline
- [ ] Distillation type chosen (logit / feature / CoT)
- [ ] Teacher outputs pre-computed/cached (or live generation for CoT)
- [ ] Distillation loss wired correctly (KL + CE combo, T scaling)
- [ ] Student trained
- [ ] Eval: distilled student > student-baseline (verified)
- [ ] User informed if distillation degraded (didn't beat baseline)
