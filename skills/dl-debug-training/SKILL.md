---
name: dl-debug-training
description: Use when a DL training run produces NaN loss, infinite loss, gradient explosion, OOM error, training divergence (loss increases monotonically), eval metric stuck at random-baseline, or the model output is nonsense (all zeros, all same token, etc). Performs root-cause triage across the data → model → optimizer → loss pipeline. Do NOT use for tabular ML failures (use ml-engineer-debug instead) or for non-training failures like data loading errors.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# Debug Training

> **Iron Law: no fix without root-cause investigation first.** (Inherited from `ml-engineer-debug`.)

Root-cause triage skill specific to DL training failures. Adopts a 5-phase decomposition specialized for DL training failures, building on the root-cause-first discipline of `ml-engineer-debug`. The five phases are: (1) read the failure, (2) form hypotheses, (3) probe to confirm, (4) apply the smallest fix, (5) escalate to a full-pipeline audit if the same step fails 3 times. Do NOT patch symptoms; do NOT skip Phase 3.

## When to invoke

- Training loss is NaN or Inf at any point.
- Loss explodes (grows by >10x in a few steps).
- Loss diverges (goes up monotonically over many steps).
- OOM during forward, backward, or optimizer step.
- Eval metric stuck at random-baseline level.
- Model outputs are degenerate (all zeros, all same token, repetitive, garbage).
- Gradient norm > 100 or < 1e-8 consistently.

## When NOT to invoke

- Tabular ML failures — use `ml-engineer-debug`.
- Data loading errors before training starts — that's a data prep problem, not a training problem.
- The user is asking about a metric being "lower than expected" but training itself completed cleanly — that's an evaluation question, not a debug question.

## Phases

### Phase 1 — Read the failure

Capture exactly:
- The error message and stack trace (if any).
- The last 100 training-step log lines (loss, grad_norm, lr, eval metric, step number).
- The model config (size, dtype, attention impl).
- The training config (lr, batch size, gradient_accumulation_steps, scheduler, optimizer, grad_clip, mixed_precision).
- The data shape (sequence length distribution, batch composition).
- The hardware state (which device, free VRAM at failure, distributed strategy if any).

Do NOT propose a fix yet.

### Phase 2 — Form hypotheses

Match the failure shape to known patterns:

| Failure shape | Most-likely root causes (ranked) |
|---|---|
| NaN loss step 0 | (1) input contains NaN/Inf (data issue); (2) lr too high; (3) loss function received invalid input (e.g., log(0)). |
| NaN loss after N steps | (1) lr too high (most common); (2) gradient explosion not caught by clipping; (3) bf16/fp16 precision overflow; (4) batch with extreme outlier. |
| Loss explodes | (1) lr too high; (2) no gradient clipping; (3) bad init for a custom layer. |
| Loss diverges (monotonic increase) | (1) wrong loss function for the task; (2) labels off-by-one (e.g., 0/1 vs -1/+1); (3) model frozen by accident (no trainable params); (4) lr scheduler is increasing instead of decreasing; (5) data shuffling broken (always same batch). |
| OOM forward | (1) batch size too large; (2) seq length too long; (3) activation checkpointing disabled; (4) model too big for the device. |
| OOM backward | (1) optimizer state dominates (use 8-bit Adam); (2) Adam moments are fp32 even though model is bf16. |
| OOM optimizer step | (1) Adam state size = 2x model size; (2) FSDP2 misconfigured (full state dict gathered to rank 0). |
| Stuck at random baseline | (1) lr too low; (2) labels shuffled; (3) loss function ignoring some classes; (4) frozen backbone with empty head; (5) wrong tokenizer for the model. |
| Degenerate output | (1) softmax temperature collapsed; (2) repetition penalty missing in generation; (3) class imbalance crushing minority predictions; (4) model never actually trained (lr=0, optimizer.step never called). |

For each plausible cause, write a one-line hypothesis with what evidence would confirm or refute it.

### Phase 3 — Probe

For each top-2 hypothesis, run a minimal probe BEFORE touching the training script:

- **Hypothesis "lr too high":** plot loss curve from logs. If loss spiked at a specific step, check the lr at that step. If lr was at peak (cosine schedule), that's the cause.
- **Hypothesis "input has NaN":** load one batch from the dataloader and run `torch.isnan(batch).any()`.
- **Hypothesis "labels off by one":** print first 5 labels, compare to the model's expected label vocabulary.
- **Hypothesis "frozen by accident":** print `[p.requires_grad for p in model.parameters()]`. If all `False`, that's the cause.
- **Hypothesis "OOM batch size":** halve the batch size or seq length, re-run; if it works, you found the cause.
- **Hypothesis "wrong tokenizer":** print `tokenizer.vocab_size` and `model.config.vocab_size`. Mismatch → cause.

Probes MUST be small scripts run via `ml-engineer-execute`, not invasive edits to the training script.

### Phase 4 — Fix the root cause

Apply the smallest fix that addresses the confirmed cause. Common fixes:

- lr too high → reduce by 3-10x; for LLM finetune, typical safe range is 1e-5 to 5e-5 (full) or 1e-4 to 5e-4 (LoRA).
- Gradient explosion → set `max_grad_norm=1.0` in TrainingArguments; for custom loop, add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.
- bf16/fp16 overflow → switch to bf16 if you were on fp16 (bf16 has wider exponent range); on T4/V100 (no bf16), reduce lr or use loss scaling.
- OOM → enable gradient checkpointing (`model.gradient_checkpointing_enable()`); reduce batch size; increase grad_accumulation_steps to compensate; switch to 8-bit Adam (`bitsandbytes.optim.Adam8bit`).
- Frozen backbone → check `requires_grad`; for PEFT, ensure `model = get_peft_model(model, peft_config)` is called.
- Wrong tokenizer → load tokenizer from the same `model_id` as the model.
- Labels broken → fix the data prep, not the loss function.

### Phase 5 — Three-failure escape hatch

IF the same step has failed 3 times after attempted fixes, STOP patching. Audit:

- The data pipeline end-to-end: load → tokenize → collate → batch → input to model. Look for silent label corruption, mis-shaped tensors, wrong dtypes.
- The model architecture vs the loss function: are they compatible? E.g., regression model with cross-entropy loss = silent failure.
- The configuration assumptions: is the model in train mode? Are the trainable params what you expect? Is the optimizer stepping?

Report the audit to the user. Do NOT continue patching beyond 3 failures without user direction.

## Recipe template

The orchestrator generates probe scripts via `ml-engineer-write-code` Layout A and runs them via `ml-engineer-execute`. Below is a template probe library the orchestrator adapts per hypothesis.

### `<workdir>/debug_probes/_probe_lib.py`

```python
"""Library of one-shot probes for common DL training failures.

Each probe is a small function that returns a dict {confirmed: bool, evidence: str}.
The orchestrator picks 1-2 to run per hypothesis (Phase 3), reads the results,
then applies the matching fix from Phase 4.
"""
import json
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def probe_input_has_nan(dataloader_factory):
    """Returns confirmed=True if any tensor in the first batch contains NaN/Inf.

    Handles three common batch shapes: tensor, dict (HF collator), tuple/list (vanilla Dataset).
    """
    import torch
    loader = dataloader_factory()
    batch = next(iter(loader))
    if torch.is_tensor(batch):
        flat = batch
    elif isinstance(batch, dict):
        flat = next(iter(batch.values()))
    elif isinstance(batch, (list, tuple)) and len(batch) > 0:
        flat = batch[0]
    else:
        return {"confirmed": False, "evidence": f"unsupported batch type: {type(batch).__name__}"}
    has_nan = torch.isnan(flat).any().item()
    has_inf = torch.isinf(flat).any().item()
    return {
        "confirmed": bool(has_nan or has_inf),
        "evidence": f"has_nan={has_nan}, has_inf={has_inf}, shape={tuple(flat.shape)}",
    }


def probe_frozen_params(model):
    """Returns confirmed=True if zero parameters have requires_grad=True."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = list(model.parameters())
    return {
        "confirmed": len(trainable) == 0,
        "evidence": f"{len(trainable)}/{len(total)} parameters have requires_grad=True",
    }


def probe_tokenizer_mismatch(tokenizer, model):
    """Returns confirmed=True only when there are tokens unrepresented in the model's embedding.

    Healthy HF models often pad model.config.vocab_size to a multiple of 8/64 for tensor-core
    alignment, so model_size > tok_size is benign. The real bug is tok_size > model_size, which
    means the tokenizer can produce ids the embedding cannot handle (silent garbage at training).
    Use len(tokenizer) (which includes special tokens added at runtime) rather than the static
    tokenizer.vocab_size attribute.
    """
    try:
        tok_size = len(tokenizer)
    except TypeError:
        tok_size = getattr(tokenizer, "vocab_size", None)
    model_size = getattr(model.config, "vocab_size", None)
    if tok_size is None or model_size is None:
        return {"confirmed": False, "evidence": f"tok_size={tok_size}, model_size={model_size} (one missing)"}
    confirmed = tok_size > model_size  # tokens unrepresented in embedding = real bug
    return {
        "confirmed": confirmed,
        "evidence": f"len(tokenizer)={tok_size}, model.config.vocab_size={model_size}, padding_ok={model_size > tok_size}",
    }


def probe_label_distribution(dataloader_factory, num_batches: int = 5):
    """Prints the first N batches' labels for human inspection (off-by-one detection)."""
    loader = dataloader_factory()
    seen = []
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        labels = batch.get("labels") if isinstance(batch, dict) else batch[1]
        seen.append(labels.tolist() if hasattr(labels, "tolist") else labels)
    return {"confirmed": False, "evidence": f"first_{num_batches}_label_batches={seen}"}


def write_probe_result(name: str, result: dict):
    out_dir = WORKDIR / "debug_probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2))
    print(f"Probe {name}: confirmed={result['confirmed']} — {result['evidence']}")
```

### Per-hypothesis probe-script pattern

For each chosen hypothesis, write `<workdir>/debug_probes/<step>_<hypothesis>.py`:

```python
"""Probe for hypothesis <hypothesis>. Run via ml-engineer-execute."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from _probe_lib import probe_<name>, write_probe_result

# Build the factory / model / tokenizer EXACTLY as the failed training script does,
# so the probe sees the same state. Copy the imports and construction lines from
# the training script — do NOT re-invent them.
result = probe_<name>(...)
write_probe_result("<step>_<hypothesis>", result)
```

The orchestrator reads `<workdir>/debug_probes/*.json` to decide whether the hypothesis is confirmed and which Phase 4 fix to apply.

## Hard constraints

- NEVER apply a "fix" without confirming the hypothesis with a probe first. Symptom patching wastes time and can mask the real cause.
- NEVER raise `max_grad_norm` above 1.0 to "let gradients through". If clipping is hurting, the real cause is upstream (lr, init, data).
- NEVER suppress NaN by replacing with zero (`torch.nan_to_num`). It hides the bug and propagates corruption.
- NEVER reduce lr to a tiny number to "make it stable" if the real issue is broken data or wrong loss. The model will train slowly to a wrong answer.
- NEVER claim the bug is "fixed" without re-running for at least 100 steps and checking the failure mode no longer appears.

## Research hooks

DL training failure modes shift as the underlying libraries evolve. Before applying a fix in unfamiliar territory, invoke `ml-engineer-research`:

- **bf16/fp16 stability on current PyTorch.** Query: *"Known bf16/fp16 numerical-stability regressions or fixes in PyTorch as of {today}, especially around scaled dot-product attention and Adam moments."*
- **Optimizer state size on current bitsandbytes.** Query: *"Current memory footprint per param of `bitsandbytes.optim.Adam8bit` vs full Adam fp32 as of {today}."*
- **Known FSDP2 / DeepSpeed regressions.** Query: *"Open issues in HF Transformers / Accelerate / DeepSpeed related to FSDP2 sharded state-dict, ZeRO-3 CPU offload, or gradient accumulation as of {today}."*
- **PEFT freezing semantics.** Query: *"Has PEFT changed the default `requires_grad` semantics for LoRA / DoRA / QDoRA adapters as of {today}? What's the canonical pattern to verify trainable params are non-empty?"*

Coordinate with `dl-prior-art` if the failure shape resembles a known competition problem — winners often documented the exact failure mode and fix.

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- The training script ran for at least 100 steps without recurrence of the original failure.
- The fix applied is documented (a one-line comment in the script: `# fix: <root cause> — was {original symptom}`).
- The probe that confirmed the hypothesis is saved in `<workdir>/debug_probes/<step>_<hypothesis>.py` AND the result JSON is in `<workdir>/debug_probes/<step>_<hypothesis>.json` for future reference.
- After 3 consecutive failures on the same step: the audit (Phase 5) was performed and surfaced to the user before any further patching.

## Output checklist

- [ ] Failure read fully (error + last 100 log lines + configs + hardware state)
- [ ] Hypotheses ranked
- [ ] Top hypothesis confirmed with a probe
- [ ] Smallest fix applied that addresses confirmed cause
- [ ] Re-run for ≥100 steps; failure mode does not recur
- [ ] Fix documented in script comment + probe + result saved
- [ ] After 3 failures on same step: stopped patching, audited pipeline, surfaced to user
