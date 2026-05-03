---
name: dl-checkpoint
description: Use to set up checkpoint saving and resume logic before any DL training run that may take >30 minutes, may be preempted (spot instances, free Colab tier), or may need partial-state inspection later. Covers HF save_pretrained, raw state_dict, sharded checkpoints, FSDP2 state-dict gotchas. Do NOT use for short runs (<30 min) where re-running from scratch is cheaper than checkpoint plumbing.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# Checkpoint

Set up checkpoint save / resume / inspect logic. The most common silent failure mode in long DL runs is "the job died at hour 8 and there's nothing on disk." This skill prevents that.

## When to invoke

- Any training run expected to take >30 minutes.
- Any run on a preemptible / spot instance (Vast.ai, Colab free, RunPod community).
- Multi-stage workflows where a downstream step needs the model from an earlier step (e.g., SFT then DPO; train then quantize).
- User explicitly says "save checkpoints", "resume from", "shard the checkpoint".

## When NOT to invoke

- Short runs (<30 min) on stable infrastructure where the whole script is cheap to re-run.
- Inference-only scripts.
- Scripts that already use HF Trainer with `save_strategy="epoch"` or `save_strategy="steps"` configured by the orchestrator — Trainer handles checkpointing natively, this skill mainly verifies the config is sane.

## Decision rules

Pick the save strategy based on training shape:

- IF using HF Trainer AND single-GPU: set `save_strategy="steps"`, `save_steps=<every ~10% of total steps>`, `save_total_limit=3` (keep last 3 to bound disk usage).
- IF using HF Trainer AND multi-GPU with FSDP2: set `save_strategy="steps"` AND `fsdp_state_dict_type="SHARDED_STATE_DICT"`. Reason: full state dict on FSDP2 requires gathering all parameters to rank 0, which OOMs on large models. Sharded checkpoints save per-rank shards.
- IF training a LoRA adapter (PEFT): save the adapter only, not the full base model. `model.save_pretrained(<path>)` on a PeftModel saves only the adapter weights (~MB) — orders of magnitude smaller than the base model.
- IF doing full finetune with bf16/fp16 weights: save in `safetensors` format (`safe_serialization=True`). Reason: faster load, immune to pickle vulnerabilities.
- IF the run is on a remote provider with ephemeral storage: also push checkpoints to a persistent location periodically (Modal volume, RunPod network volume, S3 if configured). Otherwise the checkpoint dies with the pod.

## Process

### Step 1 — Determine training shape

Read the training script and determine: HF Trainer or custom loop? Single-GPU or distributed? LoRA or full finetune? Local or remote? Read `<workdir>/env.json.active` (produced by `dl-detect-env`) to identify the active environment.

### Step 2 — Generate the checkpoint config

Produce the relevant config block. For HF Trainer:

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="<workdir>/checkpoints",
    save_strategy="steps",
    save_steps=<computed_value>,
    save_total_limit=3,
    save_safetensors=True,
    # FSDP2 only:
    fsdp="full_shard auto_wrap" if <distributed> else "",
    fsdp_config={"fsdp_state_dict_type": "SHARDED_STATE_DICT"} if <distributed> else None,
)
```

For PEFT/LoRA save:

```python
# At end of training (and at each save_steps interval via TrainerCallback):
model.save_pretrained("<workdir>/checkpoints/adapter")
tokenizer.save_pretrained("<workdir>/checkpoints/adapter")
```

For custom loop:

```python
import torch

def save_checkpoint(model, optimizer, scheduler, step, path):
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }
    torch.save(state, path)
```

### Step 3 — Generate resume logic

For HF Trainer: `trainer.train(resume_from_checkpoint=True)` — picks up the latest checkpoint in `output_dir` automatically.

For custom loop:

```python
import torch
from pathlib import Path

ckpt_dir = Path("<workdir>/checkpoints")
ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
if ckpts:
    latest = ckpts[-1]
    state = torch.load(latest, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state["scheduler"]:
        scheduler.load_state_dict(state["scheduler"])
    start_step = state["step"] + 1
    print(f"Resumed from {latest.name} at step {start_step}")
else:
    start_step = 0
```

### Step 4 — Add remote-persistence hook (if remote)

IF `env.json.active != "local"` AND the active env has ephemeral storage, add a periodic sync step. For Modal: write to a `modal.Volume` mounted at the checkpoint path. For SSH: add `rsync -av <workdir>/checkpoints/ user@persistent-host:backups/<run_name>/` on each save (called from a `TrainerCallback` or after the save in a custom loop). Coordinate with `dl-remote-execute` for the actual transport command for the active provider.

### Step 5 — Verify

After the script is wired up, the orchestrator runs the script for ~50 steps and confirms:
- A checkpoint file appeared in `<workdir>/checkpoints/`.
- The checkpoint can be loaded back (run a tiny resume-and-step test).
- For LoRA: only the adapter is saved, not the full base model (file size sanity check).

## Recipe template

The following template is adapted by the orchestrator. Two flavors: HF Trainer (most common) and custom loop (when the user has a non-Trainer training script).

### `<workdir>/src/_checkpoint_hf_trainer.py` (HF Trainer)

```python
"""Patches the existing TrainingArguments and adds a remote-sync callback if needed.

Insert AFTER training script imports, BEFORE Trainer construction.
"""
import os
from pathlib import Path
from transformers import TrainingArguments, TrainerCallback

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()
ACTIVE = os.environ.get("DL_ACTIVE_ENV", "local")
DISTRIBUTED = os.environ.get("DL_DISTRIBUTED", "0") == "1"


def make_checkpoint_args(total_steps: int, **base_kwargs) -> TrainingArguments:
    save_steps = max(1, total_steps // 10)
    extra = {}
    if DISTRIBUTED:
        extra["fsdp"] = "full_shard auto_wrap"
        extra["fsdp_config"] = {"fsdp_state_dict_type": "SHARDED_STATE_DICT"}
    return TrainingArguments(
        output_dir=str(WORKDIR / "checkpoints"),
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        save_safetensors=True,
        **extra,
        **base_kwargs,
    )


class RemoteSyncCallback(TrainerCallback):
    """Syncs <workdir>/checkpoints/ to a persistent remote location after each save."""

    def __init__(self, remote_target: str | None):
        self.remote_target = remote_target

    def on_save(self, args, state, control, **kwargs):
        if not self.remote_target:
            return
        # Provider-specific: rsync for SSH, modal.Volume for Modal, etc.
        # The orchestrator fills in the right command at generation time.
        # Use checked subprocess.run so a silent rsync failure does NOT pretend
        # the backup succeeded — surface non-zero so training can decide.
        import subprocess
        result = subprocess.run(
            ["rsync", "-av", f"{WORKDIR}/checkpoints/", f"{self.remote_target}/"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[checkpoint] WARNING: remote sync failed (exit {result.returncode}): {result.stderr.strip()}")
```

### `<workdir>/src/_checkpoint_custom.py` (custom training loop)

```python
"""Save / resume helpers for custom (non-Trainer) loops."""
import os
import torch
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def save_checkpoint(model, optimizer, scheduler, step: int, save_total_limit: int = 3):
    ckpt_dir = WORKDIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:08d}.pt"
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }
    torch.save(state, path)
    # Bound disk usage.
    existing = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for old in existing[:-save_total_limit]:
        old.unlink()


def load_latest_checkpoint(model, optimizer, scheduler):
    ckpt_dir = WORKDIR / "checkpoints"
    if not ckpt_dir.exists():
        return 0
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        return 0
    state = torch.load(ckpts[-1], map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    return state["step"] + 1
```

## Hard constraints

- NEVER use `save_strategy="no"` for runs >30 min unless the user explicitly opts out. Crashed long runs without checkpoints waste compute and money.
- NEVER use full state-dict (`fsdp_state_dict_type="FULL_STATE_DICT"`) on FSDP2 with models larger than the rank-0 GPU's VRAM. It will OOM.
- NEVER load checkpoints with `torch.load(path)` without `weights_only=True` if the source is untrusted. Pickle deserialization can execute arbitrary code. For your own checkpoints, this is moot but `weights_only=True` is the safer default; for downloaded checkpoints, prefer `safetensors`.
- NEVER overwrite the previous best checkpoint without bound. Use `save_total_limit` to bound disk usage; keep last N + best.
- NEVER assume the remote provider's "persistent" storage survives a pod stop. Verify per provider: Modal volumes survive, RunPod network volumes survive if attached, Vast.ai instance disk does NOT survive a destroy.

## Research hooks

Checkpoint APIs change with each major HF / PyTorch release. Before generating the config for an unfamiliar combination, invoke `ml-engineer-research`:

- **FSDP2 state-dict format.** Query: *"Current recommended `fsdp_state_dict_type` and resume API for FSDP2 + HF Trainer as of {today}, including any deprecated FSDP1 paths to avoid."*
- **PEFT save format.** Query: *"Does the current version of PEFT save adapter weights via `model.save_pretrained()` only, or has the API changed for DoRA / QDoRA / IA3 as of {today}?"*
- **Remote persistent storage primitives.** Query: *"Current persistent-storage primitive on `{provider}` for syncing checkpoints during training as of {today}."*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- The training script saves at least one checkpoint within the first ~50 steps.
- The checkpoint loads back without error (a tiny resume test runs end-to-end).
- For LoRA runs: checkpoint size is in the MB range (not GB) — confirms adapter-only save (e.g., total bytes under 500 MB for adapters of any common base model).
- For FSDP2 runs: `fsdp_state_dict_type` is `"SHARDED_STATE_DICT"` (NOT `"FULL_STATE_DICT"`).
- For ephemeral remotes: a persistence hook is wired up and a sync ran successfully on at least one save.

## Output checklist

- [ ] Save strategy chosen per training shape
- [ ] Checkpoint code inserted in the training script
- [ ] Resume logic inserted (or `resume_from_checkpoint=True` if Trainer)
- [ ] Remote persistence hook added if active env is ephemeral
- [ ] Verify pass: checkpoint appears + loads back + sane size
- [ ] FSDP2 sharded state-dict confirmed when distributed
- [ ] PEFT adapter-only save confirmed when LoRA
