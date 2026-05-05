---
name: dl-distributed
description: Use to pick a distributed training strategy (single-GPU, FSDP2 via Accelerate, or DeepSpeed ZeRO-3) when a model does not fit on one GPU OR when the user explicitly asks about multi-GPU training. Generates the launcher command and config. Do NOT use when the model fits on one GPU and the user has not asked for parallelism — added complexity hurts more than it helps.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# Distributed

Decision skill: pick single-GPU / FSDP2 / DeepSpeed ZeRO-3 based on model size, GPU count, VRAM budget, and the active environment. Generates the launcher command (`accelerate launch`, `torchrun`, `deepspeed`) and the config file the launcher needs.

## When to invoke

- Estimated model + optimizer + activations memory exceeds available VRAM on a single GPU.
- User explicitly says "multi-GPU", "FSDP", "DeepSpeed", "ZeRO", "shard the model".
- Training a model >7B params (sharding is almost always the answer).

## When NOT to invoke

- Model fits on one GPU. Distributed adds latency, complexity, and failure modes; do not use it as default.
- Inference-only scripts (vLLM and SGLang have their own tensor-parallel logic; they do not need this skill).
- LoRA / QLoRA training of a model that fits on one GPU (the base model is frozen; the trainable params are tiny).

## Decision rules

Read `<workdir>/env.json.environments[active]` (produced by `dl-detect-env`) for GPU count and per-GPU VRAM. Estimate memory: model params × bytes_per_param + optimizer state (Adam = 2× params for fp32 moments) + activations. The orchestrator may have a rough estimate from `dl-detect-env` or from a research hook. If unknown, compute a conservative estimate from the model's HF config.

Apply this decision tree:

- IF model fits on one GPU AND only one GPU is available: do NOT invoke this skill — use single-GPU training.
- IF model fits on one GPU AND multiple GPUs are available AND the user did not ask for parallelism: stay single-GPU. Multi-GPU training of a single-GPU-sized model is rarely worth the complexity.
- IF model does NOT fit on one GPU AND total VRAM across GPUs >= memory requirement: use **FSDP2 via Accelerate**. Reason: PyTorch-native, simpler config, better-supported across 2025-2026 HF stack. Default for medium models.
- IF model does NOT fit even with FSDP2 across all available GPUs: use **DeepSpeed ZeRO-3**. Reason: ZeRO-3 with CPU offload can train models that don't fit on the GPU pool at all (offloads optimizer state to CPU/NVMe). Slower per step but fits larger models.
- IF user explicitly requests Unsloth (single-GPU LLM finetune) AND has only one GPU: use Unsloth. Unsloth handles its own memory layout and does NOT support multi-GPU. If the user has multiple GPUs and wants Unsloth, surface this contradiction and ask whether to switch to Axolotl (multi-GPU) or proceed single-GPU on one device.

## Process

### Step 1 — Estimate memory requirement

Compute or look up:
- Model params × bytes (bf16/fp16 = 2 bytes; fp32 = 4 bytes; 4-bit quant ≈ 0.5 bytes)
- Optimizer state (Adam fp32 = 8 bytes per param; AdamW with fused = same; SGD = 4 bytes; 8-bit Adam ≈ 2 bytes)
- Gradients (same as model params unless using gradient checkpointing → ~30% reduction in activation memory at cost of recompute)
- Activations (depends on batch size, sequence length, and recompute strategy)

**Research hook:** Before final estimate, invoke `ml-engineer-research` for: *"Current memory-per-param estimate for {detected_optimizer} on {detected_model_family} {model_size} at {bf16|fp16|4-bit} including activations for sequence length {seq_len} and batch size {batch_size}, as of {today}."* Use the research result to adjust the estimate.

### Step 2 — Apply decision rules

Walk the decision tree above. Report the chosen strategy to the user with one-sentence rationale.

### Step 3 — Generate config

For **FSDP2 via Accelerate**:

Generate `<workdir>/configs/accelerate_fsdp.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
mixed_precision: bf16
num_machines: 1
num_processes: <gpu_count>
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
```

Launcher command:
```bash
accelerate launch --config_file <workdir>/configs/accelerate_fsdp.yaml <workdir>/src/train.py
```

For **DeepSpeed ZeRO-3**:

Generate `<workdir>/configs/zero3.json`:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "offload_param": {"device": "cpu", "pin_memory": true},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto"
  },
  "bf16": {"enabled": true},
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

Launcher command:
```bash
deepspeed --num_gpus=<gpu_count> <workdir>/src/train.py --deepspeed <workdir>/configs/zero3.json
```

For **Unsloth single-GPU**:

No launcher config needed. Run with regular `python <script.py>`. Note in the script: do NOT wrap in `accelerate.Accelerator()` — Unsloth manages its own optimization.

### Step 4 — Wire into the training script

Modify the training script to one of three mutually exclusive paths:
- **FSDP2 path:** use the HF Trainer-managed FSDP integration via `TrainingArguments(fsdp="full_shard auto_wrap", fsdp_config={...})` — see `_train_fsdp.py` skeleton in the Recipe template. Do NOT also wrap the model in `Accelerator.prepare()` — Trainer-managed FSDP and standalone Accelerator are different code paths.
- **DeepSpeed path:** pass `deepspeed=<workdir>/configs/zero3.json` to `TrainingArguments` — see `_train_deepspeed.py` skeleton.
- **Unsloth path:** load the model via `FastLanguageModel.from_pretrained()` and use plain `Trainer` (or `SFTTrainer`) — see `_train_unsloth.py` skeleton. Unsloth and FSDP2 are mutually exclusive — do NOT try to combine.

## Recipe template

The orchestrator chooses one of three skeletons based on the decision rules and writes the chosen one into the training script. Coordinate with `dl-remote-execute` for the actual launch command on the active provider (different providers have different launcher conventions: Modal uses `@app.function(gpu=...)`, RunPod uses raw `accelerate launch` over SSH, etc.).

### `<workdir>/src/_train_fsdp.py` (FSDP2 path)

```python
"""Patches model + Trainer for FSDP2. Insert after model construction, before Trainer init."""
import os
from pathlib import Path
from transformers import TrainingArguments

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()
GPU_COUNT = int(os.environ.get("DL_GPU_COUNT", "1"))


def make_fsdp_args(**base_kwargs) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(WORKDIR / "checkpoints"),
        bf16=True,
        fsdp="full_shard auto_wrap",
        # NOTE: HF Transformers historically accepted both `fsdp_<key>` and bare
        # `<key>` forms inside fsdp_config. Recent versions may prefer the
        # un-prefixed form. See the "fsdp_config key naming" research hook.
        fsdp_config={
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
            "fsdp_sharding_strategy": "FULL_SHARD",
            "fsdp_use_orig_params": True,
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_backward_prefetch": "BACKWARD_PRE",
        },
        **base_kwargs,
    )
```

### `<workdir>/src/_train_deepspeed.py` (DeepSpeed ZeRO-3 path)

```python
"""Patches Trainer to use the ZeRO-3 config. Insert before Trainer init."""
import os
from pathlib import Path
from transformers import TrainingArguments

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def make_deepspeed_args(**base_kwargs) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(WORKDIR / "checkpoints"),
        bf16=True,
        deepspeed=str(WORKDIR / "configs" / "zero3.json"),
        **base_kwargs,
    )
```

### `<workdir>/src/_train_unsloth.py` (Unsloth single-GPU path)

```python
"""Loads the model via Unsloth's FastLanguageModel and configures LoRA.
Use with a single GPU only — Unsloth does not support multi-GPU.
"""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def load_model_unsloth(model_id: str, max_seq_length: int = 2048, load_in_4bit: bool = True):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,  # auto bf16/fp16 per device
    )
    # Caller wires LoRA via FastLanguageModel.get_peft_model(...) before training.
    return model, tokenizer
```

## Hard constraints

- NEVER combine Unsloth with FSDP2 or DeepSpeed. Unsloth's optimizations assume single-GPU; combining will silently degrade or crash.
- NEVER use `FULL_STATE_DICT` with FSDP2 if the full model exceeds rank-0 GPU memory. It will OOM.
- NEVER skip the memory estimate. Picking distributed strategy without knowing the requirement leads to either OOM (under-provisioned) or wasted complexity (over-provisioned).
- NEVER mix bf16 and fp16 in the same config. Pick one based on hardware (A100/H100 → bf16; T4/V100 → fp16). The accelerate config and the model dtype must match.
- NEVER use DeepSpeed ZeRO-3 without testing the resume path. ZeRO-3 sharded checkpoints have version compatibility quirks; always do a resume-from-checkpoint smoke test (coordinate with `dl-checkpoint`).

## Research hooks

Distributed-training APIs evolve frequently. Before generating configs for an unfamiliar combination, invoke `ml-engineer-research`:

- **FSDP2 vs FSDP1 status.** Query: *"Current default and recommended FSDP version for HF Accelerate as of {today}, including any breaking config changes from FSDP1."*
- **DeepSpeed ZeRO-3 stability.** Query: *"Known regressions or required pinned versions for DeepSpeed ZeRO-3 + HF Trainer + bf16 as of {today}."*
- **Unsloth multi-GPU support.** Query: *"Has Unsloth added multi-GPU support as of {today}? If so, recommended launcher and config."*
- **Memory estimates.** See Step 1 research hook above.
- **`fsdp_config` key naming.** Query: *"Current expected key naming for `TrainingArguments(fsdp_config=...)` in HF Transformers as of {today} — un-prefixed (`state_dict_type`) vs. prefixed (`fsdp_state_dict_type`)?"* Apply when generating `_train_fsdp.py` or amending the FSDP yaml config.

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- A config file was generated in `<workdir>/configs/` matching the chosen strategy (`accelerate_fsdp.yaml` for FSDP2, `zero3.json` for DeepSpeed, no config for single-GPU/Unsloth).
- The launcher command runs without error for at least one step on the actual training data shape.
- For FSDP2: `accelerate test --config_file <workdir>/configs/accelerate_fsdp.yaml` returns success on the chosen config.
- The training script's gradient accumulation × micro-batch × num_gpus equals the intended global batch size.

## Output checklist

- [ ] Memory requirement estimated (research hook used if model is unfamiliar)
- [ ] Strategy picked per decision rules
- [ ] Config file generated in `<workdir>/configs/`
- [ ] Launcher command stated
- [ ] Training script modified to match the chosen strategy
- [ ] Smoke test: at least one step runs successfully
- [ ] Unsloth/FSDP2/DeepSpeed are mutually exclusive — only one applies
