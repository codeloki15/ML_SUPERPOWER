---
name: dl-llm-lora
description: Use to set up parameter-efficient finetuning (LoRA / QLoRA / DoRA) for an LLM. Picks library — Unsloth single-GPU default (Kaggle-validated, 2-5x faster, 80% less memory) or Axolotl multi-GPU. User can override at any point. Do NOT use for full finetune (rare; user must explicitly request), encoder NLP (use dl-nlp-classify), or non-LLM tasks.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# LLM LoRA

Pick the parameter-efficient finetuning method (LoRA / QLoRA / DoRA / full) and the library (Unsloth / Axolotl / TRL) per task and hardware. Generates the model+adapter construction code; hands off to `dl-llm-instruction-tune` (or `dl-llm-pref-opt`) for the training loop.

## When to invoke

- User wants to finetune any LLM >1B params (PEFT is almost always the right choice).
- User explicitly asks for LoRA / QLoRA / DoRA / Unsloth / Axolotl / TRL.
- After `dl-detect-env` (env.json must exist) and after `dl-load-data` for SFT or preference data.

## When NOT to invoke

- User wants a FULL finetune (rare; only when adapter merge fidelity is unacceptable). Hand off to `dl-finetune-loop` directly.
- Encoder NLP classification (use `dl-nlp-classify`).
- Vision-language model finetune (use `dl-vlm-finetune`).

## Decision rules

### Method (LoRA vs QLoRA vs DoRA vs full)

- **QLoRA (default for memory-constrained)**: model loaded in 4-bit (bitsandbytes); LoRA adapters in fp16/bf16. Fits 7B in ~10GB, 13B in ~16GB, 70B in ~46GB. Use when local GPU is <40GB OR user has not asked for max quality.
- **LoRA (16-bit base, no quantization)**: when VRAM is plentiful and quality matters more than memory. Modestly higher quality than QLoRA per Kaggle 2024-2025 winners.
- **DoRA**: incremental winner per spec (NVIDIA 2024+). LoRA but with magnitude+direction decomposition. ~3-5% better than LoRA on most tasks; same memory footprint. Use when prior-art surfaces it as a winner OR user explicitly requests.
- **Full finetune**: NOT this skill's path — hand off to `dl-finetune-loop` directly. PEFT loses ~1-2% quality vs full but saves 10-100x compute; full only when those points matter.

### Library (Unsloth vs Axolotl vs TRL)

- **Unsloth (default for single-GPU)**: 2-5x faster training, 80% less memory than HF baseline (per spec; Kaggle-validated 3+ wins 2024-2025). NO multi-GPU support — single-GPU only. Wraps HF + flash-attention + custom Triton kernels.
- **Axolotl (default for multi-GPU)**: yaml-config-driven; multi-GPU + multi-node support; broader model coverage; supports VLM (Qwen-VL, LLaVA, etc.).
- **TRL (when user explicitly names it)**: HF's own PEFT layer; thinnest abstraction; use when integrating with custom Trainer subclasses.

Decision flow:
1. IF env.json `active.kind == "local"` AND single GPU detected → Unsloth.
2. IF env.json shows multi-GPU OR remote with multi-GPU → Axolotl.
3. IF user named "TRL" or "trl" or "PEFT-only" explicitly → TRL.
4. IF user named "Unsloth" but env is multi-GPU → SURFACE the contradiction; ask whether to drop to single-GPU on one device OR switch to Axolotl. Do NOT silently combine Unsloth + multi-GPU.

### `target_modules` (research-hooked)

target_modules vary per model family (Llama: q/k/v/o + gate/up/down; Mistral: similar; Qwen: same; Gemma: differs). The spec calls these "facts that go stale" — do NOT bake values; consult `ml-engineer-research` for current recommendations.

Common defaults (overridable):
- Llama / Qwen / Mistral 1B-70B: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.
- Gemma family: same plus `embed_tokens`, `lm_head` (debated; check research).
- Phi family: differs significantly; always research.

### LoRA hyperparameters (overridable)

- `r` (rank): default 16 for QLoRA, 32 for LoRA without quantization. NEVER set r > 64 without justification (overfit risk).
- `lora_alpha`: default `2 * r` (standard convention).
- `lora_dropout`: 0.0-0.1; 0.05 is a safe default.
- `bias`: "none" (rarely toggled).

## Process

### Step 1 — Consult prior art

Invoke `dl-prior-art` for the user's specific task (e.g., "Kaggle LLM Science Exam 2024 LoRA recipe"). Note recommended method, library, target_modules, hyperparameters.

### Step 2 — Read env.json

Determine: single-GPU vs multi-GPU. VRAM available. Active env.

### Step 3 — Apply decision rules

Pick method (LoRA/QLoRA/DoRA) + library (Unsloth/Axolotl/TRL). Surface choice with one-sentence rationale.

### Step 4 — Confirm with user

Show menu:

> Suggested setup:
> - Method: QLoRA (4-bit base, fits 7B in ~10GB on your GPU).
> - Library: Unsloth (single-GPU detected, 2-5x speedup).
> - Rank: 16. target_modules: Llama-style (q/k/v/o + gate/up/down).
>
> Use this, override the library, or specify different hyperparameters?

Wait for confirmation.

### Step 5 — Build the model + adapter

Generate code per chosen library. Print `model.print_trainable_parameters()` to verify only adapter params are trainable.

### Step 6 — Hand off to training step

For SFT: hand off to `dl-llm-instruction-tune` (Phase 3, this Phase).
For preference tuning: hand off to `dl-llm-pref-opt`.

## Recipe template

### `<workdir>/src/_lora_unsloth.py` (single-GPU default)

```python
"""Unsloth single-GPU LoRA setup. Default for Phase 3 LLM finetune."""
import json
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def build_unsloth_lora(model_id: str, r: int = 16, max_seq_length: int = 2048,
                       load_in_4bit: bool = True, target_modules: list[str] | None = None):
    """Returns (model, tokenizer) with LoRA adapters attached, ready for SFT/DPO/etc."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,  # Unsloth picks bf16/fp16 per device
    )
    target_modules = target_modules or [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=2 * r,
        lora_dropout=0.0,  # Unsloth requires 0.0 for kernel optimizations
        bias="none",
        target_modules=target_modules,
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized variant
        random_state=42,
    )
    model.print_trainable_parameters()
    return model, tokenizer
```

### `<workdir>/configs/axolotl_qlora.yaml` (multi-GPU)

```yaml
# Axolotl QLoRA config. Multi-GPU via accelerate launch.
base_model: <model_id>
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

load_in_4bit: true
adapter: qlora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

datasets:
  - path: <workdir>/data/train.jsonl
    type: alpaca

sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true

bf16: auto
gradient_accumulation_steps: 4
micro_batch_size: 2

num_epochs: 3
optimizer: adamw_bnb_8bit
learning_rate: 2e-4
warmup_ratio: 0.03
lr_scheduler: cosine

output_dir: <workdir>/checkpoints
```

Launch: `accelerate launch -m axolotl.cli.train <workdir>/configs/axolotl_qlora.yaml`.

### `<workdir>/src/_lora_trl.py` (TRL fallback)

```python
"""TRL PEFT setup. Use when neither Unsloth nor Axolotl fits — e.g., custom Trainer subclass."""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def build_trl_lora(model_id: str, r: int = 16, target_modules: list[str] | None = None,
                   load_in_4bit: bool = True):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = target_modules or [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora_config = LoraConfig(
        r=r,
        lora_alpha=2 * r,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer
```

## Hard constraints

- NEVER mix Unsloth with FSDP / DeepSpeed / multi-GPU launchers. Unsloth's Triton kernels assume single-device; combining produces silent corruption or crashes. (`dl-distributed` enforces a complementary check.)
- NEVER use `bitsandbytes` 4-bit quantization for SERVING. It's training-only; for serving use AWQ / GPTQ / GGUF (`dl-llm-quantize`).
- NEVER set LoRA rank `r > 64` without justification. Higher rank rarely helps and increases overfit risk on small data.
- NEVER use a tokenizer from a different model than the base. Mismatch silently produces fluent garbage at low loss.
- NEVER skip `model.print_trainable_parameters()`. The number should be ~0.1-1% of total — if it's 0% the adapter wasn't attached; if it's >50% something is wrong (bias=all, target_modules too broad, etc.).
- NEVER bake `target_modules` for unfamiliar model families without research. Get the current correct list from `ml-engineer-research`.

## Research hooks

PEFT API and target_modules conventions evolve.

- **Current `target_modules` recommendation per family.** Query: *"Current recommended LoRA `target_modules` for `{model_family}` (Llama 3 / Qwen 2.5 / Mistral / Gemma 2 / Phi 4) as of {today}."*
- **DoRA effectiveness updates.** Query: *"Does DoRA still meaningfully outperform LoRA on `{task_type}` finetune as of {today}? Memory and speed cost of DoRA vs LoRA."*
- **Unsloth model-support list.** Query: *"Current list of model families Unsloth supports (Llama / Qwen / Mistral / Gemma / Phi / etc.) and any unsupported architectures as of {today}."*
- **QLoRA vs LoRA quality gap.** Query: *"Latest measurement of QLoRA-4bit vs LoRA-16bit quality gap on `{benchmark}` as of {today}."*

## Verification gates

After this skill runs (and `dl-llm-instruction-tune` or `dl-llm-pref-opt` has trained), `ml-engineer-verify` MUST check:

- `model.print_trainable_parameters()` was called and the trainable fraction is between 0.05% and 5% (typical PEFT range).
- For Unsloth path: `env.json` shows single GPU, AND `dl-distributed` is NOT active.
- For Axolotl path: yaml config saved to `<workdir>/configs/axolotl_*.yaml`.
- Adapter saved with `model.save_pretrained(...)` (NOT the merged base+adapter — adapter only).
- Adapter file size is in MB range (not GB) — confirms adapter-only save.
- Tokenizer matches base model (`tokenizer.name_or_path == model_id`).
- A 1-step smoke test ran (loss is non-NaN, decreased or stable).

## Output checklist

- [ ] `dl-prior-art` consulted for method/library recommendation
- [ ] env.json read; single-GPU vs multi-GPU determined
- [ ] Method (LoRA/QLoRA/DoRA) chosen with rationale
- [ ] Library (Unsloth/Axolotl/TRL) chosen per env + user confirmation
- [ ] target_modules sourced from research, not baked
- [ ] Model + adapter constructed; `print_trainable_parameters` shows 0.05-5%
- [ ] Handed off to `dl-llm-instruction-tune` or `dl-llm-pref-opt`
- [ ] Unsloth path NOT combined with FSDP/DeepSpeed
