---
name: dl-vlm-finetune
description: Use to finetune a vision-language model (Qwen2-VL / Qwen2.5-VL / Pixtral / LLaVA / SmolVLM2 / Idefics) on image+text data. Substitutes for dl-llm-instruction-tune at the training step. TRL Qwen2-VL cookbook + Axolotl recipes. Do NOT use for text-only LLM (use dl-llm-instruction-tune), pure vision tasks (use dl-cv-*), or VLM evaluation only (eval is part of dl-llm-eval with vision benchmarks).
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# VLM Finetune

Finetune a vision-language model on image+text instruction data. Pick VLM family + library + image preprocessing rules. Mandatory image-preprocessing-matches-base check (the #1 silent VLM failure mode).

## When to invoke

- User has image+text instruction data (e.g., `{"image": ..., "instruction": ..., "response": ...}`).
- User says "VLM", "Qwen-VL", "Pixtral", "LLaVA", "Idefics", "vision-language finetune".

## When NOT to invoke

- Text-only LLM finetune (use `dl-llm-instruction-tune`).
- Pure vision tasks (image classification / detection / segmentation — use `dl-cv-*`).
- VLM evaluation only (use `dl-llm-eval` with vision benchmarks like MMMU, MathVista).

## Decision rules

### VLM family

1. Invoke `dl-prior-art` for the user's specific task.
2. Default suggestions:
   - **Qwen2.5-VL (general default)**: best general VLM 2024-2025; broad task coverage; multiple sizes (3B / 7B / 32B / 72B).
   - **Pixtral**: image-text reasoning; Mistral's VLM.
   - **LLaVA-1.6 / LLaVA-NeXT**: research baseline; well-documented.
   - **SmolVLM2**: small / edge inference (256M / 500M / 2.2B params).
   - **Idefics3**: HF's latest, multi-image support.

User confirms after seeing the menu.

### Library

- **TRL (default)**: HF official, has the Qwen2-VL cookbook recipe. Single-GPU fine.
- **Axolotl (multi-GPU)**: yaml-driven, supports Qwen2.5-VL / Pixtral / LLaVA / SmolVLM2 directly.
- **Unsloth**: VLM support is newer and limited; check if the user's chosen VLM is supported (research-hooked) before using.

### Image preprocessing

CRITICAL: each VLM family has different image preprocessing requirements:
- Resolution (Qwen2.5-VL: dynamic up to 8K patches; Pixtral: native res; LLaVA: 336x336 or 672x672; SmolVLM2: 384x384).
- Channel order (RGB always for these families).
- Pixel value range (most: 0-1 normalized; some: ImageNet mean/std normalized).
- Image token format (`<|vision_start|><|image_pad|><|vision_end|>` for Qwen-VL; `[IMG]` for Pixtral; `<image>` for LLaVA).

ALWAYS use the model's bundled `AutoProcessor` to preprocess. Hand-rolled preprocessing silently produces garbage.

## Process

### Step 1 — Consult prior art

Invoke `dl-prior-art`. Note recommendation.

### Step 2 — Pick VLM + library + ask user

Surface menu + rationale.

### Step 3 — Format data with the model's processor

Use `AutoProcessor.from_pretrained(model_id)` for image+text formatting. Save 5 formatted examples to `<workdir>/samples/vlm_formatted.txt` for human inspection.

### Step 4 — Build the model + LoRA

Use `dl-llm-lora` decision tree to pick LoRA / QLoRA / DoRA — same as text LLM. VLM target_modules differ from text LLM (research-hooked).

### Step 5 — Wire SFT trainer

TRL `SFTTrainer` with image columns OR Axolotl yaml.

### Step 6 — Smoke test (10 steps + 1 generation)

Loss decreasing AND a generation on a held-out image+prompt produces sensible text.

### Step 7 — Generation sanity check (MANDATORY) on 5 held-out image+prompt pairs

Save to `<workdir>/samples/post_train_vlm_generation.txt`. Inspect manually — VLMs fail more silently than text LLMs (image preprocessing mismatch produces fluent text that ignores the image).

### Step 8 — Hand off to eval

`dl-llm-eval` with vision benchmarks (MMMU, MathVista, ChartQA, etc.).

## Recipe template

### `<workdir>/src/_vlm_qwen_trl.py` (TRL Qwen2-VL cookbook adaptation)

```python
"""TRL SFTTrainer for Qwen2-VL / Qwen2.5-VL."""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def build_qwen_vl(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", load_in_4bit: bool = True):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    import torch

    bnb_config = None
    if load_in_4bit:
        # CRITICAL: skip the vision encoder ('visual') from quantization. Per Hard constraints,
        # quantizing image encoder weights silently degrades quality 5-15%. Only the LLM
        # backbone gets quantized; the visual tower stays in bf16/fp16.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual"],
        )

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, processor


def attach_vlm_lora(model, r: int = 16, target_modules: list[str] | None = None):
    """LoRA on Qwen2-VL. target_modules MUST come from research — VLM target_modules
    differ from text-only LLM and from each other across VLM families. Do NOT default
    to text-LLM target_modules; that often touches the vision projection layers
    incorrectly.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if target_modules is None:
        raise ValueError(
            "target_modules is required for VLM LoRA. Invoke ml-engineer-research for the "
            "current recommendation for this VLM family — text-LLM defaults will NOT work."
        )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=r, lora_alpha=2 * r, lora_dropout=0.05,
        target_modules=target_modules, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def format_vlm_example(processor, image, instruction: str, response: str | None = None):
    """Format a single (image, instruction[, response]) into Qwen2-VL chat template."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": instruction},
    ]}]
    if response is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=(response is None))
    return text
```

### `<workdir>/configs/axolotl_vlm.yaml` (Axolotl multi-GPU VLM)

```yaml
base_model: Qwen/Qwen2.5-VL-7B-Instruct
model_type: AutoModelForVision2Seq
processor_type: AutoProcessor

load_in_4bit: true
adapter: qlora
lora_r: 16
lora_alpha: 32
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj

datasets:
  - path: <workdir>/data/vlm_train.jsonl
    type: chat_template
    chat_template: qwen2_vl

sequence_len: 4096
sample_packing: false   # VLM data is image-heavy; packing rarely helps
bf16: auto
gradient_accumulation_steps: 4
micro_batch_size: 1
num_epochs: 3
learning_rate: 2e-4
warmup_ratio: 0.03
lr_scheduler: cosine

output_dir: <workdir>/checkpoints
```

## Hard constraints

- NEVER hand-roll image preprocessing for a VLM. Always use the bundled `AutoProcessor`. Hand-rolling silently produces fluent text that ignores the image.
- NEVER mix image tokenization across families. Qwen-VL uses `<|vision_start|>...`; Pixtral uses `[IMG]`; LLaVA uses `<image>`. Wrong format = model can't see the image at all.
- NEVER assume LoRA target_modules are the same as text-only LLM. VLMs have separate vision encoders + projection layers; target_modules differ. Research before setting.
- ALWAYS run a generation sanity check on held-out image+prompt pairs after training. VLMs hide preprocessing bugs behind fluent text more often than text LLMs.
- NEVER use sample_packing=true for image-heavy VLM data unless `dl-prior-art` confirms a winner used it. Image features dominate the sequence; packing rarely helps and can corrupt image positions.
- NEVER quantize VLM image encoder weights. Quantize the LLM backbone only; image encoder is small and quantization quality drops sharply.

## Research hooks

- **Current VLM family rankings.** Query: *"Latest VLM family rankings on `{benchmark}` (MMMU / MathVista / ChartQA) as of {today}."*
- **VLM LoRA target_modules.** Query: *"Recommended LoRA `target_modules` for `{vlm_family}` (Qwen2.5-VL / Pixtral / LLaVA / SmolVLM2) — differs from text LLM, as of {today}."*
- **Axolotl VLM support status.** Query: *"Current Axolotl-supported VLM families and their yaml `model_type` settings as of {today}."*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- AutoProcessor was used for image+text formatting (NOT hand-rolled).
- 5 formatted training examples saved + inspected.
- 5 post-training generations on held-out image+prompt pairs saved.
- Image encoder weights were NOT quantized (only the LLM backbone).
- For Qwen-VL: chat template uses `<|vision_start|><|image_pad|><|vision_end|>`.
- For Pixtral: chat template uses `[IMG]`.
- For LLaVA: chat template uses `<image>`.
- LoRA print_trainable_parameters shows 0.05-5% trainable (PEFT range).

## Output checklist

- [ ] `dl-prior-art` consulted
- [ ] VLM family + library chosen with user confirm
- [ ] AutoProcessor used (no hand-rolled preprocessing)
- [ ] 5 formatted examples saved + inspected
- [ ] LoRA configured with VLM-specific target_modules
- [ ] Smoke test (10 steps + 1 image+prompt generation) clean
- [ ] Full training ran
- [ ] Post-training generation on 5 held-out image+prompt pairs
- [ ] Adapter saved
- [ ] Handed off to `dl-llm-eval` with vision benchmarks
