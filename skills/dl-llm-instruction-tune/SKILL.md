---
name: dl-llm-instruction-tune
description: Use to run supervised fine-tuning (SFT) on an LLM after dl-llm-lora has built the model+adapter. Covers chat template selection, sequence packing, and response-only masking. Substitutes for the standard finetune loop on chat / instruction data. Do NOT use for preference tuning (use dl-llm-pref-opt), pretraining, or non-LLM tasks.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# LLM Instruction Tune

Run supervised fine-tuning on chat / instruction data. Lock the chat template against the base model. Decide on sequence packing. Decide on response-only loss masking. Wire TRL `SFTTrainer` (or Unsloth equivalent). Mandatory generation sanity check after training — loss alone is not proof.

## When to invoke

- After `dl-llm-lora` has constructed the model+adapter.
- After `dl-load-data` has loaded SFT data (chat-formatted JSONL or HF dataset with `messages`/`text` field).
- User says "instruction tune", "SFT", "chat finetune", "fine-tune on conversations".

## When NOT to invoke

- Preference tuning (use `dl-llm-pref-opt`).
- Pre-training from scratch (out of v1 scope).
- Encoder NLP classification (use `dl-nlp-classify`).
- VLM tasks (use `dl-vlm-finetune`).

## Decision rules

### Chat template

- READ from the base model's tokenizer (`tokenizer.chat_template`). If non-null, use it as-is — it's the model's training-time format.
- IF `tokenizer.chat_template` is None, the model is base (not instruct-tuned). Apply a template appropriate to the user's intent: ChatML for general chat, Alpaca for instruction-tuning, raw text for completion-style.
- NEVER swap templates between training and inference. The template used to format training data MUST match the template used at generation time.

### Packing

- ENABLED by default (Unsloth + Axolotl both default to packing) for short-format datasets (<512 tokens average) — packs multiple short examples into one sequence to maximize GPU utilization.
- DISABLED for long-format datasets (>2048 tokens average) where each example fills the context — packing would truncate.
- DISABLED if the dataset has documents that must NOT be cross-contaminated (e.g., medical case notes — packing two unrelated patients in one sequence is wrong).

### Response-only masking

- ENABLED for instruction-style data (`Instruction → Response` pairs) where the instruction can be long. Without masking, model spends capacity reproducing instructions.
- DISABLED for full-text completion (no instruction prefix).
- TRL's `SFTTrainer` supports via `DataCollatorForCompletionOnlyLM(response_template=...)`. Unsloth supports via `train_on_responses_only(...)`.

### Hyperparameters (overridable)

- `learning_rate`: 2e-4 (LoRA), 2e-5 (full finetune; not this skill's path).
- `lr_scheduler_type`: cosine (standard).
- `warmup_ratio`: 0.03-0.10.
- `num_train_epochs`: 1-3 (LLM SFT often overfits past 3 epochs on small data).
- `gradient_accumulation_steps`: pick to reach a global batch size of 32-128.

## Process

### Step 1 — Lock chat template

Read `tokenizer.chat_template`. If present, save it to `<workdir>/chat_template.txt`. If None, ask the user to pick (ChatML / Alpaca / raw / custom).

### Step 2 — Format training data

Apply the chat template to every example. Save 5 formatted examples to `<workdir>/samples/sft_formatted.txt` for human inspection. The user reads these to confirm the format is right BEFORE training.

### Step 3 — Decide packing + response-only masking

Apply decision rules. Surface choice + rationale.

### Step 4 — Build the SFT trainer

Generate code per chosen library (Unsloth from `dl-llm-lora` step or TRL `SFTTrainer`).

### Step 5 — Smoke test (10 steps)

Run 10 steps. Verify: loss decreased OR is stable, no NaN/Inf, gradient norm is bounded.

### Step 6 — Train

Run the full training. Wire `dl-experiment-track`'s tracker and `dl-checkpoint`'s save_strategy.

### Step 7 — Generation sanity check (MANDATORY)

After training, generate 5 completions on held-out prompts. Save to `<workdir>/samples/post_train_generation.txt`. The user inspects these manually — loss alone is not proof. Common failure: low loss + fluent garbage = chat template mismatch.

### Step 8 — Hand off to eval

Hand off to `dl-llm-eval` for benchmark eval.

## Recipe template

### `<workdir>/src/_sft_trl.py` (TRL SFTTrainer path)

```python
"""TRL SFTTrainer for LLM SFT. Use when not using Unsloth's wrapper."""
import json
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def make_sft_trainer(model, tokenizer, train_dataset, eval_dataset=None,
                     packing: bool = True, response_template: str | None = None,
                     **base_kwargs):
    """response_template: e.g., '\\n### Response:\\n' for Alpaca; if None, no masking."""
    from trl import SFTConfig, SFTTrainer
    from trl import DataCollatorForCompletionOnlyLM

    args = SFTConfig(
        output_dir=str(WORKDIR / "checkpoints"),
        num_train_epochs=base_kwargs.pop("num_train_epochs", 3),
        per_device_train_batch_size=base_kwargs.pop("per_device_train_batch_size", 2),
        gradient_accumulation_steps=base_kwargs.pop("gradient_accumulation_steps", 8),
        learning_rate=base_kwargs.pop("learning_rate", 2e-4),
        lr_scheduler_type=base_kwargs.pop("lr_scheduler_type", "cosine"),
        warmup_ratio=base_kwargs.pop("warmup_ratio", 0.03),
        bf16=base_kwargs.pop("bf16", True),
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        report_to=base_kwargs.pop("report_to", ["wandb"]),
        run_name=base_kwargs.pop("run_name", WORKDIR.name),
        max_seq_length=base_kwargs.pop("max_seq_length", 2048),
        packing=packing,
        **base_kwargs,
    )

    data_collator = None
    if response_template:
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template, tokenizer=tokenizer
        )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer
```

### `<workdir>/src/_sft_unsloth.py` (Unsloth path)

```python
"""Unsloth SFTTrainer wrapper. Use after dl-llm-lora built model with Unsloth."""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def make_unsloth_sft_trainer(model, tokenizer, train_dataset, response_template: str | None = None,
                              **base_kwargs):
    from unsloth import is_bfloat16_supported
    from trl import SFTConfig, SFTTrainer

    args = SFTConfig(
        output_dir=str(WORKDIR / "checkpoints"),
        num_train_epochs=base_kwargs.pop("num_train_epochs", 3),
        per_device_train_batch_size=base_kwargs.pop("per_device_train_batch_size", 2),
        gradient_accumulation_steps=base_kwargs.pop("gradient_accumulation_steps", 8),
        learning_rate=base_kwargs.pop("learning_rate", 2e-4),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        max_seq_length=base_kwargs.pop("max_seq_length", 2048),
        packing=base_kwargs.pop("packing", True),
        report_to=base_kwargs.pop("report_to", ["wandb"]),
        run_name=base_kwargs.pop("run_name", WORKDIR.name),
        **base_kwargs,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Unsloth's response-only masking helper:
    if response_template:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part=response_template.split("Response")[0] if "Response" in response_template else response_template,
            response_part=response_template,
        )
    return trainer


def generation_sanity_check(model, tokenizer, prompts: list[str], max_new_tokens: int = 128):
    """MANDATORY post-training check: generate 5 completions on held-out prompts."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    out_path = WORKDIR / "samples" / "post_train_generation.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            decoded = tokenizer.decode(output[0], skip_special_tokens=False)
            f.write(f"--- prompt {i} ---\n{prompt}\n--- generation ---\n{decoded}\n\n")
    print(f"Generation samples saved to {out_path}")
```

## Hard constraints

- NEVER train without verifying 5 formatted examples manually (Step 2). A wrong chat template is the #1 silent SFT failure mode — model trains to low loss producing garbage at inference.
- NEVER skip the post-training generation sanity check (Step 7). Loss curves can look perfect while the model produces fluent nonsense.
- NEVER mix chat templates between training and inference. If you trained with ChatML, you generate with ChatML.
- NEVER pack sequences across documents that must remain isolated (e.g., medical, legal). Packing assumes unrelated examples are safe to concatenate.
- NEVER skip response-only masking on long-instruction data. Without it, the model wastes capacity learning to reproduce instructions.
- NEVER train past 3 epochs on small SFT data without justification. SFT overfits faster than pretraining.

## Research hooks

- **Current TRL `SFTTrainer` API surface.** Query: *"Has TRL `SFTTrainer` deprecated `dataset_text_field` / `formatting_func` / `tokenizer` arg as of {today}? Current `SFTConfig` defaults."*
- **Chat template registry per model family.** Query: *"Current `tokenizer.chat_template` defaults for {model_family} (Llama 3 / Qwen 2.5 / Mistral / Gemma 2) as of {today}, including any breaking changes from base to instruct variants."*
- **Response-only masking helpers.** Query: *"Current TRL `DataCollatorForCompletionOnlyLM` and Unsloth `train_on_responses_only` API as of {today}."*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- `<workdir>/chat_template.txt` exists and is non-empty.
- `<workdir>/samples/sft_formatted.txt` contains 5 formatted training examples.
- `<workdir>/samples/post_train_generation.txt` contains 5 generations on held-out prompts (post-training mandatory check).
- Trainer's `report_to` is set per `dl-experiment-track` (or `[]` with explicit no-tracking banner).
- Save strategy + save_steps wired per `dl-checkpoint`.
- 10-step smoke test loss < initial loss OR stable (NOT NaN/Inf).
- Adapter (NOT merged base+adapter) saved at end via `model.save_pretrained(...)`.

## Output checklist

- [ ] Chat template locked (read from tokenizer or user-picked)
- [ ] 5 formatted examples saved + inspected
- [ ] Packing decision per data shape
- [ ] Response-only masking decision per data shape
- [ ] Smoke test (10 steps) clean
- [ ] Tracker + checkpoint wired
- [ ] Full training ran
- [ ] Post-training generation sanity check on 5 held-out prompts (MANDATORY)
- [ ] Adapter saved (not merged base)
