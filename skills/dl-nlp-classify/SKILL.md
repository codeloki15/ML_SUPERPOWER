---
name: dl-nlp-classify
description: Use to finetune an encoder model for sequence classification (single-label or multi-label). ModernBERT and DeBERTa-v3 are default suggestions; user confirms via dl-prior-art recommendation. Do NOT use for token classification / NER (use dl-nlp-token), generative tasks (use llm-engineer), or non-text data.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# NLP Classify

Finetune an encoder for text classification. Pick backbone via `dl-prior-art` + user confirmation; default suggestions are ModernBERT-base or DeBERTa-v3-base. Hand off to `dl-finetune-loop` (Trainer path).

## When to invoke

- User wants to train a text classifier (single-label or multi-label).
- After `dl-load-data` has run (tokenizer + max_length locked in `<workdir>/data_policy.json`).

## When NOT to invoke

- Token classification / NER (use `dl-nlp-token`).
- Generative / instruction-tuning (use `llm-engineer`).
- Inference / evaluation only (use `dl-nlp-eval-classify`).

## Decision rules

### Backbone

1. Invoke `dl-prior-art` for the user's specific task.
2. Apply default suggestions if no strong prior:
   - **English text, ≤512 tokens, modern recipe**: `answerdotai/ModernBERT-base` (2024+, sequence-length 8192, faster than BERT-base).
   - **English text, fine-grained / hard classification**: `microsoft/deberta-v3-base` or `microsoft/deberta-v3-large`.
   - **Multilingual**: `microsoft/mdeberta-v3-base` or `xlm-roberta-base`.
   - **Long documents (>512 tokens)**: `answerdotai/ModernBERT-base` (handles 8k natively) OR Longformer / BigBird.
3. Surface choice + rationale; wait for user confirmation.

### Head

- HF AutoModelForSequenceClassification handles the head automatically. Pass `num_labels=N`.
- For multi-label: pass `problem_type="multi_label_classification"` — uses BCEWithLogitsLoss internally.

### Loss

- Single-label: HF default (CrossEntropy).
- Multi-label: HF default for `problem_type="multi_label_classification"` (BCEWithLogitsLoss).
- Imbalance: pass `class_weight` to a custom Trainer subclass, OR use focal loss via a `compute_loss` override.

### Optimizer / lr

- AdamW, lr=2e-5 (DeBERTa, ModernBERT) or 3e-5 (RoBERTa-style).
- Warmup ratio: 0.06-0.10.
- Weight decay: 0.01.
- These are standard recipes; tune only if `dl-prior-art` says winners did.

## Process

### Step 1 — Consult prior art

Invoke `dl-prior-art`. Note recommendation.

### Step 2 — Pick backbone (with user confirmation)

Surface menu + rationale. Wait for user.

### Step 3 — Pre-tokenization sanity check

Verify the tokenizer in `<workdir>/data_policy.json` matches the chosen model. If not, re-invoke `dl-load-data` with the correct `tokenizer_id`.

Print first 5 tokenized examples (input_ids decoded back) to verify tokenization is doing what you expect.

### Step 4 — Build the model

Use `AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=N, problem_type=...)`.

### Step 5 — Hand off to `dl-finetune-loop`

Trainer path (standard supervised loss). Pass `compute_metrics` callback computing accuracy + F1.

### Step 6 — Verify

Hand off to `dl-nlp-eval-classify` for the full eval suite.

## Recipe template

### `<workdir>/src/_model_nlp_classify.py`

```python
"""Build the text classification model. Adapt backbone per dl-prior-art recommendation."""
import json
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def build_model(model_id: str = "answerdotai/ModernBERT-base", num_labels: int | None = None,
                multi_label: bool = False, id2label: dict | None = None):
    from transformers import AutoModelForSequenceClassification

    policy = json.loads((WORKDIR / "data_policy.json").read_text()) if (WORKDIR / "data_policy.json").exists() else {}
    num_labels = num_labels or policy.get("num_classes")
    if num_labels is None:
        raise ValueError("num_labels not provided and num_classes not in data_policy.json")

    label2id = {v: k for k, v in (id2label or {}).items()} if id2label else None
    problem_type = "multi_label_classification" if multi_label else "single_label_classification"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        problem_type=problem_type,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model


def compute_metrics(eval_pred):
    """Pass to Trainer(compute_metrics=...). Computes accuracy + F1."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = logits.argmax(axis=-1) if logits.ndim > 1 else (logits > 0).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def verify_tokenization(tokenizer, examples: list[str], n: int = 5):
    """Sanity check: tokenize and decode-back to confirm tokenizer doesn't strip critical content."""
    for i, ex in enumerate(examples[:n]):
        ids = tokenizer.encode(ex, truncation=True, max_length=512)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"[{i}] orig: {ex[:80]!r}")
        print(f"    decoded: {decoded[:80]!r}")
        print(f"    n_tokens: {len(ids)}")
```

## Hard constraints

- NEVER use a tokenizer from a different model than the backbone. Mismatch silently produces fluent garbage.
- NEVER skip the pre-tokenization sanity check (Step 3). Print 5 decoded examples; verify nothing critical was stripped.
- NEVER use single-label CrossEntropy for multi-label tasks. HF auto-picks correctly via `problem_type`, but if you write a custom loss, do NOT mix.
- NEVER finetune with the head's lr equal to the backbone's lr on small datasets. Differential lr (head 5-10x faster than backbone) is the standard.
- NEVER train an encoder for text classification with sequence-length > model's `max_position_embeddings`. ModernBERT supports 8192; classic BERT supports 512.

## Research hooks

- **Current SOTA encoder for text classification.** Always invoke `dl-prior-art` first; fallback query: *"Current top encoder model for `{language}` text classification on `{dataset_size}` data as of {today}."*
- **Long-context encoder options.** Query: *"Current encoder options for >512 token text classification (ModernBERT vs Longformer vs BigBird vs XLM-R) as of {today}."*
- **lr / warmup recipes per backbone family.** Query: *"Current recommended lr / warmup ratio / weight_decay for `{model_family}` finetune as of {today}."*

## Verification gates

After this skill runs (and `dl-finetune-loop` has trained), `ml-engineer-verify` MUST check:

- Tokenizer in `<workdir>/data_policy.json` matches the chosen `model_id`.
- 5 decoded tokenization samples were printed and inspected (not silent).
- For multi-label: `problem_type="multi_label_classification"` is set on the model config.
- For single-label: model output shape is `(batch, num_labels)` with logits.
- Eval F1 (macro) is non-trivially above `1/num_labels` (sanity threshold).
- Pre-trained backbone weights loaded successfully (`ignore_mismatched_sizes=True` was used if needed).

## Output checklist

- [ ] `dl-prior-art` consulted
- [ ] User confirmed backbone choice
- [ ] Tokenizer matched to model
- [ ] 5 tokenization samples printed
- [ ] Model + compute_metrics built
- [ ] Differential lr applied if small data
- [ ] Handed off to `dl-finetune-loop` (Trainer path)
