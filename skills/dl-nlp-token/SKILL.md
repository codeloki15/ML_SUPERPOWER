---
name: dl-nlp-token
description: Use to finetune a token-classification model — NER (named entity recognition), POS tagging, or extractive question answering. Different loss / decoding from sequence classification. Mandatory BIO alignment verification before training. Do NOT use for sequence classification (use dl-nlp-classify), generative tasks (use llm-engineer), or non-text data.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# NLP Token

Finetune a token-classifier (NER / POS / extractive QA). The hard part is BIO alignment between word-level labels and subword-tokenized inputs — this skill enforces verification before training.

## When to invoke

- User wants to train NER / POS / extractive QA.
- After `dl-load-data` has tokenized the text.

## When NOT to invoke

- Sequence classification (use `dl-nlp-classify`).
- Generative tasks (use `llm-engineer`).
- Inference / evaluation only (use `dl-nlp-eval-token`).

## Decision rules

### Backbone

1. Invoke `dl-prior-art`.
2. Default suggestions (apply if no strong prior):
   - **English NER**: `answerdotai/ModernBERT-base` or `microsoft/deberta-v3-base`.
   - **Multilingual NER**: `xlm-roberta-base` (still SOTA for cross-lingual).
   - **Domain-specific**: `dmis-lab/biobert-v1.1` (biomed), `nlpaueb/legal-bert-base-uncased` (legal), `allenai/scibert_scivocab_uncased` (science).
3. Surface + ask user.

### Tag scheme

- BIO (Begin / Inside / Outside) — most common.
- BIOES (Begin / Inside / Outside / End / Single) — slightly better F1 on some tasks.
- IO — rare, only for tasks where entity boundaries don't matter.
- Read or infer from `<workdir>/data_policy.json` annotations; flag if mixed schemes detected.

### Loss

- Token-classification head + CrossEntropy on per-token labels.
- HF default; do NOT override.
- For span-level extractive QA: `AutoModelForQuestionAnswering` (different head; this skill handles both).

### Subword alignment

- The MOST common silent NER bug: tokenizer splits "WashingtonD.C." into ["Wash", "##ington", "D", ".", "C", "."] but labels are at the word level.
- HF convention: assign label only to the FIRST subword of each word; assign `-100` (ignore index) to subsequent subwords.
- This skill mandates verification before training.

## Process

### Step 1 — Consult prior art

Invoke `dl-prior-art`. Note recommendation.

### Step 2 — Pick backbone (user confirms)

Surface menu. Wait for confirmation.

### Step 3 — BIO alignment verification (MANDATORY)

For 5 sample sentences from training data:
- Tokenize and align labels using the chosen scheme.
- Print: `(token, label)` pairs side-by-side.
- Verify: every entity span in the original is preserved AND every subword inherits the correct alignment (-100 or first-subword-only).

If alignment is wrong, halt; fix the alignment function before training. Misaligned NER training produces silently bad models.

### Step 4 — Build the model

Use `AutoModelForTokenClassification.from_pretrained(model_id, num_labels=N, id2label=..., label2id=...)`. For extractive QA, use `AutoModelForQuestionAnswering`.

### Step 5 — Hand off to `dl-finetune-loop`

Trainer path. Use `DataCollatorForTokenClassification` (handles dynamic padding + label padding to -100).

Pass `compute_metrics` that decodes BIO predictions back to spans and computes span-F1 via seqeval.

### Step 6 — Verify

Hand off to `dl-nlp-eval-token` for full eval.

## Recipe template

### `<workdir>/src/_model_nlp_token.py`

```python
"""Build the NER / token-classification model. Adapt backbone per dl-prior-art."""
import json
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def build_model(model_id: str = "answerdotai/ModernBERT-base", num_labels: int | None = None,
                id2label: dict | None = None, task: str = "ner"):
    """task in {ner, pos, qa}."""
    if task == "qa":
        from transformers import AutoModelForQuestionAnswering
        return AutoModelForQuestionAnswering.from_pretrained(model_id)

    from transformers import AutoModelForTokenClassification
    label2id = {v: k for k, v in (id2label or {}).items()} if id2label else None
    return AutoModelForTokenClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


def align_labels_to_subwords(words: list[str], word_labels: list[int],
                              tokenizer, label_all_subwords: bool = False) -> dict:
    """Tokenize words and align word-level labels to subword tokens.

    HF convention: first subword inherits the word's label; subsequent subwords get -100.
    Set label_all_subwords=True to label every subword (rare; only for some POS tasks).
    """
    tokenized = tokenizer(words, is_split_into_words=True, truncation=True)
    word_ids = tokenized.word_ids()
    aligned_labels = []
    prev_word_id = None
    for wid in word_ids:
        if wid is None:
            aligned_labels.append(-100)
        elif wid != prev_word_id:
            aligned_labels.append(word_labels[wid])
        else:
            aligned_labels.append(word_labels[wid] if label_all_subwords else -100)
        prev_word_id = wid
    tokenized["labels"] = aligned_labels
    return tokenized


def verify_alignment(words: list[list[str]], word_labels: list[list[int]],
                      tokenizer, id2label: dict, n: int = 5):
    """Print n examples with (token, label) pairs side-by-side. Manually inspect."""
    for i, (w, l) in enumerate(zip(words[:n], word_labels[:n])):
        aligned = align_labels_to_subwords(w, l, tokenizer)
        tokens = tokenizer.convert_ids_to_tokens(aligned["input_ids"])
        labels = aligned["labels"]
        print(f"--- example {i} ---")
        for t, lb in zip(tokens, labels):
            tag = id2label.get(lb, "IGNORE") if lb != -100 else "IGNORE"
            print(f"  {t:20s}  {tag}")


def compute_metrics_seqeval(eval_pred, id2label: dict):
    """Decode BIO predictions and compute span-F1 via seqeval."""
    import numpy as np
    from seqeval.metrics import f1_score, precision_score, recall_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    true_labels = [
        [id2label[l] for l, p in zip(label_seq, pred_seq) if l != -100]
        for label_seq, pred_seq in zip(labels, preds)
    ]
    true_predictions = [
        [id2label[p] for l, p in zip(label_seq, pred_seq) if l != -100]
        for label_seq, pred_seq in zip(labels, preds)
    ]
    return {
        "f1_span": f1_score(true_labels, true_predictions, zero_division=0),
        "precision_span": precision_score(true_labels, true_predictions, zero_division=0),
        "recall_span": recall_score(true_labels, true_predictions, zero_division=0),
    }
```

## Hard constraints

- NEVER skip BIO alignment verification (Step 3). Misaligned NER silently produces models that look fine on token-level metrics but score 0 on span-F1.
- NEVER use a tokenizer from a different model than the backbone. Mismatch breaks alignment and produces fluent garbage.
- NEVER label all subwords with the same word's label without thinking. HF convention is first-subword-only with -100 elsewhere; deviating requires justification (some POS tasks do label all).
- NEVER apply word-level augmentation (synonym replacement, random insertion) to NER data without explicit BIO-tag tracking. The augmented entity boundaries must still align.
- NEVER report token-level F1 as the headline for NER. Span-F1 (via seqeval) is the right metric — token F1 inflates the score and hides span-boundary errors.
- NEVER cross-validate by random sentence shuffling on document-level NER. Use document-level groupKFold to avoid leaking entity context across folds.

## Research hooks

- **Current SOTA NER backbone for `{language}`.** Invoke `dl-prior-art` first; query as fallback: *"Current top NER backbone for `{language}` `{domain}` (general / biomed / legal / scientific) as of {today}."*
- **BIO vs BIOES for `{task_type}`.** Query: *"Does BIOES outperform BIO for `{task_type}` NER on `{benchmark}` as of {today}, and is the gain worth the implementation complexity?"*
- **HF subword alignment current best practice.** Query: *"Current HF DataCollatorForTokenClassification and `align_labels_to_subwords` recipe as of {today}."*

## Verification gates

After this skill runs (and `dl-finetune-loop` has trained), `ml-engineer-verify` MUST check:

- BIO alignment verification ran (5 samples printed with (token, label) side-by-side; output saved to `<workdir>/samples/alignment_check.txt`).
- Tokenizer in `<workdir>/data_policy.json` matches the chosen `model_id`.
- For NER: span-F1 (NOT token-F1) is the headline metric.
- For NER: 5 sample predictions saved with predicted spans highlighted.
- Pre-trained backbone weights loaded successfully.
- Span-F1 is non-trivially above 0 (a random NER scores ~0; even a broken model usually scores >5%).

## Output checklist

- [ ] `dl-prior-art` consulted
- [ ] Backbone confirmed by user
- [ ] BIO alignment verified on 5 samples
- [ ] Model + DataCollatorForTokenClassification + compute_metrics_seqeval wired
- [ ] Handed off to `dl-finetune-loop`
- [ ] Span-F1 (not token-F1) used as the headline
