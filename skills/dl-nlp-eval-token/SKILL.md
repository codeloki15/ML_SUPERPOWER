---
name: dl-nlp-eval-token
description: Use to evaluate a token-classification model — span-F1 via seqeval (NER, POS), entity-level precision/recall, exact-match. Pairs with dl-nlp-token. Do NOT use for sequence classification (use dl-nlp-eval-classify), generative tasks (use dl-nlp-eval-generative), or image data.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# NLP Eval — Token

Compute span-level metrics for NER / POS / extractive QA: span-F1 (entity-level via seqeval), per-entity-type breakdown, exact-match rate. Token-level F1 is reported but NEVER as the headline — span-F1 is the right metric.

## When to invoke

- After a token-classifier is trained (via `dl-nlp-token` + `dl-finetune-loop`).
- User asks "evaluate this NER" or "what's the span-F1".
- Before declaring an NER / token-classification task complete.

## When NOT to invoke

- Sequence classification (use `dl-nlp-eval-classify`).
- Generative tasks (use `dl-nlp-eval-generative`).
- Image data.

## Decision rules

### Metric set

- **Always (NER/POS)**: span-F1 (via seqeval) — entity-level. Headline metric.
- **Always**: per-entity-type precision / recall / F1.
- **Add token-F1** as a secondary metric (always > span-F1; surfaces alignment issues).
- **Add exact-match rate** for extractive QA: fraction of predictions that match the gold span exactly.
- **For QA**: also report token-level F1 (HuggingFace SQuAD-style) which is partial-credit at the token level within the predicted vs gold span.

### Tag scheme handling

- IF source uses BIO: `seqeval.metrics` works directly.
- IF BIOES: `seqeval` handles it; pass `scheme="BIOES"` if seqeval supports it (it does in modern versions).
- IF IO (no boundary tags): only token-level F1 is meaningful; span-F1 is undefined.

### Boundary error analysis

- Surface 5 false-positive entities (predicted but not in GT).
- Surface 5 false-negative entities (in GT but not predicted).
- Surface 5 boundary mismatches (correct entity type, wrong span).

## Process

### Step 1 — Read predictions + labels

Run `model.predict(...)` on eval set. Decode logits to BIO tags using `id2label`. Strip `-100` ignore positions.

### Step 2 — Compute metrics

Save to `<workdir>/metrics.json`:

```json
{
  "span_f1_macro": 0.XXX,
  "span_precision": 0.XXX,
  "span_recall": 0.XXX,
  "token_f1": 0.XXX,
  "exact_match": 0.XXX,
  "per_entity_type": {"PER": {"precision": ..., "recall": ..., "f1": ..., "support": N}, ...},
  "worst_entity_types": [{"type": "X", "f1": 0.XX, "support": N}, ...],
  "false_positive_examples": [...],
  "false_negative_examples": [...],
  "boundary_mismatch_examples": [...]
}
```

### Step 3 — Save error analysis

Save FP/FN/boundary-mismatch examples to `<workdir>/predictions/error_analysis.txt`.
Save predicted vs gold tags side-by-side for 10 random examples to `<workdir>/predictions/sample_predictions.txt`.

### Step 4 — Surface insights

Print span-F1 headline + worst 3 entity types + counts of FP/FN/boundary errors.

## Recipe template

### `<workdir>/src/_eval_nlp_token.py`

```python
"""NER / token-classification evaluation harness."""
import json
import os
from pathlib import Path

import numpy as np

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def compute_metrics(predictions: list[list[str]], labels: list[list[str]],
                    scheme: str = "IOB2") -> dict:
    """predictions, labels: per-example list of per-token BIO tags."""
    from seqeval.metrics import (classification_report, f1_score, precision_score,
                                  recall_score, accuracy_score)

    metrics = {
        "span_f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "span_f1_micro": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "span_precision": float(precision_score(labels, predictions, average="micro", zero_division=0)),
        "span_recall": float(recall_score(labels, predictions, average="micro", zero_division=0)),
        "token_accuracy": float(accuracy_score(labels, predictions)),
    }

    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    metrics["per_entity_type"] = {k: v for k, v in report.items()
                                   if isinstance(v, dict) and k not in ("micro avg", "macro avg", "weighted avg")}
    sorted_types = sorted(
        [(k, v) for k, v in metrics["per_entity_type"].items()],
        key=lambda x: x[1].get("f1-score", 0),
    )
    metrics["worst_entity_types"] = [
        {"type": k, "f1": v["f1-score"], "support": int(v["support"])}
        for k, v in sorted_types[:3]
    ]
    return metrics


def extract_entities(tags: list[str], tokens: list[str]) -> list[tuple]:
    """Return list of (entity_type, start_idx, end_idx, text) from BIO tags."""
    entities = []
    current = None
    for i, (tag, tok) in enumerate(zip(tags, tokens)):
        if tag.startswith("B-"):
            if current:
                entities.append(current)
            current = (tag[2:], i, i + 1, tok)
        elif tag.startswith("I-") and current and tag[2:] == current[0]:
            current = (current[0], current[1], i + 1, current[3] + " " + tok)
        else:
            if current:
                entities.append(current)
            current = None
    if current:
        entities.append(current)
    return entities


def error_analysis(predictions: list[list[str]], labels: list[list[str]],
                    tokens: list[list[str]], n_examples: int = 5) -> dict:
    """Find FP, FN, and boundary-mismatch entities."""
    false_positives = []
    false_negatives = []
    boundary_mismatches = []

    for sent_idx in range(len(predictions)):
        pred_ents = set(extract_entities(predictions[sent_idx], tokens[sent_idx]))
        gold_ents = set(extract_entities(labels[sent_idx], tokens[sent_idx]))
        gold_ent_types = {e[0]: (e[1], e[2]) for e in gold_ents}
        pred_ent_types = {e[0]: (e[1], e[2]) for e in pred_ents}

        for e in pred_ents - gold_ents:
            if e[0] in gold_ent_types and gold_ent_types[e[0]] != (e[1], e[2]):
                boundary_mismatches.append((sent_idx, e, gold_ent_types[e[0]]))
            else:
                false_positives.append((sent_idx, e))
        for e in gold_ents - pred_ents:
            if e[0] not in pred_ent_types:
                false_negatives.append((sent_idx, e))

    return {
        "false_positive_examples": false_positives[:n_examples],
        "false_negative_examples": false_negatives[:n_examples],
        "boundary_mismatch_examples": boundary_mismatches[:n_examples],
    }


def save_metrics(metrics: dict):
    p = WORKDIR / "metrics.json"
    p.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"Metrics saved to {p}")


def compute_squad_em_f1(predicted_spans: list[str], gold_spans: list[str]) -> dict:
    """Exact-match + token-F1 for extractive QA, SQuAD style."""
    import re
    import string
    from collections import Counter

    def normalize(s):
        s = s.lower()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = "".join(ch for ch in s if ch not in string.punctuation)
        s = " ".join(s.split())
        return s

    em_count = 0
    f1_sum = 0.0
    for pred, gold in zip(predicted_spans, gold_spans):
        n_pred = normalize(pred)
        n_gold = normalize(gold)
        if n_pred == n_gold:
            em_count += 1
        pred_toks = n_pred.split()
        gold_toks = n_gold.split()
        common = Counter(pred_toks) & Counter(gold_toks)
        n_common = sum(common.values())
        if n_common == 0:
            continue
        precision = n_common / len(pred_toks)
        recall = n_common / len(gold_toks)
        f1_sum += 2 * precision * recall / (precision + recall)
    n = len(predicted_spans)
    return {
        "exact_match": em_count / n if n else 0.0,
        "token_f1": f1_sum / n if n else 0.0,
    }
```

## Hard constraints

- NEVER report token-F1 as the headline for NER. Span-F1 is the standard; token-F1 inflates the score.
- NEVER skip error analysis. Surface FP / FN / boundary-mismatch examples — they reveal annotation bugs and class-imbalance issues.
- NEVER apply augmentation to the eval dataloader.
- NEVER mix BIO and BIOES schemes between training and evaluation. The decoder must use the same scheme as training.
- NEVER cache predictions across model versions.
- NEVER report span-F1 on extractive QA — use SQuAD EM/F1 instead (different shape).

## Research hooks

- **seqeval scheme support.** Query: *"Current seqeval support for BIOES, BMES, IOB1, IOB2 schemes as of {today}."*
- **NER eval beyond span-F1.** Query: *"NER evaluation metrics beyond span-F1 (entity normalization F1, partial-match F1, MUC F1) and when each is appropriate as of {today}."*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- `<workdir>/metrics.json` exists with at minimum `span_f1_macro`, `span_precision`, `span_recall`, `per_entity_type`, `worst_entity_types`.
- Eval dataloader had NO augmentation.
- Error analysis ran (FP/FN/boundary examples saved to `<workdir>/predictions/error_analysis.txt`).
- 10 sample predicted-vs-gold tag side-by-sides saved.
- For NER: span-F1 was reported as the headline (NOT token-F1).
- Span-F1 is non-trivially above 0 (random NER scores ~0).

## Output checklist

- [ ] BIO/BIOES tag decoding correct
- [ ] Span-F1 headline computed via seqeval
- [ ] Per-entity-type breakdown surfaced
- [ ] Worst entity types listed
- [ ] Error analysis (FP / FN / boundary mismatches) saved
- [ ] 10 sample predictions saved side-by-side with gold
- [ ] Metrics JSON written
- [ ] Eval was NOT augmented
