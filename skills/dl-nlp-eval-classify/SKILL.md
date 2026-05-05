---
name: dl-nlp-eval-classify
description: Use to evaluate a text classification model — accuracy, macro/micro F1, MCC (Matthews correlation coefficient), per-class breakdown, calibration. Pairs with dl-nlp-classify. Do NOT use for token classification / NER (use dl-nlp-eval-token), generative outputs (use dl-nlp-eval-generative), or image classification (use dl-cv-eval-classify).
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# NLP Eval — Classify

Compute the standard text-classification eval suite: accuracy, F1 (macro / micro / weighted), Matthews correlation coefficient, per-class breakdown, calibration (ECE). Save metrics + confusion matrix + per-class F1 chart. Surface insights — worst classes + calibration verdict.

## When to invoke

- After a text classifier is trained (via `dl-nlp-classify` + `dl-finetune-loop`).
- User asks "evaluate this classifier" / "what's the F1".
- Before declaring an NLP classification task complete.

## When NOT to invoke

- Token classification / NER (use `dl-nlp-eval-token`).
- Generative outputs (use `dl-nlp-eval-generative`).
- Image classification (use `dl-cv-eval-classify`).

## Decision rules

### Metric set

- Always: accuracy, macro F1, micro F1, weighted F1.
- Always: per-class precision / recall / F1 / support.
- Add MCC (Matthews correlation coefficient) for binary or imbalanced multi-class — robust to imbalance, ranges -1 to 1.
- Add Cohen's kappa for ordinal / agreement-style tasks.
- Add ECE (Expected Calibration Error, 15 bins) — a finetuned LLM-derived classifier is often miscalibrated.

### Confusion matrix

- ≤30 classes: full N×N heatmap.
- >30 classes: top-20 confusing pairs as a list.

### Multi-label

- For multi-label: subset accuracy + per-label F1 + macro / micro F1 over all labels.
- Add Hamming loss (fraction of mislabeled labels).

## Process

### Step 1 — Read predictions + labels

Run `model.predict(...)` on eval set OR read cached `<workdir>/predictions/eval_logits.npz`.

### Step 2 — Compute metrics

Save to `<workdir>/metrics.json`:

```json
{
  "accuracy": 0.XXX,
  "f1_macro": 0.XXX,
  "f1_micro": 0.XXX,
  "f1_weighted": 0.XXX,
  "mcc": 0.XXX,
  "cohen_kappa": 0.XXX,
  "ece": 0.XXX,
  "per_class": {"class_0": {"precision": ..., "recall": ..., "f1": ..., "support": ...}, ...},
  "worst_classes": [{"class": "X", "f1": 0.XX, "support": N}, ...]
}
```

### Step 3 — Save charts

- `<workdir>/charts/confusion_matrix.png`.
- `<workdir>/charts/per_class_f1.png` (sorted ascending).
- `<workdir>/charts/reliability.png`.

### Step 4 — Surface insights

Print the worst 5 classes by F1, plus calibration verdict, plus headline numbers.

### Step 5 — Save predictions

`<workdir>/predictions/eval_predictions.csv` with `text, true_label, pred_label, top1_prob`.

## Recipe template

### `<workdir>/src/_eval_nlp_classify.py`

```python
"""Text classification evaluation harness."""
import json
import os
from pathlib import Path

import numpy as np

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def compute_metrics(logits: np.ndarray, labels: np.ndarray, class_names: list[str] | None = None,
                    multi_label: bool = False) -> dict:
    """logits: (N, C). labels: (N,) for single-label or (N, C) binary for multi-label."""
    from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                                 f1_score, hamming_loss, matthews_corrcoef)
    import torch
    import torch.nn.functional as F

    metrics = {}
    if multi_label:
        preds = (logits > 0).astype(int)
        metrics["accuracy_subset"] = float(accuracy_score(labels, preds))
        metrics["f1_macro"] = float(f1_score(labels, preds, average="macro", zero_division=0))
        metrics["f1_micro"] = float(f1_score(labels, preds, average="micro", zero_division=0))
        metrics["hamming_loss"] = float(hamming_loss(labels, preds))
        return metrics

    preds = logits.argmax(axis=-1)
    probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
    metrics["accuracy"] = float(accuracy_score(labels, preds))
    metrics["f1_macro"] = float(f1_score(labels, preds, average="macro", zero_division=0))
    metrics["f1_micro"] = float(f1_score(labels, preds, average="micro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(labels, preds, average="weighted", zero_division=0))
    metrics["mcc"] = float(matthews_corrcoef(labels, preds))
    metrics["cohen_kappa"] = float(cohen_kappa_score(labels, preds))
    metrics["ece"] = compute_ece(probs, labels, n_bins=15)

    report = classification_report(labels, preds, output_dict=True, zero_division=0,
                                   target_names=class_names)
    metrics["per_class"] = {k: v for k, v in report.items() if isinstance(v, dict) and k != "accuracy"}

    sorted_classes = sorted(
        [(k, v) for k, v in metrics["per_class"].items() if "f1-score" in v],
        key=lambda x: x[1]["f1-score"],
    )
    metrics["worst_classes"] = [
        {"class": k, "f1": v["f1-score"], "support": int(v["support"])}
        for k, v in sorted_classes[:5]
    ]
    return metrics


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    accuracies = (predictions == labels).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if in_bin.any():
            bin_acc = accuracies[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            ece += in_bin.mean() * abs(bin_acc - bin_conf)
    return float(ece)


def save_charts(preds: np.ndarray, labels: np.ndarray, probs: np.ndarray,
                class_names: list[str] | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix as sk_cm

    charts_dir = WORKDIR / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    cm = sk_cm(labels, preds)
    if cm.shape[0] <= 30:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(cm, cmap="Blues")
        ax.set_title("NLP confusion matrix")
        plt.tight_layout()
        plt.savefig(charts_dir / "confusion_matrix.png", dpi=120)
        plt.close()
        print("Chart saved as confusion_matrix.png")


def save_metrics(metrics: dict):
    p = WORKDIR / "metrics.json"
    p.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {p}")
```

## Hard constraints

- NEVER apply augmentation to the eval dataloader.
- NEVER report only one F1 averaging mode on imbalanced data — both macro and weighted (or macro and micro).
- NEVER skip MCC for binary or heavily imbalanced multi-class tasks. Accuracy can be 95% on a 95/5 split with a constant predictor; MCC catches it.
- NEVER plot a >30×30 confusion matrix.
- NEVER cache predictions across model versions.
- NEVER report calibration on multi-label without per-label calibration. Single ECE on multi-label is misleading.

## Research hooks

- **Calibration metrics for NLP.** Query: *"Current calibration metrics (ECE, Adaptive ECE, Brier) for NLP classification with imbalanced classes as of {today}."*
- **Imbalanced multi-class evaluation best practice.** Query: *"Recommended evaluation suite for `{task_type}` text classification with imbalance ratio `{ratio}` as of {today}."*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- `<workdir>/metrics.json` exists with at minimum `accuracy`, `f1_macro`, `f1_micro`, `mcc`, `per_class`.
- Eval dataloader had NO augmentation.
- For ≤30 classes: confusion matrix saved.
- Predictions saved to `<workdir>/predictions/eval_predictions.csv`.
- Accuracy is non-trivially above `1/num_labels`.
- MCC is reported for binary / imbalanced tasks.

## Output checklist

- [ ] Predictions read or re-computed
- [ ] Standard metrics + MCC + ECE computed
- [ ] Per-class breakdown surfaced
- [ ] Worst 5 classes listed
- [ ] Confusion matrix + reliability + per-class F1 charts saved
- [ ] Metrics JSON + predictions CSV written
- [ ] Eval was NOT augmented
