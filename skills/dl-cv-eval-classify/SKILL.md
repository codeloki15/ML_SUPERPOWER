---
name: dl-cv-eval-classify
description: Use to evaluate an image classification model — top-k accuracy, per-class F1, confusion matrix, calibration (ECE), prediction visualization. Pairs with dl-cv-classify. Do NOT use for object detection (use dl-cv-eval-detect), segmentation (use dl-cv-eval-segment), or NLP classification (use dl-nlp-eval-classify).
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# CV Eval — Classify

Compute the standard image classification eval suite: top-k accuracy, per-class F1, confusion matrix, calibration (ECE). Save charts and a JSON metrics file. Surface insights (worst classes, calibration drift) — not just raw numbers.

## When to invoke

- After a classification model is trained (via `dl-cv-classify` + `dl-finetune-loop`).
- User asks "evaluate this classifier" or "what's the F1 / accuracy / confusion matrix".
- Before declaring a classification task complete.

## When NOT to invoke

- Detection (use `dl-cv-eval-detect`).
- Segmentation (use `dl-cv-eval-segment`).
- NLP classification (use `dl-nlp-eval-classify`) — different libraries.

## Decision rules

### Top-k

- Always compute top-1 accuracy.
- Add top-5 IF `num_classes >= 100` (helpful for fine-grained tasks).
- For multi-label: report subset accuracy + per-label F1 (top-k doesn't apply).

### Per-class metrics

- For balanced classes: macro-F1 (treats classes equally).
- For imbalanced: micro-F1 (weighted by frequency) AND macro-F1 (so the user sees both views).
- Per-class precision/recall/F1 for the worst 10 classes — surface for triage.

### Confusion matrix

- For ≤30 classes: full N×N matrix as a heatmap saved to `<workdir>/charts/confusion_matrix.png`.
- For >30 classes: top-20 confusing class pairs as a list (avoids unreadable plots).

### Calibration

- Compute ECE (Expected Calibration Error) with 15 bins.
- Compute reliability diagram saved to `<workdir>/charts/reliability.png`.
- IF ECE > 0.05: surface "model is overconfident; consider temperature scaling".

## Process

### Step 1 — Read predictions + labels

Read predictions from the trained model on the eval split. Either:
- Re-run `model.predict(...)` on the eval dataloader, OR
- Read cached predictions from `<workdir>/predictions/eval_logits.npz` if available.

### Step 2 — Compute metrics

Use `torchmetrics` and `sklearn.metrics`. Save to `<workdir>/metrics.json`:

```json
{
  "top_1_accuracy": 0.XXX,
  "top_5_accuracy": 0.XXX,
  "macro_f1": 0.XXX,
  "micro_f1": 0.XXX,
  "weighted_f1": 0.XXX,
  "ece": 0.XXX,
  "per_class": {"class_0": {"precision": ..., "recall": ..., "f1": ..., "support": ...}, ...},
  "worst_classes": [{"class": "X", "f1": 0.XX, "n_errors": N}, ...]
}
```

### Step 3 — Save charts

- `<workdir>/charts/confusion_matrix.png` (or top-20 pairs list).
- `<workdir>/charts/reliability.png` (calibration).
- `<workdir>/charts/per_class_f1.png` (bar chart sorted ascending — worst classes leftmost).

### Step 4 — Surface insights

Print a markdown table of the worst 5 classes by F1, plus calibration verdict (calibrated / overconfident / underconfident), plus accuracy headlines. The user reads this; the JSON is for programmatic consumption.

### Step 5 — Save predictions for downstream use

Save `<workdir>/predictions/eval_predictions.csv` with columns `image_path, true_label, pred_label, top1_prob`. Useful for downstream error analysis, ensembling.

## Recipe template

### `<workdir>/src/_eval_cv_classify.py`

```python
"""Image classification evaluation harness."""
import json
import os
from pathlib import Path

import numpy as np

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def compute_metrics(logits: np.ndarray, labels: np.ndarray, class_names: list[str] | None = None,
                    top_k: tuple[int, ...] = (1, 5)):
    """logits: (N, C). labels: (N,) integer class ids."""
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    import torch
    import torch.nn.functional as F

    logits_t = torch.from_numpy(logits)
    labels_t = torch.from_numpy(labels)
    probs = F.softmax(logits_t, dim=-1).numpy()
    preds = logits.argmax(axis=-1)

    metrics = {
        "top_1_accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }

    num_classes = logits.shape[-1]
    if 5 in top_k and num_classes >= 5:
        top5 = torch.topk(logits_t, k=5, dim=-1).indices.numpy()
        metrics["top_5_accuracy"] = float(np.mean([labels[i] in top5[i] for i in range(len(labels))]))

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
    return metrics, preds, probs


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
            bin_weight = in_bin.mean()
            ece += bin_weight * abs(bin_acc - bin_conf)
    return float(ece)


def save_charts(preds: np.ndarray, labels: np.ndarray, probs: np.ndarray,
                class_names: list[str] | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix as sk_cm

    charts_dir = WORKDIR / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    cm = sk_cm(labels, preds, labels=list(range(probs.shape[-1])))
    if cm.shape[0] <= 30:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion matrix")
        plt.tight_layout()
        plt.savefig(charts_dir / "confusion_matrix.png", dpi=120)
        plt.close()
        print("Chart saved as confusion_matrix.png")

    # Reliability diagram
    confs = probs.max(axis=-1)
    accs = (preds == labels).astype(float)
    n_bins = 15
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accs = []
    for i in range(n_bins):
        m = (confs > bin_edges[i]) & (confs <= bin_edges[i + 1])
        bin_accs.append(accs[m].mean() if m.any() else 0.0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(bin_centers, bin_accs, width=1 / n_bins, edgecolor="black")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram")
    ax.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "reliability.png", dpi=120)
    plt.close()
    print("Chart saved as reliability.png")


def save_metrics(metrics: dict):
    out = WORKDIR / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {out}")
```

## Hard constraints

- NEVER apply augmentation to the eval dataloader. Augmented eval data inflates apparent variance and contaminates the metric.
- NEVER skip the calibration check (ECE). A model with 95% accuracy and ECE=0.20 is dangerous in any decision-making downstream use.
- NEVER report only one F1 averaging mode on imbalanced data. Always report both macro and micro (or macro and weighted) so the user sees the imbalance impact.
- NEVER plot a >30×30 confusion matrix as a heatmap. The user can't read it; surface the top-20 confusing pairs instead.
- NEVER cache predictions across model versions. If the model retrained, re-predict from scratch.

## Research hooks

- **ECE alternatives.** Query: *"Current calibration metrics beyond ECE (Adaptive ECE, MCE, Brier score) and when each is preferred as of {today}."*
- **Imbalanced-data F1 best practices.** Query: *"Recommended F1 averaging mode and threshold for imbalanced multi-class image classification with imbalance ratio `{ratio}` as of {today}."*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- `<workdir>/metrics.json` exists, is valid JSON, contains at minimum `top_1_accuracy`, `macro_f1`, `micro_f1`, `ece`, `per_class`, `worst_classes`.
- Eval dataloader has NO augmentation (cross-check `<workdir>/data_policy.json` was used in eval mode).
- For ≤30 classes: `<workdir>/charts/confusion_matrix.png` exists.
- `<workdir>/charts/reliability.png` exists.
- Predictions saved to `<workdir>/predictions/eval_predictions.csv`.
- Top-1 accuracy is non-trivially above `1/num_classes` (a sanity threshold).

## Output checklist

- [ ] Predictions read or re-computed
- [ ] Metrics computed (top-k, F1 modes, ECE, per-class)
- [ ] Charts saved (confusion matrix or top-pairs list, reliability diagram, per-class F1 bar)
- [ ] Insights surfaced to user (worst classes + calibration verdict)
- [ ] `metrics.json` and `eval_predictions.csv` saved
- [ ] Eval was NOT augmented
