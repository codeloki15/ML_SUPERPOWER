---
name: dl-cv-eval-segment
description: Use to evaluate a segmentation model (semantic / instance / panoptic) — mean IoU, Dice, Hausdorff distance, boundary F1, per-class breakdown. Pairs with dl-cv-segment. Do NOT use for classification (use dl-cv-eval-classify) or detection (use dl-cv-eval-detect).
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# CV Eval — Segment

Compute segmentation metrics: mean IoU (Jaccard), Dice (F1 on pixels), Hausdorff distance (boundary error in pixels), boundary F1, per-class breakdown. Visualize predicted vs GT masks side-by-side.

## When to invoke

- After a segmenter is trained (via `dl-cv-segment` + `dl-finetune-loop` or via SAM zero-shot).
- User asks "evaluate this segmenter" or "what's the IoU / Dice".
- Before declaring a segmentation task complete.

## When NOT to invoke

- Classification (use `dl-cv-eval-classify`).
- Detection without masks (use `dl-cv-eval-detect`).
- Instance / panoptic-only eval is supported here for COCO-style mask AP — call out the mode in the call.

## Decision rules

### Metric set

- **Always**: mean IoU (mIoU) and per-class IoU. The standard semantic-segmentation headline.
- **Always**: Dice (mean and per-class). Dice = 2·IoU/(IoU+1); reported in medical literature.
- **Add Hausdorff distance** if the task is medical / pixel-precise (per-instance boundary error).
- **Add boundary F1** if boundary precision matters (cell segmentation, lane detection, etc.).
- **For instance segmentation**: also compute mask AP via pycocotools (`iouType="segm"`).

### Background class

- Semantic seg often has a "background" class at index 0. By default, exclude it from `mean_iou` (Pascal VOC convention). Surface both with-and-without-background numbers.

### Eval library

- `torchmetrics.JaccardIndex` for IoU.
- `torchmetrics.Dice` for Dice.
- `medpy.metric.binary.hd95` for 95th-percentile Hausdorff (more robust than max Hausdorff).
- For instance: `pycocotools.cocoeval.COCOeval(iouType="segm")`.

## Process

### Step 1 — Read predictions + GT masks

Predictions: per-pixel class labels (semantic) OR per-instance binary masks (instance/panoptic).
GT: same format.

### Step 2 — Compute metrics

Save to `<workdir>/metrics.json`:

```json
{
  "mIoU": 0.XXX,
  "mIoU_no_bg": 0.XXX,
  "per_class_IoU": {"class_X": 0.XXX, ...},
  "mDice": 0.XXX,
  "per_class_Dice": {"class_X": 0.XXX, ...},
  "mean_HD95_pixels": 0.XXX,
  "boundary_F1": 0.XXX,
  "worst_classes": [{"class": "X", "IoU": 0.XXX, "support": N}, ...]
}
```

For instance segmentation, also include:
```json
{
  "mask_mAP_50_95": 0.XXX,
  "mask_mAP_50": 0.XXX,
  "mask_mAP_75": 0.XXX,
  "per_class_mask_AP": {...}
}
```

### Step 3 — Save visualizations

For 10 sample images: image | GT mask | predicted mask | overlay (GT-pred) side-by-side. Save to `<workdir>/predictions/sample_<id>.png`. Color-code class IDs consistently.

### Step 4 — Surface insights

Print mIoU + Dice headline. Surface 3 worst classes (often: small classes with little training data). Surface boundary verdict if Hausdorff > 5 pixels (the model's masks are blob-shaped, missing fine boundaries).

## Recipe template

### `<workdir>/src/_eval_cv_segment.py`

```python
"""Segmentation evaluation harness."""
import json
import os
from pathlib import Path

import numpy as np

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def compute_iou_dice(preds: np.ndarray, gt: np.ndarray, num_classes: int,
                     ignore_index: int | None = None) -> dict:
    """preds, gt: (N, H, W) integer class labels."""
    per_class_iou = []
    per_class_dice = []
    for c in range(num_classes):
        if c == ignore_index:
            per_class_iou.append(float("nan"))
            per_class_dice.append(float("nan"))
            continue
        p = preds == c
        g = gt == c
        intersection = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        iou = intersection / union if union > 0 else float("nan")
        dice = 2 * intersection / (p.sum() + g.sum()) if (p.sum() + g.sum()) > 0 else float("nan")
        per_class_iou.append(float(iou))
        per_class_dice.append(float(dice))

    valid_iou = [v for v in per_class_iou if not np.isnan(v)]
    valid_dice = [v for v in per_class_dice if not np.isnan(v)]
    return {
        "mIoU": float(np.mean(valid_iou)) if valid_iou else 0.0,
        "per_class_IoU": {f"class_{i}": v for i, v in enumerate(per_class_iou)},
        "mDice": float(np.mean(valid_dice)) if valid_dice else 0.0,
        "per_class_Dice": {f"class_{i}": v for i, v in enumerate(per_class_dice)},
    }


def compute_hd95(preds: np.ndarray, gt: np.ndarray, num_classes: int,
                 ignore_index: int | None = None) -> float:
    """Mean HD95 across non-empty class masks."""
    try:
        from medpy.metric.binary import hd95
    except ImportError:
        return float("nan")
    distances = []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        for i in range(len(preds)):
            p = preds[i] == c
            g = gt[i] == c
            if p.any() and g.any():
                try:
                    d = hd95(p, g)
                    distances.append(d)
                except Exception:
                    pass  # empty mask edge case
    return float(np.mean(distances)) if distances else float("nan")


def compute_boundary_f1(preds: np.ndarray, gt: np.ndarray, num_classes: int,
                        tolerance_px: int = 2) -> float:
    """Boundary F1 — F1 over contour pixels within tolerance_px.

    Per-image-per-class scoring rules:
    - Both pred and gt empty: skip (no contribution to mean).
    - Exactly one empty: contributes 0.0 (false alarm or missed entity).
    - Both non-empty with non-zero boundary: standard precision/recall F1.
    """
    from scipy.ndimage import binary_dilation
    f1_scores = []
    for c in range(num_classes):
        for i in range(len(preds)):
            p = (preds[i] == c).astype(np.uint8)
            g = (gt[i] == c).astype(np.uint8)
            p_has, g_has = p.sum() > 0, g.sum() > 0
            if not p_has and not g_has:
                continue  # neither side has this class in this image; skip
            if not p_has or not g_has:
                f1_scores.append(0.0)  # one-sided absence = 0
                continue
            p_boundary = p ^ binary_dilation(p, iterations=1)
            g_boundary = g ^ binary_dilation(g, iterations=1)
            if p_boundary.sum() == 0 or g_boundary.sum() == 0:
                f1_scores.append(0.0)
                continue
            g_dilated = binary_dilation(g_boundary, iterations=tolerance_px)
            p_dilated = binary_dilation(p_boundary, iterations=tolerance_px)
            tp_p = np.logical_and(p_boundary, g_dilated).sum()
            tp_g = np.logical_and(g_boundary, p_dilated).sum()
            precision = tp_p / p_boundary.sum() if p_boundary.sum() > 0 else 0
            recall = tp_g / g_boundary.sum() if g_boundary.sum() > 0 else 0
            f1_scores.append(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def visualize_predictions(images: list, gt_masks: np.ndarray, pred_masks: np.ndarray, n: int = 10):
    """Render image | GT mask | predicted mask | overlay for n samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = WORKDIR / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(min(n, len(images))):
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].imshow(images[i]); axs[0].set_title("image"); axs[0].axis("off")
        axs[1].imshow(gt_masks[i], cmap="tab20"); axs[1].set_title("GT mask"); axs[1].axis("off")
        axs[2].imshow(pred_masks[i], cmap="tab20"); axs[2].set_title("pred mask"); axs[2].axis("off")
        diff = (gt_masks[i] != pred_masks[i]).astype(int)
        axs[3].imshow(diff, cmap="Reds"); axs[3].set_title("error map"); axs[3].axis("off")
        plt.tight_layout()
        plt.savefig(out / f"sample_{i}.png", dpi=100)
        plt.close()
        print(f"Chart saved as sample_{i}.png")


def save_metrics(metrics: dict):
    """Serialize, replacing NaN floats with null (json doesn't allow NaN literals)."""
    import math
    def sanitize(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, dict):
            return {k: sanitize(x) for k, x in v.items()}
        if isinstance(v, list):
            return [sanitize(x) for x in v]
        return v
    p = WORKDIR / "metrics.json"
    p.write_text(json.dumps(sanitize(metrics), indent=2))
    print(f"Metrics saved to {p}")
```

## Hard constraints

- NEVER apply augmentation to the eval dataloader.
- NEVER use bilinear interpolation when resizing predicted masks back to original resolution. Use nearest-neighbor; bilinear creates fractional class IDs.
- NEVER report mIoU as a single number on heavily imbalanced segmentation. Report mIoU AND mIoU_no_bg AND per-class IoU.
- NEVER skip per-class breakdown. The mIoU headline can hide a class at IoU=0.
- NEVER report Hausdorff alone without context — it's measured in pixels and depends on image resolution.
- NEVER cache predictions across model versions.

## Research hooks

- **Boundary metric choices.** Query: *"Current best practice for boundary segmentation metrics (boundary F1 vs HD95 vs ASSD vs Trimap-IoU) for `{data_domain}` as of {today}."*
- **Instance vs semantic eval libs.** Query: *"Status of `mmseg`, `mmdet`, `pycocotools`, `torchmetrics` for segmentation evaluation as of {today}."*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- `<workdir>/metrics.json` exists with at minimum `mIoU`, `per_class_IoU`, `mDice`.
- Eval dataloader had NO augmentation.
- Predicted masks were resized back to GT resolution using nearest-neighbor (NOT bilinear).
- 10 sample side-by-side visualizations saved to `<workdir>/predictions/`.
- mIoU is non-trivially above `1/num_classes` (sanity threshold).
- No per-class IoU exceeds 1.0 (sanity check — if one does, label space is broken).
- IF any class IoU is 0: alerted user and triaged.

## Output checklist

- [ ] Per-class IoU + Dice computed
- [ ] HD95 / boundary F1 added if task warrants
- [ ] Worst classes surfaced
- [ ] 10 sample visualizations saved (image | GT | pred | error map)
- [ ] Metrics JSON written
- [ ] Eval was NOT augmented
- [ ] Resize used nearest-neighbor for masks
