---
name: dl-cv-eval-detect
description: Use to evaluate an object detection model — mAP@[.5:.95], mAP-50, mAP-75, per-class AP via pycocotools (COCO-style evaluation). Pairs with dl-cv-detect. Do NOT use for classification (use dl-cv-eval-classify) or segmentation (use dl-cv-eval-segment).
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# CV Eval — Detect

Compute COCO-style detection metrics: mAP@[.5:.95] (primary), mAP-50, mAP-75, per-class AP, mAR. Save predictions in COCO JSON format for re-evaluation. Visualize predicted boxes on sample images.

## When to invoke

- After a detector is trained (via `dl-cv-detect` + `dl-finetune-loop` or via the YOLO Ultralytics API).
- User asks "evaluate this detector" or "what's the mAP".
- Before declaring a detection task complete.

## When NOT to invoke

- Classification (use `dl-cv-eval-classify`).
- Segmentation (use `dl-cv-eval-segment`).
- For real-time inference benchmarking (separate concern; not in Phase 2 scope).

## Decision rules

### Eval library

- Default: `pycocotools` (`COCO`, `COCOeval`). Standard for COCO-style mAP.
- Alternative: `torchmetrics.detection.MeanAveragePrecision` (no pycocotools install needed; matches COCO numbers within rounding).
- For YOLO models: use ultralytics' built-in `model.val(...)` which produces COCO-style metrics directly.

### Annotation format

- Predictions MUST be in COCO format: list of dicts with `image_id`, `category_id`, `bbox` (xywh, absolute pixel coords), `score`.
- Ground truth MUST be in COCO format: a JSON file with `images`, `annotations`, `categories`.
- IF source data is YOLO-format: convert via the same conversion logic from `dl-cv-detect`.

### IoU thresholds

- Standard: report mAP@[.5:.95] (averaged over IoU 0.5 to 0.95 in 0.05 steps), mAP-50, mAP-75.
- For high-IoU-sensitive tasks (medical, satellite): also report mAP-90.
- For low-IoU-tolerance tasks (small objects in dense scenes): mAP-50 is often the headline.

## Process

### Step 1 — Convert predictions to COCO JSON

If model output is in YOLO format or HF transformers format, convert to COCO predictions JSON. Save to `<workdir>/predictions/eval_predictions_coco.json`.

### Step 2 — Load ground truth in COCO format

Load `<workdir>/data/annotations/eval_gt.json` (or wherever `dl-load-data` placed it). If GT is in another format, convert first.

### Step 3 — Run COCO eval

Use `pycocotools.cocoeval.COCOeval` (or `torchmetrics.detection.MeanAveragePrecision`). Compute all standard mAP slices. Save to `<workdir>/metrics.json`:

```json
{
  "mAP_50_95": 0.XXX,
  "mAP_50": 0.XXX,
  "mAP_75": 0.XXX,
  "mAP_small": 0.XXX,
  "mAP_medium": 0.XXX,
  "mAP_large": 0.XXX,
  "mAR_max1": 0.XXX,
  "mAR_max10": 0.XXX,
  "mAR_max100": 0.XXX,
  "per_class_AP": {"class_X": 0.XXX, ...},
  "worst_classes": [{"class": "X", "AP": 0.XXX, "n_gt": N}, ...]
}
```

### Step 4 — Save visualization

For 10 sample images: render ground-truth boxes (green) and predicted boxes (red) with IoU annotation. Save to `<workdir>/predictions/sample_<id>.png`.

### Step 5 — Surface insights

Print mAP@[.5:.95] headline + the 3 worst classes by AP. Note any classes with AP=0 (model never detects them — usually a label-mapping bug, not a model issue).

## Recipe template

### `<workdir>/src/_eval_cv_detect.py`

```python
"""COCO-style detection evaluation."""
import json
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def coco_eval(gt_json: str, pred_json: str) -> dict:
    """Run pycocotools eval. Returns the standard 12-metric COCO summary as a dict."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(gt_json)
    dt = gt.loadRes(pred_json)
    e = COCOeval(gt, dt, iouType="bbox")
    e.evaluate()
    e.accumulate()
    e.summarize()

    stats = e.stats
    return {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
        "mAR_max1": float(stats[6]),
        "mAR_max10": float(stats[7]),
        "mAR_max100": float(stats[8]),
        "mAR_small": float(stats[9]),
        "mAR_medium": float(stats[10]),
        "mAR_large": float(stats[11]),
    }


def per_class_ap(gt_json: str, pred_json: str) -> dict:
    """Compute per-class AP@[.5:.95] for surfacing weak classes."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np

    gt = COCO(gt_json)
    dt = gt.loadRes(pred_json)
    e = COCOeval(gt, dt, iouType="bbox")
    e.evaluate()
    e.accumulate()

    # precision shape: [TxRxKxAxM] — T=IoU thr, R=recall thr, K=cats, A=areas, M=maxDets
    precision = e.eval["precision"]  # noqa
    cats = gt.loadCats(gt.getCatIds())
    out = {}
    for k, cat in enumerate(cats):
        prec = precision[:, :, k, 0, -1]  # all IoU, all recall, k-th cat, all areas, maxDet=100
        prec = prec[prec > -1]
        ap = float(np.mean(prec)) if prec.size > 0 else 0.0
        out[cat["name"]] = ap
    return out


def visualize_predictions(image_dir: str, gt_json: str, pred_json: str, n_samples: int = 10):
    """Render GT (green) + predicted (red) boxes on n_samples images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    import random

    out_dir = WORKDIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = json.loads(Path(gt_json).read_text())
    preds = json.loads(Path(pred_json).read_text())
    image_id_to_file = {im["id"]: im["file_name"] for im in gt["images"]}
    gt_by_image = {}
    for ann in gt["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)
    pred_by_image = {}
    for p in preds:
        pred_by_image.setdefault(p["image_id"], []).append(p)

    sample_ids = random.sample(list(image_id_to_file.keys()), min(n_samples, len(image_id_to_file)))
    for img_id in sample_ids:
        img_path = Path(image_dir) / image_id_to_file[img_id]
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        for ann in gt_by_image.get(img_id, []):
            x, y, w, h = ann["bbox"]
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="green", linewidth=2))
        for p in pred_by_image.get(img_id, []):
            if p["score"] < 0.3:
                continue
            x, y, w, h = p["bbox"]
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2))
        ax.set_title(f"image {img_id}: GT (green), pred (red, score>=0.3)")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"sample_{img_id}.png", dpi=100)
        plt.close()
        print(f"Chart saved as sample_{img_id}.png")


def save_metrics(metrics: dict):
    out = WORKDIR / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {out}")
```

## Hard constraints

- NEVER apply augmentation to the eval dataloader. Augmented eval invalidates mAP.
- NEVER skip per-class AP. A high mAP can hide a class with AP=0 (often a label-mapping bug).
- NEVER report mAP-50 alone for a task with strict IoU requirements (medical, autonomous driving). Report mAP@[.5:.95] as the headline.
- NEVER use box format `xywh` interchangeably with `xyxy`. COCO uses xywh in absolute pixel coords; YOLO uses xywh in normalized coords; Pascal VOC uses xyxy. Pick one and stick to it.
- NEVER ignore the "AP=0 for class X" signal. Either the model never sees that class, or label IDs are misaligned. Triage immediately.
- NEVER cache predictions across model versions.

## Research hooks

- **mAP variant trade-offs.** Query: *"When to prefer mAP@[.5:.95] vs mAP-50 vs mAP-75 vs PR-AUC for object detection on `{data_domain}` as of {today}."*
- **pycocotools alternatives.** Query: *"Status of `pycocotools` vs `torchmetrics.detection.MeanAveragePrecision` vs `pycocotools-fast` for COCO-style detection eval as of {today}."*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- `<workdir>/metrics.json` exists with at minimum `mAP_50_95`, `mAP_50`, `mAP_75`, `per_class_AP`, `worst_classes`.
- Eval dataloader had NO augmentation.
- `<workdir>/predictions/eval_predictions_coco.json` exists in valid COCO format (list of `{image_id, category_id, bbox, score}` dicts).
- 10 sample visualizations saved to `<workdir>/predictions/sample_<id>.png` with GT green + predicted red.
- mAP@[.5:.95] is non-trivially above 0 (>5% even for poor models; if 0, almost always a label-mapping or coordinate-format bug).
- IF any per-class AP is 0: the user was alerted and the issue triaged.

## Output checklist

- [ ] Predictions converted to COCO JSON
- [ ] COCO eval ran cleanly
- [ ] Per-class AP computed; worst classes surfaced
- [ ] mAP@[.5:.95] headline reported
- [ ] 10 sample visualizations saved
- [ ] Metrics JSON written
- [ ] Eval was NOT augmented
- [ ] AP=0 classes triaged if any
