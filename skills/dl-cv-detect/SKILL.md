---
name: dl-cv-detect
description: Use to finetune an object detection model. Backbone family (YOLO11, RT-DETR, Detectron2, etc.) is chosen at runtime via dl-prior-art recommendation + user confirmation — no baked-in default. Coordinates with dl-augment for bbox-aware augmentation. Do NOT use for classification (use dl-cv-classify), segmentation (use dl-cv-segment), or non-image data.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# CV Detect

Finetune an object detector. The backbone family is NOT pre-decided — always invoke `dl-prior-art` first, then ask the user to confirm. The decision space (YOLO11 / RT-DETR / Detectron2 / DETR / Faster-RCNN) is wide and what wins depends on data domain (real-time vs throughput, small objects vs large, dense scenes vs sparse).

## When to invoke

- User wants to train an object detector (bounding box detection).
- After `dl-load-data` has loaded the data and `dl-augment` has set up bbox-aware augmentation.

## When NOT to invoke

- Classification (use `dl-cv-classify`).
- Segmentation (use `dl-cv-segment`) — though some skills like SAM2 do detection+segmentation jointly.
- Inference / evaluation only (use `dl-cv-eval-detect`).

## Decision rules

### Backbone — runtime selection (NO default)

This is a Q4 brainstorm decision: the skill MUST consult the user at runtime.

1. Invoke `dl-prior-art` for the user's specific problem (e.g., "Kaggle wheat detection 2024", "satellite ship detection SOTA"). Note the recommended family and confidence level.
2. Surface a multiple-choice prompt to the user:
   > "For object detection on your task, three families typically work:
   > 1. **YOLO11/YOLOv12** (Ultralytics) — real-time, easy to deploy, strong on standard benchmarks. Default if speed matters.
   > 2. **RT-DETR** (HF transformers) — transformer-based, slightly better mAP at the cost of speed.
   > 3. **Detectron2** (Facebook) — academic baselines, Mask-RCNN family, slow.
   >
   > Prior art on your problem suggests: `<recommendation from dl-prior-art>` (confidence: `<low|medium|high>`).
   >
   > Which backbone? (1/2/3, or specify a HuggingFace model id)"

Wait for confirmation. Do NOT proceed without explicit user choice.

### Annotation format

- IF using YOLO family: convert annotations to YOLO format (`<class> <cx> <cy> <w> <h>` normalized 0-1 per image, one txt per image).
- IF using RT-DETR / DETR / Detectron2: convert to COCO JSON format.
- IF input data is in a different format (Pascal VOC XML, custom JSON), flag the conversion as an explicit step before training.

### Loss

- YOLO family: built-in (CIoU + objectness + classification, all weighted internally).
- RT-DETR / DETR family: Hungarian-matched bipartite loss (built-in).
- Don't override unless the user has a strong reason.

### Optimizer / lr

- YOLO: ultralytics defaults (SGD with cosine, lr0=0.01) work well; tune only if `dl-prior-art` says winners did.
- RT-DETR: AdamW, lr=1e-4 for the head, lr=1e-5 for the backbone (transformer-style differential lr).

## Process

### Step 1 — Consult prior art

Invoke `dl-prior-art` for the user's specific detection task. Note recommended backbone and confidence.

### Step 2 — Ask user for backbone choice

Surface the menu (above) with the prior-art recommendation. Wait for user response. Do NOT proceed without confirmation.

### Step 3 — Convert annotations to the required format

Check the dataset's annotation format from `<workdir>/data_policy.json`. If it doesn't match what the chosen backbone expects, generate conversion code (Pascal VOC → COCO, COCO → YOLO, etc.). Save converted annotations next to the originals; do NOT overwrite.

### Step 4 — Build the model

Generate model construction code per the chosen backbone. For YOLO, this is `YOLO("yolo11n.pt")` from ultralytics. For RT-DETR, `AutoModelForObjectDetection.from_pretrained(model_id)` from HF transformers.

### Step 5 — Hand off to `dl-finetune-loop`

For YOLO: `dl-finetune-loop` recognizes Ultralytics models and uses ultralytics' built-in `model.train(...)` rather than HF Trainer (Trainer doesn't fit YOLO's API). Surface this special case.

For RT-DETR / DETR: `dl-finetune-loop` Trainer path works directly.

### Step 6 — Verify

Hand off to `dl-cv-eval-detect` for mAP evaluation. Inspect 10 sample predictions visually (saved to `<workdir>/predictions/sample_<id>.png` with predicted boxes drawn).

## Recipe template

### `<workdir>/src/_model_cv_detect_yolo.py` (YOLO11 / Ultralytics path)

```python
"""Build and train a YOLO11 detector. Uses Ultralytics API; not HF Trainer."""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def build_yolo(model_size: str = "n", pretrained: bool = True):
    """model_size in {n,s,m,l,x} for YOLO11; n is fastest, x is most accurate."""
    from ultralytics import YOLO
    weights = f"yolo11{model_size}.pt" if pretrained else f"yolo11{model_size}.yaml"
    return YOLO(weights)


def train_yolo(model, data_yaml: str, epochs: int = 50, imgsz: int = 640, batch: int = 16, **kwargs):
    """Train via Ultralytics API. data_yaml points to the YOLO-format dataset config."""
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(WORKDIR / "checkpoints"),
        name="yolo_train",
        **kwargs,
    )
    return results
```

### `<workdir>/src/_model_cv_detect_rtdetr.py` (RT-DETR / HF path)

```python
"""Build an RT-DETR detector. Uses HF transformers; works with dl-finetune-loop Trainer path."""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def build_rtdetr(model_id: str = "PekingU/rtdetr_r50vd_coco_o365", num_labels: int | None = None):
    from transformers import AutoModelForObjectDetection, AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(model_id)
    model_kwargs = {}
    if num_labels is not None:
        model_kwargs["num_labels"] = num_labels
        model_kwargs["ignore_mismatched_sizes"] = True
    model = AutoModelForObjectDetection.from_pretrained(model_id, **model_kwargs)
    return model, processor


def make_param_groups(model, head_lr: float = 1e-4, backbone_lr: float = 1e-5):
    """Differential lr: head trains faster than the pretrained backbone."""
    head_params = [p for n, p in model.named_parameters() if "class_embed" in n or "bbox_embed" in n]
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    other_params = [p for n, p in model.named_parameters()
                    if "class_embed" not in n and "bbox_embed" not in n and "backbone" not in n]
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": other_params, "lr": head_lr},
        {"params": head_params, "lr": head_lr * 10},
    ]
```

## Hard constraints

- NEVER pick a backbone without invoking `dl-prior-art` first AND getting explicit user confirmation. Detection backbone choice has 5-15% mAP swings depending on data; defaults are dangerous here.
- NEVER use bbox-blind augmentation. `dl-augment` MUST have been invoked with `task_type=object_detection` (which sets `bbox_params`); if not, halt and re-invoke `dl-augment` first.
- NEVER mix annotation formats mid-training. Convert once before training; commit to one format.
- NEVER use the YOLO family with HF Trainer. YOLO ships its own training API; trying to wrap it in Trainer breaks both.
- NEVER finetune a detector without freezing the backbone for the first 1-3 epochs unless `dl-prior-art` explicitly says winners didn't. Cold-start full-finetune often diverges on small detection datasets.
- NEVER skip the visual inspection of predictions (Step 6). mAP can look reasonable while predictions are visibly nonsense (off by class, off by box scale).

## Research hooks

Detection moves fast — the SOTA family changes year-over-year.

- **Current SOTA detector for `{data_domain}`.** Always invoke `dl-prior-art` first; this query is a fallback. Query: *"Current top object detector for `{data_domain}` (real-time vs throughput tradeoff) as of {today}."*
- **YOLO version recommendation.** Query: *"Current latest YOLO version with stable API and pretrained weights as of {today} (YOLO11 vs YOLO12 vs YOLOv13 status)."*
- **Annotation conversion gotchas.** Query: *"Common pitfalls when converting `{source_format}` to `{target_format}` annotations as of {today} (especially small-object precision, normalized vs absolute coords)."*

## Verification gates

After this skill runs (and `dl-finetune-loop` has trained the detector), `ml-engineer-verify` MUST check:

- The user explicitly confirmed the backbone choice (don't proceed silently on the prior-art recommendation).
- The augmentation pipeline (in `<workdir>/data_policy.json`) has bbox-aware transforms (NOT the classification/segmentation pipelines).
- For YOLO: training ran via `model.train(...)` from ultralytics, NOT through HF Trainer.
- For HF detectors (RT-DETR / DETR): the model loaded with `ignore_mismatched_sizes=True` if the pretrained head's num_classes differs from the user's task.
- 10 sample predictions saved to `<workdir>/predictions/` with boxes drawn (visual sanity).
- mAP@0.5 from `dl-cv-eval-detect` is meaningfully above random (a random detector mAP ≈ 0; even a broken trained model usually gets >5%).

## Output checklist

- [ ] `dl-prior-art` consulted; recommendation surfaced
- [ ] User explicitly confirmed backbone choice
- [ ] Annotation format converted if needed
- [ ] Model + training loop wired per chosen backbone
- [ ] YOLO uses ultralytics API; HF detectors use Trainer path
- [ ] Visual inspection of 10 predictions complete
- [ ] Handed off to `dl-cv-eval-detect`
