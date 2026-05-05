---
name: dl-cv-segment
description: Use to finetune a segmentation model — semantic, instance, or panoptic. Decision tree picks SAM2/SAM3 (zero-shot/promptable), YOLO-seg (real-time), U-Net family (medical/scientific), or HF AutoModelForSemanticSegmentation per task. Coordinates with dl-augment for mask-aware augmentation. Do NOT use for classification (use dl-cv-classify), detection (use dl-cv-detect), or non-image data.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# CV Segment

Finetune a segmentation model. Decide between zero-shot promptable (SAM2/SAM3), real-time (YOLO-seg), medical/scientific (U-Net family), or general (HF AutoModelForSemanticSegmentation) based on task type and `dl-prior-art`.

## When to invoke

- User wants to train semantic / instance / panoptic segmentation.
- After `dl-load-data` and `dl-augment` (with `task_type=segmentation` for mask-aware transforms).

## When NOT to invoke

- Classification (use `dl-cv-classify`).
- Detection without masks (use `dl-cv-detect`); detection-to-mask handoff is fine here.
- Inference / evaluation only (use `dl-cv-eval-segment`).

## Decision rules

### Model family

1. Invoke `dl-prior-art` for the user's specific task. Note recommendation.
2. Apply the decision tree:

- IF the task is **zero-shot or promptable** (user has bounding-box prompts or text prompts; no labeled masks): use **SAM2** (Ultralytics) or **SAM3** (with text prompts). NO finetuning needed for many use cases — surface this.
- IF the task is **real-time instance segmentation** (latency matters, output is per-instance masks): use **YOLO-seg** (`yolo11{n,s,m,l,x}-seg.pt`). Same Ultralytics API as detection.
- IF the task is **medical / satellite / scientific** semantic segmentation with limited data and pixel-precise boundaries needed: use **U-Net family** (`segmentation-models-pytorch`'s `Unet`, `UnetPlusPlus`, `DeepLabV3Plus`).
- IF the task is **general semantic segmentation** with HF-compatible data: use **HF AutoModelForSemanticSegmentation** (Mask2Former, SegFormer, BEiT, etc.).
- DEFAULT (no strong signal): ask the user.

### Loss

- Semantic segmentation:
  - Single-class binary: BCEWithLogits + Dice (combo).
  - Multi-class: CrossEntropy + optional Dice.
- Instance segmentation: model-specific (Mask2Former / YOLO-seg ship their own).
- Heavy class imbalance (e.g., medical, where background dominates): Focal loss, or use sample weighting.

### Image size

- Read from `<workdir>/data_policy.json` (set by `dl-load-data`); default 512 if not set.
- For medical / satellite, often 512+ — do NOT downscale aggressively if pixel-precision matters.

## Process

### Step 1 — Consult prior art

Invoke `dl-prior-art` for the user's specific segmentation task.

### Step 2 — Pick model family + ask user

Apply decision tree. Surface choice + rationale to user. For zero-shot SAM, also surface "you may not need to finetune at all".

### Step 3 — Convert annotations to required format

- HF semantic seg: per-pixel class labels as PNG (palette mode or single-channel int).
- YOLO-seg: polygon coordinates per instance, normalized.
- U-Net (smp): per-pixel binary or one-hot tensors.

### Step 4 — Build model + loss

Generate code per family.

### Step 5 — Hand off to `dl-finetune-loop`

YOLO-seg uses ultralytics API (special case, like `dl-cv-detect`). Others use HF Trainer path.

### Step 6 — Verify

Hand off to `dl-cv-eval-segment`. Inspect 10 sample predictions (image + predicted-mask overlay) saved to `<workdir>/predictions/`.

## Recipe template

### `<workdir>/src/_model_cv_segment_sam.py` (SAM2/SAM3 zero-shot)

```python
"""Use SAM2 or SAM3 for zero-shot / promptable segmentation. Often no finetuning needed."""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def load_sam2(model_size: str = "b"):
    """model_size in {b (base), l (large), h (huge)} for SAM2."""
    from ultralytics import SAM
    return SAM(f"sam2.1_{model_size}.pt")


def predict_with_box_prompts(model, image_path: str, boxes: list):
    """boxes: list of [x1, y1, x2, y2] in image coordinates."""
    return model(image_path, bboxes=boxes)
```

### `<workdir>/src/_model_cv_segment_yoloseg.py` (YOLO-seg real-time)

```python
"""YOLO-seg for real-time instance segmentation. Ultralytics API."""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def build_yoloseg(model_size: str = "n"):
    from ultralytics import YOLO
    return YOLO(f"yolo11{model_size}-seg.pt")


def train_yoloseg(model, data_yaml: str, epochs: int = 50, imgsz: int = 640, batch: int = 16, **kwargs):
    return model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(WORKDIR / "checkpoints"),
        name="yoloseg_train",
        **kwargs,
    )
```

### `<workdir>/src/_model_cv_segment_unet.py` (U-Net family for medical/scientific)

```python
"""U-Net family from segmentation-models-pytorch (smp). Strong for medical/satellite."""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def build_unet(encoder: str = "resnet34", encoder_weights: str = "imagenet",
               in_channels: int = 3, classes: int = 1, arch: str = "Unet"):
    """arch in {Unet, UnetPlusPlus, DeepLabV3Plus, FPN, Linknet, MAnet}."""
    import segmentation_models_pytorch as smp
    return getattr(smp, arch)(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


def build_loss(num_classes: int, use_dice: bool = True, focal: bool = False):
    """Combo loss matching the Decision rules.

    - focal=True: FocalLoss alone (multiclass or binary).
    - num_classes == 1 and use_dice: BCE + Dice combo.
    - num_classes == 1 and not use_dice: BCEWithLogitsLoss alone.
    - num_classes > 1 and use_dice: CrossEntropy + Dice combo.
    - num_classes > 1 and not use_dice: CrossEntropy alone.

    Returns a callable `loss(pred, target) -> tensor`. Never returns None.
    """
    import torch.nn as nn
    import segmentation_models_pytorch as smp

    if focal:
        return smp.losses.FocalLoss(mode="multiclass" if num_classes > 1 else "binary")

    if num_classes == 1:
        bce = nn.BCEWithLogitsLoss()
        if not use_dice:
            return bce
        dice = smp.losses.DiceLoss(mode="binary")

        def combo(pred, target):
            return bce(pred, target.float()) + dice(pred, target)
        return combo

    ce = nn.CrossEntropyLoss()
    if not use_dice:
        return ce
    dice = smp.losses.DiceLoss(mode="multiclass")

    def combo(pred, target):
        return ce(pred, target) + dice(pred, target)
    return combo
```

### `<workdir>/src/_model_cv_segment_hf.py` (HF semantic segmentation)

```python
"""HF AutoModelForSemanticSegmentation — Mask2Former, SegFormer, BEiT, etc."""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def build_hf_segmenter(model_id: str = "nvidia/mit-b0", num_labels: int = 1, id2label: dict | None = None):
    from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(model_id)
    label2id = {v: k for k, v in (id2label or {}).items()} if id2label else None
    model = AutoModelForSemanticSegmentation.from_pretrained(
        model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model, processor
```

## Hard constraints

- NEVER use `dl-augment` without `task_type=segmentation`. Mask-blind augmentation silently mis-aligns image and mask.
- NEVER skip the per-pixel class consistency check after augmentation. Confirm `unique(mask)` is a subset of valid class IDs (no spurious values from interpolation).
- NEVER use a binary loss (BCE) on a multi-class task. CE is required when classes > 1.
- NEVER assume SAM zero-shot quality on out-of-distribution data (medical, satellite, microscopy). SAM was trained on natural images; verify with a small inspection set before committing to it.
- NEVER use bilinear interpolation on integer mask labels. Use nearest-neighbor (`Image.NEAREST`) when resizing masks; bilinear creates fractional values that map to no class.
- NEVER finetune a U-Net on tiny data (<200 images) without aggressive augmentation AND a frozen encoder for the first few epochs.

## Research hooks

- **Current SOTA segmentation family for `{data_domain}`.** Invoke `dl-prior-art` first; query as fallback: *"Current top segmentation architecture for `{data_domain}` ({medical, satellite, natural, scientific}) as of {today}."*
- **SAM family status.** Query: *"Current SAM family options (SAM, SAM2, SAM3, MobileSAM, FastSAM) and their tradeoffs as of {today}."*
- **Loss function for class imbalance.** Query: *"Recommended segmentation loss for class imbalance ratio `{ratio}` (Dice vs Focal vs Tversky vs combo) as of {today}."*

## Verification gates

After this skill runs (and `dl-finetune-loop` has trained), `ml-engineer-verify` MUST check:

- Augmentation pipeline (in `<workdir>/data_policy.json`) has `task_type=segmentation` AND `additional_targets={"mask": "mask"}` if albumentations.
- Mask resizes used nearest-neighbor (NOT bilinear), or augmentation library default is mask-aware.
- For SAM zero-shot: a sanity-check on out-of-distribution data was done before committing.
- For U-Net on small data: encoder frozen for at least 2 epochs.
- Multi-class tasks use CE (or Dice multiclass) — NOT BCE.
- 10 sample predictions saved as image + mask overlay to `<workdir>/predictions/`.
- IoU from `dl-cv-eval-segment` is meaningfully above random.

## Output checklist

- [ ] `dl-prior-art` consulted
- [ ] Model family picked + user confirmed
- [ ] Annotations converted to required format
- [ ] Mask-aware augmentation pipeline confirmed
- [ ] Model + loss wired per family
- [ ] YOLO-seg uses ultralytics API; others use HF Trainer
- [ ] 10 prediction overlays saved for visual inspection
- [ ] Handed off to `dl-cv-eval-segment`
