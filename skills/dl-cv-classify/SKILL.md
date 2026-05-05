---
name: dl-cv-classify
description: Use to finetune an image classification model. Handles single-label and multi-label classification via timm backbones (default) or HF AutoModelForImageClassification (when staying in the HF ecosystem). Do NOT use for object detection (use dl-cv-detect), segmentation (use dl-cv-segment), or non-image data.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# CV Classify

Finetune an image classifier. Picks backbone + head + loss per task; hands off to `dl-finetune-loop` for the training loop.

## When to invoke

- User wants to train an image classifier (single-label or multi-label).
- After `dl-load-data` and `dl-augment` have run.

## When NOT to invoke

- Object detection (use `dl-cv-detect`).
- Segmentation (use `dl-cv-segment`).
- Inference / evaluation only (use `dl-cv-eval-classify`).

## Decision rules

### Backbone

- Default: invoke `dl-prior-art` first to see what winners on similar problems used. If `Confidence: high`, use that backbone.
- Otherwise: timm default per data domain:
  - Natural images, modest dataset: `timm.create_model("efficientnet_b0", pretrained=True, num_classes=N)`.
  - Larger dataset / need top accuracy: `timm.create_model("efficientnetv2_rw_m", pretrained=True)`.
  - Vision Transformer territory: `timm.create_model("vit_base_patch16_224", pretrained=True)`.
  - Small / mobile / edge: `timm.create_model("mobilenetv3_small_100", pretrained=True)`.

User can override with any timm model id. Confirm at runtime.

### Head

- `num_classes=N` passed to `timm.create_model(...)` replaces the head automatically.
- For multi-label: use `timm.create_model(..., num_classes=N)` AND set loss to `BCEWithLogitsLoss` (NOT CrossEntropy).

### Loss

- Single-label: CrossEntropy with optional label smoothing (0.1 for large datasets, 0.0 for small / fine-grained).
- Multi-label: BCEWithLogits (raw logits per class; sigmoid at inference).
- Class imbalance: weight classes inversely by frequency, OR use focal loss (`timm.loss.AsymmetricLossMultiLabel` for multi-label imbalance).

### Optimizer

- Default: AdamW, lr=1e-4 for the head, lr=1e-5 for the backbone (differential lr — common Kaggle practice).
- For very small datasets: SGD with momentum, lr=1e-3.

## Process

### Step 1 — Consult prior art

Invoke `dl-prior-art` for the user's specific task (e.g., "Kaggle medical CT classification 2024"). Use the recommended backbone if confidence is high.

### Step 2 — Pick backbone (with user confirmation)

Default suggestion comes from prior art OR the decision rules above. Surface to the user:

> Suggested backbone: `efficientnet_b0` (timm). Reason: prior art on similar problems / fast & accurate baseline. Use this, or specify another?

Wait for confirmation.

### Step 3 — Build the model

Generate model construction code per the chosen backbone, num_classes, and loss.

### Step 4 — Hand off to `dl-finetune-loop`

`dl-finetune-loop` will pick Trainer (default for classification — standard loss) and wire the training loop, mixed precision, callbacks.

### Step 5 — Verify

`dl-finetune-loop`'s smoke test runs. Then hand off to `dl-cv-eval-classify` for evaluation.

## Recipe template

### `<workdir>/src/_model_cv_classify.py`

```python
"""Build the image classification model. Adapt backbone per dl-prior-art recommendation."""
import json
import os
from pathlib import Path

import timm
import torch.nn as nn

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()
POLICY = json.loads((WORKDIR / "data_policy.json").read_text())


def build_model(backbone: str = "efficientnet_b0", num_classes: int | None = None, multi_label: bool = False):
    num_classes = num_classes or POLICY.get("num_classes")
    if num_classes is None:
        raise ValueError("num_classes not in data_policy.json — re-run dl-load-data")
    model = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
    return model


def build_loss(multi_label: bool = False, label_smoothing: float = 0.1, class_weights: "torch.Tensor | None" = None):
    if multi_label:
        return nn.BCEWithLogitsLoss(weight=class_weights)
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)


def make_param_groups(model, head_lr: float = 1e-4, backbone_lr: float = 1e-5):
    """Differential lr: head trains faster than the pretrained backbone."""
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name or "classifier" in name or "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]
```

## Hard constraints

- NEVER use CrossEntropy loss for multi-label classification. CE assumes mutually exclusive classes; multi-label needs BCEWithLogits.
- NEVER skip prior-art lookup for an unfamiliar domain. The wrong backbone family can lose 5-10% accuracy compared to the winner's choice.
- NEVER train the backbone with the same lr as the head on a small dataset. Differential lr (backbone 10x lower) is the standard.
- NEVER load a timm backbone without `pretrained=True` unless the user is doing self-supervised pretraining (`dl-cv-pretrain`, Phase 3).

## Research hooks

- **Current SOTA backbone for `{data_domain}` classification.** Query: *"Current top timm backbone for `{data_domain}` image classification on `{dataset_size}` data as of {today}, considering speed/accuracy tradeoff."*
- **Class imbalance handling.** Query: *"Recommended loss / sampling strategy for image classification with class imbalance ratio `{ratio}` as of {today}."*

Always invoke `dl-prior-art` for the user's specific competition / dataset before invoking this skill.

## Verification gates

After this skill runs (and `dl-finetune-loop` has built the loop), `ml-engineer-verify` MUST check:

- A model object is constructable from the recipe (a 1-batch forward pass returns logits of the right shape).
- For multi-label: loss is BCEWithLogits, NOT CrossEntropy.
- For single-label: model output shape is `(batch, num_classes)`.
- The backbone was loaded with `pretrained=True`.
- Differential lr is set if dataset is small.

## Output checklist

- [ ] `dl-prior-art` consulted for backbone recommendation
- [ ] User confirmed the chosen backbone
- [ ] Model + loss + optimizer constructed per task
- [ ] Differential lr applied if small dataset
- [ ] Handed off to `dl-finetune-loop`
