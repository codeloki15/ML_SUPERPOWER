---
name: dl-cv-pretrain
description: Use ONLY when standard timm pretrained backbones don't transfer to the user's domain (medical imaging, satellite, microscopy, scientific) AND user has lots of unlabeled images (≥100k). Self-supervised pretraining via SimCLR / DINO / MAE. Skill's primary job is recognizing when to SKIP — most users should NOT pretrain. Do NOT use for natural-image tasks (timm pretrained beats anything you'd do), with <100k unlabeled images, or as a substitute for trying timm pretrained first.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# CV Pretrain

Self-supervised pretraining for domains where ImageNet/CLIP-pretrained backbones don't transfer. SKIP IF YOU CAN. Most users should NEVER pretrain — timm pretrained backbones cover natural images, satellite, medical imaging via specialist models (RadImageNet, BioCLIP), etc.

## When to invoke

- User has a domain where pretrained backbones empirically underperform: medical (CT/MRI/histopathology with custom modality), satellite (specific spectral bands not in standard pretraining), microscopy (electron / fluorescence with non-natural-image statistics), scientific imaging.
- User has ≥100k unlabeled images in the domain.
- User has tried timm pretrained AND `dl-prior-art` confirms self-supervised pretraining helped on similar problems.

## When NOT to invoke (i.e., SKIP)

- User hasn't tried timm pretrained yet — try that FIRST.
- Natural images (faces, animals, objects) — timm pretrained beats any pretraining you can do.
- <100k unlabeled images — pretraining underperforms with small data.
- User isn't certain pretrained backbones underperform — confirm with a baseline experiment first.
- User is short on compute — pretraining is 10-100x more expensive than finetuning.

If 80% of users SKIP this skill on first invocation, that's the right outcome.

## Decision rules

### Recognize-when-to-skip (FIRST)

Apply this gate before anything else:

1. Has the user tried timm pretrained? IF NO — return: "Try `timm.create_model('<backbone>', pretrained=True)` first. Pretrained ImageNet backbones transfer surprisingly well to many specialist domains. Only invoke `dl-cv-pretrain` if pretrained backbones are clearly underperforming."
2. Does the user have ≥100k unlabeled images? IF NO — return: "Self-supervised pretraining needs lots of unlabeled data (≥100k images). With less, finetune a timm pretrained backbone instead."
3. Has `dl-prior-art` confirmed pretraining helped on similar problems? IF NO — return: "Invoke `dl-prior-art` first; if winners on similar problems didn't pretrain, neither should you."

If ALL three checks pass, proceed.

### Method (when proceeding)

- **DINO (default)**: best across most settings. Self-distillation; no negatives needed. Vision Transformer + ConvNeXt friendly.
- **SimCLR**: contrastive; needs careful negative sampling. Older but well-understood.
- **MAE (Masked Autoencoder)**: reconstruction-based; good for ViT.

### Library

- **lightly**: SimCLR / MoCo / BYOL / DINO recipes; PyTorch.
- **dino-vits / dinov2 (Facebook)**: official DINO + DINOv2 implementation.
- **timm**: ConvNeXt + ViT pretraining recipes (limited but available).

## Process

### Step 1 — Skip-check gate (mandatory)

Apply the recognize-when-to-skip rules. If skip — return the appropriate message and STOP. Do NOT proceed to pretraining.

### Step 2 — Pick method + library + ask user

If proceeding (rare), surface choice + rationale + warning ("This will take ~3-7 days on a single GPU; consider remote compute via `dl-remote-execute`").

### Step 3 — Hand off to library

Write the pretraining script using the chosen library. Pretraining is long-running; recommend `dl-remote-execute` to a multi-GPU env.

### Step 4 — Verify

After pretraining, the resulting backbone should improve downstream finetune metric over timm pretrained baseline. If it doesn't — pretraining wasted compute; use timm baseline.

## Recipe template

### `<workdir>/src/_pretrain_dino.py` (DINO via lightly)

Thin recipe pointer — full DINO recipes are out of v1 scope:

```python
"""Thin DINO pretraining pointer. Full implementation deferred to v2."""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def dino_pretrain_skeleton(unlabeled_image_dir: str, backbone: str = "vit_small_patch16_224",
                            epochs: int = 100, batch_size: int = 256):
    """
    Pre-training entry point. Use lightly's DINO implementation:

    https://docs.lightly.ai/self-supervised-learning/examples/dino.html

    Skeleton (pseudocode — full impl deferred to v2):
    1. Load timm backbone (no pretrained weights, OR start from pretrained as warm-init).
    2. Wrap with lightly.DINOHead + DINO loss.
    3. Train on unlabeled_image_dir with global+local crops.
    4. Save backbone weights to <workdir>/pretrained_backbone/.

    For a complete recipe, invoke ml-engineer-research with:
    'Current DINOv2 / DINO recipe for {domain} domain pretraining as of {today}.'
    """
    raise NotImplementedError(
        "Full DINO recipe deferred to v2. "
        "Use lightly library directly with the pseudocode above, "
        "OR invoke ml-engineer-research for a current full recipe."
    )
```

## Hard constraints

- ALWAYS apply the skip-check gate (Step 1) before any pretraining work. If user hasn't tried timm pretrained, halt and tell them.
- NEVER pretrain when timm pretrained beats your domain. Wasted compute, worse outcomes.
- NEVER pretrain without confirming the unlabeled data is large enough (≥100k images).
- NEVER pretrain without first checking `dl-prior-art` for the user's domain. Domain-specific pretrained backbones (RadImageNet for medical, BioCLIP for biology, ScaleMAE for satellite) often beat your custom pretraining.
- NEVER skip evaluating the pretrained backbone against timm pretrained on the same downstream task. If your pretrained backbone doesn't beat timm baseline, pretraining failed.

## Research hooks

- **Domain-specific pretrained backbones.** Query: *"Are there domain-specific pretrained backbones for `{domain}` (medical / satellite / microscopy) that beat ImageNet pretrained as of {today}? E.g., RadImageNet, BioCLIP, ScaleMAE."*
- **Self-supervised method comparison.** Query: *"Latest comparison of DINOv2 / SimCLR / MAE / iBOT for self-supervised pretraining on `{domain}` as of {today}."*
- **Minimum data threshold.** Query: *"Minimum unlabeled dataset size for self-supervised pretraining to beat ImageNet pretrained on `{domain}` as of {today}."*

## Verification gates

After this skill runs (whether it ran pretraining OR returned skip), `ml-engineer-verify` MUST check:

- Skip-check gate ran (Step 1) — log its verdict.
- IF skip returned: user got the appropriate "try X first" message; no pretraining ran.
- IF pretraining ran: pretrained backbone evaluated against timm pretrained on same downstream task; user informed if pretraining did NOT improve over baseline.
- IF pretraining ran: estimated compute was surfaced (`~3-7 days on single GPU`) and user explicitly accepted.

## Output checklist

- [ ] Skip-check gate applied
- [ ] If skip: appropriate message returned; STOP
- [ ] If proceed: prior-art consulted; method+library chosen
- [ ] User warned about compute cost; explicit accept
- [ ] Pretraining ran (or remote-execute handoff)
- [ ] Pretrained backbone evaluated vs timm baseline; user informed of outcome
