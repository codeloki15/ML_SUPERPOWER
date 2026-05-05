---
name: dl-augment
description: Use after dl-load-data and before dl-finetune-loop, when training data benefits from augmentation. Picks the right library and recipe per use case (CV → albumentations + mixup/cutmix; NLP → conditional back-translation / MLM noise / EDA). Reads <workdir>/data_policy.json to know task type. Do NOT use for evaluation pipelines (eval data must NOT be augmented) or for tasks where augmentation hurts (small-text NER often does).
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# Augment

Pick the right augmentation library and recipe per task. No baked-in default — the choice depends on data shape, task type, and what `dl-prior-art` says winners on similar problems used.

## When to invoke

- After `dl-load-data` has locked the data policy AND before `dl-finetune-loop` constructs the training loop.
- Re-invoke when adding a new augmentation strategy mid-iteration (e.g., baseline → add mixup → check if score improves).

## When NOT to invoke

- For evaluation / test pipelines. Eval data MUST NOT be augmented.
- For tasks where augmentation has been shown to hurt (very small NER datasets, span-sensitive tasks).
- When `dl-prior-art` returned `Confidence: high` saying "winners did not augment for this task".

## Decision rules

Read `<workdir>/data_policy.json` to determine `task_type`. Then:

### CV tasks

- IF `task_type == "image_classification"`:
  - **Spatial / color**: albumentations recipe (HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, GaussianNoise). Use `A.Compose` with `bbox_params=None`.
  - **Mixup / Cutmix**: timm's `Mixup` class (handles soft labels properly). Apply at the collator level, not in the per-example transform.
  - **Strong recipe (when winners used it)**: add RandAugment or AutoAugment via timm.

- IF `task_type == "object_detection"`:
  - **Spatial-aware**: albumentations with `bbox_params=BboxParams(format="yolo")` or `format="coco"`. NEVER use bbox-blind transforms (HorizontalFlip on its own will silently corrupt boxes).
  - **Mosaic**: ultralytics-style mosaic augmentation. Combine 4 images into one — strong default for YOLO-family.
  - **MixUp**: not standard for detection; skip unless `dl-prior-art` says winners used it.

- IF `task_type == "segmentation"`:
  - **Mask-aware**: albumentations with `additional_targets={"mask": "mask"}`. Spatial transforms apply identically to image and mask.
  - **Pixel-level**: avoid color transforms that change semantic meaning (e.g., for medical / satellite, color shifts may invert what classes mean).

### NLP tasks

NLP augmentation is conditional. Check `dl-prior-art` first; many NLP tasks do NOT benefit from augmentation.

- IF `task_type == "text_classification"` AND dataset is small (<1k examples per class):
  - **Back-translation**: translate text to a pivot language and back. Effective but expensive.
  - **EDA-style** (Easy Data Augmentation): synonym replacement, random insertion / swap / deletion. Cheap; works on short text.

- IF `task_type == "token_classification"` (NER):
  - **Avoid most augmentation** — span boundaries are sensitive. Word-level perturbations break BIO tags.
  - **Allowed**: random masking of non-entity tokens during training (MLM-style). Skip if dataset is small (<1k sentences); the regularization hurts more than it helps below that.

- IF `task_type == "generative"`:
  - **No augmentation** at the loss level. If you need data variety, generate via the model itself (synthetic data), but that's a `dl-llm-instruction-tune` concern, not this skill.

## Process

### Step 1 — Read data policy + consult prior art

Read `<workdir>/data_policy.json`. Note `task_type`.

Optionally invoke `dl-prior-art` to see what augmentation winners on similar problems used. If `dl-prior-art` returned a strong recommendation, follow it.

### Step 2 — Pick library + recipe per decision rules

Walk the decision tree above. Report the chosen library, recipe, and rationale to the user.

### Step 3 — Generate the augmentation pipeline

Produce the augmentation code:

- For CV with albumentations: `A.Compose([...], bbox_params=...)`.
- For CV with mixup/cutmix: a `Mixup` collator.
- For NLP with back-translation: a `transform` function that calls a translation model.

Insert into the dataloader's `with_transform` (deterministic) OR collator (mixup-style) — NOT into a `map` call (would freeze the augmentation).

### Step 4 — Update data policy

Append augmentation choices to `<workdir>/data_policy.json`:

```json
{
  ...
  "augmentation": {
    "library": "albumentations",
    "recipe": "spatial_color_classification_v1",
    "transforms": ["HorizontalFlip(p=0.5)", "RandomBrightnessContrast(p=0.3)", "ShiftScaleRotate(p=0.3)"],
    "mixup": {"alpha": 0.2, "enabled": true}
  }
}
```

### Step 5 — Verify

Sample 10 augmented examples from the training dataloader. Save to `<workdir>/charts/augmented_samples.png` (CV) or `<workdir>/samples/augmented_examples.txt` (NLP). The user may inspect to confirm augmentations look sane.

## Recipe template

### CV — `<workdir>/src/_augment_cv.py`

```python
"""Augmentation pipeline for CV tasks. Reads data_policy.json to pick the right recipe."""
import json
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def _load_policy() -> dict:
    """Lazy-load data_policy.json — call inside builders, not at import time."""
    return json.loads((WORKDIR / "data_policy.json").read_text())


def make_classification_augment(image_size: int | None = None):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    if image_size is None:
        image_size = _load_policy().get("image_size", [224, 224])[0]

    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def make_detection_augment(image_size: int = 640, bbox_format: str = "yolo"):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.LongestMaxSize(image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format=bbox_format, label_fields=["labels"]))


def make_segmentation_augment(image_size: int = 512):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


def make_mixup_collator(num_classes: int, alpha: float = 0.2):
    """Mixup at the collator level — not per-example. Returns soft labels."""
    from timm.data import Mixup
    return Mixup(mixup_alpha=alpha, cutmix_alpha=alpha, num_classes=num_classes, label_smoothing=0.1)
```

### NLP — `<workdir>/src/_augment_nlp.py`

```python
"""Augmentation pipeline for NLP tasks. Conditional — many NLP tasks skip this entirely."""
import os
import random
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def eda_synonym_replacement(text: str, n: int = 1) -> str:
    """Replace n random non-stopwords with WordNet synonyms.
    Falls back to identity if NLTK / WordNet not installed.
    """
    try:
        import nltk
        from nltk.corpus import wordnet
    except ImportError:
        return text
    words = text.split()
    if len(words) < 2:
        return text
    candidate_indices = list(range(len(words)))
    random.shuffle(candidate_indices)
    replaced = 0
    for idx in candidate_indices:
        synsets = wordnet.synsets(words[idx])
        if synsets:
            synonyms = [lemma.name() for syn in synsets for lemma in syn.lemmas()]
            synonyms = [s for s in synonyms if s.lower() != words[idx].lower()]
            if synonyms:
                words[idx] = random.choice(synonyms).replace("_", " ")
                replaced += 1
                if replaced >= n:
                    break
    return " ".join(words)


def random_token_mask(
    text: str,
    mask_token: str = "[MASK]",
    p: float = 0.15,
    entity_mask: list[bool] | None = None,
) -> str:
    """Randomly mask tokens (MLM-style).

    For NER training data, pass `entity_mask` (True at entity positions) so entity
    tokens are NEVER masked — masking entity tokens silently corrupts BIO alignment
    and breaks the model's learning signal.
    """
    words = text.split()
    if entity_mask is not None and len(entity_mask) != len(words):
        raise ValueError(f"entity_mask length {len(entity_mask)} != tokens {len(words)}")
    out = []
    for i, w in enumerate(words):
        if entity_mask is not None and entity_mask[i]:
            out.append(w)  # never mask entity tokens
        elif random.random() < p:
            out.append(mask_token)
        else:
            out.append(w)
    return " ".join(out)
```

## Hard constraints

- NEVER apply augmentation to evaluation / test data. Evaluation reflects real-world performance; augmentation contaminates it.
- NEVER cache augmented examples via `dataset.map(augment_fn)`. `map` runs once and freezes the augmentation; use `with_transform` or a collator.
- NEVER use bbox-blind transforms on detection data. Even a single `HorizontalFlip` outside `bbox_params` will silently corrupt boxes.
- NEVER apply color-shift transforms to medical / satellite / scientific imagery without confirming the colors carry no semantic meaning.
- NEVER apply augmentation to NER data without confirming the augmentation preserves BIO tag alignment.
- NEVER stack mixup with strong augmentation (RandAugment) on small datasets — the regularization compounds and the model under-fits.

## Research hooks

Augmentation best practices shift. Before generating a pipeline for an unfamiliar problem, invoke `ml-engineer-research` and `dl-prior-art`:

- **What augmentations did Kaggle winners on `<similar_competition>` use?** Invoke `dl-prior-art` first.
- **Current SOTA augmentation recipe for `{task_type}`.** Query: *"Current strongest augmentation recipe for `{task_type}` on `{data_domain}` as of {today}."*
- **NLP augmentation effectiveness by dataset size.** Query: *"At what dataset size does `{augmentation_method}` (back-translation, EDA, MLM masking) help vs hurt for `{task_type}` as of {today}?"*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- `<workdir>/data_policy.json` was updated with an `augmentation` block.
- A sample of 10 augmented examples was saved to `<workdir>/charts/augmented_samples.png` (CV) or `<workdir>/samples/augmented_examples.txt` (NLP).
- For detection: bounding boxes in augmented samples are still valid (within image bounds, non-zero area).
- For segmentation: mask shapes match image shapes after augmentation.
- For NER: BIO tags in augmented samples align with token boundaries.
- The eval pipeline (if any has been built) does NOT include the augmentation transforms.

## Output checklist

- [ ] Data policy read; `task_type` determined
- [ ] Library + recipe picked per decision rules
- [ ] Augmentation pipeline inserted into the dataloader (via `with_transform` or collator, NOT `map`)
- [ ] Data policy updated with `augmentation` block
- [ ] Sample augmentations saved for inspection
- [ ] Eval pipeline confirmed augmentation-free
