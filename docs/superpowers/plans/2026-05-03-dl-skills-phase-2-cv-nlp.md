# DL Skills Phase 2 — CV + NLP Breadth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `ml-engineer` plugin v0.2.0-alpha.2 — add 14 deep-learning skills covering CV training (3) + CV evaluation (3) + NLP training (2) + NLP evaluation (3) + cross-domain data/training (3). After Phase 2, end-to-end CV and NLP tasks work via the existing sub-agents (`cv-engineer`, `nlp-engineer`).

**Architecture:** Pure markdown additions to a Claude Code plugin. Each skill is a self-contained `SKILL.md` with frontmatter trigger + body following the structure established in Phase 1 (When to invoke / When NOT to invoke / Decision rules / Process / Recipe template / Hard constraints / Research hooks / Verification gates / Output checklist). Hybrid writing style per spec: terse headers/bullets, full prose for decisions and constraints.

**Tech Stack:** Markdown (CommonMark + GitHub-flavored). YAML frontmatter. Recipe templates show Python with HuggingFace ecosystem (transformers, datasets, peft, timm, albumentations, torchmetrics, seqeval, evaluate, ultralytics).

**Reference docs:**
- Spec: [`docs/superpowers/specs/2026-05-01-dl-skills-design.md`](../specs/2026-05-01-dl-skills-design.md)
- env.json schema: [`docs/superpowers/specs/dl-env-json-schema.md`](../specs/dl-env-json-schema.md)
- Phase 1 plan (template precedent): [`docs/superpowers/plans/2026-05-02-dl-skills-phase-1-foundation.md`](2026-05-02-dl-skills-phase-1-foundation.md)
- Phase 1 review lessons (encode in every Phase 2 skill from the start):
  - Include "Recipe template" + "Research hooks" sections per design spec.
  - Include explicit cross-references to sister skills (`dl-detect-env`, `dl-finetune-loop`, etc.).
  - Use checked subprocess calls instead of `os.system(...)` in Recipe template Python.
  - Use `weights_only=True` in any `torch.load()` call.
  - Include `WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()` idiom for any Python skeleton.
  - Decision rules and Hard constraints: full prose with explicit IF/THEN/NEVER/MUST.
  - No baked-in version pins, no benchmark numbers as facts.

**Phase 2 scope (recap from spec):**
- CV training (3): `dl-cv-classify`, `dl-cv-detect`, `dl-cv-segment`.
- CV evaluation (3): `dl-cv-eval-classify`, `dl-cv-eval-detect`, `dl-cv-eval-segment`.
- NLP training (2): `dl-nlp-classify`, `dl-nlp-token`.
- NLP evaluation (3): `dl-nlp-eval-classify`, `dl-nlp-eval-token`, `dl-nlp-eval-generative`.
- Cross-domain data + training (3): `dl-load-data`, `dl-augment`, `dl-finetune-loop`.

**Out of phase scope (deferred to Phase 3):** All LLM/VLM-specific skills (`dl-llm-*`, `dl-vlm-finetune`), `dl-pseudo-label`, `dl-distillation`, `dl-ensemble-tta`, `dl-cv-pretrain`.

---

## File Structure

**New files (15):**

```
skills/
├── dl-load-data/SKILL.md          (cross-domain — implement first; everything else depends on it)
├── dl-augment/SKILL.md            (cross-domain — depends on dl-load-data)
├── dl-finetune-loop/SKILL.md      (cross-domain — depends on dl-load-data + dl-experiment-track + dl-checkpoint)
├── dl-cv-classify/SKILL.md        (CV training — depends on dl-load-data + dl-augment + dl-finetune-loop)
├── dl-cv-detect/SKILL.md          (CV training)
├── dl-cv-segment/SKILL.md         (CV training)
├── dl-cv-eval-classify/SKILL.md   (CV eval)
├── dl-cv-eval-detect/SKILL.md     (CV eval)
├── dl-cv-eval-segment/SKILL.md    (CV eval)
├── dl-nlp-classify/SKILL.md       (NLP training)
├── dl-nlp-token/SKILL.md          (NLP training)
├── dl-nlp-eval-classify/SKILL.md  (NLP eval)
├── dl-nlp-eval-token/SKILL.md     (NLP eval)
├── dl-nlp-eval-generative/SKILL.md (NLP eval)
└── (no 15th — directory listing only counts 14 skill files)
```

**Modified files (5):**

```
.claude-plugin/plugin.json              (version 0.2.0-alpha.1 → 0.2.0-alpha.2)
agents/cv-engineer.md                   (Phase 1 limitation paragraph + skills table updated for Phase 2 availability)
agents/nlp-engineer.md                  (same as cv-engineer)
agents/llm-engineer.md                  (mention that dl-finetune-loop is now available as the generic fallback)
README.md                               (Deep learning support section: bump skill count and add Phase 2 status)
```

**Implementation order (dependency-driven):**
1. Cross-domain foundations first: `dl-load-data` → `dl-augment` → `dl-finetune-loop`. CV/NLP skills all reference these.
2. CV training: `dl-cv-classify` → `dl-cv-detect` → `dl-cv-segment`.
3. CV evaluation: `dl-cv-eval-classify` → `dl-cv-eval-detect` → `dl-cv-eval-segment`.
4. NLP training: `dl-nlp-classify` → `dl-nlp-token`.
5. NLP evaluation: `dl-nlp-eval-classify` → `dl-nlp-eval-token` → `dl-nlp-eval-generative`.
6. Wire-up: update sub-agents + version bump + README.
7. Cross-validation pass + final review.

---

## Pre-flight: Verify environment

- [ ] **Step 0.1: Confirm working directory and branch**

Run: `cd /Users/lokesh/Desktop/code/Bio_hacking/ML_Engineer && git status && git branch --show-current`
Expected: clean working tree on `phase-2-cv-nlp`, base commit is `e6224d0` (the spec backport) or later.

- [ ] **Step 0.2: Verify Phase 1 deliverables are present**

Run: `ls skills/dl-detect-env skills/dl-remote-execute skills/dl-prior-art agents/cv-engineer.md agents/nlp-engineer.md agents/llm-engineer.md`
Expected: all six paths exist (Phase 2 builds on these).

- [ ] **Step 0.3: Verify spec was backported**

Run: `grep -c "33 skills" docs/superpowers/specs/2026-05-01-dl-skills-design.md`
Expected: 2 or more matches.

Run: `grep -c "dl-cv-eval-classify\|dl-cv-eval-detect\|dl-cv-eval-segment" docs/superpowers/specs/2026-05-01-dl-skills-design.md`
Expected: at least 3 matches (one per eval sub-skill in the taxonomy table).

---

## Tasks

### Cross-domain foundations (Tasks 1-3)

These three are referenced by every CV and NLP skill that follows. Implement them first.

---

### Task 1: Create skill `dl-load-data`

**Files:**
- Create: `skills/dl-load-data/SKILL.md`

This skill loads training data for any DL task — HF datasets, image folders, webdataset, text corpora, jsonl. It absorbs tokenizer + max_length policy decisions for NLP tasks (the spec calls out "Tokenization is folded in here as one chapter of data prep"). For CV tasks, it covers ImageFolder, HF Dataset, and webdataset shards. It does NOT do augmentation — that's `dl-augment`'s job.

- [ ] **Step 1.1: Create the skill directory**

Run: `mkdir -p skills/dl-load-data`

- [ ] **Step 1.2: Write the SKILL.md**

Write to `skills/dl-load-data/SKILL.md`:

````markdown
---
name: dl-load-data
description: Use when starting any DL training or evaluation task — loads images, text, audio, or tabular data into a format the trainer expects. Picks the right loader (HF datasets, image folder, webdataset shards, jsonl, csv) by data shape. For NLP tasks, locks the tokenizer + max_length + padding policy. Do NOT use for the actual augmentation step (use dl-augment) or for one-off ad-hoc data inspection (use ml-engineer-write-code Layout A directly).
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# Load Data

Load DL training/eval data into the right format for downstream training. Pick the loader by data shape; lock tokenizer + padding policy for NLP; surface dataset size + class balance + token-length distribution to inform later decisions.

## When to invoke

- Start of any CV / NLP training or evaluation task, after EDA but before model construction.
- User uploads a dataset and asks "load this for finetuning".
- Switching from one dataset to another mid-project (re-invoke with the new path).

## When NOT to invoke

- Ad-hoc EDA (use `ml-engineer-write-code` Layout A + `ml-engineer-execute` directly).
- Augmentation setup (use `dl-augment`).
- Already loaded in this session AND no data change has occurred.

## Decision rules

Pick the loader by data shape:

- IF data is at a HF Hub repo path (e.g., `username/dataset-name`): use `datasets.load_dataset("repo")`.
- IF data is a local directory of class-named subfolders (`train/cat/img1.jpg`, `train/dog/img2.jpg`): use `datasets.load_dataset("imagefolder", data_dir=...)` for CV; use `datasets.load_dataset("text", data_files=...)` for NLP.
- IF data is jsonl with `{"text": ..., "label": ...}` rows: use `datasets.load_dataset("json", data_files=...)`.
- IF data is csv with text columns: use `datasets.load_dataset("csv", data_files=...)`. State the text column name explicitly.
- IF data is sharded webdataset tarballs (`.tar` files in a directory): use `webdataset.WebDataset` with appropriate decoders. Useful for large CV datasets that don't fit in memory.
- IF dataset is large (>10 GB) AND won't fit in RAM: stream with `datasets.load_dataset(..., streaming=True)`. Disables length and shuffling — flag this to the user.

For NLP tasks, also pick:

- IF tokenizer is bundled with the model (HF model id known): load via `AutoTokenizer.from_pretrained(model_id)`. NEVER use a tokenizer from a different model.
- IF max_length is unset: compute the 95th percentile of token lengths from a sample, round to nearest power of 2. Hard cap at 4096 unless model supports more AND user has VRAM headroom.
- IF padding side matters (decoder-only models often need left-padding): set explicitly based on model architecture.

For CV tasks, also pick:

- Image size: use the backbone's expected input size if known (timm `data_config["input_size"]`); else 224 for classification, 640 for detection (YOLO-family default), 512 for segmentation.
- Channel order: RGB by default. Check the backbone — some legacy models expect BGR.

## Process

### Step 1 — Inspect the data path

Read the directory layout / first few files to determine shape. Print:
- Total size on disk.
- File count by extension.
- For images: a 5-image sample with shape and dtype.
- For text: first 5 examples with text column + label column identified.
- For HF Hub: print `dataset.info` summary.

### Step 2 — Pick loader per decision rules

Apply the decision rules above. Report the chosen loader and rationale to the user.

### Step 3 — Build the dataloader

Generate dataloader-construction code that:
- Uses `datasets.load_dataset(...)` (or `webdataset.WebDataset(...)`).
- Applies `with_transform(...)` for per-batch transforms (NOT `map(...)` for augmentation — `map` runs once and caches; augmentation must run per epoch).
- Sets `num_workers` based on `os.cpu_count()` (rule of thumb: half the cores, capped at 8).
- Sets `pin_memory=True` only when CUDA is the active device (read from `<workdir>/env.json`).
- For NLP: applies the tokenizer with the chosen max_length and padding policy.

### Step 4 — Lock the policy

Save the chosen policy to `<workdir>/data_policy.json` so downstream skills (`dl-augment`, `dl-finetune-loop`, eval skills) can read consistent values:

```json
{
  "loader": "imagefolder|hf|json|csv|webdataset",
  "task_type": "image_classification|object_detection|segmentation|text_classification|token_classification|generative",
  "image_size": [224, 224],
  "channel_order": "rgb",
  "tokenizer_id": "answerdotai/ModernBERT-base",
  "max_length": 512,
  "padding_side": "right",
  "num_classes": 10,
  "num_workers": 4,
  "pin_memory": true
}
```

### Step 5 — Verify

Run a single dataloader iteration (one batch). Confirm shape, dtype, and label distribution match expectations. If anything is off, surface immediately — broken data loaders silently break training.

## Recipe template

The orchestrator adapts the template below per data shape. The skeleton below covers the HF Hub path; image folder + webdataset + jsonl variants follow the same structure with the loader call swapped.

### `<workdir>/src/_load_data.py`

```python
"""Load training/eval data into a Trainer-compatible format. Adapt the loader call
per data shape per the SKILL's decision rules.
"""
import json
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def load_image_classification(data_path: str, model_id: str | None = None, image_size: int = 224):
    from datasets import load_dataset
    from torchvision import transforms

    dataset = load_dataset("imagefolder", data_dir=data_path)
    num_classes = dataset["train"].features["label"].num_classes if "train" in dataset else None

    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def transform(batch):
        batch["pixel_values"] = [base_transform(img.convert("RGB")) for img in batch["image"]]
        return batch

    dataset = dataset.with_transform(transform)
    return dataset, num_classes


def load_text_classification(data_path: str, tokenizer_id: str, max_length: int = 512, text_column: str = "text"):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if data_path.endswith(".jsonl") or data_path.endswith(".json"):
        dataset = load_dataset("json", data_files=data_path)
    elif data_path.endswith(".csv"):
        dataset = load_dataset("csv", data_files=data_path)
    else:
        dataset = load_dataset(data_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.vocab_size != tokenizer.vocab_size:  # placeholder; real check is len(tokenizer) <= model.config.vocab_size
        pass

    def tokenize(batch):
        return tokenizer(batch[text_column], truncation=True, max_length=max_length, padding=False)

    dataset = dataset.map(tokenize, batched=True)
    return dataset, tokenizer


def save_policy(**kwargs):
    policy_path = WORKDIR / "data_policy.json"
    policy_path.write_text(json.dumps(kwargs, indent=2))
    print(f"Data policy written to {policy_path}")
```

## Hard constraints

- NEVER load augmentation transforms in this skill. Augmentation is `dl-augment`'s job; loaders use only the deterministic preprocess (resize / normalize / tokenize). Mixing concerns means re-running the loader to change augmentations.
- NEVER cache an augmented dataset to disk via `dataset.map(augment_fn)`. `map` runs once; augmentation must run per epoch via `with_transform` or a custom collator.
- NEVER load a tokenizer from a different `model_id` than the model. Mismatch silently produces fluent garbage at training time.
- NEVER set `streaming=True` without telling the user it disables shuffling and length info.
- NEVER assume the dataset has a `train` split. HF Hub datasets sometimes ship as a single split — handle both.
- NEVER skip Step 5 (single-batch verification). A broken dataloader looks the same as a working one until training crashes.

## Research hooks

Data formats and loader libraries evolve. Before finalizing the loader for an unfamiliar shape, invoke `ml-engineer-research`:

- **HF datasets streaming behavior.** Query: *"Current best practice for `datasets.load_dataset(streaming=True)` shuffling and resumption as of {today}."*
- **Tokenizer max_length conventions per model family.** Query: *"Current recommended max_length and padding side for `{model_family}` (e.g., ModernBERT, DeBERTa-v3, Llama 3) as of {today}."*
- **Webdataset best practices for large CV datasets.** Query: *"Current `webdataset.WebDataset` shuffling, decoding, and num_workers recommendations as of {today}."*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- `<workdir>/data_policy.json` exists, is valid JSON, and contains the keys for the chosen task type.
- A single batch was successfully loaded (Step 5 ran exit 0 with non-empty batch).
- For NLP: `len(tokenizer) <= model.config.vocab_size` (no embedding-vocab mismatch).
- For CV: pixel value range is sane (0-1 if normalized to that range; ~[-2, 2] if normalized with ImageNet mean/std).
- For multi-class tasks: the `num_classes` recorded in the policy matches the actual unique label count in the training set.

## Output checklist

- [ ] Data path inspected; loader chosen per decision rules
- [ ] Tokenizer + max_length + padding locked (NLP)
- [ ] Image size + channel order locked (CV)
- [ ] `<workdir>/data_policy.json` written
- [ ] Single-batch verification ran clean
- [ ] No augmentation in the loader (deferred to `dl-augment`)
````

- [ ] **Step 1.3: Verify frontmatter parses**

Run: `python3 -c "import yaml; yaml.safe_load(open('skills/dl-load-data/SKILL.md').read().split('---')[1])"`
Expected: no output, exit 0.

- [ ] **Step 1.4: Commit**

```bash
git add skills/dl-load-data/SKILL.md
git commit -m "Add dl-load-data skill: HF datasets / image folder / webdataset / jsonl loader with NLP tokenizer policy"
```

---

### Task 2: Create skill `dl-augment`

**Files:**
- Create: `skills/dl-augment/SKILL.md`

This skill picks the right augmentation library/recipe per use case (Q5 brainstorm decision: no baked-in default). For CV: albumentations + timm.data.mixup are the Kaggle stack. For NLP: back-translation + MLM noise + EDA-style perturbation, all conditional. The skill reads `<workdir>/data_policy.json` to know what task type it's augmenting.

- [ ] **Step 2.1: Create directory**

Run: `mkdir -p skills/dl-augment`

- [ ] **Step 2.2: Write the SKILL.md**

Write to `skills/dl-augment/SKILL.md`:

````markdown
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
  - **Allowed**: random masking of non-entity tokens during training (MLM-style). Skip if dataset is small.

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
POLICY = json.loads((WORKDIR / "data_policy.json").read_text())


def make_classification_augment(image_size: int = 224):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

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


def random_token_mask(text: str, mask_token: str = "[MASK]", p: float = 0.15) -> str:
    """Randomly mask tokens (MLM-style). Useful for token classification with non-entity tokens."""
    words = text.split()
    return " ".join(mask_token if random.random() < p else w for w in words)
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
````

- [ ] **Step 2.3: Verify frontmatter parses**

Run: `python3 -c "import yaml; yaml.safe_load(open('skills/dl-augment/SKILL.md').read().split('---')[1])"`

- [ ] **Step 2.4: Commit**

```bash
git add skills/dl-augment/SKILL.md
git commit -m "Add dl-augment skill: per-use-case CV/NLP augmentation with bbox/mask awareness"
```

---

### Task 3: Create skill `dl-finetune-loop`

**Files:**
- Create: `skills/dl-finetune-loop/SKILL.md`

This skill picks Trainer vs Accelerate based on task complexity (Q3 brainstorm decision). It's the generic finetune loop that CV/NLP both use, and that LLM uses as fallback. Reads `<workdir>/data_policy.json` for batch shape; reads `<workdir>/env.json` for device.

- [ ] **Step 3.1: Create directory**

Run: `mkdir -p skills/dl-finetune-loop`

- [ ] **Step 3.2: Write the SKILL.md**

Write to `skills/dl-finetune-loop/SKILL.md`:

````markdown
---
name: dl-finetune-loop
description: Use to construct the training loop for any DL task (CV, NLP, LLM, VLM) once data is loaded, augmentation is wired, env is detected, tracking + checkpointing are configured. Picks HF Trainer (default for standard finetune) vs Accelerate (custom loop with bespoke loss / multi-task / manual gradient handling) based on task complexity. Do NOT use for inference-only scripts or for evaluation harnesses (use the relevant dl-{cv,nlp,llm}-eval-* skill).
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# Finetune Loop

Construct the training loop. Decide between HF Trainer (high-level, handles 90% of cases) and Accelerate (low-level, for custom loops). Wire mixed precision, gradient accumulation, lr scheduling, and the callbacks for tracking and checkpointing — but DO NOT do those things itself; those are owned by `dl-experiment-track` and `dl-checkpoint` already.

## When to invoke

- After data is loaded (`dl-load-data`), augmentation is wired (`dl-augment`), env is known (`dl-detect-env`), tracking is wired (`dl-experiment-track`), checkpointing is wired (`dl-checkpoint`).
- The task is a "standard" finetune of a HF-compatible model (CV, NLP, encoder/decoder LLM).
- User asks for a "training loop" or "finetune".

## When NOT to invoke

- Inference-only scripts (use `dl-llm-serve` or write inference directly).
- Evaluation harnesses (use `dl-cv-eval-*` or `dl-nlp-eval-*`).
- LLM finetune that already uses Unsloth (Unsloth ships its own loop; `dl-llm-lora` handles).

## Decision rules

Pick Trainer vs Accelerate based on the task's needs:

- **HF Trainer** is the default. Use it WHEN:
  - Standard supervised loss (CrossEntropy, MSE, etc.) handled by the model's `forward(labels=...)`.
  - Standard eval loop (predict on val set, compute metric, log).
  - Standard callbacks (logging, eval, checkpointing).
  - FSDP via `TrainingArguments(fsdp=...)` is sufficient for distributed.

- **Accelerate** (custom loop with `Accelerator.prepare()`). Use it WHEN:
  - Loss combines multiple terms with custom weighting.
  - Multi-task training with separate forward passes.
  - Manual gradient manipulation (e.g., GradCache, gradient surgery).
  - Custom batch construction outside the dataloader.
  - User explicitly requests "custom training loop" or "raw PyTorch".

- IF the task is genuinely simple (single loss, single eval, single dataloader): default to **Trainer**. Do NOT introduce Accelerate complexity unless the task needs it.

- IF Unsloth is the chosen LoRA path (`dl-llm-lora` decided so): do NOT invoke this skill. Unsloth ships its own loop.

## Process

### Step 1 — Determine task complexity

Read `<workdir>/data_policy.json` and `<workdir>/env.json`. Determine:
- Task type (single-task vs multi-task).
- Loss shape (single vs combined).
- Distributed (single-GPU vs FSDP / DeepSpeed — read from `dl-distributed`'s output).
- Whether a custom loop is needed (user request, multi-task, custom loss).

### Step 2 — Apply decision rules

Pick Trainer or Accelerate. Report the choice with one-sentence rationale.

### Step 3 — Generate the training loop

For **HF Trainer** path: generate a `train.py` that:
- Loads model via `AutoModelFor{TaskType}.from_pretrained(model_id)`.
- Constructs `TrainingArguments` with lr, batch size, grad_accum, scheduler, mixed precision.
- Wires the tracker callback (already configured by `dl-experiment-track`).
- Wires checkpointing (`save_strategy="steps"`, `save_steps`, `save_total_limit` — already configured by `dl-checkpoint`).
- Constructs `Trainer(model, args, train_dataset, eval_dataset, tokenizer/data_collator, compute_metrics)`.
- Calls `trainer.train()` and `trainer.save_model()`.

For **Accelerate** path: generate a `train.py` that:
- `accelerator = Accelerator(mixed_precision="bf16")`.
- `model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)`.
- Manual training loop with `accelerator.backward(loss)`, `accelerator.log({...})`.
- Manual checkpoint save via `accelerator.save_state(...)`.

### Step 4 — Wire mixed precision

Pick the precision based on hardware (read from `<workdir>/env.json`):
- A100, H100, RTX 30xx/40xx → `bf16=True`.
- T4, V100 → `fp16=True`.
- CPU/MPS → `fp16=False, bf16=False` (or use bf16 on MPS if available; otherwise stay fp32).

For Trainer: set in `TrainingArguments(bf16=True)` or `fp16=True`.
For Accelerate: pass to `Accelerator(mixed_precision="bf16")`.

### Step 5 — Verify

Run a 10-step smoke test on the training data. Confirm:
- Loss decreases (or at least changes — flat loss = broken).
- No OOM, no NaN.
- Tracker shows step count + metrics.
- Checkpoint dir starts to populate.

If anything fails, hand off to `dl-debug-training` with the failure context.

## Recipe template

### `<workdir>/src/_train_trainer.py` (HF Trainer path)

```python
"""HF Trainer training loop. Adapt model/dataset/metric per task."""
import json
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()
ENV = json.loads((WORKDIR / "env.json").read_text())
POLICY = json.loads((WORKDIR / "data_policy.json").read_text())

ACTIVE_ENV = ENV["environments"][ENV["active"]]
DEVICE = ACTIVE_ENV.get("device", "cpu")


def pick_precision():
    """Pick bf16 vs fp16 based on the active device."""
    if DEVICE == "cuda":
        return {"bf16": True}  # safe default for A100/H100/30xx/40xx
    if DEVICE == "mps":
        return {"bf16": False, "fp16": False}  # MPS bf16 support is uneven; default to fp32
    return {"bf16": False, "fp16": False}


def make_training_args(output_dir: str | None = None, **base_kwargs) -> "TrainingArguments":
    from transformers import TrainingArguments
    output_dir = output_dir or str(WORKDIR / "checkpoints")
    return TrainingArguments(
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=base_kwargs.pop("save_steps", 100),
        save_total_limit=3,
        save_safetensors=True,
        logging_steps=base_kwargs.pop("logging_steps", 10),
        evaluation_strategy=base_kwargs.pop("evaluation_strategy", "steps"),
        eval_steps=base_kwargs.pop("eval_steps", 100),
        report_to=base_kwargs.pop("report_to", ["wandb"]),
        run_name=base_kwargs.pop("run_name", WORKDIR.name),
        **pick_precision(),
        **base_kwargs,
    )


def run_smoke_test(trainer, max_steps: int = 10):
    """Run a tiny number of steps to confirm the loop works before committing to a full run."""
    original_max_steps = trainer.args.max_steps
    trainer.args.max_steps = max_steps
    trainer.train()
    trainer.args.max_steps = original_max_steps
    print(f"Smoke test: {max_steps} steps completed without error.")
```

### `<workdir>/src/_train_accelerate.py` (Accelerate custom loop)

```python
"""Accelerate custom training loop. Use when standard Trainer can't express the task."""
import json
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def make_accelerator():
    from accelerate import Accelerator
    env = json.loads((WORKDIR / "env.json").read_text())
    device = env["environments"][env["active"]].get("device", "cpu")
    precision = "bf16" if device == "cuda" else "no"
    return Accelerator(mixed_precision=precision, log_with="wandb")


def train_loop(accelerator, model, optimizer, scheduler, train_loader, eval_loader, num_epochs, compute_loss):
    """Generic Accelerate loop. compute_loss(model, batch) -> scalar tensor."""
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )
    accelerator.init_trackers(WORKDIR.name)

    step = 0
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                loss = compute_loss(model, batch)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step % 10 == 0:
                accelerator.log({"train/loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=step)
            step += 1

    accelerator.end_training()
```

## Hard constraints

- NEVER use `Trainer` AND `Accelerator.prepare(model)` together. They're mutually exclusive code paths; combining will produce subtle distributed bugs.
- NEVER skip the smoke test (Step 5). A 10-step smoke is cheap; a 10-hour broken training run is not.
- NEVER set `bf16=True` on hardware that does not support bf16 (T4, V100, pre-Volta GPUs). Use `fp16=True` instead.
- NEVER mix `bf16` and `fp16` flags. Pick one.
- NEVER re-implement what `dl-experiment-track` and `dl-checkpoint` already provide. Wire their outputs in; do not duplicate.
- NEVER set `report_to=[]` silently. If user declined tracking, set `report_to=[]` AND surface a `[no tracking]` banner.

## Research hooks

Trainer/Accelerate APIs evolve with each HF release. Before generating the loop for an unfamiliar combination, invoke `ml-engineer-research`:

- **TrainingArguments deprecations.** Query: *"Recently deprecated `TrainingArguments` parameters in HF Transformers as of {today} (e.g., `evaluation_strategy` → `eval_strategy`)."*
- **Accelerate prepare semantics.** Query: *"Current `Accelerator.prepare()` ordering and side-effects for distributed training as of {today}."*
- **Mixed precision recommendations per GPU.** Query: *"Current bf16 vs fp16 recommendation for `{gpu_class}` as of {today}."*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- A training script exists at `<workdir>/src/train.py` (or wherever the orchestrator placed it).
- The script imports the right Trainer/Accelerator class per the chosen path.
- The script wires `dl-experiment-track`'s init AND finish calls.
- The script wires `dl-checkpoint`'s `save_strategy` / `save_steps` / `save_total_limit`.
- A 10-step smoke test ran without OOM / NaN / Inf.
- For Trainer path: `report_to` is set (either to a real tracker OR `[]` with a `[no tracking]` banner).

## Output checklist

- [ ] Task complexity assessed; Trainer or Accelerate chosen
- [ ] Training script generated in `<workdir>/src/train.py`
- [ ] Mixed precision picked per active device
- [ ] Tracker + checkpoint wired (not duplicated)
- [ ] 10-step smoke test ran clean
- [ ] No mutual-exclusion violations (Trainer + Accelerate, bf16 + fp16, Unsloth + this skill)
````

- [ ] **Step 3.3: Verify frontmatter parses**

Run: `python3 -c "import yaml; yaml.safe_load(open('skills/dl-finetune-loop/SKILL.md').read().split('---')[1])"`

- [ ] **Step 3.4: Commit**

```bash
git add skills/dl-finetune-loop/SKILL.md
git commit -m "Add dl-finetune-loop skill: Trainer-vs-Accelerate selector with smoke test gate"
```

---

### Task 4: Create skill `dl-cv-classify`

**Files:**
- Create: `skills/dl-cv-classify/SKILL.md`

Image classification finetune via timm backbones (or HF AutoModelForImageClassification). Standard supervised loss; uses `dl-finetune-loop` Trainer path.

- [ ] **Step 4.1: Create directory and write the SKILL.md**

Run: `mkdir -p skills/dl-cv-classify`

Write to `skills/dl-cv-classify/SKILL.md`:

````markdown
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
````

- [ ] **Step 4.2: Verify frontmatter parses**

Run: `python3 -c "import yaml; yaml.safe_load(open('skills/dl-cv-classify/SKILL.md').read().split('---')[1])"`

- [ ] **Step 4.3: Commit**

```bash
git add skills/dl-cv-classify/SKILL.md
git commit -m "Add dl-cv-classify skill: timm backbone finetune with prior-art recommendation"
```

---

### Tasks 5-14: Remaining 11 skills

Following the same pattern as Task 4. Each task creates one skill file with the same body structure (frontmatter / When / NotWhen / Decision rules / Process / Recipe template / Hard constraints / Research hooks / Verification gates / Output checklist), commits with a clear message, and verifies frontmatter parses.

The remaining skills:

- **Task 5: `dl-cv-detect`** — Object detection. NO baked-in backbone (Q4 brainstorm decision); always asks user at runtime after `dl-prior-art` recommendation. Recipe template covers YOLO11 (ultralytics) and RT-DETR (HF transformers) skeletons. Bbox-aware augmentation handoff to `dl-augment`.
  - Commit: `Add dl-cv-detect skill: detection finetune with runtime backbone selection`

- **Task 6: `dl-cv-segment`** — Semantic / instance / panoptic segmentation. SAM2/3 zero-shot, YOLO-seg for real-time, U-Net family for medical. Recipe template covers each. Mask-aware augmentation handoff.
  - Commit: `Add dl-cv-segment skill: SAM/YOLO-seg/U-Net family with mask-aware augmentation`

- **Task 7: `dl-cv-eval-classify`** — top-k accuracy, per-class F1, confusion matrix, ECE calibration. Uses torchmetrics + sklearn. Saves charts to `<workdir>/charts/`.
  - Commit: `Add dl-cv-eval-classify skill: top-k accuracy + confusion matrix + ECE`

- **Task 8: `dl-cv-eval-detect`** — mAP@[.5:.95], mAP-50, mAP-75, per-class AP via pycocotools. Saves COCO-format predictions and annotations.
  - Commit: `Add dl-cv-eval-detect skill: COCO-style mAP via pycocotools`

- **Task 9: `dl-cv-eval-segment`** — mean IoU, Dice, Hausdorff via torchmetrics + medpy (for Hausdorff). Per-class breakdown.
  - Commit: `Add dl-cv-eval-segment skill: IoU/Dice/Hausdorff with per-class breakdown`

- **Task 10: `dl-nlp-classify`** — Encoder finetune via HF AutoModelForSequenceClassification. ModernBERT / DeBERTa-v3 default suggestions; user confirms via prior art. Multi-label via BCEWithLogits.
  - Commit: `Add dl-nlp-classify skill: encoder finetune with ModernBERT/DeBERTa default`

- **Task 11: `dl-nlp-token`** — Token classification (NER, extractive QA). HF AutoModelForTokenClassification. BIO alignment verification mandatory in pre-train check.
  - Commit: `Add dl-nlp-token skill: NER/token classification with BIO alignment verification`

- **Task 12: `dl-nlp-eval-classify`** — accuracy, macro/micro F1, MCC, calibration via sklearn + torchmetrics.
  - Commit: `Add dl-nlp-eval-classify skill: F1/MCC/calibration for NLP classification`

- **Task 13: `dl-nlp-eval-token`** — span-F1 via seqeval, entity-level precision/recall, exact-match.
  - Commit: `Add dl-nlp-eval-token skill: seqeval span-F1 for NER/token tasks`

- **Task 14: `dl-nlp-eval-generative`** — ROUGE-1/2/L, BLEU, BERTScore, perplexity via `evaluate` library + bert-score.
  - Commit: `Add dl-nlp-eval-generative skill: ROUGE/BLEU/BERTScore/perplexity`

Each task has the same internal structure as Task 4 (and the Phase 1 skill tasks). The orchestrator dispatching subagents will be given the full template content per task at dispatch time, mirroring how Phase 1 was executed.

---

### Task 15: Update sub-agents to mark Phase 2 skills as available

**Files:**
- Modify: `agents/cv-engineer.md`
- Modify: `agents/nlp-engineer.md`
- Modify: `agents/llm-engineer.md`

For each sub-agent, update the Phase 1 limitation paragraph to reflect that Phase 2 skills are now available, and remove `(Phase 2)` annotations from the skills table where the skill now exists.

- [ ] **Step 15.1: Update cv-engineer.md**

In `agents/cv-engineer.md`, update the "Phase 1 limitation" section. Change:
> In Phase 1, only the infra skills ... CV-specific skills (`dl-cv-classify`, `dl-cv-detect`, `dl-cv-segment`, `dl-cv-eval`, `dl-load-data`, `dl-augment`, `dl-finetune-loop`) ship in Phase 2. ...

REPLACE with:
> **Phase 2 status (this release):** CV training (`dl-cv-classify`, `dl-cv-detect`, `dl-cv-segment`), CV evaluation (`dl-cv-eval-classify`, `dl-cv-eval-detect`, `dl-cv-eval-segment`), data loading (`dl-load-data`), augmentation (`dl-augment`), and the generic finetune loop (`dl-finetune-loop`) are now available. Phase 3 will add cross-domain extras (`dl-pseudo-label`, `dl-distillation`, `dl-cv-pretrain`, `dl-ensemble-tta`).

Also remove `(Phase 2)` annotations from the skills table for the now-available skills.

Also update `dl-cv-eval` references in the loop and skills table to point to the appropriate sub-skill (`dl-cv-eval-classify` / `dl-cv-eval-detect` / `dl-cv-eval-segment`).

- [ ] **Step 15.2: Update nlp-engineer.md**

Same pattern. Phase 2 status: `dl-nlp-classify`, `dl-nlp-token`, `dl-nlp-eval-classify`, `dl-nlp-eval-token`, `dl-nlp-eval-generative`, plus cross-domain `dl-load-data`, `dl-augment`, `dl-finetune-loop`. Phase 3 deferred: `dl-pseudo-label`, `dl-distillation`, `dl-ensemble-tta`.

Update `dl-nlp-eval` references to point to the appropriate sub-skill.

- [ ] **Step 15.3: Update llm-engineer.md**

LLM sub-agent gets `dl-finetune-loop` as the generic Trainer fallback (Phase 2). Phase 3 still deferred for all `dl-llm-*` and `dl-vlm-finetune`.

- [ ] **Step 15.4: Verify all three frontmatters still parse**

```bash
for f in agents/cv-engineer.md agents/nlp-engineer.md agents/llm-engineer.md; do
  python3 -c "import yaml; yaml.safe_load(open('$f').read().split('---')[1])" || echo "FAIL $f"
done
```

- [ ] **Step 15.5: Commit**

```bash
git add agents/cv-engineer.md agents/nlp-engineer.md agents/llm-engineer.md
git commit -m "Sub-agents: mark Phase 2 skills as available, point to eval sub-skills"
```

---

### Task 16: Bump plugin version to 0.2.0-alpha.2

**Files:**
- Modify: `.claude-plugin/plugin.json`

- [ ] **Step 16.1: Update version**

Edit `.claude-plugin/plugin.json`. Change `"version": "0.2.0-alpha.1"` to `"version": "0.2.0-alpha.2"`.

- [ ] **Step 16.2: Validate JSON**

Run: `python3 -c "import json; json.load(open('.claude-plugin/plugin.json'))"`

- [ ] **Step 16.3: Commit**

```bash
git add .claude-plugin/plugin.json
git commit -m "Bump plugin version to 0.2.0-alpha.2 for DL skills phase 2"
```

---

### Task 17: Update README for Phase 2

**Files:**
- Modify: `README.md`

- [ ] **Step 17.1: Update directory tree**

In the "What's in here" section, expand the `dl-*` skills enumeration to include all 14 Phase 2 skills (in addition to the 7 Phase 1 skills already listed). Total dl-* count: 21. Total taxonomy target: 33.

- [ ] **Step 17.2: Add Phase 2 status section**

After the "Deep learning support (v0.2.0-alpha.1, Phase 1)" section, add a new "Phase 2 status" subsection (or update the existing one to mention both phases shipped). List all 14 new skills with one-line descriptions.

- [ ] **Step 17.3: Commit**

```bash
git add README.md
git commit -m "README: add Phase 2 status section, update directory tree"
```

---

### Task 18: Cross-validation pass

Validate the entire Phase 2 deliverable hangs together:

- [ ] All 14 skill files exist with valid frontmatter.
- [ ] Cross-references between skills resolve to existing skill names.
- [ ] `data_policy.json` consumers (`dl-augment`, `dl-finetune-loop`, eval skills) all read it consistently.
- [ ] No baked-in version pins.
- [ ] Sub-agent updates correctly reflect Phase 2 availability.
- [ ] Plugin version bumped.
- [ ] README updated.

Run the same validation pattern as Phase 1 Task 14.

---

### Task 19: Phase 2 completion review

Final end-of-phase review:

- [ ] Code reviewer: phase-wide review (coherence, drift, scope, risk register).
- [ ] Verdict must be `release` or `release-with-caveats`.
- [ ] Any Important issues addressed in a finalization commit.
- [ ] Push `phase-2-cv-nlp` to remote.

---

## Acceptance criteria for Phase 2 (recap from spec)

After Phase 2, the plugin can:

- [ ] Take an image classification task end-to-end: `dl-load-data` → `dl-augment` → `dl-cv-classify` → `dl-finetune-loop` → `dl-cv-eval-classify`.
- [ ] Take an object detection task end-to-end with runtime backbone selection.
- [ ] Take an NER task end-to-end: `dl-load-data` → `dl-nlp-token` → `dl-finetune-loop` → `dl-nlp-eval-token`.
- [ ] LLM tasks still route to `llm-engineer` but now use `dl-finetune-loop` as the generic Trainer fallback (Phase 3 will add LLM-specific recipes).

What it CANNOT yet do (deferred to Phase 3):
- LLM-specific recipes (Unsloth/Axolotl LoRA, DPO/GRPO, mergekit, AWQ).
- VLM finetuning.
- Pseudo-labeling, distillation, self-supervised pretraining.
- Ensembling + TTA.

This is the expected Phase 2 surface. Phase 3 plan will be written after Phase 2 lands.
