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
- Prerequisite: `dl-detect-env` should have run first (writes `<workdir>/env.json`); this skill reads it to set `pin_memory` and `num_workers`. If `env.json` is missing, default `pin_memory=False` and `num_workers=2`, and warn the user to run `dl-detect-env`.

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
    # Vocab-size check happens in dl-finetune-loop after model is loaded;
    # at load time we do not yet have model.config.vocab_size.

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
- NEVER cache an augmented (stochastic) dataset to disk via `dataset.map(augment_fn)`. `map` runs once and freezes its output; stochastic augmentation must run per epoch via `with_transform` or a custom collator. Deterministic preprocessing (tokenization, normalize-to-tensor, resize) MAY use `dataset.map` since the result is identical every epoch.
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
