---
name: dl-llm-merge
description: Use to merge two or more finetuned LLMs into a single model via mergekit (SLERP for 2-model interpolation, TIES for 3+ models with task arithmetic, DARE-TIES for further dropout-based generalization). Cheap relative to training and often top-of-leaderboard per spec. Do NOT use for adapter merging into base (use peft.merge_and_unload), tokenizer surgery, or different-architecture combinations.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# LLM Merge

Merge 2+ finetuned LLMs of the same architecture into one. Pick method (SLERP / TIES / DARE-TIES) by model count and goal. Use mergekit's yaml-driven interface.

## When to invoke

- User has 2+ finetuned LLMs of the same architecture and wants to combine.
- User asks to "merge", "ensemble at weight level", "model soup", "ties merge", "slerp".

## When NOT to invoke

- Merging a LoRA adapter into its base model — use `peft.merge_and_unload` directly (single-line).
- Tokenizer / vocab surgery (different concern; not mergekit's strength).
- Combining models of different architectures (Llama + Mistral) — won't work; weights aren't compatible.

## Decision rules

### Method

- **SLERP (Spherical Linear Interpolation)**: 2-model interpolation. Default for 2 models. Preserves direction better than linear interpolation. Hyperparameter: `t` (interpolation factor 0-1).
- **TIES (Task arithmetic with sign Trim, Elect, Disjoint merge)**: 3+ models. Resolves sign conflicts; trims small-magnitude updates. Default for >2 models.
- **DARE-TIES**: TIES + random dropout of small updates. Better generalization on some tasks. Use when prior-art surfaces it; 2024-2025 SOTA on some leaderboards.
- **Linear (model soup)**: simplest; equal weighting. Often beaten by SLERP/TIES.

### Base model

mergekit needs a `base_model` reference. Use the common ancestor (the pretrained base both finetunes started from). If finetunes started from different bases — can't merge cleanly.

### Density / weight tuning

TIES / DARE-TIES have `density` (fraction of params kept) and `weight` (relative contribution per source model). Defaults: density 0.5-0.7, weight 1.0 each. Tune via prior-art recommendations.

## Process

### Step 1 — Verify same architecture + base

Read each source model's `config.json`. Confirm `model_type`, `architectures`, `hidden_size`, `num_hidden_layers` all match. If any mismatch — halt and ask user.

### Step 2 — Consult prior art

Invoke `dl-prior-art` for "what merge method/density worked for similar problems". For HF leaderboard, mergekit recipes are widely shared.

### Step 3 — Pick method + write yaml

Apply decision rules. Save mergekit yaml to `<workdir>/configs/merge.yaml`.

### Step 4 — Run mergekit

`mergekit-yaml <workdir>/configs/merge.yaml <workdir>/merged_model/`. Mergekit handles loading + merging + saving.

### Step 5 — Eval the merge

ALWAYS hand off to `dl-llm-eval` to confirm the merge didn't degrade. Compare against the source models' individual scores.

## Recipe template

### `<workdir>/configs/slerp.yaml` (2-model SLERP)

```yaml
slices:
  - sources:
      - model: <model_a_path_or_hf_id>
        layer_range: [0, 32]
      - model: <model_b_path_or_hf_id>
        layer_range: [0, 32]
merge_method: slerp
base_model: <model_a_path_or_hf_id>
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5
dtype: bfloat16
```

### `<workdir>/configs/ties.yaml` (3+ model TIES)

```yaml
models:
  - model: <model_a>
    parameters:
      density: 0.5
      weight: 0.5
  - model: <model_b>
    parameters:
      density: 0.5
      weight: 0.3
  - model: <model_c>
    parameters:
      density: 0.5
      weight: 0.2
merge_method: ties
base_model: <common_base_path_or_hf_id>
parameters:
  normalize: true
  int8_mask: true
dtype: bfloat16
```

### `<workdir>/configs/dare_ties.yaml` (DARE-TIES)

```yaml
models:
  - model: <model_a>
    parameters:
      density: 0.53
      weight: 0.5
  - model: <model_b>
    parameters:
      density: 0.53
      weight: 0.5
merge_method: dare_ties
base_model: <common_base>
parameters:
  int8_mask: true
dtype: bfloat16
```

### `<workdir>/src/_run_merge.py`

```python
"""Wrapper around mergekit CLI."""
import os
import subprocess
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def run_mergekit(yaml_path: str, output_dir: str | None = None,
                  copy_tokenizer: bool = True, lazy_unpickle: bool = False):
    """Run mergekit-yaml CLI. Returns the path to the merged model."""
    out = output_dir or str(WORKDIR / "merged_model")
    cmd = ["mergekit-yaml", yaml_path, out]
    if copy_tokenizer:
        cmd.append("--copy-tokenizer")
    if lazy_unpickle:
        cmd.append("--lazy-unpickle")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(result.stdout)
    if result.stderr:
        print(f"[mergekit stderr] {result.stderr}")
    return out


def verify_merge(merged_path: str, source_paths: list[str]) -> dict:
    """Sanity-check the merged model: file exists, config compatible."""
    import json
    merged_config = json.loads(Path(merged_path, "config.json").read_text())
    source_configs = [json.loads(Path(p, "config.json").read_text()) for p in source_paths]
    arch_match = all(c.get("model_type") == merged_config.get("model_type") for c in source_configs)
    return {
        "merged_path": merged_path,
        "merged_model_type": merged_config.get("model_type"),
        "all_sources_arch_match": arch_match,
    }
```

## Hard constraints

- NEVER merge models with different tokenizers. Vocab IDs are not interchangeable; merging produces nonsense outputs.
- NEVER merge models with different architectures (Llama-7B + Mistral-7B). Layer shapes won't match; mergekit will error or silently corrupt.
- ALWAYS evaluate the merge with `dl-llm-eval` after merging. Merging can silently degrade if weights conflict — sometimes by 5-15%.
- NEVER specify a `base_model` that wasn't an ancestor of the finetunes. The "base" should be the pretrained model both/all source finetunes started from.
- NEVER skip the architecture-match verification (Step 1). A 2-minute config-diff check prevents 30 minutes of mergekit running before erroring.

## Research hooks

- **Current mergekit method options.** Query: *"Latest mergekit-supported merge methods (slerp, ties, dare_ties, dare_linear, breadcrumbs, model_stock, etc.) and which is winning HF Open LLM Leaderboard as of {today}."*
- **TIES density / weight defaults per family.** Query: *"Recommended TIES `density` and `weight` parameters for `{model_family}` merges as of {today}."*
- **DARE-TIES vs TIES quality gap.** Query: *"Latest measurement of DARE-TIES vs TIES on `{benchmark_subset}` as of {today}."*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- All source models share the same `model_type`, `hidden_size`, `num_hidden_layers`, `vocab_size`.
- `<workdir>/configs/<method>.yaml` exists.
- `<workdir>/merged_model/` exists with `config.json`, `tokenizer.json`, weight shards.
- Merged model loads cleanly via `AutoModelForCausalLM.from_pretrained(<merged_path>)`.
- A small eval run via `dl-llm-eval` was performed; results saved.
- Merged eval score is within ±5% of the BEST source model's score on the same benchmark — if much worse, the merge is degraded; flag for user.

## Output checklist

- [ ] All source models share architecture + base ancestor
- [ ] `dl-prior-art` consulted for method/density recommendation
- [ ] mergekit yaml saved
- [ ] Merge ran successfully
- [ ] Merged model loads
- [ ] Eval ran via `dl-llm-eval`; compared to source models
- [ ] User informed if merge degraded vs source best
