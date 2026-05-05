---
name: dl-llm-eval
description: Use to evaluate an LLM on academic benchmarks (MMLU / HellaSwag / ARC / TruthfulQA / GSM8K / BBH) via lm-evaluation-harness OR HF leaderboard subset via lighteval, with vLLM or SGLang as the inference backend. Internal mode selector — one skill, three modes. Do NOT use for small custom NLP eval (use dl-nlp-eval-* from Phase 2) or for LLM-as-judge subjective quality scoring.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# LLM Eval

Run academic-benchmark evaluation on a finetuned LLM. Pick library (lm-eval-harness for breadth, lighteval for HF-leaderboard speed, custom for user-rubric). Pick backend (vLLM default, SGLang for RAG/agents). Save metrics + per-task breakdown.

## When to invoke

- After SFT (`dl-llm-instruction-tune`) or preference tuning (`dl-llm-pref-opt`).
- User asks to "evaluate", "benchmark", "run MMLU/HellaSwag/etc", or "compare to baseline".

## When NOT to invoke

- Small custom NLP classification / NER / generative eval — use `dl-nlp-eval-{classify,token,generative}` from Phase 2.
- LLM-as-judge subjective scoring (out of v1 scope).
- Reward modeling for RLHF (different concern; lives in `dl-llm-pref-opt`'s GRPO mode).

## Decision rules

### Library

- **lm-evaluation-harness (default for academic benchmarks)**: MMLU, HellaSwag, ARC, TruthfulQA, GSM8K, BBH, ARC-Challenge, WinoGrande, etc. Slowest but most comprehensive.
- **lighteval (default for HF leaderboard subset)**: faster than lm-eval-harness on the leaderboard subset; integrates cleanly with vLLM. Use when iterating quickly on a known suite.
- **custom (when user has a rubric)**: write a custom eval harness; out of recipe scope here — just hand off.

### Backend

- **vLLM (default)**: high throughput, broad model support, good for any benchmark.
- **SGLang**: RadixAttention prefix caching — faster on multi-turn / RAG / agent benchmarks. Use when benchmark involves shared prefixes (long system prompts).

### Benchmark suite

NEVER cherry-pick. Standard small suites:

- **Reasoning suite**: MMLU, ARC-Challenge, HellaSwag, WinoGrande, TruthfulQA, GSM8K (= the original HF leaderboard).
- **Code suite**: HumanEval, MBPP.
- **Math suite**: GSM8K, MATH.

## Process

### Step 1 — Pick library + backend per decision rules

Surface choice. User can override.

### Step 2 — Pick benchmark suite

Default to leaderboard suite for general LLMs; specialty suite if user has specialty model. Surface; user confirms.

### Step 3 — Run eval

Use the chosen library's CLI or API. Stream progress.

### Step 4 — Save metrics

`<workdir>/metrics_eval.json` with per-task scores AND aggregate. Include backend version + library version + model adapter checksum (so the metric is reproducible).

### Step 5 — Surface insights

Print headline (MMLU 5-shot, GSM8K, etc.) + worst-task breakdown.

## Recipe template

### `<workdir>/src/_eval_lm_harness.py`

```python
"""lm-evaluation-harness driver via Python API. Use for academic benchmarks."""
import json
import os
import subprocess
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def run_lm_harness(model_path: str, tasks: list[str], num_fewshot: int = 5,
                    backend: str = "vllm", batch_size: str | int = "auto") -> dict:
    """Run lm-eval CLI; parse JSON output."""
    out_dir = WORKDIR / "lm_eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "lm_eval",
        "--model", backend,
        "--model_args", f"pretrained={model_path}",
        "--tasks", ",".join(tasks),
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--output_path", str(out_dir),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Find latest results JSON
    results_files = sorted(out_dir.rglob("results_*.json"))
    if results_files:
        return json.loads(results_files[-1].read_text())
    return {}


def save_eval_metrics(metrics: dict, library: str, backend: str, model_path: str):
    """Save metrics with full reproducibility metadata."""
    import hashlib
    out = WORKDIR / "metrics_eval.json"
    record = {
        "library": library,
        "backend": backend,
        "model_path": str(model_path),
        "model_path_hash": hashlib.sha256(str(model_path).encode()).hexdigest()[:16],
        "metrics": metrics,
    }
    out.write_text(json.dumps(record, indent=2, default=str))
    print(f"Eval metrics saved to {out}")
```

### `<workdir>/src/_eval_lighteval.py`

```python
"""lighteval Python driver. Faster on HF leaderboard subset."""
import os
import subprocess
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def run_lighteval(model_path: str, tasks: str = "leaderboard", backend: str = "vllm") -> dict:
    """tasks: 'leaderboard' (HF leaderboard subset) or 'lighteval|<task>' for individual."""
    out_dir = WORKDIR / "lighteval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "lighteval", backend,
        "--model_args", f"pretrained={model_path}",
        "--tasks", tasks,
        "--output_dir", str(out_dir),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
```

## Hard constraints

- NEVER report eval scores without naming the exact benchmark version. MMLU has multiple variants (original, MMLU-Pro, MMLU-Redux); not interchangeable.
- NEVER eval on training data. Even one accidental overlap inflates scores.
- ALWAYS pin the backend version (vLLM / SGLang). Generation determinism varies across versions; scores from different versions are not comparable.
- NEVER cherry-pick benchmarks. Pick a small fixed suite and report all of it. Selective reporting is misleading.
- NEVER skip the metric-reproducibility metadata (library version, backend version, model checksum). Eval scores without provenance can't be replicated.
- NEVER report eval metrics without comparison to a baseline (the pre-finetune model OR a known-good reference checkpoint).

## Research hooks

- **Current lm-evaluation-harness task list.** Query: *"Latest lm-evaluation-harness supported tasks and any deprecated/renamed tasks as of {today}."*
- **lighteval vs lm-evaluation-harness number agreement.** Query: *"Do lighteval and lm-evaluation-harness produce identical numbers on `{benchmark}` as of {today}? Known discrepancies."*
- **vLLM determinism for eval.** Query: *"Current vLLM determinism guarantees for eval (sampling, prefix caching, KV cache eviction effects) as of {today}."*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- `<workdir>/metrics_eval.json` exists with `library`, `backend`, `model_hash`, `metrics`.
- Reported metrics are non-trivially above random baseline (MMLU random is 25%; below 30% suggests broken eval).
- For finetuned models: scores compared against the pre-finetune baseline.
- Backend + library versions recorded in the metrics JSON.
- No cherry-picking: at least the standard suite (≥4 benchmarks) reported.

## Output checklist

- [ ] Library (lm-eval-harness / lighteval / custom) chosen per decision rules
- [ ] Backend (vLLM / SGLang) chosen per decision rules
- [ ] Standard suite picked (no cherry-picking)
- [ ] Eval ran end-to-end
- [ ] Metrics JSON saved with full reproducibility metadata
- [ ] Comparison to pre-finetune baseline reported
