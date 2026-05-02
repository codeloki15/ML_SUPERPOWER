---
name: dl-detect-env
description: Use at the start of any deep learning task (CV, NLP, LLM, VLM finetuning) to probe local compute and available remote handoffs. Writes <workdir>/env.json. Do NOT use for tabular ML tasks (those don't need GPU detection) or for re-probing within the same session unless a remote-execute call failed.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# Detect Environment

Probe the user's compute fleet — local device and every reachable remote handoff — and write a single source-of-truth `env.json` that all later DL skills read. Treat this as a one-time setup at the start of the workdir; do not re-probe unless explicitly invoked or a remote-execute call failed.

## When to invoke

- First step of any DL task (CV, NLP, LLM, VLM) after the workdir is created.
- User explicitly says "re-detect environment", "what GPUs do I have", "refresh remote configs".
- `dl-remote-execute` reports an unexpected provider error (auth expired, CLI missing, host unreachable) — re-probe to update `env.json` then surface fresh candidates.

## When NOT to invoke

- The task is tabular ML — no GPU/handoff detection needed.
- `env.json` already exists in the current workdir AND no failure has occurred AND the user did not ask to refresh.
- Inside a per-step loop. This is a setup skill, not a per-step probe.

## Process

### Step 1 — Probe local compute

Generate a probe script written via `ml-engineer-write-code` Layout A and executed via `ml-engineer-execute`. The script must:

- Import `torch` and report `torch.__version__`, `torch.cuda.is_available()`, `torch.backends.mps.is_available()`.
- If CUDA: report `torch.version.cuda`, count GPUs via `torch.cuda.device_count()`, and for each GPU report name (`torch.cuda.get_device_name(i)`) and total VRAM in GB (`torch.cuda.get_device_properties(i).total_memory // 1024**3`).
- If MPS: report `device = "mps"`, `vram_gb = 0` (Apple Silicon shares system memory; we treat it as "small model only" until a future skill estimates available unified memory).
- If CPU only: report `device = "cpu"`, `vram_gb = 0`.
- Probe importable libraries from this list and write the names that import cleanly: `transformers`, `peft`, `trl`, `accelerate`, `deepspeed`, `bitsandbytes`, `unsloth`, `axolotl`, `vllm`, `sglang`, `wandb`, `mlflow`, `aim`, `albumentations`, `timm`, `mergekit`, `lighteval`, `lm_eval`. Use `importlib.util.find_spec()` — do NOT actually import them (avoid heavy initialization).
- Detect Colab runtime by checking `os.environ.get("COLAB_RELEASE_TAG")`. Colab is treated as a *local* environment when present (it IS the local runtime in disguise). Record `colab_runtime_detected: true | false` in the local probe output; the composer in Step 3 uses this to add a `colab` entry to `environments` only when detected.

Write the probe output to `<workdir>/_local_probe.json` so the parent script can read it.

### Step 2 — Probe remote handoffs

For each remote provider, run the corresponding shell-level check. Use the `Bash` tool. Each probe is independent and may fail silently (provider not configured is the normal case).

| Provider | Probe |
|---|---|
| Modal | `which modal && modal token current 2>&1` — present if exit 0 AND output contains a token id. |
| RunPod | `which runpodctl && runpodctl config 2>&1` — present if exit 0 AND output contains a configured API key. |
| Vast.ai | `which vastai && vastai show user 2>&1` — present if exit 0 AND output contains a user id. |
| Lambda Labs | check for `~/.lambda_cloud/credentials` OR `LAMBDA_API_KEY` env var. |
| Beam | `which beam && beam config show 2>&1` — present if exit 0 AND output mentions `default` or a workspace. |
| Generic SSH | parse `~/.ssh/config` for `Host` blocks whose pattern is NOT a pure wildcard (skip blocks like `Host *`, `Host *.example.com`, `Host ?*`). For each non-wildcard host, do NOT auto-classify as GPU — list them as candidate `ssh-<name>` environments with `auth: "ok"` if the block has `IdentityFile` set. The user can later annotate which are GPU boxes. |

### Step 3 — Compose env.json

Merge the local probe and remote probe results into the schema defined in `docs/superpowers/specs/dl-env-json-schema.md`. Set `active = "local"` on first write.

If `env.json` already exists and this invocation is a refresh (failure-triggered or user-requested), preserve the existing `active` value UNLESS the environment is no longer present in the new `environments` map OR its `auth` is no longer `"ok"`. In either case, fall back to `"local"` and surface the change in the Step 4 summary table (e.g., `Active: local (was modal — auth expired)`).

### Step 4 — Report to user

Print a short summary table to stdout:

```
Detected environments:
  local              MPS, 0 GB VRAM, torch 2.5.1
  modal              auth ok, GPUs: T4 / A10G / A100-40GB / H100
  ssh-gpu-box        cuda, 24 GB VRAM (annotated by user)
  runpod             not configured
  vastai             not configured
  lambda             not configured
  beam               auth ok
  colab              not detected

env.json written to <workdir>/env.json
Active: local
```

Do NOT exhaustively list every probe that returned negative — only show "not configured" for providers the user might reasonably have. Skip mention of providers entirely absent from the host (e.g., `runpodctl` binary not installed).

## Recipe template

The probe script written via `ml-engineer-write-code` Layout A. The orchestrator adapts paths and handler details but should preserve the structure below to keep probes consistent across sessions.

### `<workdir>/_probe_local.py`

```python
import importlib.util
import json
import os
import sys
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()
LIBS = [
    "transformers", "peft", "trl", "accelerate", "deepspeed",
    "bitsandbytes", "unsloth", "axolotl", "vllm", "sglang",
    "wandb", "mlflow", "aim",
    "albumentations", "timm",
    "mergekit", "lighteval", "lm_eval",
]


def detect_libs():
    return sorted(name for name in LIBS if importlib.util.find_spec(name) is not None)


def detect_device():
    try:
        import torch
    except ImportError:
        return {"device": "cpu", "vram_gb": 0, "torch_version": None, "cuda_version": None, "gpus": []}
    info = {"torch_version": torch.__version__, "cuda_version": None, "gpus": []}
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["cuda_version"] = torch.version.cuda
        gpus = []
        max_vram = 0
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory // 1024**3
            gpus.append({"index": i, "name": props.name, "vram_gb": int(vram_gb)})
            max_vram = max(max_vram, vram_gb)
        info["gpus"] = gpus
        info["vram_gb"] = int(max_vram)
    elif torch.backends.mps.is_available():
        info.update({"device": "mps", "vram_gb": 0})
    else:
        info.update({"device": "cpu", "vram_gb": 0})
    return info


def detect_colab():
    return os.environ.get("COLAB_RELEASE_TAG") is not None


def main():
    out = detect_device()
    out["available_libs"] = detect_libs()
    out["colab_runtime_detected"] = detect_colab()
    target = WORKDIR / "_local_probe.json"
    target.write_text(json.dumps(out, indent=2))
    print(f"Local probe written to {target}")


if __name__ == "__main__":
    main()
```

### Composing `env.json`

After running the local probe and the Step 2 remote probes, the orchestrator merges results into a single object that conforms to the schema in `docs/superpowers/specs/dl-env-json-schema.md`. Pseudocode:

```python
import json
from pathlib import Path

WORKDIR = Path(...)
local = json.loads((WORKDIR / "_local_probe.json").read_text())
env = {
    "active": "local",
    "environments": {
        "local": {
            "kind": "local",
            "device": local["device"],
            "vram_gb": local["vram_gb"],
            "torch_version": local["torch_version"],
            "cuda_version": local["cuda_version"],
            "available_libs": local["available_libs"],
        },
    },
}
# Append each remote probe result (Modal, RunPod, Vast, Lambda, Beam, SSH, Colab) here.
# Only include providers whose probe succeeded with auth == "ok" or with explicit user-annotated SSH hosts.

# On refresh: read existing env.json, preserve `active` if the same env still exists with auth == "ok".

(WORKDIR / "env.json").write_text(json.dumps(env, indent=2))
(WORKDIR / "_local_probe.json").unlink()  # cleanup intermediate
print(f"env.json written. Active: {env['active']}")
```

## Hard constraints

- Do NOT actually import heavy libraries during the probe. Use `importlib.util.find_spec()`. Importing `vllm` or `transformers` can take seconds and load CUDA contexts.
- Do NOT auto-classify SSH hosts as GPU-equipped. We have no way to know without connecting. List them as candidates and let the user annotate (a later sub-task may add an interactive annotate step; not in Phase 1).
- Do NOT write to any path outside `<workdir>/`.
- Do NOT cache results across sessions. `env.json` lives in the per-task workdir, not in the user's home directory. Each new task re-probes.
- Do NOT echo any tokens, API keys, or credentials to stdout or into `env.json`. Only record `auth: "ok" | "missing" | "expired"` — never the secret itself.
- If a provider's CLI is present but auth fails, record `auth: "missing"` or `auth: "expired"` and do NOT include the provider in `active` candidates.
- DELETE `<workdir>/_local_probe.json` after Step 3 successfully writes `env.json`. The intermediate file is a transport between probe and composer; leaving it creates clutter and confuses future invocations into thinking they have stale results to merge.

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- `<workdir>/env.json` exists and is valid JSON.
- The file matches the schema in `docs/superpowers/specs/dl-env-json-schema.md` — top-level keys `active` and `environments`, each environment has at minimum `kind`, and `vram_gb` is present for `local` and SSH-style remote envs (omitted for menu-style providers per the schema).
- `active` references an environment that exists in the `environments` map.
- No secrets appear in the file (`grep -iE "token|key|secret|password" <workdir>/env.json` returns no matches).

If any check fails, this skill's run is `failed`, not `verified`. Re-run with the failure as input.

## Output checklist

- [ ] `<workdir>/env.json` written with valid JSON
- [ ] Local environment probed; CUDA / MPS / CPU classified correctly
- [ ] All available remote provider CLIs probed
- [ ] Summary table printed to stdout
- [ ] No secrets in the file
- [ ] `active` defaults to `"local"` on first write, preserved on refresh
