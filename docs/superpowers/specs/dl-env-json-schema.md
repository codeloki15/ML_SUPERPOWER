# `env.json` Schema

**Location:** `<workdir>/env.json`
**Writer:** `dl-detect-env` skill
**Readers:** `dl-remote-execute`, `dl-distributed`, `dl-llm-lora`, `dl-llm-instruction-tune`, all DL skills that need to know what compute is available.

## Top-level shape

```json
{
  "active": "<environment-name>",
  "environments": {
    "<name>": { ... environment object ... }
  }
}
```

- `active` — the environment name the orchestrator is currently using. Updated by `dl-remote-execute` when the user picks a remote, by `dl-detect-env` on first write (defaults to `"local"`).
- `environments` — map of environment name to environment object. Names are user-friendly identifiers (`"local"`, `"modal"`, `"runpod-h100"`, `"ssh-gpu-box"`).

## Environment object — common fields

| Field | Required | Type | Notes |
|---|---|---|---|
| `kind` | yes | `"local"` \| `"remote"` | Switch on this for routing logic. |
| `device` | local + ssh | `"cuda"` \| `"mps"` \| `"cpu"` | Probed via torch on local; via SSH on remote shell hosts. Not set for menu-style providers (Modal, RunPod, Vast, Lambda, Beam). |
| `vram_gb` | local + ssh | integer | 0 for CPU/MPS-only. For SSH remotes, the per-host GPU's VRAM. For menu-style providers (Modal, RunPod, Vast, Lambda, Beam), VRAM is per-GPU and determined per-run when the user picks from `available_gpus`; omit at probe time. |
| `torch_version` | local-only | string | From `torch.__version__`. |
| `cuda_version` | local-only | string \| null | From `torch.version.cuda` if available. |
| `available_libs` | local-only | list of strings | Detected ML libs (`transformers`, `peft`, `trl`, `unsloth`, `axolotl`, `vllm`, `sglang`, `bitsandbytes`, `accelerate`, `deepspeed`, `wandb`, `mlflow`, `aim`, `albumentations`, `timm`, `mergekit`, `lighteval`). |

## Environment object — remote-only fields

Two shapes of remote environment exist:
- **Shell-style** (`ssh`, sometimes `runpod`/`lambda` when accessed as long-running pods) — a real machine you can SSH into. These DO carry `device` and `vram_gb` like local envs, plus the remote-specific fields below.
- **Menu-style** (`modal`, `beam`, `runpod` serverless, `vastai`, `lambda` API-reserved) — the provider exposes a list of GPU classes (`available_gpus`) and the user picks one per-run. These omit `device` and `vram_gb` at probe time; the actual GPU is chosen by `dl-remote-execute` when the script runs.

| Field | Required | Type | Notes |
|---|---|---|---|
| `provider` | yes | `"modal"` \| `"runpod"` \| `"vastai"` \| `"lambda"` \| `"beam"` \| `"ssh"` \| `"colab"` | Drives which sub-mode of `dl-remote-execute` runs. |
| `auth` | yes | `"ok"` \| `"missing"` \| `"expired"` | Probe result. Only environments with `"ok"` are offered. |
| `default_gpu` | provider-specific | string | E.g., `"A10G"` for Modal default. |
| `available_gpus` | provider-specific | list of strings | E.g., `["T4", "A10G", "A100-40GB", "H100"]`. |
| `ssh_target` | ssh/runpod | string | `user@host` for SSH-based providers. |
| `host` | ssh-only | string | Hostname only when `provider == "ssh"`. |
| `detected_runtime` | colab-only | boolean | True only when running inside a Colab notebook. |

## Example

```json
{
  "active": "local",
  "environments": {
    "local": {
      "kind": "local",
      "device": "mps",
      "vram_gb": 0,
      "torch_version": "2.5.1",
      "cuda_version": null,
      "available_libs": ["transformers", "peft", "accelerate", "wandb"]
    },
    "modal": {
      "kind": "remote",
      "provider": "modal",
      "auth": "ok",
      "default_gpu": "A10G",
      "available_gpus": ["T4", "A10G", "A100-40GB", "H100"]
    },
    "ssh-gpu-box": {
      "kind": "remote",
      "provider": "ssh",
      "host": "gpu-box.example",
      "ssh_target": "lokesh@gpu-box.example",
      "device": "cuda",
      "vram_gb": 24,
      "auth": "ok"
    }
  }
}
```

## Refresh policy

`env.json` is refreshed only when:
1. The user explicitly invokes `dl-detect-env`.
2. `dl-remote-execute` fails to reach the `active` environment (auto-refresh, then re-prompt the user).

It is NOT refreshed on every step. Skills assume the file is current within the session.
