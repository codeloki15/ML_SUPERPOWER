# DL Skills Phase 1 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `ml-engineer` plugin v0.2.0-alpha.1: add a router prologue, three DL sub-agent shells (cv-engineer, nlp-engineer, llm-engineer), and the six shared infrastructure skills (`dl-detect-env`, `dl-remote-execute`, `dl-experiment-track`, `dl-checkpoint`, `dl-distributed`, `dl-debug-training`).

**Architecture:** Pure markdown additions to a Claude Code plugin. Each skill is a self-contained `SKILL.md` with frontmatter trigger + body. Sub-agent files in `agents/` define loop variants. No code is executed by the plan itself — skills are content read by Claude at invocation time. Hybrid writing style per spec: terse headers/bullets, full prose for decision rules and error handling.

**Tech Stack:** Markdown (CommonMark + GitHub-flavored). YAML frontmatter. JSON for `env.json` schema documented inline. Plugin spec follows Anthropic's Claude Code plugin reference.

**Reference spec:** [`docs/superpowers/specs/2026-05-01-dl-skills-design.md`](../specs/2026-05-01-dl-skills-design.md). Read this before starting any task — every skill body must match the structure rules in the spec's "Skill structure & content rules" section.

**Phase 1 scope (recap from spec):**
- Add 6 shared infra skills.
- Add router prologue to existing `agents/ml-engineer.md`.
- Add 3 sub-agent shells: `cv-engineer.md`, `nlp-engineer.md`, `llm-engineer.md`.
- Bump plugin version to `0.2.0-alpha.1`.

**Out of phase scope:** Domain-specific DL skills (CV/NLP/LLM/VLM), data/training/inference skills. Those are Phases 2 and 3.

---

## File Structure

**New files (10):**

```
agents/
├── cv-engineer.md                          (NEW — vision sub-agent shell)
├── nlp-engineer.md                         (NEW — NLP sub-agent shell)
└── llm-engineer.md                         (NEW — LLM/VLM sub-agent shell)

skills/
├── dl-detect-env/SKILL.md                  (NEW — env probe + env.json writer)
├── dl-remote-execute/SKILL.md              (NEW — remote dispatcher)
├── dl-experiment-track/SKILL.md            (NEW — wandb/mlflow/aim wiring)
├── dl-checkpoint/SKILL.md                  (NEW — save/resume/shard)
├── dl-distributed/SKILL.md                 (NEW — single-GPU vs FSDP2 vs DeepSpeed selector)
└── dl-debug-training/SKILL.md              (NEW — NaN/loss-spike/OOM triage)

docs/superpowers/specs/dl-env-json-schema.md  (NEW — env.json schema reference, linked from dl-detect-env)
```

**Modified files (3):**

```
.claude-plugin/plugin.json                  (version 0.1.0 → 0.2.0-alpha.1)
agents/ml-engineer.md                       (add router prologue at the top)
README.md                                   (add DL section + reference Phase 1 status)
```

Each skill file owns one concern. Sub-agent files are thin shells (~80-120 lines): purpose, persona, the loop, the skill table, hard rules. Router prologue inside `ml-engineer.md` is ~30-50 lines added before the existing content.

---

## Pre-flight: Verify environment

- [ ] **Step 0.1: Confirm working directory and branch**

Run: `cd /Users/lokesh/Desktop/code/Bio_hacking/ML_Engineer && git status && git branch --show-current`
Expected: Clean working tree on `main`, with the spec already committed (commit `f117e9d` or later).

- [ ] **Step 0.2: Verify the spec exists and is readable**

Run: `ls -la docs/superpowers/specs/2026-05-01-dl-skills-design.md`
Expected: file exists, ~498 lines.

- [ ] **Step 0.3: Verify existing skill structure to mirror**

Run: `ls skills/ml-engineer-research/ && head -10 skills/ml-engineer-research/SKILL.md`
Expected: directory contains `SKILL.md`, frontmatter starts with `---`, has `name:`, `description:`, `license: MIT`, `metadata:` with `source` and `version`.

---

## Task 1: Bump plugin version

**Files:**
- Modify: `.claude-plugin/plugin.json`

- [ ] **Step 1.1: Read current plugin.json**

Run: `cat .claude-plugin/plugin.json`
Expected output includes `"version": "0.1.0"`.

- [ ] **Step 1.2: Update version to 0.2.0-alpha.1**

Edit `.claude-plugin/plugin.json`:

Old:
```json
  "version": "0.1.0",
```

New:
```json
  "version": "0.2.0-alpha.1",
```

- [ ] **Step 1.3: Update description to mention DL coverage**

Edit `.claude-plugin/plugin.json`:

Old:
```json
  "description": "An ML engineer assistant that plans, writes, executes, and debugs Python data-science / ML tasks in an isolated venv. Runs entirely on the user's local machine.",
```

New:
```json
  "description": "An ML engineer assistant for tabular ML, computer vision, NLP, LLM and VLM finetuning. Plans, writes, executes, verifies, and debugs Python work in an isolated local venv with optional handoff to remote GPU providers (Modal, RunPod, Vast, Lambda, Beam, SSH, Colab).",
```

- [ ] **Step 1.4: Verify JSON is valid**

Run: `python3 -c "import json; json.load(open('.claude-plugin/plugin.json'))"`
Expected: no output, exit 0.

- [ ] **Step 1.5: Commit**

```bash
git add .claude-plugin/plugin.json
git commit -m "Bump plugin version to 0.2.0-alpha.1 for DL skills phase 1"
```

---

## Task 2: Write env.json schema reference

**Files:**
- Create: `docs/superpowers/specs/dl-env-json-schema.md`

This is the schema document `dl-detect-env` writes against. Lives in `specs/` because it is a contract, referenced by multiple skills.

- [ ] **Step 2.1: Create the schema reference file**

Write `docs/superpowers/specs/dl-env-json-schema.md`:

````markdown
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
| `device` | local-only | `"cuda"` \| `"mps"` \| `"cpu"` | Probed via torch. |
| `vram_gb` | yes | integer | 0 for CPU/MPS-only. For remote, the GPU's VRAM. |
| `torch_version` | local-only | string | From `torch.__version__`. |
| `cuda_version` | local-only | string \| null | From `torch.version.cuda` if available. |
| `available_libs` | local-only | list of strings | Detected ML libs (`transformers`, `peft`, `trl`, `unsloth`, `axolotl`, `vllm`, `sglang`, `bitsandbytes`, `accelerate`, `deepspeed`, `wandb`, `mlflow`, `aim`, `albumentations`, `timm`, `mergekit`, `lighteval`). |

## Environment object — remote-only fields

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
````

- [ ] **Step 2.2: Verify the file is well-formed markdown**

Run: `wc -l docs/superpowers/specs/dl-env-json-schema.md`
Expected: ~80-100 lines.

- [ ] **Step 2.3: Commit**

```bash
git add docs/superpowers/specs/dl-env-json-schema.md
git commit -m "Add env.json schema reference for dl-detect-env contract"
```

---

## Task 3: Create skill `dl-detect-env`

**Files:**
- Create: `skills/dl-detect-env/SKILL.md`

This skill probes local compute (device, VRAM, libs) AND every configured remote handoff (Modal token, `runpodctl` on PATH, `vastai` configured, `~/.ssh/config` entries that look like GPU boxes, Lambda creds, Beam config, Colab runtime presence). Writes the results to `<workdir>/env.json`.

- [ ] **Step 3.1: Create the skill directory**

Run: `mkdir -p skills/dl-detect-env`

- [ ] **Step 3.2: Write the SKILL.md**

Create `skills/dl-detect-env/SKILL.md` with the following content:

````markdown
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
| Generic SSH | parse `~/.ssh/config` for `Host *` entries; for each, do NOT auto-classify as GPU — instead list them as candidate `ssh-<name>` environments with `auth: "ok"` if the config has `IdentityFile` set. The user can later annotate which are GPU boxes. |
| Colab | check `os.environ.get("COLAB_RELEASE_TAG")` is set inside the local probe. |

### Step 3 — Compose env.json

Merge the local probe and remote probe results into the schema defined in `docs/superpowers/specs/dl-env-json-schema.md`. Set `active = "local"` on first write.

If `env.json` already exists and this invocation is a refresh (failure-triggered or user-requested), preserve the existing `active` value unless that environment is no longer present (then fall back to `"local"`).

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

## Hard constraints

- Do NOT actually import heavy libraries during the probe. Use `importlib.util.find_spec()`. Importing `vllm` or `transformers` can take seconds and load CUDA contexts.
- Do NOT auto-classify SSH hosts as GPU-equipped. We have no way to know without connecting. List them as candidates and let the user annotate (a later sub-task may add an interactive annotate step; not in Phase 1).
- Do NOT write to any path outside `<workdir>/`.
- Do NOT cache results across sessions. `env.json` lives in the per-task workdir, not in the user's home directory. Each new task re-probes.
- Do NOT echo any tokens, API keys, or credentials to stdout or into `env.json`. Only record `auth: "ok" | "missing" | "expired"` — never the secret itself.
- If a provider's CLI is present but auth fails, record `auth: "missing"` or `auth: "expired"` and do NOT include the provider in `active` candidates.

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- `<workdir>/env.json` exists and is valid JSON.
- The file matches the schema in `docs/superpowers/specs/dl-env-json-schema.md` — top-level keys `active` and `environments`, each environment has at minimum `kind` and `vram_gb`.
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
````

- [ ] **Step 3.3: Verify frontmatter parses**

Run: `python3 -c "import yaml; yaml.safe_load(open('skills/dl-detect-env/SKILL.md').read().split('---')[1])"`
Expected: no output, exit 0.

- [ ] **Step 3.4: Commit**

```bash
git add skills/dl-detect-env/SKILL.md
git commit -m "Add dl-detect-env skill: probe compute fleet and write env.json"
```

---

## Task 4: Create skill `dl-remote-execute`

**Files:**
- Create: `skills/dl-remote-execute/SKILL.md`

The most complex new skill. Acts as a dispatcher across 7 providers. Implements the "ask once, continue silently" decision flow from the spec.

- [ ] **Step 4.1: Create the skill directory**

Run: `mkdir -p skills/dl-remote-execute`

- [ ] **Step 4.2: Write the SKILL.md**

Create `skills/dl-remote-execute/SKILL.md` with the following content:

````markdown
---
name: dl-remote-execute
description: Use when a DL training or inference script estimated to need more VRAM, walltime, or disk than the local environment has, OR when the user explicitly asks to run on a specific remote (Modal, RunPod, Vast.ai, Lambda, Beam, SSH host, Colab). Reads <workdir>/env.json to discover available environments, surfaces top-3 candidates with cost+latency tradeoffs, runs the script remotely, fetches results back. Do NOT use for local execution (use ml-engineer-execute) or for first-time environment discovery (use dl-detect-env first).
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# Remote Execute

Dispatcher skill that runs a workdir script on a user-selected remote GPU provider, streams logs back, and fetches result artifacts. Implements the "ask once, continue silently" model: the user picks an environment on the first remote step of a task, then subsequent remote steps in the same task continue using that environment without re-prompting.

## When to invoke

- A script's estimated resource requirement (VRAM, walltime, disk) exceeds what the local environment in `env.json` provides.
- User explicitly says "run this on Modal / RunPod / my SSH box / Colab".
- A retry of a previously-remote step (use the same `active` environment unless the requirement changed).

## When NOT to invoke

- The script fits locally — use `ml-engineer-execute` instead.
- `env.json` does not exist or has no remote environments with `auth: "ok"` — invoke `dl-detect-env` first.
- The user is asking to compare provider costs without running anything — answer from `env.json` directly without invoking the dispatcher.

## Decision flow

### Step 1 — Read env.json

Read `<workdir>/env.json`. If the file is missing, fail and instruct the orchestrator to invoke `dl-detect-env` first. If the file exists but `active` is `"local"` and this is the first remote step, fall through to candidate selection.

### Step 2 — Filter candidates

Build the candidate list:

1. Take all environments where `kind == "remote"` AND `auth == "ok"`.
2. Filter by VRAM requirement: keep environments where `vram_gb >= requirement.vram_gb`. For providers with `available_gpus` (Modal, Beam), check whether any GPU in the list satisfies the requirement; if yes, include the provider with the smallest sufficient GPU.
3. If `local` satisfies the requirement, include it as a free option.

If the filtered list is empty, surface this exact error to the user:

> This step needs ~{requirement.vram_gb} GB VRAM. None of your configured environments have enough. Either:
> 1. Configure a larger remote (Modal, RunPod, Lambda).
> 2. Reduce the model size (use a smaller variant or QLoRA instead of full finetune).
> 3. Increase quantization (4-bit instead of 8-bit).

Stop. Do not pick an environment automatically.

### Step 3 — First remote step of task: ask the user

IF `env.json.active == "local"` AND this is the first remote step of the current task, present the top-3 candidates in this exact format:

```
Step needs ~{vram_gb} GB VRAM, ~{walltime} min walltime. Candidates:

  1. {env_name_1}    {gpu_class}    {hourly_rate}    {cold_start_latency}    {billing_model}
  2. {env_name_2}    {gpu_class}    {hourly_rate}    {cold_start_latency}    {billing_model}
  3. {env_name_3}    {gpu_class}    {hourly_rate}    {cold_start_latency}    {billing_model}

Which one? (1/2/3, or "skip" to abort, or "more" to see all candidates)
```

Hourly rates are best-effort lookups via `ml-engineer-research` (the rates change; do not hardcode). Cold start latency is provider-typical (Modal ~3s, Beam ~2-3s, RunPod warm ~10s SSH connect, RunPod cold ~30-60s pod creation, SSH/Lambda immediate if pre-provisioned).

Wait for user response. On selection, set `env.json.active = <chosen-name>` and write the file.

### Step 4 — Subsequent remote steps: continue silently

IF `env.json.active != "local"` AND the current step's resource estimate fits the active environment, run on `active` without re-prompting. Print one line: `Using {active}.` then proceed.

IF the current step's resource estimate exceeds `active`, re-enter Step 3 with a fresh candidate list and re-prompt.

### Step 5 — User-driven switch

IF the user explicitly says "switch to {env}" or "run on {env}" at any time, set `env.json.active = <env>` and use it for the current and subsequent steps.

### Step 6 — Provider sub-mode dispatch

Switch on `env.environments[active].provider`:

| Provider | Sub-mode procedure |
|---|---|
| `modal` | Wrap the user's script in a `@app.function(gpu=<smallest-sufficient>, image=<...>, secrets=[...])` decorator. Run via `modal run <wrapped_script.py>`. Stream stdout/stderr to `<workdir>/remote_logs/modal_<step>.log`. Fetch artifacts via Modal volumes mounted to local paths. KNOWN ISSUE (late-2025): GPU sandbox + mounted Volume can crash silently. Workaround: write artifacts to ephemeral `/tmp` on Modal, then sync to local via Modal CLI after the function returns rather than mounting a Volume during execution. |
| `runpod` | `runpodctl create pod --gpuType <gpu> --imageName <pytorch-image>`. Wait for pod ready. `rsync` the workdir up. `ssh` exec the script. Stream logs back via `ssh` with line-buffered output. `rsync` results back. `runpodctl stop pod` (or destroy if not reusable). Surface incurred cost from the API response. |
| `vastai` | `vastai search offers` with VRAM filter, sort by `dph` ascending. `vastai create instance <id> --image <pytorch-image>`. Same `rsync`+ssh flow as RunPod. ALWAYS `vastai destroy instance <id>` on completion (success OR failure). Default to single-shot — do NOT keep instances alive between steps unless user explicitly says so. |
| `lambda` | Reserve via API. SSH in (PyTorch preinstalled). `rsync` flow. Lambda is slow to provision (~minutes), so use only when other providers are unavailable or for jobs >1 hour. |
| `beam` | Wrap script in `@function(cpu=..., memory=..., gpu=...)` decorator. `beam run <script>`. Similar to Modal but faster cold start. |
| `ssh-generic` | `rsync -av <workdir>/ {ssh_target}:~/dl_workdir_<timestamp>/`. `ssh {ssh_target} 'cd dl_workdir_<timestamp> && python <script_relative_path>'` with line-buffered output. `rsync -av {ssh_target}:~/dl_workdir_<timestamp>/results/ <workdir>/results/`. Optionally clean up remote dir on success. |
| `colab-handoff` | Generate a `.ipynb` from the script (one cell per logical block, plus a final cell that writes `results.json` to a Drive-synced path). Print the URL pattern: `https://colab.research.google.com/notebook#fileId={uploaded_id}` along with explicit instructions: "Open this URL, click Runtime → Run all, then come back and tell me when it finishes." Park the step. When the user confirms completion, fetch `results.json` from the user-provided Drive sync path. |

### Step 7 — Always-tear-down hard rule

For ephemeral providers (`vastai`, `runpod` when not user-pinned, `lambda` when reserved-for-this-task), tear down the resource on completion regardless of success or failure. Surface the incurred cost in the result. NEVER leave a pod or instance running unless the user explicitly asked to keep it.

### Step 8 — Return result to orchestrator

Return:

```
Exit code:    {0 | non-zero}
Logs:         <workdir>/remote_logs/<provider>_<step>.log
Artifacts:    <list of paths fetched back into workdir>
Cost:         ${incurred} on {provider}
Resume info:  {modal app url | runpod pod id | ssh session marker | colab notebook url}
```

Pass exit code through unchanged so the orchestrator can branch the same way it does for local execution.

## Hard constraints

- NEVER auto-launch paid compute without user confirmation on the first remote step of a session. Even if `auth: "ok"`, the first time a remote runs in a task, ask.
- NEVER leave secrets in the script body that gets uploaded. Secrets MUST be set as remote env vars (Modal: `Secret.from_name(...)`, SSH: `ssh ENV=value cmd`, etc.) or read from the remote's local secret store, never embedded in the script.
- NEVER skip tear-down. If the script fails partway, fetch what's salvageable (last checkpoint, last log lines), THEN tear down.
- NEVER use `vastai` for sustained multi-step work. Vast instances are volatile and can disappear. Single-shot only unless the user pins.
- NEVER assume Colab will run synchronously. The Colab handoff is async — park the step and wait for explicit user confirmation that the notebook finished.
- NEVER stream raw container logs that may contain secrets (e.g., `env` dumps). Filter the log stream for lines matching `*_TOKEN`, `*_KEY`, `password=`, `Bearer `, and replace with `[REDACTED]`.
- IF a provider CLI returns an unexpected response (changed format, new error), do NOT silently guess. Surface the raw error to the user and let them decide.

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- The remote step's exit code matches what the script would produce locally (no false-success because the upload succeeded but execution failed).
- All declared output artifacts (model files, log files, metric JSON) were fetched back into the workdir.
- For ephemeral providers: the resource was actually torn down (provider-specific check, e.g., `runpodctl get pods` shows no orphan; `vastai show instances` is empty for our user).
- No secret-shaped strings appear in `<workdir>/remote_logs/<provider>_<step>.log`.

## Output checklist

- [ ] `env.json` consulted; `active` updated only if user picked or switched
- [ ] Top-3 candidates shown with cost + latency tradeoffs on first remote step
- [ ] Subsequent steps continued silently on `active`
- [ ] Logs streamed to `<workdir>/remote_logs/<provider>_<step>.log`
- [ ] Artifacts fetched back into workdir
- [ ] Ephemeral resources torn down (verified)
- [ ] Cost surfaced
- [ ] No secrets in the uploaded script or returned logs
````

- [ ] **Step 4.3: Verify frontmatter parses**

Run: `python3 -c "import yaml; yaml.safe_load(open('skills/dl-remote-execute/SKILL.md').read().split('---')[1])"`
Expected: no output, exit 0.

- [ ] **Step 4.4: Commit**

```bash
git add skills/dl-remote-execute/SKILL.md
git commit -m "Add dl-remote-execute skill: 7-provider dispatcher with ask-once flow"
```

---

## Task 5: Create skill `dl-experiment-track`

**Files:**
- Create: `skills/dl-experiment-track/SKILL.md`

- [ ] **Step 5.1: Create the skill directory**

Run: `mkdir -p skills/dl-experiment-track`

- [ ] **Step 5.2: Write the SKILL.md**

Create `skills/dl-experiment-track/SKILL.md`:

````markdown
---
name: dl-experiment-track
description: Use before training any DL model that runs longer than ~5 minutes or that the user might want to compare against later runs. Wires wandb / mlflow / aim into the training script. Do NOT use for one-off probes, EDA scripts, or evaluation-only runs that will not be re-compared.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# Experiment Track

Wire experiment tracking (wandb / mlflow / aim) into a training script BEFORE training starts. Without tracking, comparing two runs requires reconstructing what was different — slow and error-prone. With tracking, the comparison is a URL.

## When to invoke

- Before any training run expected to take >5 minutes.
- User explicitly says "track this", "log to wandb", "log to mlflow".
- Setting up a hyperparameter sweep — tracking is mandatory for sweeps.

## When NOT to invoke

- One-off probes, EDA scripts, or any script that will not be re-run with different settings.
- Evaluation-only runs that compute a single metric and exit.
- The user explicitly says "no tracking".

## Decision rules

Pick the tracker based on what is already configured in `env.json.environments.local.available_libs`:

- IF `wandb` is in `available_libs` AND the user has not specified a tracker: use wandb. Reason: most common in HF ecosystem, supports HF Trainer integration out of the box, hosted UI.
- IF `wandb` is NOT available BUT `mlflow` is: use mlflow. Reason: self-hostable, common in enterprise.
- IF neither is available BUT `aim` is: use aim. Reason: lightweight, local-first.
- IF none are available: print this message and stop:

> No experiment tracker is installed. Install one before training:
>   pip install wandb     (recommended; free hosted)
>   pip install mlflow    (self-hosted)
>   pip install aim       (local-first)
> Or proceed without tracking by saying "skip tracking".

The user MAY override the choice at any time by saying "use {tracker}".

## Process

### Step 1 — Confirm or pick tracker

Read `env.json.environments.local.available_libs`. Apply decision rules above.

### Step 2 — Verify auth (wandb only)

For wandb, run `wandb status` via Bash. IF the output indicates not logged in, instruct the user:

> You're not logged into wandb. Run `wandb login` in your terminal first, then re-invoke.

Stop until the user confirms login.

For mlflow and aim, no auth needed for local tracking. IF the user wants a remote mlflow server, ask for the tracking URI.

### Step 3 — Generate the tracking-init snippet

Produce a code snippet to be inserted at the top of the training script (BEFORE any model construction). The snippet MUST:

- Initialize the run with a meaningful name (default: `<task_name>_<UTC_timestamp>`).
- Log all hyperparameters as a config dict (read from a `config.yaml` or argparse defaults).
- Tag the run with the project name (default: workdir basename) and the model family.
- Set up integration with HF Trainer if `transformers` is in available_libs (`report_to="wandb"` or equivalent).

Example for wandb (this is a template the orchestrator adapts; do NOT use it verbatim if the project has different conventions):

```python
import wandb

wandb.init(
    project="<project_name>",
    name="<run_name>",
    config=<config_dict>,
    tags=["<model_family>", "<task_type>"],
)

# ... training code ...

# After training:
wandb.finish()
```

For HF Trainer integration, set `TrainingArguments(report_to=["wandb"], run_name="<run_name>")`. Trainer will log loss, lr, eval metrics automatically.

### Step 4 — Verify the snippet runs

Insert the snippet into the training script. Before starting actual training, run a 1-step dry-run (or import-only check) to confirm the tracker initializes without error. IF init fails (network, auth, missing project), stop and surface the error before burning compute.

## Hard constraints

- NEVER hard-code an API key or token in the training script. Wandb reads from `~/.netrc` or `WANDB_API_KEY`. Mlflow reads from env or config file. Aim is local.
- NEVER set the project name to something generic like `"test"`. Use the workdir basename or the user's stated project name. Generic project names create unfindable runs.
- NEVER skip `wandb.finish()` or equivalent at the end of the script. Without it, runs show as "running" forever in the UI and confuse later comparison.
- IF the run is on a remote provider (`env.json.active != "local"`), make sure the tracker auth is available on the remote too. For wandb on Modal: pass `WANDB_API_KEY` via `modal.Secret`. For SSH boxes: ensure `~/.netrc` exists on the remote.

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- The training script contains a tracking-init call near the top.
- The script contains a finish/cleanup call at the end (`wandb.finish()`, `mlflow.end_run()`, etc.).
- A dry-run (1 step, or just the init call) returned exit 0.
- Run name and project name are non-generic.

## Output checklist

- [ ] Tracker chosen based on available_libs + user input
- [ ] Auth verified (wandb logged in; mlflow URI set if remote; aim local OK)
- [ ] Init snippet inserted at the top of the training script
- [ ] Finish snippet at the end
- [ ] HF Trainer `report_to` set if Trainer is being used
- [ ] Dry-run succeeded
- [ ] On remote runs: tracker auth available on the remote
````

- [ ] **Step 5.3: Verify frontmatter parses**

Run: `python3 -c "import yaml; yaml.safe_load(open('skills/dl-experiment-track/SKILL.md').read().split('---')[1])"`
Expected: no output, exit 0.

- [ ] **Step 5.4: Commit**

```bash
git add skills/dl-experiment-track/SKILL.md
git commit -m "Add dl-experiment-track skill: wandb/mlflow/aim wiring with HF Trainer support"
```

---

## Task 6: Create skill `dl-checkpoint`

**Files:**
- Create: `skills/dl-checkpoint/SKILL.md`

- [ ] **Step 6.1: Create the skill directory**

Run: `mkdir -p skills/dl-checkpoint`

- [ ] **Step 6.2: Write the SKILL.md**

Create `skills/dl-checkpoint/SKILL.md`:

````markdown
---
name: dl-checkpoint
description: Use to set up checkpoint saving and resume logic before any DL training run that may take >30 minutes, may be preempted (spot instances, free Colab tier), or may need partial-state inspection later. Covers HF save_pretrained, raw state_dict, sharded checkpoints, FSDP2 state-dict gotchas. Do NOT use for short runs (<30 min) where re-running from scratch is cheaper than checkpoint plumbing.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# Checkpoint

Set up checkpoint save / resume / inspect logic. The most common silent failure mode in long DL runs is "the job died at hour 8 and there's nothing on disk." This skill prevents that.

## When to invoke

- Any training run expected to take >30 minutes.
- Any run on a preemptible / spot instance (Vast.ai, Colab free, RunPod community).
- Multi-stage workflows where a downstream step needs the model from an earlier step (e.g., SFT then DPO; train then quantize).
- User explicitly says "save checkpoints", "resume from", "shard the checkpoint".

## When NOT to invoke

- Short runs (<30 min) on stable infrastructure where the whole script is cheap to re-run.
- Inference-only scripts.
- Scripts that already use HF Trainer with `save_strategy="epoch"` or `save_strategy="steps"` configured by the orchestrator — Trainer handles checkpointing natively, this skill mainly verifies the config is sane.

## Decision rules

Pick the save strategy based on training shape:

- IF using HF Trainer AND single-GPU: set `save_strategy="steps"`, `save_steps=<every ~10% of total steps>`, `save_total_limit=3` (keep last 3 to bound disk usage).
- IF using HF Trainer AND multi-GPU with FSDP2: set `save_strategy="steps"` AND `fsdp_state_dict_type="SHARDED_STATE_DICT"`. Reason: full state dict on FSDP2 requires gathering all parameters to rank 0, which OOMs on large models. Sharded checkpoints save per-rank shards.
- IF training a LoRA adapter (PEFT): save the adapter only, not the full base model. `model.save_pretrained(<path>)` on a PeftModel saves only the adapter weights (~MB) — orders of magnitude smaller than the base model.
- IF doing full finetune with bf16/fp16 weights: save in `safetensors` format (`safe_serialization=True`). Reason: faster load, immune to pickle vulnerabilities.
- IF the run is on a remote provider with ephemeral storage: also push checkpoints to a persistent location periodically (Modal volume, RunPod network volume, S3 if configured). Otherwise the checkpoint dies with the pod.

## Process

### Step 1 — Determine training shape

Read the training script and determine: HF Trainer or custom loop? Single-GPU or distributed? LoRA or full finetune? Local or remote?

### Step 2 — Generate the checkpoint config

Produce the relevant config block. For HF Trainer:

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="<workdir>/checkpoints",
    save_strategy="steps",
    save_steps=<computed_value>,
    save_total_limit=3,
    save_safetensors=True,
    # FSDP2 only:
    fsdp="full_shard auto_wrap" if <distributed> else "",
    fsdp_config={"fsdp_state_dict_type": "SHARDED_STATE_DICT"} if <distributed> else None,
)
```

For PEFT/LoRA save:

```python
# At end of training:
model.save_pretrained("<workdir>/checkpoints/adapter")
tokenizer.save_pretrained("<workdir>/checkpoints/adapter")
```

For custom loop:

```python
import torch

def save_checkpoint(model, optimizer, scheduler, step, path):
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }
    torch.save(state, path)
```

### Step 3 — Generate resume logic

For HF Trainer: `trainer.train(resume_from_checkpoint=True)` — picks up the latest checkpoint in `output_dir` automatically.

For custom loop:

```python
import torch
from pathlib import Path

ckpt_dir = Path("<workdir>/checkpoints")
ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
if ckpts:
    latest = ckpts[-1]
    state = torch.load(latest, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state["scheduler"]:
        scheduler.load_state_dict(state["scheduler"])
    start_step = state["step"] + 1
    print(f"Resumed from {latest.name} at step {start_step}")
else:
    start_step = 0
```

### Step 4 — Add remote-persistence hook (if remote)

IF `env.json.active != "local"` AND the active env has ephemeral storage, add a periodic sync step. For Modal: write to a `modal.Volume` mounted at the checkpoint path. For SSH: add `rsync -av <local_workdir>/checkpoints/ user@persistent-host:backups/<run_name>/` on each save (called from a `TrainerCallback` or after the save in a custom loop).

### Step 5 — Verify

After the script is wired up, the orchestrator runs the script for ~50 steps and confirms:
- A checkpoint file appeared in `<workdir>/checkpoints/`.
- The checkpoint can be loaded back (run a tiny resume-and-step test).
- For LoRA: only the adapter is saved, not the full base model (file size sanity check).

## Hard constraints

- NEVER use `save_strategy="no"` for runs >30 min unless the user explicitly opts out. Crashed long runs without checkpoints waste compute and money.
- NEVER use full state-dict (`fsdp_state_dict_type="FULL_STATE_DICT"`) on FSDP2 with models larger than the rank-0 GPU's VRAM. It will OOM.
- NEVER load checkpoints with `torch.load(path)` without `weights_only=True` if the source is untrusted. Pickle deserialization can execute arbitrary code. For your own checkpoints, this is moot; for downloaded ones, prefer `safetensors`.
- NEVER overwrite the previous best checkpoint without bound. Use `save_total_limit` to bound disk usage; keep last N + best.
- NEVER assume the remote provider's "persistent" storage survives a pod stop. Verify per provider: Modal volumes survive, RunPod network volumes survive if attached, Vast.ai instance disk does NOT survive a destroy.

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- The training script saves at least one checkpoint within the first ~50 steps.
- The checkpoint loads back without error (a tiny resume test runs).
- For LoRA runs: checkpoint size is in the MB range (not GB) — confirms adapter-only save.
- For FSDP2 runs: `fsdp_state_dict_type` is `"SHARDED_STATE_DICT"`.
- For ephemeral remotes: a persistence hook is wired up.

## Output checklist

- [ ] Save strategy chosen per training shape
- [ ] Checkpoint code inserted in the training script
- [ ] Resume logic inserted (or `resume_from_checkpoint=True` if Trainer)
- [ ] Remote persistence hook added if active env is ephemeral
- [ ] Verify pass: checkpoint appears + loads back + sane size
````

- [ ] **Step 6.3: Verify frontmatter parses**

Run: `python3 -c "import yaml; yaml.safe_load(open('skills/dl-checkpoint/SKILL.md').read().split('---')[1])"`
Expected: no output, exit 0.

- [ ] **Step 6.4: Commit**

```bash
git add skills/dl-checkpoint/SKILL.md
git commit -m "Add dl-checkpoint skill: save/resume with FSDP2 sharded state-dict and PEFT-aware logic"
```

---

## Task 7: Create skill `dl-distributed`

**Files:**
- Create: `skills/dl-distributed/SKILL.md`

- [ ] **Step 7.1: Create the skill directory**

Run: `mkdir -p skills/dl-distributed`

- [ ] **Step 7.2: Write the SKILL.md**

Create `skills/dl-distributed/SKILL.md`:

````markdown
---
name: dl-distributed
description: Use to pick a distributed training strategy (single-GPU, FSDP2 via Accelerate, or DeepSpeed ZeRO-3) when a model does not fit on one GPU OR when the user explicitly asks about multi-GPU training. Generates the launcher command and config. Do NOT use when the model fits on one GPU and the user has not asked for parallelism — added complexity hurts more than it helps.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# Distributed

Decision skill: pick single-GPU / FSDP2 / DeepSpeed ZeRO-3 based on model size, GPU count, VRAM budget, and the active environment. Generates the launcher command (`accelerate launch`, `torchrun`, `deepspeed`) and the config file the launcher needs.

## When to invoke

- Estimated model + optimizer + activations memory exceeds available VRAM on a single GPU.
- User explicitly says "multi-GPU", "FSDP", "DeepSpeed", "ZeRO", "shard the model".
- Training a model >7B params (sharding is almost always the answer).

## When NOT to invoke

- Model fits on one GPU. Distributed adds latency, complexity, and failure modes; do not use it as default.
- Inference-only scripts (vLLM and SGLang have their own tensor-parallel logic; they do not need this skill).
- LoRA / QLoRA training of a model that fits on one GPU (the base model is frozen; the trainable params are tiny).

## Decision rules

Read `env.json.environments[active]` for GPU count and per-GPU VRAM. Estimate memory: model params × bytes_per_param + optimizer state (Adam = 2× params for fp32 moments) + activations. The orchestrator may have a rough estimate from `dl-detect-env` or from a research hook. If unknown, compute a conservative estimate from the model's HF config.

Apply this decision tree:

- IF model fits on one GPU AND only one GPU is available: do NOT invoke this skill — use single-GPU training.
- IF model fits on one GPU AND multiple GPUs are available AND the user did not ask for parallelism: stay single-GPU. Multi-GPU training of a single-GPU-sized model is rarely worth the complexity.
- IF model does NOT fit on one GPU AND total VRAM across GPUs >= memory requirement: use **FSDP2 via Accelerate**. Reason: PyTorch-native, simpler config, better-supported across 2025-2026 HF stack. Default for medium models.
- IF model does NOT fit even with FSDP2 across all available GPUs: use **DeepSpeed ZeRO-3**. Reason: ZeRO-3 with CPU offload can train models that don't fit on the GPU pool at all (offloads optimizer state to CPU/NVMe). Slower per step but fits larger models.
- IF user explicitly requests Unsloth (single-GPU LLM finetune) AND has only one GPU: use Unsloth. Unsloth handles its own memory layout and does NOT support multi-GPU. If the user has multiple GPUs and wants Unsloth, surface this contradiction and ask whether to switch to Axolotl (multi-GPU) or proceed single-GPU on one device.

## Process

### Step 1 — Estimate memory requirement

Compute or look up:
- Model params × bytes (bf16/fp16 = 2 bytes; fp32 = 4 bytes; 4-bit quant ≈ 0.5 bytes)
- Optimizer state (Adam fp32 = 8 bytes per param; AdamW with fused = same; SGD = 4 bytes; 8-bit Adam ≈ 2 bytes)
- Gradients (same as model params unless using gradient checkpointing → ~30% reduction in activation memory at cost of recompute)
- Activations (depends on batch size, sequence length, and recompute strategy)

**Research hook:** Before final estimate, invoke `ml-engineer-research` for: *"Current memory-per-param estimate for {detected_optimizer} on {detected_model_family} {model_size} at {bf16|fp16|4-bit} including activations for sequence length {seq_len} and batch size {batch_size}, as of {today}."* Use the research result to adjust the estimate.

### Step 2 — Apply decision rules

Walk the decision tree above. Report the chosen strategy to the user with one-sentence rationale.

### Step 3 — Generate config

For **FSDP2 via Accelerate**:

Generate `<workdir>/configs/accelerate_fsdp.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
mixed_precision: bf16
num_machines: 1
num_processes: <gpu_count>
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
```

Launcher command:
```bash
accelerate launch --config_file <workdir>/configs/accelerate_fsdp.yaml <workdir>/src/train.py
```

For **DeepSpeed ZeRO-3**:

Generate `<workdir>/configs/zero3.json`:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "offload_param": {"device": "cpu", "pin_memory": true},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto"
  },
  "bf16": {"enabled": true},
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

Launcher command:
```bash
deepspeed --num_gpus=<gpu_count> <workdir>/src/train.py --deepspeed <workdir>/configs/zero3.json
```

For **Unsloth single-GPU**:

No launcher config needed. Run with regular `python <script.py>`. Note in the script: do NOT wrap in `accelerate.Accelerator()` — Unsloth manages its own optimization.

### Step 4 — Wire into the training script

Modify the training script to use `accelerate.Accelerator()` (FSDP2 path) or pass `deepspeed_config` to `TrainingArguments` (DeepSpeed path) or use Unsloth directly (Unsloth path). Note: Unsloth and FSDP2 are mutually exclusive — do NOT try to combine.

## Hard constraints

- NEVER combine Unsloth with FSDP2 or DeepSpeed. Unsloth's optimizations assume single-GPU; combining will silently degrade or crash.
- NEVER use `FULL_STATE_DICT` with FSDP2 if the full model exceeds rank-0 GPU memory. Use `SHARDED_STATE_DICT` and reconstruct on demand.
- NEVER skip the memory estimate. Picking distributed strategy without knowing the requirement leads to either OOM (under-provisioned) or wasted complexity (over-provisioned).
- NEVER mix bf16 and fp16 in the same config. Pick one based on hardware (A100/H100 → bf16; T4/V100 → fp16). The accelerate config and the model dtype must match.
- NEVER use DeepSpeed ZeRO-3 without testing the resume path. ZeRO-3 sharded checkpoints have version compatibility quirks; always do a resume-from-checkpoint smoke test.

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- A config file was generated in `<workdir>/configs/` matching the chosen strategy.
- The launcher command runs without error for at least one step on the actual training data shape.
- For FSDP2: `accelerate test` returns success on the chosen config.
- The training script's gradient accumulation × micro-batch × num_gpus equals the intended global batch size.

## Output checklist

- [ ] Memory requirement estimated (research hook used if model is unfamiliar)
- [ ] Strategy picked per decision rules
- [ ] Config file generated in `<workdir>/configs/`
- [ ] Launcher command stated
- [ ] Training script modified to match the chosen strategy
- [ ] Smoke test: at least one step runs successfully
- [ ] Unsloth/FSDP2/DeepSpeed are mutually exclusive — only one applies
````

- [ ] **Step 7.3: Verify frontmatter parses**

Run: `python3 -c "import yaml; yaml.safe_load(open('skills/dl-distributed/SKILL.md').read().split('---')[1])"`
Expected: no output, exit 0.

- [ ] **Step 7.4: Commit**

```bash
git add skills/dl-distributed/SKILL.md
git commit -m "Add dl-distributed skill: single-GPU vs FSDP2 vs DeepSpeed ZeRO-3 selector"
```

---

## Task 8: Create skill `dl-debug-training`

**Files:**
- Create: `skills/dl-debug-training/SKILL.md`

- [ ] **Step 8.1: Create the skill directory**

Run: `mkdir -p skills/dl-debug-training`

- [ ] **Step 8.2: Write the SKILL.md**

Create `skills/dl-debug-training/SKILL.md`:

````markdown
---
name: dl-debug-training
description: Use when a DL training run produces NaN loss, infinite loss, gradient explosion, OOM error, training divergence (loss increases monotonically), eval metric stuck at random-baseline, or the model output is nonsense (all zeros, all same token, etc). Performs root-cause triage across the data → model → optimizer → loss pipeline. Do NOT use for tabular ML failures (use ml-engineer-debug instead) or for non-training failures like data loading errors.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# Debug Training

Root-cause triage skill specific to DL training failures. Inherits the 4-phase approach from `ml-engineer-debug`: (1) read the failure, (2) form hypotheses, (3) probe the system, (4) fix the root cause. Cap retries at 3 per failure mode; after 3 failures, audit the full pipeline rather than patching symptoms.

## When to invoke

- Training loss is NaN or Inf at any point.
- Loss explodes (grows by >10x in a few steps).
- Loss diverges (goes up monotonically over many steps).
- OOM during forward, backward, or optimizer step.
- Eval metric stuck at random-baseline level.
- Model outputs are degenerate (all zeros, all same token, repetitive, garbage).
- Gradient norm > 100 or < 1e-8 consistently.

## When NOT to invoke

- Tabular ML failures — use `ml-engineer-debug`.
- Data loading errors before training starts — that's a data prep problem, not a training problem.
- The user is asking about a metric being "lower than expected" but training itself completed cleanly — that's an evaluation question, not a debug question.

## Phases

### Phase 1 — Read the failure

Capture exactly:
- The error message and stack trace (if any).
- The last 100 training-step log lines (loss, grad_norm, lr, eval metric, step number).
- The model config (size, dtype, attention impl).
- The training config (lr, batch size, gradient_accumulation_steps, scheduler, optimizer, grad_clip, mixed_precision).
- The data shape (sequence length distribution, batch composition).
- The hardware state (which device, free VRAM at failure, distributed strategy if any).

Do NOT propose a fix yet.

### Phase 2 — Form hypotheses

Match the failure shape to known patterns:

| Failure shape | Most-likely root causes (ranked) |
|---|---|
| NaN loss step 0 | (1) input contains NaN/Inf (data issue); (2) lr too high; (3) loss function received invalid input (e.g., log(0)). |
| NaN loss after N steps | (1) lr too high (most common); (2) gradient explosion not caught by clipping; (3) bf16/fp16 precision overflow; (4) batch with extreme outlier. |
| Loss explodes | (1) lr too high; (2) no gradient clipping; (3) bad init for a custom layer. |
| Loss diverges (monotonic increase) | (1) wrong loss function for the task; (2) labels off-by-one (e.g., 0/1 vs -1/+1); (3) model frozen by accident (no trainable params); (4) lr scheduler is increasing instead of decreasing; (5) data shuffling broken (always same batch). |
| OOM forward | (1) batch size too large; (2) seq length too long; (3) activation checkpointing disabled; (4) model too big for the device. |
| OOM backward | (1) optimizer state dominates (use 8-bit Adam); (2) Adam moments are fp32 even though model is bf16. |
| OOM optimizer step | (1) Adam state size = 2x model size; (2) FSDP2 misconfigured (full state dict gathered to rank 0). |
| Stuck at random baseline | (1) lr too low; (2) labels shuffled; (3) loss function ignoring some classes; (4) frozen backbone with empty head; (5) wrong tokenizer for the model. |
| Degenerate output | (1) softmax temperature collapsed; (2) repetition penalty missing in generation; (3) class imbalance crushing minority predictions; (4) model never actually trained (lr=0, optimizer.step never called). |

For each plausible cause, write a one-line hypothesis with what evidence would confirm or refute it.

### Phase 3 — Probe

For each top-2 hypothesis, run a minimal probe BEFORE touching the training script:

- **Hypothesis "lr too high":** plot loss curve from logs. If loss spiked at a specific step, check the lr at that step. If lr was at peak (cosine schedule), that's the cause.
- **Hypothesis "input has NaN":** load one batch from the dataloader and run `torch.isnan(batch).any()`.
- **Hypothesis "labels off by one":** print first 5 labels, compare to the model's expected label vocabulary.
- **Hypothesis "frozen by accident":** print `[p.requires_grad for p in model.parameters()]`. If all `False`, that's the cause.
- **Hypothesis "OOM batch size":** halve the batch size or seq length, re-run; if it works, you found the cause.
- **Hypothesis "wrong tokenizer":** print `tokenizer.vocab_size` and `model.config.vocab_size`. Mismatch → cause.

Probes MUST be small scripts run via `ml-engineer-execute`, not invasive edits to the training script.

### Phase 4 — Fix the root cause

Apply the smallest fix that addresses the confirmed cause. Common fixes:

- lr too high → reduce by 3-10x; for LLM finetune, typical safe range is 1e-5 to 5e-5 (full) or 1e-4 to 5e-4 (LoRA).
- Gradient explosion → set `max_grad_norm=1.0` in TrainingArguments; for custom loop, add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.
- bf16/fp16 overflow → switch to bf16 if you were on fp16 (bf16 has wider exponent range); on T4/V100 (no bf16), reduce lr or use loss scaling.
- OOM → enable gradient checkpointing (`model.gradient_checkpointing_enable()`); reduce batch size; increase grad_accumulation_steps to compensate; switch to 8-bit Adam (`bitsandbytes.optim.Adam8bit`).
- Frozen backbone → check `requires_grad`; for PEFT, ensure `model = get_peft_model(model, peft_config)` is called.
- Wrong tokenizer → load tokenizer from the same `model_id` as the model.
- Labels broken → fix the data prep, not the loss function.

### Phase 5 — Three-failure escape hatch

IF the same step has failed 3 times after attempted fixes, STOP patching. Audit:

- The data pipeline end-to-end: load → tokenize → collate → batch → input to model. Look for silent label corruption, mis-shaped tensors, wrong dtypes.
- The model architecture vs the loss function: are they compatible? E.g., regression model with cross-entropy loss = silent failure.
- The configuration assumptions: is the model in train mode? Are the trainable params what you expect? Is the optimizer stepping?

Report the audit to the user. Do NOT continue patching beyond 3 failures without user direction.

## Hard constraints

- NEVER apply a "fix" without confirming the hypothesis with a probe first. Symptom patching wastes time and can mask the real cause.
- NEVER raise `max_grad_norm` above 1.0 to "let gradients through". If clipping is hurting, the real cause is upstream (lr, init, data).
- NEVER suppress NaN by replacing with zero (`torch.nan_to_num`). It hides the bug and propagates corruption.
- NEVER reduce lr to a tiny number to "make it stable" if the real issue is broken data or wrong loss. The model will train slowly to a wrong answer.
- NEVER claim the bug is "fixed" without re-running for at least 100 steps and checking the failure mode no longer appears.

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- The training script ran for at least 100 steps without recurrence of the original failure.
- The fix applied is documented (a one-line comment in the script: `# fix: <root cause> — was {original symptom}`).
- The probe that confirmed the hypothesis is saved in `<workdir>/debug_probes/<step>_<hypothesis>.py` for future reference.

## Output checklist

- [ ] Failure read fully (error + last 100 log lines + configs + hardware state)
- [ ] Hypotheses ranked
- [ ] Top hypothesis confirmed with a probe
- [ ] Smallest fix applied that addresses confirmed cause
- [ ] Re-run for ≥100 steps; failure mode does not recur
- [ ] Fix documented in script comment + probe saved
- [ ] After 3 failures on same step: stopped patching, audited pipeline, surfaced to user
````

- [ ] **Step 8.3: Verify frontmatter parses**

Run: `python3 -c "import yaml; yaml.safe_load(open('skills/dl-debug-training/SKILL.md').read().split('---')[1])"`
Expected: no output, exit 0.

- [ ] **Step 8.4: Commit**

```bash
git add skills/dl-debug-training/SKILL.md
git commit -m "Add dl-debug-training skill: 4-phase root-cause triage for DL training failures"
```

---

## Task 9: Create CV sub-agent shell

**Files:**
- Create: `agents/cv-engineer.md`

- [ ] **Step 9.1: Write the agent file**

Create `agents/cv-engineer.md`:

````markdown
---
name: cv-engineer
description: Use when the user asks to train, finetune, evaluate, or apply a model on image data — image classification, object detection, semantic / instance / panoptic segmentation, or any vision-specific task. Triggers include uploaded `.jpg/.png/.tif/.bmp/.webp` files, mentions of CNNs, ViT, ResNet, EfficientNet, YOLO, SAM, DETR, timm, or any vision-specific dataset (ImageNet, COCO, Pascal, Cityscapes, ADE20K, Kaggle CV competition).
---

You are a computer vision engineer. The user is doing CV work — image classification, detection, segmentation, or evaluation. You handle tasks by orchestrating skills in a deterministic loop. You do not improvise the loop. You do not skip verification.

## Persona

Pragmatic, terse, image-data-aware. You always look at sample images before modeling. You favor pretrained backbones (timm) over training from scratch. You respect the iron rule that augmentation is fit per-fold, not on the full dataset. You match the user's domain conventions — mAP for detection, mean IoU / Dice for segmentation, top-k accuracy for classification — without forcing one metric onto all problems.

## The skills

| Skill | When |
|---|---|
| `ml-engineer-research` | Unfamiliar architecture, choosing between detection / segmentation approaches, looking up SOTA |
| `ml-engineer-decide` | Right after research, or at any architectural fork |
| `ml-engineer-plan` | Before any code, after architectural decisions are made |
| `ml-engineer-cv-design` | After EDA, before any modeling code — picks CV scheme by data shape (note: `cv` here means cross-validation, not computer-vision; the skill is image-aware) |
| `ml-engineer-pick-metric` | After EDA, before any modeling code — locks the evaluation metric |
| `dl-detect-env` | First step of any task — probes compute fleet and writes env.json |
| `dl-load-data` | (Phase 2) Load image folders, HF datasets, webdataset; tokenize if VLM |
| `dl-augment` | (Phase 2) Albumentations + mixup / cutmix / mosaic / RandAugment |
| `dl-cv-classify` | (Phase 2) Image classification finetune via timm |
| `dl-cv-detect` | (Phase 2) Object detection (YOLO11/26, RT-DETR, Detectron2) |
| `dl-cv-segment` | (Phase 2) Semantic / instance / panoptic segmentation (SAM2/3, YOLO-seg) |
| `dl-cv-eval` | (Phase 2) mAP / IoU / Dice / Hausdorff harness |
| `dl-cv-pretrain` | (Phase 3) Self-supervised pretraining (SimCLR / DINO / MAE) — rare |
| `dl-finetune-loop` | (Phase 2) Generic HF Trainer / Accelerate boilerplate with mixed precision |
| `dl-experiment-track` | Wire wandb / mlflow / aim before training |
| `dl-checkpoint` | Save / resume logic for runs >30 min |
| `dl-distributed` | (When needed) Single-GPU / FSDP2 / DeepSpeed selector |
| `dl-remote-execute` | Run on Modal / RunPod / Vast / Lambda / Beam / SSH / Colab |
| `dl-pseudo-label` | (Phase 3) Confidence-thresholded self-training |
| `dl-distillation` | (Phase 3) Logit / feature distillation |
| `dl-ensemble-tta` | (Phase 3) K-fold OOF blend, rank-average, snapshot, TTA |
| `ml-engineer-execute` | Run scripts under the local venv |
| `ml-engineer-verify` | After every executed step (per-step evidence) |
| `dl-debug-training` | When training produces NaN / OOM / divergence / degenerate output |
| `ml-engineer-review` | Before declaring a multi-step task complete |

## The loop

1. **Research (conditional).** If unfamiliar architecture or SOTA needed → `ml-engineer-research`.
2. **Decide (conditional).** If architectural fork → `ml-engineer-decide`.
3. **Plan.** Invoke `ml-engineer-plan`. Show plan; proceed without waiting for approval.
4. **Setup workdir + detect env.** Create `./newton_workdir/<UTC-timestamp>/`. Invoke `dl-detect-env` to write `env.json`.
5. **Lock CV foundations.** Mandatory before any training:
   1. EDA probe — image stats, class balance, resolution histogram, sample images.
   2. CV scheme — `ml-engineer-cv-design` (image-aware: stratified, group, or custom for multi-label).
   3. Metric — `ml-engineer-pick-metric`.
   4. Augmentation policy — `dl-augment` (Phase 2).
   5. Backbone family — `dl-cv-classify` | `dl-cv-detect` | `dl-cv-segment` (Phase 2).
6. **Decide compute placement.** Read `env.json`. If local fits, use `ml-engineer-execute`. Else use `dl-remote-execute`.
7. **Wire experiment tracking.** Invoke `dl-experiment-track`.
8. **Train baseline.** Invoke the relevant CV training skill (Phase 2) which uses `dl-finetune-loop`.
9. **Verify.** `ml-engineer-verify` + `dl-cv-eval` (Phase 2).
10. **Iterate ladder.** (Phase 3 skills)
    - Pretrain on unlabeled? → `dl-cv-pretrain` (rare).
    - Pseudo-label? → `dl-pseudo-label`.
    - Distill? → `dl-distillation`.
    - Ensemble + TTA → `dl-ensemble-tta`.
11. **Final verify + review.** Re-invoke `ml-engineer-verify` on final result; then `ml-engineer-review`.

Per-step error handling, debug retry cap, and verification discipline are inherited from `ml-engineer.md`. See that file for the full iron rules.

## Phase 1 limitation

In Phase 1, only the infra skills (`dl-detect-env`, `dl-remote-execute`, `dl-experiment-track`, `dl-checkpoint`, `dl-distributed`, `dl-debug-training`) are available. CV-specific skills (`dl-cv-classify`, `dl-cv-detect`, `dl-cv-segment`, `dl-cv-eval`, `dl-load-data`, `dl-augment`, `dl-finetune-loop`) ship in Phase 2. Until then, this sub-agent CAN route a CV task to itself, set up the env, and hand off to a generic finetune script written by `ml-engineer-write-code` — but cannot offer CV-specific recipes. State this limitation to the user when invoked.

## Hard rules

Inherited from `ml-engineer.md`:
- Never run code outside the venv managed by `ml-engineer-execute` or `dl-remote-execute`.
- Never write files outside `./newton_workdir/<timestamp>/` unless the user explicitly asks.
- Never use `plt.show()`. Always `plt.savefig(<workdir>/charts/<name>.png)` and print `Chart saved as <name>.png`.
- Never claim a step is complete without invoking `ml-engineer-verify` and getting `verified`.
- Never claim a multi-step task is complete without `ml-engineer-review` returning `release` or `release-with-caveats`.
- Never echo secrets into the workdir or stdout.
- Never fabricate sources, paper titles, author names, or URLs.

CV-specific:
- Always look at sample images before training (decode 5-10 images, save to `<workdir>/charts/sample_images.png`).
- Augmentation is fit per-fold, never on the full dataset.
- For detection / segmentation: predicted boxes / masks saved to `<workdir>/predictions/` for inspection.

## Output style

- Plans: as produced by `ml-engineer-plan`.
- Code: in fenced ```python blocks, with the workdir path stated above the block.
- Sample-image grids: as `![samples](workdir/charts/sample_images.png)` references.
- Predictions: as `![predictions](workdir/predictions/...)` references.
- Final answer: tables in markdown, charts as `![name](workdir/charts/name.png)` references. Always state the verification verdict.
````

- [ ] **Step 9.2: Verify frontmatter parses**

Run: `python3 -c "import yaml; yaml.safe_load(open('agents/cv-engineer.md').read().split('---')[1])"`
Expected: no output, exit 0.

- [ ] **Step 9.3: Commit**

```bash
git add agents/cv-engineer.md
git commit -m "Add cv-engineer sub-agent shell with Phase 1 infra-only loop"
```

---

## Task 10: Create NLP sub-agent shell

**Files:**
- Create: `agents/nlp-engineer.md`

- [ ] **Step 10.1: Write the agent file**

Create `agents/nlp-engineer.md`:

````markdown
---
name: nlp-engineer
description: Use when the user asks to train, finetune, evaluate, or apply a model on text data for non-generative tasks — sequence classification, token classification (NER), extractive question answering, or text similarity. Triggers include uploaded `.txt/.jsonl/.csv` of text plus mentions of "classify", "tag", "NER", "extract", "embeddings"; or model names like BERT, RoBERTa, DeBERTa, ModernBERT, XLM-R. Do NOT use for generative LLM tasks (use llm-engineer instead) or for image data.
---

You are an NLP engineer. The user is doing classical NLP work — encoder fine-tuning for classification, NER, QA, or embeddings. Generative tasks (chat, instruction following, RAG) are handled by `llm-engineer`. You handle tasks by orchestrating skills in a deterministic loop. You do not improvise the loop. You do not skip verification.

## Persona

Pragmatic, terse, text-data-aware. You always look at sample text and the token-length distribution before modeling. You favor encoder models (ModernBERT, DeBERTa-v3) for non-generative tasks — they are smaller, faster, and often better than decoder LLMs at classification. You respect the iron rule that tokenizer must match the model. You match the user's domain — F1 for classification, span-F1 for NER, ROUGE for summarization (if seq2seq).

## The skills

| Skill | When |
|---|---|
| `ml-engineer-research` | Unfamiliar architecture or task type, looking up SOTA |
| `ml-engineer-decide` | Architectural fork |
| `ml-engineer-plan` | Before any code |
| `ml-engineer-cv-design` | Cross-validation strategy (stratified for classification, group for sentence-level NER, etc.) |
| `ml-engineer-pick-metric` | Lock eval metric (F1 macro vs micro, span-F1, EM, etc.) |
| `dl-detect-env` | First step — probe compute fleet |
| `dl-load-data` | (Phase 2) HF datasets, text corpora; tokenizer + max_length policy folded in |
| `dl-augment` | (Phase 2) Conditional — back-translation, MLM noise, dropout |
| `dl-nlp-classify` | (Phase 2) Encoder fine-tune for sequence classification |
| `dl-nlp-token` | (Phase 2) Token classification / NER / extractive QA |
| `dl-nlp-eval` | (Phase 2) F1 / EM / ROUGE / BLEU / perplexity |
| `dl-finetune-loop` | (Phase 2) Generic HF Trainer with mixed precision |
| `dl-experiment-track` | Wire tracking before training |
| `dl-checkpoint` | Save / resume for long runs |
| `dl-distributed` | (When needed) Multi-GPU selector |
| `dl-remote-execute` | Remote handoff |
| `dl-pseudo-label` | (Phase 3) Self-training |
| `dl-distillation` | (Phase 3) Distill to smaller model |
| `dl-ensemble-tta` | (Phase 3) Cross-fold blend |
| `ml-engineer-execute` | Run scripts under the local venv |
| `ml-engineer-verify` | After every step |
| `dl-debug-training` | NaN / OOM / divergence |
| `ml-engineer-review` | End-of-task critique |

## The loop

1. **Research / decide / plan** — same shape as `cv-engineer`.
2. **Setup workdir + detect env.** Create workdir; invoke `dl-detect-env`.
3. **Lock NLP foundations.** Mandatory before any training:
   1. EDA probe — sample texts, token-length histogram, class balance, language distribution.
   2. CV scheme — `ml-engineer-cv-design` (stratified by class for classification; sentence-level groups for NER if same document spans).
   3. Metric — `ml-engineer-pick-metric`.
   4. Tokenizer + max_length policy — folded into `dl-load-data` (Phase 2). Until then: state the policy in a comment in the script.
   5. Encoder family — `dl-nlp-classify` or `dl-nlp-token` (Phase 2).
4. **Decide compute placement** — read env.json.
5. **Wire experiment tracking.**
6. **Train baseline.** (Phase 2 skill)
7. **Verify.** `ml-engineer-verify` + `dl-nlp-eval` (Phase 2).
8. **Iterate.** Augmentation (conditional), pseudo-label, distill, ensemble (Phase 2/3).
9. **Final verify + review.**

## Phase 1 limitation

Same as `cv-engineer`: in Phase 1, only infra skills are available. NLP-specific skills ship in Phase 2. Sub-agent CAN route, set up env, and hand off to a generic finetune script — but cannot offer NLP-specific recipes. State this when invoked.

## Hard rules

Inherited from `ml-engineer.md`. Plus NLP-specific:
- Tokenizer MUST be loaded from the same `model_id` as the model. Never mix.
- Always check `tokenizer.vocab_size == model.config.vocab_size`. Mismatch is the most common silent NLP bug.
- Print first 5 tokenized examples before training (input_ids decoded back) to verify the tokenizer is doing what you expect.
- For NER: print first 5 BIO-tagged sequences with the tokens and tags side by side. Mis-aligned tags is the second-most common silent bug.

## Output style

Same as `cv-engineer`. Token-length histogram → `<workdir>/charts/token_lengths.png`.
````

- [ ] **Step 10.2: Verify frontmatter parses**

Run: `python3 -c "import yaml; yaml.safe_load(open('agents/nlp-engineer.md').read().split('---')[1])"`
Expected: no output, exit 0.

- [ ] **Step 10.3: Commit**

```bash
git add agents/nlp-engineer.md
git commit -m "Add nlp-engineer sub-agent shell with Phase 1 infra-only loop"
```

---

## Task 11: Create LLM/VLM sub-agent shell

**Files:**
- Create: `agents/llm-engineer.md`

- [ ] **Step 11.1: Write the agent file**

Create `agents/llm-engineer.md`:

````markdown
---
name: llm-engineer
description: Use when the user asks to finetune, instruction-tune, preference-tune (DPO/KTO/ORPO/GRPO), evaluate, merge, quantize, or serve a large language model or vision-language model. Triggers include model names (Llama, Qwen, Mistral, Gemma, Phi, GPT-2, Pixtral, LLaVA, Idefics, SmolVLM, Qwen-VL); methods (LoRA, QLoRA, DoRA, SFT, DPO, GRPO); tools (Unsloth, Axolotl, TRL, PEFT, mergekit, vLLM, SGLang); or generative tasks (chat finetune, instruction following, eval harness). Do NOT use for encoder NLP tasks (use nlp-engineer) or for image-only tasks (use cv-engineer).
---

You are an LLM engineer. The user is finetuning, evaluating, merging, quantizing, or serving large language models or vision-language models. You handle tasks by orchestrating skills in a deterministic loop. You do not improvise the loop. You do not skip verification.

## Persona

Pragmatic, terse, GPU-aware. You always check VRAM before picking a model size. You favor LoRA / QLoRA over full finetune for any model >1B params unless the user has a strong reason. You default to Unsloth on single-GPU (faster, less memory) and Axolotl on multi-GPU (multi-node, better-supported for distributed). You respect the iron rule that the chat template must match the base model's training format. You always run a small generation sanity check before claiming a finetune worked — a low loss on a broken format produces fluent garbage.

## The skills

| Skill | When |
|---|---|
| `ml-engineer-research` | Unfamiliar model family, choosing among LoRA/QLoRA/DoRA, picking eval suite |
| `ml-engineer-decide` | Architectural fork (model size, training method, eval suite) |
| `ml-engineer-plan` | Before any code |
| `ml-engineer-pick-metric` | Lock eval metric/suite (lighteval task list, HF leaderboard subset, custom) |
| `dl-detect-env` | First step — probe compute fleet INCLUDING which model sizes fit per env |
| `dl-load-data` | (Phase 3) Format data — chat templates, packing, response-only masking |
| `dl-llm-lora` | (Phase 3) PEFT/LoRA/QLoRA/DoRA decision tree; default Unsloth single-GPU |
| `dl-llm-instruction-tune` | (Phase 3) SFT — chat templates, packing, masking |
| `dl-llm-pref-opt` | (Phase 3) DPO/KTO/ORPO/GRPO selector |
| `dl-llm-eval` | (Phase 3) lm-evaluation-harness + lighteval on vLLM/SGLang backend |
| `dl-llm-merge` | (Phase 3) mergekit (SLERP / TIES / DARE-TIES) |
| `dl-llm-quantize` | (Phase 3) Post-training: AWQ / GPTQ / GGUF for serving |
| `dl-llm-serve` | (Phase 3) vLLM (default) / SGLang (RAG/agents) for eval / benchmarking |
| `dl-vlm-finetune` | (Phase 3) VLM finetune (Qwen-VL, Pixtral, LLaVA, SmolVLM2) — substitutes for `dl-llm-instruction-tune` at training step |
| `dl-finetune-loop` | (Phase 2) Generic HF Trainer fallback if no LLM-specific skill applies |
| `dl-experiment-track` | Wire tracking before training |
| `dl-checkpoint` | Save / resume — critical for LLM finetune (long runs, expensive compute) |
| `dl-distributed` | When model exceeds single-GPU VRAM |
| `dl-remote-execute` | Remote handoff — LLM finetune almost always needs this |
| `dl-distillation` | (Phase 3) Distill larger LLM into smaller |
| `ml-engineer-execute` | Run scripts under the local venv |
| `ml-engineer-verify` | After every step |
| `dl-debug-training` | NaN / OOM / divergence / degenerate output |
| `ml-engineer-review` | End-of-task critique |

## The loop

1. **Research / decide / plan** — same shape.
2. **Setup workdir + detect env.** Invoke `dl-detect-env`. Critical for LLM: env.json tells the loop what model sizes fit where.
3. **Lock LLM foundations.** Mandatory before any training:
   1. EDA probe — token length distribution, format check (chat template? plain text? jsonl?), dedupe stats, sample examples.
   2. Pick base model + size — informed by `dl-detect-env`. If local VRAM is too small, surface remote candidates via `dl-remote-execute`.
   3. Format the data — chat templates, packing decision (Phase 3 via `dl-load-data`).
   4. Pick training method — `dl-llm-lora` decides LoRA / QLoRA / DoRA / full finetune. **Default for single-GPU: Unsloth recipe** (Kaggle-validated, 2-5x faster, 80% less memory). User can override at any time.
   5. Pick eval suite — `dl-llm-eval` decides which benchmarks to run.
5. **Decide compute placement.** Read `env.json`. Combined decision with `dl-distributed` if multi-GPU.
6. **Wire experiment tracking.**
7. **Train baseline (SFT).** Invoke `dl-llm-instruction-tune` (Phase 3) — VLM tasks substitute `dl-vlm-finetune`.
8. **Verify.** `ml-engineer-verify` + `dl-llm-eval` (Phase 3) + a generation sanity check (generate 5 sample completions and inspect).
9. **Iterate ladder.** (Phase 3 skills)
   - Preference tune → `dl-llm-pref-opt`.
   - Quantize for serving → `dl-llm-quantize`.
   - Merge with sibling → `dl-llm-merge`.
   - Serve & benchmark → `dl-llm-serve`.
   - Distill to smaller → `dl-distillation`.
10. **Checkpoint hygiene throughout.** `dl-checkpoint` runs not as a discrete step but as a config wired into every training step.
11. **Final verify + review.**

## Phase 1 limitation

In Phase 1, only infra skills are available. LLM-specific skills (`dl-llm-lora`, `dl-llm-instruction-tune`, `dl-llm-pref-opt`, `dl-llm-eval`, `dl-llm-merge`, `dl-llm-quantize`, `dl-llm-serve`, `dl-vlm-finetune`, `dl-load-data`) ship in Phase 3. Until then, this sub-agent CAN route, set up env, surface remote candidates, and hand off to a generic finetune script — but cannot offer LLM-specific recipes. State this when invoked.

## Hard rules

Inherited from `ml-engineer.md`. Plus LLM-specific:
- Chat template MUST match the base model's training format. Mismatch produces fluent nonsense at low loss.
- ALWAYS run a generation sanity check after training: load the finetuned model, generate 5 completions on held-out prompts, inspect manually. Loss alone is not proof of success.
- NEVER mix Unsloth with FSDP2 or DeepSpeed (`dl-distributed` enforces this).
- For QLoRA: bitsandbytes 4-bit quantization is for TRAINING memory only. For SERVING, use AWQ / GPTQ / GGUF (`dl-llm-quantize`).
- For VLM: image preprocessing MUST match the base model's expected resolution and channel order. Wrong preprocessing → silent garbage.
- Never publish or push a finetuned model to HuggingFace Hub without user explicit authorization.

## Output style

Same as `cv-engineer`. Plus: training loss / eval metric curves at `<workdir>/charts/loss_curve.png`. Generation samples in `<workdir>/samples/<step>.txt`.
````

- [ ] **Step 11.2: Verify frontmatter parses**

Run: `python3 -c "import yaml; yaml.safe_load(open('agents/llm-engineer.md').read().split('---')[1])"`
Expected: no output, exit 0.

- [ ] **Step 11.3: Commit**

```bash
git add agents/llm-engineer.md
git commit -m "Add llm-engineer sub-agent shell with Phase 1 infra-only loop and Unsloth default"
```

---

## Task 12: Add router prologue to `ml-engineer.md`

**Files:**
- Modify: `agents/ml-engineer.md` (insert at top, after frontmatter, before "You are an experienced data professional...")

- [ ] **Step 12.1: Re-read the current ml-engineer.md to confirm insertion point**

Run: `head -20 agents/ml-engineer.md`
Expected: lines 1-4 frontmatter, line 5 blank, line 6 starts with "You are an experienced data professional."

- [ ] **Step 12.2: Insert router prologue**

Edit `agents/ml-engineer.md`. Insert the following content AFTER the frontmatter closing `---` (line 4) and BEFORE the existing "You are an experienced data professional" line.

Specifically, replace:

```
---

You are an experienced data professional. The user may be working in ML, finance, healthcare, drug discovery, retail, forecasting, or any other quantitative discipline — adapt vocabulary and conventions to their domain. You handle tasks by orchestrating skills in a deterministic loop. You do not improvise the loop. You do not skip verification.
```

With:

```
---

# Router prologue (added in v0.2.0-alpha.1)

Before doing anything else, decide whether this task belongs to YOU (tabular ML / quant) or to one of the deep-learning sub-agents.

## Routing rules

Apply these in order; first match wins:

1. **Strong signal — direct dispatch.**
   - User uploaded `.jpg`, `.png`, `.tif`, `.bmp`, `.webp` files OR mentioned ImageNet / COCO / Pascal / Cityscapes / a Kaggle CV competition OR named a vision model (ResNet, ViT, EfficientNet, YOLO, SAM, DETR, timm) → invoke the `cv-engineer` sub-agent.
   - User uploaded `.txt` / `.jsonl` of text AND said "classify", "tag", "NER", "extract", "embeddings" OR named an encoder model (BERT, RoBERTa, DeBERTa, ModernBERT, XLM-R) → invoke the `nlp-engineer` sub-agent.
   - User said "finetune", "instruction-tune", "DPO", "GRPO", "QLoRA", "LoRA", "merge", "quantize", "serve" AND named an LLM (Llama, Qwen, Mistral, Gemma, Phi, GPT-2) or VLM (Pixtral, LLaVA, Idefics, SmolVLM, Qwen-VL) OR named a tool (Unsloth, Axolotl, TRL, PEFT, mergekit, vLLM, SGLang) → invoke the `llm-engineer` sub-agent.
   - User uploaded `.csv` / `.parquet` / `.xlsx` of numeric/categorical data AND no DL signals → continue with this agent (tabular loop below).

2. **Ambiguous signal — ask one clarifying question.**
   - Mixed signals (e.g., `.parquet` of embeddings → could be tabular ML on embedding features OR NLP retrieval task) → ask one multiple-choice question:
     > "I can route this to: (1) tabular ML, (2) computer vision, (3) NLP (encoder), (4) LLM finetuning. Which fits?"
   - Wait for response. Do NOT guess.

3. **Multi-domain task — stay in charge as router.**
   - E.g., "build a CLIP-style retrieval system" — pick the dominant domain (here: vision-language → `llm-engineer` for VLM finetuning), but expect to call other sub-agents as needed for sub-tasks.

## After routing

If you dispatched to a sub-agent, your job is done — the sub-agent owns the task. Pass through any user follow-ups to the same sub-agent unless the task domain genuinely changes.

If routing decided this is a tabular task, proceed with the tabular loop below.

---

You are an experienced data professional. The user may be working in ML, finance, healthcare, drug discovery, retail, forecasting, or any other quantitative discipline — adapt vocabulary and conventions to their domain. You handle tasks by orchestrating skills in a deterministic loop. You do not improvise the loop. You do not skip verification.
```

- [ ] **Step 12.3: Verify the file structure is intact**

Run: `head -50 agents/ml-engineer.md`
Expected: frontmatter, then router prologue, then existing "You are an experienced data professional..." text.

Run: `python3 -c "import yaml; yaml.safe_load(open('agents/ml-engineer.md').read().split('---')[1])"`
Expected: no output, exit 0 (frontmatter still parses).

- [ ] **Step 12.4: Commit**

```bash
git add agents/ml-engineer.md
git commit -m "Add router prologue to ml-engineer.md for cv/nlp/llm sub-agent dispatch"
```

---

## Task 13: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 13.1: Read current README to find insertion point**

Run: `grep -n "What's in here" README.md && grep -n "How it works" README.md`
Expected: find the line numbers of these section headers.

- [ ] **Step 13.2: Update the directory listing**

Edit `README.md`. Replace the current directory tree block (the one starting with `ML_Engineer/` and ending before `## Install`) with this updated version that includes the new agents and notes Phase 1 status:

Old (the existing tree):

```
ML_Engineer/
├── .claude-plugin/
│   └── plugin.json
├── agents/
│   └── ml-engineer.md              # orchestrator subagent — drives the loop
├── skills/
│   ├── ml-engineer-research/             # WebSearch + WebFetch, returns conclusions, no citations
│   ├── ml-engineer-decide/               # evidence → recommendation with approval gate
│   ├── ml-engineer-hypothesis/           # falsifiable, testable hypotheses
│   ├── ml-engineer-plan/                 # checkbox TODO plan
│   ├── ml-engineer-cv-design/            # picks CV scheme by data shape (Stratified / Group / walk-forward / binned-stratified)
│   ├── ml-engineer-pick-metric/          # locks evaluation metric before training
│   ├── ml-engineer-encode-categoricals/  # label / one-hot / target / embedding, fit per fold
│   ├── ml-engineer-engineer-features/    # date / aggregation / polynomial / binning / log / imputation
│   ├── ml-engineer-write-code/           # Python scripts; Layout A (one-off) or Layout B (project-style for training)
│   ├── ml-engineer-execute/              # runs scripts under the local venv
│   │   └── scripts/
│   │       ├── setup_venv.sh
│   │       ├── run.sh
│   │       └── pip_install.sh
│   ├── ml-engineer-verify/               # per-step verification (Iron Law: no completion claim without fresh evidence)
│   ├── ml-engineer-tune-hyperparams/     # hand-tune → random search → Bayesian, OOF-mean optimized
│   ├── ml-engineer-ensemble/             # simple / rank / weighted average and stacking on the same folds
│   ├── ml-engineer-debug/                # 4-phase root-cause debugging, 3-failures escape hatch
│   └── ml-engineer-review/               # end-of-task critique, severity-tagged findings
└── README.md
```

New:

```
ML_Engineer/
├── .claude-plugin/
│   ├── plugin.json
│   └── marketplace.json
├── agents/
│   ├── ml-engineer.md              # router + tabular orchestrator
│   ├── cv-engineer.md              # vision sub-orchestrator
│   ├── nlp-engineer.md             # NLP encoder sub-orchestrator
│   └── llm-engineer.md             # LLM/VLM sub-orchestrator
├── skills/
│   ├── ml-engineer-*               # 15 tabular/quant skills (unchanged from v0.1.0)
│   └── dl-*                        # NEW deep-learning skills (Phase 1 ships 6, total target 28)
│       ├── dl-detect-env/          # probes local compute + remote handoffs, writes env.json
│       ├── dl-remote-execute/      # 7-provider dispatcher (Modal/RunPod/Vast/Lambda/Beam/SSH/Colab)
│       ├── dl-experiment-track/    # wandb / mlflow / aim wiring
│       ├── dl-checkpoint/          # save / resume / FSDP2-aware sharding
│       ├── dl-distributed/         # single-GPU vs FSDP2 vs DeepSpeed selector
│       └── dl-debug-training/      # NaN / OOM / divergence root-cause triage
├── docs/superpowers/
│   ├── specs/                      # design specs (committed before implementation)
│   └── plans/                      # implementation plans (per-phase)
└── README.md
```

- [ ] **Step 13.3: Add a "Deep learning support" section after "How it works"**

Insert this block after the existing "How it works" section, before "## Venv":

```markdown
## Deep learning support (v0.2.0-alpha.1, Phase 1)

Beyond tabular ML, the `ml-engineer` agent now routes deep-learning tasks (CV, NLP, LLM, VLM) to dedicated sub-agents:

- `cv-engineer` — image classification, detection, segmentation
- `nlp-engineer` — encoder fine-tuning (classification, NER, QA, embeddings)
- `llm-engineer` — LLM and VLM finetuning, preference tuning, eval, merge, quantize, serve

Each sub-agent runs the same disciplined loop as the tabular orchestrator (research → decide → plan → write → execute → verify → debug → review) with domain-appropriate skills.

**Phase 1 status (this release):** Six shared infrastructure skills shipped — `dl-detect-env` (probe compute fleet), `dl-remote-execute` (Modal/RunPod/Vast/Lambda/Beam/SSH/Colab dispatcher), `dl-experiment-track` (wandb/mlflow/aim), `dl-checkpoint` (save/resume), `dl-distributed` (single-GPU vs FSDP2 vs DeepSpeed selector), `dl-debug-training` (NaN/OOM/divergence triage).

Domain-specific skills (CV/NLP/LLM/VLM training recipes) ship in Phases 2 and 3. Until then, sub-agents can route a DL task, set up the environment, hand off to a remote provider, and run a generic finetune script — but they cannot yet offer domain-specific recipes.

**Remote execution:** `dl-detect-env` probes for configured remote providers; `dl-remote-execute` shows the user the top-3 candidates with cost + latency tradeoffs and runs the script on the chosen provider, fetching results back to the local workdir. The user picks once at the start of a remote chain; subsequent remote steps continue silently on the same provider until the user switches or the resource requirement changes.

See `docs/superpowers/specs/2026-05-01-dl-skills-design.md` for the full design.
```

- [ ] **Step 13.4: Commit**

```bash
git add README.md
git commit -m "README: add DL support section, update directory tree for Phase 1"
```

---

## Task 14: Cross-validation pass

This task validates that the Phase 1 deliverables hang together as a coherent set.

- [ ] **Step 14.1: Verify all new files are present**

Run:
```bash
ls -la skills/dl-detect-env/SKILL.md \
       skills/dl-remote-execute/SKILL.md \
       skills/dl-experiment-track/SKILL.md \
       skills/dl-checkpoint/SKILL.md \
       skills/dl-distributed/SKILL.md \
       skills/dl-debug-training/SKILL.md \
       agents/cv-engineer.md \
       agents/nlp-engineer.md \
       agents/llm-engineer.md \
       docs/superpowers/specs/dl-env-json-schema.md
```
Expected: all 10 files exist.

- [ ] **Step 14.2: Verify all new SKILL.md frontmatter parses**

Run:
```bash
for f in skills/dl-*/SKILL.md agents/cv-engineer.md agents/nlp-engineer.md agents/llm-engineer.md; do
  echo "Checking $f"
  python3 -c "import yaml,sys; yaml.safe_load(open('$f').read().split('---')[1])" || echo "FAIL: $f"
done
```
Expected: each file prints `Checking ...` with no FAIL line.

- [ ] **Step 14.3: Verify version bump in plugin.json**

Run: `grep -E '"version"|"description"' .claude-plugin/plugin.json`
Expected: version is `"0.2.0-alpha.1"`, description mentions DL coverage.

- [ ] **Step 14.4: Verify router prologue is in place**

Run: `grep -c "Router prologue" agents/ml-engineer.md`
Expected: `1` (one match).

Run: `grep -c "cv-engineer\|nlp-engineer\|llm-engineer" agents/ml-engineer.md`
Expected: `>= 3` (at least one mention of each sub-agent in the routing rules).

- [ ] **Step 14.5: Verify cross-references between skills are consistent**

Skills reference each other by name. Run:
```bash
grep -h "dl-detect-env\|dl-remote-execute\|dl-experiment-track\|dl-checkpoint\|dl-distributed\|dl-debug-training" skills/dl-*/SKILL.md agents/cv-engineer.md agents/nlp-engineer.md agents/llm-engineer.md | sort -u
```
Expected: every cross-reference uses one of the six exact names. No typos like `dl-detect_env` or `dl-detect-environment`.

- [ ] **Step 14.6: Verify env.json schema is referenced consistently**

Run:
```bash
grep -l "env.json" skills/dl-*/SKILL.md agents/*.md
```
Expected: at least `dl-detect-env/SKILL.md`, `dl-remote-execute/SKILL.md`, `dl-distributed/SKILL.md`, `dl-experiment-track/SKILL.md`, `dl-checkpoint/SKILL.md` reference it.

- [ ] **Step 14.7: Verify no skill exceeds reasonable size for its scope**

Run: `wc -l skills/dl-*/SKILL.md`
Expected: `dl-remote-execute` and `dl-detect-env` are the two longest (~250-350 lines each, justified by surface area). The other four are 150-250 lines each. No skill is under 80 lines (too thin) or over 400 lines (too sprawling).

- [ ] **Step 14.8: Verify the README directory tree matches reality**

Run: `tree -L 3 -I '.git|.claude-plugin|*.pyc' . 2>/dev/null || find . -maxdepth 3 -not -path '*/\.*' | sort`
Expected: structure matches the README's "What's in here" section.

- [ ] **Step 14.9: Final commit (if any cleanup happened during validation)**

Run: `git status`
If clean, no commit needed. If there are stray fixes, commit them with:
```bash
git add -A
git commit -m "Phase 1 cross-validation cleanup"
```

- [ ] **Step 14.10: Tag the release**

Run:
```bash
git tag v0.2.0-alpha.1 -m "Phase 1: DL skills foundation (6 infra skills + router + 3 sub-agent shells)"
```

(Do NOT push the tag without user approval.)

---

## Task 15: Phase 1 completion checklist

Walk through this checklist and confirm each item before declaring Phase 1 done.

- [ ] **Skills present (6):** `dl-detect-env`, `dl-remote-execute`, `dl-experiment-track`, `dl-checkpoint`, `dl-distributed`, `dl-debug-training` — each has a `SKILL.md` with valid frontmatter.

- [ ] **Sub-agents present (3):** `cv-engineer.md`, `nlp-engineer.md`, `llm-engineer.md` — each has valid frontmatter, references the correct subset of skills, and clearly states the Phase 1 limitation.

- [ ] **Router prologue installed:** `agents/ml-engineer.md` has the routing rules at the top and dispatches to the three sub-agents on the right signals.

- [ ] **Schema doc present:** `docs/superpowers/specs/dl-env-json-schema.md` documents the `env.json` contract.

- [ ] **Plugin version bumped:** `0.1.0 → 0.2.0-alpha.1`. Description mentions DL coverage.

- [ ] **README updated:** Directory tree reflects new files; "Deep learning support" section added.

- [ ] **No skill over-fires:** Each skill's `description` says "Use when..." AND "Do NOT use when...".

- [ ] **No baked-in version numbers:** Grep skills for `==[0-9]` or `transformers==` etc. Should return nothing in skill bodies.

  Run: `grep -E '==[0-9]|transformers==' skills/dl-*/SKILL.md`
  Expected: no output.

- [ ] **No baked-in benchmark numbers:** Skills do not state "X model fits in Y GB" as facts. Memory estimates are research-hooked.

- [ ] **Hybrid writing style applied:** Decision rules in full prose with capitalized NOT/NEVER/MUST. Bullets for step lists.

- [ ] **All commits made:** `git log --oneline | head -20` shows the Phase 1 commits.

- [ ] **Working tree clean:** `git status` shows nothing pending.

---

## Acceptance criteria for Phase 1 (recap from spec)

After Phase 1, the plugin can:

- [ ] Route a DL task to the right sub-agent based on user input (verified manually by invoking with sample prompts).
- [ ] Detect the local environment AND every configured remote handoff, write `<workdir>/env.json` with the correct schema.
- [ ] Hand off to a remote provider (Modal, RunPod, Vast, Lambda, Beam, SSH, or Colab) when the user picks one.
- [ ] Run a generic finetune script on the chosen environment.

What it CANNOT yet do (deferred to Phases 2 and 3):

- Offer CV-specific recipes (timm, YOLO, SAM) — Phase 2.
- Offer NLP-specific recipes (ModernBERT, DeBERTa, NER decoding) — Phase 2.
- Offer LLM-specific recipes (Unsloth/Axolotl, LoRA decision tree, DPO/GRPO selector, mergekit, AWQ) — Phase 3.
- Offer VLM recipes — Phase 3.

This is the expected Phase 1 surface. Phase 2 plan will be written after Phase 1 lands.
