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
2. Filter by VRAM requirement.
   - For **shell-style** environments (have `vram_gb` set: `local`, `ssh` hosts, `runpod` pods, `lambda` long-running): keep when `vram_gb >= requirement.vram_gb`.
   - For **menu-style** environments (have `available_gpus` set, omit `vram_gb`: `modal`, `beam`, `vastai`, `runpod` serverless, `lambda` API-reserved): include the provider whenever any GPU in `available_gpus` satisfies the requirement. Look up the per-class VRAM via the research hook in Research hooks below.
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
| `ssh` | (generic SSH host) `rsync -av <workdir>/ {ssh_target}:~/dl_workdir_<timestamp>/`. `ssh {ssh_target} 'cd dl_workdir_<timestamp> && python <script_relative_path>'` with line-buffered output. `rsync -av {ssh_target}:~/dl_workdir_<timestamp>/results/ <workdir>/results/`. Optionally clean up remote dir on success. |
| `colab` | (async handoff) Generate a `.ipynb` from the script (one cell per logical block, plus a final cell that writes `results.json` to a Drive-synced path). Print the URL pattern: `https://colab.research.google.com/notebook#fileId={uploaded_id}` along with explicit instructions: "Open this URL, click Runtime → Run all, then come back and tell me when it finishes." Park the step. When the user confirms completion, fetch `results.json` from the user-provided Drive sync path. |

### Step 7 — Always-tear-down hard rule

For ephemeral providers (`vastai`, `runpod` when not user-pinned, `lambda` when reserved-for-this-task), tear down the resource on completion regardless of success or failure. Surface the incurred cost in the result. NEVER leave a pod or instance running unless the user explicitly asked to keep it.

### Step 8 — Return result to orchestrator

Return:

```
Exit code:    {0 | non-zero}
Logs:         <workdir>/remote_logs/<provider>_<step>.log
Artifacts:    <list of paths fetched back into workdir>
Cost:         ${incurred} on {provider}
Resume info:  {modal app url | runpod pod id | vastai instance id | lambda reservation id | beam app url | ssh session marker | colab notebook url}
```

Pass exit code through unchanged so the orchestrator can branch the same way it does for local execution.

## Recipe template

This skill is primarily a dispatcher; the actual remote-run logic lives in provider-specific sub-modules generated by `ml-engineer-write-code` per provider. Below is the dispatcher skeleton plus one representative sub-mode (Modal) as a template the orchestrator adapts.

### `<workdir>/_remote_dispatch.py` (skeleton)

```python
import json
import os
import sys
import time
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def load_env():
    return json.loads((WORKDIR / "env.json").read_text())


def save_env(env):
    (WORKDIR / "env.json").write_text(json.dumps(env, indent=2))


def filter_candidates(env, requirement):
    candidates = []
    for name, e in env["environments"].items():
        if e.get("kind") != "remote" or e.get("auth") != "ok":
            continue
        # menu-style: check available_gpus has any GPU large enough.
        # The per-class VRAM map is provider-specific and may drift; the orchestrator
        # populates it at runtime via the research hook in the SKILL body. The skeleton
        # below assumes a precomputed `gpu_vram_gb` dict keyed by GPU class name.
        gpus = e.get("available_gpus") or []
        if gpus:
            gpu_vram_gb = e.get("gpu_vram_gb_lookup", {})  # populated by research hook
            sufficient = [g for g in gpus if gpu_vram_gb.get(g, 0) >= requirement["vram_gb"]]
            if sufficient:
                candidates.append((name, e))
            continue
        elif e.get("vram_gb", 0) >= requirement["vram_gb"]:
            candidates.append((name, e))
    if env["environments"].get("local", {}).get("vram_gb", 0) >= requirement["vram_gb"]:
        candidates.append(("local", env["environments"]["local"]))
    return candidates


def run_remote(active_name, active_env, script_path, requirement):
    provider = active_env["provider"]
    if provider == "modal":
        return run_modal(active_env, script_path, requirement)
    if provider == "runpod":
        return run_runpod(active_env, script_path, requirement)
    if provider == "vastai":
        return run_vastai(active_env, script_path, requirement)
    if provider == "lambda":
        return run_lambda(active_env, script_path, requirement)
    if provider == "beam":
        return run_beam(active_env, script_path, requirement)
    if provider == "ssh":
        return run_ssh(active_env, script_path, requirement)
    if provider == "colab":
        return run_colab_handoff(active_env, script_path, requirement)
    raise RuntimeError(f"Unknown provider {provider!r}")
```

### Modal sub-mode (representative)

```python
import subprocess
from pathlib import Path


def run_modal(env, script_path, requirement):
    """
    Wrap the user's script in a Modal @app.function decorator and run it.
    Workaround for the late-2025 GPU-sandbox-with-Volume crash: write artifacts
    to /tmp inside Modal, then sync to local after the function returns.
    """
    log_dir = WORKDIR / "remote_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"modal_{int(time.time())}.log"

    wrapper = WORKDIR / "_modal_wrapper.py"
    wrapper.write_text(_render_modal_wrapper(script_path, env, requirement))

    cmd = ["modal", "run", str(wrapper)]
    with log_path.open("w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)

    artifacts = _fetch_modal_artifacts(env, WORKDIR / "results")
    cost = _read_modal_cost(env)
    return {
        "exit_code": proc.returncode,
        "log": str(log_path),
        "artifacts": artifacts,
        "cost": cost,
        "resume": _modal_app_url(env),
    }


def _render_modal_wrapper(script_path, env, requirement):
    # Returns Python source that imports the user's script, decorates it as
    # @app.function(gpu=..., secrets=[modal.Secret.from_name(...)]) and calls
    # the entrypoint inside Modal. NEVER inline secret values into the body —
    # use Secret.from_name. Returns string source.
    raise NotImplementedError("orchestrator fills in")
```

The remaining sub-modes (`run_runpod`, `run_vastai`, `run_lambda`, `run_beam`, `run_ssh`, `run_colab_handoff`) follow the same shape: take `(env, script_path, requirement)`, return a dict with `exit_code / log / artifacts / cost / resume`. Each is generated by `ml-engineer-write-code` on first use and cached in the workdir.

## Hard constraints

- NEVER auto-launch paid compute without user confirmation on the first remote step of a session. Even if `auth: "ok"`, the first time a remote runs in a task, ask.
- NEVER leave secrets in the script body that gets uploaded. Secrets MUST be set as remote env vars (Modal: `Secret.from_name(...)`, SSH: `ssh ENV=value cmd`, etc.) or read from the remote's local secret store, never embedded in the script.
- NEVER skip tear-down. If the script fails partway, fetch what's salvageable (last checkpoint, last log lines), THEN tear down.
- NEVER use `vastai` for sustained multi-step work. Vast instances are volatile and can disappear. Single-shot only unless the user pins.
- NEVER assume Colab will run synchronously. The Colab handoff is async — park the step and wait for explicit user confirmation that the notebook finished.
- NEVER stream raw container logs that may contain secrets (e.g., `env` dumps). Filter the log stream for lines matching `*_TOKEN`, `*_KEY`, `password=`, `Bearer `, and replace with `[REDACTED]`.
- IF a provider CLI returns an unexpected response (changed format, new error), do NOT silently guess. Surface the raw error to the user and let them decide.

## Research hooks

The following facts go stale within 6 months. Before quoting any of them to the user OR before generating provider-specific code, invoke `ml-engineer-research` with the listed query. Use the research result verbatim; do NOT cache values across sessions.

- **Per-provider hourly GPU rates.** Query: *"Current published hourly rate for `{gpu_class}` on `{provider}` as of {today} (community vs reserved tier if applicable)."* Apply the result when rendering the candidate menu in Step 3.
- **Cold-start latency.** Query: *"Typical cold-start time for `{provider}` `{gpu_class}` in {month, year} (warm vs cold pod, container pull time included)."* Apply when annotating menu candidates.
- **Modal Volume bug status.** Query: *"Has Modal resolved the GPU-sandbox-with-Volume silent crash reported in late 2025?"* If the issue is fixed, drop the `/tmp` workaround in the Modal sub-mode and mount Volumes normally.
- **Provider CLI command shapes.** Query: *"Current CLI command and flags for `{provider}` to (a) check auth, (b) launch a GPU job, (c) tear down resources, as of {today}."* Apply when generating the provider sub-module.
- **Available GPU classes per provider.** Query: *"Currently available GPU classes on `{provider}` and their VRAM, as of {today}."* Apply when filtering menu-style providers in Step 2.

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- The remote step's exit code matches what the script would produce locally (no false-success because the upload succeeded but execution failed).
- All declared output artifacts (model files, log files, metric JSON) were fetched back into the workdir.
- For ephemeral providers: the resource was actually torn down. Provider-specific checks (research the current CLI command via Research hooks):
  - RunPod: `runpodctl get pods` shows no orphan pod from this task.
  - Vast.ai: `vastai show instances` is empty for our user.
  - Lambda Labs (when reserved-for-this-task): the relevant API listing shows no active reservation. Use the current Lambda CLI command (research-hooked).
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
