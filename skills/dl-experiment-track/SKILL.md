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

## Recipe template

The following template skeletons are adapted by the orchestrator for the chosen tracker.

### `<workdir>/src/_init_tracker.py` (wandb)

```python
"""Tracker init — insert this at the TOP of the training script, before model construction."""
import os
import time
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()
TRACKER = "wandb"  # set by dl-experiment-track decision rules


def init_tracker(config: dict, model_family: str, task_type: str):
    project = WORKDIR.name
    run_name = f"{task_type}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    if TRACKER == "wandb":
        import wandb
        wandb.init(
            project=project,
            name=run_name,
            config=config,
            tags=[model_family, task_type],
        )
        return wandb
    if TRACKER == "mlflow":
        import mlflow
        mlflow.set_experiment(project)
        run = mlflow.start_run(run_name=run_name, tags={"model_family": model_family, "task_type": task_type})
        mlflow.log_params(config)
        return mlflow
    if TRACKER == "aim":
        import aim
        run = aim.Run(experiment=project, log_system_params=True)
        run.name = run_name
        run["hparams"] = config
        run.add_tag(model_family); run.add_tag(task_type)
        return run
    raise RuntimeError(f"Unknown tracker {TRACKER!r}")


def finish_tracker(handle):
    if TRACKER == "wandb":
        handle.finish()
    elif TRACKER == "mlflow":
        handle.end_run()
    elif TRACKER == "aim":
        handle.close()
```

### Hugging Face Trainer integration

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir=str(WORKDIR / "checkpoints"),
    report_to=["wandb"],   # or ["mlflow"] / ["aim"] / [] to disable
    run_name=run_name,
    # ... other training args ...
)
```

When using Trainer + `report_to`, you do NOT need to call `wandb.log()` manually for loss/lr/eval metrics — Trainer handles them. Custom metrics (e.g., per-class F1, per-fold OOF) still need explicit `wandb.log({...})` calls inside training callbacks or after `trainer.evaluate()`.

## Hard constraints

- NEVER hard-code an API key or token in the training script. Wandb reads from `~/.netrc` or `WANDB_API_KEY`. Mlflow reads from env or config file. Aim is local.
- NEVER set the project name to something generic like `"test"`. Use the workdir basename or the user's stated project name. Generic project names create unfindable runs.
- NEVER skip `wandb.finish()` or equivalent at the end of the script. Without it, runs show as "running" forever in the UI and confuse later comparison.
- IF the run is on a remote provider (`env.json.active != "local"`), make sure the tracker auth is available on the remote too. For wandb on Modal: pass `WANDB_API_KEY` via `modal.Secret`. For SSH boxes: ensure `~/.netrc` exists on the remote.
- NEVER use the same `run_name` across two distinct configs. Use a UTC timestamp suffix to disambiguate; otherwise old runs get visually merged with new ones in the UI.

## Research hooks

Tracker libraries evolve quickly. Before generating the init snippet for an unfamiliar tracker version OR when the user asks "what's the current best practice", invoke `ml-engineer-research`:

- **Tracker version compatibility.** Query: *"Current minimum compatible version of `{tracker}` for HF Trainer `report_to=` integration as of {today}."*
- **Sweep / hyperparameter search setup.** Query: *"Recommended config-as-yaml format for `{tracker}` sweep on HF Trainer as of {today}."*
- **Remote-provider secret patterns.** Query: *"How to securely pass `WANDB_API_KEY` to `{provider}` remote runs as of {today}."* Apply when the active env is remote.

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- The training script contains a tracking-init call near the top (within the first ~30 lines, BEFORE any model or trainer construction).
- The script contains a finish/cleanup call at the end (`wandb.finish()`, `mlflow.end_run()`, `aim_run.close()`).
- A dry-run (1 step, or just the init call) returned exit 0.
- Run name and project name are non-generic (NOT `"test"`, NOT empty string, NOT default placeholders).

## Output checklist

- [ ] Tracker chosen based on available_libs + user input
- [ ] Auth verified (wandb logged in; mlflow URI set if remote; aim local OK)
- [ ] Init snippet inserted at the top of the training script
- [ ] Finish snippet at the end
- [ ] HF Trainer `report_to` set if Trainer is being used
- [ ] Dry-run succeeded
- [ ] On remote runs: tracker auth available on the remote
- [ ] Run name is unique (UTC timestamp suffix); project name is non-generic
