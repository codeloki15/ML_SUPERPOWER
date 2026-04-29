---
name: ml-engineer-execute
description: Runs a Python script written by ml-engineer-write-code under an isolated local venv. Captures stdout, stderr, and exit code. Use after a script has been written to newton_workdir/<timestamp>/. On first use this skill creates the venv (one-time, ~30s) after asking the user for approval.
license: MIT
metadata:
  source: ml-engineer
  version: 0.1.0
---

# ML Engineer — Execute

Run a script under the project's local venv. Install packages on demand. Report results.

## Step 1 — Ensure venv exists (ask user once)

The venv lives at `${CLAUDE_PLUGIN_DATA}/venv` (typically `~/.claude/ml-engineer/venv`). It is shared across all sessions of this plugin.

Check whether it already exists:

```bash
ls "${CLAUDE_PLUGIN_DATA:-$HOME/.claude/ml-engineer}/venv/bin/python" 2>/dev/null
```

- **Exists:** skip to Step 2.
- **Does not exist:** ask the user exactly this:

  > Newton needs a Python venv at `<path>` to run code locally. This is a one-time setup (~30 seconds, ~50MB). Approve?

  On `yes` → run `bash ${CLAUDE_PLUGIN_ROOT}/skills/ml-engineer-execute/scripts/setup_venv.sh`. On `no` → stop, tell the user code execution is unavailable until they approve.

Do not ask again in subsequent runs of the same session.

## Step 2 — Run the script

From the workdir:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/skills/ml-engineer-execute/scripts/run.sh <workdir>/step_<N>_<name>.py
```

Capture stdout, stderr, and exit code.

## Step 3 — Handle missing packages

If stderr contains `ModuleNotFoundError: No module named '<pkg>'`:

1. Tell the user: `Script needs <pkg>. Install into venv? (y/n)`
2. On `y`, run:
   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/skills/ml-engineer-execute/scripts/pip_install.sh <pkg>
   ```
3. Re-run the script. Loop up to 3 times for cascading missing imports. After 3 install attempts in a row, stop and ask the user how to proceed.

Map common import names to PyPI names when they differ:

| import name | pip name |
|---|---|
| `cv2` | `opencv-python` |
| `sklearn` | `scikit-learn` |
| `PIL` | `Pillow` |
| `yaml` | `PyYAML` |
| `bs4` | `beautifulsoup4` |

## Step 4 — Report

Return to the orchestrator with:

- `exit_code`: integer
- `stdout`: full stdout (truncate to last 200 lines if huge, note truncation)
- `stderr`: full stderr
- `chart_files`: list of files written to `<workdir>/charts/` since the run started

## Rules

- Never run `python` or `python3` directly. Always `run.sh`.
- Never `pip install` outside `pip_install.sh`. No `--user`, no system pip, no `sudo`.
- Never run a script outside `<workdir>/`. If the orchestrator points to a path outside the workdir, refuse and surface the error.
- If exit code is non-zero, do not retry blindly — hand off to `ml-engineer-debug`.

## Output checklist

- [ ] Confirmed venv exists (or got user approval and created it)
- [ ] Ran the script via `run.sh`
- [ ] Captured exit code, stdout, stderr separately
- [ ] Listed any new chart files
- [ ] On `ModuleNotFoundError`, asked before installing
