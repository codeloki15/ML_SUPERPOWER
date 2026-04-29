---
name: ml-engineer-write-code
description: Use when an approved plan exists and the orchestrator is implementing a specific step in Python. Do NOT use without an approved plan, for general code edits, or for writing scripts outside the current session workdir.
license: MIT
metadata:
  source: ml-engineer
  version: 0.1.0
---

# ML Engineer — Write Code

Generate a complete, executable Python script for one plan step. The script will run unattended under the project venv and must exit cleanly.

## Output contract

- Save the script to `<workdir>/step_<N>_<short-name>.py` where `<workdir>` is the current session's `newton_workdir/<timestamp>/`.
- Wrap the code in a fenced ```python block in your reply, with the file path stated on the line above the block.
- The script must be runnable as `python step_N.py` with the venv active. No CLI args expected unless explicitly required.

## Hard requirements

- **Complete and executable** — no placeholders, no `pass`, no `TODO`. If you can't fill in a value, ask the user before writing.
- **No interaction** — never use `input()`, `getpass`, `argparse` prompts, or any function that waits on stdin.
- **No display calls** — never use `plt.show()`, `df.head()` without `print()`, or Jupyter-style implicit display. Always `print(...)` what you want visible.
- **Charts saved, not shown** — `plt.savefig(os.path.join(CHART_DIR, '<name>.png'))` then `plt.close()` then `print(f"Chart saved as <name>.png")`. `CHART_DIR` is `<workdir>/charts/` (create it with `os.makedirs(CHART_DIR, exist_ok=True)`).
- **No web servers, no daemons** — never use Flask, FastAPI, uvicorn, gradio, streamlit, dash, or any code that listens on a port.
- **Bounded execution** — every loop has a clear exit condition. Long-running operations (training, fitting on >100k rows) must print progress at intervals.
- **No file deletion** — never `os.remove`, `shutil.rmtree`, or `rm -rf` anything. The user manages cleanup.
- **No network calls** unless the plan step explicitly requires fetching data from a URL the user named.

## Pandas display defaults

If the script uses pandas, include this near the top:

```python
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 200)
```

## Error handling

- Wrap risky operations (file IO, model fitting, parsing) in `try/except`.
- In each `except`, print the error with context and `raise` — do not silently swallow.
- Validate inputs before processing: file exists, columns present, dtypes as expected.

## Imports

- Standard library first, then third-party, alphabetized within group.
- Only import what you use. No `import *`.

## Style

- PEP 8. Functions for any logic block longer than ~15 lines.
- Docstrings on every function, one line each.
- Inline comments only where the code is non-obvious (a workaround, a numerical-stability trick, etc.).

## Output checklist

Before returning, verify:

- [ ] File path stated above the code block
- [ ] No `plt.show()`, `input()`, `argparse`, web server, or `os.remove` in the code
- [ ] Every chart has `plt.savefig(...)` + `plt.close()` + `print("Chart saved as ...")`
- [ ] All `except` blocks re-raise after logging
- [ ] Pandas display options set if pandas is used
- [ ] Script ends with a clear summary `print(...)` so the executor sees what happened
