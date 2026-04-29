---
name: ml-engineer-debug
description: Diagnoses a failed Python script run (non-zero exit code, traceback, or verification failure) and produces a corrected version. Use when ml-engineer-execute or ml-engineer-verify reports failure, regardless of domain. Do NOT use for general code review or refactoring.
license: MIT
metadata:
  source: ml-engineer
  version: 0.1.0
---

# ML Engineer — Debug

Diagnose one failed run and produce a fixed script. One pass per invocation.

## Inputs you need

The orchestrator must provide:

1. **Failed script path** — `<workdir>/step_<N>_<name>.py`
2. **stderr** — full, untruncated
3. **stdout** — full
4. **exit code**

If any of these is missing, ask for it before proceeding.

## Diagnosis structure

Reply in exactly this format:

```
### Error analysis
<one line: what kind of error — ImportError / KeyError / shape mismatch / value out of range / logic bug>

### Root cause
<2-3 sentences: why it happened, citing the line number from the traceback>

### Fix
<one sentence: what change resolves it>

### Patched script
<file path>
```python
<full corrected script>
```
```

## Rules

- **Patch the whole file** — return the complete corrected script, not a diff. The execute skill replaces the file wholesale.
- **One root cause per pass** — if there are multiple unrelated bugs, fix the one that's blocking execution and note the others under `### Follow-ups`. Do not bundle unrelated changes.
- **Preserve the workdir convention** — chart paths, workdir-relative file IO, all rules from `ml-engineer-write-code` still apply.
- **No new dependencies** unless the original error was `ModuleNotFoundError`. If you need a new library to fix the bug, justify it in `### Root cause`.
- **No `try/except` to mask the error** — fix the underlying cause. Catching and ignoring is not a fix.

## Common cases and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `KeyError: 'col_name'` | Column missing or renamed | Inspect actual columns first; use `df.columns.tolist()` to confirm |
| `ValueError: could not convert string to float` | Mixed dtypes in a numeric column | Add `pd.to_numeric(..., errors='coerce')` and handle NaN |
| `ValueError: shapes (X,) (Y,) not aligned` | Train/test split or feature mismatch | Print shapes before the failing op |
| `FileNotFoundError` | Wrong path or missing dir | `os.makedirs(..., exist_ok=True)` for output dirs; absolute paths for inputs |
| `MemoryError` | Loaded full dataset eagerly | Add `nrows=`, `chunksize=`, or `dtype=` to `read_csv` |
| Plot is empty | Wrote after `plt.close()` or before any `plot()` call | Reorder: plot → savefig → close |

## What to do if you can't diagnose

If stderr is unhelpful (e.g. `Killed`, OOM, or no traceback):

1. State that the cause is unclear from the available output.
2. Suggest the smallest possible diagnostic script (one that prints shapes/dtypes/memory) before retrying the original.
3. Do not fabricate a root cause.

## Output checklist

- [ ] One-line `### Error analysis`
- [ ] `### Root cause` cites a line number from the traceback
- [ ] Full corrected script in `### Patched script`
- [ ] Script still respects all `ml-engineer-write-code` rules
- [ ] No diff format — the whole file
