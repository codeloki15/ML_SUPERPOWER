---
name: ml-engineer-debug
description: Use when ml-engineer-execute returns a non-zero exit code, ml-engineer-verify returns a failed verdict, or a script produced an exception, NaN/inf, or implausible metric. Do NOT use for general code review, refactoring, or proactive cleanups.
license: MIT
metadata:
  source: ml-engineer
  version: 0.1.0
---

# Debug

## Iron Law

> **No fix without root-cause investigation first.**

Patching the symptom and re-running is not debugging. It is gambling. If you can't explain *why* the bug happened, you cannot tell whether your fix actually fixed it — and ML pipelines are full of bugs that produce plausible numbers anyway.

## The four phases

Every debug pass goes through these phases, in order. Skipping ahead is a violation of the Iron Law.

### Phase 1 — Root cause

What is the **smallest** change in input or code that would have prevented this error? Be specific to a line, a column, a value, a shape — not "something with the data."

The output of this phase is **one sentence** that names the actual cause. Not the symptom.

- ❌ Symptom: "KeyError: 'price'"
- ✅ Root cause: "The cleaning step renamed 'price' to 'price_usd' on line 42, but the modeling step still looks up 'price'."

If you can't write that sentence, you don't have a root cause yet. Run a small probe script that prints shapes, dtypes, columns, NaN counts, sample values — whatever narrows it down. Don't guess.

### Phase 2 — Pattern

What category of bug is this? Pattern recognition lets you anticipate related bugs and write a fix that doesn't just paper over one instance.

Common ML / data-science patterns:

| Pattern | Tells |
|---|---|
| Schema drift | Column missing, renamed, or dtype changed between two steps |
| Silent NaN | Aggregate row count dropped; means look round; metric is suspiciously high or zero |
| Leakage / look-ahead | Metric is implausibly good; feature has near-perfect target correlation |
| Split contamination | Validation IDs/timestamps overlap with training |
| Shape mismatch | `(N,)` vs `(N, 1)`, train vs predict feature count differs |
| Encoding leakage | Group statistics, target encoding, or scaling fit on full data instead of train-only |
| Off-by-one in time | First/last period weird; weekday histogram skewed; forecast horizon shifted |
| Unit / scale | Numbers ~1000× off; mixed currencies, timezones, or units |
| Memory exhaustion | `Killed`, OOM, or process truncated mid-run |
| Numerical instability | NaN/inf appearing in loss, gradient, or matrix inverse |
| Indexing bug | Wrong values in features because indices got reordered after a sort or shuffle |

Name the pattern explicitly. If multiple apply, name the primary one.

### Phase 3 — Hypothesis

State the fix as a **falsifiable** prediction:

> "If I change `<specific change>` then `<specific observation>` will change from `<X>` to `<Y>`."

Examples:

- "If I rename the column lookup on line 87 from 'price' to 'price_usd', the script will run past line 87 and print the next stage's output."
- "If I split before scaling instead of after, validation R² will drop from 0.99 to a plausible 0.6-0.8."
- "If I exclude the `target_lag_1` feature, train/val gap will narrow from 0.35 to <0.1."

A non-falsifiable fix is not a fix. "Add error handling" predicts nothing observable.

### Phase 4 — Implementation

Apply the fix. Return the **full** corrected script. Note the predicted observation alongside it so the orchestrator and `ml-engineer-verify` can confirm.

## Output format

Reply in exactly this structure:

```
### Phase 1 — Root cause
<one sentence naming the actual cause, with a line reference>

### Phase 2 — Pattern
<one of the patterns from the table, or "other: <description>">

### Phase 3 — Hypothesis
If I <change>, then <observable> will change from <X> to <Y>.

### Phase 4 — Patched script

<workdir>/step_<N>_<name>.py

```python
<full corrected script>
```

### Predicted next observation
<what the verifier should see if the fix worked>
```

## The 3-failures escape hatch

After **3 fix attempts on the same step have failed**, stop fixing. The bug is probably not where you've been looking.

Step out and audit:

1. **Data pipeline.** Print the actual data going into the failing step at every prior stage. The bug is often upstream of where it manifests.
2. **Assumptions.** Re-read the plan step. Are the inputs really shaped like you assumed?
3. **Architecture.** Is the approach itself wrong for this data? An imbalanced binary task with a regression model fails in many creative ways before the architecture is admitted as the issue.
4. **Domain conventions.** Did you violate a domain-specific rule (walk-forward CV in time series, scaffold splits in molecular ML, calibration in healthcare)?

Output:

```
### Escape hatch triggered
Three fix attempts on step <N> have failed. Stopping symptom patches.

### Audit
- Data pipeline: <what you found upstream>
- Assumptions: <which one was wrong>
- Architecture: <is the method right for this data?>
- Domain conventions: <any rules violated?>

### Recommended action
<one of: re-plan this step / abandon this approach / hand to user>
```

Tell the user; do not silently keep retrying.

## Hard rules

- **Patch the whole file** — return the complete corrected script, not a diff. Execute replaces wholesale.
- **One root cause per pass.** If multiple bugs exist, fix the one blocking execution and note the others under `### Follow-ups`.
- **No try/except to mask the error.** Catching and ignoring is not a fix. Fix the underlying cause.
- **No new dependencies** unless the original error was `ModuleNotFoundError`. If a new library is needed for a fix, justify it in `### Phase 1 — Root cause`.
- **No "improvements" beyond the fix.** Reformatting, renaming, factoring out helpers — all out of scope. Fix the bug, return the script.

## Common cases — root causes, not symptoms

| Symptom | Likely root cause | Test for it |
|---|---|---|
| `KeyError: 'X'` | Column renamed upstream, or never existed | Print `df.columns.tolist()` before the failing line |
| `ValueError: could not convert string to float` | Mixed dtype in a "numeric" column | Print value_counts of unique non-numeric entries |
| `ValueError: shapes not aligned` | Train vs predict feature count or order differs | Print shapes of both at the failing op |
| `FileNotFoundError` | Wrong path, missing dir, or workdir not created | Print `os.getcwd()` and `os.listdir(workdir)` |
| `MemoryError` / `Killed` | Loaded full dataset eagerly | Switch to `nrows=`, `chunksize=`, narrow `dtype=` |
| Plot is empty / blank | `plt.close()` called before `savefig`, or no `plot()` called | Print `len(ax.lines)` before saving |
| Metric implausibly high | Leakage or eval-on-train | Audit feature/target correlations; confirm split |
| Metric implausibly low | Wrong target column, inverted labels, or unit mismatch | Hand-check 5 rows of (input, target, prediction) |

## Output checklist

- [ ] Phase 1 names a root cause with a line reference
- [ ] Phase 2 names one of the patterns
- [ ] Phase 3 is falsifiable (predicts an observable change with magnitude)
- [ ] Phase 4 returns the full corrected script
- [ ] Predicted next observation stated
- [ ] If 3rd consecutive failure on same step: escape hatch invoked instead
- [ ] No improvements beyond the fix
