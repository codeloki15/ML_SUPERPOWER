---
name: ml-engineer-verify
description: Verifies that a completed step actually did what was claimed before reporting completion. Use after every executed step and especially before declaring a task complete. Required by the orchestrator — never skip. Works across any quantitative domain — catches silent failures, wrong-shape outputs, leaked data, look-ahead, off-by-one joins, unit mismatches, and inverted metrics.
license: MIT
metadata:
  source: ml-engineer
  version: 0.1.0
---

# Verify Before Completion

Trust nothing. Re-check that the step did what was claimed. This skill is invoked after every executed step and is mandatory before the orchestrator says "done".

Domain-agnostic. The same verification discipline catches a leaked target in an ML model, a look-ahead feature in a backtest, a forgotten timezone in a logistics forecast, and a unit mismatch in a clinical analysis.

## When to invoke

- After `ml-engineer-execute` returns `exit 0` and the orchestrator is about to move to the next step.
- Before reporting overall task completion to the user.
- When a result looks "too good" or "too bad" to be true.

## Universal failure patterns to check

These bugs appear in every quantitative discipline and are the most common silent failures. Always consider them, regardless of domain:

| Pattern | What it looks like | Cheap check |
|---|---|---|
| **Silent NaN propagation** | Aggregate has fewer rows than expected, or means look "round" | `df.isna().sum()`, row count before/after |
| **Look-ahead / leakage** | Metric is implausibly good | Inspect features for any column derived from the outcome or future timestamps |
| **Wrong join cardinality** | Row count multiplied or shrunk after a merge | Compare `len()` before and after, check uniqueness of join keys |
| **Unit / scale mismatch** | Numbers are ~1000x off | Print min/max of key columns, sanity-check units |
| **Timezone or off-by-one in time** | First/last day looks weird, or weekday distributions are skewed | Print min/max timestamp, weekday histogram |
| **Inverted sign / metric** | Metric is the right magnitude but wrong direction | Check 2-3 representative rows by hand |
| **Sampling bias** | Validation/holdout overlaps with training | Print intersection of IDs / timestamps between splits |
| **Distribution shift between splits** | Train and test have visibly different distributions | Print mean/std/quantiles per split for key columns |
| **Aggregation double-counting** | Totals don't match raw sum | Compare summed total to a known total |
| **Encoding leakage** | Group-level statistics computed on full data, then used per-row | Confirm any group-encoded feature was fit only on training data |

The orchestrator should pick checks that match the step's claim, not run all of them.

## Claim-type → checks

### "Loaded N rows / M columns"
- Re-read the first 5 rows; confirm shape matches the claim.
- Spot-check: does column 1 actually contain what the user said it contains?
- Are there NaN/inf values that the prior step silently coerced?

### "Cleaned / transformed / normalized"
- Run the inverse check: `df.duplicated().sum() == 0`, `df.isna().sum() == 0`, normalized columns have mean ≈ 0 and std ≈ 1, etc.
- Compare row count before and after — did we lose more data than expected?

### "Joined / merged / aggregated"
- Row count before and after, with expected ratio.
- Uniqueness of join keys.
- Spot-check 3 rows of the joined result against the source tables.

### "Computed feature / metric / score"
- Print the new column alongside its source columns for 5 rows.
- Sanity-check value range, dtype, and whether any value lies outside a plausible domain (negative ages, returns > 100%, probabilities > 1).

### "Trained / fit a model / strategy"
- Re-predict on the held-out set, recompute the metric, confirm it matches the claim.
- **Leakage check:** any feature with implausibly high importance? Any feature derived from the outcome or from future data?
- **Split integrity:** confirm train/val/test were separated *before* any preprocessing that uses outcome statistics (target encoding, scaling fit on full data, group statistics).
- **Walk-forward / temporal split** for time-indexed data: confirm no future leakage into the training window.

### "Evaluated / scored / benchmarked"
- For classification-like outcomes: confusion matrix, not just headline metric. Per-class precision/recall when classes are imbalanced.
- For regression-like outcomes: residual plot or distribution; check for systematic bias across subgroups.
- For ranking outcomes: top-K precision, not just AUC.
- For time-series forecasting: error by horizon, not just average; check if errors are autocorrelated (a sign of model misspec).
- For backtests: confirm transaction costs included, no survivorship bias, walk-forward (not k-fold) CV.
- Confirm the metric was computed on a held-out / out-of-sample set, not the training set.

### "Generated chart / figure"
- Confirm the file exists and is non-empty (`os.path.getsize > 1024`).
- The chart should reflect the actual data — print 2-3 cells / values from the underlying data and confirm they're consistent with what the figure shows.

### "Forecasted / projected / simulated"
- Confirm forecast horizon is correct.
- Check that the forecast doesn't extrapolate beyond plausible bounds.
- Confirm the model wasn't fit on the period it's now forecasting.

## Process

### Step 1 — Read the step's output

What was the executed script's stdout? What did it claim?

### Step 2 — Pick the verification(s)

Based on the claim type(s), pick 1-2 cheap checks from the tables above. Always also consider the universal failure patterns — at least one should be checked when the step transforms or merges data.

### Step 3 — Run a verification script

Write a small Python script (`<workdir>/verify_step_<N>.py`) that runs the chosen checks. Use the existing `ml-engineer-write-code` skill's output rules.

Run it via `ml-engineer-execute`.

### Step 4 — Report

Output exactly this format:

```
## Verification: step <N>

### Claim
<what the step claimed>

### Checks run
- <check 1>: <pass | fail | mismatch>
- <check 2>: <pass | fail | mismatch>

### Verdict
<verified | suspect | failed>

### Notes
<one sentence — anything the user should know before we proceed>
```

### Step 5 — Branch

- **verified** → orchestrator continues to next step.
- **suspect** → tell the user what's odd, ask whether to proceed or investigate.
- **failed** → hand back to `ml-engineer-debug` with the verification output as evidence.

## Hard rules

- **Never trust headline metrics.** A 99% accuracy / 3.0 Sharpe / 0.99 R² claim without supporting checks is `suspect` by default.
- **Never accept "no errors" as verification.** A script can exit 0 and still produce wrong output.
- **Always check for leakage / look-ahead** when verifying any predictive or evaluative step. This is the #1 silent failure across every quantitative domain.
- **Always re-read the file** when verifying loading. A script that prints `(1000, 50)` may have actually loaded `(10, 5)` due to a `nrows=` left in from debugging.
- Verification scripts must be cheap (≤ 30 seconds). If the check would take longer than the original step, skip it and report `Verdict: unverified — too expensive to check`.

## Anti-patterns to avoid

- **Self-confirming verification:** running the exact same script that produced the original claim and getting the same number. That's not verification, that's repetition. Use a different code path.
- **Vague verification:** "Looks reasonable." Be specific or don't bother.
- **Skipping on "obvious" steps:** "It just loaded a CSV, no need to verify." Especially verify the boring steps — that's where silent failures hide.

## Output checklist

- [ ] Claim restated in one line
- [ ] At least one check from the appropriate category
- [ ] At least one universal-failure-pattern check considered (and run if applicable)
- [ ] Verification ran via execute skill, with its own script (not the original)
- [ ] Verdict is one of: verified | suspect | failed
- [ ] On `failed`, handed off to debug skill
- [ ] On `suspect`, surfaced to user before proceeding
