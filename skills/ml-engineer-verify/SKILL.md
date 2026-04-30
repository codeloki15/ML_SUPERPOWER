---
name: ml-engineer-verify
description: Use after every successful script execution, before declaring any step or task complete, and whenever a result looks "too good" or "too bad" to be true. Do NOT skip — completion claims without fresh verification evidence are forbidden.
license: MIT
metadata:
  source: ml-engineer
  version: 0.1.0
---

# Verify Before Completion

## Iron Law

> **No completion claim without fresh verification evidence. Ever.**

This is non-negotiable. The orchestrator does not say "done", "verified", "trained", "loaded", "computed", or any synonym until this skill has run a separate check and returned `verified`.

A script exiting with code 0 is **not** verification. A printed metric is **not** verification. A saved chart file is **not** verification. The only thing that counts is fresh output from a check the orchestrator did **not** run as part of the original step.

## Claim / Requires / Not Sufficient

For each common claim type, the table fixes what counts as evidence and what doesn't:

| Claim | Requires | Not Sufficient |
|---|---|---|
| "Loaded N rows / M cols" | Re-read first 5 rows + print shape from a separate read | The original load script's stdout |
| "Cleaned / dropped duplicates" | `df.duplicated().sum() == 0` printed by a fresh script | "duplicates removed" message from the cleaning step |
| "Filled missing values" | `df.isna().sum()` per column == 0, fresh | The filler step said it filled them |
| "Joined / merged" | Row count before & after, uniqueness of join key, 3 spot-checked rows | "merge complete" with a row count |
| "Trained model" | Re-predict on held-out set, recompute the headline metric, confirm match. **If metric is implausibly perfect (AUC≥0.999, R²>0.99, F1=1.0): verdict is `failed`, audit for leakage immediately.** | Loss curve / training log |
| "Model achieves <metric=X>" | Recomputed metric on the **held-out** set + a sanity check (confusion matrix, residual plot, or per-class breakdown). **For OOF / k-fold metrics: confirm reported score is mean across folds, not single-fold.** | The original eval call's printout |
| "No leakage" | Feature-target correlation scan + feature importance audit + confirmation that splits preceded preprocessing | "I split before fitting" in the code comments |
| "Backtest Sharpe = X" | Re-run with the **same** seed and confirm; verify transaction costs included; confirm walk-forward not k-fold | The backtest function's return value |
| "Forecast generated" | Confirm horizon, plausible bounds, model wasn't fit on the forecast period | The forecast file exists |
| "Chart saved" | File exists AND non-empty AND 2-3 underlying values cross-checked against the visual | `plt.savefig()` was called |
| "Statistical test significant" | Re-run on a permuted/shuffled label and confirm it's non-significant; check sample size and assumptions | A small p-value |

If the current step's claim isn't in the table, infer the analogous row. The pattern is always: **a separate check via a different code path that could falsify the claim**.

## Universal failure patterns to scan for

Always consider these when deciding what to verify, regardless of the specific claim:

| Pattern | Symptom | Cheap check |
|---|---|---|
| Silent NaN propagation | Aggregates have fewer rows than expected | `df.isna().sum()`, row count before/after |
| Look-ahead / leakage | Metric is implausibly good | Inspect features for any column derived from the target or future timestamps |
| Wrong join cardinality | Row count multiplied or shrunk after merge | Compare `len()` before/after, check uniqueness of join keys |
| Unit / scale mismatch | Numbers are ~1000× off | Print min/max of key columns, sanity-check units |
| Timezone / off-by-one | First/last day looks weird, weekday distribution skewed | Print min/max timestamp, weekday histogram |
| Inverted sign / metric | Right magnitude, wrong direction | Hand-check 2-3 representative rows |
| Sampling bias | Validation overlaps with training | Print intersection of IDs / timestamps between splits |
| Distribution shift between splits | Train / test visibly different | Print mean/std/quantiles per split for key columns |
| Aggregation double-counting | Totals don't match raw sum | Compare summed total to a known total |
| Encoding leakage | Group-level stats fit on full data, used per-row | Confirm any group-encoded feature was fit only on training split |

## ML-specific leakage patterns (auto-Critical)

These are the silent killers Thakur (AAAMLP) calls out repeatedly. If any of them is detected, the verdict is **failed** (not "suspect"), and the orchestrator must fix before continuing — no exceptions.

| Pattern | Symptom | How to detect |
|---|---|---|
| **AUC = 1.0 (or > 0.999)** | "Perfect" classifier | Almost always a validation bug. Audit features for target derivatives; confirm splits respect group/time |
| **AUC < 0.5** | Worse than random | Predictions inverted, or labels flipped during preprocessing. Audit the prediction pipeline |
| **R² > 0.99** | "Perfect" regressor | Same as AUC=1 — almost always leakage |
| **F1 = 1.0 / accuracy = 1.0** | Perfect classification | Same — almost always leakage |
| **p-value < 1e-10 with N < 10000** | "Highly significant" | Almost always assumption violation, multiple-testing failure, or leakage |
| **Target encoding fit on full data** | Non-fold-aware encoding | Grep the script for `groupby(col)[target].mean()` outside a per-fold loop |
| **Scaling / standardization fit on full data** | `StandardScaler().fit(X)` then split | Grep for fit calls before the train/val split |
| **Feature selection done before CV split** | `SelectKBest.fit(X_full, y_full)` then split | Selection must be inside the fold loop |
| **HPO score reported on test set** | Tuning on the held-out set instead of inner CV | Confirm HPO returned mean OOF score, not single-split test score |
| **Group leakage** (same patient/customer/scaffold in train+val) | Grouped data without GroupKFold | Compare ID sets across splits; intersection must be empty |
| **Time leakage** (validation timestamp ≤ training timestamp) | Random split on time series | Print `train_max_ts` and `val_min_ts`; train_max must be < val_min |
| **Same row in train and validation** | Data was duplicated before split | Compare row hashes / index sets across splits |
| **Categorical encoder fit on test data** | `LabelEncoder.fit(X_full)` | Encoder fit must use only training-fold data; rare/unknown categories at test time use a separate fallback |
| **Different random state across folds breaking OOF alignment** | Re-creating folds in each script | Confirm `kfold` column is read from disk, not regenerated |

## Rationalizations to reject

When tempted to skip verification, the following thoughts are **all wrong**. Reject them:

| Rationalization | Why it's wrong |
|---|---|
| "The script exited cleanly, it's fine" | Silent failures exit 0 every day |
| "The metric printed by the script is the metric" | The script could be measuring the wrong thing |
| "The chart got saved, so the chart is right" | A blank or wrong chart also gets saved |
| "It's just a CSV load, no need to verify" | Wrong file, wrong columns, wrong dtypes — all common |
| "I already ran this kind of thing before" | Past correctness doesn't transfer to new data |
| "Verification will take too long" | If verification is expensive, the original step is suspect — investigate |
| "The user is in a hurry" | A wrong answer is slower than a right one |
| "I'm pretty confident" | Confidence is not evidence |

## Process

### Step 1 — Read the step's output

What did the script claim? Match it to a row in the Claim/Requires table.

### Step 2 — Pick the verification

From the Requires column. Add at least one universal-failure-pattern check if the step transformed, joined, or modeled.

### Step 3 — Write a separate verification script

Create `<workdir>/verify_step_<N>.py`. **Different code path** from the original — re-load from disk, recompute via different functions, hand-check sample rows. Never just re-run the original.

### Step 4 — Execute via `ml-engineer-execute`

Capture exit code and output.

### Step 5 — Report

Output exactly this format:

```
## Verification: step <N>

### Claim
<what the step claimed>

### Required evidence
<from the Claim/Requires table>

### Checks run
- <check 1>: <pass | fail | mismatch>
- <check 2>: <pass | fail | mismatch>

### Verdict
<verified | suspect | failed>

### Notes
<one sentence — anything the user should know before we proceed>
```

### Step 6 — Branch

- **verified** → orchestrator continues to next step.
- **suspect** → surface to user with the mismatch; wait for direction.
- **failed** → hand off to `ml-engineer-debug` with the verification output as evidence.

## Hard rules

- The verification script must use a **different code path** than the step it's verifying. Same code, same bug.
- Verification must be **cheap** (≤ 30 seconds). If a thorough check would take longer than the original step, return `Verdict: unverified — too expensive` and tell the user.
- **Never** accept "exit 0" as verification.
- **Never** accept the original step's printed values as verification.
- **Never** skip verification because the step "looks right".

## Output checklist

- [ ] Claim restated in one line, matched to a row in the Claim/Requires table
- [ ] Verification script written separately (`verify_step_<N>.py`), not a re-run of the original
- [ ] At least one universal-failure-pattern check considered
- [ ] Verdict is one of: verified | suspect | failed
- [ ] On `failed`, handed off to debug skill
- [ ] On `suspect`, surfaced to user with specifics
