---
name: ml-engineer-pick-metric
description: Use after EDA and before any training code. Picks the primary evaluation metric from task type and target distribution, and locks it into the plan so train, eval, HPO, and ensembling all optimize the same thing. Do NOT skip — the wrong metric (especially accuracy on imbalanced data) makes every later result meaningless.
license: MIT
metadata:
  source: ml-engineer
  version: 0.1.0
---

# Pick Metric

## Iron Law

> **The metric is decided before the first line of training code is written. The same metric is used in train, eval, HPO, and ensembling.**

A metric chosen mid-project, or a different metric for HPO than for evaluation, makes every comparison invalid. Lock it once.

## Decision rule

Mechanical, given task type and target distribution:

```
TASK = {binary classification, multi-class classification, multi-label
        classification, regression, ranking, ordinal classification}

For BINARY CLASSIFICATION:
  If classes are balanced (minority > 35%) → accuracy + F1
  If classes are imbalanced                → AUC primary; F1, precision/recall secondary
  If TP/TN/FP/FN tradeoff matters & skewed → MCC
  If well-calibrated probabilities matter  → log loss

For MULTI-CLASS CLASSIFICATION:
  Balanced classes → accuracy + macro F1
  Imbalanced       → macro F1 (penalizes minority-class failure equally)
  Class frequency reflects business value → weighted F1
  Ordinal-like target (e.g. severity 1-5)  → quadratic weighted kappa (QWK), aim >0.85

For MULTI-LABEL CLASSIFICATION:
  Top-k matters → P@k, AP@k, MAP@k
  All labels matter → mean column-wise log loss

For REGRESSION:
  General                          → RMSE (or MAE if outliers should not dominate)
  Target is positive & skewed (counts, prices, durations) → RMSLE
    → If RMSLE: train on log(1+y) and apply expm1 on predictions.
  Percentage-error semantics matter → MAPE
  Goodness-of-fit interpretability  → R² (alongside, not instead of, an error metric)

For RANKING:
  AUC if pairwise; NDCG@k for graded relevance; MAP@k for binary relevance.

For TIME-SERIES FORECASTING:
  sMAPE, MASE, or weighted RMSE by horizon. Always compare against a naive
  baseline (last-value, seasonal-naive).
```

## Hard rules

- **Never use accuracy on imbalanced classification.** A 90/10 split lets a constant predictor score 90%. Use AUC or F1.
- **Never use a single number for evaluation.** Always print the supporting evidence:
  - Classification → confusion matrix, per-class precision/recall
  - Regression → residual plot or distribution
  - Ranking → top-K precision per K
- **Never optimize HPO on one metric and report another.** Whatever the primary metric is, that is what HPO returns and that is what gets reported.
- **AUC = 1, R² > 0.99, F1 = 1, p < 1e-10 → suspect by default.** These are validation bugs more often than they are real results. The verification skill must audit, not celebrate.

## When the metric implies a target transform

These pairings are non-negotiable:

| Metric | Required transform |
|---|---|
| RMSLE | Train on `log(1 + y)`, predict, then `expm1(pred)` |
| Log loss | Output probabilities, not class predictions; use `predict_proba` |
| AUC | Output probabilities; threshold not needed for the metric itself but required for class predictions |
| QWK | Output integer class predictions; `weights="quadratic"` in `cohen_kappa_score` |
| MCC | Output class predictions, not probabilities |

## Threshold selection (binary classification)

For probability outputs, the default 0.5 threshold is rarely optimal. After training, write a small script that:

1. Computes precision and recall over a sweep of thresholds (e.g. 0.05 to 0.95 step 0.05)
2. Picks the threshold that:
   - Maximizes F1 → balanced default
   - Or matches a target precision (e.g. ≥0.9) when false positives are costly
   - Or matches a target recall (e.g. ≥0.9) when false negatives are costly (medical screening, fraud)
3. Saves the chosen threshold to `<workdir>/models/threshold.txt` for inference reuse

This is a separate analysis step. State it in the plan. Never use 0.5 silently.

## Process

### Step 1 — Confirm task type

Inspect the target column:
- dtype, unique values, value_counts
- Is it binary / multi-class / continuous / multi-label / ranked?

### Step 2 — Apply the decision rule

Walk through the table for the relevant task type. Pick a primary metric and at most 2 supporting metrics. Cite the data property that drove the choice (e.g. "minority class is 12%, so AUC primary").

### Step 3 — Document in the plan

Add to the plan output:

```
## Evaluation
- **Task:** <binary classification | multi-class | regression | ...>
- **Primary metric:** <name> — <one-sentence reason>
- **Supporting metrics:** <list>
- **Required target transform:** <e.g. log(1+y) for RMSLE, or "none">
- **Threshold selection:** <required after training | not applicable>
- **Baseline to beat:** <what counts as "better than nothing" — e.g.
    "majority-class baseline = 0.88 accuracy / 0 F1", or
    "seasonal-naive forecast sMAPE = 12%">
```

### Step 4 — Lock the metric for downstream skills

The plan now constrains:
- `ml-engineer-write-code` — `train.py` evaluates with this metric
- `ml-engineer-verify` — recomputes this metric on hold-out, not the original print
- `ml-engineer-tune-hyperparams` — optimizes this metric via OOF mean
- `ml-engineer-ensemble` — averages probabilities (or ranks if AUC)
- The orchestrator's final report leads with this metric

## Anti-patterns

- **"Let's just use accuracy and see."** No. Pick before training.
- **Optimizing log-loss but reporting AUC.** Pick one or report both with explicit framing.
- **Reporting a single headline number with no supporting evidence.** A 99% accuracy without a confusion matrix is a lie.
- **Switching the metric mid-project because the first one looked bad.** That's p-hacking. Either the original choice was wrong (acknowledge it explicitly and re-run from scratch) or accept the result.
- **Defaulting threshold to 0.5 on imbalanced binary.** Silent precision/recall imbalance — surface and pick deliberately.

## Output checklist

- [ ] Inspected target column and confirmed task type
- [ ] Picked primary metric per the decision rule with cited reason
- [ ] Required target transform stated (or "none")
- [ ] Threshold-selection requirement stated (or "not applicable")
- [ ] Baseline-to-beat named — what is the trivial number we must beat?
- [ ] Documented in the plan with the standard block above
