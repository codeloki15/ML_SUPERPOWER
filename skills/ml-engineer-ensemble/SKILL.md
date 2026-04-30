---
name: ml-engineer-ensemble
description: Use after 2+ models have been trained on the same folds with OOF predictions saved, when the user asks to ensemble / stack / blend / combine models, or when single-model performance has plateaued. Combines OOF predictions via simple/weighted/rank average or stacking. Do NOT use until the OOF predictions exist and were generated on the same kfold column.
license: MIT
metadata:
  source: ml-engineer
  version: 0.1.0
---

# Ensemble & Stack

## Iron Law (stated twice on purpose, per Thakur)

> **The first rule of ensembling is that you always create folds before starting with ensembling.**
> **The second rule of ensembling is that you always create folds before starting with ensembling.**

Ensembling models trained on different folds, or worse, on the full dataset, is leakage. The OOF predictions used as L2 features must come from the same `kfold` column that all base models used.

## Prerequisites

This skill refuses to run unless **all** of these are true:

- Two or more base models trained via `train.py --fold N --model X` (Layout B)
- OOF prediction files exist at `<workdir>/models/oof_<model>_fold<N>.csv` for every model and every fold
- All OOF files used the same `kfold` column from `<workdir>/input/<name>_folds.csv`
- Models are reasonably uncorrelated (Pearson r between OOF predictions < 0.95) — highly correlated models don't ensemble usefully

If any prerequisite fails, refuse with the specific reason and route back.

## Strategy ladder (escalate only if prior plateaus)

```
1. Simple average of probabilities         (zero risk, often near-optimal)
2. Rank average                            (when metric is AUC — averages ranks, not probs)
3. Weighted average via fmin               (Dirichlet-init weights, optimized on OOF)
4. Stacking (L2 model on OOF features)     (more powerful, more overfit risk)
5. Blending                                (stacking with single hold-out instead of folds)
```

Thakur's recommendation: try simple average first. It's astonishingly close to weighted/stacked in many cases and has zero overfitting risk on the L2 step.

## Building OOF predictions (recap)

This is what `train.py` already produces in Layout B. Re-stating for clarity:

```python
# Inside train.py, after model.fit on training fold and predict on validation fold:
oof = pd.DataFrame({
    "row_id": df_valid.index,
    "y_true": y_valid,
    "y_pred": valid_preds,    # probability for class 1, or regression prediction
    "fold": fold,
    "model": model_name,
})
oof.to_csv(f"../models/oof_{model_name}_fold{fold}.csv", index=False)
```

To assemble all OOF predictions for one model into a single Series aligned with the original training data:

```python
def assemble_oof(model_name: str, n_folds: int = 5) -> pd.Series:
    parts = []
    for fold in range(n_folds):
        parts.append(pd.read_csv(f"../models/oof_{model_name}_fold{fold}.csv"))
    df = pd.concat(parts).sort_values("row_id").reset_index(drop=True)
    return df.set_index("row_id")["y_pred"]
```

## Strategy 1 — simple average

```python
# src/ensemble_avg.py
import numpy as np
import pandas as pd
from sklearn import metrics

import config

MODELS = ["logreg", "rf", "xgb"]


def assemble_oof(model_name, n_folds=config.N_FOLDS):
    parts = [pd.read_csv(f"../models/oof_{model_name}_fold{f}.csv") for f in range(n_folds)]
    return pd.concat(parts).sort_values("row_id").reset_index(drop=True)


if __name__ == "__main__":
    base = None
    cols = []
    for m in MODELS:
        oof = assemble_oof(m)
        if base is None:
            base = oof[["row_id", "y_true"]].copy()
        base[m] = oof["y_pred"].values
        cols.append(m)

    base["avg"] = base[cols].mean(axis=1)

    # Per-model and ensemble OOF metrics
    for col in cols + ["avg"]:
        score = metrics.roc_auc_score(base["y_true"], base[col])
        print(f"OOF {config.METRIC} {col:>10s}: {score:.5f}")
```

## Strategy 2 — rank average (for AUC)

AUC only cares about ranks, so averaging ranks is more stable than averaging probabilities:

```python
from scipy.stats import rankdata

ranked = np.column_stack([rankdata(base[m]) for m in MODELS])
base["rank_avg"] = ranked.mean(axis=1)
score = metrics.roc_auc_score(base["y_true"], base["rank_avg"])
```

## Strategy 3 — weighted average via fmin

Optimize weights on OOF predictions to maximize the locked metric:

```python
# src/ensemble_weighted.py
from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import fmin
from sklearn import metrics

import config

MODELS = ["logreg", "rf", "xgb"]


def neg_auc(weights, X, y):
    """Negative AUC of weighted prediction (fmin minimizes)."""
    pred = np.sum(X * weights, axis=1)
    return -metrics.roc_auc_score(y, pred)


def assemble(model_name, n=config.N_FOLDS):
    parts = [pd.read_csv(f"../models/oof_{model_name}_fold{f}.csv") for f in range(n)]
    return pd.concat(parts).sort_values("row_id").reset_index(drop=True)


if __name__ == "__main__":
    parts = [assemble(m) for m in MODELS]
    y = parts[0]["y_true"].values
    X = np.column_stack([p["y_pred"].values for p in parts])

    # Initialize with Dirichlet so weights start as a probability vector
    init = np.random.default_rng(42).dirichlet(np.ones(len(MODELS)))
    loss = partial(neg_auc, X=X, y=y)
    best = fmin(loss, init, disp=True)

    print(f"Best weights: {dict(zip(MODELS, best))}")
    final = np.sum(X * best, axis=1)
    print(f"Weighted-avg OOF {config.METRIC}: {metrics.roc_auc_score(y, final):.5f}")
```

Note: weights from `fmin` may not sum to 1 — that's fine when the metric is rank-based (AUC). For metrics that care about scale (log loss, RMSE), normalize weights or use constrained optimization.

## Strategy 4 — stacking with L2 model

L1 OOF predictions become features for an L2 model, fit on the **same folds**:

```python
# src/stack.py
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics

import config

L1_MODELS = ["logreg", "rf", "xgb"]


def assemble(model_name, n=config.N_FOLDS):
    parts = [pd.read_csv(f"../models/oof_{model_name}_fold{f}.csv") for f in range(n)]
    return pd.concat(parts).sort_values("row_id").reset_index(drop=True)


if __name__ == "__main__":
    base = assemble(L1_MODELS[0])[["row_id", "y_true"]].copy()
    base["fold"] = pd.read_csv(config.TRAINING_FILE).sort_index().reset_index(drop=True)["kfold"].values
    for m in L1_MODELS:
        base[m] = assemble(m)["y_pred"].values

    # L2 model trained on L1 OOF features, same folds
    l2_oof = np.zeros(len(base))
    for fold in range(config.N_FOLDS):
        train_idx = base["fold"] != fold
        valid_idx = base["fold"] == fold
        l2 = linear_model.LogisticRegression()
        l2.fit(base.loc[train_idx, L1_MODELS], base.loc[train_idx, "y_true"])
        l2_oof[valid_idx] = l2.predict_proba(base.loc[valid_idx, L1_MODELS])[:, 1]

    print(f"Stacked OOF {config.METRIC}: {metrics.roc_auc_score(base['y_true'], l2_oof):.5f}")
```

L2 model choice:
- **Logistic regression / Ridge / Lasso** — standard, low risk of overfitting on top of L1
- **XGBoost / LightGBM with shallow depth** — more powerful, but watch for L2 overfitting to L1 noise
- **Simple linear blender** is almost always competitive with fancier L2 choices

## Strategy 5 — blending

Stacking with a single hold-out instead of folds. Simpler, less robust. Use only when:
- Computing OOF for L1 is too expensive
- The dataset is huge enough that one large hold-out is statistically reliable

Otherwise prefer stacking.

## Hard rules

- **All L1 models trained on identical folds.** If `model_a` used folds A and `model_b` used folds B, refuse to ensemble.
- **L2 model trained on the same folds as L1.** Re-using a different split for L2 leaks future-fold L1 predictions into past-fold L2 training.
- **L2 features = L1 OOF predictions only**, not L1 + raw features. Adding raw features back is technically valid (called "stacking with original features") but doubles the overfitting risk; only attempt after the simple stack works.
- **Ensemble metric must beat the best single model on OOF, not just match it.** If your ensemble matches the best single model, the ensemble adds complexity for no value — drop it.
- **Correlation check before ensembling.** Print the Pearson r matrix of L1 OOF predictions. If two models have r > 0.97, ensembling them is a no-op.
- **Diversity matters more than individual scores.** A logreg with AUC=0.78 + an xgb with AUC=0.85 often ensembles to 0.86. Two xgbs with AUC=0.85 each often ensemble to 0.85.

## Process

### Step 1 — Verify prerequisites

```python
# Refuse and explain if any of these fail
oof_files_exist = all(
    os.path.exists(f"../models/oof_{m}_fold{f}.csv")
    for m in MODELS for f in range(config.N_FOLDS)
)
```

### Step 2 — Print correlation matrix

```python
oof_df = pd.DataFrame({m: assemble(m)["y_pred"].values for m in MODELS})
print("OOF correlation matrix:")
print(oof_df.corr().round(3))
```

If any pair has r > 0.97, suggest dropping one before ensembling.

### Step 3 — Apply strategy ladder

Run simple average. If that doesn't beat the best single model meaningfully, try rank average (for AUC) or weighted average. Stack only if the linear combinations plateau.

### Step 4 — Document in the plan

```
## Ensemble
- **Base models:** [logreg, rf, xgb]
- **OOF correlations:** logreg-rf=0.62, logreg-xgb=0.71, rf-xgb=0.83
- **Strategies tried:**
  - simple avg: AUC=0.871 (best single: rf=0.864 → +0.7pp)
  - rank avg:   AUC=0.873
  - weighted:   AUC=0.875 (weights: logreg=0.18, rf=0.42, xgb=0.40)
  - stack:      AUC=0.876 (L2: logreg)
- **Chosen:** rank avg — simplest strategy beating the best single model by ≥0.5pp
```

### Step 5 — Verify

`ml-engineer-verify` will check:
- All L1 models used the same fold IDs (Critical if not)
- L2 model uses the same folds as L1 (Critical if not)
- Ensemble metric is computed on OOF, not on training predictions (Critical if not)
- Ensemble beats best single model on OOF — if not, the ensemble adds complexity for no value (Important)

## Anti-patterns

- **Ensembling models trained on different splits.** Wrong by construction.
- **Stacking with L2 trained on full L1 predictions.** That's training L2 on data L1 has seen — pure leakage.
- **Adding 5 nearly-identical XGBoost models with different seeds.** They all see the same biases; ensembling gives ~no improvement.
- **Not checking correlation before ensembling.** You'll happily combine two models that disagree on 2% of cases and conclude ensembling "doesn't work."
- **Reporting ensemble score without OOF correlation matrix.** Reviewer can't tell if the diversity claim is real.
- **Using a complex L2 (XGBoost) when linear blending suffices.** More moving parts, more overfit risk to the same OOF.

## Output checklist

- [ ] All L1 models trained on identical folds (verified)
- [ ] OOF prediction files complete (every model × every fold)
- [ ] Correlation matrix printed
- [ ] Simple average tried first
- [ ] Final ensemble beats the best single model on OOF by a meaningful margin
- [ ] Strategy and weights documented in the plan
- [ ] L2 model (if used) trained on same folds as L1
- [ ] Ensemble metric computed on OOF, not training predictions
