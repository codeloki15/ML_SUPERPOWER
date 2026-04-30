---
name: ml-engineer-tune-hyperparams
description: Use after a baseline model is trained and verified, when the user asks to tune / optimize / search hyperparameters, or when the baseline result is competitive but not yet optimal. Runs HPO inside the existing CV folds, returns mean OOF score, never tunes on the test set. Do NOT use before a baseline exists — tuning a broken pipeline wastes compute.
license: MIT
metadata:
  source: ml-engineer
  version: 0.1.0
---

# Tune Hyperparameters

## Iron Law

> **HPO runs inside the locked CV folds. The score it returns is mean OOF across folds. The test set is never touched during HPO.**

Tuning on the test set is the most common HPO failure. It produces beautifully optimistic numbers that collapse in production.

## Order of operations

Always in this order — escalate only if the previous step plateaus:

```
1. Hand-tune (1-3 iterations) — learn the model's sensitivities
2. Random search (~30-100 evals) — broad exploration
3. Bayesian: gp_minimize / hyperopt TPE / Optuna (~50-200 evals) — fine-tuning
```

Quote from Thakur (AAAMLP, p. 183): *"to learn, one must start with tuning the hyper-parameters manually... Hand tuning will help you learn the basics."*

Skip hand-tune only when the user has already done it or the model is well-understood for this dataset.

## Per-model starting ranges (Thakur, AAAMLP p. 184)

These are the practical ranges to search. Don't search outside without a reason.

| Model | Param | Range |
|---|---|---|
| **Linear Regression** | fit_intercept | True/False |
| | normalize | True/False |
| **Ridge** | alpha | 0.01, 0.1, 1.0, 10, 100 |
| | fit_intercept | True/False |
| **Lasso** | alpha | 0.1, 1.0, 10 |
| **Logistic Regression** | penalty | l1, l2 |
| | C | 0.001, 0.01, 0.1, 1, 10, 100 |
| **k-NN** | n_neighbors | 2, 4, 8, 16 |
| | p | 2, 3 |
| **SVM** | C | 0.001, 0.01, 0.1, 1, 10, 100, 1000 |
| | gamma | 'auto', RS* |
| | class_weight | 'balanced', None |
| **Random Forest** | n_estimators | 120, 300, 500, 800, 1200 |
| | max_depth | 5, 8, 15, 25, 30, None |
| | min_samples_split | 1, 2, 5, 10, 15, 100 |
| | min_samples_leaf | 1, 2, 5, 10 |
| | max_features | log2, sqrt, None |
| **XGBoost** | eta | 0.01, 0.015, 0.025, 0.05, 0.1 |
| | gamma | 0.05-0.1, 0.3, 0.5, 0.7, 0.9, 1.0 |
| | max_depth | 3, 5, 7, 9, 12, 15, 17, 25 |
| | min_child_weight | 1, 3, 5, 7 |
| | subsample | 0.6, 0.7, 0.8, 0.9, 1.0 |
| | colsample_bytree | 0.6, 0.7, 0.8, 0.9, 1.0 |
| | lambda | 0.01-0.1, 1.0, RS* |
| | alpha | 0, 0.1, 0.5, 1.0, RS* |

`RS*` = random search recommended for that param.

**For LightGBM**, use the XGBoost ranges as a starting point — most parameters are analogous (`learning_rate ↔ eta`, `num_leaves` typically 15-127).

## Hard rules

- **HPO score is mean OOF across all folds, not single-fold.** `gp_minimize` or `hyperopt` should return `-1 * np.mean(fold_scores)` (negative because they minimize).
- **Same `kfold` column throughout.** Read from `<workdir>/input/<name>_folds.csv`. Don't re-create folds.
- **The test set is never seen.** No "let me check on test to confirm" mid-search. After HPO finishes, the chosen params get one final OOF run plus optionally a single test evaluation if a held-out test exists.
- **Optimize the locked metric.** Whatever `ml-engineer-pick-metric` picked is what HPO minimizes/maximizes. Don't optimize log-loss and report AUC.
- **Search budget bounded.** Random search ≤ 100 iters by default. Bayesian ≤ 200. Anything more requires explicit user opt-in — users get bored before models do.
- **Parameters from one dataset don't transfer.** Don't import "good XGBoost params from Kaggle" — re-search per dataset.
- **HPO doesn't fix bad CV.** If validation has leakage, HPO finds the leakiest config. Run `ml-engineer-verify` on the baseline model before tuning.

## Templates

All templates assume Layout B is in place: `<workdir>/input/<name>_folds.csv` exists, `train.py` works for `--fold N --model X`.

### Random search (sklearn)

```python
# src/tune_random.py
import argparse
import numpy as np
import pandas as pd
from sklearn import ensemble, model_selection, metrics

import config


def evaluate(params: dict, df: pd.DataFrame, feature_cols: list) -> float:
    """Run 5-fold CV with the given params, return mean OOF metric."""
    scores = []
    for fold in range(config.N_FOLDS):
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        x_train = df_train[feature_cols].values
        y_train = df_train[config.TARGET].values
        x_valid = df_valid[feature_cols].values
        y_valid = df_valid[config.TARGET].values

        model = ensemble.RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(x_train, y_train)
        preds = model.predict_proba(x_valid)[:, 1]
        scores.append(metrics.roc_auc_score(y_valid, preds))
    return float(np.mean(scores))


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    feature_cols = [c for c in df.columns if c not in ("kfold", config.TARGET)]

    param_grid = {
        "n_estimators": [120, 300, 500, 800, 1200],
        "max_depth": [5, 8, 15, 25, 30, None],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["log2", "sqrt", None],
    }

    rng = np.random.RandomState(42)
    n_iter = 50
    results = []
    for i in range(n_iter):
        params = {k: rng.choice(v) for k, v in param_grid.items()}
        # Convert numpy types to Python types (sklearn dislikes np.int64 here)
        params = {k: (None if v is None else (int(v) if isinstance(v, (np.integer,)) else v)) for k, v in params.items()}
        score = evaluate(params, df, feature_cols)
        results.append((score, params))
        print(f"[{i+1}/{n_iter}] {config.METRIC}={score:.5f} params={params}")

    results.sort(reverse=True)
    print("\nTop 5:")
    for score, params in results[:5]:
        print(f"  {config.METRIC}={score:.5f} params={params}")
```

### Bayesian — gp_minimize (scikit-optimize)

```python
# src/tune_bayes.py
from functools import partial

import numpy as np
import pandas as pd
from skopt import gp_minimize, space
from sklearn import ensemble, metrics

import config


def evaluate(params, param_names, df, feature_cols, target):
    p = dict(zip(param_names, params))
    scores = []
    for fold in range(config.N_FOLDS):
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        model = ensemble.RandomForestClassifier(
            n_estimators=int(p["n_estimators"]),
            max_depth=int(p["max_depth"]),
            criterion=p["criterion"],
            max_features=p["max_features"],
            random_state=42, n_jobs=-1,
        )
        model.fit(df_train[feature_cols].values, df_train[target].values)
        preds = model.predict_proba(df_valid[feature_cols].values)[:, 1]
        scores.append(metrics.roc_auc_score(df_valid[target].values, preds))
    return -1.0 * np.mean(scores)   # negative because gp_minimize minimizes


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    feature_cols = [c for c in df.columns if c not in ("kfold", config.TARGET)]

    param_space = [
        space.Integer(3, 30, name="max_depth"),
        space.Integer(100, 1500, name="n_estimators"),
        space.Categorical(["gini", "entropy"], name="criterion"),
        space.Real(0.01, 1.0, prior="uniform", name="max_features"),
    ]
    param_names = ["max_depth", "n_estimators", "criterion", "max_features"]

    objective = partial(evaluate, param_names=param_names, df=df,
                        feature_cols=feature_cols, target=config.TARGET)

    result = gp_minimize(
        objective,
        dimensions=param_space,
        n_calls=50,
        n_random_starts=10,
        random_state=42,
        verbose=True,
    )

    best = dict(zip(param_names, result.x))
    print(f"\nBest mean OOF {config.METRIC}: {-result.fun:.5f}")
    print(f"Best params: {best}")
```

### Bayesian — hyperopt TPE

```python
# src/tune_hyperopt.py
import numpy as np
import pandas as pd
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
from sklearn import ensemble, metrics

import config


def evaluate(params, df, feature_cols, target):
    scores = []
    for fold in range(config.N_FOLDS):
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        model = ensemble.RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(df_train[feature_cols].values, df_train[target].values)
        preds = model.predict_proba(df_valid[feature_cols].values)[:, 1]
        scores.append(metrics.roc_auc_score(df_valid[target].values, preds))
    return -1.0 * np.mean(scores)


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    feature_cols = [c for c in df.columns if c not in ("kfold", config.TARGET)]

    space = {
        "max_depth": scope.int(hp.quniform("max_depth", 3, 30, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 1500, 50)),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "max_features": hp.uniform("max_features", 0.1, 1.0),
    }

    trials = Trials()
    best = fmin(
        fn=lambda p: evaluate(p, df, feature_cols, config.TARGET),
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        rstate=np.random.default_rng(42),
    )
    print(f"\nBest params: {best}")
    print(f"Best mean OOF: {-min(t['result']['loss'] for t in trials.trials):.5f}")
```

## When the metric isn't built-in

Use `make_scorer` to wrap a custom metric (e.g., quadratic weighted kappa for ordinal classification):

```python
from sklearn.metrics import make_scorer, cohen_kappa_score

def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

qwk_scorer = make_scorer(qwk, greater_is_better=True)
# Pass scoring=qwk_scorer to GridSearchCV / RandomizedSearchCV / cross_val_score
```

For `gp_minimize` / `hyperopt`, just call your custom metric inside the objective and negate it.

## Process

### Step 1 — Confirm baseline exists

If no baseline has been trained and verified yet, refuse and route back to baseline training. HPO without a baseline is wasted compute.

### Step 2 — Pick search strategy

- Hand-tune first if model is unfamiliar to the user
- Random search if 4+ params being tuned
- Bayesian if individual evals are cheap (< 1 min) and we have budget for ~50-100

### Step 3 — Constrain search space

Use the table above. Document the chosen ranges in the plan:

```
## HPO
- **Model:** RandomForest
- **Strategy:** random search, 50 iters
- **Search space:**
  - n_estimators: [120, 300, 500, 800, 1200]
  - max_depth: [5, 8, 15, 25, 30, None]
  - ...
- **Metric:** AUC (locked from pick-metric)
- **CV:** read from kfold column in <name>_folds.csv (5 folds)
- **Budget:** 50 evals, ~10 minutes expected
```

### Step 4 — Write tune_X.py and run

Per templates above. Always returns mean OOF score, never single-fold.

### Step 5 — Re-train with best params

After HPO finishes, update `model_dispatcher.py` with the chosen params and re-run `train.py --fold N --model <name>` for all folds. Save final OOF predictions for downstream ensembling.

### Step 6 — Verify

`ml-engineer-verify` will check:
- HPO returned mean OOF, not single-fold (auto-Critical if violated)
- Test set was not touched during HPO (auto-Critical)
- Final model uses the chosen params on the same folds

## Anti-patterns

- **Tuning on the held-out test set.** Auto-Critical.
- **Re-creating folds with a different seed inside the tuner.** Breaks comparability with baseline.
- **Tuning before verification of the baseline.** If baseline has a leakage bug, HPO maximizes the leak.
- **Searching outside Thakur's ranges without justification.** `n_estimators=10000` is rarely better than 1500 and 6× slower.
- **Optimizing one metric and reporting another.** If AUC is locked, optimize AUC.
- **Running 1000+ iters because "more is better".** Diminishing returns kick in fast; the model and data quality dominate.

## Output checklist

- [ ] Baseline exists and is verified
- [ ] Strategy picked (hand-tune / random / Bayesian) with rationale
- [ ] Search space matches Thakur's table or has explicit justification for deviation
- [ ] Objective returns mean OOF, not single-fold
- [ ] kfold column read from disk, not regenerated
- [ ] Test set untouched
- [ ] Locked metric optimized
- [ ] Best params printed and used to retrain via `model_dispatcher.py`
- [ ] Search budget reasonable (≤100 random / ≤200 Bayesian by default)
