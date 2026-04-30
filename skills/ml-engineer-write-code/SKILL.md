---
name: ml-engineer-write-code
description: Use when an approved plan exists and the orchestrator is implementing a specific step in Python. Do NOT use without an approved plan, for general code edits, or for writing scripts outside the current session workdir.
license: MIT
metadata:
  source: ml-engineer
  version: 0.1.0
---

# Write Code

Generate complete, executable Python for one plan step. Scripts run unattended under the venv and exit cleanly.

## Two layouts — pick one before writing

### Layout A — flat (for EDA, cleaning, charting, one-off analysis)

Save to `<workdir>/step_<N>_<short-name>.py`. Single script, no CLI args.

Use this when the step is genuinely one-shot: load + describe, plot, compute a statistic, clean and save a CSV. ~80% of non-modeling steps.

### Layout B — project layout (for any model training)

Once a CV scheme is locked (via `ml-engineer-cv-design`) and a metric is locked (via `ml-engineer-pick-metric`), training code uses this structure under `<workdir>/`:

```
<workdir>/
├── input/                          # data + folds CSV
│   ├── <name>.csv
│   └── <name>_folds.csv
├── src/
│   ├── config.py                   # paths, fold count, metric name
│   ├── create_folds.py             # written by ml-engineer-cv-design
│   ├── model_dispatcher.py         # name → model instance
│   ├── train.py                    # CLI: --fold N --model NAME
│   ├── inference.py                # loads models/, predicts on new data
│   └── run.sh                      # loops folds × models
├── models/                         # joblib artifacts per fold per model
│   └── <model_name>_fold<N>.bin
└── charts/
```

Why this exists (from Thakur, AAAMLP):

- **One fold, one model, one process.** `train.py --fold 0 --model rf` is independently runnable. Memory leaks across folds disappear.
- **Adding a model is one entry in `model_dispatcher.py`.** No edits to `train.py`.
- **OOF predictions are cleanly produced per fold.** Required for any future ensembling/stacking.
- **Inference reuses `models/`.** Train once, predict many.

When the plan step is "train a model", "tune hyperparameters", or "evaluate a model", **Layout B is mandatory**. Don't write a flat training script.

## Layout B templates

### `config.py`

```python
TRAINING_FILE = "../input/<name>_folds.csv"
MODEL_OUTPUT = "../models/"
N_FOLDS = 5
TARGET = "<target_col>"
METRIC = "<from pick-metric step>"  # used only as a label; the metric function lives in train.py
```

### `model_dispatcher.py`

```python
from sklearn import ensemble, linear_model, tree
import xgboost as xgb

models = {
    "logreg": linear_model.LogisticRegression(max_iter=1000),
    "rf":     ensemble.RandomForestClassifier(n_jobs=-1, random_state=42),
    "xgb":    xgb.XGBClassifier(n_jobs=-1, max_depth=7, n_estimators=200,
                                use_label_encoder=False, eval_metric="logloss"),
}
```

Adapt to the task (regressors for regression). Tree models accept label-encoded data; linear/SVM/NN need one-hot or scaling.

### `train.py`

```python
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics

import config
import model_dispatcher


def run(fold: int, model_name: str) -> None:
    """Train one model on one fold, save the artifact, save OOF predictions."""
    df = pd.read_csv(config.TRAINING_FILE)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ("kfold", config.TARGET)]
    x_train = df_train[feature_cols].values
    y_train = df_train[config.TARGET].values
    x_valid = df_valid[feature_cols].values
    y_valid = df_valid[config.TARGET].values

    model = model_dispatcher.models[model_name]
    model.fit(x_train, y_train)

    # For classification AUC; swap for the locked metric.
    valid_preds = model.predict_proba(x_valid)[:, 1] if hasattr(model, "predict_proba") else model.predict(x_valid)
    score = metrics.roc_auc_score(y_valid, valid_preds)
    print(f"Fold={fold} Model={model_name} {config.METRIC}={score:.5f}")

    os.makedirs(config.MODEL_OUTPUT, exist_ok=True)
    joblib.dump(model, os.path.join(config.MODEL_OUTPUT, f"{model_name}_fold{fold}.bin"))

    # Save OOF predictions for this fold (required for ensembling later)
    oof = pd.DataFrame({
        "row_id": df_valid.index,
        "y_true": y_valid,
        "y_pred": valid_preds,
        "fold": fold,
        "model": model_name,
    })
    oof_path = os.path.join(config.MODEL_OUTPUT, f"oof_{model_name}_fold{fold}.csv")
    oof.to_csv(oof_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    run(fold=args.fold, model_name=args.model)
```

Adapt the metric line to whatever `pick-metric` locked in (RMSE for regression, log-loss, F1, QWK, etc.). For RMSLE, train on `np.log1p(y_train)` and apply `np.expm1(...)` to predictions before scoring.

### `run.sh`

```bash
#!/usr/bin/env bash
set -e

MODELS=("logreg" "rf" "xgb")
N_FOLDS=5

for model in "${MODELS[@]}"; do
    for fold in $(seq 0 $((N_FOLDS - 1))); do
        python train.py --fold "$fold" --model "$model"
    done
done
```

### `inference.py` (when needed)

```python
import os

import joblib
import numpy as np
import pandas as pd

import config


def predict(test_csv: str, model_name: str) -> np.ndarray:
    """Average predictions across all folds of a single model."""
    test = pd.read_csv(test_csv)
    feature_cols = [c for c in test.columns if c != config.TARGET]
    preds = np.zeros(len(test))
    for fold in range(config.N_FOLDS):
        model = joblib.load(os.path.join(config.MODEL_OUTPUT, f"{model_name}_fold{fold}.bin"))
        if hasattr(model, "predict_proba"):
            preds += model.predict_proba(test[feature_cols].values)[:, 1]
        else:
            preds += model.predict(test[feature_cols].values)
    return preds / config.N_FOLDS
```

## Hard requirements (apply to both layouts)

- **Complete and executable** — no placeholders, no `pass`, no `TODO`. If you can't fill in a value, ask before writing.
- **No interaction** — never `input()`, `getpass`, or any prompt that waits on stdin. (`argparse` is allowed in Layout B for `--fold` / `--model`, but no interactive input.)
- **No display calls** — never `plt.show()`, never bare `df.head()`. Always `print(...)`.
- **Charts saved, not shown** — `plt.savefig(os.path.join(CHART_DIR, '<name>.png'))` then `plt.close()` then `print(f"Chart saved as <name>.png")`. `CHART_DIR = "../charts/"` in Layout B, `<workdir>/charts/` in Layout A.
- **No web servers, no daemons** — no Flask / FastAPI / uvicorn / gradio / streamlit / dash.
- **Bounded execution** — every loop has a clear exit. Print progress for long ops.
- **No file deletion** — never `os.remove`, `shutil.rmtree`, `rm -rf`.
- **No network calls** unless the plan step explicitly requires fetching from a named URL.
- **Random seed fixed** for any stochastic step. Default: `random_state=42`.

## Modeling-specific rules (Layout B)

These prevent the silent failures Thakur calls out:

- **Read folds from disk; never re-create them inside `train.py`.** The `kfold` column in `<name>_folds.csv` is the source of truth.
- **Categorical encoding fit on training fold only**, never on full data. `LabelEncoder.fit(df_train[col])` then `.transform(df_valid[col])`. For unseen categories at validation time, use the rare-category mechanic (see below).
- **Scaling fit on training fold only.** Never `StandardScaler().fit(full_df)` then split.
- **Target encoding is per-fold.** Compute the train-fold mean per category, then map onto validation. Never compute on full data and then split.
- **Always `fillna` categorical columns before encoding.** `df[col] = df[col].astype(str).fillna("NONE")`. `LabelEncoder` raises on NaN.
- **Rare category handling.** If a category's count in training is below threshold (default 1% of N or 10, whichever larger), replace with `"RARE"`. Apply the same rule at validation time so unseen categories also map to `"RARE"`.
- **For RMSLE metric**: train on `np.log1p(y)` and apply `np.expm1` on predictions before scoring.
- **For AUC / log loss**: `predict_proba`, not `predict`. Output the positive-class probability.
- **For tree models (RF, XGB, LightGBM, CatBoost)**: label encoding is fine, no scaling needed, no one-hot needed.
- **For linear / SVM / NN models**: one-hot encode categoricals, scale numerical features, ideally inside a `Pipeline` so the scaler fits per fold automatically.

## Pandas display defaults

If the script uses pandas, include near the top:

```python
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 200)
```

## Error handling

- Wrap risky ops (file IO, model fitting, parsing) in `try/except`.
- In each `except`, print the error with context and `raise`. Never silently swallow.
- Validate inputs before processing: file exists, columns present, dtypes as expected.

## Imports

Standard library first, then third-party, alphabetized within group. Only import what you use. No `import *`.

## Style

PEP 8. Functions for any logic block longer than ~15 lines. One-line docstrings. Inline comments only where the code is non-obvious.

## Output checklist

- [ ] Picked the right layout (A for EDA / one-off, B for any training)
- [ ] File path(s) stated above the code block(s)
- [ ] No `plt.show()`, `input()`, web server, or `os.remove`
- [ ] Charts saved with `savefig` + `close` + `print`
- [ ] All `except` blocks re-raise
- [ ] Pandas display options set if pandas used
- [ ] Random seed fixed
- [ ] **Layout B only:** folds read from disk (not re-created); encoding/scaling fit on training fold only; OOF predictions saved; per-fold per-model artifacts saved to `models/`
- [ ] Script ends with a clear summary `print(...)`
