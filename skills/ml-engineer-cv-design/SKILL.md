---
name: ml-engineer-cv-design
description: Use whenever a modeling task is approved and before any training code is written. Picks the cross-validation scheme based on data shape (classification balance, time order, group structure, sample size) and produces a create_folds.py that adds a kfold column. Do NOT skip — choosing the wrong CV scheme is the #1 silent killer of ML projects.
license: MIT
metadata:
  source: ml-engineer
  version: 0.1.0
---

# CV Design

## Iron Law

> **The CV scheme is decided before the first line of training code is written. Never after.**

Cross-validation is not a hyperparameter. It is the contract between training and reality. If validation does not represent how the model will be used, every metric downstream is a lie.

## Decision rule

Pick the scheme by inspecting the data, not by habit. The decision is mechanical given the answers to four questions:

```
1. Is there a time/sequence dimension that must not leak?
   → YES → walk-forward / hold-out on the future. STOP.
   → NO  → continue.

2. Are there grouped rows (multiple rows per entity — patient, customer,
   user, molecule scaffold, store)?
   → YES → GroupKFold on the group key. STOP.
   → NO  → continue.

3. Is the task classification?
   → YES → StratifiedKFold (always, regardless of balance).
   → NO  → continue (regression).

4. Is the regression target distribution uneven (heavy-tailed, multimodal,
   small N)?
   → YES → bin the target with Sturge's rule (bins = 1 + log2(N)),
           then StratifiedKFold on the bins. Drop the bin column before
           training.
   → NO  → KFold.
```

If `N > 1,000,000` and a single fold is fast enough for what you need, swap CV for a held-out tail (e.g. last 100k rows or last 10% by time). State this explicitly in the plan.

## Hard rules

- **Same fold IDs throughout the project.** Save them once in a `kfold` column. Every script — training, target encoding, feature selection, HPO, stacking — reads the same file. Never re-shuffle, never re-create folds with a different seed mid-project.
- **GroupKFold beats StratifiedKFold when groups exist.** A patient appearing in train and validation is leakage even if the labels are stratified. If both group structure and class imbalance are present, use GroupKFold and accept the imbalance — there is no clean StratifiedGroupKFold in vanilla sklearn for binary; if you need it, write it yourself or use sklearn ≥1.0's `StratifiedGroupKFold`.
- **Walk-forward, not k-fold, for time-indexed data.** Never randomly shuffle a time series before splitting. Confirm the maximum training timestamp is strictly before the minimum validation timestamp.
- **Defaults: 5 folds, `shuffle=True`, fixed `random_state=42`** unless there's a reason to deviate. Document any deviation in the plan.

## Process

### Step 1 — Inspect the data

Run a small probe script (via `ml-engineer-execute`) that prints:

- `df.shape`
- `df.dtypes` for the target and any obvious group / time columns
- For classification: `df[target].value_counts(normalize=True)` to see balance
- For regression: `df[target].describe()` and a histogram (saved to charts) to see distribution shape
- If a candidate group column exists (`patient_id`, `user_id`, `customer_id`, `scaffold_id`, etc.): `df.groupby(group_col).size().describe()` — are there really multiple rows per entity?
- If a candidate time column exists: `df[time_col].min(), df[time_col].max(), df[time_col].is_monotonic_increasing`

### Step 2 — Apply the decision rule

Walk through the four questions above. Write the answer to each one in your reply, with the column name(s) cited. No guessing.

### Step 3 — Write `create_folds.py`

Save to `<workdir>/src/create_folds.py` (creating `<workdir>/src/` if it doesn't exist — this aligns with `ml-engineer-write-code`'s project layout).

The script reads the input data, adds a `kfold` integer column (0 to `n_splits-1`), saves to `<workdir>/input/<name>_folds.csv`, and prints fold sizes + per-fold target distribution for verification.

Templates:

**Stratified k-fold (classification)**
```python
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/<filename>.csv")
    df["kfold"] = -1
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    y = df.<target>.values
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (_, v_idx) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_idx, "kfold"] = f
    df.to_csv("../input/<filename>_folds.csv", index=False)
    print(df.kfold.value_counts())
    for k in range(5):
        print(f"Fold {k} target distribution:")
        print(df[df.kfold == k].<target>.value_counts(normalize=True))
```

**KFold (regression, even target)**
```python
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/<filename>.csv")
    df["kfold"] = -1
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    kf = model_selection.KFold(n_splits=5)
    for f, (_, v_idx) in enumerate(kf.split(X=df)):
        df.loc[v_idx, "kfold"] = f
    df.to_csv("../input/<filename>_folds.csv", index=False)
    print(df.kfold.value_counts())
```

**Stratified k-fold on binned target (regression, uneven target)**
```python
import numpy as np
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/<filename>.csv")
    df["kfold"] = -1
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    num_bins = int(np.floor(1 + np.log2(len(df))))   # Sturge's rule
    df.loc[:, "bins"] = pd.cut(df.<target>, bins=num_bins, labels=False)
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (_, v_idx) in enumerate(kf.split(X=df, y=df.bins.values)):
        df.loc[v_idx, "kfold"] = f
    df = df.drop("bins", axis=1)
    df.to_csv("../input/<filename>_folds.csv", index=False)
    print(df.kfold.value_counts())
```

**GroupKFold (grouped data)**
```python
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/<filename>.csv")
    df["kfold"] = -1
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    groups = df.<group_col>.values
    gkf = model_selection.GroupKFold(n_splits=5)
    for f, (_, v_idx) in enumerate(gkf.split(X=df, groups=groups)):
        df.loc[v_idx, "kfold"] = f
    df.to_csv("../input/<filename>_folds.csv", index=False)
    print(df.kfold.value_counts())
    # CRITICAL: confirm no group leakage
    for k in range(5):
        train_groups = set(df[df.kfold != k].<group_col>)
        val_groups = set(df[df.kfold == k].<group_col>)
        assert not (train_groups & val_groups), f"Group leakage in fold {k}"
    print("No group leakage detected.")
```

**Walk-forward (time series)**
```python
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("../input/<filename>.csv")
    df = df.sort_values("<time_col>").reset_index(drop=True)
    df["kfold"] = -1
    n = len(df)
    fold_size = n // 6   # 5 expanding-window folds
    for f in range(5):
        start = (f + 1) * fold_size
        end = (f + 2) * fold_size
        df.loc[start:end - 1, "kfold"] = f
    df = df[df.kfold != -1].reset_index(drop=True)
    df.to_csv("../input/<filename>_folds.csv", index=False)
    # CRITICAL: confirm no time leakage
    for k in range(5):
        train_max = df[df.kfold < k]["<time_col>"].max() if k > 0 else None
        val_min = df[df.kfold == k]["<time_col>"].min()
        if train_max is not None:
            assert train_max < val_min, f"Time leakage in fold {k}"
    print("No time leakage detected.")
```

### Step 4 — Run create_folds.py and verify

Execute via `ml-engineer-execute`. Then verify (via `ml-engineer-verify`):

- All 5 folds have similar sizes (within 5% of each other)
- For classification: per-fold target distribution is similar to the global distribution
- For GroupKFold: assertion in the script passed (no shared groups)
- For walk-forward: assertion in the script passed (no time leakage)

### Step 5 — Document the choice

In the plan output, add one paragraph stating:

```
## CV Scheme
- **Type:** <StratifiedKFold | KFold | StratifiedKFold-on-bins | GroupKFold | Walk-forward>
- **Reason:** <one sentence citing the data property that drove the choice>
- **n_splits:** 5
- **Group / time column:** <name, or N/A>
- **Folds file:** <workdir>/input/<filename>_folds.csv
```

Every downstream skill (`write-code`, `pick-metric`, `verify`, future `tune-hyperparams`, `ensemble`) reads this file. Never re-create folds with a different seed.

## Anti-patterns

- **Random k-fold on a time series.** Future leaks into past. Don't.
- **StratifiedKFold when groups exist.** Same patient in train + validation. Metrics will look great, deployment will fail.
- **Re-creating folds in each script with a different seed.** OOF predictions stop aligning. Stacking breaks. Comparisons across runs become meaningless.
- **"I'll just use train_test_split."** That is a hold-out, not CV. It is acceptable only when N is huge and you stated it in the plan.
- **Skipping the per-fold target-distribution check** for classification. If a fold is missing a class, k-fold gave you a corrupt split.

## Output checklist

- [ ] Probed data: shape, target distribution, group/time columns identified
- [ ] Walked through the 4-question decision rule with column names cited
- [ ] Wrote `<workdir>/src/create_folds.py` using the right template
- [ ] Executed it and got non-trivial fold sizes
- [ ] For grouped/time data: leakage assertion passed
- [ ] Documented the choice in the plan with the standard block above
