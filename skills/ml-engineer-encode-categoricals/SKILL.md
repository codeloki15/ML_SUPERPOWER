---
name: ml-engineer-encode-categoricals
description: Use after CV is designed and before training, whenever the dataset has categorical columns (object dtype, string columns, low-cardinality integers used as categories). Picks the right encoding (label / one-hot / target / entity embedding) for the model family and applies it without leakage. Do NOT skip when categoricals exist — wrong encoding is the second-most-common silent ML failure after wrong CV.
license: MIT
metadata:
  source: ml-engineer
  version: 0.1.0
---

# Encode Categoricals

## Iron Law

> **Encoders are fit on training-fold data only. Never on the full dataset. Never on validation. Never on test.**

The only exception is the safe trick when test data is available **and** there is no live serving: concatenate train+test, fit `LabelEncoder` once, then split. This is acceptable for one-shot Kaggle-style workflows. It is **not** acceptable when the model will see new categories in production.

## Decision rule

Pick the encoding by the **model family**, not by habit:

```
MODEL FAMILY                          ENCODING
─────────────────────────────────────────────────────────────────────
Tree-based                            Label encoding
  (Random Forest, ExtraTrees,         (or target encoding for high
   XGBoost, LightGBM, CatBoost)        cardinality, with smoothing)

Linear / SVM / Naive Bayes            One-hot (sparse for high cardinality)
  (LogReg, Ridge, Lasso, SVC, SVR)    + numeric feature scaling

Neural networks                       Entity embeddings if cardinality > 50
                                      otherwise one-hot

K-NN                                  One-hot + scaling (KNN cares about
                                      distance; label-encoded ordinal is wrong)
```

Inside each family:

- **Cardinality < ~50** → use the family default
- **Cardinality 50-1000** → consider target encoding for trees, sparse one-hot for linear
- **Cardinality > 1000** → entity embeddings or target encoding (with smoothing); plain one-hot is wasteful

## Hard rules — applied before encoding

These four steps run **before** any encoder touches the data:

### 1. Always `fillna` first

`LabelEncoder` raises on NaN. `OneHotEncoder` (default) sometimes silently produces a row of zeros. The only safe approach:

```python
df[col] = df[col].astype(str).fillna("NONE")
```

Cast to string then fill. NaN becomes the literal string `"NONE"` and is treated as its own category. This is almost always the right call — see Thakur, AAAMLP categorical chapter.

### 2. Apply the rare-category mechanic

Categories that appear fewer than `max(10, 0.01 * N)` times in training fold are unstable. Replace them with the literal string `"RARE"`:

```python
counts = df_train[col].value_counts()
threshold = max(10, int(0.01 * len(df_train)))
rare_categories = counts[counts < threshold].index
df_train.loc[df_train[col].isin(rare_categories), col] = "RARE"
df_valid.loc[df_valid[col].isin(rare_categories), col] = "RARE"
# At inference: same threshold, but also map any unseen category to "RARE"
df_test.loc[~df_test[col].isin(df_train[col].unique()), col] = "RARE"
```

This makes the model robust to new categories at test time. Without it, `LabelEncoder.transform` on an unseen category raises `ValueError`.

### 3. Confirm fold-aware fitting

Wrong:
```python
lbl = LabelEncoder()
df[col] = lbl.fit_transform(df[col])    # fit on full data — leaks
df_train, df_valid = split(df)
```

Right:
```python
df_train, df_valid = split(df)            # split first
lbl = LabelEncoder()
lbl.fit(df_train[col])                    # fit on train only
df_train[col] = lbl.transform(df_train[col])
df_valid[col] = lbl.transform(df_valid[col])  # may need rare-category fallback
```

### 4. Memory: prefer sparse for high-cardinality one-hot

```python
ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
X_train_sparse = ohe.fit_transform(df_train[cat_cols])
X_valid_sparse = ohe.transform(df_valid[cat_cols])
```

For 100k rows × 1000 categories, dense one-hot is ~800MB; sparse is ~30MB. Tree models can't consume sparse directly (XGBoost can, RF can't); use sparse only with linear models or XGBoost/LightGBM.

## Encoder templates

These are templates to drop into `train.py` (Layout B) inside the fold loop, **after** train/valid split. Adapt to your data.

### Label encoding (tree models)

```python
from sklearn.preprocessing import LabelEncoder

# After df_train, df_valid are split
cat_cols = [c for c in feature_cols if df[c].dtype == "object"]

for col in cat_cols:
    df_train[col] = df_train[col].astype(str).fillna("NONE")
    df_valid[col] = df_valid[col].astype(str).fillna("NONE")
    # Rare-category mechanic
    counts = df_train[col].value_counts()
    threshold = max(10, int(0.01 * len(df_train)))
    rare = counts[counts < threshold].index
    df_train.loc[df_train[col].isin(rare), col] = "RARE"
    df_valid.loc[df_valid[col].isin(rare), col] = "RARE"
    # Map unseen valid categories to RARE
    seen = set(df_train[col].unique())
    df_valid.loc[~df_valid[col].isin(seen), col] = "RARE"
    # Fit only on training fold
    lbl = LabelEncoder()
    df_train[col] = lbl.fit_transform(df_train[col])
    df_valid[col] = lbl.transform(df_valid[col])
```

### One-hot encoding (linear / NN)

```python
from sklearn.preprocessing import OneHotEncoder

cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
num_cols = [c for c in feature_cols if c not in cat_cols]

# Fillna and rare-category as above

ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
X_train_cat = ohe.fit_transform(df_train[cat_cols])
X_valid_cat = ohe.transform(df_valid[cat_cols])

# Combine with numeric
from scipy.sparse import hstack
X_train = hstack([X_train_cat, df_train[num_cols].values])
X_valid = hstack([X_valid_cat, df_valid[num_cols].values])
```

### Target (mean) encoding — fold-aware, with smoothing

This is where leakage usually happens. Read the code carefully:

```python
def target_encode(df_train, df_valid, col, target, smoothing=10):
    """
    Mean target encoding fit on training fold only, with smoothing.
    Returns (encoded_train, encoded_valid).
    """
    global_mean = df_train[target].mean()
    agg = df_train.groupby(col)[target].agg(["mean", "count"])
    # Smoothed mean: pulls categories with few samples toward the global mean
    smoothed = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
    encoded_train = df_train[col].map(smoothed).fillna(global_mean)
    encoded_valid = df_valid[col].map(smoothed).fillna(global_mean)
    return encoded_train, encoded_valid

# Inside the fold loop:
for col in cat_cols:
    df_train[f"{col}_te"], df_valid[f"{col}_te"] = target_encode(
        df_train, df_valid, col, target=config.TARGET, smoothing=10
    )
```

**Critical:** the encoding map (`smoothed`) is built from `df_train` only. `df_valid` is mapped through it. If you write `df.groupby(col)[target].mean()` outside the fold loop, you have leakage — `ml-engineer-verify` will mark it `failed`.

### Entity embeddings (very high cardinality + neural net)

For this case, generate the model in Keras/PyTorch with an `Embedding` layer per categorical column, embedding dim = `min(50, ceil(n_unique / 2))`. Out of scope for a single-file template; consult Thakur's entity embeddings recipe (AAAMLP categoricals chapter) and request a deeper implementation if needed.

### Combination features

Create new categorical features by concatenating two existing ones with a separator:

```python
import itertools

for c1, c2 in itertools.combinations(cat_cols, 2):
    df[f"{c1}__{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)
```

This often improves tree models. Watch the cardinality explode — apply rare-category to the combination columns too.

## Process

### Step 1 — Identify categorical columns

```python
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
# Also flag low-cardinality integer columns that are really categorical
for col in df.select_dtypes(include=["int64", "int32"]).columns:
    if df[col].nunique() < 20:
        print(f"WARNING: {col} has {df[col].nunique()} unique values — consider categorical")
```

### Step 2 — Pick encoder by model family

State the choice in the plan:

```
## Categorical encoding
- **Categorical columns:** [col1, col2, ...]
- **Cardinalities:** col1=12, col2=347, col3=8500
- **Model family:** tree / linear / NN
- **Strategy:** label encoding (tree) | sparse one-hot (linear) | embeddings (NN)
- **Rare-category threshold:** 10 occurrences (or 1% of N)
- **Target encoding columns:** [col3] with smoothing=10 (high cardinality only)
```

### Step 3 — Apply inside the fold loop

The encoder code goes inside `train.py` (Layout B) **after** the fold split. Never preprocess full data and then split.

### Step 4 — Verify

`ml-engineer-verify` will scan for:
- Encoder fit on full dataset (auto-Critical)
- Target encoding outside the fold loop (auto-Critical)
- Unfilled NaN in categorical columns at training time (Critical)
- One-hot dimensionality blowing up beyond available memory (warning)

## Anti-patterns

- **`pd.get_dummies(df)` on the whole dataset** before splitting. Convenient but leaks if any column has different categories across splits.
- **Target encoding without smoothing.** Categories with one sample get an encoding equal to that sample's target — pure overfitting.
- **Target encoding outside CV.** "I computed the means once, then used them everywhere" → leakage.
- **Skipping `astype(str).fillna("NONE")`.** `LabelEncoder` will crash at validation time when a NaN sneaks through.
- **Mixing label-encoded categoricals into a linear model.** Logistic regression treats label-encoded categories as ordinal. `RARE > NONE > A > B` is meaningless.
- **One-hot encoding 10k-cardinality column densely.** OOM, slow, and hurts most models.

## Output checklist

- [ ] Categorical columns identified, cardinalities printed
- [ ] Encoding choice matches model family, written into the plan
- [ ] `astype(str).fillna("NONE")` applied before encoding
- [ ] Rare-category mechanic applied with threshold stated
- [ ] Encoder `fit` called on training fold only (not full data, not validation)
- [ ] Target encoding (if used) is per-fold with smoothing
- [ ] Sparse format used when one-hot cardinality is high
- [ ] Code lives inside `train.py` fold loop, not in a separate full-data preprocessing script
