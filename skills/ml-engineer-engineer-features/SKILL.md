---
name: ml-engineer-engineer-features
description: Use after EDA when raw features look weak or limited, when the task involves dates / time-series / list-valued / high-variance numeric data, or when the user asks to "create features", "engineer features", "improve features". Generates date features, aggregation features, polynomial features, binning, log transforms, and missing-value imputation — all fit per-fold to avoid leakage.
license: MIT
metadata:
  source: ml-engineer
  version: 0.1.0
---

# Engineer Features

## Iron Law

> **Any feature that uses the target, or aggregates across rows, must be fit on training-fold data only.**

Features built from the full dataset (including validation) are leakage. Examples that leak: groupby-target means, KNN imputation fit on full data, polynomial features that include the target.

## When to invoke

- The plan or `ml-engineer-decide` flagged "feature engineering" as a step
- Baseline model on raw features is mediocre and there are obvious unexploited signals (date columns, transaction lists, high-cardinality categories that haven't been combined, heavy-tailed numerics)
- The user explicitly asks for new features

## When NOT to invoke

- Baseline isn't trained yet — engineer features after you know what's missing
- Tree models on already-good tabular data — feature engineering helps less than for linear models
- Deep learning workflows that learn representations end-to-end (the network does the FE)

## Feature recipes

Each recipe is a self-contained transformation. Mix and match — most projects use 2-4.

### 1. Date features

For any datetime column, extract:

```python
df["year"]       = df[date_col].dt.year
df["month"]      = df[date_col].dt.month
df["day"]        = df[date_col].dt.day
df["dayofweek"]  = df[date_col].dt.dayofweek
df["dayofyear"]  = df[date_col].dt.dayofyear
df["weekofyear"] = df[date_col].dt.isocalendar().week.astype(int)
df["quarter"]    = df[date_col].dt.quarter
df["hour"]       = df[date_col].dt.hour              # if time component exists
df["weekend"]    = (df[date_col].dt.dayofweek >= 5).astype(int)
df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
df["is_month_end"]   = df[date_col].dt.is_month_end.astype(int)
```

For cyclic features (hour, dayofweek, month), encode as sin/cos pairs so the model knows 23:00 is close to 00:00:

```python
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
```

**Leakage concern:** none — date features are deterministic per row.

### 2. Aggregation features (groupby on entity)

When data has multiple rows per entity (`customer_id`, `user_id`, `device_id`), aggregate within each group:

```python
def make_aggregates(df, group_col, num_cols):
    """Per-group aggregations. Fit on training fold only when target is involved."""
    aggs = {}
    for col in num_cols:
        aggs[col] = ["mean", "std", "min", "max", "sum", "nunique"]
    agg_df = df.groupby(group_col).agg(aggs)
    agg_df.columns = [f"{c}_{stat}" for c, stat in agg_df.columns]
    agg_df = agg_df.reset_index()
    return df.merge(agg_df, on=group_col, how="left")
```

**Leakage concern:** if you aggregate over both train and test rows, validation rows of an entity see training rows of the same entity. Two safe approaches:
- Aggregate over **train only**, merge onto train + valid via the group key (validation rows of unseen entities get NaN — fillna with global mean)
- Use only past-data aggregates if there's a time dimension (groupby + cumulative + shift)

### 3. Time-series list features

When a row has a list/array of values (transactions, sensor readings, click sequence), extract statistical summaries:

```python
def list_features(values: np.ndarray) -> dict:
    return {
        "mean":   np.mean(values),
        "std":    np.std(values),
        "min":    np.min(values),
        "max":    np.max(values),
        "median": np.median(values),
        "ptp":    np.ptp(values),
        "var":    np.var(values),
        "skew":   pd.Series(values).skew(),
        "kurt":   pd.Series(values).kurtosis(),
        "p10":    np.percentile(values, 10),
        "p90":    np.percentile(values, 90),
        "q05":    np.quantile(values, 0.05),
        "q95":    np.quantile(values, 0.95),
        "abs_energy": np.sum(values ** 2),
    }
```

For more, use the `tsfresh` library — hundreds of pre-built time-series features. Install on demand in the venv.

**Leakage concern:** none if the list is per-row data. Concern arises only if the list spans the train/test boundary — then truncate to past-only.

### 4. Polynomial features

For numeric columns, create degree-2 interactions:

```python
from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = pf.fit_transform(df_train[num_cols])
# Keep names if useful:
poly_cols = pf.get_feature_names_out(num_cols)
df_train_poly = pd.DataFrame(X_poly, columns=poly_cols, index=df_train.index)
```

For categorical-categorical interactions (string concat, see `ml-engineer-encode-categoricals`):

```python
import itertools
for c1, c2 in itertools.combinations(cat_cols, 2):
    df[f"{c1}__{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)
```

**Leakage concern:** `PolynomialFeatures.fit_transform` on full data is fine (no target involved, deterministic). Just don't include the target column.

### 5. Binning numeric features

Convert numeric → categorical (sometimes helps tree models capture non-linear cuts the tree wouldn't find):

```python
df["age_bin_10"]  = pd.cut(df["age"], bins=10, labels=False)
df["age_bin_100"] = pd.cut(df["age"], bins=100, labels=False)
```

Keep both the binned and original column — the model picks. Apply the same bin edges to validation by saving them:

```python
df_train["age_bin"], bin_edges = pd.cut(df_train["age"], bins=10, labels=False, retbins=True)
df_valid["age_bin"] = pd.cut(df_valid["age"], bins=bin_edges, labels=False).astype(float)
```

**Leakage concern:** if you `pd.qcut` the full data, validation quantile assignments depend on validation values. Use `pd.cut` with bin_edges from train only, or `pd.qcut` on train then apply the same edges.

### 6. Log / power transforms

For heavy-tailed positive features (counts, prices, durations):

```python
df["amount_log"] = np.log1p(df["amount"])
df["amount_sqrt"] = np.sqrt(df["amount"])
```

For metrics like RMSLE, this is required (already covered in `ml-engineer-pick-metric`).

**Leakage concern:** none — element-wise transforms.

### 7. Missing-value imputation

Strategy depends on column type and downstream model:

| Strategy | When |
|---|---|
| Treat NaN as a category | Categorical features (always start here) |
| Fill with 0 | Numeric, when 0 is a meaningful "absence" |
| Fill with mean / median | Numeric, no special pattern |
| KNN imputation (`sklearn.impute.KNNImputer`) | Numeric, when missing is informative |
| Iterative regression imputation | Numeric, multivariate dependencies |
| Leave NaN | Tree models (XGBoost, LightGBM, CatBoost handle NaN natively) |

**Leakage concern:** every imputer except "treat as category" and "leave NaN" is fit on data — must fit on training fold only.

```python
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="mean")
df_train[num_cols] = imp.fit_transform(df_train[num_cols])  # fit on train
df_valid[num_cols] = imp.transform(df_valid[num_cols])      # transform valid
```

## Hard rules

- **Imputers, scalers, polynomial transforms that involve aggregation → fit on training fold only.** Never `imputer.fit(full_df)` then split.
- **Aggregation features over groups → if target is involved, fit per fold.** Otherwise OK on train+valid (e.g. count of records per customer is fine to compute over all rows, since it doesn't use the target).
- **Save fit objects (imputers, encoders, scalers, bin_edges) per fold** so inference can reproduce them.
- **Don't drop the original column when adding a derived feature.** Let the model decide which to use. Drop only if the derivative strictly dominates (e.g., `log(amount)` replacing `amount` when the metric is RMSLE).
- **Don't generate 1000 features without selection.** Use `ml-engineer-verify` and feature importance to prune. Curse of dimensionality is real for linear and KNN models.

## Process

### Step 1 — Identify candidates

Inspect the data:
- Date columns? → recipe 1
- Group columns with multiple rows per entity? → recipe 2
- List-valued columns (variable-length per row)? → recipe 3
- Heavy-tailed numerics (skew > 2)? → recipe 6
- High cardinality categoricals? → polynomial / combination via `ml-engineer-encode-categoricals`
- Missing values? → recipe 7

### Step 2 — Pick 2-4 recipes

Don't apply all of them. Each new feature increases the risk of overfit and the cost of debugging.

State the choice in the plan:

```
## Feature engineering
- **Date features:** year, month, dayofweek, weekend, hour_sin, hour_cos from <date_col>
- **Aggregations:** mean/std/sum/nunique of <amount, duration> grouped by <customer_id>, fit on train
- **Log transform:** np.log1p applied to amount, balance (skew=4.2, 3.7)
- **Imputation:** SimpleImputer(strategy='median') for <num_cols>, fit per fold
- **Skipped:** polynomial features (already 50+ raw features), entity embeddings (cardinality < 50)
```

### Step 3 — Implement inside the fold loop

Feature engineering that requires fitting goes inside the fold loop in `train.py`. Stateless transforms (date features, log) can go in a separate `prepare_data.py` that runs once.

```python
# Inside train.py fold loop
df_train, df_valid = ...split...

# Stateless transforms (already applied earlier, just illustrating)
# df_train["amount_log"] = np.log1p(df_train["amount"])

# Stateful: imputation, aggregations using target
imp = SimpleImputer(strategy="median")
df_train[num_cols] = imp.fit_transform(df_train[num_cols])
df_valid[num_cols] = imp.transform(df_valid[num_cols])

# Save imputer for inference
joblib.dump(imp, f"../models/imputer_fold{fold}.bin")
```

### Step 4 — Verify the new features helped

Re-run training with new features. Compare OOF metric to baseline:
- Improved by ≥0.5pp → keep
- Improved by < 0.5pp → keep but acknowledge the marginal benefit
- No improvement or regression → drop the features, document the negative result

### Step 5 — Verify no leakage

`ml-engineer-verify` scans for:
- Any imputer / scaler / target-aggregation that touched validation data
- Polynomial features that included the target column
- Bin edges computed on full data instead of train-only

## Anti-patterns

- **Generating polynomial features on top of polynomial features.** Combinatorial blow-up.
- **Aggregating over the full dataset when target is in the aggregation.** That's target encoding, do it per-fold.
- **Adding 200 tsfresh features without selection.** Train a baseline first, prune with `ml-engineer-verify` + feature importance, then keep top-K.
- **Fitting `KNNImputer` on the full dataset.** Validation values inform imputation of training rows. Subtle leakage.
- **`pd.qcut` on the full dataset.** Same as above — quantiles depend on validation values.
- **Dropping the original column.** Let the model choose.

## Output checklist

- [ ] Inspected data, identified candidate recipes
- [ ] Picked 2-4 recipes (not all of them)
- [ ] Documented choice in the plan with rationale
- [ ] Stateful transforms inside fold loop, fit on train only
- [ ] Stateless transforms applied once, before training
- [ ] Imputers / scalers / encoders saved per fold for inference reuse
- [ ] Feature impact measured: OOF metric before vs after
- [ ] Negative results documented (don't silently drop "didn't help")
