---
name: dl-ensemble-tta
description: Use to combine multiple models or augmented test predictions — k-fold OOF blending, rank-average, weighted average, snapshot ensembles, test-time augmentation (TTA). Cross-domain (CV + NLP + tabular) — internal sub-modes per domain. Always sanity-check ensemble vs best single model. Do NOT use during training (this is inference-time), with a single model (no ensemble), or as a substitute for fixing a broken baseline.
license: MIT
metadata:
  source: ml-engineer
  version: 0.2.0
---

# Ensemble + TTA

Combine multiple model predictions at inference time. Pick simple-average / rank-average / weighted / stacking / TTA based on number of models, prediction calibration, and domain. Cross-domain: CV uses inverse-augmentation TTA; NLP uses cross-fold blend; tabular uses OOF rank-average.

## When to invoke

- 2+ trained models with predictions on the same eval set.
- CV: a single model + augmented test images for TTA.
- After Phase 2 training; before declaring a multi-model task complete.

## When NOT to invoke

- Single model only — no ensemble possible.
- During training (this is inference-time only).
- As a substitute for fixing a broken baseline. Ensemble of broken models = still broken.

## Decision rules

### Method (by model count + calibration)

- **Simple average (mean of probs)**: 2+ well-calibrated models. Default starting point.
- **Rank average**: when model confidences are on different scales (one model outputs probs in [0.1, 0.9], another in [0.45, 0.55]). Convert each model's predictions to ranks per-example, then average ranks.
- **Weighted average**: when models differ meaningfully in quality. Weight by validation metric. Avoid over-fitting weights — use OOF metric.
- **Stacking**: 5+ models with diverse error patterns. Train a meta-model on OOF predictions. Risk: meta-model overfits if folds were not properly held out during base training.
- **TTA (test-time augmentation)**: single CV model + N augmented versions of the test image. Average the N predictions. Always boosts CV score by 0.5-2%.

### Domain mode

- **CV mode**: TTA via inverse augmentation (horizontal flip, multi-scale, multi-crop) + OOF blend across folds.
- **NLP mode**: cross-fold blend (no TTA — text augmentation is rarely safe at inference).
- **Tabular mode**: OOF rank-average across fold-models + diverse base models (LightGBM + XGBoost + CatBoost).

## Process

### Step 1 — Detect domain + load predictions

Read `<workdir>/data_policy.json` for `task_type`. Load per-model predictions from `<workdir>/predictions/`.

### Step 2 — Pick method per decision rules

Surface choice + rationale.

### Step 3 — Apply ensemble

For TTA: run the model on N augmented versions of the test set; invert the augmentation on the predictions where applicable (e.g., flip back the predicted boxes for horizontal-flip TTA on detection).

For multi-model blend: average / rank-average / weighted average the per-example predictions.

For stacking: train a meta-model (LogisticRegression / lightgbm) on OOF predictions; predict on test.

### Step 4 — Sanity check

Compare ensemble metric to:
- Best single model's metric.
- Average single model's metric.

Ensemble should beat the BEST single model. If not, the ensemble is hurting (often: one model dominates and others add noise) — drop weakest models.

### Step 5 — Save ensembled predictions

`<workdir>/predictions/ensemble.csv` (or coco json for detection, or per-pixel masks for segmentation).

### Step 6 — Surface insights

Print headline metric + per-model contributions (correlation between models, leave-one-out drop in ensemble metric).

## Recipe template

### `<workdir>/src/_ensemble.py`

```python
"""Cross-domain ensembling. Pick method by decision rules."""
import json
import os
from pathlib import Path

import numpy as np

WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve()


def simple_average(preds_list: list) -> np.ndarray:
    """preds_list: list of (N, C) probability arrays."""
    return np.mean(np.stack(preds_list, axis=0), axis=0)


def rank_average(preds_list: list) -> np.ndarray:
    """Average per-class ranks across examples — robust to scale differences across models.

    Per the ensembling convention: for each class column, rank all N example scores;
    each model contributes a (N, C) matrix of per-class ranks; we average across models.

    Use this when models output probabilities on different scales (one model says 0.1-0.9,
    another says 0.45-0.55). Ranks normalize to a common ordinal scale.
    """
    from scipy.stats import rankdata
    ranks = []
    for p in preds_list:
        # rank within each column (each class), across all examples
        rank_p = np.array([rankdata(p[:, c]) for c in range(p.shape[1])]).T
        ranks.append(rank_p)
    return np.mean(np.stack(ranks, axis=0), axis=0)


def weighted_average(preds_list: list, weights: list[float]) -> np.ndarray:
    """weights: per-model contribution; usually inverse-error or OOF metric."""
    weights = np.array(weights) / sum(weights)
    return np.sum(np.stack(preds_list, axis=0) * weights[:, None, None], axis=0)


def stacking_meta_model(oof_preds: list, oof_labels: np.ndarray,
                         test_preds: list, model_class: str = "lr") -> np.ndarray:
    """Train a meta-model on OOF preds; predict on test. model_class in {'lr', 'lgb'}."""
    from sklearn.linear_model import LogisticRegression
    X_train = np.concatenate(oof_preds, axis=-1)  # stack along feature dim
    X_test = np.concatenate(test_preds, axis=-1)

    if model_class == "lr":
        meta = LogisticRegression(C=1.0, max_iter=1000)
    elif model_class == "lgb":
        import lightgbm as lgb
        meta = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05)
    else:
        raise ValueError(model_class)

    meta.fit(X_train, oof_labels)
    return meta.predict_proba(X_test)


def cv_tta_classify(model, image, tta_transforms: list, original_predict_fn) -> np.ndarray:
    """TTA for image classification: predict on N augmented versions; average.

    tta_transforms: list of dicts each with a "forward" callable that produces an augmented
    version of the image. For classification, predictions don't need to be inverted (only
    the image was changed). For detection/segmentation, you'd invert the predicted boxes/masks
    after applying the inverse spatial transform; that's not handled by this helper.
    """
    preds = [original_predict_fn(model, image)]  # original
    for tf in tta_transforms:
        augmented = tf["forward"](image)
        pred = original_predict_fn(model, augmented)
        preds.append(pred)
    return np.mean(np.stack(preds, axis=0), axis=0)


def sanity_check_ensemble(ensemble_metric: float, single_metrics: list[float]) -> dict:
    """Check ensemble actually helps. Return verdict + reasoning."""
    best_single = max(single_metrics)
    avg_single = np.mean(single_metrics)
    return {
        "ensemble_metric": ensemble_metric,
        "best_single_metric": best_single,
        "avg_single_metric": avg_single,
        "ensemble_beats_best": ensemble_metric > best_single,
        "verdict": "OK — ensemble helps" if ensemble_metric > best_single else "WARNING — ensemble degrades vs best single; drop weak models",
    }


def save_ensemble(predictions: np.ndarray, method: str, n_models: int):
    """Save ensembled predictions + metadata."""
    out_dir = WORKDIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "ensemble.npy", predictions)
    metadata = {"method": method, "n_models": n_models}
    (out_dir / "ensemble_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Ensemble predictions saved to {out_dir}/ensemble.npy")
```

## Hard constraints

- NEVER blend predictions before applying the same calibration. If model A outputs softmax probs and model B outputs raw logits, blending is meaningless. Calibrate first.
- NEVER stack on the same folds used for base-model training. The meta-model will overfit. Use proper OOF predictions: model-K's OOF is from the model that didn't see fold-K.
- ALWAYS sanity-check ensemble vs BEST single model. Ensembling often hurts when one model dominates and others add noise.
- NEVER blend models trained on different label spaces (different num_classes, different class IDs). Output shapes must match.
- NEVER use TTA augmentations that change the prediction's meaning. Horizontal-flip OK for natural-image classification; NOT OK for OCR or text-in-image tasks (flipping text breaks it).
- NEVER apply TTA at training-time. Augmentation at training is `dl-augment`'s job; TTA is inference-only.
- NEVER report ensemble metric without recording how many models contributed. "0.92 mAP from ensemble" is meaningless without knowing it's a 5-model ensemble vs single.

## Research hooks

- **Current ensemble methods on Kaggle.** Invoke `dl-prior-art` for the user's specific competition; query as fallback: *"Current top ensembling methods on Kaggle as of {today} (rank-avg / power-avg / stacking / blend)."*
- **TTA recipes for `{task_type}`.** Query: *"Standard TTA augmentations for `{task_type}` (image classify / detect / segment) as of {today} that boost vs hurt the score."*
- **Snapshot ensembling status.** Query: *"Is snapshot ensembling (cyclical lr + saved checkpoints) still competitive vs proper k-fold ensemble as of {today}?"*

## Verification gates

After this skill runs, `ml-engineer-verify` MUST check:

- Method (simple / rank / weighted / stacking / TTA) recorded in `<workdir>/predictions/ensemble_metadata.json`.
- Ensemble metric > best single model metric (sanity check). If not, user was alerted.
- For stacking: meta-model trained on OOF predictions (not on the same folds as base training).
- For TTA: augmentations used are documented; no augmentation that changes the prediction meaning was used (no horizontal-flip on OCR, etc.).
- For weighted average: weights derived from OOF metrics, not from test metrics.

## Output checklist

- [ ] Domain mode (CV / NLP / tabular) detected
- [ ] Ensemble method picked per decision rules
- [ ] All base predictions same shape + same label space
- [ ] Ensemble computed
- [ ] Sanity check vs best single model passed (or user alerted if failed)
- [ ] `<workdir>/predictions/ensemble.{npy,json,csv}` saved
- [ ] Metadata (method, n_models) saved
