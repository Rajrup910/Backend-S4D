"""Leak-free inner cross-validation for fitted ensemble methods.

The 6 backbones are frozen, so their `val`-split predictions are already out-of-fold with
respect to *base-model* training. The remaining overfitting risk is the *meta-learner*
(Nelder-Mead weights, the stacking regressor, Caruana selection) fitting and being scored
on the exact same validation rows -- that inflates the val score used to pick a winning
method. This module runs 5-fold stratified, lesion-grouped CV *within the val split* to
produce an honest out-of-fold score for method selection; the final model applied to test
is then refit once on the full val split (never touched during folding).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

FitFn = Callable[[np.ndarray, np.ndarray], object]  # (train_probs_or_logits, train_y) -> fitted_state
PredictFn = Callable[[object, np.ndarray], np.ndarray]  # (fitted_state, holdout_features) -> (N, C) scores


def oof_scores(
    features: np.ndarray,
    y_true: np.ndarray,
    lesion_ids: np.ndarray,
    fit_fn: FitFn,
    predict_fn: PredictFn,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Return out-of-fold (N, C) scores: each row predicted by a model that never saw it."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    num_classes = int(y_true.max()) + 1
    oof = np.zeros((len(y_true), num_classes))
    filled = np.zeros(len(y_true), dtype=bool)

    for train_idx, holdout_idx in splitter.split(features, y_true, groups=lesion_ids):
        state = fit_fn(features[train_idx], y_true[train_idx])
        oof[holdout_idx] = predict_fn(state, features[holdout_idx])
        filled[holdout_idx] = True

    assert filled.all(), "StratifiedGroupKFold left some samples unassigned to a fold"
    return oof
