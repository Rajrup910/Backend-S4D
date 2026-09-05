"""Statistical rigor for the winning ensemble vs. the single-best baseline (ConvNeXt-Tiny).

Bootstrap CIs quantify how much the headline macro-F1 could plausibly swing under
resampling; McNemar's test asks whether the ensemble's errors differ from the baseline's
on the *same* held-out images, which is the right paired test for two classifiers scored
on one test set.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from ml.evaluation.metrics import compute_metrics


def bootstrap_macro_f1_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[i] = compute_metrics(y_true[idx], y_pred[idx])["macro_f1"]
    return {
        "mean": float(scores.mean()),
        "ci_low": float(np.percentile(scores, 2.5)),
        "ci_high": float(np.percentile(scores, 97.5)),
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """Paired McNemar's test on correct/incorrect disagreement between two classifiers.

    Uses the chi-square form with continuity correction; exact binomial would be more
    conservative but chi-square is standard and adequate for the ~1500-image test split.
    """
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    only_a = int(np.sum(a_correct & ~b_correct))
    only_b = int(np.sum(~a_correct & b_correct))

    discordant = only_a + only_b
    if discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "only_a_correct": only_a, "only_b_correct": only_b}

    statistic = (abs(only_a - only_b) - 1) ** 2 / discordant
    p_value = float(chi2.sf(statistic, df=1))
    return {"statistic": float(statistic), "p_value": p_value, "only_a_correct": only_a, "only_b_correct": only_b}
