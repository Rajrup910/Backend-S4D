"""DeLong's test for paired ROC-AUC comparison (Sun & Xu 2014 fast implementation, via the
structural components of DeLong et al. 1988), applied one-vs-rest per class.

Multi-class macro-ROC-AUC has no closed-form DeLong covariance, so this module tests the
thing DeLong's test is actually defined for: per-class one-vs-rest AUC, on the same paired
test cases, between two classifiers. The macro-AUC comparison in the ablation report instead
uses the paired bootstrap distribution already computed for the CI (see stats.py) -- that is
honest about what DeLong does and does not cover, rather than approximating a multi-class
extension nobody has agreed on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks, needed because DeLong's structural components assume no ties."""
    order = np.argsort(x, kind="mergesort")
    n = len(x)
    sorted_x = x[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    result = np.empty(n, dtype=float)
    result[order] = ranks
    return result


def _fast_delong_structural_components(predictions: np.ndarray, positive: np.ndarray):
    """predictions: (K, N) scores for K classifiers on N cases; positive: (N,) bool label."""
    pos = predictions[:, positive]
    neg = predictions[:, ~positive]
    m, n = pos.shape[1], neg.shape[1]
    k = predictions.shape[0]

    # The Sun & Xu recurrences index tz as [positives | negatives], so the pooled midranks
    # must be taken over that reordering -- not over the original case order.
    pooled = np.concatenate([pos, neg], axis=1)

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(pooled[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    return aucs, v01, v10, m, n


@dataclass(frozen=True)
class DeLongResult:
    class_code: str
    auc_a: float
    auc_b: float
    z: float
    p_value: float
    n_positive: int
    n_negative: int


def delong_paired_auc_test(
    y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray, positive_class: int
) -> DeLongResult | None:
    """Paired DeLong test between two one-vs-rest score vectors for one class.

    Returns None if the class has fewer than 2 positives or 2 negatives on this split
    (the covariance estimate is undefined -- exactly the `df`/`vasc` situation).
    """
    positive = y_true == positive_class
    if positive.sum() < 2 or (~positive).sum() < 2:
        return None

    predictions = np.vstack([prob_a, prob_b])
    aucs, v01, v10, m, n = _fast_delong_structural_components(predictions, positive)

    sx = np.cov(v01, ddof=1)
    sy = np.cov(v10, ddof=1)
    cov = sx / m + sy / n  # 2x2 covariance of [auc_a, auc_b]

    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var_diff <= 0:
        return None
    z = (aucs[0] - aucs[1]) / np.sqrt(var_diff)
    p = float(2 * norm.sf(abs(z)))
    return DeLongResult(
        class_code="",
        auc_a=float(aucs[0]),
        auc_b=float(aucs[1]),
        z=float(z),
        p_value=p,
        n_positive=int(positive.sum()),
        n_negative=int((~positive).sum()),
    )


def delong_per_class(
    y_true: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray, class_codes: list[str]
) -> list[DeLongResult]:
    """Run the paired one-vs-rest DeLong test for every class; skips undefined classes."""
    results = []
    for index, code in enumerate(class_codes):
        result = delong_paired_auc_test(y_true, probs_a[:, index], probs_b[:, index], index)
        if result is not None:
            results.append(
                DeLongResult(
                    class_code=code,
                    auc_a=result.auc_a,
                    auc_b=result.auc_b,
                    z=result.z,
                    p_value=result.p_value,
                    n_positive=result.n_positive,
                    n_negative=result.n_negative,
                )
            )
    return results
