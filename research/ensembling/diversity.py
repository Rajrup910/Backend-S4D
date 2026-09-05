"""Pairwise diversity metrics: do the architectures make *different* mistakes?

An ensemble only helps if its members are individually decent and jointly de-correlated.
These metrics quantify that de-correlation across the six frozen backbones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.ensembling.data import PredictionMatrix


def _pairwise_matrix(archs: tuple[str, ...], pair_fn) -> pd.DataFrame:
    k = len(archs)
    matrix = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            value = pair_fn(i, j)
            matrix[i, j] = matrix[j, i] = value
    return pd.DataFrame(matrix, index=archs, columns=archs)


def pairwise_disagreement(matrix: PredictionMatrix) -> pd.DataFrame:
    """Fraction of samples where two models predict a different class."""
    preds = matrix.probs.argmax(axis=2)  # (N, K)

    def pair_fn(i: int, j: int) -> float:
        return float(np.mean(preds[:, i] != preds[:, j]))

    return _pairwise_matrix(matrix.archs, pair_fn)


def yules_q(matrix: PredictionMatrix) -> pd.DataFrame:
    """Yule's Q-statistic: correlation of correctness between two classifiers.

    +1 => the two models are always right/wrong together (redundant).
    -1 => one is right exactly when the other is wrong (maximally complementary).
     0 => independent error patterns.
    """
    preds = matrix.probs.argmax(axis=2)  # (N, K)
    correct = preds == matrix.y_true[:, None]  # (N, K)

    def pair_fn(i: int, j: int) -> float:
        both_right = np.sum(correct[:, i] & correct[:, j])
        both_wrong = np.sum(~correct[:, i] & ~correct[:, j])
        i_only = np.sum(correct[:, i] & ~correct[:, j])
        j_only = np.sum(~correct[:, i] & correct[:, j])
        numerator = both_right * both_wrong - i_only * j_only
        denominator = both_right * both_wrong + i_only * j_only
        return float(numerator / denominator) if denominator else 0.0

    return _pairwise_matrix(matrix.archs, pair_fn)


def double_fault_ratio(matrix: PredictionMatrix) -> pd.DataFrame:
    """Fraction of samples where BOTH models are wrong -- lower is more complementary."""
    preds = matrix.probs.argmax(axis=2)
    correct = preds == matrix.y_true[:, None]

    def pair_fn(i: int, j: int) -> float:
        return float(np.mean(~correct[:, i] & ~correct[:, j]))

    return _pairwise_matrix(matrix.archs, pair_fn)


def diversity_report(matrix: PredictionMatrix) -> dict[str, object]:
    disagreement = pairwise_disagreement(matrix)
    q_stat = yules_q(matrix)
    double_fault = double_fault_ratio(matrix)

    k = len(matrix.archs)
    off_diag = ~np.eye(k, dtype=bool)
    return {
        "disagreement": disagreement,
        "yules_q": q_stat,
        "double_fault": double_fault,
        "mean_disagreement": float(disagreement.to_numpy()[off_diag].mean()),
        "mean_yules_q": float(q_stat.to_numpy()[off_diag].mean()),
        "mean_double_fault": float(double_fault.to_numpy()[off_diag].mean()),
    }
