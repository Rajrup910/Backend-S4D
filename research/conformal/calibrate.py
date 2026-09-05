"""Conformal quantiles and prediction-set construction, marginal and Mondrian.

The whole method is one number per calibration set: the score threshold below which
`1 - alpha` of calibration points fall, with a finite-sample correction. Everything else
is bookkeeping.

**Marginal vs Mondrian.** A single threshold over all calibration points guarantees that
90% of *all* future cases have the truth in their set — a promise that, on a dataset that
is 67% `nv`, can be kept almost entirely by covering moles. Mondrian (class-conditional)
calibration computes a separate threshold per class, so the guarantee holds *within* each
class, including melanoma. That is the only version of the guarantee worth quoting to a
clinician, and it is strictly more demanding: the price is larger sets for the rare
classes, paid honestly rather than hidden in an average.

**When a class has too few calibration points**, the required quantile level exceeds 1 and
no finite threshold can be certified. That case returns +inf — the set then contains every
class, which is the correct statement that the data cannot support a narrower claim at
this alpha. It is reported rather than silently clipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalState:
    """Fitted thresholds plus the bookkeeping needed to report them honestly."""

    method: str
    alpha: float
    mondrian: bool
    quantiles: np.ndarray            # (C,) — all entries equal when marginal
    calibration_counts: np.ndarray   # (C,) points behind each quantile
    degenerate_classes: tuple[int, ...]  # classes whose quantile is +inf

    @property
    def is_degenerate(self) -> bool:
        return len(self.degenerate_classes) > 0


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The (1-alpha) conformal quantile with the finite-sample (n+1) correction.

    Returns +inf when `n` is too small for the level to exist, i.e. when
    ceil((n+1)(1-alpha)) > n. Rounding *up* to an observed score (`method="higher"`) is
    what makes the guarantee hold for finite n rather than only asymptotically.
    """
    n = len(scores)
    if n == 0:
        return float("inf")
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        return float("inf")
    return float(np.quantile(scores, level, method="higher"))


def fit(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    num_classes: int,
    alpha: float,
    method: str,
    mondrian: bool,
) -> ConformalState:
    """Fit one threshold (marginal) or one per class (Mondrian) on calibration data."""
    if mondrian:
        quantiles = np.empty(num_classes, dtype=np.float64)
        counts = np.zeros(num_classes, dtype=np.int64)
        for c in range(num_classes):
            in_class = calibration_scores[calibration_labels == c]
            counts[c] = len(in_class)
            quantiles[c] = conformal_quantile(in_class, alpha)
    else:
        shared = conformal_quantile(calibration_scores, alpha)
        quantiles = np.full(num_classes, shared, dtype=np.float64)
        counts = np.full(num_classes, len(calibration_scores), dtype=np.int64)

    return ConformalState(
        method=method,
        alpha=alpha,
        mondrian=mondrian,
        quantiles=quantiles,
        calibration_counts=counts,
        degenerate_classes=tuple(int(c) for c in np.where(~np.isfinite(quantiles))[0]),
    )


def prediction_sets(state: ConformalState, score_matrix: np.ndarray) -> np.ndarray:
    """(N, C) boolean membership: class c is in image i's set iff its score clears c's bar.

    Note the threshold is indexed by the *candidate* class, not by the prediction — that
    is what makes the Mondrian guarantee class-conditional, and it is why a set can hold
    a rare class whose own bar is lenient while excluding a common one.
    """
    return score_matrix <= state.quantiles[None, :]


def grouped_halves(
    lesion_ids: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into two halves that never share a lesion.

    Two images of the same lesion are near-duplicates; letting one land in the tuning
    half and the other in the calibration half would make the tuned hyperparameters look
    better than they are, the same leak the project's splits already guard against at the
    dataset level.
    """
    unique = np.unique(lesion_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    first = set(shuffled[: len(shuffled) // 2].tolist())

    mask = np.array([lesion_id in first for lesion_id in lesion_ids])
    return np.where(mask)[0], np.where(~mask)[0]
