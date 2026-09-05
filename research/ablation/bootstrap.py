"""Lesion-grouped bootstrap confidence intervals.

`research/ensembling/stats.py::bootstrap_macro_f1_ci` resamples *images* uniformly. HAM10000
has 10015 images on only 7470 lesions, and several lesions contribute 2-3 images each (the
whole reason the train/val/test split is grouped by lesion_id in the first place -- see hard
rule 1). Resampling images independently would let a bootstrap draw pick two images of the
same lesion as if they were independent evidence, understating the true sampling variance.
This module resamples *lesion IDs* with replacement and pools every image belonging to each
drawn lesion, which is the correct unit of resampling given how the data was collected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.evaluation.metrics import compute_metrics


@dataclass(frozen=True)
class BootstrapCI:
    metric: str
    point_estimate: float
    mean: float
    ci_low: float
    ci_high: float
    n_boot: int


_METRIC_GETTERS = {
    "macro_f1": lambda m: m["macro_f1"],
    "balanced_accuracy": lambda m: m["balanced_accuracy"],
    "roc_auc_macro": lambda m: m.get("roc_auc_macro"),
    "escalation_sensitivity": lambda m: m["clinical"]["binary_sensitivity"],
}


def grouped_bootstrap_ci(
    y_true: np.ndarray,
    probs: np.ndarray,
    lesion_ids: np.ndarray,
    metrics: tuple[str, ...] = ("macro_f1", "balanced_accuracy", "escalation_sensitivity"),
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, BootstrapCI]:
    """1000x stratified-by-nothing, grouped-by-lesion bootstrap over (y_true, probs).

    `probs` (not `y_pred`) is resampled so that `roc_auc_macro` can be requested too; hard
    predictions are re-derived from the resampled probabilities via argmax each draw.
    """
    rng = np.random.default_rng(seed)
    unique_lesions = np.unique(lesion_ids)
    n_lesions = len(unique_lesions)

    # index lists per lesion, built once
    lesion_to_rows: dict = {lesion: np.flatnonzero(lesion_ids == lesion) for lesion in unique_lesions}

    point = compute_metrics(y_true, probs.argmax(axis=1), probs)
    draws: dict[str, list[float]] = {m: [] for m in metrics}

    for _ in range(n_boot):
        sampled_lesions = rng.choice(unique_lesions, size=n_lesions, replace=True)
        row_idx = np.concatenate([lesion_to_rows[lesion] for lesion in sampled_lesions])
        y_b = y_true[row_idx]
        p_b = probs[row_idx]
        m_b = compute_metrics(y_b, p_b.argmax(axis=1), p_b)
        for metric in metrics:
            value = _METRIC_GETTERS[metric](m_b)
            if value is not None:
                draws[metric].append(value)

    results = {}
    for metric in metrics:
        values = np.asarray(draws[metric])
        results[metric] = BootstrapCI(
            metric=metric,
            point_estimate=float(_METRIC_GETTERS[metric](point) or float("nan")),
            mean=float(values.mean()) if len(values) else float("nan"),
            ci_low=float(np.percentile(values, 2.5)) if len(values) else float("nan"),
            ci_high=float(np.percentile(values, 97.5)) if len(values) else float("nan"),
            n_boot=len(values),
        )
    return results


def lesion_resample_indices(
    lesion_ids: np.ndarray, n_boot: int = 1000, seed: int = 42
):
    """Yield `n_boot` row-index arrays, each one a lesion-level resample with replacement.

    The two functions above inline this loop. It is exposed separately -- rather than
    refactoring them onto it -- because any change to their RNG consumption would silently
    move published intervals, and because a caller whose statistic is not a `compute_metrics`
    key (net benefit, for one: it scores a *decision*, not an argmax over probabilities)
    otherwise has to re-implement the resampling and can get the unit wrong. Resampling
    lesions rather than images is the whole point; see the module docstring.
    """
    rng = np.random.default_rng(seed)
    unique_lesions = np.unique(lesion_ids)
    lesion_to_rows: dict = {lesion: np.flatnonzero(lesion_ids == lesion) for lesion in unique_lesions}
    for _ in range(n_boot):
        sampled = rng.choice(unique_lesions, size=len(unique_lesions), replace=True)
        yield np.concatenate([lesion_to_rows[lesion] for lesion in sampled])


def grouped_bootstrap_diff_ci(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    lesion_ids: np.ndarray,
    metric: str = "macro_f1",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap CI for (metric_a - metric_b) on the same lesion resample each draw."""
    rng = np.random.default_rng(seed)
    unique_lesions = np.unique(lesion_ids)
    n_lesions = len(unique_lesions)
    lesion_to_rows: dict = {lesion: np.flatnonzero(lesion_ids == lesion) for lesion in unique_lesions}
    getter = _METRIC_GETTERS[metric]

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sampled_lesions = rng.choice(unique_lesions, size=n_lesions, replace=True)
        row_idx = np.concatenate([lesion_to_rows[lesion] for lesion in sampled_lesions])
        y_b = y_true[row_idx]
        ma = compute_metrics(y_b, probs_a[row_idx].argmax(axis=1), probs_a[row_idx])
        mb = compute_metrics(y_b, probs_b[row_idx].argmax(axis=1), probs_b[row_idx])
        diffs[i] = getter(ma) - getter(mb)

    point = getter(compute_metrics(y_true, probs_a.argmax(axis=1), probs_a)) - getter(
        compute_metrics(y_true, probs_b.argmax(axis=1), probs_b)
    )
    return {
        "metric": metric,
        "point_estimate": float(point),
        "mean": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
        "n_boot": n_boot,
    }
