"""Per-class cost-sensitive decision thresholds, fit on a held-out split (val or OOF).

Decision rule: predict argmax_c (p_c - theta_c) -- a per-class score shift, so theta > 0
makes a class harder to predict (raises the bar) and theta < 0 makes it easier (e.g.
biasing toward escalating classes). Thresholds are found by coordinate-descent grid
search minimizing expected clinical cost (`cost_matrix.py`) subject to a specificity
floor on the binary escalation view, via a penalty term -- joint threshold optimization
over 7 classes has no simple closed form, and this dominates a handful of coordinate-
descent passes cheaply given only ~1500 validation rows per pass.

`ThresholdState`'s fitted-cost fields are named `fit_*` rather than `val_*`: since
session 6 the fitting split can be either validation or the K-fold out-of-fold
predictions over the train split (`research.fitsplit`), and a field called `val_cost`
holding an OOF number is the kind of quiet mislabelling that survives into a paper.

**`optimize_thresholds_by_group` and its capacity warning.** Group-conditional thresholds
are what an age-, sex- or site-conditional decision rule needs, and this is the place for
them. But the search has 7 free parameters over a 13-point grid, so it needs a *lot* of
positives per group to mean anything: the `min_group_positives` gate exists to make the
sample-size requirement explicit, and groups below it fall back to the pooled thresholds
rather than getting a 7-vector fitted on a handful of cases. A group with ~20 escalating
cases should not be given its own 7-vector at all -- a single scalar operating point on
one statistic is the defensible construction at that sample size, and it is a different
function from this one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.paths import ClassMapping, load_class_mapping
from research.thresholds.cost_matrix import expected_cost


@dataclass(frozen=True)
class ThresholdState:
    thresholds: np.ndarray  # (C,)
    fit_cost: float
    fit_specificity: float


def _escalation_specificity(y_true: np.ndarray, y_pred: np.ndarray, escalating_idx: set[int]) -> float:
    true_escalates = np.isin(y_true, list(escalating_idx))
    pred_escalates = np.isin(y_pred, list(escalating_idx))
    true_negative = np.sum(~true_escalates & ~pred_escalates)
    false_positive = np.sum(~true_escalates & pred_escalates)
    denom = true_negative + false_positive
    return float(true_negative / denom) if denom else 1.0


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (probs - thresholds).argmax(axis=1)


def optimize_thresholds(
    probs: np.ndarray,
    y_true: np.ndarray,
    cost_matrix: np.ndarray,
    mapping: ClassMapping | None = None,
    min_specificity: float = 0.85,
    grid: np.ndarray | None = None,
    passes: int = 3,
    penalty_weight: float = 50.0,
) -> ThresholdState:
    mapping = mapping or load_class_mapping()
    escalating_idx = {c.index for c in mapping.classes if c.needs_escalation}
    grid = grid if grid is not None else np.linspace(-0.3, 0.3, 13)
    num_classes = probs.shape[1]

    def objective(thetas: np.ndarray) -> tuple[float, float, float]:
        preds = apply_thresholds(probs, thetas)
        cost = expected_cost(y_true, preds, cost_matrix)
        specificity = _escalation_specificity(y_true, preds, escalating_idx)
        penalty = penalty_weight * max(0.0, min_specificity - specificity)
        return cost + penalty, cost, specificity

    thetas = np.zeros(num_classes)
    best_score, best_cost, best_spec = objective(thetas)

    for _ in range(passes):
        for class_idx in range(num_classes):
            best_delta = thetas[class_idx]
            for delta in grid:
                trial = thetas.copy()
                trial[class_idx] = delta
                score, cost, spec = objective(trial)
                if score < best_score:
                    best_score, best_cost, best_spec = score, cost, spec
                    best_delta = delta
            thetas[class_idx] = best_delta

    return ThresholdState(thresholds=thetas, fit_cost=best_cost, fit_specificity=best_spec)


@dataclass(frozen=True)
class GroupThresholdState:
    """Per-group thresholds, with the groups that were too small to fit named explicitly."""

    pooled: ThresholdState
    by_group: dict[str, ThresholdState]      # groups that cleared the positives gate
    fitted_groups: tuple[str, ...]
    fallback_groups: tuple[str, ...]         # groups falling back to `pooled`
    group_sizes: dict[str, int]
    group_positives: dict[str, int]          # true *escalating* cases, the binding count
    min_group_positives: int

    def thresholds_for(self, group: str) -> np.ndarray:
        state = self.by_group.get(group)
        return (state or self.pooled).thresholds

    def as_dict(self) -> dict:
        """Flat, serialisable form for `research.fitsplit.write_fit_state`."""
        return {
            "pooled": {
                "thresholds": self.pooled.thresholds,
                "fit_cost": self.pooled.fit_cost,
                "fit_specificity": self.pooled.fit_specificity,
            },
            "by_group": {
                group: {
                    "thresholds": state.thresholds,
                    "fit_cost": state.fit_cost,
                    "fit_specificity": state.fit_specificity,
                }
                for group, state in self.by_group.items()
            },
            "fitted_groups": list(self.fitted_groups),
            "fallback_groups": list(self.fallback_groups),
            "group_sizes": self.group_sizes,
            "group_positives": self.group_positives,
            "min_group_positives": self.min_group_positives,
        }


def apply_group_thresholds(
    probs: np.ndarray, groups: np.ndarray, state: GroupThresholdState
) -> np.ndarray:
    """Apply each row's group thresholds, or the pooled ones where the group did not qualify."""
    if len(groups) != len(probs):
        raise ValueError(f"groups has {len(groups)} rows, probs has {len(probs)}")
    preds = np.empty(len(probs), dtype=int)
    for group in np.unique(groups):
        mask = groups == group
        preds[mask] = apply_thresholds(probs[mask], state.thresholds_for(str(group)))
    return preds


def optimize_thresholds_by_group(
    probs: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    cost_matrix: np.ndarray,
    mapping: ClassMapping | None = None,
    min_group_positives: int = 30,
    **kwargs,
) -> GroupThresholdState:
    """Fit one threshold vector per group, falling back to pooled where data is thin.

    The gate counts **true escalating cases**, not group size: the objective is dominated
    by the escalation specificity penalty and by the cost of missing a malignancy, so a
    group of 400 benign nevi with 8 malignancies carries no more information about where
    the escalation boundary should sit than those 8 cases do. Groups below the gate are
    not silently dropped -- they are listed in `fallback_groups` and take the pooled
    thresholds, so every row still gets a decision and the report can state which groups
    got their own rule.
    """
    mapping = mapping or load_class_mapping()
    escalating_idx = {c.index for c in mapping.classes if c.needs_escalation}
    groups = np.asarray(groups).astype(str)
    if len(groups) != len(y_true):
        raise ValueError(f"groups has {len(groups)} rows, y_true has {len(y_true)}")

    pooled = optimize_thresholds(probs, y_true, cost_matrix, mapping, **kwargs)

    sizes: dict[str, int] = {}
    positives: dict[str, int] = {}
    by_group: dict[str, ThresholdState] = {}
    fitted, fallback = [], []

    for group in sorted(np.unique(groups)):
        mask = groups == group
        n_positive = int(np.isin(y_true[mask], list(escalating_idx)).sum())
        sizes[group] = int(mask.sum())
        positives[group] = n_positive
        if n_positive >= min_group_positives:
            by_group[group] = optimize_thresholds(
                probs[mask], y_true[mask], cost_matrix, mapping, **kwargs
            )
            fitted.append(group)
        else:
            fallback.append(group)

    return GroupThresholdState(
        pooled=pooled,
        by_group=by_group,
        fitted_groups=tuple(fitted),
        fallback_groups=tuple(fallback),
        group_sizes=sizes,
        group_positives=positives,
        min_group_positives=min_group_positives,
    )
