"""Evaluation of prediction sets: coverage, efficiency, and the clinical failure modes.

Coverage and average set size are the standard pair, but neither is the number that
decides whether this is safe to deploy. A prediction set is handed to a clinician as a
shortlist, so the question is what the shortlist *says*:

  * A set holding only benign classes when the lesion is malignant is a **false
    reassurance** — worse than a large set, and invisible to marginal coverage, which
    counts it as a single miss among hundreds of correctly covered moles.
  * An **empty** set (possible under LAC) is not an abstention; it is a shortlist with
    nothing on it, and it needs to be reported rather than folded into "coverage 90%".
  * A **singleton** set is the system committing to one diagnosis, which is where its
    output is actually actionable, so the fraction of singletons is the efficiency
    measure that matters clinically more than the mean size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml.paths import load_class_mapping


@dataclass(frozen=True)
class SetMetrics:
    method: str
    alpha: float
    mondrian: bool
    n: int
    marginal_coverage: float
    mean_set_size: float
    median_set_size: float
    empty_rate: float
    singleton_rate: float
    full_set_rate: float
    escalating_coverage: float       # coverage restricted to truly serious cases
    false_reassurance: int           # malignant truth, set contains no escalating class
    false_reassurance_rate: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)


def evaluate(
    sets: np.ndarray, y_true: np.ndarray, method: str, alpha: float, mondrian: bool
) -> SetMetrics:
    """Full metric set for one (N, C) boolean prediction-set matrix."""
    mapping = load_class_mapping()
    escalating = [c.index for c in mapping.classes if c.needs_escalation]

    n, num_classes = sets.shape
    covered = sets[np.arange(n), y_true]
    sizes = sets.sum(axis=1)

    is_serious = np.isin(y_true, escalating)
    set_has_escalating = sets[:, escalating].any(axis=1)
    reassured = is_serious & ~set_has_escalating

    per_class = {}
    for skin_class in mapping.classes:
        in_class = y_true == skin_class.index
        if not in_class.any():
            continue
        per_class[skin_class.code] = {
            "n": int(in_class.sum()),
            "coverage": float(covered[in_class].mean()),
            "mean_set_size": float(sizes[in_class].mean()),
        }

    return SetMetrics(
        method=method,
        alpha=alpha,
        mondrian=mondrian,
        n=n,
        marginal_coverage=float(covered.mean()),
        mean_set_size=float(sizes.mean()),
        median_set_size=float(np.median(sizes)),
        empty_rate=float((sizes == 0).mean()),
        singleton_rate=float((sizes == 1).mean()),
        full_set_rate=float((sizes == num_classes).mean()),
        escalating_coverage=float(covered[is_serious].mean()) if is_serious.any() else float("nan"),
        false_reassurance=int(reassured.sum()),
        false_reassurance_rate=float(reassured.sum() / is_serious.sum()) if is_serious.any() else float("nan"),
        per_class=per_class,
    )


def worst_class_coverage(metrics: SetMetrics) -> tuple[str, float]:
    """The class the guarantee serves worst — the honest summary of conditional coverage."""
    if not metrics.per_class:
        return ("none", float("nan"))
    code = min(metrics.per_class, key=lambda k: metrics.per_class[k]["coverage"])
    return (code, metrics.per_class[code]["coverage"])
