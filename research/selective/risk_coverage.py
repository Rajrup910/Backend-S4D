"""Risk-coverage analysis for an abstaining classifier.

The clinical framing: the system is allowed to say "I don't know" and refer the case
for in-person biopsy. Referring a case is never a diagnostic error — it is the correct
conservative action — so the metrics that matter are computed on the *retained* cases
only, alongside a description of what landed in the referral pile.

Two things are kept strictly separate:

  * **Where the abstention threshold comes from.** It is the quantile of the uncertainty
    score on the *validation* split that leaves the target coverage. The test split is
    then thresholded with that fixed number, and the coverage it actually achieves is
    reported rather than forced. Taking the quantile on test instead would let the test
    labels' own difficulty distribution set the operating point — a leak that silently
    flatters every number downstream.
  * **How well the score ranks errors.** AURC and its excess over the optimal ranking are
    computed on whichever split is being described, and are a property of the score, not
    of a chosen threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.evaluation.metrics import compute_metrics
from ml.paths import load_class_mapping

DEFAULT_ABSTENTION_RATES = (0.0, 0.05, 0.10, 0.15, 0.20)


@dataclass(frozen=True)
class SelectivePoint:
    """One operating point: a threshold, what it kept, and how the kept cases scored."""

    abstention_rate: float          # target, set on val
    threshold: float                # val quantile of the uncertainty score
    coverage: float                 # achieved on the split being evaluated
    num_kept: int
    num_abstained: int
    macro_f1: float
    balanced_accuracy: float
    accuracy: float
    escalation_sensitivity: float
    missed_serious: int             # among retained cases only
    abstained_escalating: int       # serious cases correctly sent for biopsy
    abstained_would_be_missed: int  # of those, ones the model would have got wrong


def coverage_threshold(val_scores: np.ndarray, abstention_rate: float) -> float:
    """Score above which a case is referred, set to abstain on `abstention_rate` of val."""
    if not 0.0 <= abstention_rate < 1.0:
        raise ValueError(f"abstention_rate must be in [0, 1), got {abstention_rate}")
    if abstention_rate == 0.0:
        return float(np.inf)
    return float(np.quantile(val_scores, 1.0 - abstention_rate))


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    abstention_rate: float,
) -> SelectivePoint:
    """Metrics on the cases scoring at or below `threshold`, plus the referral pile."""
    mapping = load_class_mapping()
    escalating = [c.index for c in mapping.classes if c.needs_escalation]

    keep = scores <= threshold
    if keep.sum() == 0:
        raise ValueError(f"threshold {threshold} abstains on every case")

    kept = compute_metrics(y_true[keep], y_pred[keep], probs[keep])

    referred = ~keep
    referred_serious = np.isin(y_true[referred], escalating)
    referred_wrong = y_pred[referred] != y_true[referred]

    return SelectivePoint(
        abstention_rate=abstention_rate,
        threshold=threshold,
        coverage=float(keep.mean()),
        num_kept=int(keep.sum()),
        num_abstained=int(referred.sum()),
        macro_f1=float(kept["macro_f1"]),
        balanced_accuracy=float(kept["balanced_accuracy"]),
        accuracy=float(kept["accuracy"]),
        escalation_sensitivity=float(kept["clinical"]["binary_sensitivity"]),
        missed_serious=int(kept["clinical"]["missed_serious_cases"]),
        abstained_escalating=int(referred_serious.sum()),
        abstained_would_be_missed=int((referred_serious & referred_wrong).sum()),
    )


def risk_coverage_curve(
    y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Selective risk (0-1 error rate) as coverage sweeps from 1/N to 1.

    Cases are sorted by ascending uncertainty and admitted one at a time, so this is the
    score's ranking quality, independent of any threshold choice.
    """
    order = np.argsort(scores, kind="stable")
    errors = (y_pred != y_true).astype(np.float64)[order]
    n = len(errors)
    coverage = np.arange(1, n + 1, dtype=np.float64) / n
    risk = np.cumsum(errors) / np.arange(1, n + 1)
    return coverage, risk


def aurc(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    _, risk = risk_coverage_curve(y_true, y_pred, scores)
    return float(risk.mean())


def excess_aurc(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> float:
    """AURC minus the AURC of a perfect ranking (all errors last) on the same errors.

    Subtracting the oracle removes the part of AURC that is just the model's overall
    error rate, leaving only how well the uncertainty score orders mistakes. Comparing
    raw AURC across models with different accuracies would otherwise conflate the two.
    """
    errors = (y_pred != y_true).astype(np.float64)
    optimal_scores = errors  # correct cases rank first, errors last
    return aurc(y_true, y_pred, scores) - aurc(y_true, y_pred, optimal_scores)


def sweep(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    rates: tuple[float, ...] = DEFAULT_ABSTENTION_RATES,
) -> list[SelectivePoint]:
    """Evaluate every target abstention rate with thresholds taken from `val_scores`."""
    return [
        evaluate_at_threshold(
            y_true, y_pred, probs, test_scores, coverage_threshold(val_scores, rate), rate
        )
        for rate in rates
    ]
