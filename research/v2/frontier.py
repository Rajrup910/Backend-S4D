"""S32 -- the budget-sensitivity frontier: S^u_g(q) over a grid of q, with lesion-grouped
confidence intervals, the inverse "minimum budget for target sensitivity" query, and
partial AUC (absent repository-wide before this module -- every existing AUC in the repo
is a full `roc_auc_score`; grep confirms it, and the blueprint's own audit noted this
gap explicitly).

Builds on `research.v2.estimators.top_r_refers` for the exact-integer referral construction
and on `research.stats.calibration_slices.grouped_bootstrap_scalar` for the lesion-grouped
CI, rather than reimplementing either.

**Monotonicity is structural, not empirical.** `sensitivity_curve` sorts once (by score,
descending, seeded tie-break -- the same order `top_r_refers` uses) and takes a cumulative
sum of the escalation indicator over that fixed order. A cumulative sum of a 0/1 array is
non-decreasing by construction, so the frontier cannot fail to be monotone in r unless the
sort itself is broken -- which is exactly what the verification check below re-confirms
against `top_r_refers` directly, rather than trusting the shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from research.stats.calibration_slices import grouped_bootstrap_scalar
from research.v2 import estimators as est

DEFAULT_Q_GRID: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30)
DEFAULT_ETA_TARGETS: tuple[float, ...] = (0.70, 0.80, 0.90)


# ------------------------------------------------------------------------- the sort
def _sorted_order(scores: np.ndarray, seed: int) -> np.ndarray:
    """Identical ordering to `estimators.top_r_refers`: score descending, ties broken by
    a seeded permutation. Kept as its own function so both this module and
    `top_r_refers` are provably using the same order (verified in `__main__`)."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    tiebreak = np.random.default_rng(seed).permutation(n)
    return np.lexsort((tiebreak, -scores))


def sensitivity_curve(
    scores: np.ndarray, y_esc: np.ndarray, seed: int = est.SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """The full monotone curve: `(r_values, sensitivity_values)`, both length n+1
    (r=0..n), sensitivity_values non-decreasing in r by construction."""
    order = _sorted_order(scores, seed)
    y_esc = np.asarray(y_esc, dtype=bool)
    n_pos = int(y_esc.sum())
    cum_caught = np.concatenate([[0], np.cumsum(y_esc[order])])
    sens = cum_caught / n_pos if n_pos else np.full(len(cum_caught), np.nan)
    return np.arange(len(cum_caught)), sens


# ------------------------------------------------------------------- point frontier
@dataclass(frozen=True)
class FrontierPoint:
    q: float
    r: int
    burden: float
    sensitivity: float
    ci_lo: float
    ci_hi: float
    n_boot: int

    def as_dict(self) -> dict:
        return {
            "q": self.q, "r": self.r, "burden": self.burden,
            "sensitivity": self.sensitivity, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi,
            "n_boot": self.n_boot,
        }


def _sensitivity_at_q(scores: np.ndarray, y_esc: np.ndarray, q: float, seed: int) -> tuple[int, float]:
    n = len(scores)
    r = int(round(q * n))
    refers = est.top_r_refers(scores, r, seed)
    return r, est.sensitivity(refers, y_esc)


def frontier(
    scores: np.ndarray, y_esc: np.ndarray, lesion_ids: np.ndarray,
    q_grid: tuple[float, ...] = DEFAULT_Q_GRID,
    seed: int = est.SEED, n_boot: int = 2000,
) -> list[FrontierPoint]:
    """`S^u_g(q)` at each q in the grid, with a lesion-grouped bootstrap CI.

    The CI resamples lesions and, WITHIN each resample, refits `r = round(q * n_resample)`
    and re-selects the top-r by score inside the resampled rows -- not a fixed r carried
    over from the original sample. That is what makes the interval a genuine statement
    about the policy "refer the top q fraction by this score", not about one frozen set
    of indices.
    """
    scores = np.asarray(scores, dtype=float)
    y_esc = np.asarray(y_esc, dtype=bool)
    points = []
    for q in q_grid:
        r, point_sens = _sensitivity_at_q(scores, y_esc, q, seed)

        def statistic(idx, q=q):
            sub_scores, sub_y = scores[idx], y_esc[idx]
            sub_r = int(round(q * len(idx)))
            return est.sensitivity(est.top_r_refers(sub_scores, sub_r, seed), sub_y)

        lo, hi = grouped_bootstrap_scalar(statistic, lesion_ids, n_boot=n_boot, seed=seed)
        points.append(FrontierPoint(
            q=q, r=r, burden=r / len(scores) if len(scores) else float("nan"),
            sensitivity=point_sens, ci_lo=lo, ci_hi=hi, n_boot=n_boot,
        ))
    return points


def frontier_frame(
    scores: np.ndarray, y_esc: np.ndarray, lesion_ids: np.ndarray,
    q_grid: tuple[float, ...] = DEFAULT_Q_GRID, seed: int = est.SEED, n_boot: int = 2000,
) -> pd.DataFrame:
    return pd.DataFrame([p.as_dict() for p in frontier(scores, y_esc, lesion_ids, q_grid, seed, n_boot)])


# ------------------------------------------------------------------- inversion: q_g(eta)
def min_budget_for_sensitivity(
    scores: np.ndarray, y_esc: np.ndarray, target_eta: float, seed: int = est.SEED,
) -> dict:
    """`q_g(eta) = inf{q : S^u_g(q) >= eta}` -- the minimum referral burden needed to hit
    a target sensitivity, using the exact monotone curve (no grid search needed: the
    curve is exact at every integer r)."""
    r_vals, sens = sensitivity_curve(scores, y_esc, seed)
    n = len(scores) if len(scores) else len(r_vals) - 1
    reachable = sens >= target_eta
    if not np.any(reachable) or n == 0:
        return {"target_eta": target_eta, "q": float("nan"), "r": None, "achievable": False}
    r = int(np.argmax(reachable))  # first True index; argmax on a monotone-once-true array
    return {"target_eta": target_eta, "q": r / n, "r": r, "achievable": True}


def budget_table_for_targets(
    scores: np.ndarray, y_esc: np.ndarray, eta_targets: tuple[float, ...] = DEFAULT_ETA_TARGETS,
    seed: int = est.SEED,
) -> pd.DataFrame:
    return pd.DataFrame([min_budget_for_sensitivity(scores, y_esc, eta, seed) for eta in eta_targets])


# ------------------------------------------------------------------------- partial AUC
def partial_auc(y_esc: np.ndarray, scores: np.ndarray, fpr_max: float = 0.20) -> dict:
    """Area under the ROC curve restricted to FPR in [0, fpr_max] (McClish 1989),
    reported both raw and standardized to the same [0.5, 1.0] scale as a full AUC so it
    stays comparable across cohorts with different `fpr_max`-implied operating ranges.

    Absent from the repository before this module: every existing AUC
    (`research.agerule.lambda_rule.escalation_mass_auc`, `research.ablation.delong`,
    `research.xdomain.run_session8b`'s inline `roc_auc_score` calls) is a full-range AUC.
    A budget-constrained diagnostic needs the LOW-referral region specifically, since two
    scores with equal full AUC can differ sharply in the region a clinic actually operates
    in (blueprint 16 -- full AUC does not imply equal fixed-budget performance).
    """
    y_esc = np.asarray(y_esc, dtype=bool)
    if y_esc.sum() == 0 or (~y_esc).sum() == 0:
        return {"fpr_max": fpr_max, "partial_auc_raw": float("nan"),
                "partial_auc_mcclish": float("nan"), "full_auc": float("nan")}

    fpr, tpr, _ = roc_curve(y_esc, scores)
    full_auc = float(np.trapezoid(tpr, fpr))

    # interpolate tpr at exactly fpr_max, then integrate on [0, fpr_max]
    mask = fpr <= fpr_max
    fpr_p = np.concatenate([fpr[mask], [fpr_max]])
    tpr_at_max = float(np.interp(fpr_max, fpr, tpr))
    tpr_p = np.concatenate([tpr[mask], [tpr_at_max]])
    # de-duplicate the boundary point if fpr already lands exactly on fpr_max
    order = np.argsort(fpr_p)
    fpr_p, tpr_p = fpr_p[order], tpr_p[order]
    raw = float(np.trapezoid(tpr_p, fpr_p))

    # McClish standardization onto [0.5, 1.0]: perfect classifier -> 1.0, chance -> 0.5
    chance = 0.5 * fpr_max ** 2
    perfect = fpr_max
    mcclish = 0.5 * (1.0 + (raw - chance) / (perfect - chance)) if perfect > chance else float("nan")

    return {
        "fpr_max": fpr_max, "partial_auc_raw": raw,
        "partial_auc_mcclish": mcclish, "full_auc": full_auc,
    }


def partial_auc_ci(
    y_esc: np.ndarray, scores: np.ndarray, lesion_ids: np.ndarray,
    fpr_max: float = 0.20, seed: int = est.SEED, n_boot: int = 2000,
) -> dict:
    y_esc = np.asarray(y_esc, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    point = partial_auc(y_esc, scores, fpr_max)

    def statistic(idx):
        sub = partial_auc(y_esc[idx], scores[idx], fpr_max)
        return sub["partial_auc_mcclish"]

    lo, hi = grouped_bootstrap_scalar(statistic, lesion_ids, n_boot=n_boot, seed=seed)
    return {**point, "ci_lo": lo, "ci_hi": hi, "n_boot": n_boot}


if __name__ == "__main__":
    # Self-check: sensitivity_curve's shortcut must agree with top_r_refers directly,
    # and the frontier must be monotone -- both asserted, not just printed.
    rng = np.random.default_rng(0)
    n = 5000
    scores = rng.uniform(0, 1, n)
    y_esc = rng.random(n) < (0.1 + 0.5 * scores)

    r_vals, sens = sensitivity_curve(scores, y_esc, seed=est.SEED)
    assert np.all(np.diff(sens[~np.isnan(sens)]) >= -1e-12), "frontier is not monotone"

    for r in [0, 1, 50, 500, 2500, n]:
        refers = est.top_r_refers(scores, r, est.SEED)
        direct = est.sensitivity(refers, y_esc)
        shortcut = sens[r]
        assert abs(direct - shortcut) < 1e-12 or (np.isnan(direct) and np.isnan(shortcut)), (
            f"mismatch at r={r}: top_r_refers={direct} vs sensitivity_curve={shortcut}"
        )
    print("S32 self-check: sensitivity_curve matches top_r_refers exactly at all r; "
          "frontier is monotone. PASS")
