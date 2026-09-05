"""A single-parameter, age-conditional escalation rule (the lambda rule).

Session 4 measured an under-40 blind spot: escalation sensitivity 0.143 in patients <40
on the test split, against ~0.78 in both older bands, and abstention did not rescue it
(only 11.1% of the misses were referred). This module builds the instrument that acts on
it, and the diagnostic that says whether acting on it is even the right move.

**The diagnostic comes first, because it decides what kind of failure this is.**
`escalation_mass_auc` scores each lesion by the probability mass the model puts on the
escalating classes and asks how well that ranks true escalating cases within a band. If
the AUC is high while argmax sensitivity is low, the *representation* separates the
classes and the *decision rule* is throwing that separation away -- a failure at the
operating point, which a threshold can fix. If the AUC were also low, no threshold would
help and the honest conclusion would be that the model cannot see melanoma in this band.
These are different papers, and one number decides which one this is.

**Why one scalar and not a 7-vector.** `research.thresholds.optimize` already offers
`optimize_thresholds_by_group`, which fits a full per-class threshold vector per group.
Its own docstring warns against using it at this sample size, and the warning binds here:
the <40 band has 64 escalating cases out of 6,981 OOF rows, so a 7-parameter search over a
13-point grid has roughly nine positives per free parameter. This module therefore fits
**one** number per band -- a uniform bonus lambda added to every escalating class's score:

    predict argmax_c ( p_c + lambda * 1[c escalates] )

which is exactly `apply_thresholds` with theta_c = -lambda on the escalating classes and 0
elsewhere, so the decision rule is the same family the project already uses and defends,
restricted to its one clinically meaningful direction. `fit_group_vector_ablation` fits
the 7-vector anyway, purely so the report can show how much less stable it is rather than
asserting it.

**Cross-fitting.** Under `--fit-split oof` the Dirichlet calibrator is fitted on the same
6,981 OOF rows lambda is fitted on, so calibrated probabilities on those rows are
in-sample with respect to the calibrator, and lambda would inherit that optimism.
`crossfit_calibration` refits the calibrator K times on K-1 folds and scores each held-out
fold with a map that never saw it. This is the same hazard S4 caught in the Mahalanobis
score, in a different place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.paths import ClassMapping, load_class_mapping
from research.calibration.methods import apply_calibration, fit_dirichlet_calibration
from research.thresholds.cost_matrix import expected_cost
from research.thresholds.optimize import _escalation_specificity, apply_thresholds

EPS = 1e-12
# Upper end chosen after measuring, not guessed: at 0.60 the 40-59 band and the pooled fit
# both selected the largest value on the grid, i.e. the grid was the binding constraint
# rather than the data. `fit_lambda` now refuses to return a boundary solution outright, so
# a clipped parameter cannot be frozen into a pre-registration by accident.
DEFAULT_GRID = np.round(np.arange(0.0, 1.201, 0.01), 4)
MIN_GROUP_POSITIVES = 30


class GridBoundaryError(RuntimeError):
    """The cost-minimising lambda sits at the top of the search grid, so it is not a minimum."""


# --------------------------------------------------------------------------------------
# diagnostic: is this a decision-rule failure or a representation failure?
# --------------------------------------------------------------------------------------

def escalating_indices(mapping: ClassMapping | None = None) -> list[int]:
    mapping = mapping or load_class_mapping()
    return [c.index for c in mapping.classes if c.needs_escalation]


def escalation_mass(probs: np.ndarray, escalating_idx: list[int]) -> np.ndarray:
    """Total probability assigned to the escalating classes -- the rule-free severity score."""
    return probs[:, escalating_idx].sum(axis=1)


def escalation_mass_auc(
    y_true: np.ndarray, probs: np.ndarray, escalating_idx: list[int]
) -> float:
    """AUC of escalation mass against the binary escalate/do-not label.

    Deliberately independent of any threshold, calibration map or class prior: it measures
    only whether the model *ranks* escalating lesions above benign ones inside this set.
    """
    labels = np.isin(y_true, escalating_idx)
    if labels.all() or not labels.any():
        return float("nan")
    return float(roc_auc_score(labels, escalation_mass(probs, escalating_idx)))


def crossfit_calibration(
    probs: np.ndarray, y_true: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    """Dirichlet-calibrated probabilities where no row was seen by the map that scores it."""
    out = np.empty_like(probs)
    for fold in np.unique(folds):
        held_out = folds == fold
        state = fit_dirichlet_calibration(probs[~held_out], y_true[~held_out])
        out[held_out] = apply_calibration(state, np.log(np.clip(probs[held_out], EPS, None)))
    return out


# --------------------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------------------

def lambda_thresholds(lam: float, num_classes: int, escalating_idx: list[int]) -> np.ndarray:
    """The lambda rule expressed in the project's existing theta parameterisation."""
    theta = np.zeros(num_classes)
    theta[escalating_idx] = -lam
    return theta


def apply_lambda(probs: np.ndarray, lam: float, escalating_idx: list[int]) -> np.ndarray:
    return apply_thresholds(probs, lambda_thresholds(lam, probs.shape[1], escalating_idx))


@dataclass(frozen=True)
class LambdaState:
    group: str
    lam: float
    n: int
    n_positive: int
    fit_cost: float
    fit_sensitivity: float
    fit_specificity: float
    fit_referral_rate: float
    baseline_cost: float
    baseline_sensitivity: float
    baseline_specificity: float
    baseline_referral_rate: float
    fitted: bool             # False => this group fell back to the pooled lambda

    def as_dict(self) -> dict:
        return asdict(self)


def _operating_point(
    probs: np.ndarray,
    y_true: np.ndarray,
    lam: float,
    cost_matrix: np.ndarray,
    escalating_idx: list[int],
) -> tuple[float, float, float, float]:
    """(expected cost, escalation sensitivity, escalation specificity, referral rate)."""
    preds = apply_lambda(probs, lam, escalating_idx)
    cost = expected_cost(y_true, preds, cost_matrix)
    true_esc = np.isin(y_true, escalating_idx)
    pred_esc = np.isin(preds, escalating_idx)
    positives = int(true_esc.sum())
    sensitivity = float((true_esc & pred_esc).sum() / positives) if positives else float("nan")
    specificity = _escalation_specificity(y_true, preds, set(escalating_idx))
    return float(cost), sensitivity, float(specificity), float(pred_esc.mean())


def sweep_lambda(
    probs: np.ndarray,
    y_true: np.ndarray,
    cost_matrix: np.ndarray,
    escalating_idx: list[int],
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """The whole operating-point curve, so any point can be read off rather than refitted."""
    grid = DEFAULT_GRID if grid is None else grid
    rows = []
    for lam in grid:
        cost, sens, spec, referral = _operating_point(
            probs, y_true, float(lam), cost_matrix, escalating_idx
        )
        rows.append(
            {"lambda": float(lam), "expected_cost": cost, "escalation_sens": sens,
             "escalation_spec": spec, "referral_rate": referral}
        )
    return pd.DataFrame(rows)


def fit_lambda(
    probs: np.ndarray,
    y_true: np.ndarray,
    cost_matrix: np.ndarray,
    escalating_idx: list[int],
    group: str = "pooled",
    grid: np.ndarray | None = None,
    min_specificity: float = 0.85,
    fitted: bool = True,
    strict: bool = True,
) -> LambdaState:
    """Cost-minimising lambda subject to an escalation-specificity floor.

    One free parameter over a 61-point grid, so the search is exhaustive and there is no
    local-minimum story to tell. Ties go to the *smaller* lambda: when two operating points
    cost the same, the one that refers fewer patients is the one to deploy.
    """
    grid = DEFAULT_GRID if grid is None else grid
    base = _operating_point(probs, y_true, 0.0, cost_matrix, escalating_idx)

    best_lam, best = 0.0, base
    for lam in grid:
        cost, sens, spec, referral = _operating_point(
            probs, y_true, float(lam), cost_matrix, escalating_idx
        )
        if spec < min_specificity:
            continue
        if cost < best[0] - 1e-12:
            best_lam, best = float(lam), (cost, sens, spec, referral)

    if strict and best_lam >= float(np.max(grid)) - 1e-12 and best_lam > 0.0:
        raise GridBoundaryError(
            f"group {group!r}: the cost-minimising lambda={best_lam} is the largest value on "
            f"the grid, so the grid is the binding constraint and this is not an interior "
            f"optimum. Widen `grid` and refit -- do not freeze a clipped parameter."
        )

    return LambdaState(
        group=group, lam=best_lam, n=len(y_true),
        n_positive=int(np.isin(y_true, escalating_idx).sum()),
        fit_cost=best[0], fit_sensitivity=best[1],
        fit_specificity=best[2], fit_referral_rate=best[3],
        baseline_cost=base[0], baseline_sensitivity=base[1],
        baseline_specificity=base[2], baseline_referral_rate=base[3],
        fitted=fitted,
    )


def fit_lambda_by_group(
    probs: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    cost_matrix: np.ndarray,
    escalating_idx: list[int],
    min_group_positives: int = MIN_GROUP_POSITIVES,
    **kwargs,
) -> tuple[LambdaState, dict[str, LambdaState]]:
    """Pooled lambda plus one per qualifying group; thin groups fall back to pooled.

    The gate counts true escalating cases rather than group size, for the reason
    `optimize_thresholds_by_group` gives: a band of 400 nevi with 8 malignancies says
    nothing about where its escalation boundary belongs beyond what those 8 cases say.
    """
    groups = np.asarray(groups).astype(str)
    pooled = fit_lambda(probs, y_true, cost_matrix, escalating_idx, group="pooled", **kwargs)

    by_group: dict[str, LambdaState] = {}
    for group in sorted(np.unique(groups)):
        mask = groups == group
        n_positive = int(np.isin(y_true[mask], escalating_idx).sum())
        if n_positive >= min_group_positives:
            by_group[group] = fit_lambda(
                probs[mask], y_true[mask], cost_matrix, escalating_idx, group=group, **kwargs
            )
        else:
            base = _operating_point(probs[mask], y_true[mask], 0.0, cost_matrix, escalating_idx)
            applied = _operating_point(
                probs[mask], y_true[mask], pooled.lam, cost_matrix, escalating_idx
            )
            by_group[group] = LambdaState(
                group=group, lam=pooled.lam, n=int(mask.sum()), n_positive=n_positive,
                fit_cost=applied[0], fit_sensitivity=applied[1],
                fit_specificity=applied[2], fit_referral_rate=applied[3],
                baseline_cost=base[0], baseline_sensitivity=base[1],
                baseline_specificity=base[2], baseline_referral_rate=base[3],
                fitted=False,
            )
    return pooled, by_group


def apply_group_lambda(
    probs: np.ndarray,
    groups: np.ndarray,
    by_group: dict[str, LambdaState],
    pooled: LambdaState,
    escalating_idx: list[int],
) -> np.ndarray:
    """Apply each row's band lambda. Unseen bands take the pooled value, never crash."""
    groups = np.asarray(groups).astype(str)
    preds = np.empty(len(probs), dtype=int)
    for group in np.unique(groups):
        mask = groups == group
        state = by_group.get(str(group))
        lam = pooled.lam if state is None else state.lam
        preds[mask] = apply_lambda(probs[mask], lam, escalating_idx)
    return preds


# --------------------------------------------------------------------------------------
# stability
# --------------------------------------------------------------------------------------

def bootstrap_lambda(
    probs: np.ndarray,
    y_true: np.ndarray,
    lesion_ids: np.ndarray,
    cost_matrix: np.ndarray,
    escalating_idx: list[int],
    n_boot: int = 400,
    seed: int = 20260904,
    **kwargs,
) -> np.ndarray:
    """Lesion-resampled distribution of the fitted lambda.

    Resamples **lesions**, not images, for the same reason `research.ablation.bootstrap`
    does: several images of one lesion are not independent draws, and resampling images
    would report an interval narrower than the data supports.
    """
    rng = np.random.default_rng(seed)
    unique_lesions = np.unique(lesion_ids)
    index_by_lesion = {les: np.flatnonzero(lesion_ids == les) for les in unique_lesions}

    out = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(unique_lesions, size=len(unique_lesions), replace=True)
        idx = np.concatenate([index_by_lesion[les] for les in drawn])
        if np.isin(y_true[idx], escalating_idx).sum() == 0:
            out[b] = np.nan
            continue
        # `strict=False`: a resample that happens to push lambda to the grid edge is a
        # property of that resample, not a mis-specified grid, and killing the whole
        # bootstrap over one draw would hide the spread this function exists to measure.
        out[b] = fit_lambda(
            probs[idx], y_true[idx], cost_matrix, escalating_idx, strict=False, **kwargs
        ).lam
    return out


def fit_group_vector_ablation(
    probs: np.ndarray,
    y_true: np.ndarray,
    lesion_ids: np.ndarray,
    cost_matrix: np.ndarray,
    mapping: ClassMapping,
    escalating_idx: list[int],
    n_boot: int = 100,
    seed: int = 20260904,
) -> dict:
    """Bootstrap spread of the 7-parameter threshold vector, for the "why not this" row.

    Returns the mean per-class standard deviation of the fitted theta across lesion
    resamples, alongside the same statistic for the 1-parameter rule, so the choice of
    instrument is evidenced rather than argued.

    **The raw standard deviations are not comparable and must not be reported as if they
    were.** `optimize_thresholds` searches theta over [-0.3, 0.3] (width 0.6) while
    `fit_lambda` searches lambda over [0, 1.2] (width 1.2), so an identical raw spread
    means twice the instability for the vector. Both are therefore also reported as a
    fraction of the range each parameter was actually searched over, and that normalised
    figure is the one the report cites.
    """
    from research.thresholds.optimize import optimize_thresholds

    rng = np.random.default_rng(seed)
    unique_lesions = np.unique(lesion_ids)
    index_by_lesion = {les: np.flatnonzero(lesion_ids == les) for les in unique_lesions}

    theta_draws, lambda_draws = [], []
    for _ in range(n_boot):
        drawn = rng.choice(unique_lesions, size=len(unique_lesions), replace=True)
        idx = np.concatenate([index_by_lesion[les] for les in drawn])
        if np.isin(y_true[idx], escalating_idx).sum() == 0:
            continue
        theta_draws.append(
            optimize_thresholds(probs[idx], y_true[idx], cost_matrix, mapping).thresholds
        )
        lambda_draws.append(
            fit_lambda(probs[idx], y_true[idx], cost_matrix, escalating_idx, strict=False).lam
        )

    theta = np.vstack(theta_draws)
    theta_range = 0.6   # optimize_thresholds' default grid, np.linspace(-0.3, 0.3, 13)
    lambda_range = float(np.max(DEFAULT_GRID) - np.min(DEFAULT_GRID))
    vector_std = float(theta[:, escalating_idx].std(axis=0).mean())
    lambda_std = float(np.std(lambda_draws))
    return {
        "n_boot": len(theta_draws),
        "vector_per_class_std": theta.std(axis=0).tolist(),
        "vector_mean_per_class_std": float(theta.std(axis=0).mean()),
        "vector_escalating_std": vector_std,
        "lambda_std": lambda_std,
        "vector_search_range": theta_range,
        "lambda_search_range": lambda_range,
        "vector_std_as_share_of_range": vector_std / theta_range,
        "lambda_std_as_share_of_range": lambda_std / lambda_range,
        "free_parameters_vector": int(theta.shape[1]),
        "free_parameters_lambda": 1,
        "class_codes": list(mapping.codes),
    }
