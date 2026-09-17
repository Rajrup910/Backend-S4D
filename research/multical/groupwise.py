"""Fit one Dirichlet map per age band instead of one map for everyone.

S7 measured the thing this module acts on (`research/stats/results_oof/band_calibration.csv`):
after the **global** Dirichlet map, the residual signed calibration gap *flips sign* across age
bands -- under-40 stays under-confident (-0.027) while 60+ is pushed into over-confidence
(+0.041) -- so an aggregate ECE of 0.017-0.025 is hiding two opposite-signed errors. A single
affine map has one set of coefficients for every patient and cannot close two gaps that point in
different directions. This is exactly the motivation Hebert-Johnson et al. (2018) give for
multicalibration: calibrate within each of several overlapping subgroups, not just marginally.

**Why some bands get their own map and some don't.** A Dirichlet map on C=7 classes has
C*(C-1) off-diagonal weights plus C diagonal weights and C biases fit by ODIR-regularised
LBFGS -- on the order of 50 free numbers. The `unknown`-age band has 38 OOF rows. Fitting a
band-specific map there would not be group-conditional calibration, it would be overfitting
dressed up as one, so a band below `MIN_FIT_SIZE` falls back to the global map and is named as
having done so, exactly as `research.agerule.lambda_rule.fit_lambda_by_group` falls back to the
pooled lambda for bands under its own positives floor.

**Leak discipline.** Reporting a band's calibration under a map fitted on that same band's rows
would be in-sample and would flatter the fix -- the same hazard `crossfit_calibration` exists to
remove for the global map. `crossfit_group_dirichlet` therefore K-fold cross-fits *within* each
band using the project's existing OOF fold assignment, and falls back to the (also cross-fitted)
global map for rows in an under-powered band. The **deployed** per-band states this module also
produces -- fit on the whole band's OOF rows, no cross-fitting -- are what a future session would
apply to validation, the reserved cohort, or an external panel; they are not used to score OOF
here, for the same reason the deployed global map is not used to score the OOF fit split either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research.calibration.methods import CalibrationState, apply_calibration, fit_dirichlet_calibration

EPS = 1e-12

#: A Dirichlet map has ~C*(C+1) free parameters (56 at C=7). 300 rows is roughly 5 rows per
#: parameter -- thin, but this is a calibration map (bias + near-diagonal weight), not a
#: classifier, and it is the same order of magnitude as the smallest band that *does* get its
#: own map in this project (<40, 1,319 OOF rows). Below this a band-specific fit is overfitting,
#: not multicalibration, and falls back to the global map.
MIN_FIT_SIZE = 300


@dataclass(frozen=True)
class GroupDirichletState:
    band: str
    n: int
    fitted: bool  # False => this band has no map of its own; it uses the global fallback
    state: CalibrationState | None  # None when fitted is False


def fit_group_dirichlet(
    probs: np.ndarray,
    y_true: np.ndarray,
    bands: np.ndarray,
    *,
    min_fit_size: int = MIN_FIT_SIZE,
    l2_lambda: float = 0.1,
) -> dict[str, GroupDirichletState]:
    """One Dirichlet map per band with >= `min_fit_size` rows, fit on that band's rows alone.

    Deployed (not cross-fitted) -- for applying to a split this fit never saw.
    """
    bands = np.asarray(bands).astype(str)
    out: dict[str, GroupDirichletState] = {}
    for band in sorted(np.unique(bands)):
        mask = bands == band
        n = int(mask.sum())
        if n >= min_fit_size:
            state = fit_dirichlet_calibration(probs[mask], y_true[mask], l2_lambda=l2_lambda)
            out[band] = GroupDirichletState(band=band, n=n, fitted=True, state=state)
        else:
            out[band] = GroupDirichletState(band=band, n=n, fitted=False, state=None)
    return out


def crossfit_group_dirichlet(
    probs: np.ndarray,
    y_true: np.ndarray,
    bands: np.ndarray,
    folds: np.ndarray,
    global_crossfit: np.ndarray,
    *,
    min_fit_size: int = MIN_FIT_SIZE,
    l2_lambda: float = 0.1,
) -> np.ndarray:
    """Leak-free per-band calibrated probabilities for reporting ECE on the fit split itself.

    For a band with enough rows, each fold's rows are scored by a map fit on that band's
    *other* folds only. For an under-powered band, its rows keep `global_crossfit` -- itself
    already cross-fitted by the caller (`research.agerule.lambda_rule.crossfit_calibration`) --
    so no row here is ever scored by a map that has seen it.
    """
    bands = np.asarray(bands).astype(str)
    out = global_crossfit.copy()
    for band in np.unique(bands):
        band_mask = bands == band
        if band_mask.sum() < min_fit_size:
            continue  # already carries the global cross-fit value
        band_folds = folds[band_mask]
        band_probs = probs[band_mask]
        band_y = y_true[band_mask]
        band_out = np.empty_like(band_probs)
        for fold in np.unique(band_folds):
            held_out = band_folds == fold
            state = fit_dirichlet_calibration(
                band_probs[~held_out], band_y[~held_out], l2_lambda=l2_lambda
            )
            band_out[held_out] = apply_calibration(
                state, np.log(np.clip(band_probs[held_out], EPS, None))
            )
        out[band_mask] = band_out
    return out


def apply_group_dirichlet(
    states: dict[str, GroupDirichletState],
    global_state: CalibrationState,
    probs: np.ndarray,
    bands: np.ndarray,
) -> np.ndarray:
    """Apply the deployed per-band maps, falling back to the deployed global map by band."""
    bands = np.asarray(bands).astype(str)
    out = np.empty_like(probs)
    log_probs = np.log(np.clip(probs, EPS, None))
    for band in np.unique(bands):
        mask = bands == band
        entry = states.get(band)
        state = entry.state if (entry is not None and entry.fitted) else global_state
        out[mask] = apply_calibration(state, log_probs[mask])
    return out
