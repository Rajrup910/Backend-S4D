"""S57a -- age resolution: 5-year diagnostic, the S5 baseline, and candidate lambda(age) fits.

**Terminology.** "Age" is the *patient's* recorded chronological age (`age` in the HAM manifest,
`age_approx` in ISIC). No lesion-age target exists anywhere in this corpus.

**Not H6 under a new name.** V3 asked whether *removing* age signal from the representation
helps, and H6 falsified it. S57 asks the complementary question: given that the age signal is
present and load-bearing, can the *decision layer* use it more precisely? Conditioning is not
invariance.

The frozen S5 rule is ``argmax_c (p_c + lambda_band * 1[c escalates])`` with one lambda per
coarse band. This module replaces ``lambda_band`` with ``lambda(age)`` and fits the S57b ablation
family on development data only:

    A0  argmax (lambda = 0)
    A1  frozen 3-band lambda, loaded through `frozen_params` -- never typed
    A2  independent 5-year lambda, S5's minimum-positives gate, pooled fallback
    A3  kernel-smoothed lambda(a): local cost minimisation, Gaussian kernel, CV bandwidth
    A4  linear lambda(a)        } lambda = softplus(B(a) @ beta), fitted on the S5 objective
    A5  quadratic lambda(a)     }
    A6  penalised cubic B-spline lambda(a), second-difference (P-spline) ridge penalty, CV kappa
    A7  shrunk 5-year lambda: alpha_b * local_b + (1 - alpha_b) * A6(a_b), alpha_b = P_b/(P_b+tau)

**The objective is loaded, not invented**: expected clinical cost under
`research.thresholds.cost_matrix.build_cost_matrix`, subject to escalation specificity >= 0.85
(`lambda_rule.py:187`). A1's floor was applied *per coarse band*, so every curve here is held to
the same floor in each of `<40`, `40-59`, `60+` -- the same unit, so A1 and the curves are
comparable. A curve is scored against it with a hinge penalty inside the optimiser and the
realised per-band specificity is always reported.

**Why a softplus link everywhere.** A negative lambda suppresses escalating classes, the opposite
of intent (runbook §S57a), so A4/A5/A6 are linear/quadratic/spline *in the link*, not in lambda.
Declared: A4 is therefore "log-linear-ish", not literally ``beta0 + beta1 * a``.

**Why the loss, not a Ridge target.** The runbook names `sklearn.linear_model.Ridge` for the
penalised coefficients. The S5 objective is a step function of lambda, not a squared error, so
there is no response for Ridge to regress on; the spline carries the *ridge penalty* on its
coefficient second differences (a P-spline, Eilers & Marx 1996) and is optimised directly on the
loaded objective. Ridge is used where it fits -- to project A1's band values onto the basis as a
warm start. Declared deviation D1.

**Fast exact evaluation.** Adding lambda to every escalating class never changes *which*
escalating or *which* non-escalating class wins, only which group wins. So each row reduces to
``d_i = max_{c not in E} p_c - max_{c in E} p_c``: the row is called escalating iff
``lambda > d_i`` (or ``lambda == d_i`` and the escalating winner has the lower index -- numpy's
argmax tie rule). `--selftest` proves this equals `lambda_rule.apply_lambda` row-for-row.

**The age representation.** HAM records age in 5-year steps (0, 5, ..., 85), so the 5-year bins
[0,5), ..., [75,80), [80, inf) contain exactly one recorded age each except 80+ (80 and 85).
Every arm is frozen as an 18-value table at the recorded ages ``0..85``; curve arms (A3-A6) are
applied by linear interpolation clamped at the ends, bin arms (A2, A7) by bin lookup. Missing age
takes the arm's pooled lambda -- the S5 behaviour -- and **never** an estimated age (see
`research/v4/age_estimator.py`).

**The coverage regime (runbook: "fit on top of whatever S56 establishes").** Fitted at full
coverage. S5, the comparator, was fitted at full coverage, and S56 showed its referral budget does
not transfer off HAM (nominal 20% realises ~40% on reserved), so conditioning lambda on a
referral set whose size is unstable would freeze a non-transferable quantity into the curve.
Composition with S56 is S57b's question. Declared deviation D2.

**Dependency graph (audit item 3)** -- written into `lambda_candidates.json` and asserted by
`nested_cv_panels`:

    Dirichlet_deployed   <- all 6,981 OOF rows          -> scores val, reserved
    Dirichlet_k          <- OOF rows with fold != k     -> scores OOF fold k   (S5 cross-fit)
    lambda(.) (reported) <- Dirichlet_k outputs, all folds
    CV loss for fold k   <- lambda fitted on rows with fold != k, whose probabilities come from
                            an INNER cross-fit over those four folds only; fold k scored by a map
                            fitted on the four folds. Fold k's labels touch neither the
                            calibrator nor lambda that score it.

The *reported* full-data fits consume S5's cross-fitted probabilities, exactly as S5 did, so
their OOF numbers are in-sample with respect to lambda (labelled so). S57b cross-fits lambda.

Reserved is read **only** for audit item 2 (is the deployed 40-59 lambda inside the 0.85 floor
there?), under its own receipt, and only escalation *specificity* is computed -- no under-40
sensitivity, so nothing in S57 can be selected on it.

    $py -m research.v4.lambda_age --selftest
    $py -m research.v4.lambda_age --run                     # phases 1-3 on OOF (+ val descriptive)
    $py -m research.v4.lambda_age --audit-reserved --smoke  # rehearsal on HAM val
    $py -m research.v4.lambda_age --audit-reserved          # the one reserved read (receipt)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "v4"
S57A_DIR = OUT_DIR / "s57a"
DIAGNOSTIC_CSV = OUT_DIR / "age_bin_diagnostic.csv"
BASELINE_JSON = OUT_DIR / "lambda_baseline_s5.json"
CANDIDATES_JSON = OUT_DIR / "lambda_candidates.json"
FIGURE = REPO_ROOT / "paper" / "figures" / "lambda_age_diagnostic.png"
AUDIT_RECEIPT = S57A_DIR / "reserved_audit_receipt.json"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"
SESSION = "v4_s57a"

BIN_WIDTH = 5
TOP_BIN = 80                     # [80, inf) -- "80+"
AGE_KNOTS = np.arange(0.0, 85.0 + 1e-9, 5.0)   # the recorded HAM ages, 18 values
AGE_MAX = 85.0
FLOOR_BANDS = ("<40", "40-59", "60+")
MIN_SPECIFICITY = 0.85           # lambda_rule.py:187
VIOLATION_WEIGHT = 10.0          # hinge on (0.85 - spec_band); cost values are ~0.3
AUC_MIN_POSITIVES = 10           # below this a bin's AUC is "not estimable"
ECE_MIN_N = 200                  # below this a bin's ECE is "not meaningful"
PREVALENCES = (0.01, 0.03, 0.05)
N_BOOT = 400                     # S5's bootstrap count for lambda and AUC
SEED = 20260904                  # S5's seed

KERNEL_BANDWIDTHS = (2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 100.0)
SPLINE_INTERIOR_KNOTS = (17.0, 34.0, 51.0, 68.0)
SPLINE_KAPPAS = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
SHRINK_TAUS = (0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0, float("inf"))
N_STARTS = 6
EPS = 1e-12


# ============================================================================ helpers
def esc_indices() -> list[int]:
    from research.external import frozen_params as fp

    return fp.escalating_indices()


def cost_matrix() -> np.ndarray:
    from ml.paths import load_class_mapping
    from research.thresholds.cost_matrix import build_cost_matrix

    return build_cost_matrix(load_class_mapping())


def age_bin(ages: np.ndarray) -> np.ndarray:
    """5-year bin lower edge, 80 for 80+, -1 for missing age."""
    ages = np.asarray(ages, dtype=float)
    out = np.full(len(ages), -1, dtype=int)
    known = ~np.isnan(ages)
    out[known] = np.minimum((ages[known] // BIN_WIDTH) * BIN_WIDTH, TOP_BIN).astype(int)
    return out


def bin_label(b: int) -> str:
    if b < 0:
        return "unknown"
    return f"{TOP_BIN}+" if b >= TOP_BIN else f"{b}-{b + BIN_WIDTH - 1}"


BINS = tuple(range(0, TOP_BIN + 1, BIN_WIDTH))   # 17 bins


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def softplus_inv(y: np.ndarray) -> np.ndarray:
    y = np.maximum(np.asarray(y, dtype=float), 1e-3)
    return y + np.log(-np.expm1(-y))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, default=_json_default))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(type(obj))


def _clean(obj: Any) -> Any:
    """NaN/inf -> None/str so the JSON is strict."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if np.isnan(f):
            return None
        if np.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ============================================================================ panels
@dataclass
class Panel:
    name: str
    probs: np.ndarray
    y: np.ndarray
    ages: np.ndarray
    lesion: np.ndarray
    bands: np.ndarray
    folds: np.ndarray | None = None
    raw: np.ndarray | None = None
    bins: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.bins = age_bin(self.ages)
        self.lesion = np.asarray(self.lesion).astype(str)
        self.bands = np.asarray(self.bands).astype(str)

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray, probs: np.ndarray | None = None, name: str | None = None) -> "Panel":
        return Panel(name or self.name, self.probs[idx] if probs is None else probs, self.y[idx],
                     self.ages[idx], self.lesion[idx], self.bands[idx],
                     None if self.folds is None else self.folds[idx],
                     None if self.raw is None else self.raw[idx])


def dev_panel() -> Panel:
    """HAM OOF, S5's cross-fitted Dirichlet -- the probabilities the frozen lambdas were fit on."""
    from research.agerule import lambda_rule as lr
    from research.external import frozen_params as fp
    from research.run_session55_multical import _fold_column

    raw = fp.load_ham_oof_panel(calibrate_probs=False)
    folds = _fold_column(fp.OOF_PREDICTIONS_DIR, raw.image_ids)
    probs = lr.crossfit_calibration(raw.probs_raw, raw.y_true, folds)
    return Panel("oof", probs, raw.y_true, raw.ages, raw.lesion_ids, raw.bands, folds, raw.probs_raw)


def val_panel() -> Panel:
    """HAM val, deployed map (fit on OOF). Descriptive only: 22 under-40 escalating cases."""
    from research.ensembling.data import load_split_matrix
    from research.ensembling.methods import soft_vote_arithmetic
    from research.external import frozen_params as fp

    matrix = load_split_matrix("val", predictions_dir="research/predictions_tta")
    probs = fp.calibrate(soft_vote_arithmetic(matrix.probs))
    ages = fp._ages_for(matrix.image_ids)
    return Panel("val", probs, matrix.y_true, ages, matrix.lesion_ids, fp.age_bands(ages))


# ============================================================================ the rule
@dataclass
class Decision:
    """Per-row reduction of the lambda rule (see module docstring)."""
    d: np.ndarray          # max non-escalating p - max escalating p (for display/sorting)
    p_esc: np.ndarray      # max escalating p
    p_non: np.ndarray      # max non-escalating p
    tie_to_esc: np.ndarray  # at lambda == d, numpy's argmax picks the escalating winner
    cost_esc: np.ndarray   # cost if called escalating (the escalating winner)
    cost_non: np.ndarray   # cost if not
    pred_esc_cls: np.ndarray
    pred_non_cls: np.ndarray
    true_esc: np.ndarray

    def flagged(self, lam_row: np.ndarray) -> np.ndarray:
        # the same float expression argmax sees (p - (-lambda) == p + lambda in IEEE), so
        # rounding cannot separate this from apply_lambda
        lifted = self.p_esc + lam_row
        return (lifted > self.p_non) | ((lifted == self.p_non) & self.tie_to_esc)

    def preds(self, lam_row: np.ndarray) -> np.ndarray:
        return np.where(self.flagged(lam_row), self.pred_esc_cls, self.pred_non_cls)


def decision(probs: np.ndarray, y: np.ndarray, esc: list[int], cm: np.ndarray) -> Decision:
    k = probs.shape[1]
    non = [c for c in range(k) if c not in esc]
    esc_arr, non_arr = np.asarray(esc), np.asarray(non)
    e_cls = esc_arr[probs[:, esc_arr].argmax(axis=1)]
    n_cls = non_arr[probs[:, non_arr].argmax(axis=1)]
    rows = np.arange(len(probs))
    d = probs[rows, n_cls] - probs[rows, e_cls]
    return Decision(d=d, p_esc=probs[rows, e_cls], p_non=probs[rows, n_cls], tie_to_esc=e_cls < n_cls, cost_esc=cm[y, e_cls], cost_non=cm[y, n_cls],
                    pred_esc_cls=e_cls, pred_non_cls=n_cls, true_esc=np.isin(y, esc))


class Scorer:
    """Objective + per-band specificity for a per-row lambda, O(N) with no argmax."""

    def __init__(self, panel: Panel, esc: list[int], cm: np.ndarray,
                 min_spec: float = MIN_SPECIFICITY):
        self.panel = panel
        self.min_spec = min_spec
        self.dec = decision(panel.probs, panel.y, esc, cm)
        self.band_codes = {b: i for i, b in enumerate(FLOOR_BANDS)}
        self.band_idx = np.array([self.band_codes.get(b, -1) for b in panel.bands])
        neg = ~self.dec.true_esc
        self.neg_band = [neg & (self.band_idx == i) for i in range(len(FLOOR_BANDS))]
        self.n_neg_band = np.array([m.sum() for m in self.neg_band])
        # recorded ages index into AGE_KNOTS; off-knot ages are never produced by HAM
        known = ~np.isnan(panel.ages)
        self.known = known
        self.knot_idx = np.full(len(panel), -1)
        self.knot_idx[known] = np.clip(np.round(panel.ages[known] / BIN_WIDTH), 0, len(AGE_KNOTS) - 1).astype(int)
        self.on_knots = bool(np.allclose(AGE_KNOTS[self.knot_idx[known]], panel.ages[known]))

    def cost(self, lam_row: np.ndarray) -> float:
        f = self.dec.flagged(lam_row)
        return float(np.where(f, self.dec.cost_esc, self.dec.cost_non).mean())

    def band_spec(self, lam_row: np.ndarray) -> np.ndarray:
        f = self.dec.flagged(lam_row)
        fp_ = np.array([(f & m).sum() for m in self.neg_band])
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.n_neg_band > 0, 1.0 - fp_ / self.n_neg_band, 1.0)

    def objective(self, lam_row: np.ndarray) -> float:
        viol = np.maximum(0.0, self.min_spec - self.band_spec(lam_row)).sum()
        return self.cost(lam_row) + VIOLATION_WEIGHT * viol

    def rows_from_knots(self, knot_values: np.ndarray, pooled: float) -> np.ndarray:
        lam = np.full(len(self.panel), pooled, dtype=float)
        lam[self.known] = knot_values[self.knot_idx[self.known]]
        return lam


# ============================================================================ arms
@dataclass
class Arm:
    """One frozen lambda(age): an 18-value table at AGE_KNOTS plus the missing-age value."""
    name: str
    kind: str                      # "step" (bin lookup) or "interp" (linear, clamped)
    knots: np.ndarray
    pooled: float
    meta: dict[str, Any] = field(default_factory=dict)

    def lam_for(self, ages: np.ndarray) -> np.ndarray:
        ages = np.asarray(ages, dtype=float)
        out = np.full(len(ages), self.pooled, dtype=float)
        known = ~np.isnan(ages)
        if self.kind == "interp":
            out[known] = np.interp(np.clip(ages[known], 0.0, AGE_MAX), AGE_KNOTS, self.knots)
        else:
            b = age_bin(ages[known])
            out[known] = self.knots[np.searchsorted(AGE_KNOTS, b.astype(float))]
        return out

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind,
                "table": {f"{int(a)}": float(v) for a, v in zip(AGE_KNOTS, self.knots)},
                "missing_age_lambda": float(self.pooled), **self.meta}


def band_arm(name: str, lam_by_band: dict[str, float], pooled: float, meta: dict | None = None) -> Arm:
    from research.external import frozen_params as fp

    bands = fp.age_bands(AGE_KNOTS)
    return Arm(name, "step", np.array([lam_by_band.get(b, pooled) for b in bands]), pooled, meta or {})


def arm_a0() -> Arm:
    return Arm("A0", "step", np.zeros(len(AGE_KNOTS)), 0.0, {"mechanism": "argmax"})


def arm_a1() -> Arm:
    """Frozen S5 lambdas, loaded -- the primary comparator."""
    from research.external import frozen_params as fp

    lam = fp.load_lambda_by_band()
    return band_arm("A1", lam, lam["pooled"], {"mechanism": "frozen 3-band lambda (S5)",
                                               "source": fp.LAMBDA_STATE})


def refit_a1(panel: Panel, esc: list[int], cm: np.ndarray) -> Arm:
    """S5's fit, re-run on a panel (CV folds only -- the reported A1 is always the frozen one)."""
    from research.agerule import lambda_rule as lr

    pooled, by = lr.fit_lambda_by_group(panel.probs, panel.y, panel.bands, cm, esc,
                                        min_specificity=MIN_SPECIFICITY, strict=False)
    return band_arm("A1", {b: s.lam for b, s in by.items()}, pooled.lam)


def fit_pooled(panel: Panel, esc: list[int], cm: np.ndarray,
               min_spec: float = MIN_SPECIFICITY) -> float:
    from research.agerule import lambda_rule as lr

    return lr.fit_lambda(panel.probs, panel.y, cm, esc, min_specificity=min_spec,
                         strict=False).lam


def local_fits(panel: Panel, esc: list[int], cm: np.ndarray,
               min_spec: float = MIN_SPECIFICITY) -> pd.DataFrame:
    """Per-bin S5 fit (strict=False), whatever the bin's size. Shrinkage inputs, not results."""
    from research.agerule import lambda_rule as lr

    grid_max = float(np.max(lr.DEFAULT_GRID))
    rows = []
    for b in BINS:
        mask = panel.bins == b
        pos = int(np.isin(panel.y[mask], esc).sum())
        lam = float("nan")
        if pos > 0:
            lam = lr.fit_lambda(panel.probs[mask], panel.y[mask], cm, esc, group=bin_label(b),
                                min_specificity=min_spec, strict=False).lam
        rows.append({"bin": b, "n": int(mask.sum()), "n_positive": pos, "lam_local": lam,
                     "at_grid_max": bool(lam >= grid_max - 1e-12)})
    return pd.DataFrame(rows)


def arm_a2(panel: Panel, esc: list[int], cm: np.ndarray, pooled: float) -> Arm:
    """Independent 5-year lambda with S5's gate; bins below it take the pooled lambda."""
    from research.agerule import lambda_rule as lr

    groups = np.array([bin_label(b) for b in panel.bins])
    _, by = lr.fit_lambda_by_group(panel.probs, panel.y, groups, cm, esc,
                                   min_group_positives=lr.MIN_GROUP_POSITIVES,
                                   min_specificity=MIN_SPECIFICITY, strict=False)
    table = np.array([by[bin_label(age_bin(np.array([a]))[0])].lam
                      if bin_label(age_bin(np.array([a]))[0]) in by else pooled for a in AGE_KNOTS])
    fitted = {k: bool(v.fitted) for k, v in by.items() if k != "unknown"}
    return Arm("A2", "step", table, pooled,
               {"mechanism": "independent 5-year lambda, min-positive fallback to pooled",
                "gate": lr.MIN_GROUP_POSITIVES, "fitted_bins": fitted})


# ---------------------------------------------------------------- A3: kernel
class GridTables:
    """Per-row cost and flag at every grid lambda, for kernel-weighted local fits."""

    def __init__(self, scorer: Scorer):
        from research.agerule import lambda_rule as lr

        self.grid = lr.DEFAULT_GRID
        dec = scorer.dec
        lifted = dec.p_esc[:, None] + self.grid[None, :]
        flags = (lifted > dec.p_non[:, None]) | (
            (lifted == dec.p_non[:, None]) & dec.tie_to_esc[:, None])
        self.cost = np.where(flags, dec.cost_esc[:, None], dec.cost_non[:, None])
        self.fp = flags & ~dec.true_esc[:, None]
        self.neg = ~dec.true_esc
        self.scorer = scorer

    def local_lambda(self, weights: np.ndarray) -> float:
        wsum = weights.sum()
        cost = weights @ self.cost / wsum
        wneg = weights[self.neg].sum()
        spec = 1.0 - (weights @ self.fp) / wneg if wneg > 0 else np.ones(len(self.grid))
        # exactly fit_lambda: start at lambda=0 whatever its specificity, then take a feasible
        # grid point only if it is strictly cheaper (ties -> smaller lambda)
        best_i, best_cost = 0, cost[0]
        for i in np.flatnonzero(spec >= self.scorer.min_spec):
            if cost[i] < best_cost - 1e-12:
                best_i, best_cost = i, cost[i]
        return float(self.grid[best_i])


def arm_a3(tables: GridTables, h: float, pooled: float) -> Arm:
    ages = tables.scorer.panel.ages
    known = ~np.isnan(ages)
    table = np.empty(len(AGE_KNOTS))
    for j, a0 in enumerate(AGE_KNOTS):
        w = np.zeros(len(ages))
        w[known] = np.exp(-0.5 * ((ages[known] - a0) / h) ** 2)
        table[j] = tables.local_lambda(w)
    return Arm("A3", "interp", table, pooled,
               {"mechanism": "kernel-smoothed local cost minimisation (Gaussian)", "bandwidth": h})


# ---------------------------------------------------------------- A4-A6: link-space curves
def basis(kind: str, ages: np.ndarray) -> np.ndarray:
    from scipy.interpolate import BSpline

    a = np.clip(np.asarray(ages, dtype=float), 0.0, AGE_MAX)
    z = (a - 45.0) / 25.0
    if kind == "linear":
        return np.column_stack([np.ones_like(z), z])
    if kind == "quadratic":
        return np.column_stack([np.ones_like(z), z, z ** 2])
    if kind == "spline":
        k = 3
        t = np.r_[[0.0] * (k + 1), list(SPLINE_INTERIOR_KNOTS), [AGE_MAX] * (k + 1)]
        return BSpline.design_matrix(a, t, k).toarray()
    raise ValueError(kind)


def second_difference(p: int) -> np.ndarray:
    return np.diff(np.eye(p), n=2, axis=0)


def fit_link_curve(scorer: Scorer, kind: str, pooled: float, kappa: float = 0.0,
                   warm: np.ndarray | None = None, seed: int = SEED,
                   n_starts: int = N_STARTS) -> tuple[np.ndarray, dict[str, Any]]:
    """Minimise the loaded objective over beta; returns the knot table and fit metadata."""
    from scipy.optimize import minimize
    from sklearn.linear_model import Ridge

    if not scorer.on_knots:
        raise ValueError("development ages are not on the 5-year knots")
    B = basis(kind, AGE_KNOTS)
    p = B.shape[1]
    D = second_difference(p) if kind == "spline" and p > 2 else np.zeros((0, p))

    def knots_of(beta: np.ndarray) -> np.ndarray:
        return softplus(B @ beta)

    def loss(beta: np.ndarray) -> float:
        lam = scorer.rows_from_knots(knots_of(beta), pooled)
        return scorer.objective(lam) + kappa * float(np.sum((D @ beta) ** 2))

    starts = [np.linalg.lstsq(B, np.full(len(AGE_KNOTS), softplus_inv(pooled)), rcond=None)[0]]
    if warm is not None:   # Ridge projection of a knot table (A1's bands) onto the basis
        ridge = Ridge(alpha=1e-3, fit_intercept=False).fit(B, softplus_inv(warm))
        starts.append(ridge.coef_)
    rng = np.random.default_rng(seed)
    n_base = len(starts)
    while len(starts) < n_starts:
        starts.append(starts[len(starts) % n_base] + rng.normal(scale=0.5, size=p))

    best = None
    for s in starts:
        res = minimize(loss, s, method="Powell", options={"xtol": 1e-3, "ftol": 1e-9, "maxfev": 4000})
        res = minimize(loss, res.x, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-10, "maxfev": 3000})
        if best is None or res.fun < best.fun - 1e-12:
            best = res
    table = knots_of(best.x)
    return table, {"beta": best.x.tolist(), "loss": float(best.fun), "kappa": kappa,
                   "basis": kind, "n_starts": n_starts}


def arm_link(name: str, scorer: Scorer, kind: str, pooled: float, kappa: float,
             warm: np.ndarray | None) -> Arm:
    table, meta = fit_link_curve(scorer, kind, pooled, kappa, warm)
    mech = {"linear": "softplus(b0 + b1 z)", "quadratic": "softplus(b0 + b1 z + b2 z^2)",
            "spline": "softplus(cubic B-spline), P-spline ridge penalty"}[kind]
    if kind == "spline":
        meta["interior_knots"] = list(SPLINE_INTERIOR_KNOTS)
    return Arm(name, "interp", table, pooled, {"mechanism": mech, "z": "(age - 45) / 25", **meta})


# ---------------------------------------------------------------- A7: shrinkage
def arm_a7(local: pd.DataFrame, prior: Arm, tau: float, pooled: float) -> Arm:
    shrunk = {}
    rows = []
    for _, r in local.iterrows():
        b = int(r["bin"])
        ages_in_bin = AGE_KNOTS[(AGE_KNOTS >= b) & ((AGE_KNOTS < b + BIN_WIDTH) | (b >= TOP_BIN))]
        prior_b = float(prior.lam_for(ages_in_bin).mean())
        pos = int(r["n_positive"])
        if np.isinf(tau) or pos == 0 or np.isnan(r["lam_local"]):
            alpha = 0.0
        else:
            alpha = pos / (pos + tau) if (pos + tau) > 0 else 1.0
        local_b = 0.0 if np.isnan(r["lam_local"]) else float(r["lam_local"])
        shrunk[b] = alpha * local_b + (1.0 - alpha) * prior_b
        rows.append({"bin": bin_label(b), "n_positive": pos, "alpha": alpha,
                     "lam_local": None if np.isnan(r["lam_local"]) else float(r["lam_local"]),
                     "lam_prior": prior_b, "lam_shrunk": shrunk[b]})
    table = np.array([shrunk[int(age_bin(np.array([a]))[0])] for a in AGE_KNOTS])
    return Arm("A7", "step", table, pooled,
               {"mechanism": "shrunk 5-year lambda toward A6", "tau": tau,
                "alpha_rule": "n_positive / (n_positive + tau)", "bins": rows})


# ---------------------------------------------------------------- floor repair
def repair_floor(arm: Arm, scorer: Scorer, step: float = 0.01) -> Arm:
    """Hold every fitted arm to A1's constraint: escalation specificity >= floor in each band.

    A3's kernel fit enforces the floor on kernel-weighted specificity and A7's shrinkage has no
    constraint at all, so either can leave a coarse band under the floor -- and a cheaper
    objective bought that way is not a fair comparison with A1 (S57a's first run: A3's lower
    CV cost came with 60+ specificity 0.77). The repair lowers the lambda of every knot in a
    violating band by the smallest uniform delta on a 0.01 grid (clipped at 0) that restores the
    floor. Declared deviation D4; the deltas are reported.
    """
    from research.external import frozen_params as fp

    knot_band = fp.age_bands(AGE_KNOTS)
    table = arm.knots.copy()
    deltas = {}
    for i, band in enumerate(FLOOR_BANDS):
        in_band = knot_band == band
        delta = 0.0
        while True:
            trial = table.copy()
            trial[in_band] = np.maximum(0.0, arm.knots[in_band] - delta)
            probe = Arm(arm.name, arm.kind, trial, arm.pooled)
            spec = scorer.band_spec(probe.lam_for(scorer.panel.ages))[i]
            if spec >= scorer.min_spec or not (trial[in_band] > 0).any():
                break
            delta = round(delta + step, 10)
        table = trial
        deltas[band] = delta
    return Arm(arm.name, arm.kind, table, arm.pooled, {**arm.meta, "floor_repair_delta": deltas})


def arm_pooled(pooled: float) -> Arm:
    return Arm("AP", "step", np.full(len(AGE_KNOTS), pooled), pooled,
               {"mechanism": "one pooled lambda (reference outside the family; not floor-repaired)"})


# ============================================================================ evaluation
def evaluate(panel: Panel, lam_row: np.ndarray, esc: list[int], cm: np.ndarray, *,
             intervals: bool = False) -> dict[str, Any]:
    """Exact predictions via `lambda_rule.apply_lambda`, then the S57 metric set."""
    from ml.evaluation.metrics import compute_metrics
    from research.agerule import lambda_rule as lr
    from research.session9 import nnb as nb
    from research.stats.intervals import clopper_pearson
    from research.thresholds.optimize import _escalation_specificity

    preds = np.empty(len(panel), dtype=int)
    for value in np.unique(lam_row):
        m = lam_row == value
        preds[m] = lr.apply_lambda(panel.probs[m], float(value), esc)
    metrics = compute_metrics(panel.y, preds, panel.probs)
    te, pe = np.isin(panel.y, esc), np.isin(preds, esc)

    def block(mask: np.ndarray) -> dict[str, Any]:
        pos = int(te[mask].sum())
        caught = int((te & pe & mask).sum())
        out = {"n": int(mask.sum()), "n_escalating": pos,
               "sensitivity": caught / pos if pos else float("nan"),
               "missed_serious": pos - caught,
               "specificity": _escalation_specificity(panel.y[mask], preds[mask], set(esc)),
               "referral_rate": float(pe[mask].mean()) if mask.any() else float("nan"),
               "expected_cost": float(cm[panel.y[mask], preds[mask]].mean()) if mask.any() else float("nan")}
        if intervals and pos:
            out["sensitivity_cp"] = list(clopper_pearson(caught, pos))
        if pos and mask.any():
            burden = nb.biopsy_burden(panel.y[mask], preds[mask], esc, cohort=panel.name, rule="s57a")
            out.update({f"nnb_pi{p:.2f}": nb.nnb_at(burden, p) for p in PREVALENCES})
        return out

    result = {"all": block(np.ones(len(panel), bool)),
              **{b: block(panel.bands == b) for b in FLOOR_BANDS}}
    result["all"].update(macro_f1=float(metrics["macro_f1"]),
                         balanced_accuracy=float(metrics["balanced_accuracy"]),
                         ece_probs=float(metrics["expected_calibration_error"]))
    return result


# ============================================================================ phase 1
def diagnostic(panel: Panel, esc: list[int], cm: np.ndarray, a1: Arm, n_boot: int) -> pd.DataFrame:
    from ml.evaluation.metrics import expected_calibration_error
    from research.agerule import lambda_rule as lr
    from research.run_session5_agerule import _bootstrap_auc
    from research.stats.intervals import clopper_pearson

    grid_max = float(np.max(lr.DEFAULT_GRID))
    preds = panel.probs.argmax(axis=1)
    te, pe = np.isin(panel.y, esc), np.isin(preds, esc)
    rows = []
    for b in list(BINS) + [-1]:
        m = panel.bins == b
        n, pos = int(m.sum()), int(te[m].sum())
        caught = int((te & pe & m).sum())
        row = {"bin": bin_label(b), "age_lo": b if b >= 0 else None,
               "ages_recorded": ",".join(str(int(a)) for a in np.unique(panel.ages[m])) if b >= 0 else "",
               "n": n, "n_escalating": pos, "prevalence": pos / n if n else float("nan")}
        row["prevalence_cp_lo"], row["prevalence_cp_hi"] = clopper_pearson(pos, n)
        row["argmax_sensitivity"] = caught / pos if pos else float("nan")
        row["sens_cp_lo"], row["sens_cp_hi"] = clopper_pearson(caught, pos) if pos else (np.nan, np.nan)
        row["missed_escalating"] = pos - caught
        if pos >= AUC_MIN_POSITIVES and pos < n:
            row["esc_mass_auc"] = lr.escalation_mass_auc(panel.y[m], panel.probs[m], esc)
            row["auc_ci_lo"], row["auc_ci_hi"] = _bootstrap_auc(
                panel.y[m], panel.probs[m], panel.lesion[m], esc, n_boot)
            row["auc_status"] = "estimated"
        else:
            row.update(esc_mass_auc=np.nan, auc_ci_lo=np.nan, auc_ci_hi=np.nan,
                       auc_status=f"not estimable (<{AUC_MIN_POSITIVES} positives)")
        row["referral_rate_argmax"] = float(pe[m].mean()) if n else float("nan")
        if n >= ECE_MIN_N:
            conf = panel.probs[m].max(axis=1)
            row["ece"] = float(expected_calibration_error(conf, preds[m] == panel.y[m]))
        else:
            row["ece"] = np.nan
        row["lambda_in_force_A1"] = float(a1.lam_for(panel.ages[m][:1])[0]) if n else np.nan
        if b >= 0 and pos >= lr.MIN_GROUP_POSITIVES:
            st = lr.fit_lambda(panel.probs[m], panel.y[m], cm, esc, group=bin_label(b),
                               min_specificity=MIN_SPECIFICITY, strict=False)
            draws = lr.bootstrap_lambda(panel.probs[m], panel.y[m], panel.lesion[m], cm, esc,
                                        n_boot=n_boot, min_specificity=MIN_SPECIFICITY)
            draws = draws[~np.isnan(draws)]
            row.update(optimal_lambda=st.lam, opt_lambda_ci_lo=float(np.percentile(draws, 2.5)),
                       opt_lambda_ci_hi=float(np.percentile(draws, 97.5)),
                       opt_lambda_at_grid_max=bool(st.lam >= grid_max - 1e-12),
                       optimal_lambda_status="estimated")
        else:
            row.update(optimal_lambda=np.nan, opt_lambda_ci_lo=np.nan, opt_lambda_ci_hi=np.nan,
                       opt_lambda_at_grid_max=False,
                       optimal_lambda_status=(f"not estimable (<{lr.MIN_GROUP_POSITIVES} positives)"
                                              if b >= 0 else "missing age -> pooled"))
        rows.append(row)
    return pd.DataFrame(rows)


def audit_gate(panel: Panel, esc: list[int], cm: np.ndarray) -> dict[str, Any]:
    """Audit item 1: how S5's gate and fallback behave on 5-year bins."""
    from research.agerule import lambda_rule as lr

    groups = np.array([bin_label(b) for b in panel.bins])
    pooled, by = lr.fit_lambda_by_group(panel.probs, panel.y, groups, cm, esc,
                                        min_specificity=MIN_SPECIFICITY, strict=False)
    rows = [{"bin": k, "n_positive": s.n_positive, "fitted": s.fitted, "lam": s.lam}
            for k, s in sorted(by.items(), key=lambda kv: (age_bin_from_label(kv[0]), kv[0]))]
    below = [r["bin"] for r in rows if not r["fitted"]]
    under40 = [r for r in rows if 0 <= age_bin_from_label(r["bin"]) < 40]
    return {
        "gate_min_positives": lr.MIN_GROUP_POSITIVES,
        "fallback": "pooled lambda fitted on all rows (fit_lambda_by_group)",
        "pooled_lambda": pooled.lam,
        "bins": rows,
        "bins_below_gate": below,
        "under40_bins_fitted": [r["bin"] for r in under40 if r["fitted"]],
        "under40_bins_fallback": [r["bin"] for r in under40 if not r["fitted"]],
        "reading": (
            f"{len([r for r in under40 if not r['fitted']])} of {len(under40)} under-40 bins fall "
            f"below the gate and take the pooled lambda {pooled.lam:.2f} -- larger than the "
            f"frozen <40 band value, so the fallback gives the sparsest bins the strongest "
            f"correction the data least support."),
    }


def age_bin_from_label(label: str) -> int:
    if label == "unknown":
        return 999
    return int(label.split("-")[0].rstrip("+"))


# ============================================================================ CV (nested)
def nested_cv_panels(dev: Panel) -> list[tuple[Panel, Panel, dict[str, Any]]]:
    """Per outer fold: (train panel with inner-cross-fitted probs, held-out panel, provenance)."""
    from research.agerule import lambda_rule as lr
    from research.calibration.methods import apply_calibration, fit_dirichlet_calibration

    assert dev.raw is not None and dev.folds is not None
    out = []
    for k in np.unique(dev.folds):
        tr, ho = np.flatnonzero(dev.folds != k), np.flatnonzero(dev.folds == k)
        inner = lr.crossfit_calibration(dev.raw[tr], dev.y[tr], dev.folds[tr])
        state = fit_dirichlet_calibration(dev.raw[tr], dev.y[tr])
        held = apply_calibration(state, np.log(np.clip(dev.raw[ho], EPS, None)))
        prov = {"outer_fold": int(k),
                "calibrator_for_heldout_saw_folds": sorted(int(f) for f in np.unique(dev.folds[tr])),
                "inner_calibrators_saw_folds": sorted(int(f) for f in np.unique(dev.folds[tr])),
                "lambda_fit_rows_folds": sorted(int(f) for f in np.unique(dev.folds[tr]))}
        for key in ("calibrator_for_heldout_saw_folds", "inner_calibrators_saw_folds",
                    "lambda_fit_rows_folds"):
            assert int(k) not in prov[key], f"leak: fold {k} in {key}"
        assert not set(dev.lesion[tr]) & set(dev.lesion[ho]), "lesion crosses outer folds"
        out.append((dev.subset(tr, inner, f"cv{k}_train"), dev.subset(ho, held, f"cv{k}_heldout"), prov))
    return out


def cv_loss(folds: list[tuple[Panel, Panel, dict]], fitter: Callable[[Panel, Scorer], Arm],
            esc: list[int], cm: np.ndarray, cache: dict | None = None) -> dict[str, Any]:
    losses, sizes = [], []
    fp_, neg = np.zeros(len(FLOOR_BANDS)), np.zeros(len(FLOOR_BANDS))
    for tr, ho, _ in folds:
        scorer_tr = cache[tr.name] if cache is not None else Scorer(tr, esc, cm)
        arm = fitter(tr, scorer_tr)
        scorer_ho = Scorer(ho, esc, cm)
        lam = arm.lam_for(ho.ages)
        losses.append(scorer_ho.cost(lam))
        sizes.append(len(ho))
        flagged = scorer_ho.dec.flagged(lam)
        fp_ += [(flagged & m).sum() for m in scorer_ho.neg_band]
        neg += scorer_ho.n_neg_band
    losses, sizes = np.array(losses), np.array(sizes)
    mean = float(np.average(losses, weights=sizes))
    se = float(np.std(losses, ddof=1) / np.sqrt(len(losses)))
    return {"cv_cost": mean, "cv_se": se, "per_fold": losses.tolist(),
            "heldout_spec": {b: float(1 - f / n) for b, f, n in zip(FLOOR_BANDS, fp_, neg)}}


def one_se_pick(table: list[dict[str, Any]], key: str, smoother_is_larger: bool = True) -> dict:
    """Smallest CV cost, then the smoothest value within one SE of it (declared rule)."""
    best = min(table, key=lambda r: r["cv_cost"])
    within = [r for r in table if r["cv_cost"] <= best["cv_cost"] + best["cv_se"]]
    vals = [r[key] for r in within]
    pick = max(vals) if smoother_is_larger else min(vals)
    finite = [r[key] for r in table if not np.isinf(r[key])]
    return {"selected": pick, "min_cv_value": best[key], "rule": "1-SE toward smoother",
            "at_grid_boundary": bool(pick == max(finite))}


# ============================================================================ phase 3 driver
def fit_candidates(dev: Panel, esc: list[int], cm: np.ndarray, *, cv: bool = True,
                   verbose: bool = True) -> dict[str, Any]:
    from research.external import frozen_params as fp

    say = print if verbose else (lambda *a, **k: None)
    scorer = Scorer(dev, esc, cm)
    pooled = fit_pooled(dev, esc, cm)
    frozen = fp.load_lambda_by_band()
    a1 = arm_a1()
    a1_table = a1.knots
    local = local_fits(dev, esc, cm)
    tables = GridTables(scorer)

    folds = nested_cv_panels(dev) if cv else []
    cache = {tr.name: Scorer(tr, esc, cm) for tr, _, _ in folds}
    table_cache = {tr.name: GridTables(cache[tr.name]) for tr, _, _ in folds}
    selection: dict[str, Any] = {}

    def pooled_of(p: Panel) -> float:
        return fit_pooled(p, esc, cm)

    pooled_cache = {tr.name: pooled_of(tr) for tr, _, _ in folds}

    # A3 bandwidth
    if cv:
        rows = []
        for h in KERNEL_BANDWIDTHS:
            r = cv_loss(folds, lambda p, s, h=h: repair_floor(
                arm_a3(table_cache[p.name], h, pooled_cache[p.name]), s), esc, cm, cache)
            rows.append({"bandwidth": h, **r})
            say(f"  A3 h={h:5.1f}  cv cost {r['cv_cost']:.5f} +- {r['cv_se']:.5f}")
        selection["A3"] = {"grid": rows, **one_se_pick(rows, "bandwidth")}
    h_star = selection.get("A3", {}).get("selected", 10.0)

    # A6 smoothing
    if cv:
        rows = []
        for kappa in SPLINE_KAPPAS:
            r = cv_loss(folds, lambda p, s, kp=kappa: arm_link(
                "A6", s, "spline", pooled_cache[p.name], kp, a1_table), esc, cm, cache)
            rows.append({"kappa": kappa, **r})
            say(f"  A6 kappa={kappa:g}  cv cost {r['cv_cost']:.5f} +- {r['cv_se']:.5f}")
        selection["A6"] = {"grid": rows, **one_se_pick(rows, "kappa")}
    kappa_star = selection.get("A6", {}).get("selected", 1e-2)

    # A7 tau (A6 prior refit inside each fold at kappa*)
    if cv:
        prior_cache = {tr.name: arm_link("A6", cache[tr.name], "spline", pooled_cache[tr.name],
                                         kappa_star, a1_table) for tr, _, _ in folds}
        local_cache = {tr.name: local_fits(tr, esc, cm) for tr, _, _ in folds}
        rows = []
        for tau in SHRINK_TAUS:
            r = cv_loss(folds, lambda p, s, t=tau: repair_floor(arm_a7(
                local_cache[p.name], prior_cache[p.name], t, pooled_cache[p.name]), s), esc, cm, cache)
            rows.append({"tau": tau, **r})
            say(f"  A7 tau={tau:g}  cv cost {r['cv_cost']:.5f} +- {r['cv_se']:.5f}")
        selection["A7"] = {"grid": rows, **one_se_pick(rows, "tau")}
    tau_star = selection.get("A7", {}).get("selected", 30.0)

    arms = {
        "A0": arm_a0(),
        "A1": a1,
        "A2": arm_a2(dev, esc, cm, pooled),
        "A3": repair_floor(arm_a3(tables, h_star, pooled), scorer),
        "A4": arm_link("A4", scorer, "linear", pooled, 0.0, a1_table),
        "A5": arm_link("A5", scorer, "quadratic", pooled, 0.0, a1_table),
        "A6": arm_link("A6", scorer, "spline", pooled, kappa_star, a1_table),
    }
    arms["A7"] = repair_floor(arm_a7(local, arms["A6"], tau_star, pooled), scorer)
    for name in ("A2", "A4", "A5", "A6"):   # already constrained: the repair is a no-op or is reported
        arms[name] = repair_floor(arms[name], scorer)

    # tau sensitivity on the full development fit (in-sample, descriptive)
    tau_sens = []
    for tau in SHRINK_TAUS:
        arm = repair_floor(arm_a7(local, arms["A6"], tau, pooled), scorer)
        lam = arm.lam_for(dev.ages)
        spec = scorer.band_spec(lam)
        u40 = dev.bands == "<40"
        f = scorer.dec.flagged(lam)
        tau_sens.append({"tau": tau, "in_sample_cost": scorer.cost(lam),
                         "under40_sensitivity": float((f & scorer.dec.true_esc & u40).sum()
                                                      / (scorer.dec.true_esc & u40).sum()),
                         "under40_referral": float(f[u40].mean()),
                         **{f"spec_{b}": float(s) for b, s in zip(FLOOR_BANDS, spec)},
                         "table": arm.knots.tolist()})

    # every arm's CV cost at its selected setting -- a leak-free comparison on the objective
    arm_cv = {}
    if cv:
        fitters: dict[str, Callable[[Panel, Scorer], Arm]] = {
            "AP": lambda p, s: arm_pooled(pooled_cache[p.name]),
            "A0": lambda p, s: arm_a0(),
            "A1": lambda p, s: refit_a1(p, esc, cm),
            "A2": lambda p, s: repair_floor(arm_a2(p, esc, cm, pooled_cache[p.name]), s),
            "A3": lambda p, s: repair_floor(arm_a3(table_cache[p.name], h_star, pooled_cache[p.name]), s),
            "A4": lambda p, s: repair_floor(
                arm_link("A4", s, "linear", pooled_cache[p.name], 0.0, a1_table), s),
            "A5": lambda p, s: repair_floor(
                arm_link("A5", s, "quadratic", pooled_cache[p.name], 0.0, a1_table), s),
            "A6": lambda p, s: repair_floor(prior_cache[p.name], s),
            "A7": lambda p, s: repair_floor(arm_a7(local_cache[p.name], prior_cache[p.name], tau_star,
                                                   pooled_cache[p.name]), s),
        }
        for name, fitter in fitters.items():
            arm_cv[name] = cv_loss(folds, fitter, esc, cm, cache)
            hs = arm_cv[name]["heldout_spec"]
            say(f"  CV {name}: {arm_cv[name]['cv_cost']:.5f} +- {arm_cv[name]['cv_se']:.5f}  held-out spec "
                + " ".join(f"{b} {v:.3f}" for b, v in hs.items()))

    return {
        "pooled_refit": pooled, "pooled_frozen": frozen["pooled"],
        "arms": arms, "local": local, "selection": selection, "tau_sensitivity": tau_sens,
        "arm_cv": arm_cv, "provenance": [p for _, _, p in folds],
        "hyper": {"bandwidth": h_star, "kappa": kappa_star, "tau": tau_star},
    }


# ============================================================================ figure
def plot(diag: pd.DataFrame, arms: dict[str, Arm]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = diag[diag["bin"] != "unknown"].copy()
    x = d["age_lo"].astype(float) + 2.5
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    ax.errorbar(x, d["prevalence"], yerr=[d["prevalence"] - d["prevalence_cp_lo"],
                                          d["prevalence_cp_hi"] - d["prevalence"]],
                fmt="o", color="C0", capsize=2)
    ax.set_title("Escalation prevalence (OOF)")
    ax.set_ylabel("share escalating (Clopper-Pearson 95%)")
    ax = axes[1]
    ax.errorbar(x, d["argmax_sensitivity"],
                yerr=[d["argmax_sensitivity"] - d["sens_cp_lo"], d["sens_cp_hi"] - d["argmax_sensitivity"]],
                fmt="o", color="C1", capsize=2)
    for _, r in d.iterrows():
        ax.annotate(str(int(r["n_escalating"])), (r["age_lo"] + 2.5, 0.02), ha="center", fontsize=7,
                    color="0.4")
    ax.set_ylim(0, 1.05)
    ax.set_title("Argmax escalation sensitivity (n positives at foot)")
    ax = axes[2]
    est = d[d["optimal_lambda_status"] == "estimated"]
    ax.errorbar(est["age_lo"].astype(float) + 2.5, est["optimal_lambda"],
                yerr=[est["optimal_lambda"] - est["opt_lambda_ci_lo"],
                      est["opt_lambda_ci_hi"] - est["optimal_lambda"]],
                fmt="o", color="k", capsize=2, label="bin optimum (>=30 pos, bootstrap 95%)")
    fine = np.linspace(0, AGE_MAX, 200)
    ax.step(AGE_KNOTS, arms["A1"].knots, where="post", color="C3", label="A1 frozen 3-band")
    ax.plot(fine, arms["A6"].lam_for(fine), color="C2", label="A6 spline")
    ax.step(AGE_KNOTS, arms["A7"].knots, where="post", color="C4", ls="--",
            label=f"A7 shrunk (tau={arms['A7'].meta['tau']:g})")
    ax.set_title("Empirical optimal lambda by age (diagnostic)")
    ax.legend(fontsize=7, loc="upper left")
    for a in axes:
        a.set_xlabel("patient age (5-year bin)")
        a.axvline(40, color="0.8", lw=0.8)
        a.axvline(60, color="0.8", lw=0.8)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)


# ============================================================================ ledger
def write_ledger(rows: list[dict[str, Any]], prune_prefix: str) -> None:
    """Prune this session's own rows with this prefix, then append (the S20 hazard)."""
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    frame = pd.DataFrame([{"timestamp": stamp, "session": SESSION, **r} for r in rows])
    old = pd.read_csv(LEDGER_PATH, low_memory=False)
    kept = old[~((old["session"] == SESSION) & old["method"].astype(str).str.startswith(prune_prefix))]
    print(f"ledger: pruned {len(old) - len(kept)} prior {SESSION} row(s), appended {len(frame)}")
    pd.concat([kept, frame], ignore_index=True).to_csv(LEDGER_PATH, index=False)


def _ledger_row(method: str, split: str, ev: dict[str, Any], note: str) -> dict[str, Any]:
    a = ev["all"]
    return {"method": method, "split": split, "macro_f1": a["macro_f1"],
            "balanced_accuracy": a["balanced_accuracy"], "ece": a["ece_probs"],
            "escalation_sens": a["sensitivity"], "missed_serious": a["missed_serious"],
            "notes": note}


# ============================================================================ runs
def run(n_boot: int, cv: bool = True) -> int:
    from research.external import frozen_params as fp
    from research.agerule import lambda_rule as lr

    esc, cm = esc_indices(), cost_matrix()
    dev, val = dev_panel(), val_panel()
    a1 = arm_a1()
    lam_state = REPO_ROOT / fp.LAMBDA_STATE

    # ---- phase 2: baseline, loaded not typed; must reproduce S5's own fitted numbers
    frozen_state = json.loads(lam_state.read_text(encoding="utf-8"))
    base_lam = fp.load_lambda_by_band()
    via_loader = fp.apply_age_rule(dev.probs, bands=dev.bands)
    via_arm = np.empty(len(dev), int)
    lam_rows = a1.lam_for(dev.ages)
    for v in np.unique(lam_rows):
        m = lam_rows == v
        via_arm[m] = lr.apply_lambda(dev.probs[m], float(v), esc)
    assert np.array_equal(via_loader, via_arm), "A1 table disagrees with frozen_params.apply_age_rule"
    repro = {}
    for band, st in frozen_state["by_band"].items():
        m = dev.bands == band
        te, pe = np.isin(dev.y[m], esc), np.isin(via_loader[m], esc)
        sens = float((te & pe).sum() / te.sum())
        repro[band] = {"frozen_fit_sensitivity": st["fit_sensitivity"], "reproduced": sens,
                       "match": abs(sens - st["fit_sensitivity"]) < 1e-12}
    assert all(r["match"] for r in repro.values()), f"S5 baseline not reproduced: {repro}"

    baseline = {
        "session": SESSION, "arm": "A1", "rule": frozen_state["rule"],
        "lambda_by_band": base_lam, "source": fp.LAMBDA_STATE, "source_sha256": sha256(lam_state),
        "loader": "research.external.frozen_params.load_lambda_by_band / apply_age_rule",
        "reproduction_of_s5_fit_sensitivity": repro,
        "oof_crossfitted_in_sample": {"A0_argmax": evaluate(dev, np.zeros(len(dev)), esc, cm, intervals=True),
                                      "A1_frozen": evaluate(dev, lam_rows, esc, cm, intervals=True)},
        "val_heldout_descriptive": {"A0_argmax": evaluate(val, np.zeros(len(val)), esc, cm, intervals=True),
                                    "A1_frozen": evaluate(val, a1.lam_for(val.ages), esc, cm, intervals=True)},
        "notes": ["OOF numbers are in-sample for lambda (S5 fit these rows); val is held out but "
                  "has 22 under-40 escalating cases -- descriptive only.",
                  "ece_probs is the ECE of the calibrated probabilities; the lambda rule does not "
                  "change probabilities, so it is identical across arms.",
                  "NNB counts every escalating call as a biopsy, re-weighted to each prevalence."],
        "comparator_for": "S57b",
    }
    _dump(BASELINE_JSON, _clean(baseline))
    b = baseline["oof_crossfitted_in_sample"]["A1_frozen"]
    print(f"[phase 2] A1 reproduced ({', '.join(f'{k} {v['reproduced']:.4f}' for k, v in repro.items())})")
    print(f"          OOF all sens {b['all']['sensitivity']:.4f}  <40 {b['<40']['sensitivity']:.4f}  "
          f"MF1 {b['all']['macro_f1']:.4f}  spec 40-59 {b['40-59']['specificity']:.4f}")

    # ---- phase 1: diagnostic + audit item 1
    diag = diagnostic(dev, esc, cm, a1, n_boot)
    DIAGNOSTIC_CSV.parent.mkdir(parents=True, exist_ok=True)
    diag.to_csv(DIAGNOSTIC_CSV, index=False, lineterminator="\n")
    gate = audit_gate(dev, esc, cm)
    print(f"[phase 1] {len(diag)} rows -> {rel(DIAGNOSTIC_CSV)}; bins below gate: {gate['bins_below_gate']}")

    # ---- phase 3: candidates
    print("[phase 3] fitting A0-A7 (nested CV for bandwidth / kappa / tau)")
    fits = fit_candidates(dev, esc, cm, cv=cv)
    assert abs(fits["pooled_refit"] - fits["pooled_frozen"]) < 1e-12, "pooled refit != frozen pooled"
    arms: dict[str, Arm] = fits["arms"]

    evals = {}
    for name, arm in arms.items():
        evals[name] = {"oof_in_sample": evaluate(dev, arm.lam_for(dev.ages), esc, cm, intervals=True),
                       "val_descriptive": evaluate(val, arm.lam_for(val.ages), esc, cm, intervals=True)}

    specs_ok = {name: all(evals[name]["oof_in_sample"][b]["specificity"] >= MIN_SPECIFICITY
                          for b in FLOOR_BANDS) for name in arms}
    candidates = {
        "session": SESSION,
        "question": "can the decision layer use patient age more finely than three bands?",
        "objective": {"loss": "expected clinical cost, build_cost_matrix",
                      "constraint": f"escalation specificity >= {MIN_SPECIFICITY} in each of {list(FLOOR_BANDS)}",
                      "constraint_handling": f"hinge penalty x{VIOLATION_WEIGHT} inside the optimiser (A4-A6); "
                                             "hard filter on the grid (A1-A3, A7 local fits)"},
        "age_representation": {"bins": [bin_label(b) for b in BINS], "knots": AGE_KNOTS.tolist(),
                               "curve_application": "linear interpolation clamped to [0, 85]",
                               "bin_application": "5-year lower-edge lookup; 80+ merges 80 and 85",
                               "missing_age": "the arm's pooled lambda (never an estimated age)"},
        "coverage_regime": "full coverage (declared deviation D2, see module docstring)",
        "deviations": {
            "D1": "A6 is optimised on the S5 objective with a P-spline ridge penalty; sklearn Ridge "
                  "is used only for the warm start (no squared-error response exists).",
            "D2": "fitted at full coverage, not on S56's referral regime (budget does not transfer).",
            "D3": "A4/A5 are linear/quadratic in the softplus link, so lambda >= 0 holds.",
            "D4": "every fitted arm is floor-repaired per band (repair_floor) so all arms carry A1's "
                  "constraint; A3's kernel-weighted floor and A7's unconstrained shrinkage do not.",
            "D5": "AP (one pooled lambda) is a CV reference outside the family, not floor-repaired.",
        },
        "dependency_graph": {
            "Dirichlet_deployed": "fit on all OOF rows; scores val and reserved",
            "Dirichlet_k": "fit on OOF folds != k; scores OOF fold k (S5 cross-fit)",
            "lambda_reported": "fit on Dirichlet_k outputs over all folds (in-sample for lambda)",
            "cv_fold_k": "lambda fit on folds != k with an inner cross-fit over those folds; fold k "
                         "scored by a map fit on folds != k",
            "asserted": fits["provenance"],
            "for_S57b": "cross-fit lambda itself with this nesting; no row may be scored by an age "
                        "function or calibrator fitted on it",
        },
        "audit_item_1_gate_on_5_year_bins": gate,
        "hyperparameters": fits["hyper"],
        "selection": fits["selection"],
        "cv_cost_by_arm": fits["arm_cv"],
        "tau_sensitivity": fits["tau_sensitivity"],
        "arms": {name: arm.as_dict() for name, arm in arms.items()},
        "evaluation": evals,
        "specificity_floor_met_in_sample": specs_ok,
        "status": "candidates only -- no gate is read in S57a; S57b cross-fits, bootstraps and freezes",
    }
    _dump(CANDIDATES_JSON, _clean(candidates))
    plot(diag, arms)

    print("\n arm  table(<40 ages 0..35)                         OOF<40  OOF all  MF1     spec(40-59) CV cost")
    for name, arm in arms.items():
        e = evals[name]["oof_in_sample"]
        cvc = fits["arm_cv"].get(name, {}).get("cv_cost", float("nan"))
        print(f" {name}  {np.array2string(arm.knots[:8], precision=2, separator=',')[:44]:44s} "
              f"{e['<40']['sensitivity']:.4f}  {e['all']['sensitivity']:.4f}  {e['all']['macro_f1']:.4f}  "
              f"{e['40-59']['specificity']:.4f}      {cvc:.5f}")

    rows = []
    for name in arms:
        for split, key in (("oof", "oof_in_sample"), ("val", "val_descriptive")):
            e = evals[name][key]
            rows.append(_ledger_row(
                f"S57a_{name}", split, e,
                f"S57a {name} ({arms[name].meta.get('mechanism', '')}); {key}; <40 sens "
                f"{e['<40']['sensitivity']:.4f} referral {e['<40']['referral_rate']:.4f}; "
                f"spec min band {min(e[b]['specificity'] for b in FLOOR_BANDS):.4f}; candidates only"))
    write_ledger(rows, prune_prefix="S57a_A")
    return 0


# ---------------------------------------------------------------- audit item 2 (reserved)
def audit_reserved(rerun_reason: str | None, smoke: bool) -> int:
    """Is the deployed 40-59 lambda inside the 0.85 specificity floor on reserved? Specificity only."""
    from research.external import frozen_params as fp
    from research.stats.intervals import clopper_pearson
    from research.v4.s54_guard import git_head, now

    esc = esc_indices()
    lam_state = REPO_ROOT / fp.LAMBDA_STATE
    out_path = (S57A_DIR / "smoke" / "reserved_audit.json") if smoke else (S57A_DIR / "reserved_audit.json")
    receipt = None
    if not smoke:
        receipt = (json.loads(AUDIT_RECEIPT.read_text(encoding="utf-8")) if AUDIT_RECEIPT.is_file()
                   else {"cohort": "manifest_v4 split=reserved (S57a audit item 2)",
                         "declared_quantities": ["escalation specificity per band, A0 and A1"],
                         "lambda_sha256": sha256(lam_state), "executions": []})
        if receipt["lambda_sha256"] != sha256(lam_state):
            raise SystemExit("frozen lambda file changed since the audit was declared")
        if receipt["executions"] and not rerun_reason:
            raise SystemExit("S57a has already read reserved for the audit; a repeat needs --rerun-reason")
        receipt["executions"].append({"execution": len(receipt["executions"]) + 1, "status": "started",
                                      "started_at": now(), "rerun_reason": rerun_reason,
                                      "git_head": git_head()})
        _dump(AUDIT_RECEIPT, receipt)
        from research.v4.s56_abstention import reserved_panel

        panel, meta = reserved_panel()
        probs, y, bands, name = panel.probs, panel.y7, panel.bands, "reserved"
    else:
        val = val_panel()
        probs, y, bands, name, meta = val.probs, val.y, val.bands, "ham_val (smoke)", {}

    rows = {}
    for arm, preds in (("A0", probs.argmax(axis=1)), ("A1", fp.apply_age_rule(probs, bands=bands))):
        pe = np.isin(preds, esc)
        neg = ~np.isin(y, esc)
        for band in ("all",) + FLOOR_BANDS:
            m = neg & (np.ones(len(y), bool) if band == "all" else bands == band)
            tn = int((m & ~pe).sum())
            rows[f"{arm}|{band}"] = {"arm": arm, "band": band, "negatives": int(m.sum()),
                                     "specificity": tn / m.sum() if m.sum() else float("nan"),
                                     "cp": list(clopper_pearson(tn, int(m.sum())))}
    s = rows["A1|40-59"]
    verdict = ("floor met" if s["specificity"] >= MIN_SPECIFICITY
               else "FLOOR VIOLATED -- a finding about the deployed rule, not a reason to relax the floor")
    payload = {"panel": name, "lambda_by_band": fp.load_lambda_by_band(), "min_specificity": MIN_SPECIFICITY,
               "rows": list(rows.values()), "deployed_40_59": {**s, "verdict": verdict},
               "not_computed": "no sensitivity of any kind (S57 must not select on reserved)",
               "panel_meta": meta}
    _dump(out_path, _clean(payload))
    print(f"[audit 2] {name}: A1 40-59 specificity {s['specificity']:.4f} "
          f"[{s['cp'][0]:.4f}, {s['cp'][1]:.4f}] -> {verdict}")
    for r in rows.values():
        print(f"   {r['arm']} {r['band']:6s} spec {r['specificity']:.4f}  (neg {r['negatives']})")
    if receipt is not None:
        receipt["executions"][-1].update(status="completed", completed_at=now(),
                                         items={rel(out_path): sha256(out_path)})
        _dump(AUDIT_RECEIPT, receipt)
        write_ledger([{"method": "S57a_audit_reserved_spec_40_59", "split": "reserved",
                       "notes": f"S57a audit 2: deployed A1 lambda 40-59 escalation specificity "
                                f"{s['specificity']:.4f} [{s['cp'][0]:.4f}, {s['cp'][1]:.4f}] vs floor "
                                f"{MIN_SPECIFICITY} -> {verdict.split(' --')[0]}; A0 {rows['A0|40-59']['specificity']:.4f}"}],
                     prune_prefix="S57a_audit")
    return 0


# ============================================================================ selftest
def synthetic(truth: Callable[[np.ndarray], np.ndarray], n_per_age: int, seed: int,
              slope: float = 25.0) -> Panel:
    """Two-class world (mel vs nv) where the cost-optimal lambda at age a is truth(a) exactly.

    With p = (p_mel, p_nv) and d = p_nv - p_mel, calling mel costs C[nv, mel] = 1 on a nevus and
    calling nv costs C[mel, nv] = 10 on a melanoma, so flagging pays iff P(mel | d) > 1/11. Setting
    P(mel | d, a) = sigmoid(s (truth(a) - d) + logit(1/11)) puts the boundary at d = truth(a).
    """
    rng = np.random.default_rng(seed)
    ages = np.repeat(AGE_KNOTS, n_per_age)
    d = rng.uniform(-1.0, 1.0, size=len(ages))
    p_mel = (1.0 - d) / 2.0
    logit_q = np.log((1 / 11) / (1 - 1 / 11))
    prob = 1.0 / (1.0 + np.exp(-(slope * (truth(ages) - d) + logit_q)))
    is_mel = rng.uniform(size=len(ages)) < prob
    probs = np.zeros((len(ages), 7))
    probs[:, 4], probs[:, 5] = p_mel, 1.0 - p_mel
    y = np.where(is_mel, 4, 5)
    from research.external import frozen_params as fp

    return Panel("synthetic", probs, y, ages.astype(float), np.arange(len(ages)).astype(str),
                 fp.age_bands(ages), folds=np.arange(len(ages)) % 5)


def selftest() -> int:
    from research.agerule import lambda_rule as lr
    from research.external import frozen_params as fp

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    esc, cm = esc_indices(), cost_matrix()
    rng = np.random.default_rng(0)

    # 1. the d-reduction equals apply_lambda row for row, including exact ties
    p = rng.dirichlet(np.ones(7) * 0.5, size=4000)
    p[:50] = np.round(p[:50], 1)
    p[:50] /= p[:50].sum(axis=1, keepdims=True)
    y = rng.integers(0, 7, 4000)
    dec = decision(p, y, esc, cm)
    bad = 0
    for lam in lr.DEFAULT_GRID:
        bad += int((dec.preds(np.full(4000, lam)) != lr.apply_lambda(p, float(lam), esc)).sum())
    check("d-reduction == apply_lambda", bad == 0, f"{bad} disagreements over {len(lr.DEFAULT_GRID)} lambdas")

    # 2. bins
    b = age_bin(np.array([0, 4, 5, 39, 40, 79, 80, 85, 90, np.nan]))
    check("age_bin", b.tolist() == [0, 0, 5, 35, 40, 75, 80, 80, 80, -1], str(b.tolist()))

    # 3. A1 table == frozen_params, and == apply_age_rule on random ages
    a1 = arm_a1()
    ages = rng.choice(np.r_[AGE_KNOTS, np.nan], size=3000)
    lam_rows = a1.lam_for(ages)
    via = np.empty(3000, int)
    for v in np.unique(lam_rows):
        m = lam_rows == v
        via[m] = lr.apply_lambda(p[:3000][m], float(v), esc)
    check("A1 == frozen_params.apply_age_rule",
          np.array_equal(via, fp.apply_age_rule(p[:3000], ages)), "3000 random rows incl. missing age")

    # 4. no typed lambda: A1 values come from the loader
    lam = fp.load_lambda_by_band()
    check("A1 values loaded", set(np.round(a1.knots, 6)) == {round(lam[k], 6) for k in FLOOR_BANDS}
          and a1.pooled == lam["pooled"], f"{sorted(set(a1.knots))} pooled {a1.pooled}")

    # 5-7. synthetic recovery of a known curve
    truth = lambda a: 0.15 + 0.5 * np.exp(-((np.asarray(a) - 60.0) / 18.0) ** 2)  # noqa: E731
    syn = synthetic(truth, 1500, seed=1)
    # with the floor on, a synthetic world's optimum is the floor's, not the truth's -- the
    # floor is checked on real data (check 3 of `run`), recovery is checked with it off
    scorer = Scorer(syn, esc, cm, min_spec=0.0)
    check("synthetic ages on knots", scorer.on_knots, "")
    pooled = fit_pooled(syn, esc, cm, 0.0)
    # kernel with an enormous bandwidth must reproduce the pooled S5 fit, floor on and off
    for ms in (0.0, MIN_SPECIFICITY):
        sc = Scorer(syn, esc, cm, min_spec=ms)
        wide = arm_a3(GridTables(sc), 1e6, 0.0)
        ref = fit_pooled(syn, esc, cm, ms)
        check(f"A3(h=inf) == pooled fit_lambda (floor {ms})", np.allclose(wide.knots, ref),
              f"{wide.knots[0]} vs {ref}")
    a6_table, _ = fit_link_curve(scorer, "spline", pooled, 1e-4, None, n_starts=4)
    boots = []
    brng = np.random.default_rng(7)
    for _ in range(30):
        idx = np.sort(brng.choice(len(syn), size=len(syn), replace=True))
        sb = syn.subset(idx)
        t, _ = fit_link_curve(Scorer(sb, esc, cm, min_spec=0.0), "spline",
                              fit_pooled(sb, esc, cm, 0.0), 1e-4, None, n_starts=2)
        boots.append(t)
    boots = np.array(boots)
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    true_k = truth(AGE_KNOTS)
    inside = int(((true_k >= lo - 0.02) & (true_k <= hi + 0.02)).sum())
    check("A6 recovers the generating curve inside its bootstrap band", inside >= 15,
          f"{inside}/18 knots inside (+-0.02 grid tolerance); max |err| "
          f"{np.abs(a6_table - true_k).max():.3f}")
    local = local_fits(syn, esc, cm, 0.0)
    prior = Arm("A6", "interp", a6_table, pooled)
    a7 = arm_a7(local, prior, 50.0, pooled)
    check("A7 close to the truth", np.abs(a7.knots - true_k).max() < 0.12,
          f"max |err| {np.abs(a7.knots - true_k).max():.3f}")
    a7_inf = arm_a7(local, prior, float("inf"), pooled)
    a7_zero = arm_a7(local, prior, 0.0, pooled)
    check("A7 limits: tau=inf -> prior, tau=0 -> local",
          np.allclose(a7_inf.knots, [prior.lam_for(AGE_KNOTS[(AGE_KNOTS >= age_bin(np.array([a]))[0])
                                                              & ((AGE_KNOTS < age_bin(np.array([a]))[0] + 5)
                                                                 | (a >= 80))]).mean() for a in AGE_KNOTS])
          and np.allclose(a7_zero.knots, [local.set_index("bin").loc[age_bin(np.array([a]))[0], "lam_local"]
                                          for a in AGE_KNOTS]), "")

    # 8. nested CV asserts disjointness (real OOF folds)
    dev = dev_panel()
    dev_scorer = Scorer(dev, esc, cm)
    greedy = Arm("x", "interp", np.full(len(AGE_KNOTS), 1.2), 0.65)
    fixed = repair_floor(greedy, dev_scorer)
    spec_before = dev_scorer.band_spec(greedy.lam_for(dev.ages))
    spec_after = dev_scorer.band_spec(fixed.lam_for(dev.ages))
    check("repair_floor restores the per-band floor",
          (spec_before < MIN_SPECIFICITY).any() and (spec_after >= MIN_SPECIFICITY).all(),
          f"before {np.round(spec_before, 3)} after {np.round(spec_after, 3)} "
          f"deltas {fixed.meta['floor_repair_delta']}")
    ok_arm = arm_a0()
    check("repair_floor is a no-op on a feasible arm",
          np.array_equal(repair_floor(ok_arm, dev_scorer).knots, ok_arm.knots), "A0")
    folds = nested_cv_panels(dev)
    check("nested CV provenance", len(folds) == 5 and all(
        f[2]["outer_fold"] not in f[2]["calibrator_for_heldout_saw_folds"] for f in folds), "5 outer folds")

    # 9. missing age never receives a curve value
    arm = Arm("x", "interp", np.linspace(0, 1, 18), 0.42)
    check("missing age -> pooled", arm.lam_for(np.array([np.nan]))[0] == 0.42, "")

    print("SELFTEST PASSED" if not failures else f"SELFTEST FAILED: {failures}")
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--run", action="store_true", help="phases 1-3 on OOF, val descriptive")
    parser.add_argument("--no-cv", action="store_true", help="skip nested CV (debug only)")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--audit-reserved", action="store_true", help="audit item 2 (receipt-guarded)")
    parser.add_argument("--smoke", action="store_true", help="with --audit-reserved: rehearse on HAM val")
    parser.add_argument("--rerun-reason", default=None)
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.run:
        return run(args.n_boot, cv=not args.no_cv)
    if args.audit_reserved:
        return audit_reserved(args.rerun_reason, args.smoke)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
