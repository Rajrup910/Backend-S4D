"""S31 -- the minimal scalar core the whole V2 decomposition rests on.

This module holds only pure functions over arrays: the scores, the exact-integer referral
construction, and the four terms of the budget-conditional decomposition. It deliberately
contains no cross-fitting, no bootstrap, no cohort loading and no I/O -- those belong to
`frontier.py` (S32) and `decompose.py` (S33), which build on these primitives rather than
reimplementing them.

It lives in S31 rather than S33 because the synthetic validation suite has to have
something to validate, and a suite that tested a *second* implementation of the same maths
would be checking that two copies agree, not that the maths is right.

**The decomposition** (blueprint 6.1), all evaluated at one common referral count r:

    S_eta(r) - S_argmax  =  A(r) + B(r) + C(r)

    A(r) = S_eta(r)   - S_ustar(r)    ranking deficit beyond the score library
    B(r) = S_ustar(r) - S_s(r)        score-choice gap, >= 0 because s is in the library
    C(r) = S_s(r)     - S_argmax      decision gap, **SIGNED**

`S_eta` needs the true escalation posterior and is therefore computable **only on
synthetic data**. On real cohorts A is not identified and only its lower bound (B) is
reported -- which is the entire reason the certified-bound machinery exists.

**Why A can come out slightly negative in a finite sample.** `S_eta >= S_ustar` is a
population statement (Neyman-Pearson): ranking by the true posterior maximises expected
captured positives among x-measurable rankings. In a finite sample a noisier score can beat
eta by luck, so a small negative A is sampling noise, not a broken identity. The identity
`A + B + C == S_eta - S_argmax` holds exactly regardless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from research.external.frozen_params import escalating_indices
from research.selective.scores import (
    ensemble_variance,
    max_softmax,
    predictive_entropy,
    top_two_margin,
)

SEED = 42


# ------------------------------------------------------------------------------- scores
def escalation_mass(probs: np.ndarray, esc: list[int]) -> np.ndarray:
    """s(x) = sum of probability over the escalating classes."""
    return np.asarray(probs)[:, esc].sum(axis=1)


def escalation_margin(probs: np.ndarray, esc: list[int]) -> np.ndarray:
    """d(x) = max_{c in E} p_c - max_{c not in E} p_c.

    The object argmax thresholds at 0, and the object the frozen age/lambda rule
    thresholds at -lambda (blueprint 6.4). Distinct from `top_two_margin`, which is
    top-1-vs-top-2 over all classes and knows nothing about E.
    """
    probs = np.asarray(probs)
    non_esc = [c for c in range(probs.shape[1]) if c not in esc]
    return probs[:, esc].max(axis=1) - probs[:, non_esc].max(axis=1)


def argmax_refers(probs: np.ndarray, esc: list[int]) -> np.ndarray:
    """The deployed rule: refer iff the top-1 class is an escalating one."""
    return np.isin(np.asarray(probs).argmax(axis=1), esc)


def build_score_library(
    probs: np.ndarray,
    esc: list[int],
    member_probs: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """The frozen library U from `results/v2/analysis_plan.json`.

    Every score is returned in its natural direction, higher = referred first. No score is
    ever sign-flipped: because B is a max over the library, a badly-oriented candidate can
    only fail to win, never inflate the bound (plan: score_library.orientation_rule).

    `disagreement` requires the per-member (N, K, C) tensor and is omitted when it is not
    supplied -- the same availability constraint the plan records for BCN/MSKCC.
    """
    probs = np.asarray(probs)
    library = {
        "s": escalation_mass(probs, esc),
        "d": escalation_margin(probs, esc),
        "msp": max_softmax(probs),
        "entropy": predictive_entropy(probs),
        "margin": top_two_margin(probs),
    }
    if member_probs is not None:
        library["disagreement"] = ensemble_variance(np.asarray(member_probs))
    return library


# ------------------------------------------------------------------- referral policies
def top_r_refers(scores: np.ndarray, r: int, seed: int = SEED) -> np.ndarray:
    """Refer exactly `r` cases, the highest-scoring ones, ties broken by a seeded
    permutation (blueprint 9's exact-integer construction).

    Exactness matters: it is what makes a McNemar test on paired referral indicators
    valid, and what keeps "natural argmax" and "budget-matched" from being conflated.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    r = int(np.clip(r, 0, n))
    mask = np.zeros(n, dtype=bool)
    if r == 0:
        return mask
    tiebreak = np.random.default_rng(seed).permutation(n)
    order = np.lexsort((tiebreak, -scores))  # primary: score desc; secondary: seeded
    mask[order[:r]] = True
    return mask


def sensitivity(refers: np.ndarray, y_esc: np.ndarray) -> float:
    """Fraction of truly escalating cases that get referred."""
    y_esc = np.asarray(y_esc, dtype=bool)
    if y_esc.sum() == 0:
        return float("nan")
    return float(np.asarray(refers, dtype=bool)[y_esc].mean())


def frozen_threshold_refers(scores: np.ndarray, threshold: float) -> np.ndarray:
    """A deployable policy: an absolute cutoff fitted elsewhere, applied unchanged.

    Unlike `top_r_refers` this does NOT control the referral count on the target cohort --
    that is the whole point. Its realized burden is an outcome, and the gap between its
    sensitivity and the oracle's at that same burden is the transport term T.
    """
    return np.asarray(scores, dtype=float) >= float(threshold)


def threshold_for_rate(scores: np.ndarray, rate: float) -> float:
    """The absolute cutoff that refers `rate` of a development cohort."""
    return float(np.quantile(np.asarray(scores, dtype=float), 1.0 - float(rate)))


# ------------------------------------------------------------------------ decomposition
@dataclass(frozen=True)
class Decomposition:
    r: int
    n: int
    n_escalating: int
    burden: float
    s_argmax: float
    s_mass: float
    s_ustar: float
    s_eta: float | None
    best_score: str
    A: float | None
    B: float
    C: float
    identity_residual: float | None
    per_score_sensitivity: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "r": self.r, "n": self.n, "n_escalating": self.n_escalating,
            "burden": self.burden, "S_argmax": self.s_argmax, "S_mass": self.s_mass,
            "S_ustar": self.s_ustar, "S_eta": self.s_eta, "best_score": self.best_score,
            "A": self.A, "B": self.B, "C": self.C,
            "identity_residual": self.identity_residual,
        }
        for name, value in self.per_score_sensitivity.items():
            d[f"S_{name}"] = value
        return d


def decompose_at_argmax_budget(
    probs: np.ndarray,
    y_esc: np.ndarray,
    esc: list[int] | None = None,
    member_probs: np.ndarray | None = None,
    eta: np.ndarray | None = None,
    seed: int = SEED,
) -> Decomposition:
    """Decompose the safety gap at argmax's own natural referral count.

    `eta` (the true escalation posterior) is optional and available only on synthetic
    data; when omitted, `A` and `identity_residual` are None and only the certified lower
    bound `B` is reported -- exactly the real-data situation.
    """
    probs = np.asarray(probs)
    y_esc = np.asarray(y_esc, dtype=bool)
    esc = escalating_indices() if esc is None else esc

    refers_argmax = argmax_refers(probs, esc)
    r = int(refers_argmax.sum())
    n = len(probs)

    library = build_score_library(probs, esc, member_probs)
    per_score = {
        name: sensitivity(top_r_refers(score, r, seed), y_esc)
        for name, score in library.items()
    }

    s_argmax = sensitivity(refers_argmax, y_esc)
    s_mass = per_score["s"]
    best_score = max(per_score, key=lambda k: per_score[k])
    s_ustar = per_score[best_score]

    s_eta = None if eta is None else sensitivity(top_r_refers(eta, r, seed), y_esc)
    A = None if s_eta is None else s_eta - s_ustar
    B = s_ustar - s_mass
    C = s_mass - s_argmax
    residual = None if s_eta is None else (A + B + C) - (s_eta - s_argmax)

    return Decomposition(
        r=r, n=n, n_escalating=int(y_esc.sum()), burden=r / n if n else float("nan"),
        s_argmax=s_argmax, s_mass=s_mass, s_ustar=s_ustar, s_eta=s_eta,
        best_score=best_score, A=A, B=B, C=C, identity_residual=residual,
        per_score_sensitivity=per_score,
    )


def transport_term(
    dev_scores: np.ndarray,
    target_scores: np.ndarray,
    target_y_esc: np.ndarray,
    dev_rate: float,
    seed: int = SEED,
) -> dict:
    """Split transport failure into the part recalibration causes and the part it cannot.

    **Correction found by the S31 synthetic suite.** Blueprint revision 2 defined a single
    transport term `T = S_oracle(at the realized burden) - S_frozen` and claimed monotone
    recalibration acts "only through T". That is false, and provably so: a threshold on a
    strictly monotone transform of a score selects *the same set* as a threshold on the
    original score at the corresponding cutoff -- it is still a top-k set, just at a
    different k. So it lies exactly ON the oracle frontier and `T` is identically zero.
    Verified directly: the two referral masks are element-wise equal.

    The real damage from recalibration is therefore **budget mis-targeting**, not lost
    sensitivity, and the two quantities below separate cleanly:

      `T_matched`    = S_oracle(r_realized) - S_frozen.  Zero under ANY monotone
                       recalibration; positive only when the score *reorders* cases
                       relative to the development cohort. It is thus a diagnostic for
                       *which kind* of transport failure occurred -- recalibration
                       (T_matched == 0) versus genuine reordering (T_matched > 0).
      `burden_error` = realized burden - intended burden. Signed. This is what a
                       miscalibrated score actually costs a clinic: you budgeted 10% of
                       capacity and spent 3.7%, or 30%.
      `T_intended`   = S_oracle(r_intended) - S_frozen. Signed: negative means the frozen
                       rule over-referred and bought extra sensitivity it had not budgeted.
    """
    n = len(target_scores)
    tau = threshold_for_rate(dev_scores, dev_rate)
    refers_frozen = frozen_threshold_refers(target_scores, tau)
    r_realized = int(refers_frozen.sum())
    r_intended = int(round(dev_rate * n))

    s_frozen = sensitivity(refers_frozen, target_y_esc)
    s_oracle_realized = sensitivity(top_r_refers(target_scores, r_realized, seed), target_y_esc)
    s_oracle_intended = sensitivity(top_r_refers(target_scores, r_intended, seed), target_y_esc)

    return {
        "tau": tau,
        "r_intended": r_intended,
        "r_realized": r_realized,
        "burden_intended": dev_rate,
        "burden_realized": r_realized / n if n else float("nan"),
        "burden_error": (r_realized / n - dev_rate) if n else float("nan"),
        "S_frozen": s_frozen,
        "S_oracle_at_realized": s_oracle_realized,
        "S_oracle_at_intended": s_oracle_intended,
        "T_matched": s_oracle_realized - s_frozen,
        "T_intended": s_oracle_intended - s_frozen,
    }


def rescue_rate(
    probs: np.ndarray,
    y_esc: np.ndarray,
    uncertainty: np.ndarray,
    esc: list[int] | None = None,
    seed: int = SEED,
) -> dict:
    """Of the escalating cases argmax misses, what fraction does an uncertainty-ranked
    policy refer at argmax's own budget?

    Terminology is the directive's (section 23): a correctly-referred case is a **rescue**,
    never a "miss-referral rate".
    """
    probs = np.asarray(probs)
    y_esc = np.asarray(y_esc, dtype=bool)
    esc = escalating_indices() if esc is None else esc

    refers_argmax = argmax_refers(probs, esc)
    r = int(refers_argmax.sum())
    missed = y_esc & ~refers_argmax
    refers_unc = top_r_refers(uncertainty, r, seed)
    n_missed = int(missed.sum())
    return {
        "n_missed_by_argmax": n_missed,
        "n_rescued": int((missed & refers_unc).sum()),
        "rescue_rate": float((missed & refers_unc).sum() / n_missed) if n_missed else float("nan"),
    }
