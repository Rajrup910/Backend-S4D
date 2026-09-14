"""S32 -- policy construction and labeling.

Builds on `research.v2.estimators` (S31) rather than reimplementing referral
construction. This module's job is bookkeeping, not new maths: every policy comparison
this project runs must carry `threshold_type` / `fit_cohort` / `eval_cohort`, and the
frozen plan (S30, `budget_rule`) is explicit that these three fields are the entire
difference between "how much information is in a score" (oracle) and "does the policy
transport" (frozen). Mixing the two in one table without this column is an audit failure
(blueprint 9, 33), so it is enforced at the data-structure level rather than left to each
caller to remember.

Three policy kinds:

  natural            -- argmax's own rule, at its own naturally-occurring burden.
  oracle_evaluation  -- refer the top-r by some score, r fixed by argmax's burden ON THE
                        SAME cohort. Answers "how much information is in this score".
  frozen_deployable   -- a cutoff fitted on one (development) cohort, applied UNCHANGED to
                        a different (target) cohort. Answers "does it transport".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from research.v2 import estimators as est

THRESHOLD_TYPES = ("natural", "oracle_evaluation", "frozen_deployable")


@dataclass(frozen=True)
class PolicyResult:
    threshold_type: str
    score_name: str
    eval_cohort: str
    fit_cohort: str | None    # None only for "natural"; oracle_evaluation sets it == eval_cohort
    subgroup: str
    r: int
    n: int
    n_escalating: int
    burden: float
    sensitivity: float
    seed: int                 # -1 where no seeded tie-break was needed (natural, frozen)

    def as_dict(self) -> dict:
        return asdict(self)


def _burden(r: int, n: int) -> float:
    return r / n if n else float("nan")


def natural_argmax(
    probs: np.ndarray, y_esc: np.ndarray, esc: list[int], *,
    eval_cohort: str, subgroup: str = "ALL",
) -> PolicyResult:
    """The deployed rule, unmatched to any budget -- its burden is whatever it is."""
    refers = est.argmax_refers(probs, esc)
    r = int(refers.sum())
    return PolicyResult(
        threshold_type="natural", score_name="argmax", eval_cohort=eval_cohort,
        fit_cohort=None, subgroup=subgroup, r=r, n=len(probs),
        n_escalating=int(np.asarray(y_esc, dtype=bool).sum()), burden=_burden(r, len(probs)),
        sensitivity=est.sensitivity(refers, y_esc), seed=-1,
    )


def oracle_top_r(
    scores: np.ndarray, y_esc: np.ndarray, r: int, *,
    score_name: str, eval_cohort: str, subgroup: str = "ALL", seed: int = est.SEED,
) -> PolicyResult:
    """Refer exactly `r` cases by `scores`, on the SAME cohort the scores come from.

    `fit_cohort == eval_cohort` by construction -- an oracle result answers "how much
    information is in the score", never "does it transport" (blueprint 9, 15).
    """
    refers = est.top_r_refers(scores, r, seed)
    r_actual = int(refers.sum())
    if r_actual != min(r, len(scores)):
        raise ValueError(
            f"oracle_top_r for {score_name!r}: requested r={r}, referred {r_actual} -- "
            f"the exact-integer construction must be exact."
        )
    return PolicyResult(
        threshold_type="oracle_evaluation", score_name=score_name, eval_cohort=eval_cohort,
        fit_cohort=eval_cohort, subgroup=subgroup, r=r_actual, n=len(scores),
        n_escalating=int(np.asarray(y_esc, dtype=bool).sum()), burden=_burden(r_actual, len(scores)),
        sensitivity=est.sensitivity(refers, y_esc), seed=seed,
    )


def frozen_deployable(
    dev_scores: np.ndarray, dev_rate: float, eval_scores: np.ndarray, y_esc: np.ndarray, *,
    score_name: str, fit_cohort: str, eval_cohort: str, subgroup: str = "ALL",
) -> PolicyResult:
    """A cutoff fitted on `dev_scores` at `dev_rate`, applied UNCHANGED to `eval_scores`.

    The realized burden on the target cohort is an OUTCOME, not a control -- it is not
    forced to equal `dev_rate`. That gap is exactly what the transport term measures
    (`research.v2.estimators.transport_term`); this function only produces the labeled
    policy row, it does not compute the oracle comparison.
    """
    if fit_cohort == eval_cohort:
        raise ValueError(
            f"frozen_deployable requires fit_cohort != eval_cohort (got {fit_cohort!r} "
            f"twice) -- applying a cohort's own threshold to itself is oracle_evaluation, "
            f"not a transport claim."
        )
    tau = est.threshold_for_rate(dev_scores, dev_rate)
    refers = est.frozen_threshold_refers(eval_scores, tau)
    r = int(refers.sum())
    return PolicyResult(
        threshold_type="frozen_deployable", score_name=score_name, eval_cohort=eval_cohort,
        fit_cohort=fit_cohort, subgroup=subgroup, r=r, n=len(eval_scores),
        n_escalating=int(np.asarray(y_esc, dtype=bool).sum()), burden=_burden(r, len(eval_scores)),
        sensitivity=est.sensitivity(refers, y_esc), seed=-1,
    )


def matched_budget_table(
    probs: np.ndarray, y_esc: np.ndarray, esc: list[int], *,
    eval_cohort: str, subgroup: str = "ALL",
    member_probs: np.ndarray | None = None, seed: int = est.SEED,
) -> list[PolicyResult]:
    """One `natural` row for argmax, plus one `oracle_evaluation` row per score-library
    member at argmax's own referral count `r`.

    This is the policy-level table that `decompose.py` (S33) reduces to the A/B/C scalars;
    exposed here as labeled rows so every downstream consumer (S33 decomposition, S37
    rescue, S38 transport) reads the same construction instead of re-deriving `r`.
    """
    refers_argmax = est.argmax_refers(probs, esc)
    r = int(refers_argmax.sum())
    rows = [natural_argmax(probs, y_esc, esc, eval_cohort=eval_cohort, subgroup=subgroup)]
    library = est.build_score_library(probs, esc, member_probs)
    for name, scores in library.items():
        rows.append(oracle_top_r(
            scores, y_esc, r, score_name=name, eval_cohort=eval_cohort,
            subgroup=subgroup, seed=seed,
        ))
    return rows


def to_frame(rows: list[PolicyResult]) -> pd.DataFrame:
    return pd.DataFrame([r.as_dict() for r in rows])
