"""The link tier of a candidate pair, from its two pieces of evidence (protocol amendment 01).

What the gate (`research/data/build_master_v7.py`, `research/data/overlap_guard.py`) does with each tier:
* both tiers: the V7 folds are frozen per row, so a link moves no row; when a pair crosses a fold, or joins a training-only
  row to a development, spent or held-out-archive row, the guard removes the training-side row. A false link of either tier
  costs training rows (`cost.py` measures how many); a missed one leaves a leak;
* `confirmed` links are also merged into lesion groups (transitively): the groups the bootstraps and denominators resample,
  and the gate fails if one exceeds the percolation limit (31 rows). So confirmed needs the stronger evidence.

Evidence per pair: `score`, the matcher's density-normalised similarity (`density.py`), and `inliers`, the SIFT + RANSAC
inlier count on the originals (`scripts/audit_leak_keypoints.py`), a different detector. `RULE` is the rule in words; it is
written into the amendment and so cannot change after the held-out benchmark is read.
"""

from __future__ import annotations

import numpy as np

SIFT_BAR = 30      # the V6 bar: a second, different detector agrees
SIFT_STRICT = 12   # enough when the score alone already clears FMR 1e-6
SIFT_SOME = 8      # lifts a pair between FMR 1e-5 and 1e-6 to at_risk
SIFT_LOOSE = 12    # lifts a pair between FMR 1e-4 and 1e-5 to at_risk

RULE = ("thresholds T4, T5, T6 = the matcher's score at FMR 1e-4, 1e-5, 1e-6 on the held-out benchmark; "
        "confirmed: (score >= T5 and SIFT RANSAC inliers on the originals >= 30) or (score >= T6 and inliers >= 12); "
        "at_risk (if not confirmed): score >= T6, or (score >= T5 and inliers >= 8), or (score >= T4 and inliers >= 12); "
        "a matcher without T4/T6 (scored before amendment 01): at_risk score >= T5, confirmed also inliers >= 30")


def tier(score: np.ndarray, inliers: np.ndarray, thr_primary: float, thr_strict: float | None,
         thr_loose: float | None = None) -> np.ndarray:
    """Per pair: "confirmed", "at_risk" or "" (no link), by `RULE`.

    `thr_primary`, `thr_strict` and `thr_loose` are the score thresholds at FMR 1e-5, 1e-6 and 1e-4 (the last two None for a
    matcher scored before the amendment). Every condition only ever asks for MORE score or MORE inliers, so more evidence can
    never weaken a tier. Why these bars (training clusters, top-8 candidates by normalised SSCD; recomputed by `develop.py`
    into `results/v7/m1/amendment_01_development.json`):
    the 1e-5..1e-6 band with fewer than 8 inliers was 800 true pairs to 453 false ones, so the score alone is trusted only
    from 1e-6; a SIFT agreement of 8-12 inliers is about 30:1 true within 1e-4..1e-6. Same pair recall as score >= T5
    (0.479 against 0.472) with 91 false at-risk links instead of 521 (25 false confirmed against 16; largest merged
    group 21 either way), so far fewer training rows are lost to links that are not duplicates.
    """
    score, inliers = np.asarray(score, dtype=float), np.asarray(inliers, dtype=float)
    p5 = score >= thr_primary
    if thr_strict is None:
        out = np.where(p5, "at_risk", "").astype(object)
        out[p5 & (inliers >= SIFT_BAR)] = "confirmed"
        return out
    p6 = score >= thr_strict
    p4 = score >= thr_loose if thr_loose is not None else p5
    risk = p6 | (p5 & (inliers >= SIFT_SOME)) | (p4 & (inliers >= SIFT_LOOSE))
    confirmed = (p5 & (inliers >= SIFT_BAR)) | (p6 & (inliers >= SIFT_STRICT))
    out = np.where(risk, "at_risk", "").astype(object)
    out[confirmed] = "confirmed"
    return out
