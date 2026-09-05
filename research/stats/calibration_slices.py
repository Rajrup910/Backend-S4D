"""Calibration sliced by subgroup, because the aggregate number hides the interesting part.

The project already reports that the soft-vote ensemble is *under*-confident -- mean
confidence 0.7048 against accuracy 0.8609 on test, a signed gap of -0.156 that accounts
for essentially the whole ECE. That is the opposite of the Guo et al. single-network
result, and the paper already explains why: six members that disagree on the runner-up
class pull the maximum probability down.

**What the aggregate cannot say is that the distortion is not uniform across patients.**
Sliced by age band, the under-40 band is simultaneously the most "accurate" band -- it is
dominated by easy nevi -- and the most badly calibrated one. That matters twice over:

  1. It unifies two threads the manuscript currently keeps apart. The calibration story
     and the fairness story are the same story, and the band where the model is least
     honest about its own confidence is the band where it misses melanoma.
  2. It is the direct motivation for group-wise calibration (Hebert-Johnson et al. 2018,
     multicalibration) over the single global Dirichlet map the project fits today. A map
     with one set of coefficients for every patient cannot close three different gaps.

So this module reports each band **twice**: once on the raw soft-vote and once after the
global Dirichlet map. If the calibrated per-band gaps were equal, the global map would be
sufficient and there would be nothing to say. Whether they are is a measurement, and this
module makes it rather than assuming it.

**Interval discipline.** ECE and the signed gap are not proportions, so per
`research.stats.intervals` they take the lesion-grouped bootstrap and never an exact
binomial interval. Accuracy *is* a proportion and is routed through
`intervals.proportion`, which picks the interval kind from the count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from ml.evaluation.metrics import expected_calibration_error
from research.stats.intervals import N_BOOT, SEED, proportion

#: A band below this many images is reported but flagged: a 15-bin ECE over a handful of
#: images is mostly bin-occupancy noise. Matches `research.selective.fairness.MIN_GROUP_SIZE`
#: so the two tables agree about which groups are worth reading.
MIN_GROUP_SIZE = 30

NUM_BINS = 15

#: Group levels that are data-quality artefacts rather than patient populations. They are
#: always *computed and reported* -- a row that vanishes is a row nobody audits -- but they
#: are excluded from disparity spreads and from `worst_calibrated`. Without this the
#: 38-image missing-age bucket wins "worst calibrated band" on 38 images of bin noise, and
#: the reported ECE gap becomes a statement about record-keeping rather than about
#: patients. Mirrors `research.stats.intersectional.UNINTERPRETABLE`.
UNINTERPRETABLE = ("unknown",)


def _interpretable(result: "GroupCalibration") -> bool:
    return (
        result.adequately_powered
        and result.group != "ALL"
        and not any(token in result.group for token in UNINTERPRETABLE)
    )


@dataclass(frozen=True)
class GroupCalibration:
    """Calibration of one subgroup under one probability source."""

    attribute: str
    group: str
    source: str               # "uncalibrated" | "dirichlet" | any caller-supplied label
    n: int
    n_lesions: int
    accuracy: float
    mean_confidence: float
    signed_gap: float         # mean confidence - accuracy; negative == under-confident
    ece: float
    ece_ci_lo: float
    ece_ci_hi: float
    signed_gap_ci_lo: float
    signed_gap_ci_hi: float
    accuracy_ci_lo: float
    accuracy_ci_hi: float
    accuracy_interval_method: str
    adequately_powered: bool

    def as_dict(self) -> dict:
        return asdict(self)


def confidence_and_correctness(
    y_true: np.ndarray, probs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(max probability, was-the-argmax-right) -- the two arrays every ECE is built from."""
    return probs.max(axis=1), (probs.argmax(axis=1) == y_true)


def grouped_bootstrap_scalar(
    statistic: Callable[[np.ndarray], float],
    lesion_ids: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile CI for any scalar of the rows, resampling **lesions** not images.

    `statistic` receives the row indices of one resample and returns the scalar. Passing
    indices rather than sliced arrays keeps the caller free to compute a statistic over
    several arrays at once (ECE needs confidences *and* correctness) without this function
    knowing what they are.
    """
    if len(lesion_ids) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    unique = np.unique(lesion_ids)
    rows = {lesion: np.flatnonzero(lesion_ids == lesion) for lesion in unique}

    draws = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([rows[lesion] for lesion in sampled])
        value = statistic(idx)
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return (float("nan"), float("nan"))

    tail = (1.0 - confidence) / 2.0 * 100.0
    return (float(np.percentile(draws, tail)), float(np.percentile(draws, 100.0 - tail)))


def group_calibration(
    attribute: str,
    group: str,
    source: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    lesion_ids: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    num_bins: int = NUM_BINS,
) -> GroupCalibration:
    """Calibration summary for one already-masked subgroup."""
    confidences, correct = confidence_and_correctness(y_true, probs)
    accuracy = proportion(
        correct, lesion_ids, label=f"{attribute}={group} accuracy", n_boot=n_boot, seed=seed
    )

    ece_lo, ece_hi = grouped_bootstrap_scalar(
        lambda idx: expected_calibration_error(confidences[idx], correct[idx], num_bins),
        lesion_ids, n_boot=n_boot, seed=seed,
    )
    gap_lo, gap_hi = grouped_bootstrap_scalar(
        lambda idx: float(confidences[idx].mean() - correct[idx].mean()),
        lesion_ids, n_boot=n_boot, seed=seed,
    )

    return GroupCalibration(
        attribute=attribute,
        group=group,
        source=source,
        n=int(len(y_true)),
        n_lesions=int(len(np.unique(lesion_ids))),
        accuracy=float(correct.mean()),
        mean_confidence=float(confidences.mean()),
        signed_gap=float(confidences.mean() - correct.mean()),
        ece=float(expected_calibration_error(confidences, correct, num_bins)),
        ece_ci_lo=ece_lo,
        ece_ci_hi=ece_hi,
        signed_gap_ci_lo=gap_lo,
        signed_gap_ci_hi=gap_hi,
        accuracy_ci_lo=accuracy.interval[0],
        accuracy_ci_hi=accuracy.interval[1],
        accuracy_interval_method=accuracy.primary,
        adequately_powered=len(y_true) >= MIN_GROUP_SIZE,
    )


def slice_calibration(
    attribute: str,
    groups: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    lesion_ids: np.ndarray,
    source: str,
    include_all: bool = True,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    num_bins: int = NUM_BINS,
) -> list[GroupCalibration]:
    """One `GroupCalibration` per level of `attribute`, plus an ALL row for reference.

    The ALL row is not decoration: a per-band gap only means something next to the
    aggregate the paper already reports, and printing them in the same table is what makes
    "the under-40 band is worse than the headline" checkable rather than asserted.
    """
    groups = np.asarray(groups).astype(str)
    levels = sorted(set(groups.tolist()))
    if include_all:
        levels = levels + ["ALL"]

    results = []
    for level in levels:
        mask = np.ones(len(y_true), bool) if level == "ALL" else (groups == level)
        if not mask.any():
            continue
        results.append(
            group_calibration(
                attribute, level, source,
                y_true[mask], probs[mask], lesion_ids[mask],
                n_boot=n_boot, seed=seed, num_bins=num_bins,
            )
        )
    return results


def calibration_gaps(results: list[GroupCalibration]) -> dict[str, float]:
    """Max-minus-min spread of ECE and signed gap across adequately powered real groups.

    The ALL row is excluded: it is a weighted average of the others, so including it can
    only shrink an apparent disparity and never reveal one. `UNINTERPRETABLE` levels are
    excluded for the opposite reason -- they can manufacture one.
    """
    sized = [r for r in results if _interpretable(r)]
    excluded = [r.group for r in results
                if r.group != "ALL" and r not in sized]
    if len(sized) < 2:
        return {"n_groups": float(len(sized)), "excluded_groups": excluded}

    def spread(values: list[float]) -> float:
        finite = [v for v in values if np.isfinite(v)]
        return float(max(finite) - min(finite)) if len(finite) >= 2 else float("nan")

    return {
        "ece_gap": spread([r.ece for r in sized]),
        "signed_gap_spread": spread([r.signed_gap for r in sized]),
        "accuracy_gap": spread([r.accuracy for r in sized]),
        "n_groups": float(len(sized)),
        "groups": [r.group for r in sized],
        "excluded_groups": excluded,
    }


def worst_calibrated(results: list[GroupCalibration]) -> GroupCalibration | None:
    """The powered, interpretable group with the largest ECE -- the row the discussion is about."""
    sized = [r for r in results if _interpretable(r)]
    return max(sized, key=lambda r: r.ece) if sized else None
