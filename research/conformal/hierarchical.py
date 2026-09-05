"""Hierarchical bipartite Mondrian conformal calibration, and the FRR machinery.

**Why not the obvious thing.** The natural extension of class-conditional (Mondrian)
calibration to a subgroup question is a full cross: 3 age bands x 7 classes = 21 cells,
each with its own conformal quantile. On this data that does not survive contact with
arithmetic. A finite threshold at level `1-alpha` needs `ceil((n+1)(1-alpha)) <= n`, i.e.
`n >= 1/alpha - 1`: 19 calibration points at alpha=0.05, 99 at alpha=0.01. The `<40` band
holds 22 escalating images in val and 64 out-of-fold, spread over three escalating
classes -- so most of the 21 cells are degenerate by construction and the "guarantee"
collapses to "every class is in the set", which is true and useless.

**The bipartite cell.** Group instead by (age band x escalation requirement), where the
second axis is binary -- does this class oblige a referral or not. That is 6 cells rather
than 21, and it is the partition the clinical question actually asks about: what a
shortlist promises is that a lesion needing escalation will have *something escalating*
on it. Coverage conditional on (band, escalating) is exactly the statement that a young
patient's melanoma is on the list at rate `1-alpha`, which marginal coverage is free to
sacrifice and full class-conditional coverage cannot certify at this sample size.

**The hierarchy, and what it costs.** Cells are fitted first. A cell that still cannot
certify a finite threshold backs off to the all-ages class-conditional quantile for its
candidate classes, and if that is also degenerate the threshold stays `+inf` and every
class enters the set. Which of the three happened is recorded per cell and reported,
because a backed-off cell no longer carries a band-conditional guarantee -- it carries
the weaker class-conditional one, and a table that does not say so is claiming coverage
it did not certify.

**The guarantee is conditional on the true label's cell, not the prediction's.** As in
`calibrate.prediction_sets`, the threshold is indexed by the *candidate* class, so class
`c` enters image `i`'s set when its score clears the bar for `(band(i), group(c))`. This
is what makes the conditional statement hold: among truly-escalating `<40` lesions, the
relevant bar was fitted on truly-escalating `<40` calibration points.

**False Reassurance Rate.** Formalised here rather than left as a count. For a set-valued
predictor `S` on a lesion with true class `y`:

    FRR = P( S(x) contains no escalating class | y is escalating )

It is the complement of coverage of the escalating classes as a *group* rather than
individually -- a set holding `bcc` for a `mel` lesion is a misdiagnosis but not a false
reassurance, because it still routes the patient to a dermatologist. That distinction is
the reason FRR is the right primary metric for a referral system and marginal coverage is
not.

Intervals are lesion-grouped: HAM10000 holds several images per lesion, and an interval
that treats them as independent is too narrow -- the same error the project's splits
exist to avoid. Clopper-Pearson is reported alongside as the image-level exact reference,
and leads where the count is small enough that the percentile bootstrap under-covers
(Workstream G.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from research.conformal.calibrate import conformal_quantile

BENIGN = 0
ESCALATING = 1
GROUP_LABELS = ("benign", "escalating")

#: how a (band, group) cell got its threshold
SOURCE_CELL = "cell"
SOURCE_BACKOFF = "class_conditional_backoff"
SOURCE_DEGENERATE = "degenerate"

UNKNOWN_BAND = "unknown"


def class_groups(num_classes: int, escalating_idx: list[int]) -> np.ndarray:
    """(C,) array marking each class benign (0) or escalating (1)."""
    groups = np.zeros(num_classes, dtype=np.int64)
    groups[np.asarray(escalating_idx, dtype=np.int64)] = ESCALATING
    return groups


def min_calibration_n(alpha: float) -> int:
    """Smallest calibration count that can certify a finite threshold at this alpha.

    `ceil((n+1)(1-alpha)) <= n` rearranges to `n >= 1/alpha - 1`; reported so a degenerate
    cell can be explained by a number rather than by "too few".
    """
    return int(np.ceil(1.0 / alpha - 1.0))


@dataclass(frozen=True)
class BipartiteState:
    """Thresholds per (age band, escalation group), plus how each one was obtained."""

    method: str
    alpha: float
    bands: tuple[str, ...]
    class_group: np.ndarray                       # (C,) benign/escalating per class
    cell_quantiles: dict[tuple[str, int], float]  # (band, group) -> threshold, may be +inf
    cell_counts: dict[tuple[str, int], int]
    cell_source: dict[tuple[str, int], str]
    fallback_quantiles: np.ndarray                # (C,) all-ages class-conditional
    fallback_counts: np.ndarray                   # (C,)
    band_index: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def degenerate_cells(self) -> tuple[tuple[str, int], ...]:
        return tuple(k for k, v in self.cell_source.items() if v == SOURCE_DEGENERATE)

    @property
    def backed_off_cells(self) -> tuple[tuple[str, int], ...]:
        return tuple(k for k, v in self.cell_source.items() if v == SOURCE_BACKOFF)

    def thresholds_for_band(self, band: str) -> np.ndarray:
        """(C,) threshold per *candidate* class for an image in this band."""
        out = np.empty(len(self.class_group), dtype=np.float64)
        for c, group in enumerate(self.class_group):
            q = self.cell_quantiles.get((band, int(group)), float("inf"))
            if not np.isfinite(q):
                q = float(self.fallback_quantiles[c])
            out[c] = q
        return out

    def threshold_matrix(self) -> np.ndarray:
        """(n_bands, C) -- `thresholds_for_band` for every band, in `self.bands` order."""
        return np.stack([self.thresholds_for_band(b) for b in self.bands])


def fit_bipartite(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_bands: np.ndarray,
    num_classes: int,
    escalating_idx: list[int],
    alpha: float,
    method: str,
    bands: tuple[str, ...],
) -> BipartiteState:
    """Fit one quantile per (band, escalation group), backing off where it cannot certify.

    `calibration_scores` are the true-label scores of the calibration points, matching
    `calibrate.fit`'s contract, so the two calibrators are interchangeable at the call site.
    """
    groups = class_groups(num_classes, escalating_idx)
    point_group = groups[calibration_labels]

    # All-ages class-conditional fallback: the second level of the hierarchy.
    fallback = np.empty(num_classes, dtype=np.float64)
    fallback_counts = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        in_class = calibration_scores[calibration_labels == c]
        fallback_counts[c] = len(in_class)
        fallback[c] = conformal_quantile(in_class, alpha)

    quantiles: dict[tuple[str, int], float] = {}
    counts: dict[tuple[str, int], int] = {}
    source: dict[tuple[str, int], str] = {}

    for band in bands:
        in_band = calibration_bands == band
        for group in (BENIGN, ESCALATING):
            cell = in_band & (point_group == group)
            key = (band, group)
            counts[key] = int(cell.sum())
            q = conformal_quantile(calibration_scores[cell], alpha)
            quantiles[key] = q
            if np.isfinite(q):
                source[key] = SOURCE_CELL
            else:
                # Does the class-conditional level rescue every class in this group?
                members = np.where(groups == group)[0]
                source[key] = (
                    SOURCE_BACKOFF if np.isfinite(fallback[members]).all() else SOURCE_DEGENERATE
                )

    return BipartiteState(
        method=method,
        alpha=alpha,
        bands=tuple(bands),
        class_group=groups,
        cell_quantiles=quantiles,
        cell_counts=counts,
        cell_source=source,
        fallback_quantiles=fallback,
        fallback_counts=fallback_counts,
        band_index={b: i for i, b in enumerate(bands)},
    )


def prediction_sets_bipartite(
    state: BipartiteState, score_matrix: np.ndarray, bands: np.ndarray
) -> np.ndarray:
    """(N, C) boolean membership under the band-specific, group-indexed thresholds."""
    matrix = state.threshold_matrix()                       # (n_bands, C)
    default = state.band_index.get(UNKNOWN_BAND, 0)
    index = np.array([state.band_index.get(b, default) for b in bands])
    return score_matrix <= matrix[index]


# --------------------------------------------------------------------------------------
# False Reassurance Rate and its intervals
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Proportion:
    """A count-over-total with both interval kinds attached, and which one leads."""

    numerator: int
    denominator: int
    point: float
    clopper_pearson: tuple[float, float]
    grouped_bootstrap: tuple[float, float]
    primary: str          # "clopper_pearson" | "grouped_bootstrap"
    n_lesions: int

    def as_dict(self) -> dict:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "n_lesions": self.n_lesions,
            "point": self.point,
            "clopper_pearson_95": list(self.clopper_pearson),
            "grouped_bootstrap_95": list(self.grouped_bootstrap),
            "primary_interval": self.primary,
        }


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval. Inverts the Beta CDF, so it needs no normal approximation
    and does not collapse at k=0 or k=n, where the percentile bootstrap degenerates."""
    from scipy.stats import beta

    if n == 0:
        return (float("nan"), float("nan"))
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - tail, k + 1, n - k))
    return (lower, upper)


def grouped_bootstrap_proportion(
    flags: np.ndarray,
    lesion_ids: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile CI for `flags.mean()`, resampling **lesions** rather than images.

    Images of one lesion are near-duplicates; resampling them independently pretends the
    sample is larger than it is and returns an interval that is too narrow.
    """
    if len(flags) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    unique = np.unique(lesion_ids)
    rows = {lesion: np.flatnonzero(lesion_ids == lesion) for lesion in unique}

    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([rows[lesion] for lesion in sampled])
        draws[b] = flags[idx].mean()

    tail = (1.0 - confidence) / 2.0 * 100.0
    return (float(np.percentile(draws, tail)), float(np.percentile(draws, 100.0 - tail)))


def false_reassurance(
    sets: np.ndarray,
    y_true: np.ndarray,
    lesion_ids: np.ndarray,
    escalating_idx: list[int],
    n_boot: int = 2000,
    seed: int = 42,
    small_count: int = 30,
) -> Proportion:
    """FRR with both intervals: the share of escalating lesions whose set is all-benign.

    `small_count` selects the primary interval. Clopper-Pearson leads when the numerator
    is small, where the percentile bootstrap is known to under-cover (Workstream G.1); the
    lesion-grouped bootstrap leads otherwise, because it is the only one of the two that
    accounts for repeated images of a lesion.
    """
    is_serious = np.isin(y_true, escalating_idx)
    has_escalating = sets[:, escalating_idx].any(axis=1)
    reassured = (is_serious & ~has_escalating)[is_serious]

    k, n = int(reassured.sum()), int(is_serious.sum())
    serious_lesions = lesion_ids[is_serious]
    return Proportion(
        numerator=k,
        denominator=n,
        point=float(k / n) if n else float("nan"),
        clopper_pearson=clopper_pearson(k, n),
        grouped_bootstrap=grouped_bootstrap_proportion(
            reassured.astype(np.float64), serious_lesions, n_boot=n_boot, seed=seed
        ),
        primary="clopper_pearson" if k < small_count else "grouped_bootstrap",
        n_lesions=int(len(np.unique(serious_lesions))),
    )


def cell_coverage(
    sets: np.ndarray,
    y_true: np.ndarray,
    bands: np.ndarray,
    escalating_idx: list[int],
    band_order: tuple[str, ...],
) -> dict[str, dict]:
    """Achieved coverage inside each (band, group) cell -- the number the fit promises.

    Reported for every variant including the marginal and class-conditional baselines, so
    that "the bipartite calibrator helps" is a comparison of measured cell coverage rather
    than an argument from construction.
    """
    num_classes = sets.shape[1]
    groups = class_groups(num_classes, escalating_idx)
    covered = sets[np.arange(len(y_true)), y_true]
    point_group = groups[y_true]

    out: dict[str, dict] = {}
    for band in band_order:
        for group in (BENIGN, ESCALATING):
            cell = (bands == band) & (point_group == group)
            n = int(cell.sum())
            out[f"{band}|{GROUP_LABELS[group]}"] = {
                "n": n,
                "coverage": float(covered[cell].mean()) if n else float("nan"),
                "mean_set_size": float(sets[cell].sum(axis=1).mean()) if n else float("nan"),
            }
    return out
