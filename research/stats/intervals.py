"""Confidence intervals, and the rule for which kind goes where.

The project already bootstraps Macro-F1 by resampling **lesions**, which is correct: a
metric computed over images that come in near-duplicate clusters has less independent
information than the image count suggests. Workstream G.1 found that applying the same
machinery to a *proportion* at small n is not correct, and that the paper's most
distinctive number was about to be reported with an interval that is too narrow:

    cohort        k/n            Clopper-Pearson 95%    percentile bootstrap
    test <40      3/21  = 0.143  [0.030, 0.363]         [0.000, 0.286]
    val  <40      12/22 = 0.545  [0.322, 0.756]         [0.318, 0.773]
    test 60+      154/199= 0.774 [0.709, 0.830]         [0.714, 0.829]

At n=21 the percentile bootstrap misses the upper tail badly — it cannot place mass above
the largest resampled value, and with three positives the resampling distribution is
coarse and skewed. At n=199 the two agree to within a few thousandths.

**The rule this module encodes**, so it is applied consistently rather than per-caller:

  * a **proportion** (sensitivity, rescue rate, FRR, coverage) -> Clopper-Pearson exact
    when the count is small; the lesion-grouped bootstrap once the count is large enough
    that clustering matters more than discreteness;
  * a **non-proportion** (Macro-F1, balanced accuracy, AUC, ECE) -> always the
    lesion-grouped bootstrap, because there is no exact interval for it and clustering
    is the dominant correction.

`SMALL_COUNT` is the switch. It is a judgement call, stated once here rather than being
implicit in each call site, and both intervals are always reported so a reader can see
what the choice cost.

Clopper-Pearson ignores lesion clustering, which makes it slightly anti-conservative when
one lesion contributes several images. At the counts where it leads (a handful of
positives) the discreteness problem dominates that one, and the alternative — an interval
that excludes the true value more often than 5% of the time — is the worse failure for a
number the paper is built on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: numerator below which the exact interval leads (see module docstring).
SMALL_COUNT = 30

N_BOOT = 2000
SEED = 42


@dataclass(frozen=True)
class Proportion:
    """A count-over-total carrying both interval kinds, and which one leads."""

    numerator: int
    denominator: int
    point: float
    clopper_pearson: tuple[float, float]
    grouped_bootstrap: tuple[float, float]
    primary: str          # "clopper_pearson" | "grouped_bootstrap"
    n_lesions: int
    label: str = ""

    @property
    def interval(self) -> tuple[float, float]:
        """The interval that leads for this count."""
        return (
            self.clopper_pearson if self.primary == "clopper_pearson"
            else self.grouped_bootstrap
        )

    @property
    def method_short(self) -> str:
        return "CP" if self.primary == "clopper_pearson" else "boot"

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "n_lesions": self.n_lesions,
            "point": self.point,
            "clopper_pearson_95": list(self.clopper_pearson),
            "grouped_bootstrap_95": list(self.grouped_bootstrap),
            "primary_interval": self.primary,
        }


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval, by inverting the Beta CDF.

    Needs no normal approximation and stays honest at k=0 and k=n, where the percentile
    bootstrap collapses to a degenerate point.
    """
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
    n_boot: int = N_BOOT,
    seed: int = SEED,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile CI for `flags.mean()`, resampling **lesions** rather than images."""
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


def proportion(
    flags: np.ndarray,
    lesion_ids: np.ndarray,
    label: str = "",
    n_boot: int = N_BOOT,
    seed: int = SEED,
    small_count: int = SMALL_COUNT,
) -> Proportion:
    """Build a `Proportion` from a boolean flag array, applying the module's rule.

    `flags` is the event indicator over the *conditioning* population already — e.g. for
    sensitivity, one entry per truly-escalating case, True where it was caught.
    """
    flags = np.asarray(flags).astype(bool)
    k, n = int(flags.sum()), int(len(flags))
    return Proportion(
        numerator=k,
        denominator=n,
        point=float(k / n) if n else float("nan"),
        clopper_pearson=clopper_pearson(k, n),
        grouped_bootstrap=grouped_bootstrap_proportion(
            flags.astype(np.float64), lesion_ids, n_boot=n_boot, seed=seed
        ),
        primary="clopper_pearson" if k < small_count else "grouped_bootstrap",
        n_lesions=int(len(np.unique(lesion_ids))) if len(lesion_ids) else 0,
        label=label,
    )


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Whether two intervals share any point.

    Non-overlap implies a significant difference; overlap does **not** imply the absence
    of one, which is why this is only ever used to make the conservative statement.
    """
    if not all(np.isfinite([*a, *b])):
        return True
    return a[0] <= b[1] and b[0] <= a[1]
