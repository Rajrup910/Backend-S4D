"""Number Needed to Biopsy, and the prevalence correction without which it misleads.

NNB is the clinical currency for a referral rule: of every N lesions the rule sends for
biopsy, one is genuinely escalating. It is simply the reciprocal of precision on the
binary escalate / do-not-escalate decision,

    NNB = (TP + FP) / TP = 1 / PPV,

and it is the number a clinic actually feels, which is why the paper reports it rather
than precision.

**The trap, and the reason this module exists.** NNB depends on prevalence, and
HAM10000's escalating prevalence is roughly 19-20% -- a curated dermoscopy archive, not a
screening population, where the literature's 8-15 dermatologist figures were measured at
prevalences nearer 1-5%. An unadjusted NNB of 2.7 computed here and set beside a
dermatologist's 8-15 would read as a five-fold improvement when it is very largely an
artefact of who is in the dataset. Reporting it that way is exactly the claim a clinical
reviewer rejects, so this module never returns a bare observed NNB without also returning
the reference-prevalence version and the weight that produced it.

**The correction.** Sensitivity and specificity are properties of the decision rule and do
not move with prevalence; only the *mix* of cases does. So the benign cases are importance
re-weighted until the cohort's escalating prevalence equals the reference, positives
keeping weight 1:

    w = n_pos * (1 - pi_ref) / (n_neg * pi_ref)          (benign weight)
    NNB(pi_ref) = (TP + w * FP) / TP

At `pi_ref` equal to the observed prevalence, `w` is 1 and the formula returns the observed
NNB exactly, which is the check `assert` below enforces. The re-weighting assumes the
class-conditional score distributions transfer -- that a benign lesion in a screening
clinic looks to the model like a benign lesion in HAM10000. That is an assumption, it is
almost certainly optimistic (screening cohorts hold easier benign lesions and different
imaging), and it is stated in the plan rather than buried here.

Both the observed and the reference-prevalence values are always returned together. The
plan pre-registers which one is the headline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

#: Reference escalating prevalence for the primary reported NNB. Mid-point of the 1-5%
#: range quoted for primary-care skin-lesion screening; the flanking values are reported
#: as a pre-registered sensitivity range, never as alternatives to choose between.
REFERENCE_PREVALENCE = 0.03
PREVALENCE_SENSITIVITY_RANGE = (0.01, 0.05)


@dataclass(frozen=True)
class BiopsyBurden:
    """One decision rule's referral burden on one cohort, at one prevalence."""

    cohort: str
    rule: str
    n: int
    n_escalating: int
    observed_prevalence: float
    true_positives: int
    false_positives: int
    false_negatives: int
    sensitivity: float
    referral_rate: float
    ppv_observed: float
    nnb_observed: float
    reference_prevalence: float
    benign_weight: float
    ppv_reference: float
    nnb_reference: float

    def as_dict(self) -> dict:
        return asdict(self)


def benign_weight(n_positive: int, n_negative: int, reference_prevalence: float) -> float:
    """Importance weight on benign cases that moves the cohort to `reference_prevalence`."""
    if n_negative == 0 or reference_prevalence <= 0.0 or reference_prevalence >= 1.0:
        return float("nan")
    return float(n_positive * (1.0 - reference_prevalence) / (n_negative * reference_prevalence))


def biopsy_burden(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    escalating_idx: list[int],
    *,
    cohort: str,
    rule: str,
    reference_prevalence: float = REFERENCE_PREVALENCE,
) -> BiopsyBurden:
    """NNB for one cohort under one rule, observed and re-weighted to the reference.

    `y_pred` is a hard 7-class prediction; the escalate decision is "the predicted class
    is one of the escalating classes", which is the same decision every other escalation
    metric in the project uses. Keeping that definition identical is what lets NNB sit in
    the same table as escalation sensitivity without a footnote reconciling two rules.
    """
    true_esc = np.isin(y_true, escalating_idx)
    pred_esc = np.isin(y_pred, escalating_idx)

    tp = int(np.sum(true_esc & pred_esc))
    fp = int(np.sum(~true_esc & pred_esc))
    fn = int(np.sum(true_esc & ~pred_esc))
    n_pos = int(true_esc.sum())
    n_neg = int((~true_esc).sum())
    n = len(y_true)

    ppv_obs = float(tp / (tp + fp)) if (tp + fp) else float("nan")
    weight = benign_weight(n_pos, n_neg, reference_prevalence)
    weighted_fp = weight * fp
    ppv_ref = float(tp / (tp + weighted_fp)) if (tp + weighted_fp) > 0 else float("nan")

    return BiopsyBurden(
        cohort=cohort,
        rule=rule,
        n=n,
        n_escalating=n_pos,
        observed_prevalence=float(n_pos / n) if n else float("nan"),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        sensitivity=float(tp / n_pos) if n_pos else float("nan"),
        referral_rate=float(pred_esc.mean()) if n else float("nan"),
        ppv_observed=ppv_obs,
        nnb_observed=float(1.0 / ppv_obs) if ppv_obs and np.isfinite(ppv_obs) and ppv_obs > 0 else float("inf"),
        reference_prevalence=float(reference_prevalence),
        benign_weight=weight,
        ppv_reference=ppv_ref,
        nnb_reference=float(1.0 / ppv_ref) if ppv_ref and np.isfinite(ppv_ref) and ppv_ref > 0 else float("inf"),
    )


def nnb_at(burden: BiopsyBurden, reference_prevalence: float) -> float:
    """Re-price an already-computed burden at another prevalence, no re-scoring needed.

    Used for the pre-registered sensitivity range: the confusion counts are fixed, so the
    whole prevalence curve is available from them and no additional test read is involved.
    """
    weight = benign_weight(burden.n_escalating, burden.n - burden.n_escalating, reference_prevalence)
    denominator = burden.true_positives + weight * burden.false_positives
    if burden.true_positives == 0 or not np.isfinite(denominator) or denominator <= 0:
        return float("inf")
    return float(denominator / burden.true_positives)


def self_check() -> None:
    """At pi_ref == observed prevalence the correction must be the identity."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 7, size=500)
    y_pred = rng.integers(0, 7, size=500)
    esc = [0, 1, 4]
    observed = float(np.isin(y_true, esc).mean())
    burden = biopsy_burden(y_true, y_pred, esc, cohort="self-check", rule="random",
                           reference_prevalence=observed)
    assert abs(burden.benign_weight - 1.0) < 1e-9, burden.benign_weight
    assert abs(burden.nnb_reference - burden.nnb_observed) < 1e-9
