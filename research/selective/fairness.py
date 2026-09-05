"""Subgroup fairness of the classifier and of its abstention policy.

Two questions, and the second is the one an abstaining system creates:

  1. *Diagnostic parity* — does escalation sensitivity hold up in every subgroup, or is
     the headline number carried by the majority group?
  2. *Referral-burden parity* — when the system abstains, someone is told to come in for
     a biopsy. If one subgroup is referred at twice the rate of another, the model has
     quietly redistributed cost onto those patients even though its retained accuracy
     looks identical everywhere. Aggregate risk-coverage curves cannot show this.

**On skin tone.** The roadmap asks for Fitzpatrick I-VI slices. This repository has no
Fitzpatrick labels it can use: HAM10000 ships none, and while `ml/data/manifest_pad.csv`
carries the PAD-UFES-20 Fitzpatrick column for 1302 of its 2106 rows, the PAD images
themselves are not present under `data/`, so no predictions can be produced for them
here. The ITA image proxy in `ml/ood/skin_tone_slice.py` is not a substitute — that
script documents its own failure on dermoscopy, where vignetting and erythema drive the
angle onto noise. Rather than dress a broken proxy up as a fairness result, this module
slices on the attributes HAM10000 actually records — sex, age band, and lesion site —
and the report states plainly that skin tone is unanswered with this data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.evaluation.metrics import compute_metrics
from ml.paths import load_class_mapping, load_training_config, resolve

MIN_GROUP_SIZE = 30
MIN_POSITIVES = 10
AGE_BINS = [0, 40, 60, 200]
AGE_LABELS = ["<40", "40-59", "60+"]


@dataclass(frozen=True)
class GroupResult:
    attribute: str
    group: str
    n: int
    n_escalating: int
    macro_f1: float
    balanced_accuracy: float
    escalation_sensitivity: float
    escalation_fpr: float
    missed_serious: int
    predicted_escalate_rate: float
    abstention_rate: float | None
    adequately_powered: bool     # enough cases to estimate a rate at all
    positives_powered: bool      # enough true escalating cases to estimate sensitivity


def load_attributes(image_ids: np.ndarray) -> pd.DataFrame:
    """Sex, age band and lesion site for the given images, in the given order."""
    config = load_training_config()
    manifest = pd.read_csv(resolve(config["data"]["manifest"])).set_index("image_id")
    rows = manifest.loc[[str(i) for i in image_ids]]

    # `.astype(object)` before filling: a categorical from pd.cut cannot take a value
    # outside its categories, and a missing age must stay visible as its own group
    # rather than being folded into a real band.
    age_band = pd.cut(rows["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(object)
    age_band = age_band.where(age_band.notna(), "unknown")

    return pd.DataFrame(
        {
            "sex": rows["sex"].fillna("unknown").astype(str).to_numpy(),
            "age_band": age_band.astype(str).to_numpy(),
            "localization": rows["localization"].fillna("unknown").astype(str).to_numpy(),
        },
        index=range(len(rows)),
    )


def _escalation_rates(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """(sensitivity, false-positive rate, predicted-positive rate) on escalate-vs-not."""
    mapping = load_class_mapping()
    escalating = [c.index for c in mapping.classes if c.needs_escalation]
    true_serious = np.isin(y_true, escalating)
    pred_serious = np.isin(y_pred, escalating)

    positives = int(true_serious.sum())
    negatives = int((~true_serious).sum())
    sensitivity = float((true_serious & pred_serious).sum() / positives) if positives else float("nan")
    fpr = float((~true_serious & pred_serious).sum() / negatives) if negatives else float("nan")
    return sensitivity, fpr, float(pred_serious.mean())


def slice_attribute(
    attribute: str,
    groups: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    keep: np.ndarray | None = None,
) -> list[GroupResult]:
    """Per-group metrics for one attribute.

    Args:
        keep: boolean mask of retained (non-abstained) cases over the full split. When
            given, diagnostic metrics are computed on each group's retained cases and the
            group's abstention rate is reported alongside; when None, no abstention is in
            play and every case counts.
    """
    results = []
    for group in sorted(set(groups)):
        in_group = groups == group
        n_total = int(in_group.sum())
        if n_total == 0:
            continue

        if keep is None:
            evaluated = in_group
            abstention = None
        else:
            evaluated = in_group & keep
            abstention = float((in_group & ~keep).sum() / n_total)

        n_eval = int(evaluated.sum())
        if n_eval == 0:
            continue

        metrics = compute_metrics(y_true[evaluated], y_pred[evaluated], probs[evaluated])
        sensitivity, fpr, predicted_rate = _escalation_rates(y_true[evaluated], y_pred[evaluated])

        mapping = load_class_mapping()
        escalating = [c.index for c in mapping.classes if c.needs_escalation]
        n_escalating = int(np.isin(y_true[evaluated], escalating).sum())

        results.append(
            GroupResult(
                attribute=attribute,
                group=str(group),
                n=n_eval,
                n_escalating=n_escalating,
                macro_f1=float(metrics["macro_f1"]),
                balanced_accuracy=float(metrics["balanced_accuracy"]),
                escalation_sensitivity=sensitivity,
                escalation_fpr=fpr,
                missed_serious=int(metrics["clinical"]["missed_serious_cases"]),
                predicted_escalate_rate=predicted_rate,
                abstention_rate=abstention,
                adequately_powered=n_total >= MIN_GROUP_SIZE,
                positives_powered=n_escalating >= MIN_POSITIVES,
            )
        )
    return results


@dataclass(frozen=True)
class RescueResult:
    """How much of a group's diagnostic failure the abstention policy actually catches."""

    group: str
    n: int
    n_escalating: int
    sensitivity_full_coverage: float
    n_would_be_missed: int      # serious cases misclassified at full coverage
    n_rescued: int              # of those, referred instead of answered wrongly
    rescue_rate: float          # n_rescued / n_would_be_missed
    referral_rate: float        # over the whole group


def miss_rescue(
    groups: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, keep: np.ndarray
) -> list[RescueResult]:
    """Per group: of the serious cases the model gets wrong, how many does it refer?

    This is the question a risk-coverage curve cannot answer. A curve improves whenever
    abstention removes *any* errors, so a policy can look excellent in aggregate while
    being silent exactly where it is needed — on a subgroup the model is confidently
    wrong about. A low rescue rate next to a low referral rate is the signature of
    confident error, and it means uncertainty-based abstention is not a safety net for
    that subgroup at all.
    """
    mapping = load_class_mapping()
    escalating = [c.index for c in mapping.classes if c.needs_escalation]

    results = []
    for group in sorted(set(groups)):
        in_group = groups == group
        serious = in_group & np.isin(y_true, escalating)
        n_serious = int(serious.sum())
        if n_serious == 0:
            continue

        caught = np.isin(y_pred[serious], escalating)
        would_miss = serious & ~np.isin(y_pred, escalating)
        n_would_miss = int(would_miss.sum())
        rescued = int((would_miss & ~keep).sum())

        results.append(
            RescueResult(
                group=str(group),
                n=int(in_group.sum()),
                n_escalating=n_serious,
                sensitivity_full_coverage=float(caught.mean()),
                n_would_be_missed=n_would_miss,
                n_rescued=rescued,
                rescue_rate=float(rescued / n_would_miss) if n_would_miss else float("nan"),
                referral_rate=float((in_group & ~keep).sum() / in_group.sum()),
            )
        )
    return results


def gaps(results: list[GroupResult]) -> dict[str, float]:
    """Max-minus-min spreads across sufficiently powered groups of one attribute.

    Reported as gaps rather than ratios so a group with zero positives does not produce
    an infinite disparity. Each gap uses its own power gate, because they are estimated
    from different denominators: rates over all cases (referral burden, predicted-positive
    rate, macro-F1) need `MIN_GROUP_SIZE` cases, while **sensitivity needs
    `MIN_POSITIVES` actual escalating cases**. Without that second gate a group holding
    three melanomas can miss one and post a 0.33 sensitivity, manufacturing a disparity
    that is entirely sampling error — the single most common way a fairness table
    misleads.
    """

    def spread(values: list[float]) -> float:
        finite = [v for v in values if np.isfinite(v)]
        return float(max(finite) - min(finite)) if len(finite) >= 2 else float("nan")

    sized = [r for r in results if r.adequately_powered]
    positive_powered = [r for r in sized if r.positives_powered]
    if len(sized) < 2:
        return {}

    out: dict[str, float] = {
        "demographic_parity_gap": spread([r.predicted_escalate_rate for r in sized]),
        "macro_f1_gap": spread([r.macro_f1 for r in sized]),
    }
    if len(positive_powered) >= 2:
        out["equalized_odds_tpr_gap"] = spread([r.escalation_sensitivity for r in positive_powered])
        out["equalized_odds_fpr_gap"] = spread([r.escalation_fpr for r in positive_powered])
    if all(r.abstention_rate is not None for r in sized):
        out["referral_burden_gap"] = spread([float(r.abstention_rate) for r in sized])
    return out
