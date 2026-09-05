"""Intersectional subgroup slices, with the power gates enforced rather than hoped for.

`research.selective.fairness` slices one attribute at a time -- sex, age band, lesion
site. A one-at-a-time table can miss a disparity that lives in a cell: if the under-40
deficit is carried entirely by one sex, neither the age table nor the sex table shows it,
because each marginalises over the other.

**Why this is OOF-only.** The cells are small, and this module refuses to pretend
otherwise. Validation carries 22 escalating images under 40 and test carries 21, so an
age x sex cell holds roughly 10-11 escalating cases -- at or below the `MIN_POSITIVES=10`
gate `research.selective.fairness` already imposes, which means a cell could miss one case
and post a sensitivity that moves by ten points on pure sampling error. The OOF split
carries 64 escalating images under 40, so its cells hold around 32, which is reportable.
The gates are the module's own, imported rather than redefined, so this table and the
single-attribute fairness table agree about what counts as readable.

**Suppressed cells are named, never dropped.** A table that silently omits its
underpowered cells reads as though those patients were fine. Every cell appears with its
counts; the ones that fail a gate carry `suppressed=True` and a reason, and their metrics
are withheld from the disparity spreads rather than from the reader.

Every proportion here goes through `research.stats.intervals.proportion`, so a cell with
32 positives gets an exact Clopper-Pearson interval rather than a percentile bootstrap
that cannot reach the upper tail at that count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.paths import load_class_mapping
from research.selective.fairness import MIN_GROUP_SIZE, MIN_POSITIVES
from research.stats.intervals import N_BOOT, SEED, proportion

#: Cells whose group label contains one of these are reported but never used to claim a
#: disparity: "unknown" is a data-quality artefact, not a patient population.
UNINTERPRETABLE = ("unknown",)


def combine(*attributes: np.ndarray, separator: str = " x ") -> np.ndarray:
    """Cartesian group label per row, e.g. ('<40', 'male') -> '<40 x male'."""
    columns = [np.asarray(a).astype(str) for a in attributes]
    return np.array([separator.join(values) for values in zip(*columns, strict=True)])


def _suppression_reason(n: int, n_positive: int, group: str) -> str:
    reasons = []
    if n < MIN_GROUP_SIZE:
        reasons.append(f"n={n} < MIN_GROUP_SIZE={MIN_GROUP_SIZE}")
    if n_positive < MIN_POSITIVES:
        reasons.append(f"escalating={n_positive} < MIN_POSITIVES={MIN_POSITIVES}")
    if any(token in group for token in UNINTERPRETABLE):
        reasons.append("group contains an 'unknown' attribute level")
    return "; ".join(reasons)


def intersectional_table(
    groups: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lesion_ids: np.ndarray,
    keep: np.ndarray | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> pd.DataFrame:
    """One row per cell: counts, escalation sensitivity with an interval, referral burden.

    Args:
        keep: boolean mask of retained (non-abstained) cases. When given, the referral
            rate is reported per cell and the rescue count says how many of the cell's
            missed escalating cases the abstention policy referred anyway -- the statistic
            that distinguishes a confidently-wrong subgroup from an uncertain one.
    """
    mapping = load_class_mapping()
    escalating = [c.index for c in mapping.classes if c.needs_escalation]
    groups = np.asarray(groups).astype(str)
    true_esc = np.isin(y_true, escalating)
    pred_esc = np.isin(y_pred, escalating)

    rows = []
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        n = int(mask.sum())
        if n == 0:
            continue
        positives = mask & true_esc
        n_positive = int(positives.sum())

        caught = pred_esc[positives]
        sens = proportion(
            caught, lesion_ids[positives], label=f"{group} sensitivity",
            n_boot=n_boot, seed=seed,
        ) if n_positive else None

        reason = _suppression_reason(n, n_positive, group)
        row = {
            "group": group,
            "n": n,
            "n_lesions": int(len(np.unique(lesion_ids[mask]))),
            "n_escalating": n_positive,
            "escalating_prior": float(n_positive / n),
            "n_caught": int(caught.sum()) if n_positive else 0,
            "escalation_sensitivity": sens.point if sens else float("nan"),
            "sens_ci_lo": sens.interval[0] if sens else float("nan"),
            "sens_ci_hi": sens.interval[1] if sens else float("nan"),
            "sens_interval_method": sens.method_short if sens else "",
            "predicted_escalate_rate": float(pred_esc[mask].mean()),
            "suppressed": bool(reason),
            "suppression_reason": reason,
        }

        if keep is not None:
            missed = positives & ~pred_esc
            n_missed = int(missed.sum())
            row["referral_rate"] = float((mask & ~keep).sum() / n)
            row["n_missed"] = n_missed
            row["n_missed_referred"] = int((missed & ~keep).sum())
            row["miss_rescue_rate"] = (
                float((missed & ~keep).sum() / n_missed) if n_missed else float("nan")
            )
        rows.append(row)

    return pd.DataFrame(rows)


def disparities(table: pd.DataFrame) -> dict[str, float]:
    """Max-minus-min spreads over the cells that cleared every gate.

    Returns an empty dict rather than a spread when fewer than two cells survive. That is
    the honest outcome for an intersectional table on this dataset and it must be reported
    as such -- an intersectional analysis that cannot be powered is a finding about the
    data, not a gap to be filled with an underpowered number.
    """
    usable = table[~table["suppressed"]]
    if len(usable) < 2:
        return {"n_usable_cells": float(len(usable))}

    def spread(column: str) -> float:
        values = usable[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        return float(values.max() - values.min()) if len(values) >= 2 else float("nan")

    out = {
        "n_usable_cells": float(len(usable)),
        "equalized_odds_tpr_gap": spread("escalation_sensitivity"),
        "demographic_parity_gap": spread("predicted_escalate_rate"),
    }
    if "referral_rate" in usable.columns:
        out["referral_burden_gap"] = spread("referral_rate")
    return out


def worst_cell(table: pd.DataFrame) -> pd.Series | None:
    """The powered cell with the lowest escalation sensitivity, or None if none are powered."""
    usable = table[~table["suppressed"]]
    if usable.empty:
        return None
    return usable.loc[usable["escalation_sensitivity"].idxmin()]
