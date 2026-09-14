"""S33 -- the per-architecture (N, 6, 7) probability tensor, aligned to a panel's rows.

S29's panels store only the merged soft-vote, so the two ensemble-disagreement scores
(`ensemble_variance`, `mutual_information`) were unavailable from a panel alone. S30's
frozen plan recorded that gap and, on the evidence then available, restricted the F2 and F4
families to HAM-OOF because the assembled BCN/MSKCC ensemble file carries no per-member
columns.

This module closes the gap: the per-architecture CSVs exist for **all five** cohorts
(`results/external/predictions/{arch}_{cohort}.csv` for the external pair, alongside the
HAM and PAD directories the panel builder already reads), so disagreement is in fact
computable everywhere.

**This does not change any confirmatory family.** F2 and F4 stay HAM-OOF-only exactly as
frozen in `results/v2/analysis_plan.json`. Widening a confirmatory family after the freeze
is precisely what pre-registration exists to prevent; the plan's
`score_library.members.disagreement.availability` note is superseded as a statement about
*capability*, not as a licence to enlarge the declared families. The extra cohorts are
available to the exploratory family (F5) only.

Disagreement is computed on the members' own **raw** outputs, not on calibrated
probabilities: it measures how much the six architectures disagree with each other, which
is a property of the ensemble before any post-hoc map is applied.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.paths import load_class_mapping, resolve
from research.ensembling.data import ARCHS

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASS_CODES: tuple[str, ...] = tuple(load_class_mapping().codes)

#: cohort -> (directory, filename template). `{arch}` is substituted per architecture.
SOURCES: dict[str, tuple[str, str]] = {
    "ham_oof": ("research/predictions_oof_tta", "{arch}_train.csv"),
    "ham_val": ("research/predictions_tta", "{arch}_val.csv"),
    "pad": ("research/predictions_pad", "{arch}.csv"),
    "bcn20000": ("results/external/predictions", "{arch}_bcn20000.csv"),
    "mskcc": ("results/external/predictions", "{arch}_mskcc.csv"),
}


def load_member_probs(cohort: str, image_ids: np.ndarray, archs: tuple[str, ...] = ARCHS) -> np.ndarray:
    """Return `(N, n_arch, 7)` raw member probabilities, row-aligned to `image_ids`.

    Alignment is by explicit reindex on `image_id`, never by assuming the per-arch files
    share the panel's row order -- the panel builder sorts some cohorts and not others, and
    a silent misalignment here would corrupt every disagreement score without raising.
    """
    if cohort not in SOURCES:
        raise KeyError(f"no per-architecture source registered for cohort {cohort!r}; "
                       f"known: {sorted(SOURCES)}")
    directory, template = SOURCES[cohort]
    wanted = pd.Index([str(i) for i in image_ids], name="image_id")
    prob_cols = [f"p_{c}" for c in CLASS_CODES]

    stacked = []
    for arch in archs:
        path = resolve(directory) / template.format(arch=arch)
        if not path.is_file():
            raise FileNotFoundError(f"missing per-architecture file {path}")
        frame = pd.read_csv(path)
        frame["image_id"] = frame["image_id"].astype(str)
        frame = frame.set_index("image_id")

        missing = wanted.difference(frame.index)
        if len(missing):
            raise ValueError(
                f"{cohort}/{arch}: {len(missing)} panel image_ids absent from {path.name} "
                f"(first few: {list(missing[:3])})"
            )
        stacked.append(frame.reindex(wanted)[prob_cols].to_numpy(dtype=float))

    tensor = np.stack(stacked, axis=1)  # (N, n_arch, 7)

    row_sums = tensor.sum(axis=2)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        worst = float(np.abs(row_sums - 1.0).max())
        raise ValueError(f"{cohort}: member probability rows do not sum to 1 (max |sum-1| = {worst:.2e})")
    return tensor


def available_cohorts() -> list[str]:
    """Cohorts whose full per-architecture set is physically present on disk."""
    found = []
    for cohort, (directory, template) in SOURCES.items():
        if all((resolve(directory) / template.format(arch=a)).is_file() for a in ARCHS):
            found.append(cohort)
    return found


if __name__ == "__main__":
    print("cohorts with a complete per-architecture set:")
    for cohort in available_cohorts():
        panel = pd.read_csv(REPO_ROOT / "results" / "v2" / "panels" / f"{cohort}.csv")
        tensor = load_member_probs(cohort, panel["image_id"].to_numpy())
        print(f"  {cohort:10s} {tensor.shape}")
