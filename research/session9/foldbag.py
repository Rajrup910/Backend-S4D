"""Rung A8 -- the 30-member fold-bagged ensemble, and the honest account of its calibrator.

A8 is the one Macro-F1 path the S6 exhaustion result did not kill: six architectures x
five folds gives 30 members instead of 6, and `research/oof/results/diagnose_shift_report.md`
already measured fold-bagging beating the frozen full-train checkpoint on validation for
five of the six architectures (+0.020 to +0.037 Macro-F1). That is evidence it may help,
not a promise, and the plan pre-registers it two-sided.

**Uniform, with no member selection.** The plan's original wording was "member set selected
on val". This module deliberately does not do that. Session 1 already measured Caruana
greedy selection over 8 members losing to the uniform average on held-out data, and
selecting 30 members would give the winner's curse a far larger surface. Uniform bagging
takes no selection decision at all, which is strictly stronger discipline and removes the
need for a second 30-model extraction over validation. The choice is recorded in the plan
so it cannot be read as a retrospective simplification.

**The calibrator carries a real caveat.** A7-oof applies a Dirichlet map fitted on the
6-architecture OOF matrix. A8 needs a map for a 30-member bag, and no out-of-fold matrix
for that bag can exist: for any training row, 24 of the 30 members saw it. So A8 reuses
the same OOF-fitted Dirichlet map, fitted on the 6-member OOF ensemble. Bagging 30 members
pulls the maximum probability down further than bagging 6 (the same mechanism that makes
the 6-CNN soft vote under-confident), so a map fitted on the 6-member distribution is being
applied to a slightly flatter one. The direction of that mismatch is knowable -- it will
under-sharpen A8 -- and it is stated in the plan and the report rather than being papered
over. A8 is therefore a lower bound on what a properly-calibrated fold-bag would do.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.paths import load_class_mapping, resolve

#: Where `research/oof/extract_foldbag_test.py` stages the 30 test matrices.
FOLDBAG_DIR = "research/predictions_foldbag_tta"
FOLDS = (0, 1, 2, 3, 4)


def fold_paths(archs: tuple[str, ...], folds: tuple[int, ...] = FOLDS,
               out_dir: str = FOLDBAG_DIR) -> list[Path]:
    root = resolve(out_dir)
    return [root / "_folds" / f"fold{k}" / f"{arch}_test.csv" for arch in archs for k in folds]


def available(archs: tuple[str, ...], folds: tuple[int, ...] = FOLDS,
              out_dir: str = FOLDBAG_DIR) -> tuple[bool, list[str]]:
    """(all present, list of missing paths). Used to decide whether A8 can be emitted."""
    missing = [p.as_posix() for p in fold_paths(archs, folds, out_dir) if not p.is_file()]
    return (not missing, missing)


def load_members(
    archs: tuple[str, ...],
    image_ids: np.ndarray,
    folds: tuple[int, ...] = FOLDS,
    out_dir: str = FOLDBAG_DIR,
) -> np.ndarray:
    """(N, 30, C) probabilities aligned to `image_ids`, in (arch, fold) order.

    Alignment is asserted against the caller's image order rather than assumed, because a
    silently mis-ordered member would produce a plausible-looking ensemble that is simply
    wrong -- the failure mode `research/ensembling/data.py` guards for the frozen matrices.
    """
    class_codes = load_class_mapping().codes
    columns = [f"p_{c}" for c in class_codes]
    members = []
    for arch in archs:
        for fold in folds:
            path = resolve(out_dir) / "_folds" / f"fold{fold}" / f"{arch}_test.csv"
            frame = pd.read_csv(path).sort_values("image_id").reset_index(drop=True)
            if not np.array_equal(frame["image_id"].to_numpy().astype(str), image_ids.astype(str)):
                raise ValueError(
                    f"{path} is not aligned to the frozen test matrix "
                    f"({len(frame)} rows vs {len(image_ids)})"
                )
            probs = frame[columns].to_numpy(dtype=np.float64)
            if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
                raise ValueError(f"{path} has rows whose probabilities do not sum to 1")
            members.append(probs)
    return np.stack(members, axis=1)


def bag(members: np.ndarray) -> np.ndarray:
    """Uniform arithmetic mean over the member axis -- the same rule as the 6-CNN soft vote."""
    return members.mean(axis=1)
