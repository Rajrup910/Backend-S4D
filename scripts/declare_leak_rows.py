"""Freeze, before any V6 prediction on folds 1-4 exists, the development rows a leak audit found at risk, for a with-and-without read at V6-5. Metadata only.

    python scripts/declare_leak_rows.py

Not a gate, and E-primary is unchanged: the training rows stay as frozen, because changing them would split V6's runs across two pools and across V5's. The read beside E-primary drops these
lesion groups from both sides. Declared rule: a row's lesion group is at risk if any of its images (a) is a young extra under another name (`leak_audit_names.json`), or (b) has at least 20 RANSAC
inliers to a training image (`leak_audit_keypoint_pairs.csv`). 20 is above the random-pair 99th percentile (10) and below the 99.9th (30); it was fixed from the controls, and the fold-0 gain did not
move across 10, 20 and 40. The detector is known to miss copies, so this list is a lower bound.
Writes results/v6/leak_sensitivity_rows.csv and results/v6/leak_sensitivity_declaration.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import testguard  # noqa: E402
from research.v4.s71_kfold import load_assignments  # noqa: E402

INLIERS = 20
SCREENS = REPO / "results/v6/screens"
ROWS = REPO / "results/v6/leak_sensitivity_rows.csv"
DECLARATION = REPO / "results/v6/leak_sensitivity_declaration.json"


def main() -> int:
    testguard.block_test_reads("leak row declaration: metadata only")
    folds = load_assignments()[["image_id", "fold", "group_id", "archive", "age_band", "class_7"]].copy()
    folds["image_id"] = folds["image_id"].astype(str)
    named = json.loads((SCREENS / "leak_audit_names.json").read_text(encoding="utf-8"))["young_extras"]["image_ids"]
    pairs = pd.read_csv(SCREENS / "leak_audit_keypoint_pairs.csv", dtype={"a_id": str, "b_id": str})
    pairs = pairs[pairs["inliers"] >= INLIERS]
    reasons: dict[str, set[str]] = {}
    for image in named:
        reasons.setdefault(image, set()).add("name_copy_of_young_extra")
    for row in pairs.itertuples():
        reasons.setdefault(row.a_id, set()).add(f"keypoints_{row.against}")
    flagged_groups = set(folds.loc[folds["image_id"].isin(reasons), "group_id"])
    at_risk = folds[folds["group_id"].isin(flagged_groups)].copy()
    at_risk["reason"] = [";".join(sorted(reasons.get(i, {"same_lesion_group_as_a_flagged_row"}))) for i in at_risk["image_id"]]
    at_risk.sort_values("image_id").to_csv(ROWS, index=False)
    declaration = {"declared": "2026-11-05, before any V6 prediction on folds 1-4", "rule": f"name copy of a young extra, or >= {INLIERS} RANSAC inliers to a training image; whole lesion groups",
                   "role": "a read beside E-primary, on both sides; not a gate, and no training row changes", "rows": len(at_risk), "groups": len(flagged_groups), "share_of_development": len(at_risk) / len(folds),
                   "by_fold": at_risk["fold"].value_counts().sort_index().to_dict(), "by_archive": at_risk["archive"].value_counts().to_dict(), "by_band": at_risk["age_band"].value_counts().to_dict(), "by_class": at_risk["class_7"].value_counts().to_dict(),
                   "limit": "the detector misses copies: at 256 px it scores some known copies below the bar, so this list is a lower bound", "rows_sha256": hashlib.sha256(ROWS.read_bytes()).hexdigest()}
    DECLARATION.write_text(json.dumps(declaration, indent=2, default=str), encoding="utf-8")
    print(json.dumps(declaration, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
