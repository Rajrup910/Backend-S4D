"""Freeze, before any V6 prediction on folds 1-4 exists, the development rows a leak audit found at risk, for a with-and-without read at V6-5. Metadata only.

    python scripts/declare_leak_rows.py

The training rows stay as frozen, because changing them would split V6's runs across two pools and across V5's; V5 itself trained on the same extras. The read drops these lesion groups from both sides, and a system must
pass on both row sets (`scripts/read_leak_sensitivity.py`, owner decision 5 Nov 2026). Declared rule: a row's lesion group is at risk if any of its images (a) is a young extra under another name (`leak_audit_names.json`), or
(b) has at least 30 RANSAC inliers to a training image (`leak_audit_keypoint_pairs.csv`). The bar comes from the controls and by eye: all 35 known copies score 87 or more, the median random pair 0 and its 99th percentile 9, and the pairs sampled
from 30-45 inliers were almost all one lesion. The detector compares only each row's 5 nearest neighbours by DINOv2, so this is not proof that nothing else exists.
Amended 5 Nov 2026 by owner decision, before any fold 1-4 prediction: (c) rows a second audit flagged by lesion-id or patient links or by a second pixel detector (`results/v6/leak_sensitivity_side_audit_rows.csv`).
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

INLIERS = 30
SCREENS = REPO / "results/v6/screens"
ROWS = REPO / "results/v6/leak_sensitivity_rows.csv"
DECLARATION = REPO / "results/v6/leak_sensitivity_declaration.json"
SIDE_AUDIT = REPO / "results/v6/leak_sensitivity_side_audit_rows.csv"
PREVIOUS = {"rows": 380, "groups": 244, "rows_sha256": "7feaeadde15136dc3362fb0792a7e01b55eb0b4a3707f2b49a639864a7c6e483"}


def fold0_read(at_risk: set[str]) -> dict:
    """Descriptive: the go/no-go's own votes, with and without the at-risk rows. Fold 0 only."""
    import numpy as np

    from research.v6 import gate, stage_rules as st, system_reads as sr

    reads = [st.fold0_read(gate.pred_path(n, 42, t, sr.V6_TTA_DIR, checkpoint="swad_tta")) for n, t in {"anchor_h7": "v62y", "panderm_b_h7": "v63", "dinov2_b_h7": "v63", "anchor384_h7": "v63"}.items()]
    first = reads[0]
    rows, models = sr.load_matrix((sr.V5_SYSTEM, *sr.V5_TTA_SEEDS))
    position = {image: i for i, image in enumerate(rows.image_id)}
    where = np.array([position[i] for i in first["image_id"]])
    v6 = np.mean([r["p"] for r in reads], axis=0)
    combined = np.mean([*(r["p"] for r in reads), *(models[m][where] for m in sr.V5_TTA_SEEDS)], axis=0)
    v5 = models[sr.V5_SYSTEM][where]
    out = {}
    for name, keep in (("all_rows", np.ones(len(v6), bool)), ("without_at_risk_rows", ~np.isin(first["image_id"], list(at_risk)))):
        young = keep & first["young"]

        def ends(p):
            return {"macro_f1": sr._mf1(first["y"][keep], p[keep]), "pauc_all": sr._pauc(first["esc"][keep], p[keep]), "pauc_u40": sr._pauc(first["esc"][young], p[young])}

        base = ends(v5)
        out[name] = {"rows": int(keep.sum()), "u40_escalating_rows": int((young & first["esc"]).sum()), "v5": base, "v6_minus_v5": {m: v - base[m] for m, v in ends(v6).items()}, "combined_minus_v5": {m: v - base[m] for m, v in ends(combined).items()}}
    return out


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
    side = pd.read_csv(SIDE_AUDIT, dtype={"image_id": str, "group_id": str})
    for row in side.itertuples():
        reasons.setdefault(row.image_id, set()).add(row.reason)
    flagged_groups = set(folds.loc[folds["image_id"].isin(reasons), "group_id"])
    at_risk = folds[folds["group_id"].isin(flagged_groups)].copy()
    at_risk["reason"] = [";".join(sorted(reasons.get(i, {"same_lesion_group_as_a_flagged_row"}))) for i in at_risk["image_id"]]
    at_risk.sort_values("image_id").to_csv(ROWS, index=False)
    declaration = {"declared": "2026-11-05, before any V6 prediction on folds 1-4", "rule": f"name copy of a young extra, or >= {INLIERS} RANSAC inliers to a training image; whole lesion groups",
                   "role": "binding at V6-5: a system must pass E-primary and no-harm with and without these rows; no training row changes", "rows": len(at_risk), "groups": len(flagged_groups), "share_of_development": len(at_risk) / len(folds),
                   "by_fold": at_risk["fold"].value_counts().sort_index().to_dict(), "by_archive": at_risk["archive"].value_counts().to_dict(), "by_band": at_risk["age_band"].value_counts().to_dict(), "by_class": at_risk["class_7"].value_counts().to_dict(),
                   "limit": "checks only the 5 nearest neighbours by DINOv2 at 256 px, and test, reserved and HIBA pixels were not compared here", "rows_sha256": hashlib.sha256(ROWS.read_bytes()).hexdigest()}
    declaration["rule"] = f"name copy of a young extra; or >= {INLIERS} RANSAC inliers to a training image; or listed by the second audit for lesion-id or patient links or a second pixel detector ({SIDE_AUDIT.name}); whole lesion groups"
    declaration["amended"] = {"date": "2026-11-05, owner decision, before any V6 prediction on folds 1-4", "added_rows": len(side), "added_groups": int(side["group_id"].nunique()), "previous": PREVIOUS}
    declaration["fold0_v6_vote_minus_v5"] = fold0_read(set(at_risk["image_id"]))
    DECLARATION.write_text(json.dumps(declaration, indent=2, default=str), encoding="utf-8")
    print(json.dumps(declaration, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
