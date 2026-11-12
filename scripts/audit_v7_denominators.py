"""V7 denominator ledger and per-archive V6 scorecard, computed from files on disk.

    python scripts/audit_v7_denominators.py            # writes results/v7/denominators.json

Why it exists: the V7 documents quoted "12,235 lesions" and "64 young melanomas". Both are wrong, and a
verifier that only checks that a phrase appears in a document cannot catch that. Every count here is
recomputed from `results/v5/v5_oof_common_matrix.csv` (development rows only) and the V6 fold 1-4
prediction files, so documents can cite this file instead of each other.

Reads development rows only. The test lock is armed, no test, reserved or HIBA label is read. The HIBA
block reads `results/v6/hiba/decisions.csv`, which holds predictions and referral flags and no labels.
The per-archive scorecard is DESCRIPTIVE: folds 1-4 were already read at V6-5 (`results/v6/e_primary.json`).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
from research import testguard  # noqa: E402

testguard.block_test_reads("V7 denominator ledger: development rows only")

MATRIX = REPO / "results" / "v5" / "v5_oof_common_matrix.csv"
V6_TTA = REPO / "results" / "v6" / "preds_tta"
HIBA_DECISIONS = REPO / "results" / "v6" / "hiba" / "decisions.csv"
OUT = REPO / "results" / "v7" / "denominators.json"
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
ESC = [CLASSES.index(c) for c in ("akiec", "bcc", "mel")]
FINALISTS = ("anchor_h7", "panderm_b_h7", "dinov2_b_h7", "anchor384_h7")
READ_FOLDS = (1, 2, 3, 4)
FPR_MAX = 0.20


def counts(frame: pd.DataFrame) -> dict:
    """Images, lesion ids and lesion groups, kept as three separate numbers."""
    return {"images": int(len(frame)), "lesion_ids": int(frame["lesion_id"].nunique()), "lesion_groups": int(frame["group_id"].nunique())}


def young_escalating(frame: pd.DataFrame) -> dict:
    young = frame[(frame["age_band"] == "<40") & (frame["escalation_status"] == 1)]
    return {**counts(young),
            "lesion_groups_by_class": {k: int(v) for k, v in young.groupby("true_dx")["group_id"].nunique().items()},
            "images_by_class": {k: int(v) for k, v in young["true_dx"].value_counts().items()},
            "lesion_groups_by_archive": {k: int(v) for k, v in young.groupby("archive")["group_id"].nunique().items()}}


def denominators() -> dict:
    cols = ["image_id", "lesion_id", "group_id", "fold", "true_dx", "age_band", "escalation_status", "archive"]
    rows = pd.read_csv(MATRIX, usecols=cols, low_memory=False)
    read = rows[rows["fold"].isin(READ_FOLDS)]
    return {
        "source": str(MATRIX.relative_to(REPO)).replace("\\", "/"),
        "unit_note": "images are rows; lesion_groups (group_id) is the bootstrap unit; no patient count exists for HAM or BCN",
        "development_all_folds": {**counts(rows), "by_archive": {a: counts(g) for a, g in rows.groupby("archive")},
                                  "groups_spanning_archives": int((rows.groupby("group_id")["archive"].nunique() > 1).sum()),
                                  "under40_escalating": young_escalating(rows)},
        "folds_1_to_4": {**counts(read), "by_archive": {a: counts(g) for a, g in read.groupby("archive")},
                         "under40": counts(read[read["age_band"] == "<40"]),
                         "under40_escalating": young_escalating(read),
                         "classes_present_by_archive": {a: sorted(g["true_dx"].unique()) for a, g in read.groupby("archive")},
                         "class_images_by_archive": {a: {k: int(v) for k, v in g["true_dx"].value_counts().items()} for a, g in read.groupby("archive")},
                         "escalating_share_by_archive": {a: round(float(g["escalation_status"].mean()), 4) for a, g in read.groupby("archive")}},
    }


def v6_vote() -> pd.DataFrame:
    """The banked V6 uniform vote on folds 1-4: SWA weights, 24 views pooled by inverse entropy (`research/v5/tta_v5.py`)."""
    prob_cols = [f"p_{c}" for c in CLASSES]
    members = {}
    for name in FINALISTS:
        files = sorted(glob.glob(str(V6_TTA / f"{name}_f[1-4]_s42*v64_swad_tta.csv")))
        if len(files) != len(READ_FOLDS):
            raise SystemExit(f"{name}: expected {len(READ_FOLDS)} fold files, found {len(files)}")
        members[name] = pd.concat([pd.read_csv(f) for f in files]).set_index("image_id")
    base = members[FINALISTS[0]]
    probs = sum(members[n].loc[base.index, prob_cols].to_numpy() for n in FINALISTS) / len(FINALISTS)
    out = base[["group_id", "age_band", "archive", "y_true", "y_esc"]].copy()
    out[list(CLASSES)] = probs
    return out


def score(y: np.ndarray, esc: np.ndarray, probs: np.ndarray) -> dict:
    pred, present = probs.argmax(1), sorted(set(y.tolist()))
    both = len(np.unique(esc)) == 2
    return {"n_images": int(len(y)), "classes_present": int(len(present)),
            "macro_f1_all_7_labels": float(f1_score(y, pred, labels=list(range(len(CLASSES))), average="macro", zero_division=0)),
            "macro_f1_present_classes": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "escalation_pauc_mcclish_fpr20": float(roc_auc_score(esc, probs[:, ESC].sum(1), max_fpr=FPR_MAX)) if both else None}


def scorecard() -> dict:
    v6 = v6_vote()
    v5_cols = {c: [f"prob_composite_tta_s{s}_{c}" for s in (42, 43, 44)] for c in CLASSES}
    matrix = pd.read_csv(MATRIX, usecols=["image_id", *[x for v in v5_cols.values() for x in v]], low_memory=False).set_index("image_id").loc[v6.index]
    v5 = np.stack([matrix[v5_cols[c]].mean(axis=1).to_numpy() for c in CLASSES], axis=1)
    y, esc, p6 = v6["y_true"].to_numpy(), v6["y_esc"].to_numpy().astype(int), v6[list(CLASSES)].to_numpy()
    archive, band = v6["archive"].to_numpy(), v6["age_band"].to_numpy()
    out = {"status": "DESCRIPTIVE reread of folds 1-4, already read at V6-5; image level; no interval",
           "systems": {"v6": "uniform vote of the four finalists, banked predictions", "v5": "mean of composite_tta seeds 42, 43, 44"},
           "pooled": {"v6": score(y, esc, p6), "v5": score(y, esc, v5)}, "by_archive": {}, "under40_by_archive": {}}
    for name in sorted(set(archive)):
        mask = archive == name
        out["by_archive"][name] = {"v6": score(y[mask], esc[mask], p6[mask]), "v5": score(y[mask], esc[mask], v5[mask])}
        young = mask & (band == "<40")
        out["under40_by_archive"][name] = {"n_images": int(young.sum()), "n_escalating_images": int(esc[young].sum()),
                                           "v6_escalation_pauc": score(y[young], esc[young], p6[young])["escalation_pauc_mcclish_fpr20"],
                                           "v5_escalation_pauc": score(y[young], esc[young], v5[young])["escalation_pauc_mcclish_fpr20"]}
    return out


def hiba_workload() -> dict:
    """Label-free: share flagged = called escalating or referred. The 0.20 budget fixes only the referred part."""
    decisions = pd.read_csv(HIBA_DECISIONS)
    out = {"source": str(HIBA_DECISIONS.relative_to(REPO)).replace("\\", "/"), "lesions": int(len(decisions)), "labels_read": False, "systems": {}}
    for system in ("v6_vote", "combined_vote", "v5", "v1"):
        called, referred = decisions[f"{system}_pred"].isin(ESC), decisions[f"{system}_referred"].astype(bool)
        out["systems"][system] = {"called_escalating": round(float(called.mean()), 4), "referred": round(float(referred.mean()), 4),
                                  "flagged_total": round(float((called | referred).mean()), 4)}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recompute V7 denominators and the per-archive V6 scorecard from files on disk.")
    parser.add_argument("--out", type=Path, default=OUT, help="output JSON path")
    args = parser.parse_args(argv)
    report = {"denominators": denominators(), "v6_per_archive_scorecard": scorecard(), "hiba_workload": hiba_workload(),
              "test_read": False, "reserved_read": False, "hiba_labels_read": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    read = report["denominators"]["folds_1_to_4"]
    print(json.dumps({"folds_1_to_4": {k: read[k] for k in ("images", "lesion_ids", "lesion_groups")}, "under40_escalating": read["under40_escalating"],
                      "macro_f1_present_classes_v6": {a: round(v["v6"]["macro_f1_present_classes"], 4) for a, v in report["v6_per_archive_scorecard"]["by_archive"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
