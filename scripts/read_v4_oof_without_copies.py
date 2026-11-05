"""How much did copies across folds move V4's cross-fitted scores? Descriptive; frozen V4 files are not changed. CPU. Development rows only.

    python scripts/read_v4_oof_without_copies.py

V4's OOF has no young extras, so its only exposure is a row whose copy, or other photo of the same lesion, sat in another fold and so in that row's training data (>= 30 RANSAC inliers, results/v6/screens/leak_audit_keypoint_pairs.csv). Whole lesions are removed. Removing rows moves a score for ordinary reasons too, so the read also compares the removed rows with the rest. The detector finds only just over half of known same-lesion pairs, so this is a lower bound on exposure. Writes results/v4/kfold/oof_without_copies.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ml.evaluation.metrics import compute_metrics  # noqa: E402
from research import testguard  # noqa: E402
from research.v5.d0_brainstorm_diagnostics import pauc  # noqa: E402

OOF = REPO / "results/v4/kfold/oof_predictions.csv"
PAIRS = REPO / "results/v6/screens/leak_audit_keypoint_pairs.csv"
OUT = REPO / "results/v4/kfold/oof_without_copies.json"
INLIERS = 30
N_BOOT, SEED = 2000, 20261105
PROBS = [f"p_{c}" for c in ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")]
ESCALATING = {0, 1, 4}


def levels(frame: pd.DataFrame) -> dict:
    y, s = frame["y_esc"].astype(bool).to_numpy(), frame["escalation_mass"].to_numpy()
    u40 = frame["age_band"].eq("<40").to_numpy()
    called = frame["pred_index"].isin(ESCALATING).to_numpy()
    metrics = compute_metrics(frame["y_true"].to_numpy(), frame["pred_index"].to_numpy(), frame[PROBS].to_numpy())
    return {"rows": int(len(frame)), "lesions": int(frame["effective_lesion_id"].nunique()), "escalating_rows": int(y.sum()), "macro_f1": float(metrics["macro_f1"]), "pauc_all": float(pauc(y, s)), "pauc_u40": float(pauc(y[u40], s[u40])) if len(np.unique(y[u40])) == 2 else None, "escalation_sensitivity_argmax": float(called[y].mean()) if y.any() else None, "accuracy": float((frame["pred_index"] == frame["y_true"]).mean())}


def paired_difference(frame: pd.DataFrame, removed: np.ndarray, metric) -> dict:
    """All rows minus rows kept, resampling lesions; the interval says if the removal moves the level beyond resampling noise."""
    rng = np.random.default_rng(SEED)
    codes = pd.factorize(frame["effective_lesion_id"])[0]
    index = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
    point = metric(frame) - metric(frame[~removed])
    values = []
    for _ in range(N_BOOT):
        pick = np.concatenate([index[g] for g in rng.integers(0, len(index), len(index))])
        sample, flags = frame.iloc[pick], removed[pick]
        try:
            values.append(metric(sample) - metric(sample[~flags]))
        except ValueError:
            continue
    return {"point": float(point), "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]}


def main() -> int:
    testguard.block_test_reads("V4 OOF without copies: development rows only")
    frame = pd.read_csv(OOF, dtype={"image_id": str})
    pairs = pd.read_csv(PAIRS, dtype={"a_id": str})
    pairs = pairs[(pairs["inliers"] >= INLIERS) & (pairs["against"] == "other_fold")]
    lesions = set(frame.loc[frame["image_id"].isin(set(pairs["a_id"])), "effective_lesion_id"])
    removed = frame["effective_lesion_id"].isin(lesions).to_numpy()
    u40 = frame["age_band"].eq("<40")

    def macro(f: pd.DataFrame) -> float:
        return float(compute_metrics(f["y_true"].to_numpy(), f["pred_index"].to_numpy(), f[PROBS].to_numpy())["macro_f1"])

    def all_age(f: pd.DataFrame) -> float:
        return float(pauc(f["y_esc"].astype(bool).to_numpy(), f["escalation_mass"].to_numpy()))

    def young(f: pd.DataFrame) -> float:
        g = f[f["age_band"].eq("<40")]
        return float(pauc(g["y_esc"].astype(bool).to_numpy(), g["escalation_mass"].to_numpy()))

    report = {"question": "How much did copies across folds move V4's cross-fitted scores?", "source": "results/v4/kfold/oof_predictions.csv", "rule": f">= {INLIERS} RANSAC inliers to a row in another fold; whole lesions", "limit": "the keypoint search compares each row with its 5 nearest neighbours and catches only just over half of known same-lesion pairs: a lower bound", "rows_named": int(pairs["a_id"].nunique()), "rows_removed": int(removed.sum()), "u40_rows_removed": int((removed & u40.to_numpy()).sum()), "u40_escalating_rows_removed": int((removed & u40.to_numpy() & frame["y_esc"].astype(bool).to_numpy()).sum()), "by_archive": frame.loc[removed, "archive"].value_counts().to_dict(), "by_fold": frame.loc[removed, "fold"].value_counts().sort_index().to_dict(), "all_rows": levels(frame), "without_copies": levels(frame[~removed]), "the_removed_rows": levels(frame[removed]) if removed.any() else None, "all_minus_without": {"macro_f1": paired_difference(frame, removed, macro), "pauc_all": paired_difference(frame, removed, all_age), "pauc_u40": paired_difference(frame, removed, young)}}
    OUT.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rows_named", "rows_removed", "u40_rows_removed", "u40_escalating_rows_removed", "by_archive", "all_rows", "without_copies", "the_removed_rows", "all_minus_without")}, indent=0, default=float))
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
