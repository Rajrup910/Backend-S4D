"""Independent recomputation of the V6 reference read on folds 1-4 (12 Nov 2026).

Written from scratch: own loading, own vote, sklearn metrics, own McClish pAUC, own grouped bootstrap.
It does NOT import any project module. Labels come from `y_true`/`y_esc` in the banked files, cross-checked
against `ml/data/manifest_v4.csv`. These rows were read at V6-5; this re-read is a reproduction, not a new read.
Run from the repo root: python -I results/v7/audit_2026-11-12/independent_v6_reference.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "independent_v6_reference.json"
MEMBERS = ("anchor_h7_f{f}_s42_in22k_v64", "panderm_b_h7_f{f}_s42_v64", "dinov2_b_h7_f{f}_s42_v64",
           "anchor384_h7_f{f}_s42_in22k_v64")
P = ["p_akiec", "p_bcc", "p_bkl", "p_df", "p_mel", "p_nv", "p_vasc"]   # alphabetical = class_index_7 order
ESC = [0, 1, 4]                                                         # akiec, bcc, mel


def mcclish(y: np.ndarray, s: np.ndarray, max_fpr: float = 0.2) -> float:
    return float(roc_auc_score(y, s, max_fpr=max_fpr))   # sklearn returns the McClish-standardised value


def grouped_ci(fn, groups: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([idx_by[k] for k in pick])
        try:
            vals.append(fn(rows))
        except ValueError:
            continue
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main() -> int:
    frames = []
    for f in (1, 2, 3, 4):
        per = []
        for mem in MEMBERS:
            d = pd.read_csv(REPO / "results/v6/preds_tta" / f"{mem.format(f=f)}_swad_tta.csv")
            per.append(d.set_index("image_id"))
        ids = per[0].index
        assert all(set(p.index) == set(ids) for p in per), "members disagree on rows"
        probs = np.mean([p.loc[ids, P].to_numpy() for p in per], axis=0)
        base = per[0].loc[ids, ["group_id", "age_band", "archive", "y_true", "y_esc"]].copy()
        base[P] = probs
        base["fold"] = f
        frames.append(base.reset_index())
    d = pd.concat(frames, ignore_index=True)
    man = pd.read_csv(REPO / "ml/data/manifest_v4.csv", low_memory=False).set_index("image_id")
    label_ok = bool((man.loc[d["image_id"], "class_index_7"].to_numpy() == d["y_true"].to_numpy()).all())
    y, prob = d["y_true"].to_numpy(), d[P].to_numpy()
    pred = prob.argmax(1)
    esc_score = prob[:, ESC].sum(1)
    yesc = d["y_esc"].astype(bool).to_numpy()
    groups = d["group_id"].astype(str).to_numpy()
    u40 = d["age_band"].eq("<40").to_numpy()
    res = {
        "rows": int(len(d)), "groups": int(d["group_id"].nunique()), "labels_match_manifest": label_ok,
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(7)))),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float((y == pred).mean()),
        "pauc_all": mcclish(yesc, esc_score),
        "pauc_u40": mcclish(yesc[u40], esc_score[u40]),
        "u40_rows": int(u40.sum()),
        "u40_escalating_images": int((yesc & u40).sum()),
        "u40_escalating_groups": int(d.loc[yesc & u40, "group_id"].nunique()),
        "u40_negative_images": int((~yesc & u40).sum()),
        "escalating_images": int(yesc.sum()),
        "escalation_sensitivity_argmax_all": float(np.isin(pred[yesc], ESC).mean()),
        "escalation_sensitivity_argmax_u40": float(np.isin(pred[yesc & u40], ESC).mean()),
        "per_class_f1": dict(zip(["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                                 map(float, f1_score(y, pred, average=None, labels=list(range(7)))))),
        "by_archive_macro_f1": {a: float(f1_score(y[w], pred[w], average="macro", labels=list(range(7))))
                                for a, w in ((a, d["archive"].eq(a).to_numpy()) for a in sorted(d["archive"].unique()))},
        "by_band_pauc": {b: mcclish(yesc[w], esc_score[w]) for b, w in
                         ((b, d["age_band"].eq(b).to_numpy()) for b in ("<40", "40-59", "60+"))},
    }
    res["macro_f1_ci_grouped"] = grouped_ci(lambda r: f1_score(y[r], pred[r], average="macro", labels=list(range(7))), groups)
    gu = groups[u40]
    res["pauc_u40_ci_grouped"] = grouped_ci(lambda r: mcclish(yesc[u40][r], esc_score[u40][r]), gu)
    res["recorded_macro_f1"] = json.loads((REPO / "results/v6/e_primary.json").read_text(encoding="utf-8"))["descriptive"]["v6_endpoints"]["macro_f1"]
    res["reproduces_recorded"] = abs(res["macro_f1"] - res["recorded_macro_f1"]) < 1e-9
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
