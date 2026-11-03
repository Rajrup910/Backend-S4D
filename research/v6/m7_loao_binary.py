"""Does arXiv 2607.26765's out-of-domain gain show up on our own hold-out archives? CPU, existing predictions only.

    python -m research.v6.m7_loao_binary --n-boot 2000

The paper reports binary malignant-vs-benign ROC-AUC on held-out sources (+0.047 over 4 seeds). V5's `m7` arm took the paper's
physically plausible components (Planckian jitter, CLAHE, optical/grid distortion, blur and noise, 360 degree rotation; no
HSV, HEStain or gamma, under the plan's HSV ban) and ran the source-disjoint test the paper asks for, but it was read on 7-class
Macro-F1. This reads the same runs on the paper's own kind of endpoint: escalation-mass ROC-AUC and pAUC@0.20 on each hold-out,
`m7` minus `control`, with a lesion-grouped paired bootstrap. One seed per arm (42), so this is evidence about a sibling of the
paper's `mix` policy, not a test of `mix` itself. Writes results/v6/m7_loao_binary.json. No test, reserved or external read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research import testguard
from research.v5.d0_brainstorm_diagnostics import pauc

REPO = Path(__file__).resolve().parents[2]
PRED = REPO / "results" / "v5" / "preds"
OUT = REPO / "results" / "v6" / "m7_loao_binary.json"
HOLDOUTS = ("bcn20000", "ham", "mskcc")
ENDPOINTS = {"auroc": roc_auc_score, "pauc20": pauc}


def load(arm: str, holdout: str) -> pd.DataFrame:
    frame = pd.read_csv(PRED / f"{arm}_loao-{holdout}_s42_in22k_v5scr.csv")
    return frame[["image_id", "effective_lesion_id", "age_band", "y_esc", "declared_score"]]


def paired(frame: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """`frame` has y_esc, score_m7, score_control, effective_lesion_id. Resamples lesions, not images."""
    y, m7, ctl = frame["y_esc"].to_numpy(bool), frame["score_m7"].to_numpy(), frame["score_control"].to_numpy()
    groups = pd.factorize(frame["effective_lesion_id"])[0]
    members = pd.Series(np.arange(len(frame))).groupby(groups).apply(lambda s: s.to_numpy()).to_list()
    rng = np.random.default_rng(seed)
    out = {"n_images": int(len(frame)), "n_lesions": len(members), "n_positive_images": int(y.sum())}
    for name, fn in ENDPOINTS.items():
        point = {"control": float(fn(y, ctl)), "m7": float(fn(y, m7))}
        deltas = []
        for _ in range(n_boot):
            picks = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
            if y[picks].min() == y[picks].max():
                continue
            deltas.append(fn(y[picks], m7[picks]) - fn(y[picks], ctl[picks]))
        low, high = np.percentile(deltas, [2.5, 97.5])
        out[name] = {**point, "delta": point["m7"] - point["control"], "ci95": [float(low), float(high)]}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args(argv)
    testguard.block_test_reads("M7 LOAO binary read: development hold-out predictions only")
    report: dict = {"seed": 42, "n_boot": args.n_boot, "test_read": False, "reserved_read": False, "holdouts": {}}
    for i, holdout in enumerate(HOLDOUTS):
        joined = load("control", holdout).merge(load("m7", holdout), on=["image_id", "effective_lesion_id", "age_band", "y_esc"],
                                                suffixes=("_control", "_m7"), validate="one_to_one")
        joined = joined.rename(columns={"declared_score_control": "score_control", "declared_score_m7": "score_m7"})
        report["holdouts"][holdout] = {"all_ages": paired(joined, args.n_boot, 1000 + i)}
        young = joined[joined["age_band"] == "<40"]
        if young["y_esc"].sum() >= 20:
            report["holdouts"][holdout]["under_40"] = paired(young, args.n_boot, 2000 + i)
    for endpoint in ENDPOINTS:
        deltas = [report["holdouts"][h]["all_ages"][endpoint]["delta"] for h in HOLDOUTS]
        report[f"mean_delta_{endpoint}"] = float(np.mean(deltas))
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for holdout, value in report["holdouts"].items():
        for band, row in value.items():
            print(holdout, band, f"n_pos={row['n_positive_images']}",
                  *(f"{e} {row[e]['control']:.4f}->{row[e]['m7']:.4f} d {row[e]['delta']:+.4f} "
                    f"[{row[e]['ci95'][0]:+.4f},{row[e]['ci95'][1]:+.4f}]" for e in ENDPOINTS), sep="  ")
    print("mean delta", {e: round(report[f"mean_delta_{e}"], 4) for e in ENDPOINTS})
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
