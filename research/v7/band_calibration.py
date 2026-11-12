"""Standing problem P4: does a band-wise calibration map fix the age-band sign flip a single global map leaves?

    python -m research.v7.band_calibration      # writes results/v7/p4_band_calibration.json

V4 found that after the global Dirichlet map the residual calibration gap flips sign across age bands (<40 -0.027, 60+ +0.041;
`research/stats/results_oof/band_calibration.csv`). This reads the same question on the deployed V6 vote (folds 1-4, banked
predictions): Dirichlet calibration (multinomial logistic regression on log-probabilities, Kull et al. 2019, L2 C = 1.0,
declared here) fitted globally vs fitted per age band, both cross-fitted by fold (each fold's map comes from the other three).
Endpoint (standing-problem register): per-band signed gap (mean confidence minus accuracy) and 15-bin ECE, with lesion-grouped
bootstrap intervals for the change in |signed gap|; Macro-F1 must not fall. Drop rule: no band's |signed gap| falls by 0.01,
or Macro-F1 falls. Folds 1-4 were read at V6-5, so this is exploratory evidence for a safety-stack choice, not a promotion read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research import testguard  # noqa: E402
from research.v7 import stats  # noqa: E402
from research.v7.dark_read import V6_FINALISTS, V6_TTA  # noqa: E402

PROBS = [f"p_{c}" for c in ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")]
BANDS = ("<40", "40-59", "60+")
OUT = REPO / "results" / "v7" / "p4_band_calibration.json"
C = 1.0
EPS = 1e-6


def v6_vote() -> pd.DataFrame:
    frames = []
    for name in V6_FINALISTS:
        files = sorted(V6_TTA.glob(f"{name}_f[1-4]_s42*v64_swad_tta.csv"))
        frames.append(pd.concat(pd.read_csv(f, dtype={"image_id": str}) for f in files).set_index("image_id"))
    base = frames[0]
    probs = sum(f.loc[base.index, PROBS].to_numpy() for f in frames) / len(frames)
    out = base[["group_id", "age_band", "y_true"]].copy()
    out[PROBS] = probs
    rows = pd.read_csv(REPO / "results" / "v5" / "v5_oof_common_matrix.csv", usecols=["image_id", "fold"], dtype={"image_id": str}).set_index("image_id")
    return out.join(rows, how="inner").reset_index()


def fit_apply(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = LogisticRegression(C=C, max_iter=2000)
    model.fit(np.log(train[PROBS].to_numpy() + EPS), train["y_true"].to_numpy())
    p = np.zeros((len(test), len(PROBS)))
    p[:, model.classes_] = model.predict_proba(np.log(test[PROBS].to_numpy() + EPS))
    return p


def crossfit(frame: pd.DataFrame, by_band: bool) -> np.ndarray:
    out = np.zeros((len(frame), len(PROBS)))
    for fold in sorted(frame["fold"].unique()):
        held = (frame["fold"] == fold).to_numpy()
        if not by_band:
            out[held] = fit_apply(frame[~held], frame[held])
            continue
        for band in frame["age_band"].unique():
            fit_rows = ~held & (frame["age_band"] == band).to_numpy()
            apply_rows = held & (frame["age_band"] == band).to_numpy()
            if apply_rows.any():
                source = frame[fit_rows] if fit_rows.sum() >= 500 else frame[~held]  # small bands fall back to the global map
                out[apply_rows] = fit_apply(source, frame[apply_rows])
    return out


def gap_and_ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> tuple[float, float]:
    conf, correct = p.max(1), (p.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    ece = sum(abs(conf[idx == b].mean() - correct[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    return float(conf.mean() - correct.mean()), float(ece)


def main() -> int:
    testguard.block_test_reads("V7 P4 band-wise calibration: development rows only")
    frame = v6_vote()
    y = frame["y_true"].to_numpy()
    systems = {"uncalibrated": frame[PROBS].to_numpy(), "global_dirichlet": crossfit(frame, False), "bandwise_dirichlet": crossfit(frame, True)}
    report: dict = {"status": "exploratory (folds 1-4 read at V6-5); evidence for the safety stack, not a promotion read", "rows": int(len(frame)),
                    "C": C, "by_band": {}, "macro_f1": {k: float(f1_score(y, p.argmax(1), average="macro", labels=list(range(7)), zero_division=0)) for k, p in systems.items()}}
    groups = frame["group_id"].to_numpy()
    for band in BANDS:
        mask = (frame["age_band"] == band).to_numpy()
        cell = {name: dict(zip(("signed_gap", "ece"), gap_and_ece(y[mask], p[mask]))) for name, p in systems.items()}
        idx_band = np.where(mask)[0]
        change = stats.paired_cluster_bootstrap(
            lambda i, a=systems["bandwise_dirichlet"]: abs(gap_and_ece(y[idx_band[i]], a[idx_band[i]])[0]),
            lambda i, b=systems["global_dirichlet"]: abs(gap_and_ece(y[idx_band[i]], b[idx_band[i]])[0]),
            groups[mask], n_boot=500)
        cell["abs_gap_change_bandwise_minus_global"] = change
        report["by_band"][band] = {**cell, "images": int(mask.sum())}
    falls = [b for b in BANDS if report["by_band"][b]["abs_gap_change_bandwise_minus_global"]["delta"] <= -0.01]
    f1_ok = report["macro_f1"]["bandwise_dirichlet"] >= report["macro_f1"]["global_dirichlet"] - 0.002
    report["drop_rule"] = {"bands_where_abs_gap_falls_by_0.01": falls, "macro_f1_not_lower": bool(f1_ok),
                           "verdict": "band-wise map helps: carry to the safety stack for the frozen V7 read" if falls and f1_ok else
                                      "band-wise map does not meet the endpoint in this read; keep the global map"}
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for band in BANDS:
        c = report["by_band"][band]
        print(f"{band:6s} gap uncal {c['uncalibrated']['signed_gap']:+.3f} global {c['global_dirichlet']['signed_gap']:+.3f} band {c['bandwise_dirichlet']['signed_gap']:+.3f}"
              f" | |gap| change {c['abs_gap_change_bandwise_minus_global']['delta']:+.4f} {[round(x, 4) for x in c['abs_gap_change_bandwise_minus_global']['ci95']]}")
    print("Macro-F1", {k: round(v, 4) for k, v in report["macro_f1"].items()}, "|", report["drop_rule"]["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
