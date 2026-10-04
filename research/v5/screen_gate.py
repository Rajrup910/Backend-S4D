"""V5 screen gate (runsheet section 6; audit AU18-AU20, AU25) -- fold 0, selection only.

    python -m research.v5.screen_gate --null-only
    python -m research.v5.screen_gate --arm look --comparator control
    python -m research.v5.screen_gate --arm control --arm-trunk dinov3 --comparator control

Inputs are last-epoch prediction CSVs in `results/v5/preds/` (train_v5, or `infer_last` for the
banked train_v4 control runs). A file is found by glob on `<arm>_f0_s<seed>[_<trunk>]*.csv`; the
control's 224 px seeds are the `R0_kfold_f0_s<seed>*_last.csv` files `infer_last` writes.

The gate (fixed before any screen is read):
  * statistic = mean over seeds of (arm - comparator) on the same seed;
  * null = the k-seed-mean difference under "no effect": SD = pooled control pair SD / sqrt(k),
    the pair SD taken over all pairs of the control seeds 42-47 (normal approximation);
  * PASS on an endpoint if the statistic exceeds the SCREEN_NULL_PERCENTILE (80th) quantile of
    that null; the arm passes if it passes on **all-age pAUC@0.20 or pAUC_histo**, AND Macro-F1
    is retained: the mean Macro-F1 difference is >= min(-0.010, -z95 * pair SD / sqrt(k)), i.e.
    an arm fails retention only when its loss exceeds both the -0.010 floor and the seed noise
    (AU32). The mechanism falsifier is read separately (morning read).
  * with two endpoints OR-ed, a null arm passes 20-36% of the time (20% if the endpoints are
    perfectly correlated, 36% if independent); the noise-floor file reports the empirical rate.
  * descriptive only: under-40 pAUC, per-band raw ECE and signed gap (confidence - accuracy).

A screen is a compute-allocation filter, not a test (AU12). No test / reserved / external read.
"""

from __future__ import annotations

import argparse
import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ml.evaluation.metrics import compute_metrics
from research import testguard
from research.v4.recipe import REPO_ROOT
from research.v5 import arms as registry
from research.v5.d0_brainstorm_diagnostics import pauc

PRED_DIR = REPO_ROOT / "results" / "v5" / "preds"
OUT_DIR = REPO_ROOT / "results" / "v5" / "screens"
BANDS = ("<40", "40-59", "60+")
N_BINS = 15


def find_pred(arm: str, seed: int, trunk: str = "in1k", tag: str = "") -> Path:
    """One last-epoch prediction file for (arm, seed, trunk); refuses zero or several matches."""
    if arm == "control" and trunk == "in1k":
        # Banked train_v4 R0 control, re-scored at its last epoch by infer_last.
        patterns = [f"R0_kfold_f0_s{seed}_last.csv", f"R0_kfold_f0_s{seed}_*_last.csv",
                    f"control_f0_s{seed}_{tag}*.csv" if tag else f"control_f0_s{seed}.csv"]
    else:
        stem = f"{arm}_f0_s{seed}" + ("" if trunk == "in1k" else f"_{trunk}")
        patterns = [f"{stem}_{tag}.csv" if tag else f"{stem}.csv", f"{stem}_*.csv"]
    for pattern in patterns:
        hits = sorted(p for p in glob.glob(str(PRED_DIR / pattern)) if not p.endswith("_best.csv"))
        if len(hits) == 1:
            return Path(hits[0])
        if len(hits) > 1:
            raise SystemExit(f"{pattern}: {len(hits)} files match; pass --tag to disambiguate")
    raise SystemExit(f"no prediction file for {arm} seed {seed} trunk {trunk} in {PRED_DIR}")


def histo_mask(frame: pd.DataFrame) -> np.ndarray:
    """Rows of the melanoma-vs-suspicious-lesion differential (D0 section 0.2): every escalating
    row, plus benign rows whose diagnosis is **histopathology-confirmed** in any archive
    (D5, `research/v5/confirmation.py`; AU27 -- the earlier "all BCN/MSKCC benign" rule was
    wrong: BCN is 65% histopathology among benign)."""
    from research.v5.confirmation import histo_confirmed

    escalating = frame["y_esc"].astype(bool).to_numpy()
    return escalating | histo_confirmed(frame["image_id"])


def ece_and_gap(frame: pd.DataFrame) -> tuple[float, float]:
    probs = frame[[c for c in frame.columns if c.startswith("p_")]].to_numpy()
    conf = probs.max(1)
    correct = (probs.argmax(1) == frame["y_true"].to_numpy()).astype(float)
    bins = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    ece = sum(abs(conf[bins == b].mean() - correct[bins == b].mean()) * (bins == b).mean()
              for b in range(N_BINS) if (bins == b).any())
    return float(ece), float(conf.mean() - correct.mean())


def score_run(path: Path, histo: np.ndarray | None = None) -> dict[str, float]:
    frame = pd.read_csv(path, low_memory=False)
    y = frame["y_esc"].astype(bool).to_numpy()
    s = frame["declared_score"].to_numpy(dtype=float)
    if histo is None:
        histo = histo_mask(frame)
    u40 = (frame["age_band"] == "<40").to_numpy()
    metrics = compute_metrics(frame["y_true"].to_numpy(), frame["pred_index"].to_numpy(),
                              frame[[c for c in frame.columns if c.startswith("p_")]].to_numpy())
    out = {"pauc_all": pauc(y, s), "pauc_histo": pauc(y[histo], s[histo]),
           "pauc_u40": pauc(y[u40], s[u40]) if len(np.unique(y[u40])) == 2 else float("nan"),
           "macro_f1": float(metrics["macro_f1"]), "n_rows": int(len(frame))}
    out["ece"], out["signed_gap"] = ece_and_gap(frame)
    band_paucs = []
    for band in BANDS:
        sub = frame[frame["age_band"] == band]
        if len(sub):
            out[f"ece_{band}"], out[f"signed_gap_{band}"] = ece_and_gap(sub)
            yb = sub["y_esc"].astype(bool).to_numpy()
            if len(np.unique(yb)) == 2:
                out[f"pauc_{band}"] = pauc(yb, sub["declared_score"].to_numpy(dtype=float))
                band_paucs.append(out[f"pauc_{band}"])
    # Band-stratified pAUC (AU33): immune to a band-wide score shift, i.e. to a learned age prior.
    out["pauc_band_mean"] = float(np.mean(band_paucs)) if band_paucs else float("nan")
    return out


def null_model(tag: str = "") -> dict[str, object]:
    """Pooled pair SD of every endpoint over the control seeds 42-47 (224 px, in1k)."""
    seeds = registry.SEEDS_NULL
    scores = {s: score_run(find_pred("control", s, "in1k", tag)) for s in seeds}
    pair_sd = {}
    for endpoint in ("pauc_all", "pauc_histo", "macro_f1", "pauc_u40"):
        diffs = [scores[a][endpoint] - scores[b][endpoint] for a, b in combinations(seeds, 2)]
        pair_sd[endpoint] = float(np.sqrt(np.nanmean(np.square(diffs))))
    return {"seeds": list(seeds), "n_pairs": len(list(combinations(seeds, 2))),
            "pair_sd": pair_sd, "per_seed": {str(k): v for k, v in scores.items()}}


def gate(arm: str, comparator: str, null: dict, arm_trunk: str, comparator_trunk: str,
         seeds: tuple[int, ...], tag: str = "") -> dict[str, object]:
    k = len(seeds)
    z = float(norm.ppf(registry.SCREEN_NULL_PERCENTILE / 100))
    rows = []
    for seed in seeds:
        a = score_run(find_pred(arm, seed, arm_trunk, tag))
        c = score_run(find_pred(comparator, seed, comparator_trunk, tag))
        rows.append({"seed": seed, **{f"delta_{e}": a[e] - c[e] for e in
                                      ("pauc_all", "pauc_histo", "pauc_u40", "pauc_band_mean",
                                       "macro_f1", "ece", "signed_gap")},
                     "arm": a, "comparator": c})
    mean = {e: float(np.nanmean([r[f"delta_{e}"] for r in rows]))
            for e in ("pauc_all", "pauc_histo", "pauc_u40", "pauc_band_mean", "macro_f1", "ece",
                      "signed_gap")}
    thresholds = {e: z * null["pair_sd"][e] / np.sqrt(k) for e in registry.SCREEN_ENDPOINTS}
    endpoint_pass = {e: mean[e] > thresholds[e] for e in registry.SCREEN_ENDPOINTS}
    retention_bar = min(registry.SCREEN_MACRO_F1_FLOOR,
                        -float(norm.ppf(0.95)) * null["pair_sd"]["macro_f1"] / np.sqrt(k))
    retention = mean["macro_f1"] >= retention_bar
    passed = any(endpoint_pass.values()) and retention
    rescue = (not passed and arm in registry.RESCUE_384
              and max(mean["pauc_all"], mean["pauc_histo"]) > 0)
    return {"arm": arm, "arm_trunk": arm_trunk, "comparator": comparator,
            "comparator_trunk": comparator_trunk, "seeds": list(seeds),
            "null_percentile": registry.SCREEN_NULL_PERCENTILE, "z": z,
            "thresholds": thresholds, "retention_bar": retention_bar,
            "mean_delta": mean, "endpoint_pass": endpoint_pass,
            "macro_f1_retention_ok": retention, "gate_pass_before_falsifier": passed,
            "rescue_384_eligible_if_s01_picked_384": rescue,
            "descriptive_only": ["pauc_u40", "ece", "signed_gap", "per-band ECE"],
            "youngdata_within_band_check": (mean["pauc_band_mean"] > 0
                                            and mean["pauc_u40"] >= 0) if arm == "youngdata" else None,
            "per_seed": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm")
    parser.add_argument("--comparator", default=None, help="default: the registry comparator")
    parser.add_argument("--arm-trunk", default="in1k")
    parser.add_argument("--comparator-trunk", default=None, help="default: --arm-trunk")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(registry.SEEDS_SCREEN))
    parser.add_argument("--tag", default="", help="run tag, if several files match")
    parser.add_argument("--null-only", action="store_true")
    args = parser.parse_args(argv)
    testguard.block_test_reads("V5 screen gate: fold-0 development predictions only")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    null = null_model()
    (OUT_DIR / "noise_floor.json").write_text(json.dumps(null, indent=2), encoding="utf-8")
    print("noise floor (pair SD over control seeds "
          + ",".join(map(str, null["seeds"])) + "): "
          + "  ".join(f"{k}={v:.4f}" for k, v in null["pair_sd"].items()))
    if args.null_only:
        return 0
    if not args.arm:
        raise SystemExit("pass --arm (or --null-only)")
    comparator = args.comparator or registry.get_arm(args.arm).comparator or "control"
    # The trunk screen compares control-on-trunk with control-on-in1k.
    comparator_trunk = args.comparator_trunk or ("in1k" if args.arm == comparator else args.arm_trunk)
    verdict = gate(args.arm, comparator, null, args.arm_trunk, comparator_trunk,
                   tuple(args.seeds), args.tag)
    name = f"{args.arm}" + ("" if args.arm_trunk == "in1k" else f"_{args.arm_trunk}")
    out = OUT_DIR / f"gate_{name}_vs_{comparator}.json"
    out.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    m, t = verdict["mean_delta"], verdict["thresholds"]
    print(f"{name} vs {comparator}: dpAUC_all {m['pauc_all']:+.4f} (bar {t['pauc_all']:.4f})  "
          f"dpAUC_histo {m['pauc_histo']:+.4f} (bar {t['pauc_histo']:.4f})  "
          f"dMacroF1 {m['macro_f1']:+.4f}  -> "
          f"{'PASS (falsifier still to read)' if verdict['gate_pass_before_falsifier'] else 'FAIL'}"
          + ("  [384 px rescue eligible]" if verdict["rescue_384_eligible_if_s01_picked_384"] else ""))
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
