"""S40 / Phase A1 -- the N2 kill test: was S35's 0.997 a representation finding or a leak?

S35 (`research/v2/np_head.py`) reported a cross-fitted logistic probe on cached
ConvNeXt-Tiny features reaching **0.9930** under-40 partial AUC against the deployed
posterior's 0.8096, and concluded `RAISES_B` -- the representation carries escalation
signal the 7-way head does not expose. S36 selected its three trained arms on that basis.

The probe's cross-fitting is correct. The *features* are not. `research/selective/features.py`
line 41 pins `CHECKPOINT_TEMPLATE = "ml/checkpoints/{arch}_best.HAM-only.pt"` -- the model
trained on the entire train split -- and `extract_one` then runs it over
`LesionDataset(split="train")`. So `convnext_tiny_train.npz` holds **in-sample activations**
of the 6,981 images that model fitted. The `s_uniform` it was compared against comes from
cross-fitted OOF fold models (`research/predictions_oof_tta/`). One side memorized, the
other did not; cross-fitting a probe on top cannot undo a leak in the extractor.

The tell is already in `results/v2/np_head_report.json`: np_head full AUC is 0.9975 (<40),
0.9986 (40-59), 0.9991 (60+) -- near-perfect in **every** band, including the bands with no
claimed deficit. A representation finding is selective; memorization is uniform.

Three stages, cheapest first:

    stage 0   HAM val features -- 1,532 rows the HAM-only checkpoint never trained on.
              Both sides out-of-sample, so the comparison is finally like-for-like.
              CPU, seconds. Also re-runs the probe on `_train.npz` as a harness proof:
              if this code cannot reproduce S35's ~0.997 there, a low number on val would
              mean nothing.
    stage 1   The honest OOF feature panel from `extract_oof_features.py` -- all 6,981 rows,
              each scored by the fold model that held it out. The registered falsifier.
    stage 2   PAD-UFES-20 -- a different dataset entirely, out-of-sample by construction.

FALSIFIER (declared in the V3 plan before any stage was run): N2 survives only if, on
Stage-1 OOF features, the under-40 partial-AUC gain over `s_uniform` exceeds the project
MCID of 0.05 with a lesion-grouped 95% CI excluding zero. Anything less is written up as
"the S35 result was an artifact of in-sample feature extraction", naming the S36 dependency.

    $py -m research.v3.oos_probe --stage 0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.external import frozen_params as fp
from research.stats.calibration_slices import grouped_bootstrap_scalar
from research.v2 import estimators as est
from research.v2 import frontier as fr
from research.v2.np_head import cross_fitted_np_scores

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = REPO_ROOT / "results" / "v2" / "panels"
V2_FEATURES = REPO_ROOT / "research" / "selective" / "features"
V3_FEATURES = REPO_ROOT / "research" / "v3" / "features"
OUT_DIR = REPO_ROOT / "results" / "v3"
OUT_JSON = OUT_DIR / "oos_probe_report.json"
OUT_CSV = OUT_DIR / "oos_probe_bands.csv"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"

SEED = 42
FPR_MAX = 0.20
N_BOOT = 2000
MCID = 0.05
BANDS = ("<40", "40-59", "60+", "all")

# S35's published numbers, for the harness proof in stage 0.
S35_TRAIN_UNDER40_NP_PAUC = 0.9929794710048696
S35_TRAIN_ALL_NP_FULL_AUC = 0.9990320550639135
HARNESS_TOL = 0.02


def _load_npz(path: Path) -> dict:
    """Read a cached feature file directly, bypassing `features.load_features`.

    `load_features` routes through `testguard.check_split`, which is correct for its own
    callers but takes a split *name*; here the split identity is carried by the file path
    and stage, and stage 1 reads a file that belongs to no train/val/test split at all.
    No stage of this module ever opens `convnext_tiny_test.npz`.
    """
    if "test" in path.stem.split("_"):
        raise ValueError(f"refusing to read a test-split feature cache: {path.name}")
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}. For stage 1 run `$py -m research.v3.extract_oof_features` first."
        )
    data = np.load(path, allow_pickle=False)
    return {"features": data["features"], "image_ids": np.asarray([str(i) for i in data["image_ids"]])}


def _align(features: np.ndarray, feat_ids: np.ndarray, panel_ids: np.ndarray) -> np.ndarray:
    order = pd.Index(feat_ids).get_indexer(panel_ids)
    if (order < 0).any():
        raise ValueError(f"{int((order < 0).sum())} panel image_ids absent from the feature cache")
    return features[order]


def paired_delta_ci(
    y_esc: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, lesion_ids: np.ndarray,
    n_boot: int = N_BOOT, seed: int = SEED,
) -> dict:
    """Lesion-grouped CI for pAUC(a) - pAUC(b), both scored on the *same* resample.

    Paired rather than a comparison of two marginal intervals: the two scores are computed
    on identical rows, so their sampling errors are strongly correlated and independent
    intervals would be far too wide to decide the falsifier.
    """
    def point(s: np.ndarray) -> float:
        return fr.partial_auc(y_esc, s, FPR_MAX)["partial_auc_mcclish"]

    delta = point(score_a) - point(score_b)

    def statistic(idx: np.ndarray) -> float:
        sub_y = y_esc[idx]
        if sub_y.sum() == 0 or (~sub_y).sum() == 0:
            return float("nan")
        a = fr.partial_auc(sub_y, score_a[idx], FPR_MAX)["partial_auc_mcclish"]
        b = fr.partial_auc(sub_y, score_b[idx], FPR_MAX)["partial_auc_mcclish"]
        return a - b

    lo, hi = grouped_bootstrap_scalar(statistic, lesion_ids, n_boot=n_boot, seed=seed)
    return {"delta_pauc": float(delta), "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0.0), "exceeds_mcid": bool(delta > MCID)}


_paired_delta_ci = paired_delta_ci  # internal alias; `paired_delta_ci` is the shared instrument


def probe_panel(panel: pd.DataFrame, features: np.ndarray, feat_ids: np.ndarray,
                label: str, n_boot: int = N_BOOT) -> dict:
    """Run the S35 probe against `s_uniform` on one cohort, band by band.

    `cross_fitted_np_scores` is imported from `research.v2.np_head` unchanged -- the probe
    is not the thing under test, the features are.
    """
    panel_ids = panel["image_id"].astype(str).to_numpy()
    aligned = _align(features, feat_ids, panel_ids)

    lesion_ids = panel["effective_lesion_id"].astype(str).to_numpy()
    bands = panel["age_band"].astype(str).to_numpy()
    esc = fp.escalating_indices()
    y_esc = np.isin(panel["y_true"].to_numpy(), esc)

    prob_cols = [c for c in panel.columns if c.startswith("p_") and not c.startswith("p_raw_")]
    probs = panel[prob_cols].to_numpy(dtype=float)
    s_uniform = est.escalation_mass(probs, esc)

    np_scores = cross_fitted_np_scores(aligned, y_esc, lesion_ids)

    rows, deltas = [], {}
    for band in BANDS:
        mask = np.ones(len(bands), dtype=bool) if band == "all" else bands == band
        if y_esc[mask].sum() == 0 or (~y_esc[mask]).sum() == 0:
            continue
        s_stat = fr.partial_auc_ci(y_esc[mask], s_uniform[mask], lesion_ids[mask],
                                   fpr_max=FPR_MAX, seed=SEED, n_boot=n_boot)
        n_stat = fr.partial_auc_ci(y_esc[mask], np_scores[mask], lesion_ids[mask],
                                   fpr_max=FPR_MAX, seed=SEED, n_boot=n_boot)
        delta = _paired_delta_ci(y_esc[mask], np_scores[mask], s_uniform[mask],
                                 lesion_ids[mask], n_boot=n_boot)
        deltas[band] = delta
        rows.append({
            "source": label, "band": band,
            "n": int(mask.sum()), "n_escalating": int(y_esc[mask].sum()),
            "s_pauc": s_stat["partial_auc_mcclish"], "s_full_auc": s_stat["full_auc"],
            "np_pauc": n_stat["partial_auc_mcclish"], "np_full_auc": n_stat["full_auc"],
            "np_pauc_ci_lo": n_stat["ci_lo"], "np_pauc_ci_hi": n_stat["ci_hi"],
            "delta_pauc": delta["delta_pauc"],
            "delta_ci_lo": delta["ci_lo"], "delta_ci_hi": delta["ci_hi"],
            "underpowered": bool(int(y_esc[mask].sum()) < 30),
        })

    return {"label": label, "n_rows": int(len(panel)), "bands": rows, "deltas": deltas}


def _verdict(result: dict) -> dict:
    """Apply the pre-registered falsifier to the under-40 band."""
    under40 = next((r for r in result["bands"] if r["band"] == "<40"), None)
    if under40 is None:
        return {"result": "NOT_EVALUABLE", "reason": "no under-40 band in this cohort"}
    d = result["deltas"]["<40"]
    survives = d["excludes_zero"] and d["exceeds_mcid"]
    return {
        "criterion": (
            "under-40 pAUC gain over s_uniform exceeds MCID 0.05 with a lesion-grouped "
            "95% CI excluding zero"
        ),
        "delta_pauc": d["delta_pauc"], "ci": [d["ci_lo"], d["ci_hi"]],
        "excludes_zero": d["excludes_zero"], "exceeds_mcid": d["exceeds_mcid"],
        "n_escalating": under40["n_escalating"],
        "underpowered": under40["underpowered"],
        "result": "N2_SURVIVES" if survives else "N2_FALSIFIED",
    }


def _panel(name: str) -> pd.DataFrame:
    return pd.read_csv(PANEL_DIR / f"{name}.csv")


def stage0(n_boot: int) -> dict:
    """HAM val -- both sides out-of-sample -- plus the harness proof on `_train.npz`."""
    print("stage 0: HAM val features (OOS for the HAM-only checkpoint)\n")

    train_cache = _load_npz(V2_FEATURES / "convnext_tiny_train.npz")
    train_res = probe_panel(_panel("ham_oof"), train_cache["features"],
                            train_cache["image_ids"], "ham_train_INSAMPLE", n_boot=n_boot)
    t40 = next(r for r in train_res["bands"] if r["band"] == "<40")
    tall = next(r for r in train_res["bands"] if r["band"] == "all")
    harness_ok = (
        abs(t40["np_pauc"] - S35_TRAIN_UNDER40_NP_PAUC) < HARNESS_TOL
        and abs(tall["np_full_auc"] - S35_TRAIN_ALL_NP_FULL_AUC) < HARNESS_TOL
    )
    print(f"  harness proof on _train.npz (in-sample): under-40 pAUC {t40['np_pauc']:.4f} "
          f"vs S35's {S35_TRAIN_UNDER40_NP_PAUC:.4f}  ->  "
          f"{'REPRODUCED' if harness_ok else 'DIVERGED'}")

    val_cache = _load_npz(V2_FEATURES / "convnext_tiny_val.npz")
    val_res = probe_panel(_panel("ham_val"), val_cache["features"],
                          val_cache["image_ids"], "ham_val_OOS", n_boot=n_boot)

    for row in val_res["bands"]:
        flag = "  [underpowered]" if row["underpowered"] else ""
        print(f"  val {row['band']:>5}: s={row['s_pauc']:.4f}  np={row['np_pauc']:.4f}  "
              f"delta={row['delta_pauc']:+.4f} [{row['delta_ci_lo']:+.4f}, "
              f"{row['delta_ci_hi']:+.4f}]{flag}")

    return {
        "stage": 0,
        "harness_proof": {
            "train_under40_np_pauc": t40["np_pauc"],
            "s35_published": S35_TRAIN_UNDER40_NP_PAUC,
            "train_all_np_full_auc": tall["np_full_auc"],
            "reproduced": bool(harness_ok),
            "note": (
                "In-sample by construction -- features from the full-train checkpoint over "
                "the rows it trained on. Reproducing S35 here proves this harness is faithful; "
                "it is not evidence for N2."
            ),
        },
        "in_sample": train_res, "out_of_sample": val_res,
        "verdict": _verdict(val_res),
    }


def stage1(n_boot: int) -> dict:
    """The registered falsifier: all 6,981 OOF rows, each scored by the fold that held it out."""
    print("stage 1: HAM OOF features (cross-fitted fold models)\n")
    cache = _load_npz(V3_FEATURES / "convnext_tiny_oof.npz")
    res = probe_panel(_panel("ham_oof"), cache["features"], cache["image_ids"],
                      "ham_oof_CROSSFIT", n_boot=n_boot)
    for row in res["bands"]:
        print(f"  oof {row['band']:>5}: s={row['s_pauc']:.4f}  np={row['np_pauc']:.4f}  "
              f"delta={row['delta_pauc']:+.4f} [{row['delta_ci_lo']:+.4f}, "
              f"{row['delta_ci_hi']:+.4f}]")
    return {"stage": 1, "out_of_sample": res, "verdict": _verdict(res)}


def stage2(n_boot: int) -> dict:
    """PAD-UFES-20 -- a different dataset, out-of-sample by construction."""
    print("stage 2: PAD-UFES-20 features (OOS by construction)\n")
    cache = _load_npz(V2_FEATURES / "convnext_tiny_pad.npz")
    res = probe_panel(_panel("pad"), cache["features"], cache["image_ids"],
                      "pad_OOS", n_boot=n_boot)
    for row in res["bands"]:
        print(f"  pad {row['band']:>5}: s={row['s_pauc']:.4f}  np={row['np_pauc']:.4f}  "
              f"delta={row['delta_pauc']:+.4f} [{row['delta_ci_lo']:+.4f}, "
              f"{row['delta_ci_hi']:+.4f}]")
    return {"stage": 2, "out_of_sample": res, "verdict": _verdict(res)}


def _write(stage_result: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.is_file() else {
        "session": "S40", "phase": "A1",
        "statement": (
            "N2 kill test. S35's probe re-run on genuinely out-of-sample features. The probe "
            "(research.v2.np_head.cross_fitted_np_scores) is imported unchanged; only the "
            "feature source varies. Exploratory -- member of no confirmatory family."
        ),
        "mcid": MCID, "fpr_max": FPR_MAX, "n_boot": N_BOOT, "seed": SEED,
        "stages": {},
    }
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["stages"][f"stage{stage_result['stage']}"] = stage_result
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = []
    for stage in report["stages"].values():
        for key in ("in_sample", "out_of_sample"):
            if key in stage:
                rows.extend(stage[key]["bands"])
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_JSON.relative_to(REPO_ROOT)}\nwrote {OUT_CSV.relative_to(REPO_ROOT)}")


def _append_ledger(stage_result: dict) -> None:
    session = "v3_s40_oos_probe"
    v = stage_result["verdict"]
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "session": session,
        "method": f"A1_oos_probe[stage{stage_result['stage']}]", "split": "oof",
        "macro_f1": "", "accuracy": "", "balanced_accuracy": "", "weighted_f1": "",
        "macro_roc_auc": "", "ece": "", "escalation_sens": "", "missed_serious": "",
        "p_value_vs_baseline": "",
        "notes": (
            f"S40_A1 stage{stage_result['stage']}; under40 delta_pauc="
            f"{v.get('delta_pauc', float('nan')):.4f} "
            f"CI=[{v.get('ci', [float('nan')] * 2)[0]:.4f},{v.get('ci', [float('nan')] * 2)[1]:.4f}]; "
            f"verdict={v['result']}"
        ),
    }
    frame = pd.DataFrame([row])
    if LEDGER_PATH.is_file():
        old = pd.read_csv(LEDGER_PATH)
        keep = ~((old["session"] == session)
                 & (old["method"] == row["method"]))  # prune this stage's own prior row
        frame = pd.concat([old[keep], frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S40 A1 -- the N2 kill test")
    parser.add_argument("--stage", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)

    result = {0: stage0, 1: stage1, 2: stage2}[args.stage](args.n_boot)
    _write(result)
    _append_ledger(result)

    v = result["verdict"]
    print(f"\nVERDICT (stage {args.stage}): {v['result']}")
    if v["result"] == "N2_FALSIFIED":
        print("  The S35 result does not survive out-of-sample feature extraction.")
        print("  S36's three trained arms were selected on the in-sample number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
