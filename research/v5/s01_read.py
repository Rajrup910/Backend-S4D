"""S01 read (runsheet E2 / section 8 Q0): does 384 px beat 224 px on the pooled fold-0 partition?

    python -m research.v5.s01_read [--n-boot 2000]

Inputs: the last-epoch prediction files `infer_last` writes, `results/v5/preds/`
`R1_kfold_f0_s<seed>*_last.csv` (384 px) and `R0_kfold_f0_s<seed>*_last.csv` (224 px), seeds 42/43/44,
paired by seed on the same 3,059 held-out rows. Output: `results/v5/s01_decision.json`, which
sets the resolution of the composite and of confirmation (runsheet section 8.3).

The rule is transcribed, not chosen here (master plan V5-A1 "Two-Hurdle Acceptance Rule",
amended by A01 A5 and audit AU29). 384 px is adopted only if ALL of:
  1. statistical: the 95% **hierarchical** bootstrap CI of the seed-mean paired delta Macro-F1
     (resample the 3 seed pairs, then lesions within each drawn pair) has lower bound > 0;
     the t-interval over the 3 paired deltas is reported, descriptive only (AU29);
  2. magnitude: mean paired delta Macro-F1 >= +0.015 (MCID);
  3. consistency: >= 2 of 3 seeds strictly positive and no seed below -0.010;
  4. per-class safety: mean delta MEL F1 >= -0.020 and mean delta BCC recall >= -0.020.
Otherwise the composite and confirmation run at 224 px (runsheet section 8.3).

Descriptive only: all-age and under-40 escalation pAUC@0.20 deltas.
No test / reserved / external read.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ml.evaluation.metrics import compute_metrics
from ml.paths import load_class_mapping
from research import testguard
from research.v4.recipe import REPO_ROOT
from research.v5.d0_brainstorm_diagnostics import pauc

PRED_DIR = REPO_ROOT / "results" / "v5" / "preds"
OUT = REPO_ROOT / "results" / "v5" / "s01_decision.json"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"
SEEDS = (42, 43, 44)
MCID = 0.015
SEED_FLOOR = -0.010
CLASS_FLOOR = -0.020
ALPHA = 0.05


def find(rung: str, seed: int) -> Path:
    hits = sorted(set(glob.glob(str(PRED_DIR / f"{rung}_kfold_f0_s{seed}_last.csv")))
                  | set(glob.glob(str(PRED_DIR / f"{rung}_kfold_f0_s{seed}_*_last.csv"))))
    if len(hits) != 1:
        raise SystemExit(f"{rung} seed {seed}: expected one *_last.csv in {PRED_DIR}, "
                         f"found {len(hits)} -- run research.v5.infer_last first")
    return Path(hits[0])


def weighted_scores(y: np.ndarray, p: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    """[macro F1, F1 per class..., recall per class...] from a weighted confusion matrix."""
    cm = np.bincount(y * k + p, weights=w, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    support, predicted = cm.sum(1), cm.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(support > 0, tp / support, 0.0)
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return np.concatenate([[f1.mean()], f1, recall])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260930, help="bootstrap RNG seed")
    args = parser.parse_args(argv)
    testguard.block_test_reads("V5 S01 read: fold-0 development predictions only")

    mapping = load_class_mapping()
    k = mapping.num_classes
    mel, bcc = mapping.codes.index("mel"), mapping.codes.index("bcc")
    prob_cols = [f"p_{c}" for c in mapping.codes]

    pairs, inputs = [], {}
    for seed in SEEDS:
        frames = {}
        for rung, size in (("R1", 384), ("R0", 224)):
            path = find(rung, seed)
            frame = pd.read_csv(path, low_memory=False)
            if int(frame["image_size"].iloc[0]) != size or (frame["checkpoint"] != "last").any():
                raise SystemExit(f"{path.name}: expected {size} px last-epoch predictions")
            frames[size] = frame
            inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if not frames[384]["image_id"].equals(frames[224]["image_id"]):
            raise SystemExit(f"seed {seed}: 384 and 224 rows are not the same images in order")
        pairs.append((seed, frames[384], frames[224]))

    y = pairs[0][1]["y_true"].to_numpy()
    lesion_codes, lesions = pd.factorize(pairs[0][1]["effective_lesion_id"])
    n_lesions = len(lesions)
    for _, a, b in pairs:
        if not (np.array_equal(a["y_true"], y) and np.array_equal(b["y_true"], y)):
            raise SystemExit("labels differ between prediction files")

    # Point estimates from the project's metric engine; the fast weighted path must agree.
    per_seed, preds = [], []
    ones = np.ones(len(y))
    for seed, a, b in pairs:
        row = {"seed": seed}
        pa, pb = a["pred_index"].to_numpy(), b["pred_index"].to_numpy()
        preds.append((pa, pb))
        for label, frame, p in (("384", a, pa), ("224", b, pb)):
            m = compute_metrics(y, p, frame[prob_cols].to_numpy())
            fast = weighted_scores(y, p, ones, k)
            if abs(fast[0] - m["macro_f1"]) > 1e-9:
                raise AssertionError("weighted Macro-F1 disagrees with compute_metrics")
            row[f"macro_f1_{label}"] = float(m["macro_f1"])
            row[f"balanced_accuracy_{label}"] = float(m["balanced_accuracy"])
            row[f"mel_f1_{label}"] = float(fast[1 + mel])
            row[f"bcc_recall_{label}"] = float(fast[1 + k + bcc])
            esc = frame["y_esc"].astype(bool).to_numpy()
            s = frame["declared_score"].to_numpy(dtype=float)
            u40 = (frame["age_band"] == "<40").to_numpy()
            row[f"pauc_all_{label}"] = pauc(esc, s)
            row[f"pauc_u40_{label}"] = pauc(esc[u40], s[u40])
        for key in ("macro_f1", "mel_f1", "bcc_recall", "pauc_all", "pauc_u40",
                    "balanced_accuracy"):
            row[f"delta_{key}"] = row[f"{key}_384"] - row[f"{key}_224"]
        per_seed.append(row)

    deltas = np.array([r["delta_macro_f1"] for r in per_seed])
    mean_delta = float(deltas.mean())
    sd = float(deltas.std(ddof=1))
    t_half = float(stats.t.ppf(1 - ALPHA / 2, len(deltas) - 1) * sd / np.sqrt(len(deltas)))

    # Hierarchical bootstrap: draw seed pairs with replacement, then lesions with replacement
    # independently within each drawn pair; both arms of a pair share that pair's lesion draw.
    rng = np.random.default_rng(args.seed)
    boot = np.empty(args.n_boot)
    for i in range(args.n_boot):
        drawn = rng.integers(0, len(pairs), len(pairs))
        vals = []
        for j in drawn:
            counts = np.bincount(rng.integers(0, n_lesions, n_lesions), minlength=n_lesions)
            w = counts[lesion_codes].astype(float)
            pa, pb = preds[j]
            vals.append(weighted_scores(y, pa, w, k)[0] - weighted_scores(y, pb, w, k)[0])
        boot[i] = np.mean(vals)
    lo, hi = np.quantile(boot, [ALPHA / 2, 1 - ALPHA / 2])

    mel_delta = float(np.mean([r["delta_mel_f1"] for r in per_seed]))
    bcc_delta = float(np.mean([r["delta_bcc_recall"] for r in per_seed]))
    hurdles = {
        "1_hierarchical_ci_lower_gt_0": bool(lo > 0),
        "2_mean_delta_ge_mcid": bool(mean_delta >= MCID),
        "3_consistency": bool((deltas > 0).sum() >= 2 and deltas.min() >= SEED_FLOOR),
        "4_per_class_safety": bool(mel_delta >= CLASS_FLOOR and bcc_delta >= CLASS_FLOOR),
    }
    passed = all(hurdles.values())
    decision = {
        "question": "S01: 384 px (R1) vs 224 px (R0), pooled partition, fold 0, last epoch",
        "rule_source": "master plan V5-A1 two-hurdle rule; A01 A5; audit AU29; runsheet 6, 8.3",
        "seeds": list(SEEDS), "n_rows": int(len(y)), "n_lesions": int(n_lesions),
        "per_seed": per_seed,
        "mean_delta_macro_f1": mean_delta,
        "hierarchical_bootstrap": {"n_boot": args.n_boot, "rng_seed": args.seed,
                                   "ci95": [float(lo), float(hi)],
                                   "p_delta_le_0": float((boot <= 0).mean())},
        "t_interval_descriptive": [mean_delta - t_half, mean_delta + t_half],
        "seed_sd_of_delta": sd,
        "mean_delta_mel_f1": mel_delta, "mean_delta_bcc_recall": bcc_delta,
        "mean_delta_pauc_all_descriptive": float(np.mean([r["delta_pauc_all"] for r in per_seed])),
        "mean_delta_pauc_u40_descriptive": float(np.mean([r["delta_pauc_u40"] for r in per_seed])),
        "thresholds": {"mcid": MCID, "seed_floor": SEED_FLOOR, "class_floor": CLASS_FLOOR},
        "hurdles": hurdles, "pass": passed,
        "resolution": 384 if passed else 224,
        "consequence": ("composite and confirmation at 384 px" if passed else
                        "runsheet 8.3 fallback: composite and confirmation at 224 px; the 384 px "
                        "rescue (AU20) cannot fire"),
        "inputs_sha256": inputs,
        "test_read": False,
    }
    OUT.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    ledger_row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(), "session": "v5_s01_read",
        "method": "s01_384_vs_224_f0_last", "split": "f0",
        "macro_f1": mean_delta,
        "notes": (f"mean paired dMacroF1 (384-224) over seeds 42/43/44; hierarchical CI "
                  f"[{lo:.4f}, {hi:.4f}]; pass={passed}; resolution={decision['resolution']}")}])
    if LEDGER_PATH.is_file():
        old = pd.read_csv(LEDGER_PATH, low_memory=False)
        old = old[~((old["session"] == "v5_s01_read") & (old["method"] == "s01_384_vs_224_f0_last"))]
        ledger_row = pd.concat([old, ledger_row], ignore_index=True)
    ledger_row.to_csv(LEDGER_PATH, index=False)

    for r in per_seed:
        print(f"s{r['seed']}: MacroF1 384 {r['macro_f1_384']:.4f}  224 {r['macro_f1_224']:.4f}  "
              f"d {r['delta_macro_f1']:+.4f}  dMEL-F1 {r['delta_mel_f1']:+.4f}  "
              f"dBCC-rec {r['delta_bcc_recall']:+.4f}")
    print(f"mean d {mean_delta:+.4f}; hierarchical 95% CI [{lo:+.4f}, {hi:+.4f}]; "
          f"t-interval (descriptive) [{mean_delta - t_half:+.4f}, {mean_delta + t_half:+.4f}]")
    print("hurdles: " + ", ".join(f"{k}={v}" for k, v in hurdles.items()))
    print(f"S01 {'PASS -> 384 px' if passed else 'FAIL -> 224 px'}; wrote "
          f"{OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
