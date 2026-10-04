"""V5 confirmation read (runsheet §11.3 E14 / E18): locked composite vs in1k control on held-out folds 1-4.

    python -m research.v5.confirm_read --seeds 42                 # Review-2 minimum -> confirm_s42.json
    python -m research.v5.confirm_read --seeds 42 43              # two seeds, Gate D provisional
    python -m research.v5.confirm_read --seeds 42 43 44           # final

Every operational choice was declared in CHANGELOG ("E14 confirmation read — operational choices DECLARED
BEFORE THE READ", 2026-10-04) before this file first read a fold 1-4 prediction:
  * rows: pooled folds 1-4 per seed, aligned by image_id; score = each arm's registry declared_score;
  * uncertainty: the S01 hierarchical bootstrap (seed pairs, then effective_lesion_id clusters), 2,000 resamples;
  * Gate A  d<40 pAUC@0.20 >= +0.050 and CI lower > 0;  Gate B  dMacro-F1 >= -0.010;
    Gate C  d sensitivity at fixed specificity 0.80, bands 40-59 and 60+, >= -0.030;
    Gate D  per-seed d all-age pAUC: mean > 0, >= 2 of 3 > 0, none < -0.010 (3 seeds only);
  * descriptive: pAUC_histo, balanced accuracy, argmax escalation sensitivity, B2 (MEL-vs-NV AUC <40,
    class-standardised <40 pAUC), B4 (per-archive <40 pAUC, archive-fixed-effects pooled delta);
  * secondary family D1 (seed 42): m4 vs control, composite vs m4 on d all-age pAUC, Holm across 2.
Development folds only: the test lock is armed and no reserved / external row is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.evaluation.metrics import compute_metrics
from research import testguard
from research.v4.recipe import REPO_ROOT
from research.v5.d0_brainstorm_diagnostics import pauc
from research.v5.s01_read import weighted_scores
from research.v5.screen_gate import histo_mask

PRED = REPO_ROOT / "results" / "v5" / "preds"
OUT_DIR = REPO_ROOT / "results" / "v5"
LEDGER = REPO_ROOT / "research" / "experiments.csv"
FOLDS = (1, 2, 3, 4)
K = 7
PROB = ["p_akiec", "p_bcc", "p_bkl", "p_df", "p_mel", "p_nv", "p_vasc"]
MEL, NV = 4, 5
ALPHA = 0.05
GATE_A, GATE_B, GATE_C, SEED_FLOOR = 0.050, -0.010, -0.030, -0.010
SPEC = 0.80
ARCHIVES = ("bcn20000", "ham", "mskcc")


def files(arm: str, seed: int) -> list[Path]:
    if arm == "control":
        tag = "" if seed == 42 else "_v5ctl"
        return [PRED / f"R0_kfold_f{k}_s{seed}{tag}_last.csv" for k in FOLDS]
    return [PRED / f"{arm}_f{k}_s{seed}_in22k_v5conf.csv" for k in FOLDS]


def load(arm: str, seed: int, sha: dict) -> pd.DataFrame:
    parts = []
    for p in files(arm, seed):
        if not p.is_file():
            raise SystemExit(f"missing {p.relative_to(REPO_ROOT)}")
        f = pd.read_csv(p, low_memory=False)
        if "checkpoint" in f and (f["checkpoint"] != "last").any():
            raise SystemExit(f"{p.name}: not last-epoch predictions")
        sha[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        parts.append(f)
    frame = pd.concat(parts, ignore_index=True)
    if set(frame["fold"].unique()) != set(FOLDS):
        raise SystemExit(f"{arm} s{seed}: folds {sorted(frame['fold'].unique())}, expected 1-4")
    return frame.sort_values("image_id").reset_index(drop=True)


def wauc(y: np.ndarray, s: np.ndarray, w: np.ndarray, max_fpr: float | None = None) -> float:
    keep = w > 0
    y, s, w = y[keep], s[keep], w[keep]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s, max_fpr=max_fpr, sample_weight=w))


def sens_at_spec(esc: np.ndarray, s: np.ndarray, band: np.ndarray, w: np.ndarray) -> float:
    """Sensitivity in `band` at the threshold giving specificity SPEC on all non-escalating rows."""
    neg = (~esc) & (w > 0)
    order = np.argsort(s[neg])
    cw = np.cumsum(w[neg][order]) / w[neg].sum()
    thr = s[neg][order][min(np.searchsorted(cw, SPEC), len(order) - 1)]
    pos = esc & band & (w > 0)
    return float((w[pos] * (s[pos] > thr)).sum() / w[pos].sum()) if w[pos].sum() else float("nan")


class Ctx:
    """Row-level quantities shared by every arm (labels, bands, masks) for one aligned row set."""

    def __init__(self, f: pd.DataFrame):
        self.y = f["y_true"].to_numpy()
        self.esc = f["y_esc"].astype(bool).to_numpy()
        self.u40 = (f["age_band"] == "<40").to_numpy()
        self.b4059 = (f["age_band"] == "40-59").to_numpy()
        self.b60 = (f["age_band"] == "60+").to_numpy()
        self.histo = histo_mask(f)
        self.archive = f["archive"].to_numpy()
        self.lesion, uniq = pd.factorize(f["effective_lesion_id"])
        self.n_lesions = len(uniq)
        # B2 class-standardisation weights: pooled class share / <40 class share, on <40 rows.
        pooled = np.bincount(self.y, minlength=K) / len(self.y)
        young = np.bincount(self.y[self.u40], minlength=K) / max(self.u40.sum(), 1)
        ratio = np.divide(pooled, young, out=np.zeros(K), where=young > 0)
        self.std_w = ratio[self.y]
        self.melnv_u40 = self.u40 & np.isin(self.y, [MEL, NV])
        self.esc_les = {a: len(np.unique(self.lesion[self.esc & self.u40 & (self.archive == a)]))
                        for a in ARCHIVES}


def metrics(ctx: Ctx, f: pd.DataFrame, w: np.ndarray) -> dict[str, float]:
    s = f["declared_score"].to_numpy(dtype=float)
    p = f["pred_index"].to_numpy()
    sc = weighted_scores(ctx.y, p, w, K)
    esc_pred = np.isin(p, [0, 1, 4])
    out = {
        "pauc_all": wauc(ctx.esc, s, w, 0.2),
        "pauc_u40": wauc(ctx.esc[ctx.u40], s[ctx.u40], w[ctx.u40], 0.2),
        "pauc_histo": wauc(ctx.esc[ctx.histo], s[ctx.histo], w[ctx.histo], 0.2),
        "macro_f1": float(sc[0]),
        "balanced_accuracy": float(sc[1 + K:].mean()),
        "esc_sens_argmax": float((w * (ctx.esc & esc_pred)).sum() / (w * ctx.esc).sum()),
        "sens_at_spec80_40-59": sens_at_spec(ctx.esc, s, ctx.b4059, w),
        "sens_at_spec80_60+": sens_at_spec(ctx.esc, s, ctx.b60, w),
        "b2_melnv_auc_u40": wauc(ctx.y[ctx.melnv_u40] == MEL, s[ctx.melnv_u40], w[ctx.melnv_u40]),
        "b2_std_pauc_u40": wauc(ctx.esc[ctx.u40], s[ctx.u40], (w * ctx.std_w)[ctx.u40], 0.2),
    }
    for a in ARCHIVES:
        m = ctx.u40 & (ctx.archive == a)
        out[f"b4_pauc_u40_{a}"] = wauc(ctx.esc[m], s[m], w[m], 0.2)
    return out


ENDPOINTS = ["pauc_all", "pauc_u40", "pauc_histo", "macro_f1", "balanced_accuracy", "esc_sens_argmax",
             "sens_at_spec80_40-59", "sens_at_spec80_60+", "b2_melnv_auc_u40", "b2_std_pauc_u40"] + \
            [f"b4_pauc_u40_{a}" for a in ARCHIVES]


def b4_pooled(ctx: Ctx, d: dict) -> float:
    vals = [(ctx.esc_les[a], d[f"b4_pauc_u40_{a}"]) for a in ARCHIVES
            if ctx.esc_les[a] > 0 and np.isfinite(d[f"b4_pauc_u40_{a}"])]
    return float(sum(n * v for n, v in vals) / sum(n for n, _ in vals)) if vals else float("nan")


def contrast(name: str, pairs: list[tuple[int, pd.DataFrame, pd.DataFrame]], ctx: Ctx,
             n_boot: int, rng_seed: int) -> dict:
    """Paired deltas (arm a - arm b) per seed, and the hierarchical bootstrap over seeds and lesions."""
    ones = np.ones(len(ctx.y))
    per_seed = []
    for seed, a, b in pairs:
        ma, mb = metrics(ctx, a, ones), metrics(ctx, b, ones)
        d = {k: ma[k] - mb[k] for k in ENDPOINTS}
        d["b4_pooled_fe"] = b4_pooled(ctx, d)
        per_seed.append({"seed": seed, "a": ma, "b": mb, "delta": d})
    # Cross-check the fast weighted path against the project's metric engine on the first pair.
    cm = compute_metrics(ctx.y, pairs[0][1]["pred_index"].to_numpy(), pairs[0][1][PROB].to_numpy())
    if abs(cm["macro_f1"] - per_seed[0]["a"]["macro_f1"]) > 1e-9:
        raise AssertionError("weighted Macro-F1 disagrees with compute_metrics")
    if abs(cm["balanced_accuracy"] - per_seed[0]["a"]["balanced_accuracy"]) > 1e-9:
        raise AssertionError("weighted balanced accuracy disagrees with compute_metrics")
    s0 = pairs[0][1]["declared_score"].to_numpy(dtype=float)
    if abs(pauc(ctx.esc, s0) - per_seed[0]["a"]["pauc_all"]) > 1e-6:
        raise AssertionError("weighted pAUC disagrees with the frozen McClish pAUC")

    keys = ENDPOINTS + ["b4_pooled_fe"]
    rng = np.random.default_rng(rng_seed)
    boot = {k: [] for k in keys}
    for _ in range(n_boot):
        drawn = rng.integers(0, len(pairs), len(pairs))
        acc = {k: [] for k in keys}
        for j in drawn:
            w = np.bincount(rng.integers(0, ctx.n_lesions, ctx.n_lesions),
                            minlength=ctx.n_lesions)[ctx.lesion].astype(float)
            _, a, b = pairs[j]
            ma, mb = metrics(ctx, a, w), metrics(ctx, b, w)
            d = {k: ma[k] - mb[k] for k in ENDPOINTS}
            d["b4_pooled_fe"] = b4_pooled(ctx, d)
            for k in keys:
                acc[k].append(d[k])
        for k in keys:
            boot[k].append(float(np.nanmean(acc[k])) if np.isfinite(acc[k]).any() else float("nan"))
    summary = {}
    for k in keys:
        arr = np.array(boot[k])
        arr = arr[np.isfinite(arr)]
        point = float(np.nanmean([r["delta"][k] for r in per_seed]))
        lo, hi = (np.quantile(arr, [ALPHA / 2, 1 - ALPHA / 2]) if len(arr) else (np.nan, np.nan))
        p2 = float(min(1.0, 2 * min((arr <= 0).mean(), (arr >= 0).mean()))) if len(arr) else float("nan")
        summary[k] = {"point": point, "ci95": [float(lo), float(hi)], "p_two_sided": p2,
                      "n_boot_valid": int(len(arr))}
    return {"contrast": name, "seeds": [s for s, _, _ in pairs], "per_seed": per_seed, "delta": summary}


def gates(c: dict) -> dict:
    d, n = c["delta"], len(c["seeds"])
    seeds = np.array([r["delta"]["pauc_all"] for r in c["per_seed"]])
    gd = {"per_seed_delta_pauc_all": seeds.tolist(), "mean": float(seeds.mean()),
          "n_positive": int((seeds > 0).sum()), "min": float(seeds.min())}
    if n == 3:
        gd["pass"] = bool(seeds.mean() > 0 and (seeds > 0).sum() >= 2 and seeds.min() >= SEED_FLOOR)
        gd["status"] = "evaluated"
    else:
        gd["pass"] = None
        gd["status"] = f"provisional ({n}/3 seeds) - not a pass or fail"
    return {
        "A": {"rule": "d<40 pAUC >= +0.050 and CI lower > 0", "point": d["pauc_u40"]["point"],
              "ci95": d["pauc_u40"]["ci95"],
              "pass": bool(d["pauc_u40"]["point"] >= GATE_A and d["pauc_u40"]["ci95"][0] > 0)},
        "B": {"rule": "dMacro-F1 >= -0.010 (point)", "point": d["macro_f1"]["point"],
              "ci95": d["macro_f1"]["ci95"], "pass": bool(d["macro_f1"]["point"] >= GATE_B)},
        "C": {"rule": "d sensitivity at spec 0.80 >= -0.030 in 40-59 and 60+",
              "40-59": d["sens_at_spec80_40-59"], "60+": d["sens_at_spec80_60+"],
              "pass": bool(d["sens_at_spec80_40-59"]["point"] >= GATE_C
                           and d["sens_at_spec80_60+"]["point"] >= GATE_C)},
        "D": gd,
    }


def holm(ps: dict[str, float]) -> dict[str, float]:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (len(items) - i) * p))
        out[k] = running
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--rng-seed", type=int, default=20261004)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    testguard.block_test_reads("V5 confirmation read: development folds 1-4 only")

    sha: dict[str, str] = {}
    comp = {s: load("composite", s, sha) for s in args.seeds}
    ctrl = {s: load("control", s, sha) for s in args.seeds}
    ref = ctrl[args.seeds[0]]
    for s in args.seeds:
        for f in (comp[s], ctrl[s]):
            if not (f["image_id"].equals(ref["image_id"]) and f["y_true"].equals(ref["y_true"])):
                raise SystemExit(f"seed {s}: rows or labels differ from the reference")
    ctx = Ctx(ref)

    primary = contrast("composite - control", [(s, comp[s], ctrl[s]) for s in args.seeds], ctx,
                       args.n_boot, args.rng_seed)
    result = {
        "question": "V5 confirmation: locked composite vs in1k control, held-out folds 1-4 (runsheet E14/E18)",
        "declared": "CHANGELOG 2026-10-04 'E14 confirmation read - operational choices DECLARED BEFORE THE READ'",
        "lock": "results/v5/composite_lock.json sha256 a34bd78c4e3098d906bf8cabbc49d0e863f33f20d7fc905f5a34b0081f456059",
        "seeds": args.seeds, "n_rows": int(len(ctx.y)), "n_lesions": int(ctx.n_lesions),
        "n_u40_escalating_images": int((ctx.esc & ctx.u40).sum()),
        "n_u40_escalating_lesions": int(len(np.unique(ctx.lesion[ctx.esc & ctx.u40]))),
        "u40_escalating_lesions_by_archive": ctx.esc_les,
        "bootstrap": {"n_boot": args.n_boot, "rng_seed": args.rng_seed,
                      "scheme": "seed pairs with replacement, then effective_lesion_id clusters within each"},
        "primary": primary, "gates": gates(primary),
    }

    if 42 in args.seeds:
        m4 = load("m4", 42, sha)
        if not (m4["image_id"].equals(ref["image_id"])):
            raise SystemExit("m4 rows differ from the reference")
        sec_ctrl = contrast("m4 - control (secondary, D1)", [(42, m4, ctrl[42])], ctx, args.n_boot, args.rng_seed + 1)
        sec_comp = contrast("composite - m4 (secondary, D1)", [(42, comp[42], m4)], ctx, args.n_boot, args.rng_seed + 2)
        m4_mass = m4.copy()
        m4_mass["declared_score"] = m4_mass["escalation_mass"]
        sec_mass = contrast("m4[escalation mass] - control (descriptive)", [(42, m4_mass, ctrl[42])], ctx,
                            args.n_boot, args.rng_seed + 3)
        adj = holm({"m4 - control": sec_ctrl["delta"]["pauc_all"]["p_two_sided"],
                    "composite - m4": sec_comp["delta"]["pauc_all"]["p_two_sided"]})
        g = gates(sec_ctrl)
        g.pop("D")
        result["secondary_D1"] = {
            "family": "Holm across 2 on d all-age pAUC, seed 42",
            "m4_vs_control": sec_ctrl, "m4_vs_control_gates_secondary": g,
            "composite_vs_m4": sec_comp, "m4_escalation_mass_vs_control_descriptive": sec_mass,
            "holm_adjusted_p": adj,
        }

    result["inputs_sha256"] = sha
    result["test_read"] = False
    name = args.out or f"confirm_s{'_s'.join(str(s) for s in args.seeds)}.json"
    out = OUT_DIR / name
    out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")

    p = primary["delta"]
    note = (f"seeds {args.seeds}; d pAUC_all {p['pauc_all']['point']:+.4f} {p['pauc_all']['ci95']}; "
            f"d pAUC_u40 {p['pauc_u40']['point']:+.4f} {p['pauc_u40']['ci95']}; d MacroF1 "
            f"{p['macro_f1']['point']:+.4f}; gates A={result['gates']['A']['pass']} B={result['gates']['B']['pass']} "
            f"C={result['gates']['C']['pass']} D={result['gates']['D']['status']}")
    row = pd.DataFrame([{"timestamp": pd.Timestamp.now(tz="UTC").isoformat(), "session": "v5_confirm",
                         "method": out.stem, "split": "f1-4", "macro_f1": p["macro_f1"]["point"],
                         "balanced_accuracy": p["balanced_accuracy"]["point"], "notes": note}])
    if LEDGER.is_file():
        old = pd.read_csv(LEDGER, low_memory=False)
        old = old[~((old["session"] == "v5_confirm") & (old["method"] == out.stem))]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(LEDGER, index=False)
    print(f"wrote {out.relative_to(REPO_ROOT)}\n{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
