"""F4 (V7 plan 13.1): does the lesion-group percentile bootstrap interval cover the truth at the under-40 sample size?

P2 decides on the upper bound of the paired under-40 escalation-pAUC delta. With 65 positive lesion groups the interval's
coverage is not guaranteed; a literature source (Kottas et al. 2014) finds Wald-type AUC intervals well below nominal at
1:9 imbalance, and ours is about 1:19 in groups.

Resampling our own rows cannot measure coverage (that is the bootstrap checking itself). So this builds a synthetic
population calibrated to the banked V6 under-40 rows on folds 1-4 (measured 11 Nov, `scripts/v7_verification_strata.py`
loader): positive groups 65 with the observed group-size distribution, negative groups 1,229 likewise; escalation logits
normal with the observed means and SDs (positives 0.45, 2.12; negatives -2.96, 1.24) and the observed within-lesion
clustering (ICC about 0.93 and 0.89). Two models share a lesion-level latent with correlation `rho`; the candidate's
positives are shifted by `shift` logits. Truth is the pAUC delta on a population 200x the sample. Each trial draws a sample,
builds the paired lesion-group percentile interval exactly as `research.v7.stats.paired_cluster_bootstrap` does, and
records whether it contains the truth, and how often the upper bound falls below it (the error that would let P2 pass
a harmful candidate).

Synthetic data only; reads no predictions beyond the calibration constants above.

    python scripts/v7_interval_coverage.py --trials 400 --n-boot 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "v7" / "interval_coverage.json"
FPR_MAX = 0.20
POS_SIZES = {1: 24, 2: 16, 3: 16, 4: 5, 5: 2, 7: 1}          # 65 groups, 158 images (measured)
NEG_SIZES = {1: 821, 2: 233, 3: 112, 4: 25, 5: 10, 6: 12}   # top of the measured distribution; rest folded into 6
POS = (0.45, 2.12, 0.93)   # mean, sd, ICC (measured)
NEG = (-2.96, 1.24, 0.89)


def _sizes(table: dict[int, int], n_groups: int, rng: np.random.Generator) -> np.ndarray:
    k = np.array(list(table)); p = np.array(list(table.values()), float)
    return rng.choice(k, n_groups, p=p / p.sum())


def draw(n_pos: int, n_neg: int, rho: float, shift: float, rng: np.random.Generator):
    """Image rows for two models: (score_ref, score_cand, y, group)."""
    out = []
    gid = 0
    for label, n, sizes, (mu, sd, icc) in ((1, n_pos, POS_SIZES, POS), (0, n_neg, NEG_SIZES, NEG)):
        m = _sizes(sizes, n, rng)
        lesion = rng.normal(0, sd * np.sqrt(icc), n)                    # shared lesion latent
        # model-specific lesion deviation with correlation rho between the two models' lesion effects
        dev_r = rng.normal(0, sd * np.sqrt(icc * (1 - rho)), n)
        dev_c = rng.normal(0, sd * np.sqrt(icc * (1 - rho)), n)
        base_r = mu + np.sqrt(rho) * lesion + dev_r
        base_c = mu + np.sqrt(rho) * lesion + dev_c + (shift if label else 0.0)
        rep = np.repeat(np.arange(n), m)
        noise_sd = sd * np.sqrt(1 - icc)
        img_shared = rng.normal(0, noise_sd, len(rep))
        r = base_r[rep] + np.sqrt(rho) * img_shared + rng.normal(0, noise_sd * np.sqrt(1 - rho), len(rep))
        c = base_c[rep] + np.sqrt(rho) * img_shared + rng.normal(0, noise_sd * np.sqrt(1 - rho), len(rep))
        out.append((r, c, np.full(len(rep), label), rep + gid))
        gid += n
    return tuple(np.concatenate(x) for x in zip(*out))


def pauc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s, max_fpr=FPR_MAX))


def interval(r, c, y, g, n_boot: int, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    codes = np.unique(g, return_inverse=True)[1]
    order = np.argsort(codes, kind="stable")
    bounds = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1, len(codes)]
    members = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    point = pauc(y, c) - pauc(y, r)
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        if len(np.unique(y[idx])) == 2:
            deltas.append(pauc(y[idx], c[idx]) - pauc(y[idx], r[idx]))
    return point, np.asarray(deltas)


def scenario(rho: float, shift: float, trials: int, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    r, c, y, g = draw(65 * 200, 1229 * 200, rho, shift, rng)
    truth = pauc(y, c) - pauc(y, r)
    levels = {"95": (2.5, 97.5), "97.5": (1.25, 98.75)}
    tally = {k: {"cover": 0, "below": 0, "above": 0, "widths": []} for k in levels}
    for _ in range(trials):
        r, c, y, g = draw(65, 1229, rho, shift, rng)
        _, deltas = interval(r, c, y, g, n_boot, rng)
        for k, (a, b) in levels.items():
            lo, hi = float(np.percentile(deltas, a)), float(np.percentile(deltas, b))
            t = tally[k]
            t["cover"] += lo <= truth <= hi
            t["below"] += hi < truth   # upper bound below truth: harm read as larger than it is (P2's conservative side)
            t["above"] += lo > truth   # lower bound above truth: a gain overstated (anti-conservative for gain claims)
            t["widths"].append(hi - lo)
    n = trials
    out = {"rho": rho, "shift_logits": shift, "true_delta": truth, "trials": n, "n_boot": n_boot,
           "coverage_mc_se_at_nominal": float(np.sqrt(0.95 * 0.05 / n))}
    for k, t in tally.items():
        out[f"level_{k}"] = {"coverage": t["cover"] / n, "upper_bound_below_truth": t["below"] / n,
                             "lower_bound_above_truth": t["above"] / n, "median_width": float(np.median(t["widths"]))}
    out.update(out["level_95"])  # top-level fields keep the 95% interval (the one in use)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Coverage of the paired lesion-group bootstrap at the under-40 size.")
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args(argv)
    start = time.time()
    grid = [(0.85, 0.0), (0.95, 0.0), (0.85, -0.5), (0.95, -0.5)]   # no difference; candidate worse on positives
    rows = [scenario(rho, shift, args.trials, args.n_boot, args.seed + i) for i, (rho, shift) in enumerate(grid)]
    report = {"what": "F4: coverage of the paired lesion-group percentile 95% interval for the under-40 escalation pAUC delta",
              "calibration": {"pos_groups": 65, "neg_groups": 1229, "pos": POS, "neg": NEG,
                              "source": "banked V6 vote, under-40 rows, folds 1-4 (measured 11 Nov)"},
              "decision_rule": "coverage >= 0.93 in every scenario: keep the percentile interval; else a declared wider interval",
              "scenarios": rows, "seconds": round(time.time() - start, 1)}
    report["passes_rule"] = all(s["coverage"] >= 0.93 for s in rows)
    report["level_97.5_passes_rule"] = all(s["level_97.5"]["coverage"] >= 0.93 for s in rows)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for s in rows:
        print(f"rho {s['rho']:.2f} shift {s['shift_logits']:+.1f}: truth {s['true_delta']:+.4f} coverage {s['coverage']:.3f} "
              f"(MC se {s['coverage_mc_se_at_nominal']:.3f}) upper<truth {s['upper_bound_below_truth']:.3f} "
              f"lower>truth {s['lower_bound_above_truth']:.3f} width {s['median_width']:.3f}")
    for s in rows:
        t = s["level_97.5"]
        print(f"  97.5% interval: coverage {t['coverage']:.3f} upper<truth {t['upper_bound_below_truth']:.3f} "
              f"lower>truth {t['lower_bound_above_truth']:.3f} width {t['median_width']:.3f}")
    print("97.5% passes rule:", report["level_97.5_passes_rule"])
    print("passes rule:", report["passes_rule"], f"({report['seconds']} s) -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
