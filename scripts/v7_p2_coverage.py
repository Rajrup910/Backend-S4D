"""F4b (V7 plan 13.1, freeze audit of 11 Nov 2026): what does P2 itself do at the under-40 sample size?

`scripts/v7_interval_coverage.py` measured coverage of the under-40 pAUC interval at two true deltas and fixed the GAIN
rule. It did not answer the question P2 asks: P2 passes a candidate when the UPPER 95% bound of the paired delta
(candidate - V6) is >= 0, for (a) the under-40 escalation pAUC and (b) the all-age argmax escalation sensitivity
(`research.v6.system_reads.no_harm_verdict`; `esc_sens` is read on all rows, not on the under-40 band). So this script
simulates P2's two components separately and reports, per scenario:

* coverage of the two-sided 95% percentile interval, and each one-sided error, with Monte Carlo standard errors;
* **P2 pass rate**: how often the component passes. At a true harm this is the FALSE-PASS probability; 1 minus it is the
  power to detect that harm. At no difference it is the chance a harmless candidate survives;
* for information only (no margin is adopted here; the margin is the owner's decision): how often a non-inferiority
  rule "lower bound > -m" would pass, on a grid of illustrative margins m, with the two-sided 95% and one-sided 95% bounds.

Component (a) reuses the synthetic population of `v7_interval_coverage.py` (65 positive and 1,229 negative under-40 lesion
groups, logit means, SDs and within-lesion ICCs measured on the banked V6 vote, folds 1-4). The candidate's positives are
shifted by `shift` logits; truth is the pAUC delta on a population 200x the sample.

Component (b) is a probit model of the argmax call on the 1,683 all-age escalating lesion groups of folds 1-4 (4,258
images; group sizes as observed; V6 image-level sensitivity 0.8255; measured 11 Nov from the banked V6 vote). A lesion
latent and an image residual (latent ICC 0.93) are shared between the two systems with correlation `rho`; the candidate's
call threshold is moved so its true sensitivity is 0.8255 + delta. Truth is delta exactly.

Synthetic data only. Reads no prediction file. CPU; uses `--workers` processes.

    python scripts/v7_p2_coverage.py --trials 1000 --n-boot 400 --workers 6
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "v7" / "p2_coverage.json"
RAW = REPO / "results" / "v7" / "p2_coverage_trials.npz"
PCTS = (1.25, 2.5, 5.0, 95.0, 97.5, 98.75)
PAUC_GRID = [(rho, shift) for shift in (0.0, -0.25, -0.5, -1.0, 0.5) for rho in (0.85, 0.95)]
SENS_GRID = [(rho, delta) for delta in (0.0, -0.01, -0.02, -0.03, -0.05, 0.02) for rho in (0.85, 0.95)]
PAUC_MARGINS = (0.02, 0.03, 0.05, 0.08)      # illustrative only; none is adopted
SENS_MARGINS = (0.01, 0.02, 0.03, 0.05)      # illustrative only; none is adopted
# All-age escalating lesion groups, folds 1-4, banked V6 vote (measured 11 Nov 2026; see results/v7/freeze_audit/).
SENS_GROUP_SIZES = {1: 654, 2: 485, 3: 247, 4: 94, 5: 61, 6: 52, 7: 24, 8: 11, 9: 19, 10: 8, 11: 7, 12: 6, 13: 4, 14: 2,
                    15: 1, 17: 2, 18: 2, 19: 2, 20: 1, 21: 1}
SENS_GROUPS = 1683
SENS_REF = 0.8255
SENS_LATENT_ICC = 0.93


def _cov():
    spec = importlib.util.spec_from_file_location("v7_interval_coverage", REPO / "scripts" / "v7_interval_coverage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def pauc_truth(args: tuple[float, float, int]) -> float:
    rho, shift, seed = args
    cov = _cov()
    r, c, y, _ = cov.draw(65 * 200, 1229 * 200, rho, shift, np.random.default_rng(seed))
    return cov.pauc(y, c) - cov.pauc(y, r)


def pauc_trials(args: tuple[float, float, int, int, int]) -> np.ndarray:
    """One chunk of trials for component (a). Rows: point, then the percentiles in PCTS."""
    rho, shift, trials, n_boot, seed = args
    cov = _cov()
    rng = np.random.default_rng(seed)
    out = np.empty((trials, 1 + len(PCTS)))
    for t in range(trials):
        r, c, y, g = cov.draw(65, 1229, rho, shift, rng)
        point, deltas = cov.interval(r, c, y, g, n_boot, rng)
        out[t] = [point, *np.percentile(deltas, PCTS)]
    return out


def sens_trials(args: tuple[float, float, int, int, int]) -> np.ndarray:
    """One chunk of trials for component (b): paired argmax escalation sensitivity on all-age escalating lesion groups."""
    rho, delta, trials, n_boot, seed = args
    rng = np.random.default_rng(seed)
    sizes_k = np.array(list(SENS_GROUP_SIZES))
    sizes_p = np.array(list(SENS_GROUP_SIZES.values()), float)
    t_ref, t_cand = norm.ppf(1 - SENS_REF), norm.ppf(1 - (SENS_REF + delta))
    lesion_sd, image_sd = np.sqrt(SENS_LATENT_ICC), np.sqrt(1 - SENS_LATENT_ICC)
    out = np.empty((trials, 1 + len(PCTS)))
    for t in range(trials):
        m = rng.choice(sizes_k, SENS_GROUPS, p=sizes_p / sizes_p.sum())
        rep = np.repeat(np.arange(SENS_GROUPS), m)
        shared_l = rng.normal(0, 1, SENS_GROUPS)
        lat_r = np.sqrt(rho) * shared_l + np.sqrt(1 - rho) * rng.normal(0, 1, SENS_GROUPS)
        lat_c = np.sqrt(rho) * shared_l + np.sqrt(1 - rho) * rng.normal(0, 1, SENS_GROUPS)
        shared_i = rng.normal(0, 1, len(rep))
        img_r = np.sqrt(rho) * shared_i + np.sqrt(1 - rho) * rng.normal(0, 1, len(rep))
        img_c = np.sqrt(rho) * shared_i + np.sqrt(1 - rho) * rng.normal(0, 1, len(rep))
        called_r = (lesion_sd * lat_r[rep] + image_sd * img_r) > t_ref
        called_c = (lesion_sd * lat_c[rep] + image_sd * img_c) > t_cand
        diff = np.bincount(rep, weights=called_c.astype(float) - called_r, minlength=SENS_GROUPS)
        point = diff.sum() / m.sum()
        # lesion-group bootstrap: multinomial counts over groups, both systems scored on the same resample
        w = rng.multinomial(SENS_GROUPS, np.full(SENS_GROUPS, 1 / SENS_GROUPS), size=n_boot)
        deltas = (w @ diff) / (w @ m)
        out[t] = [point, *np.percentile(deltas, PCTS)]
    return out


def mcse(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1 - p), 0.0) / n))


def summarise(trials: np.ndarray, truth: float, margins: tuple[float, ...]) -> dict:
    n = len(trials)
    point, lo975, lo95, lo90, hi90, hi95, hi975 = trials.T   # PCTS order: 1.25, 2.5, 5, 95, 97.5, 98.75
    rate = lambda mask: {"rate": float(mask.mean()), "mc_se": mcse(float(mask.mean()), n)}  # noqa: E731
    return {
        "true_delta": float(truth), "trials": int(n), "mean_point": float(point.mean()), "sd_point": float(point.std(ddof=1)),
        "bias": float(point.mean() - truth), "median_width_95": float(np.median(hi95 - lo95)),
        "coverage_95": rate((lo95 <= truth) & (truth <= hi95)),
        "upper95_below_truth": rate(hi95 < truth), "lower95_above_truth": rate(lo95 > truth),
        "coverage_97.5": rate((lo975 <= truth) & (truth <= hi975)),
        "p2_pass_upper95_ge_0": rate(hi95 >= 0),
        "p2_pass_if_upper97.5": rate(hi975 >= 0),
        "illustrative_noninferiority_pass": {
            f"margin_{m:g}": {"two_sided_95_lower": rate(lo95 > -m), "one_sided_95_lower": rate(lo90 > -m)} for m in margins},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P2's two components at the real sample sizes: coverage, false-pass, power.")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--raw", default=str(RAW))
    args = parser.parse_args(argv)
    out, raw = Path(args.out), Path(args.raw)
    if out.exists() or raw.exists():
        raise SystemExit(f"{out.name} or {raw.name} exists: results are not overwritten; move them or pass --out/--raw")
    start = time.time()
    chunks = max(args.workers, 1) * 2
    per = int(np.ceil(args.trials / chunks))
    seeds = np.random.SeedSequence(args.seed).generate_state(4 * (len(PAUC_GRID) + len(SENS_GRID)) * (chunks + 1)).tolist()
    take = iter(seeds)
    with Pool(args.workers) as pool:
        truths = pool.map(pauc_truth, [(rho, shift, next(take)) for rho, shift in PAUC_GRID])
        pauc_jobs = [(rho, shift, per, args.n_boot, next(take)) for rho, shift in PAUC_GRID for _ in range(chunks)]
        sens_jobs = [(rho, delta, per, 2000, next(take)) for rho, delta in SENS_GRID for _ in range(chunks)]
        pauc_out = pool.map(pauc_trials, pauc_jobs, chunksize=1)
        sens_out = pool.map(sens_trials, sens_jobs, chunksize=1)
    arrays, rows_pauc, rows_sens = {}, [], []
    for i, (rho, shift) in enumerate(PAUC_GRID):
        trials = np.concatenate(pauc_out[i * chunks:(i + 1) * chunks])[:args.trials]
        arrays[f"pauc_rho{rho}_shift{shift}"] = trials
        rows_pauc.append({"rho": rho, "shift_logits": shift, "n_boot": args.n_boot, **summarise(trials, truths[i], PAUC_MARGINS)})
    for i, (rho, delta) in enumerate(SENS_GRID):
        trials = np.concatenate(sens_out[i * chunks:(i + 1) * chunks])[:args.trials]
        arrays[f"sens_rho{rho}_delta{delta}"] = trials
        rows_sens.append({"rho": rho, "n_boot": 2000, **summarise(trials, delta, SENS_MARGINS)})
    report = {
        "what": "F4b: P2's two components simulated separately: coverage, P2 pass rate (false-pass at a true harm), illustrative NI rules",
        "p2_rule_in_code": "passes when the upper 95% percentile bound of (candidate - V6) is >= 0 for under-40 escalation pAUC AND all-age "
                           "argmax escalation sensitivity (research/v6/system_reads.py::no_harm_verdict)",
        "calibration": {"pauc": {"pos_groups": 65, "neg_groups": 1229, "source": "scripts/v7_interval_coverage.py constants (banked V6 vote, under-40, folds 1-4)"},
                        "sensitivity": {"escalating_groups": SENS_GROUPS, "images": 4258, "reference_sensitivity": SENS_REF,
                                        "latent_icc": SENS_LATENT_ICC, "group_sizes": SENS_GROUP_SIZES,
                                        "source": "banked V6 vote, all ages, folds 1-4 (measured 11 Nov 2026)"}},
        "percentiles_stored_per_trial": list(PCTS), "margins_are": "illustrative only; no margin is adopted by this file",
        "under40_pauc": rows_pauc, "all_age_escalation_sensitivity": rows_sens,
        "seed": args.seed, "workers": args.workers, "seconds": round(time.time() - start, 1),
    }
    np.savez_compressed(raw, **arrays)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name, rows in (("under-40 pAUC", rows_pauc), ("all-age escalation sensitivity", rows_sens)):
        print(name)
        for s in rows:
            print(f"  rho {s['rho']:.2f} truth {s['true_delta']:+.4f}: coverage95 {s['coverage_95']['rate']:.3f} (se {s['coverage_95']['mc_se']:.3f}) "
                  f"P2 pass {s['p2_pass_upper95_ge_0']['rate']:.3f} (se {s['p2_pass_upper95_ge_0']['mc_se']:.3f}) sd {s['sd_point']:.4f}")
    print(f"{report['seconds']} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
