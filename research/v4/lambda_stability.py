"""S57b phase 7 -- lesion-grouped bootstrap stability of every lambda(age) arm, and Gate 5.

Each replicate resamples OOF *lesions* with replacement, keeps S5's cross-fitted probabilities
fixed, and refits every arm exactly as S57a fit it -- A1 by S5's `fit_lambda_by_group`, A2-A7 by
`lambda_age`, all floor-repaired per band -- at **S57a's full-data hyperparameters** (bandwidth,
kappa, tau). Re-selecting them per replicate would cost ~47 s x N_BOOT; their variability is
measured instead by the cross-fit, which re-selects them in every outer fold
(`lambda_crossfit_audit.json`). Declared.

Per replicate: the 18-knot table of each arm, in-sample under-40 and overall sensitivity,
per-band referral and specificity, and cost. From those: the 1-year-resolution curve with its
**unsmoothed** 95% percentile band (`results/v4/lambda_curve.{csv,json}`) and the figure.

**Gate 5, declared in the plan before reserved is read**, for each confirmatory candidate C:

* **5a resolution** -- mean over the 18 knots of C's bootstrap band width (97.5th - 2.5th)
  must be *smaller* than the mean |C - A1| distance between the two point tables. A band wider
  than the distance between arms means the resolution is not in the data (runbook Failure A).
* **5b sign stability** -- at every knot where C and A1 differ by more than 0.01, the share of
  replicates in which (C* - A1*) has the point estimate's sign; the median over those knots must
  be >= 0.80. A curve whose shape flips under resampling is not frozen.

Leave-one-bin-out influence is reported beside the gate, descriptively: refit C without each
5-year bin in turn and record the largest mean shift of the *other* knots.

    $py -m research.v4.lambda_stability --selftest
    $py -m research.v4.lambda_stability --run [--n-boot 400] [--workers 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.v4 import lambda_age as la  # noqa: E402

BOOT_JSON = REPO_ROOT / "results" / "v4" / "lambda_bootstrap.json"
CURVE_CSV = REPO_ROOT / "results" / "v4" / "lambda_curve.csv"
CURVE_JSON = REPO_ROOT / "results" / "v4" / "lambda_curve.json"
DRAWS_NPZ = REPO_ROOT / "results" / "v4" / "s57b" / "bootstrap_tables.npz"
FIGURE = REPO_ROOT / "paper" / "figures" / "lambda_age_curve.png"
ARMS = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")
KIND = {"A0": "step", "A1": "step", "A2": "step", "A7": "step",
        "A3": "interp", "A4": "interp", "A5": "interp", "A6": "interp"}
N_BOOT = 400                  # S5's bootstrap count
SIGN_MIN = 0.80
DIFF_MIN = 0.01
FINE_AGES = np.arange(0, 86, 1.0)

_DEV: la.Panel | None = None


def hyper() -> dict[str, float]:
    h = json.loads(la.CANDIDATES_JSON.read_text(encoding="utf-8"))["hyperparameters"]
    return {"bandwidth": float(h["bandwidth"]), "kappa": float(h["kappa"]), "tau": float(h["tau"])}


def point_tables() -> dict[str, np.ndarray]:
    arms = json.loads(la.CANDIDATES_JSON.read_text(encoding="utf-8"))["arms"]
    return {a: np.array([arms[a]["table"][str(int(k))] for k in la.AGE_KNOTS]) for a in arms}


def fit_all(panel: la.Panel, hp: dict[str, float], arms: tuple[str, ...] = ARMS) -> dict[str, la.Arm]:
    """Every arm on one panel, at fixed hyperparameters, exactly as S57a fits them."""
    esc, cm = la.esc_indices(), la.cost_matrix()
    scorer = la.Scorer(panel, esc, cm)
    pooled = la.fit_pooled(panel, esc, cm)
    a1_frozen = la.arm_a1().knots          # warm start only, as in S57a
    out: dict[str, la.Arm] = {}
    if "A1" in arms:
        out["A1"] = la.refit_a1(panel, esc, cm)
    if "A2" in arms:
        out["A2"] = la.repair_floor(la.arm_a2(panel, esc, cm, pooled), scorer)
    if "A3" in arms:
        out["A3"] = la.repair_floor(la.arm_a3(la.GridTables(scorer), hp["bandwidth"], pooled), scorer)
    if "A4" in arms:
        out["A4"] = la.repair_floor(la.arm_link("A4", scorer, "linear", pooled, 0.0, a1_frozen), scorer)
    if "A5" in arms:
        out["A5"] = la.repair_floor(la.arm_link("A5", scorer, "quadratic", pooled, 0.0, a1_frozen), scorer)
    if "A6" in arms or "A7" in arms:
        a6 = la.arm_link("A6", scorer, "spline", pooled, hp["kappa"], a1_frozen)
        out["A6"] = la.repair_floor(a6, scorer)
        if "A7" in arms:
            out["A7"] = la.repair_floor(la.arm_a7(la.local_fits(panel, esc, cm), a6, hp["tau"], pooled), scorer)
    return {a: out[a] for a in arms if a in out}


def summarise(panel: la.Panel, arm: la.Arm) -> dict[str, float]:
    esc, cm = la.esc_indices(), la.cost_matrix()
    sc = la.Scorer(panel, esc, cm)
    lam = arm.lam_for(panel.ages)
    f = sc.dec.flagged(lam)
    te = sc.dec.true_esc
    out = {"cost": sc.cost(lam), "sens_all": float(f[te].mean())}
    for b, s in zip(la.FLOOR_BANDS, sc.band_spec(lam)):
        m = panel.bands == b
        out[f"sens|{b}"] = float(f[m & te].mean()) if (m & te).any() else np.nan
        out[f"referral|{b}"] = float(f[m].mean())
        out[f"spec|{b}"] = float(s)
    return out


def _init(dummy: int) -> None:
    global _DEV
    _DEV = la.dev_panel()


def _replicate(seed: int) -> dict[str, Any]:
    assert _DEV is not None
    rng = np.random.default_rng(seed)
    idx = lesion_resample(_DEV.lesion, rng)
    panel = _DEV.subset(idx, name=f"boot{seed}")
    arms = fit_all(panel, hyper())
    return {"seed": seed, "tables": {a: arms[a].knots.tolist() for a in arms},
            "metrics": {a: summarise(panel, arms[a]) for a in arms}}


def lesion_resample(lesion: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    codes, uniq = pd.factorize(pd.Series(lesion))
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(len(uniq)))
    ends = np.r_[starts[1:], len(codes)]
    pick = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([order[starts[g]:ends[g]] for g in pick])


def gate5(point: dict[str, np.ndarray], draws: dict[str, np.ndarray], cand: str) -> dict[str, Any]:
    width = np.percentile(draws[cand], 97.5, axis=0) - np.percentile(draws[cand], 2.5, axis=0)
    dist = np.abs(point[cand] - point["A1"])
    diff_knots = np.flatnonzero(dist > DIFF_MIN)
    sign = np.sign(point[cand] - point["A1"])
    agree = [float(np.mean(np.sign(draws[cand][:, j] - draws["A1"][:, j]) == sign[j])) for j in diff_knots]
    a = {"mean_band_width": float(width.mean()), "mean_distance_to_A1": float(dist.mean()),
         "under40_mean_band_width": float(width[:8].mean()), "under40_mean_distance_to_A1": float(dist[:8].mean()),
         "pass": bool(width.mean() < dist.mean())}
    b = {"knots_compared": [int(la.AGE_KNOTS[j]) for j in diff_knots],
         "sign_agreement": agree, "median_sign_agreement": float(np.median(agree)) if agree else float("nan"),
         "threshold": SIGN_MIN, "pass": bool(agree and np.median(agree) >= SIGN_MIN)}
    return {"candidate": cand, "5a_resolution": a, "5b_sign_stability": b, "pass": bool(a["pass"] and b["pass"])}


def leave_one_bin_out(dev: la.Panel, cand: str, point: np.ndarray, hp: dict[str, float]) -> dict[str, Any]:
    shifts = {}
    for b in la.BINS:
        keep = np.flatnonzero(dev.bins != b)
        arm = fit_all(dev.subset(keep, name=f"lobo{b}"), hp, (cand,))[cand]
        others = np.array([la.age_bin(np.array([k]))[0] != b for k in la.AGE_KNOTS])
        shifts[la.bin_label(b)] = float(np.abs(arm.knots - point)[others].mean())
    worst = max(shifts, key=shifts.get)
    return {"mean_abs_shift_other_knots": shifts, "most_influential_bin": worst, "max_shift": shifts[worst],
            "label": "descriptive (not a gate)"}


def curve(point: dict[str, np.ndarray], draws: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for a in draws:
        pa = la.Arm(a, KIND[a], point[a], 0.0)
        fine_point = pa.lam_for(FINE_AGES)
        fine_draws = np.array([la.Arm(a, KIND[a], t, 0.0).lam_for(FINE_AGES) for t in draws[a]])
        for i, age in enumerate(FINE_AGES):
            rows.append({"arm": a, "age": int(age), "lambda_point": float(fine_point[i]),
                         "boot_median": float(np.median(fine_draws[:, i])),
                         "boot_lo": float(np.percentile(fine_draws[:, i], 2.5)),
                         "boot_hi": float(np.percentile(fine_draws[:, i], 97.5))})
    return pd.DataFrame(rows)


def plot(cv: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = ("A2", "A3", "A6", "A7")
    fig, axes = plt.subplots(1, len(shown), figsize=(16, 4), sharey=True)
    a1 = cv[cv.arm == "A1"]
    for ax, a in zip(axes, shown):
        d = cv[cv.arm == a]
        ax.fill_between(a1.age, a1.boot_lo, a1.boot_hi, step="post", color="C3", alpha=0.15, label="A1 95% band")
        ax.step(a1.age, a1.lambda_point, where="post", color="C3", label="A1 frozen 3-band")
        step = "post" if KIND[a] == "step" else None
        if step:
            ax.fill_between(d.age, d.boot_lo, d.boot_hi, step="post", color="C0", alpha=0.25, label=f"{a} 95% band")
            ax.step(d.age, d.lambda_point, where="post", color="C0", label=f"{a} full-data fit")
        else:
            ax.fill_between(d.age, d.boot_lo, d.boot_hi, color="C0", alpha=0.25, label=f"{a} 95% band")
            ax.plot(d.age, d.lambda_point, color="C0", label=f"{a} full-data fit")
        ax.plot(d.age, d.boot_median, color="C0", ls=":", lw=1, label="bootstrap median")
        ax.set_title(f"{a} vs A1 (lesion-grouped bootstrap, unsmoothed)", fontsize=9)
        ax.set_xlabel("patient age")
        for x in (40, 60):
            ax.axvline(x, color="0.8", lw=0.8)
        ax.legend(fontsize=6, loc="upper right")
    axes[0].set_ylabel("λ(age)")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=170)
    plt.close(fig)


def run(n_boot: int, workers: int, candidates: list[str]) -> int:
    from research import testguard

    testguard.block_test_reads("S57b stability: OOF only")
    hp = hyper()
    point = point_tables()
    dev = la.dev_panel()
    # the full-data refit at these hyperparameters must reproduce S57a's tables
    refit = fit_all(dev, hp)
    repro = {a: float(np.abs(refit[a].knots - point[a]).max()) for a in ARMS if a != "A1"}
    repro["A1_refit_vs_frozen"] = float(np.abs(refit["A1"].knots - point["A1"]).max())
    print(f"[phase 7] full-data refit vs S57a tables, max |diff|: {repro}", flush=True)
    assert max(repro.values()) < 1e-9, "stability refit does not reproduce S57a's tables"

    t0 = time.time()
    seeds = [la.SEED + 1000 + i for i in range(n_boot)]
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(0,)) as ex:
        for i, r in enumerate(ex.map(_replicate, seeds, chunksize=4)):
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{n_boot} replicates, {time.time() - t0:.0f}s", flush=True)
    draws = {a: np.array([r["tables"][a] for r in results]) for a in ARMS}
    assert all(np.isfinite(d).all() for d in draws.values()), "a bootstrap lambda is not finite"
    metrics = {a: pd.DataFrame([r["metrics"][a] for r in results]) for a in ARMS}
    DRAWS_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DRAWS_NPZ, seeds=np.array(seeds), **{a: draws[a] for a in ARMS})

    cv = curve(point, draws)
    cv.to_csv(CURVE_CSV, index=False, lineterminator="\n")
    gates = {c: gate5(point, draws, c) for c in candidates}
    lobo = {c: leave_one_bin_out(dev, c, point[c], hp) for c in candidates}
    summary = {a: {k: {"median": float(metrics[a][k].median()),
                       "lo": float(metrics[a][k].quantile(0.025)), "hi": float(metrics[a][k].quantile(0.975))}
                   for k in metrics[a].columns} for a in ARMS}
    payload = {
        "session": la.SESSION.replace("s57a", "s57b"), "phase": 7, "split": "HAM OOF, S5 cross-fitted probs",
        "test_read": False, "reserved_read": False, "n_boot": n_boot, "resampling": "lesion-grouped",
        "hyperparameters_fixed_at_s57a": hp, "refit_reproduces_s57a_tables": repro,
        "seeds": [seeds[0], seeds[-1]], "draws_file": la.rel(DRAWS_NPZ), "draws_sha256": la.sha256(DRAWS_NPZ),
        "in_sample_metric_bands": summary,
        "gate5": gates, "gate5_rule": {"5a": "mean band width < mean |C - A1|",
                                       "5b": f"median sign agreement >= {SIGN_MIN} at knots differing by > {DIFF_MIN}"},
        "leave_one_bin_out": lobo,
        "note": "A1 draws are S5 refits on each replicate (the frozen table's own sampling variability)",
    }
    la._dump(BOOT_JSON, la._clean(payload))
    la._dump(CURVE_JSON, la._clean({"resolution_years": 1, "source": la.rel(CURVE_CSV),
                                    "arms": {a: {"kind": KIND[a], "points": cv[cv.arm == a].drop(columns="arm")
                                                 .to_dict(orient="records")} for a in ARMS},
                                    "band": "percentile 2.5-97.5, lesion-grouped, not smoothed"}))
    plot(cv)
    for c, g in gates.items():
        print(f"  Gate 5 {c}: 5a width {g['5a_resolution']['mean_band_width']:.3f} vs distance "
              f"{g['5a_resolution']['mean_distance_to_A1']:.3f} -> {g['5a_resolution']['pass']}; "
              f"5b median sign {g['5b_sign_stability']['median_sign_agreement']:.2f} -> {g['5b_sign_stability']['pass']}; "
              f"LOBO worst {lobo[c]['most_influential_bin']} {lobo[c]['max_shift']:.3f}")
    for a in ARMS:
        s = summary[a]
        print(f"  {a}: <40 sens {s['sens|<40']['median']:.3f} [{s['sens|<40']['lo']:.3f}, {s['sens|<40']['hi']:.3f}]"
              f"  <40 referral {s['referral|<40']['median']:.3f}")
    print(f"  {time.time() - t0:.0f}s")
    return 0


def selftest() -> int:
    fails = 0

    def check(name: str, ok: bool) -> None:
        nonlocal fails
        fails += not ok
        print(("PASS " if ok else "FAIL ") + name)

    rng = np.random.default_rng(0)
    les = np.repeat(np.arange(30), 3).astype(str)
    idx = lesion_resample(les, rng)
    groups = pd.Series(idx).groupby(les[idx]).size()
    check("resampling keeps lesions whole", (groups % 3 == 0).all())
    pt = {"A1": np.full(18, 0.3), "C": np.r_[np.full(8, 0.8), np.full(10, 0.3)]}
    tight = {"A1": 0.3 + rng.normal(0, 0.01, (200, 18)), "C": pt["C"] + rng.normal(0, 0.01, (200, 18))}
    g = gate5(pt, tight, "C")
    check("gate 5 passes a tight, well-separated curve", g["pass"])
    loose = {"A1": 0.3 + rng.normal(0, 0.01, (200, 18)), "C": pt["C"] + rng.normal(0, 0.6, (200, 18))}
    check("gate 5 fails a curve whose band is wider than its distance", not gate5(pt, loose, "C")["pass"])
    flat = {"A1": pt["A1"], "C": pt["A1"].copy()}
    check("gate 5 fails a curve identical to A1 (no knots to compare)",
          not gate5(flat, {"A1": tight["A1"], "C": tight["A1"]}, "C")["pass"])
    a = la.Arm("A6", "interp", np.linspace(0, 1, 18), 0.5)
    v = a.lam_for(np.array([np.nan, -5, 200, 42.5]))
    check("missing age -> pooled; out-of-range ages clamp; never NaN",
          v[0] == 0.5 and v[1] == 0.0 and v[2] == 1.0 and np.isfinite(v).all())
    s = la.Arm("A7", "step", np.arange(18.0), 9.0)
    check("step arm: 80 and 85 share the 80+ value", s.lam_for(np.array([80.0, 85.0]))[0] == s.lam_for(np.array([85.0]))[0])
    print(f"{6 - fails}/6 passed")
    return int(fails > 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--candidates", nargs="+", default=None,
                    help="default: the confirmatory family in lambda_crossfit_audit.json")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.run:
        cands = args.candidates
        if cands is None:
            from research.v4.lambda_crossfit import AUDIT_JSON
            cands = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))["confirmatory_family"]
        return run(args.n_boot, args.workers, cands)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
