"""S57b phase 4 -- cross-fitting the lambda(age) function itself, plus the development ablation.

S57a's *reported* arms were fit on all 6,981 OOF rows, so their OOF numbers are in-sample for
lambda. Here every arm is fit five times, once per outer fold ``k``, on the other four folds, and
only ever applied to fold ``k``:

    Dirichlet_k^inner  <- folds != k, cross-fitted among themselves -> probs lambda is fit on
    hyperparameters_k  <- chosen by S57a's nested CV *inside* folds != k (bandwidth, kappa, tau)
    lambda_k(.)        <- folds != k only
    Dirichlet_k        <- folds != k                                 -> scores fold k
    fold k             <- scored by lambda_k over Dirichlet_k; its labels touch nothing above

That is `lambda_age.nested_cv_panels` + `lambda_age.fit_candidates(train, cv=True)`, with one
change: S57a's A1 is the *frozen* S5 table (fit on all folds), so inside a fold A1 is **refit** by
S5's own `fit_lambda_by_group` (`lambda_age.refit_a1`). Every assertion is written into
`results/v4/lambda_crossfit_audit.json`.

Also here, because they are development-only and must exist before the gates are frozen:

* the **ablation on cross-fitted OOF** (`results/v4/s57b/ablation_dev.csv`) -- attribution steps
  A0->A1 (age conditioning), A1->A2 (resolution), A2->A3/A6 (smoothing), A2->A7 (shrinkage), with
  per-band referral beside every sensitivity;
* the **dev-best rule** that names the second confirmatory candidate (declared in the plan before
  it runs): among A2-A7, the highest cross-fitted under-40 sensitivity, ties -> lower cross-fitted
  cost. A7 is always a member (the runbook's primary candidate);
* the **deviance diagnostic**: logistic P(escalating | score, age) with 3 band steps vs the same
  plus a cubic B-spline in age -- "does age carry more structure than three steps capture", a
  different and easier question than the clinical gate. Neither substitutes for the other.

    $py -m research.v4.lambda_crossfit --selftest
    $py -m research.v4.lambda_crossfit --run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.v4 import lambda_age as la  # noqa: E402

S57B_DIR = REPO_ROOT / "results" / "v4" / "s57b"
AUDIT_JSON = REPO_ROOT / "results" / "v4" / "lambda_crossfit_audit.json"
ROWS_CSV = S57B_DIR / "crossfit_rows.csv"
ABLATION_DEV_CSV = S57B_DIR / "ablation_dev.csv"
SESSION = "v4_s57b"
ARMS = ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7")
FITTED = ("A2", "A3", "A4", "A5", "A6", "A7")
PRIMARY_CANDIDATE = "A7"
STEPS = (("A0", "A1", "age conditioning"), ("A1", "A2", "5-year resolution"),
         ("A2", "A3", "kernel smoothing"), ("A2", "A6", "spline smoothing"),
         ("A2", "A7", "shrinkage"), ("A1", "A7", "primary candidate vs frozen rule"))
N_BOOT_DEV = 1000


# ============================================================================ cross-fit
def crossfit(dev: la.Panel, esc: list[int], cm: np.ndarray) -> dict[str, Any]:
    folds = la.nested_cv_panels(dev)
    n = len(dev)
    lam = {a: np.full(n, np.nan) for a in ARMS}
    probs = np.full_like(dev.probs, np.nan)
    audit, hypers = [], []
    for tr, ho, prov in folds:
        k = prov["outer_fold"]
        idx_ho = np.flatnonzero(dev.folds == k)
        idx_tr = np.flatnonzero(dev.folds != k)
        assert len(idx_ho) == len(ho) and np.array_equal(dev.y[idx_ho], ho.y)
        assert k not in set(tr.folds.tolist()), "held-out fold inside the train panel"
        assert not set(tr.lesion) & set(ho.lesion), "lesion crosses fit/evaluation"
        t0 = time.time()
        fits = la.fit_candidates(tr, esc, cm, cv=True, verbose=False)
        arms = dict(fits["arms"])
        arms["A1"] = la.refit_a1(tr, esc, cm)           # never the all-fold frozen table here
        for a in ARMS:
            vals = arms[a].lam_for(ho.ages)
            assert np.isfinite(vals).all(), f"{a} produced a non-finite lambda on fold {k}"
            lam[a][idx_ho] = vals
        probs[idx_ho] = ho.probs
        hypers.append({"outer_fold": k, **fits["hyper"],
                       "a3_at_grid_boundary": fits["selection"]["A3"]["at_grid_boundary"],
                       "a6_at_grid_boundary": fits["selection"]["A6"]["at_grid_boundary"],
                       "a7_at_grid_boundary": fits["selection"]["A7"]["at_grid_boundary"],
                       "tables": {a: arms[a].knots.tolist() for a in ARMS},
                       "pooled": float(arms["A2"].pooled)})
        audit.append({**prov,
                      "hyperparameter_selection_saw_folds": sorted(int(f) for f in np.unique(tr.folds)),
                      "a1_refit_saw_folds": sorted(int(f) for f in np.unique(tr.folds)),
                      "n_fit_rows": int(len(idx_tr)), "n_scored_rows": int(len(idx_ho)),
                      "lesions_shared_with_fit": 0, "seconds": round(time.time() - t0, 1)})
        print(f"  fold {k}: {len(idx_ho)} rows scored, hyper {fits['hyper']}, "
              f"{time.time() - t0:.0f}s", flush=True)
    assert all(np.isfinite(v).all() for v in lam.values()), "a row was never scored"
    assert np.isfinite(probs).all()
    for rec in audit:
        k = rec["outer_fold"]
        for key in ("calibrator_for_heldout_saw_folds", "inner_calibrators_saw_folds",
                    "lambda_fit_rows_folds", "hyperparameter_selection_saw_folds", "a1_refit_saw_folds"):
            assert k not in rec[key], f"leak: fold {k} in {key}"
    return {"lam": lam, "probs": probs, "audit": audit, "hypers": hypers}


# ============================================================================ evaluation
def cf_panel(dev: la.Panel, probs: np.ndarray) -> la.Panel:
    return la.Panel("oof_crossfit", probs, dev.y, dev.ages, dev.lesion, dev.bands, dev.folds)


def flags_for(panel: la.Panel, lam_row: np.ndarray, esc: list[int], cm: np.ndarray) -> np.ndarray:
    return la.decision(panel.probs, panel.y, esc, cm).flagged(lam_row)


def paired_boot(lesion: np.ndarray, stat, n_boot: int, seed: int) -> dict[str, Any]:
    """Lesion-grouped paired bootstrap of a scalar; percentile CI, two-sided bootstrap p."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(pd.Series(lesion))
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(len(uniq)))
    ends = np.r_[starts[1:], len(codes)]
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[starts[g]:ends[g]] for g in pick])
        v = stat(idx)
        if np.isfinite(v):
            draws.append(v)
    draws = np.asarray(draws)
    point = stat(np.arange(len(lesion)))
    p = 2 * min(((draws <= 0).sum() + 1) / (len(draws) + 1), ((draws >= 0).sum() + 1) / (len(draws) + 1))
    return {"delta": float(point), "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "p_boot": float(min(1.0, p)), "n_boot_valid": int(len(draws))}


def sens_in(mask_band: np.ndarray, true_esc: np.ndarray, flagged: np.ndarray, idx: np.ndarray) -> float:
    m = mask_band[idx] & true_esc[idx]
    return float(flagged[idx][m].mean()) if m.any() else np.nan


def ablation(panel: la.Panel, lam: dict[str, np.ndarray], esc: list[int], cm: np.ndarray,
             n_boot: int, split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    true_esc = np.isin(panel.y, esc)
    flags = {a: flags_for(panel, lam[a], esc, cm) for a in lam}
    evals = {a: la.evaluate(panel, lam[a], esc, cm, intervals=True) for a in lam}
    rows = []
    for a in lam:
        e = evals[a]
        row = {"split": split, "arm": a, "macro_f1": e["all"]["macro_f1"],
               "balanced_accuracy": e["all"]["balanced_accuracy"]}
        for b in ("all",) + la.FLOOR_BANDS:
            for key in ("sensitivity", "specificity", "referral_rate", "missed_serious", "expected_cost",
                        "nnb_pi0.03"):
                row[f"{key}|{b}"] = e[b].get(key, np.nan)
        rows.append(row)
    steps = {}
    u40 = panel.bands == "<40"
    for i, (a, b, what) in enumerate(STEPS):
        if a not in lam or b not in lam:
            continue
        d_sens = paired_boot(panel.lesion, lambda idx, a=a, b=b: (
            sens_in(u40, true_esc, flags[b], idx) - sens_in(u40, true_esc, flags[a], idx)),
            n_boot, la.SEED + i)
        d_ref = float(flags[b][u40].mean() - flags[a][u40].mean())
        steps[f"{a}->{b}"] = {"attributes": what, "under40_sensitivity": d_sens,
                              "under40_referral_delta": d_ref,
                              "all_sensitivity_delta": evals[b]["all"]["sensitivity"] - evals[a]["all"]["sensitivity"],
                              "macro_f1_delta": evals[b]["all"]["macro_f1"] - evals[a]["all"]["macro_f1"],
                              "label": "exploratory (development)"}
    return pd.DataFrame(rows), {"evaluations": evals, "steps": steps, "flags": flags}


def dev_best(table: pd.DataFrame, cv_cost: dict[str, float]) -> str:
    """Declared rule: among A2-A7, highest cross-fitted under-40 sensitivity; ties -> lower cost."""
    cand = table[table["arm"].isin(FITTED)].copy()
    cand["cost"] = cand["arm"].map(cv_cost)
    cand = cand.sort_values(["sensitivity|<40", "cost"], ascending=[False, True])
    return str(cand.iloc[0]["arm"])


# ============================================================================ deviance
def _logit_fit(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    from scipy.optimize import minimize

    def nll(beta: np.ndarray) -> tuple[float, np.ndarray]:
        z = X @ beta
        ll = y * z - np.logaddexp(0, z)
        p = 1 / (1 + np.exp(-z))
        return -float(ll.sum()), -(X.T @ (y - p))

    res = minimize(nll, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 5000, "gtol": 1e-8})
    return -float(res.fun), res.x


def deviance_test(panel: la.Panel, esc: list[int]) -> dict[str, Any]:
    """LR test: logit P(esc) ~ score + band steps  vs  + cubic B-spline(age). Known-age rows."""
    from scipy.stats import chi2

    known = ~np.isnan(panel.ages)
    p = panel.probs[known]
    y = np.isin(panel.y[known], esc).astype(float)
    mass = np.clip(p[:, esc].sum(axis=1), 1e-6, 1 - 1e-6)
    s = np.log(mass / (1 - mass))
    bands = panel.bands[known]
    steps = np.column_stack([np.ones(len(s))] + [(bands == b).astype(float) for b in la.FLOOR_BANDS[1:]])
    X0 = np.column_stack([s, steps])
    spline = la.basis("spline", panel.ages[known])
    X1 = np.column_stack([X0, spline])
    rank0, rank1 = np.linalg.matrix_rank(X0), np.linalg.matrix_rank(X1)
    ll0, _ = _logit_fit(X0, y)
    ll1, _ = _logit_fit(X1, y)
    stat = max(0.0, 2 * (ll1 - ll0))
    df = int(rank1 - rank0)
    out = {"model_steps": "logit P(esc) ~ logit(escalation mass) + 3 band steps",
           "model_spline": "+ cubic B-spline(age), interior knots " + str(la.SPLINE_INTERIOR_KNOTS),
           "n": int(known.sum()), "loglik_steps": ll0, "loglik_spline": ll1,
           "lr_statistic": stat, "df": df, "p_value": float(chi2.sf(stat, df)),
           "reading": "tests whether age carries structure beyond three steps; says nothing about "
                      "whether that structure clears a clinical MCID"}
    u = bands == "<40"   # the under-40 slice alone: does age inside the band matter?
    Xu0 = np.column_stack([s[u], np.ones(u.sum())])
    Xu1 = np.column_stack([Xu0, panel.ages[known][u]])
    llu0, _ = _logit_fit(Xu0, y[u])
    llu1, _ = _logit_fit(Xu1, y[u])
    su = max(0.0, 2 * (llu1 - llu0))
    out["under40_linear_age"] = {"n": int(u.sum()), "n_escalating": int(y[u].sum()),
                                 "lr_statistic": su, "df": 1, "p_value": float(chi2.sf(su, 1))}
    return out


# ============================================================================ run
def run(n_boot: int) -> int:
    from research import testguard
    from research.agerule import lambda_rule as lr

    testguard.block_test_reads("S57b cross-fit: OOF only")
    esc, cm = la.esc_indices(), la.cost_matrix()
    dev = la.dev_panel()
    t0 = time.time()
    print(f"[phase 4] cross-fitting {len(ARMS)} arms over {len(np.unique(dev.folds))} outer folds", flush=True)
    cf = crossfit(dev, esc, cm)
    panel = cf_panel(dev, cf["probs"])
    table, abl = ablation(panel, cf["lam"], esc, cm, n_boot, "oof_crossfit")
    scorer = la.Scorer(panel, esc, cm)
    cv_cost = {a: scorer.cost(cf["lam"][a]) for a in ARMS}
    table["crossfit_cost"] = table["arm"].map(cv_cost)
    best = dev_best(table, cv_cost)
    family = sorted({PRIMARY_CANDIDATE, best})
    dev_dev = deviance_test(dev, esc)
    S57B_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(ABLATION_DEV_CSV, index=False, lineterminator="\n")

    rows = pd.DataFrame({"row": np.arange(len(dev)), "fold": dev.folds, "age": dev.ages,
                         "band": dev.bands, "lesion": dev.lesion, "y": dev.y,
                         **{f"lam_{a}": cf["lam"][a] for a in ARMS},
                         **{f"flag_{a}": abl["flags"][a].astype(int) for a in ARMS}})
    rows.to_csv(ROWS_CSV, index=False, lineterminator="\n")

    hyper_frame = pd.DataFrame(cf["hypers"])
    stability_of_hyper = {k: sorted({str(v) for v in hyper_frame[k]}) for k in ("bandwidth", "kappa", "tau")}
    audit = {
        "session": SESSION, "phase": 4, "split": "HAM OOF (6,981 rows, S3 folds)", "test_read": False,
        "reserved_read": False,
        "dependency_graph": {
            "Dirichlet_k_inner": "folds != k, cross-fitted among themselves; lambda_k is fit on its outputs",
            "hyperparameters_k": "S57a's nested CV run inside folds != k",
            "lambda_k": "fit on folds != k; A1 refit by lambda_rule.fit_lambda_by_group",
            "Dirichlet_k": "fit on folds != k; scores fold k",
            "fold_k": "scored once, by lambda_k over Dirichlet_k",
        },
        "assertions": [
            "no lesion in both fit and scored rows of any outer fold",
            "the scored fold absent from: held-out calibrator, inner calibrators, lambda fit rows, "
            "hyperparameter selection, A1 refit",
            "every row scored exactly once, every lambda finite",
        ],
        "folds": cf["audit"],
        "per_fold_hyperparameters": cf["hypers"],
        "hyperparameter_values_seen": stability_of_hyper,
        "full_data_hyperparameters_s57a": json.loads(la.CANDIDATES_JSON.read_text(encoding="utf-8"))["hyperparameters"],
        "crossfit_cost": cv_cost,
        "dev_best_rule": "among A2-A7, highest cross-fitted under-40 sensitivity; ties -> lower cross-fitted cost",
        "dev_best": best,
        "confirmatory_family": family,
        "ablation_steps_dev": abl["steps"],
        "deviance_diagnostic": dev_dev,
        "lambda_rule_sha256": la.sha256(Path(lr.__file__)),
        "seconds": round(time.time() - t0, 1),
    }
    la._dump(AUDIT_JSON, la._clean(audit))

    print(f"\n cross-fitted OOF (dev) -- cost, <40 sens / referral, spec per band")
    for _, r in table.iterrows():
        print(f"  {r['arm']}  cost {r['crossfit_cost']:.4f}  <40 {r['sensitivity|<40']:.3f} / "
              f"{r['referral_rate|<40']:.3f}  all {r['sensitivity|all']:.3f}  MF1 {r['macro_f1']:.4f}  spec "
              + " ".join(f"{r[f'specificity|{b}']:.3f}" for b in la.FLOOR_BANDS))
    for k, v in abl["steps"].items():
        d = v["under40_sensitivity"]
        print(f"  {k:8s} <40 sens {d['delta']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]  "
              f"<40 referral {v['under40_referral_delta']:+.3f}  ({v['attributes']})")
    print(f"  dev-best {best}; confirmatory family {family}")
    print(f"  deviance: LR {dev_dev['lr_statistic']:.2f} on {dev_dev['df']} df, p {dev_dev['p_value']:.3g}; "
          f"<40 linear age p {dev_dev['under40_linear_age']['p_value']:.3g}")
    print(f"  hyperparameters across folds: {stability_of_hyper}")

    ledger = []
    for _, r in table.iterrows():
        ledger.append({"method": f"S57b_crossfit_{r['arm']}", "split": "oof_crossfit",
                       "macro_f1": r["macro_f1"], "balanced_accuracy": r["balanced_accuracy"],
                       "escalation_sens": r["sensitivity|all"], "missed_serious": r["missed_serious|all"],
                       "notes": (f"S57b cross-fitted lambda (nested, hyper re-selected per fold): cost "
                                 f"{r['crossfit_cost']:.4f}; <40 sens {r['sensitivity|<40']:.4f} referral "
                                 f"{r['referral_rate|<40']:.4f}; min band spec "
                                 f"{min(r[f'specificity|{b}'] for b in la.FLOOR_BANDS):.4f}")})
    ledger.append({"method": "S57b_deviance_lr", "split": "oof",
                   "notes": f"steps vs spline(age): LR {dev_dev['lr_statistic']:.3f} df {dev_dev['df']} "
                            f"p {dev_dev['p_value']:.4g}"})
    write_ledger(ledger, "S57b_crossfit")
    return 0


def write_ledger(rows: list[dict[str, Any]], prune_prefix: str) -> None:
    """Prune this session's rows with any of the prefixes about to be written, then append."""
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    frame = pd.DataFrame([{"timestamp": stamp, "session": SESSION, **r} for r in rows])
    prefixes = tuple({prune_prefix} | {str(r["method"]).split("_")[0] + "_" + str(r["method"]).split("_")[1]
                                        for r in rows})
    old = pd.read_csv(la.LEDGER_PATH, low_memory=False)
    drop = (old["session"] == SESSION) & old["method"].astype(str).str.startswith(prefixes)
    kept = old[~drop]
    print(f"ledger: pruned {int(drop.sum())} prior {SESSION} row(s) {prefixes}, appended {len(frame)}")
    pd.concat([kept, frame], ignore_index=True).reindex(columns=old.columns).to_csv(la.LEDGER_PATH, index=False)


# ============================================================================ selftest
def selftest() -> int:
    fails = 0

    def check(name: str, ok: bool) -> None:
        nonlocal fails
        fails += not ok
        print(("PASS " if ok else "FAIL ") + name)

    rng = np.random.default_rng(1)
    # paired bootstrap: identical arms -> delta 0, p 1
    les = rng.integers(0, 50, 400).astype(str)
    r = paired_boot(les, lambda idx: 0.0, 50, 0)
    check("paired bootstrap of a null difference is 0 with p=1", r["delta"] == 0 and r["p_boot"] == 1.0)
    r = paired_boot(les, lambda idx: 1.0, 50, 0)
    check("paired bootstrap of a constant positive difference excludes 0", r["ci"][0] > 0 and r["p_boot"] < 0.05)
    # deviance test on synthetic data: age matters inside a band -> spline wins
    n = 6000
    ages = rng.choice(la.AGE_KNOTS, n)
    from research.external import frozen_params as fp
    bands = fp.age_bands(ages)
    true_logit = -2 + 3 * np.sin(ages / 12.0)
    y = rng.random(n) < 1 / (1 + np.exp(-true_logit))
    probs = np.full((n, 7), 0.01)
    esc = la.esc_indices()
    probs[:, esc[0]] = 0.5
    probs /= probs.sum(axis=1, keepdims=True)
    labels = np.where(y, esc[0], 5)
    p = la.Panel("syn", probs, labels, ages, np.arange(n).astype(str), bands)
    d = deviance_test(p, esc)
    check("deviance test detects within-band age structure", d["p_value"] < 1e-6)
    true_logit = -2 + 1.5 * (bands == "60+")
    y = rng.random(n) < 1 / (1 + np.exp(-true_logit))
    p = la.Panel("syn", probs, np.where(y, esc[0], 5), ages, np.arange(n).astype(str), bands)
    d = deviance_test(p, esc)
    check("deviance test is quiet when age is exactly three steps", d["p_value"] > 0.01)
    # dev-best rule
    t = pd.DataFrame({"arm": ["A1", "A2", "A3", "A6"], "sensitivity|<40": [0.9, 0.7, 0.8, 0.8]})
    check("dev-best ignores A1 and breaks ties on cost",
          dev_best(t, {"A1": 0, "A2": 0, "A3": 0.5, "A6": 0.4}) == "A6")
    # the cross-fit machinery on a real fold: no lesion crosses, fold excluded everywhere
    dev = la.dev_panel()
    folds = la.nested_cv_panels(dev)
    ok = all(not set(tr.lesion) & set(ho.lesion) and prov["outer_fold"] not in set(tr.folds.tolist())
             for tr, ho, prov in folds)
    check("no lesion crosses fit/evaluation; scored fold absent from fit rows", ok)
    check("every OOF row scored exactly once", sum(len(ho) for _, ho, _ in folds) == len(dev))
    esc, cm = la.esc_indices(), la.cost_matrix()
    a1 = la.refit_a1(folds[0][0], esc, cm)
    check("refit A1 differs from nothing it should not: table finite, pooled finite",
          np.isfinite(a1.knots).all() and np.isfinite(a1.pooled))
    print(f"{8 - fails}/8 passed")
    return int(fails > 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEV)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.run:
        return run(args.n_boot)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
