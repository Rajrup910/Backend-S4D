"""S66 -- lambda per hospital: per-centre escalation bias for the frozen V1 ensemble.

S14's descriptive sweep put the target-fitted under-40 lambda at 0.24 (HAM), 0.12 (MSKCC) and
0.91 (BCN), a spread no single lambda covers. This session fits a per-centre lambda without
touching reserved, then reads reserved once.

**Why the V1 arm needs no GPU.** V1 never trained on BCN or MSKCC, and S13 froze its
probabilities for every image (`results/external/predictions/ensemble_dirichlet_{cohort}.csv`,
deployed HAM-OOF Dirichlet map). The V4 **train** rows of those centres are therefore clean
fitting data for V1. HAM's fitting data are the S5 cross-fitted OOF probabilities, exactly as S5
and S57a used them.

Arms, each a lambda per (centre, age band), same rule as S5:
``argmax_c (p_c + lambda * 1[c escalates])``

    P0  frozen HAM lambda everywhere (`frozen_params`, never typed)          -- comparator
    PC  one lambda per band pooled over the three centres' fitting rows     -- attribution reference
    P1  independent per-centre lambda; below S5's 30-positive gate -> PC
    P2  hierarchical (primary): alpha * local + (1 - alpha) * PC, alpha = P / (P + tau)
    P2r P2 applied with the centre S58's router *predicts* (reserved only)  -- the dependency

The objective is S5's, loaded: expected clinical cost s.t. escalation specificity >= 0.85
(`lambda_rule.fit_lambda`). P1 and P2 are floor-repaired per (centre, band) with S57a's
`repair_floor`, and PC on the pooled rows. A missing age takes the arm's per-centre "all rows"
lambda, shrunk the same way.

Declared deviations (written into the plan):
  D1  tau is chosen by lesion-grouped 5-fold CV **within** each centre, not by leave-one-centre-out.
      With a centre held out it has no local estimate, alpha is undefined, and every tau collapses
      to PC -- the criterion cannot distinguish taus. CV cost is the centre-equal-weighted mean of
      held-out expected cost (HAM would otherwise dominate at 6,981 of ~15k rows); 1-SE rule toward
      more shrinkage (S57a's).
  D2  HAM's CV folds are S57a's nested panels (inner cross-fit calibration), so a HAM held-out
      fold's labels touch neither its calibrator nor its lambda. BCN/MSKCC need no nesting: their
      calibrator was fit on HAM.
  D3  P counts escalating *images* (S5's gate unit and S57a's A7). BCN has ~3.3 images per lesion,
      so P overstates BCN's information; tau is CV-chosen and absorbs the scale.
  D4  `scc` rows are absent from S13's V1 files (pre-registered exclusion), so they are not in the
      fitting data (BCN train loses 226 images, 0 under-40 escalating). Reserved includes S54's
      146 top-up rows, as S56/S57b did.
  D5  V4 `val` (BCN/MSKCC) has **zero** under-40 escalating images, so it cannot check the
      primary; it is scored as a held-out descriptive check on older-band specificity/referral.
  D6  Measured on V1 ensemble probabilities only. S58's pooled head was trained on these same
      train rows, so a per-centre lambda over it would need cross-fitted head outputs; not built.
  D7  The dev ablation is cross-fitted at tau*, and tau* was chosen on the same folds -- mildly
      optimistic for P2, declared.

Commands:

    $py -m research.v4.s66_lambda_centre --selftest
    $py -m research.v4.s66_lambda_centre --fit                 # CV, tables, dev ablation (no reserved)
    $py -m research.v4.s66_lambda_centre --freeze-plan         # immutable plan, hashed inputs
    $py -m research.v4.s66_lambda_centre --reserved --smoke    # rehearsal on BCN/MSKCC dev rows
    $py -m research.v4.s66_lambda_centre --reserved            # the one reserved read (receipt)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.v4 import lambda_age as la

REPO_ROOT = la.REPO_ROOT
OUT_DIR = REPO_ROOT / "results" / "v4" / "s66"
TABLES_JSON = OUT_DIR / "lambda_by_centre.json"
DEV_CSV = OUT_DIR / "ablation_dev.csv"
PLAN = REPO_ROOT / "results" / "v4" / "s66_plan.json"
RECEIPT = OUT_DIR / "reserved_receipt.json"
REPORT = OUT_DIR / "s66_report.json"
RESERVED_CSV = OUT_DIR / "reserved_arms.csv"
LEDGER_PATH = la.LEDGER_PATH
SESSION = "v4_s66"

CENTRES = ("ham", "bcn20000", "mskcc")
PRIMARY_CENTRE = "bcn20000"
BANDS = la.FLOOR_BANDS
GROUPS = BANDS + ("all",)          # "all" = the centre's missing-age lambda
FITTED_ARMS = ("PC", "P1", "P2")
ARMS = ("A0", "P0") + FITTED_ARMS
TAUS = la.SHRINK_TAUS
N_FOLDS = 5
FOLD_SEED = 20260917
MCID = 0.10                        # S48, same unit as S56/S57b
ALPHA = 0.05
NI_MARGIN = 0.02                   # S57b
SPEC_NI_MARGIN = 0.02              # S57b
REFERRAL_BLOWOUT = 1.5             # S57b
N_BOOT = 2000
BOOT_SEED = 42
VERDICT_ORDER = ("ADOPT", "PROMISING", "REJECT-cost", "REJECT-router", "REJECT-flat")


# ============================================================================ data
def _v1_frame(cohort: str) -> pd.DataFrame:
    from research.v4 import s54_gate as g

    codes = g.class_codes()
    frame = pd.read_csv(g.V1_FROZEN_DIR / f"ensemble_dirichlet_{cohort}.csv",
                        usecols=["image_id", "true_index"] + [f"p_{c}" for c in codes])
    return frame.assign(image_id=frame["image_id"].astype(str))


def external_panel(centre: str, splits: tuple[str, ...], name: str) -> tuple[la.Panel, pd.DataFrame]:
    """Non-reserved manifest_v4 rows of one external centre, V1 probabilities, lesion folds."""
    from research.external import frozen_params as fp
    from research.v4 import s54_gate as g
    from sklearn.model_selection import StratifiedGroupKFold

    assert "reserved" not in splits and "ham_test" not in splits
    man = pd.read_csv(g.MANIFEST, low_memory=False)
    rows = man[(man["archive"] == centre) & man["split"].isin(splits)].copy()
    rows["image_id"] = rows["image_id"].astype(str)
    v1 = _v1_frame(centre)
    rows = rows.merge(v1, on="image_id", how="inner").sort_values("image_id").reset_index(drop=True)
    dropped_non_scc = int((man[(man["archive"] == centre) & man["split"].isin(splits)]
                           .pipe(lambda d: (~d["image_id"].astype(str).isin(v1["image_id"]))
                                 & (d["class_8"] != "scc"))).sum())
    assert dropped_non_scc == 0, f"{centre}: {dropped_non_scc} non-scc rows lack V1 probabilities"
    if not np.array_equal(rows["true_index"].to_numpy(), rows["class_index_7"].to_numpy()):
        raise ValueError(f"{centre}: V1 labels disagree with manifest_v4")
    ages = rows["age_approx"].to_numpy(dtype=float)
    bands = rows["age_band"].astype(str).to_numpy()
    assert np.array_equal(fp.age_bands(ages).astype(str), bands), f"{centre}: age_band disagrees"
    probs = rows[[f"p_{c}" for c in g.class_codes()]].to_numpy(float)
    y = rows["class_index_7"].to_numpy(int)
    groups = rows["group_id"].astype(str).to_numpy()
    strata = ((bands == "<40") & np.isin(y, fp.escalating_indices())).astype(int)
    folds = np.full(len(rows), -1)
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    for k, (_, idx) in enumerate(skf.split(probs, strata, groups)):
        folds[idx] = k
    assert (folds >= 0).all()
    return la.Panel(name, probs, y, ages, groups, bands, folds), rows


def dev_panels() -> dict[str, la.Panel]:
    out = {"ham": la.dev_panel()}
    for c in CENTRES[1:]:
        out[c] = external_panel(c, ("train",), f"{c}_train")[0]
    return out


def concat(panels: list[la.Panel], name: str) -> la.Panel:
    return la.Panel(name, np.vstack([p.probs for p in panels]), np.concatenate([p.y for p in panels]),
                    np.concatenate([p.ages for p in panels]), np.concatenate([p.lesion for p in panels]),
                    np.concatenate([p.bands for p in panels]))


def positives(panel: la.Panel, group: str, esc: list[int]) -> int:
    m = np.ones(len(panel), bool) if group == "all" else panel.bands == group
    return int(np.isin(panel.y[m], esc).sum())


# ============================================================================ fitting
def fit_group(panel: la.Panel, group: str, esc: list[int], cm: np.ndarray) -> float:
    from research.agerule import lambda_rule as lr

    m = np.ones(len(panel), bool) if group == "all" else panel.bands == group
    if not np.isin(panel.y[m], esc).any():
        return float("nan")
    return float(lr.fit_lambda(panel.probs[m], panel.y[m], cm, esc, group=group,
                               min_specificity=la.MIN_SPECIFICITY, strict=False).lam)


@dataclass
class Fit:
    """Everything tau-independent: pooled and local lambdas and their positive counts."""
    pooled: dict[str, float]
    local: dict[str, dict[str, float]]
    n_pos: dict[str, dict[str, int]]
    pooled_panel: la.Panel


def fit_components(panels: dict[str, la.Panel], esc: list[int], cm: np.ndarray) -> Fit:
    union = concat([panels[c] for c in CENTRES], "pooled")
    pooled = {g: fit_group(union, g, esc, cm) for g in GROUPS}
    local = {c: {g: fit_group(panels[c], g, esc, cm) for g in GROUPS} for c in CENTRES}
    n_pos = {c: {g: positives(panels[c], g, esc) for g in GROUPS} for c in CENTRES}
    return Fit(pooled, local, n_pos, union)


def as_arm(name: str, values: dict[str, float], meta: dict | None = None) -> la.Arm:
    return la.band_arm(name, {b: values[b] for b in BANDS}, values["all"], meta or {})


def arm_p0() -> la.Arm:
    return la.arm_a1()


def build_tables(fit: Fit, panels: dict[str, la.Panel], tau: float,
                 scorers: dict[str, la.Scorer] | None = None) -> dict[str, dict[str, la.Arm]]:
    """arm -> centre -> Arm. P1/P2 floor-repaired per centre; PC on the pooled rows."""
    from research.agerule import lambda_rule as lr

    scorers = scorers or {}
    sc = lambda key, p: scorers.get(key) or la.Scorer(p, la.esc_indices(), la.cost_matrix())  # noqa: E731
    pc = la.repair_floor(as_arm("PC", fit.pooled), sc("pooled", fit.pooled_panel))
    out: dict[str, dict[str, la.Arm]] = {"A0": {}, "P0": {}, "PC": {}, "P1": {}, "P2": {}}
    for c in CENTRES:
        p1, p2, alphas = {}, {}, {}
        for g in GROUPS:
            loc, n = fit.local[c][g], fit.n_pos[c][g]
            p1[g] = loc if (n >= lr.MIN_GROUP_POSITIVES and np.isfinite(loc)) else fit.pooled[g]
            a = 0.0 if (np.isinf(tau) or n == 0 or not np.isfinite(loc)) else (n / (n + tau) if n + tau > 0 else 1.0)
            alphas[g] = a
            p2[g] = a * (loc if np.isfinite(loc) else 0.0) + (1.0 - a) * fit.pooled[g]
        out["A0"][c] = la.arm_a0()
        out["P0"][c] = arm_p0()
        out["PC"][c] = pc
        out["P1"][c] = la.repair_floor(as_arm("P1", p1, {"gate": lr.MIN_GROUP_POSITIVES}), sc(c, panels[c]))
        out["P2"][c] = la.repair_floor(as_arm("P2", p2, {"tau": tau, "alpha": alphas}), sc(c, panels[c]))
    return out


def band_values(arm: la.Arm) -> dict[str, float]:
    from research.external import frozen_params as fp

    kb = fp.age_bands(la.AGE_KNOTS)
    return {**{b: float(arm.knots[kb == b][0]) for b in BANDS}, "all": float(arm.pooled)}


# ============================================================================ CV
def cv_splits(panels: dict[str, la.Panel]) -> list[tuple[dict[str, la.Panel], dict[str, la.Panel]]]:
    ham = {prov["outer_fold"]: (tr, ho) for tr, ho, prov in la.nested_cv_panels(panels["ham"])}
    out = []
    for k in range(N_FOLDS):
        train, held = {"ham": ham[k][0]}, {"ham": ham[k][1]}
        for c in CENTRES[1:]:
            p = panels[c]
            tr, ho = np.flatnonzero(p.folds != k), np.flatnonzero(p.folds == k)
            assert not set(p.lesion[tr]) & set(p.lesion[ho]), f"{c}: lesion crosses folds"
            train[c], held[c] = p.subset(tr, name=f"{c}_cv{k}_train"), p.subset(ho, name=f"{c}_cv{k}_heldout")
        out.append((train, held))
    return out


def run_cv(panels: dict[str, la.Panel], esc: list[int], cm: np.ndarray,
           verbose: bool = True) -> dict[str, Any]:
    say = print if verbose else (lambda *a, **k: None)
    splits = cv_splits(panels)
    fits = [fit_components(tr, esc, cm) for tr, _ in splits]
    tr_scorers = [{**{c: la.Scorer(tr[c], esc, cm) for c in CENTRES},
                   "pooled": la.Scorer(f.pooled_panel, esc, cm)} for (tr, _), f in zip(splits, fits)]
    ho_scorers = [{c: la.Scorer(ho[c], esc, cm) for c in CENTRES} for _, ho in splits]

    def score(tables_per_fold: list[dict[str, dict[str, la.Arm]]], arm: str) -> dict[str, Any]:
        per_fold, per_centre = [], {c: [] for c in CENTRES}
        for (_, ho), tables, hs in zip(splits, tables_per_fold, ho_scorers):
            costs = {c: hs[c].cost(tables[arm][c].lam_for(ho[c].ages)) for c in CENTRES}
            for c in CENTRES:
                per_centre[c].append(costs[c])
            per_fold.append(float(np.mean(list(costs.values()))))
        arr = np.array(per_fold)
        return {"cv_cost": float(arr.mean()), "cv_se": float(arr.std(ddof=1) / np.sqrt(len(arr))),
                "per_fold": per_fold, "per_centre": {c: float(np.mean(v)) for c, v in per_centre.items()}}

    grid, tables_by_tau = [], {}
    for tau in TAUS:
        tables_by_tau[tau] = [build_tables(f, tr, tau, s) for (tr, _), f, s in zip(splits, fits, tr_scorers)]
        r = score(tables_by_tau[tau], "P2")
        grid.append({"tau": tau, **r})
        say(f"  P2 tau={tau:<6g} cv cost {r['cv_cost']:.5f} +- {r['cv_se']:.5f}  "
            + " ".join(f"{c} {v:.4f}" for c, v in r["per_centre"].items()))
    pick = la.one_se_pick(grid, "tau", smoother_is_larger=True)
    tau_star = pick["selected"]
    arm_cv = {a: score(tables_by_tau[tau_star], a) for a in ARMS}
    for a, r in arm_cv.items():
        say(f"  CV {a:3s} {r['cv_cost']:.5f} +- {r['cv_se']:.5f}")

    # cross-fitted flags at tau* for the dev ablation (declared D7)
    flags: dict[str, dict[str, np.ndarray]] = {a: {} for a in ARMS}
    for c in CENTRES:
        for a in ARMS:
            flags[a][c] = np.zeros(len(panels[c]), bool)
        for k, ((_, ho), tables, hs) in enumerate(zip(splits, tables_by_tau[tau_star], ho_scorers)):
            idx = np.flatnonzero(panels[c].folds == k)
            assert np.array_equal(panels[c].y[idx], ho[c].y), f"{c} fold {k}: held-out rows misaligned"
            for a in ARMS:
                flags[a][c][idx] = hs[c].dec.flagged(tables[a][c].lam_for(ho[c].ages))
    return {"grid": grid, "selection": pick, "tau": tau_star, "arm_cv": arm_cv, "crossfit_flags": flags}


def dev_ablation(panels: dict[str, la.Panel], cv: dict[str, Any], esc: list[int]) -> pd.DataFrame:
    rows = []
    for c in CENTRES:
        p = panels[c]
        te = np.isin(p.y, esc)
        for a in ARMS:
            f = cv["crossfit_flags"][a][c]
            row = {"centre": c, "arm": a, "tau": cv["tau"], "cv_cost_centre": cv["arm_cv"][a]["per_centre"][c]}
            for b in ("all",) + BANDS:
                m = np.ones(len(p), bool) if b == "all" else p.bands == b
                pos, neg = te & m, ~te & m
                row[f"n_esc|{b}"] = int(pos.sum())
                row[f"sensitivity|{b}"] = float(f[pos].mean()) if pos.any() else np.nan
                row[f"specificity|{b}"] = float(1 - f[neg].mean()) if neg.any() else np.nan
                row[f"referral_rate|{b}"] = float(f[m].mean()) if m.any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================ evaluation helpers
def lam_rows(tables: dict[str, dict[str, la.Arm]], arm: str, ages: np.ndarray, centre: np.ndarray) -> np.ndarray:
    out = np.full(len(ages), np.nan)
    for c in CENTRES:
        m = centre == c
        if m.any():
            out[m] = tables[arm][c].lam_for(ages[m])
    assert np.isfinite(out).all(), "a row has no centre lambda"
    return out


def tables_to_json(tables: dict[str, dict[str, la.Arm]]) -> dict[str, Any]:
    return {a: {c: {"by_band": band_values(arm), **arm.as_dict()} for c, arm in per.items()}
            for a, per in tables.items()}


def tables_from_json(payload: dict[str, Any]) -> dict[str, dict[str, la.Arm]]:
    out: dict[str, dict[str, la.Arm]] = {}
    for a, per in payload.items():
        out[a] = {c: la.Arm(a, d["kind"], np.array([d["table"][str(int(k))] for k in la.AGE_KNOTS]),
                            float(d["missing_age_lambda"])) for c, d in per.items()}
    return out


def load_tables() -> dict[str, dict[str, la.Arm]]:
    return tables_from_json(json.loads(TABLES_JSON.read_text(encoding="utf-8"))["tables"])


def evaluate_centre(panel: la.Panel, lam: np.ndarray, esc: list[int], cm: np.ndarray) -> dict[str, Any]:
    return la.evaluate(panel, lam, esc, cm, intervals=True)


# ============================================================================ --fit
def run_fit(verbose: bool = True) -> int:
    from research import testguard
    from research.external import frozen_params as fp

    testguard.block_test_reads("S66 fit: HAM OOF + BCN/MSKCC V4 train rows only")
    esc, cm = la.esc_indices(), la.cost_matrix()
    panels = dev_panels()
    for c, p in panels.items():
        u = (p.bands == "<40") & np.isin(p.y, esc)
        print(f"[dev] {c:9s} rows {len(p):5d}  <40 escalating {int(u.sum()):3d} images / "
              f"{len(set(p.lesion[u])):3d} lesions")
    print("[cv] tau grid (P2), centre-equal-weighted held-out cost")
    cv = run_cv(panels, esc, cm, verbose)
    fit = fit_components(panels, esc, cm)
    tables = build_tables(fit, panels, cv["tau"])

    # S5 reproduction: HAM's local band lambdas are S5's frozen ones
    frozen = fp.load_lambda_by_band()
    repro = {g: {"local": fit.local["ham"][g], "frozen": frozen[g if g != "all" else "pooled"]} for g in GROUPS}
    assert all(abs(v["local"] - v["frozen"]) < 1e-12 for v in repro.values()), f"S5 not reproduced: {repro}"

    abl = dev_ablation(panels, cv, esc)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    abl.to_csv(DEV_CSV, index=False, lineterminator="\n")

    # held-out descriptive: V4 val (BCN/MSKCC; no <40 positives) and HAM val
    val = {c: external_panel(c, ("val",), f"{c}_v4val")[0] for c in CENTRES[1:]}
    val["ham"] = la.val_panel()
    val_eval = {c: {a: evaluate_centre(val[c], tables[a][c].lam_for(val[c].ages), esc, cm) for a in ARMS}
                for c in CENTRES}
    dev_eval = {c: {a: evaluate_centre(panels[c], tables[a][c].lam_for(panels[c].ages), esc, cm) for a in ARMS}
                for c in CENTRES}

    payload = {
        "session": SESSION,
        "question": "does a per-centre lambda (hierarchically shrunk) beat the frozen HAM lambda off HAM?",
        "fitting_data": {c: {"rows": len(panels[c]), "positives": fit.n_pos[c]} for c in CENTRES},
        "rule": "argmax_c (p_c + lambda[centre, band] * 1[c escalates])",
        "objective": "expected clinical cost (build_cost_matrix) s.t. escalation specificity >= 0.85 per band",
        "tau": cv["tau"], "tau_selection": {**cv["selection"], "grid": cv["grid"]},
        "cv_cost_by_arm": cv["arm_cv"],
        "components": {"pooled": fit.pooled, "local": fit.local, "n_positive_images": fit.n_pos,
                       "s5_reproduction_ham": repro},
        "tables": tables_to_json(tables),
        "dev_in_sample": dev_eval,
        "val_heldout_descriptive": val_eval,
        "router_dependency": "P1/P2 need the centre at inference; S58's router (3-way reserved accuracy "
                             "0.932) supplies it. P2r measures the cost on reserved.",
        "deviations": {k: v for k, v in DEVIATIONS.items()},
    }
    la._dump(TABLES_JSON, la._clean(payload))
    print(f"\n[tables] tau* = {cv['tau']:g}  ({cv['selection']['rule']}; min at {cv['selection']['min_cv_value']:g})")
    print("  arm  centre     <40     40-59   60+     all(missing age)")
    for a in FITTED_ARMS + ("P0",):
        for c in CENTRES:
            v = band_values(tables[a][c])
            print(f"  {a:3s}  {c:9s} " + "  ".join(f"{v[g]:.2f}  " for g in GROUPS))
    print("\n[dev cross-fitted]  centre     arm  <40 sens  <40 ref   min spec")
    for _, r in abl.iterrows():
        spec = min(r[f"specificity|{b}"] for b in BANDS)
        print(f"                    {r['centre']:9s}  {r['arm']:3s}  {r['sensitivity|<40']:.3f}     "
              f"{r['referral_rate|<40']:.3f}     {spec:.3f}")
    rows = []
    for c in CENTRES:
        for a in ARMS:
            e = dev_eval[c][a]
            rows.append({"method": f"S66_dev_{a}_{c}", "split": "oof" if c == "ham" else "v4_train",
                         "macro_f1": e["all"]["macro_f1"], "balanced_accuracy": e["all"]["balanced_accuracy"],
                         "escalation_sens": e["all"]["sensitivity"], "missed_serious": e["all"]["missed_serious"],
                         "notes": f"S66 {a} {c} in-sample (tau*={cv['tau']:g}); <40 sens "
                                  f"{e['<40']['sensitivity']:.4f} referral {e['<40']['referral_rate']:.4f}; "
                                  f"cross-fitted <40 sens "
                                  f"{abl.set_index(['centre', 'arm']).loc[(c, a), 'sensitivity|<40']:.4f}"})
    write_ledger(rows, "S66_dev")
    return 0


DEVIATIONS = {
    "D1": "tau by lesion-grouped 5-fold CV within centres (centre-equal-weighted held-out cost, 1-SE toward "
          "more shrinkage), not leave-one-centre-out: a held-out centre has no local estimate, so every tau "
          "collapses to PC there.",
    "D2": "HAM folds use S57a's nested panels (inner cross-fit calibration); BCN/MSKCC need none (HAM-fit map).",
    "D3": "alpha counts escalating images (S5/A7 unit); BCN has ~3.3 images per lesion; tau absorbs the scale.",
    "D4": "scc rows lack S13 V1 probabilities and are not fitting data; reserved includes S54's 146 top-ups.",
    "D5": "V4 val has zero under-40 escalating images: descriptive older-band check only.",
    "D6": "V1 ensemble probabilities only; S58's pooled head would need cross-fitted outputs (not built).",
    "D7": "dev ablation cross-fitted at tau*, which was chosen on the same folds (mildly optimistic for P2).",
}


# ============================================================================ plan
def inputs() -> dict[str, Path]:
    from research.agerule import lambda_rule as lr
    from research.external import frozen_params as fp
    from research.v4 import s54_gate as g
    from research.v4 import s58_front_end as s58

    out = {"lambda_by_centre": TABLES_JSON, "ablation_dev": DEV_CSV, "manifest_v4": g.MANIFEST,
           "frozen_band_lambda": REPO_ROOT / fp.LAMBDA_STATE, "deployed_dirichlet": REPO_ROOT / fp.DIRICHLET_STATE,
           "lambda_rule_py": Path(lr.__file__), "s58_router_state": s58.STATE_PATH,
           "s58_reserved_features": s58.FEATURES_RESERVED}
    for c in g.V1_COHORTS:
        out[f"v1_{c}"] = g.V1_FROZEN_DIR / f"ensemble_dirichlet_{c}.csv"
    for p in sorted(g.topup_dir(False).glob("ensemble_dirichlet_*.csv")):
        out[f"v1_topup_{p.stem}"] = p
    return out


def hashes() -> dict[str, dict[str, str]]:
    return {k: {"path": la.rel(p), "sha256": la.sha256(p)} for k, p in inputs().items()}


def build_plan() -> dict[str, Any]:
    from research.v4.s54_guard import git_head, now

    fitted = json.loads(TABLES_JSON.read_text(encoding="utf-8"))
    return {
        "session": SESSION, "frozen_at": now(), "git_head": git_head(),
        "question": fitted["question"],
        "evaluation_surface": {"primary": "manifest_v4 split=reserved, BCN rows (79 under-40 escalating "
                                          "lesions), deployed V1 ensemble, read once",
                               "descriptive": "reserved MSKCC rows (24 under-40 escalating lesions)",
                               "ham": "HAM has no reserved rows; HAM tables are dev-only",
                               "ham_test": "not read"},
        "arms": {"A0": "argmax", "P0": "frozen HAM 3-band lambda (comparator)",
                 "PC": "pooled over centres per band (attribution reference)",
                 "P1": "independent per centre, 30-positive gate -> PC",
                 "P2": f"hierarchical, alpha = P/(P+tau), tau* = {fitted['tau']} (primary)",
                 "P2r": "P2 with S58's router-predicted centre (router 'ham' -> HAM table)"},
        "tables": {a: {c: fitted["tables"][a][c]["by_band"] for c in CENTRES} for a in ARMS},
        "primary_endpoint": {
            "quantity": "BCN reserved under-40 escalation sensitivity (image-level), P2 - P0",
            "interval": f"lesion-grouped (group_id) paired percentile bootstrap, {N_BOOT} resamples, seed "
                        f"{BOOT_SEED}, resampling lesions within BCN under-40",
            "p_value": "two-sided bootstrap p; family = {P2 vs P0 on BCN}, a single member, no Holm",
            "always_quoted_with": "BCN under-40 referral P2 vs P0 and its paired delta"},
        "secondary_descriptive": ["P1 - P0, PC - P0, P2 - PC, P2r - P0 on BCN under-40 sensitivity",
                                  "per centre x band: sensitivity, specificity, referral, NNB (pi 0.01/0.03/0.05)",
                                  "Macro-F1 and balanced accuracy per centre",
                                  "MSKCC: every quantity above, descriptive only (24 lesions)"],
        "gates": {
            "1_primary": f"delta >= {MCID} AND 95% CI lower bound > 0 AND p < {ALPHA}; "
                         f"directional if delta >= {MCID} with CI including 0",
            "2_specificity": f"BCN reserved, per band: P2 specificity >= P0's - {SPEC_NI_MARGIN}; the 0.85 floor "
                             "is reported, not relaxed (P0 already misses it at 40-59 on reserved, S57a)",
            "3_noninferiority": f"BCN Macro-F1 and balanced accuracy: lower 95% bound of (P2 - P0) > -{NI_MARGIN}; "
                                "BCN 40-59 and 60+ P2 sensitivity >= P0's Clopper-Pearson lower bound",
            "4_referral": f"BLOWOUT if BCN under-40 referral P2 / P0 > {REFERRAL_BLOWOUT}",
            "5_router": "BCN under-40 sensitivity P2r - P0: 95% CI lower bound > 0 (the gain survives routing)",
        },
        "verdict_logic": [
            "ADOPT: gates 1-5 pass",
            "REJECT-cost: gate 1 passes and gate 4 BLOWOUT",
            "PROMISING: gate 1 directional, gates 2, 3, 5 pass, no BLOWOUT (P0 stays deployed)",
            "REJECT-router: gates 1-4 pass, gate 5 fails",
            "REJECT-flat: otherwise",
        ],
        "on_adopt": "freeze tables into research/agerule/results_oof/lambda_by_centre.json and add "
                    "frozen_params.load_lambda_by_centre() as the only loader (S65 may then take P2)",
        "caveat_declared": "a BCN gain compensates for V1's poor BCN transfer (Macro-F1 0.402); it does not "
                           "fix ranking -- no lambda beats a centre's own ROC curve (S64)",
        "stated_before_the_read": {
            "tau_star_is_inf": "the tau grid is flat within 1 SE, so the declared rule takes full shrinkage: "
                               "P2 = PC except where the per-centre floor repair lowers a band. The data do not "
                               "support a per-centre deviation beyond what the floor forces.",
            "bcn_under40_is_floor_bound": "BCN's local under-40 lambda (0.66) sits where the 0.85 specificity "
                                          "floor binds, not at the cost optimum; S14's 0.91 had no floor.",
            "p0_cheapest_by_breaking_the_floor": "P0 has the lowest CV cost only because it is not floor-held: "
                                                 "its cross-fitted BCN 40-59 specificity is ~0.75. Cost across "
                                                 "arms with and without the floor is not like-for-like.",
            "gate4_expected_to_fire": "cross-fitted BCN under-40 referral P0 ~0.062 vs P2 ~0.199 (ratio ~3.2). "
                                      "The 1.5x rule is S57b's, kept unchanged; P0's base referral is below "
                                      "BCN's under-40 prevalence (0.119), so the ratio is harsh here. Extra "
                                      "referrals per extra caught case is reported beside it, descriptive.",
            "p2_moves_ham_too": "with tau* = inf HAM's under-40 lambda becomes the pooled 0.66 (frozen 0.26); "
                                "cross-fitted HAM under-40 referral 0.076 -> 0.153. HAM has no reserved rows, "
                                "so this is a dev-only finding S65 must weigh.",
            "dev_crossfit": pd.read_csv(DEV_CSV)[["centre", "arm", "sensitivity|<40", "referral_rate|<40"]]
                            .to_dict(orient="records"),
        },
        "deviations": fitted["deviations"],
        "inputs": hashes(),
    }


def freeze_plan() -> int:
    if PLAN.is_file():
        raise SystemExit(f"{la.rel(PLAN)} exists; the plan is immutable once written")
    if not TABLES_JSON.is_file():
        raise SystemExit("run --fit first")
    la._dump(PLAN, la._clean(build_plan()))
    print(f"plan frozen: {la.rel(PLAN)} sha256 {la.sha256(PLAN)}")
    return 0


# ============================================================================ the read
@dataclass
class EvalPanel:
    panel: la.Panel
    centre: np.ndarray       # true archive
    routed: np.ndarray       # S58 router prediction
    meta: dict[str, Any]


def reserved_eval() -> EvalPanel:
    from research.v4 import lambda_freeze as lf
    from research.v4 import s54_gate as g
    from research.v4 import s58_front_end as s58

    panel, meta = lf.reserved_eval_panel()
    frame = g.build_panel("reserved")
    assert np.array_equal(frame["y7"].to_numpy(), panel.y) and np.array_equal(frame["group_id"].astype(str).to_numpy(), panel.lesion)
    state = s58.load_state()
    feats, ids = s58._load_npz(s58.FEATURES_RESERVED)
    routed = s58.route(state["router"], s58.align(feats, ids, frame["image_id"]))
    return EvalPanel(panel, frame["archive"].astype(str).to_numpy(), routed, meta)


def smoke_eval() -> EvalPanel:
    """Mechanical rehearsal on BCN/MSKCC V4 train+val rows (in-sample for lambda -- numbers mean nothing)."""
    from research.v4 import s58_front_end as s58

    parts, frames = [], []
    for c in CENTRES[1:]:
        p, f = external_panel(c, ("train", "val"), f"{c}_smoke")
        parts.append(p)
        frames.append(f)
    panel = concat(parts, "smoke_dev_rows")
    frame = pd.concat(frames, ignore_index=True)
    state = s58.load_state()
    feats, ids = s58._load_npz(s58.FEATURES_FIT)
    routed = s58.route(state["router"], s58.align(feats, ids, frame["image_id"]))
    return EvalPanel(panel, frame["archive"].astype(str).to_numpy(), routed,
                     {"smoke": "BCN/MSKCC train+val rows stand in for reserved; in-sample, not a result"})


def subset(panel: la.Panel, m: np.ndarray, name: str) -> la.Panel:
    return panel.subset(np.flatnonzero(m), name=name)


def evaluate_read(ev: EvalPanel, tables: dict[str, dict[str, la.Arm]]) -> dict[str, Any]:
    from research.v4 import lambda_crossfit as lc
    from sklearn.metrics import balanced_accuracy_score, f1_score

    esc, cm = la.esc_indices(), la.cost_matrix()
    p = ev.panel
    lam = {a: lam_rows(tables, a, p.ages, ev.centre) for a in ARMS}
    routed_centre = np.where(np.isin(ev.routed, CENTRES), ev.routed, "ham")
    lam["P2r"] = lam_rows(tables, "P2", p.ages, routed_centre)
    arms = ARMS + ("P2r",)
    dec = la.decision(p.probs, p.y, esc, cm)
    flags = {a: dec.flagged(lam[a]) for a in arms}
    preds = {a: dec.preds(lam[a]) for a in arms}
    te = np.isin(p.y, esc)
    labels = list(range(p.probs.shape[1]))

    per_centre, table_rows = {}, []
    for c in CENTRES[1:]:
        mc = ev.centre == c
        if not mc.any():
            continue
        sub = subset(p, mc, c)
        per_centre[c] = {a: evaluate_centre(sub, lam[a][mc], esc, cm) for a in arms}
        for a in arms:
            e = per_centre[c][a]
            assert np.array_equal(np.isin(la.decision(sub.probs, sub.y, esc, cm).preds(lam[a][mc]), esc), flags[a][mc])
            row = {"centre": c, "arm": a, "macro_f1": e["all"]["macro_f1"],
                   "balanced_accuracy": e["all"]["balanced_accuracy"]}
            for b in ("all",) + BANDS:
                for k in ("n_escalating", "sensitivity", "missed_serious", "specificity", "referral_rate",
                          "nnb_pi0.03"):
                    row[f"{k}|{b}"] = e[b].get(k, np.nan)
            table_rows.append(row)

    def contrasts(c: str, pairs: list[tuple[str, str]], seed0: int) -> dict[str, Any]:
        mc = ev.centre == c
        u40 = np.flatnonzero(mc & (p.bands == "<40"))
        allc = np.flatnonzero(mc)
        out = {}
        for i, (a, b) in enumerate(pairs):
            if not te[u40].any():
                out[f"{a}-{b}"] = {"under40_sensitivity": None, "note": "no under-40 escalating rows"}
                continue
            fa, fb, t = flags[a][u40], flags[b][u40], te[u40]
            d_sens = lc.paired_boot(p.lesion[u40], lambda idx: (fa[idx][t[idx]].mean() - fb[idx][t[idx]].mean())
                                    if t[idx].any() else np.nan, N_BOOT, BOOT_SEED + seed0 + i)
            d_ref = lc.paired_boot(p.lesion[u40], lambda idx: fa[idx].mean() - fb[idx].mean(),
                                   N_BOOT, BOOT_SEED + seed0 + 50 + i)
            ya, pa, pb = p.y[allc], preds[a][allc], preds[b][allc]
            d_mf1 = lc.paired_boot(p.lesion[allc], lambda idx: (
                f1_score(ya[idx], pa[idx], labels=labels, average="macro", zero_division=0)
                - f1_score(ya[idx], pb[idx], labels=labels, average="macro", zero_division=0)),
                N_BOOT, BOOT_SEED + seed0 + 100 + i)
            d_ba = lc.paired_boot(p.lesion[allc], lambda idx: (
                balanced_accuracy_score(ya[idx], pa[idx]) - balanced_accuracy_score(ya[idx], pb[idx])),
                N_BOOT, BOOT_SEED + seed0 + 150 + i)
            extra_caught = int((fa & t).sum() - (fb & t).sum())
            extra_referred = int(fa.sum() - fb.sum())
            out[f"{a}-{b}"] = {"under40_sensitivity": d_sens, "under40_referral": d_ref,
                               "macro_f1": d_mf1, "balanced_accuracy": d_ba,
                               "under40_extra_caught_images": extra_caught,
                               "under40_extra_referred_images": extra_referred,
                               "under40_extra_referrals_per_extra_catch":
                                   extra_referred / extra_caught if extra_caught > 0 else None}
        return out

    pairs = [("P2", "P0"), ("P1", "P0"), ("PC", "P0"), ("P2", "PC"), ("P2r", "P0"), ("P0", "A0")]
    contrast = {c: contrasts(c, pairs, 1000 * j) for j, c in enumerate(CENTRES[1:]) if (ev.centre == c).any()}

    # ---- gates (BCN, P2 vs P0)
    c = PRIMARY_CENTRE
    k = contrast[c]["P2-P0"]
    s = k["under40_sensitivity"]
    e2, e0 = per_centre[c]["P2"], per_centre[c]["P0"]
    if s is None:
        return {"per_centre": per_centre, "contrasts": contrast, "table": pd.DataFrame(table_rows),
                "verdict": "NOT-EVALUABLE", "gates": {}, "routing": routing_summary(ev)}
    g1 = s["delta"] >= MCID and s["ci"][0] > 0 and s["p_boot"] < ALPHA
    g1_dir = s["delta"] >= MCID and not g1
    spec = {b: {"P2": e2[b]["specificity"], "P0": e0[b]["specificity"],
                "pass": e2[b]["specificity"] >= e0[b]["specificity"] - SPEC_NI_MARGIN,
                "floor_met": e2[b]["specificity"] >= la.MIN_SPECIFICITY} for b in BANDS}
    g2 = all(v["pass"] for v in spec.values())
    older = {}
    for b in ("40-59", "60+"):
        lo = e0[b]["sensitivity_cp"][0]
        older[b] = {"P2": e2[b]["sensitivity"], "P0": e0[b]["sensitivity"], "P0_cp_lo": lo,
                    "pass": e2[b]["sensitivity"] >= lo}
    g3 = (k["macro_f1"]["ci"][0] > -NI_MARGIN and k["balanced_accuracy"]["ci"][0] > -NI_MARGIN
          and all(v["pass"] for v in older.values()))
    ratio = e2["<40"]["referral_rate"] / e0["<40"]["referral_rate"] if e0["<40"]["referral_rate"] > 0 else np.inf
    blowout = bool(ratio > REFERRAL_BLOWOUT)
    r = contrast[c]["P2r-P0"]["under40_sensitivity"]
    g5 = bool(r["ci"][0] > 0)
    if g1 and g2 and g3 and not blowout and g5:
        verdict = "ADOPT"
    elif g1 and blowout:
        verdict = "REJECT-cost"
    elif g1_dir and g2 and g3 and g5 and not blowout:
        verdict = "PROMISING"
    elif g1 and g2 and g3 and not blowout and not g5:
        verdict = "REJECT-router"
    else:
        verdict = "REJECT-flat"
    gates = {"1_primary": {**s, "pass": bool(g1), "directional": bool(g1_dir), "mcid": MCID},
             "2_specificity": {"per_band": spec, "pass": bool(g2)},
             "3_noninferiority": {"macro_f1": k["macro_f1"], "balanced_accuracy": k["balanced_accuracy"],
                                  "older_bands": older, "pass": bool(g3)},
             "4_referral": {"P2": e2["<40"]["referral_rate"], "P0": e0["<40"]["referral_rate"],
                            "delta": k["under40_referral"], "ratio": float(ratio), "blowout": blowout},
             "5_router": {**r, "pass": g5}}
    return {"per_centre": per_centre, "contrasts": contrast, "table": pd.DataFrame(table_rows),
            "verdict": verdict, "gates": gates, "routing": routing_summary(ev)}


def routing_summary(ev: EvalPanel) -> dict[str, Any]:
    out = {}
    for c in CENTRES[1:]:
        m = ev.centre == c
        if m.any():
            out[c] = {"n": int(m.sum()), "routed_to": pd.Series(ev.routed[m]).value_counts().to_dict(),
                      "accuracy": float((ev.routed[m] == c).mean())}
    return out


def run_reserved(smoke: bool, rerun_reason: str | None) -> int:
    from research import testguard
    from research.v4 import lambda_freeze as lf
    from research.v4.s54_guard import git_head, now

    testguard.block_test_reads("S66 evaluation: reserved (or BCN/MSKCC dev rows in a smoke), never HAM test")
    if not PLAN.is_file():
        raise SystemExit("freeze the plan first (--freeze-plan)")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    current = hashes()
    drift = sorted(k for k in set(plan["inputs"]) | set(current)
                   if plan["inputs"].get(k, {}).get("sha256") != current.get(k, {}).get("sha256"))
    if drift:
        raise SystemExit(f"inputs changed since the plan was frozen: {drift}")
    plan_sha = la.sha256(PLAN)
    out_dir = OUT_DIR / "smoke" if smoke else OUT_DIR
    receipt = None
    if not smoke:
        receipt = (json.loads(RECEIPT.read_text(encoding="utf-8")) if RECEIPT.is_file()
                   else {"cohort": "manifest_v4 split=reserved (S66 per-centre lambda)", "plan": la.rel(PLAN),
                         "plan_sha256": plan_sha, "executions": []})
        if receipt["plan_sha256"] != plan_sha:
            raise SystemExit("the plan changed after the receipt was opened")
        if receipt["executions"] and not rerun_reason:
            raise SystemExit("S66 has already read reserved; a repeat needs --rerun-reason")
        receipt["executions"].append({"execution": len(receipt["executions"]) + 1, "status": "started",
                                      "started_at": now(), "rerun_reason": rerun_reason, "git_head": git_head()})
        la._dump(RECEIPT, receipt)

    tables = load_tables()
    for a in ARMS:   # the loaded tables must be the ones the plan declared
        for c in CENTRES:
            assert all(abs(band_values(tables[a][c])[g] - plan["tables"][a][c][g]) < 1e-12 for g in GROUPS)
    with lf.no_fitting():
        ev = smoke_eval() if smoke else reserved_eval()
        res = evaluate_read(ev, tables)

    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / RESERVED_CSV.name
    report_path = out_dir / REPORT.name
    res["table"].to_csv(table_path, index=False, lineterminator="\n")
    consequence = {
        "ADOPT": "freeze per-centre tables behind frozen_params.load_lambda_by_centre(); S65 may take P2",
        "PROMISING": "exploratory only; P0 stays deployed; S65 uses the frozen 3-band rule",
        "REJECT-cost": "gain achievable but not clinically efficient; P0 stays deployed; S65 uses the frozen rule",
        "REJECT-router": "gain does not survive routing; P0 stays deployed",
        "REJECT-flat": "per-centre lambda does not beat the frozen rule; P0 stays deployed",
        "NOT-EVALUABLE": "smoke panel lacks the primary rows",
    }[res["verdict"]]
    report = {"session": SESSION, "smoke": smoke, "panel": ev.panel.name, "plan": la.rel(PLAN),
              "plan_sha256": plan_sha, "verdict": res["verdict"], "consequence": consequence,
              "gates": res["gates"], "contrasts": res["contrasts"], "per_centre": res["per_centre"],
              "routing": res["routing"], "panel_meta": ev.meta, "not_read": "HAM test",
              "caveat": plan["caveat_declared"]}
    la._dump(report_path, la._clean(report))
    print_report(res, ev.panel.name)
    if receipt is not None:
        receipt["executions"][-1].update(status="completed", completed_at=now(), verdict=res["verdict"],
                                         items={la.rel(p): la.sha256(p) for p in (report_path, table_path)})
        la._dump(RECEIPT, receipt)
        ledger_reserved(res)
    return 0


def print_report(res: dict[str, Any], name: str) -> None:
    t = res["table"].set_index(["centre", "arm"])
    print(f"\n[{name}] centre     arm  <40 sens  <40 ref   all sens  MF1     spec <40/40-59/60+")
    for (c, a), r in t.iterrows():
        print(f"  {c:9s}  {a:3s}  {r['sensitivity|<40']:.3f}     {r['referral_rate|<40']:.3f}     "
              f"{r['sensitivity|all']:.3f}     {r['macro_f1']:.4f}  "
              + "/".join(f"{r[f'specificity|{b}']:.3f}" for b in BANDS))
    for c, per in res["contrasts"].items():
        for pair, k in per.items():
            s = k.get("under40_sensitivity")
            if s:
                print(f"  {c:9s} {pair:7s} <40 sens {s['delta']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] "
                      f"p {s['p_boot']:.4f}   <40 ref {k['under40_referral']['delta']:+.3f}   "
                      f"MF1 {k['macro_f1']['delta']:+.4f} [{k['macro_f1']['ci'][0]:+.4f}, {k['macro_f1']['ci'][1]:+.4f}]")
    g = res["gates"]
    if g:
        print(f"  gates: G1 {g['1_primary']['pass']} (dir {g['1_primary']['directional']})  G2 "
              f"{g['2_specificity']['pass']}  G3 {g['3_noninferiority']['pass']}  G4 ratio "
              f"{g['4_referral']['ratio']:.2f} blowout {g['4_referral']['blowout']}  G5 {g['5_router']['pass']}")
    print(f"  routing: {res['routing']}")
    print(f"  VERDICT: {res['verdict']}")


# ============================================================================ ledger
def write_ledger(rows: list[dict[str, Any]], prune_prefix: str) -> None:
    """Prune this session's own rows with this prefix, then append (the S20 hazard)."""
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    frame = pd.DataFrame([{"timestamp": stamp, "session": SESSION, **r} for r in rows])
    old = pd.read_csv(LEDGER_PATH, low_memory=False)
    kept = old[~((old["session"] == SESSION) & old["method"].astype(str).str.startswith(prune_prefix))]
    print(f"ledger: pruned {len(old) - len(kept)} prior {SESSION} row(s), appended {len(frame)}")
    pd.concat([kept, frame], ignore_index=True).to_csv(LEDGER_PATH, index=False)


def ledger_reserved(res: dict[str, Any]) -> None:
    rows = []
    for c, per in res["per_centre"].items():
        for a, e in per.items():
            rows.append({"method": f"S66_reserved_{a}_{c}", "split": "reserved",
                         "macro_f1": e["all"]["macro_f1"], "balanced_accuracy": e["all"]["balanced_accuracy"],
                         "escalation_sens": e["all"]["sensitivity"], "missed_serious": e["all"]["missed_serious"],
                         "notes": f"S66 reserved {c} {a}: <40 sens {e['<40']['sensitivity']:.4f} referral "
                                  f"{e['<40']['referral_rate']:.4f}; min band spec "
                                  f"{min(e[b]['specificity'] for b in BANDS):.4f}"})
    g = res["gates"]["1_primary"]
    rows.append({"method": "S66_reserved_verdict", "split": "reserved", "p_value_vs_baseline": g["p_boot"],
                 "notes": f"S66 BCN <40 sens P2-P0 {g['delta']:+.4f} [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}] "
                          f"p {g['p_boot']:.4f}; referral ratio {res['gates']['4_referral']['ratio']:.2f}; "
                          f"verdict {res['verdict']}"})
    write_ledger(rows, "S66_reserved")


# ============================================================================ selftest
def selftest() -> int:
    from research import testguard
    from research.agerule import lambda_rule as lr
    from research.external import frozen_params as fp
    from research.v4 import lambda_freeze as lf

    fails: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    testguard.block_test_reads("S66 selftest")
    esc, cm = la.esc_indices(), la.cost_matrix()

    # 1. P0 is the frozen rule, row for row
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(7) * 0.5, size=3000)
    ages = rng.choice(np.r_[la.AGE_KNOTS, np.nan], size=3000)
    tables = {"P0": {c: arm_p0() for c in CENTRES}}
    centre = rng.choice(CENTRES, size=3000)
    lam = lam_rows(tables, "P0", ages, centre)
    dec = la.decision(p, rng.integers(0, 7, 3000), esc, cm)
    check("P0 == frozen_params.apply_age_rule", np.array_equal(dec.preds(lam), fp.apply_age_rule(p, ages)))

    # 2. shrinkage limits on a synthetic two-centre world
    syn = {c: la.synthetic(lambda a, s=s: np.full(len(np.atleast_1d(a)), s), 600, seed=i)
           for i, (c, s) in enumerate(zip(CENTRES, (0.10, 0.60, 0.30)))}
    for c, pn in syn.items():
        pn.name = c
    fit = fit_components(syn, esc, cm)
    t0 = build_tables(fit, syn, 0.0)
    tinf = build_tables(fit, syn, float("inf"))
    def limit_ok(tabs: dict, target: Any) -> bool:
        for c in CENTRES:
            arm = tabs["P2"][c]
            delta = arm.meta["floor_repair_delta"]
            for gname in GROUPS:
                want, got = target(c, gname), band_values(arm)[gname]
                repaired = gname in delta and delta[gname] > 0
                if (abs(got - want) > 1e-9) if not repaired else (got > want + 1e-9):
                    return False
        return True

    check("tau=0 -> local (floor repair may only lower it)", limit_ok(t0, lambda c, gn: fit.local[c][gn]))
    check("tau=inf -> pooled", limit_ok(tinf, lambda c, gn: fit.pooled[gn]))
    spread = [fit.local[c]["all"] for c in CENTRES]
    check("synthetic per-centre optima are recovered in order", spread[0] < spread[2] < spread[1],
          f"local 'all' {np.round(spread, 2)} vs truth 0.10 / 0.60 / 0.30")

    # 3. repair keeps each centre inside its per-band floor
    fitted_real = TABLES_JSON.is_file()
    panels = dev_panels()
    for c in CENTRES:
        u = (panels[c].bands == "<40") & np.isin(panels[c].y, esc)
        print(f"     {c}: rows {len(panels[c])}, <40 escalating {int(u.sum())} / {len(set(panels[c].lesion[u]))} lesions")
    check("BCN/MSKCC fitting counts match the runbook table (117/35, 12/12)",
          [(int(((panels[c].bands == "<40") & np.isin(panels[c].y, esc)).sum()),
            len(set(panels[c].lesion[(panels[c].bands == "<40") & np.isin(panels[c].y, esc)]))) for c in CENTRES[1:]]
          == [(117, 35), (12, 12)])
    split_sets = [set(panels[c].lesion[panels[c].folds == k]) for c in CENTRES[1:] for k in range(N_FOLDS)]
    check("external folds are lesion-disjoint", all(
        not (split_sets[i] & split_sets[j]) for i in range(len(split_sets)) for j in range(i + 1, len(split_sets))
        if (i // N_FOLDS) == (j // N_FOLDS)))
    from research.v4 import s54_gate as g
    man = pd.read_csv(g.MANIFEST, low_memory=False)
    reserved_groups = set(man.loc[man["split"] == "reserved", "group_id"].astype(str))
    check("no fitting row shares a group with reserved",
          not any(set(panels[c].lesion) & reserved_groups for c in CENTRES[1:]))
    if fitted_real:
        real = load_tables()
        worst = min(la.Scorer(panels[c], esc, cm).band_spec(real[a][c].lam_for(panels[c].ages)).min()
                    for a in ("P1", "P2") for c in CENTRES)
        check("fitted P1/P2 meet the 0.85 floor in every centre x band (in-sample)", worst >= la.MIN_SPECIFICITY - 1e-12,
              f"worst {worst:.4f}")
        payload = json.loads(TABLES_JSON.read_text(encoding="utf-8"))
        rep = payload["components"]["s5_reproduction_ham"]
        check("HAM local band lambdas reproduce S5", all(abs(v["local"] - v["frozen"]) < 1e-12 for v in rep.values()))
        check("tables round-trip through JSON", all(
            np.array_equal(real[a][c].knots, tables_from_json(payload["tables"])[a][c].knots)
            for a in ARMS for c in CENTRES))

    # 4. the read cannot fit, and cannot touch HAM test
    with lf.no_fitting():
        try:
            lr.fit_lambda(None, None, None, None)
            ok = False
        except RuntimeError:
            ok = True
    check("fitting disabled inside the read", ok)
    try:
        testguard.check_split("test", "selftest")
        ok = False
    except testguard.TestSplitLocked:
        ok = True
    check("testguard armed", ok)
    try:
        external_panel("bcn20000", ("reserved",), "x")
        ok = False
    except AssertionError:
        ok = True
    check("external_panel refuses reserved", ok)
    if PLAN.is_file():
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        check("plan input hashes reproduce", {k: v["sha256"] for k, v in plan["inputs"].items()}
              == {k: v["sha256"] for k, v in hashes().items()})
    print("SELFTEST PASSED" if not fails else f"SELFTEST FAILED: {fails}")
    return int(bool(fails))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--freeze-plan", action="store_true")
    ap.add_argument("--reserved", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rerun-reason", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.fit:
        return run_fit()
    if args.freeze_plan:
        return freeze_plan()
    if args.reserved:
        return run_reserved(args.smoke, args.rerun_reason)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
