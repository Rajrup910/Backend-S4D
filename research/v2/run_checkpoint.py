"""S34 -- the registered pass: frontiers (V2) + decomposition (V3), with intervals.

Runs the pre-registered analysis from `results/v2/analysis_plan.json` and writes the
artifacts every later session reads. Nothing here chooses what to test: F1's three cohorts,
F2's four members, the two-sidedness of each, the MCID and the bootstrap settings were all
frozen in S30 before any of these numbers existed.

Two bootstrap budgets, stated rather than hidden:
  * confirmatory (F1, F2)  -- `n_boot_confirmatory` from the frozen plan (2000).
  * exploratory (frontier curves, AUC) -- `--n-boot-exploratory` (default 500), because
    these carry no significance claim and a full-cohort lesion bootstrap at 2000 across
    every cohort x band x score cell costs hours for intervals nobody is allowed to test.

    $py -m research.v2.run_checkpoint --smoke     # tiny n_boot, validates the pipeline
    $py -m research.v2.run_checkpoint             # the registered pass
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.paths import load_class_mapping
from research.ablation.bootstrap import lesion_resample_indices
from research.external.frozen_params import escalating_indices
from research.stats.intervals import proportion
from research.v2 import decompose as dec
from research.v2 import estimators as est
from research.v2 import frontier as fr
from research.v2 import members as mem
from research.v2 import multiplicity as mult

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = REPO_ROOT / "results" / "v2" / "panels"
OUT_DIR = REPO_ROOT / "results" / "v2"
PLAN_PATH = OUT_DIR / "analysis_plan.json"

CLASS_CODES = tuple(load_class_mapping().codes)
ESC_CODES = frozenset({"akiec", "bcc", "mel"})
BANDS = ("ALL", "<40", "40-59", "60+")
COHORTS = ("ham_oof", "bcn20000", "mskcc", "pad")
SEED = 42


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _slice(cohort: str, band: str) -> pd.DataFrame:
    panel = pd.read_csv(PANEL_DIR / f"{cohort}.csv")
    if band != "ALL":
        panel = panel[panel["age_band"] == band]
    return panel.reset_index(drop=True)


def _arrays(panel: pd.DataFrame):
    probs = panel[[f"p_{c}" for c in CLASS_CODES]].to_numpy(dtype=float)
    y_esc = panel["true_code"].isin(ESC_CODES).to_numpy()
    lesion_ids = panel["effective_lesion_id"].to_numpy()
    return probs, y_esc, lesion_ids


# ------------------------------------------------------------------ F1: the signed C
def paired_bootstrap_C(
    probs: np.ndarray, y_esc: np.ndarray, lesion_ids: np.ndarray, esc: list[int],
    *, n_boot: int, seed: int = SEED,
) -> dict:
    """Paired lesion bootstrap of `C = S_mass - S_argmax`, two-sided.

    Argmax is not a score-ranked policy, so its budget is recomputed inside every resample
    as argmax's own referral count in that resample -- not carried over from the full
    sample. That keeps the comparison at argmax's natural burden in every draw, which is
    what the frozen budget rule specifies.
    """
    y_esc = np.asarray(y_esc, dtype=bool)
    s = est.escalation_mass(probs, esc)
    refers_am = est.argmax_refers(probs, esc)
    r = int(refers_am.sum())
    point = (est.sensitivity(est.top_r_refers(s, r, seed), y_esc)
             - est.sensitivity(refers_am, y_esc))

    draws = []
    for idx in lesion_resample_indices(lesion_ids, n_boot=n_boot, seed=seed):
        sub_y = y_esc[idx]
        if sub_y.sum() == 0:
            continue
        sub_am = refers_am[idx]
        sub_r = int(sub_am.sum())
        d = est.sensitivity(est.top_r_refers(s[idx], sub_r, seed), sub_y) - est.sensitivity(sub_am, sub_y)
        if np.isfinite(d):
            draws.append(d)

    draws = np.asarray(draws)
    if len(draws) == 0:
        return {"C": point, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_two_sided": float("nan"), "n_boot_used": 0}
    p_lo = (1 + int(np.sum(draws <= 0))) / (1 + len(draws))
    p_hi = (1 + int(np.sum(draws >= 0))) / (1 + len(draws))
    return {
        "C": point,
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "p_two_sided": float(min(1.0, 2 * min(p_lo, p_hi))),
        "n_boot_used": int(len(draws)),
    }


# ----------------------------------------------------------------------------- driver
def run(n_boot_conf: int, n_boot_expl: int, seed: int = SEED) -> int:
    plan = _plan()
    esc = escalating_indices()
    mcid = float(plan["mcid_sensitivity"])
    started = datetime.now(timezone.utc).isoformat()

    decomposition_rows: list[dict] = []
    frontier_rows: list[dict] = []
    taxonomy_rows: list[dict] = []
    auc_rows: list[dict] = []

    print("=== V3: decomposition + band taxonomy + AUC, all cohorts x bands ===")
    for cohort in COHORTS:
        panel_full = pd.read_csv(PANEL_DIR / f"{cohort}.csv")
        members_full = mem.load_member_probs(cohort, panel_full["image_id"].to_numpy())
        for band in BANDS:
            mask = np.ones(len(panel_full), bool) if band == "ALL" else (panel_full["age_band"] == band).to_numpy()
            if mask.sum() < 30:
                continue
            panel = panel_full[mask].reset_index(drop=True)
            probs, y_esc, lesion_ids = _arrays(panel)
            if y_esc.sum() == 0:
                continue

            res = dec.decompose_panel(panel, cohort=cohort, subgroup=band,
                                      member_probs=members_full[mask], seed=seed)
            row = res.as_row()
            row["certification"] = dec.certification_verdict(res.B_certified, mcid)
            decomposition_rows.append(row)

            # band taxonomy of argmax's misses
            tax = dec.classify_misses(probs, y_esc, esc, r=res.r, seed=seed,
                                      image_ids=panel["image_id"].to_numpy())
            taxonomy_rows.append({
                "cohort": cohort, "band": band,
                "n_missed_by_argmax": len(tax),
                "compression_compatible": int(tax["compression_compatible"].sum()) if len(tax) else 0,
                "q_recoverable": int(tax["q_recoverable"].sum()) if len(tax) else 0,
                "provably_determined_benign": int(tax["provably_determined_benign"].sum()) if len(tax) else 0,
                "median_recovery_budget": float(tax["recovery_budget"].median()) if len(tax) else float("nan"),
            })

            # AUC + partial AUC on escalation mass
            s = est.escalation_mass(probs, esc)
            auc = fr.partial_auc(y_esc, s, fpr_max=0.20)
            auc_rows.append({"cohort": cohort, "band": band, "n": len(panel),
                             "n_escalating": int(y_esc.sum()), **auc})

            # frontier point estimates for every score; CIs only for s and d (exploratory)
            library = est.build_score_library(probs, esc, members_full[mask])
            for score_name, score in library.items():
                want_ci = score_name in ("s", "d") and band != "ALL"
                if want_ci:
                    pts = fr.frontier(score, y_esc, lesion_ids, seed=seed, n_boot=n_boot_expl)
                    for p in pts:
                        frontier_rows.append({"cohort": cohort, "band": band, "score": score_name,
                                              **p.as_dict()})
                else:
                    n = len(score)
                    for q in fr.DEFAULT_Q_GRID:
                        r_q = int(round(q * n))
                        frontier_rows.append({
                            "cohort": cohort, "band": band, "score": score_name, "q": q,
                            "r": r_q, "burden": r_q / n if n else float("nan"),
                            "sensitivity": est.sensitivity(est.top_r_refers(score, r_q, seed), y_esc),
                            "ci_lo": float("nan"), "ci_hi": float("nan"), "n_boot": 0,
                        })
            print(f"  {cohort:10s} {band:6s} n={len(panel):6d} esc={int(y_esc.sum()):5d} "
                  f"r={res.r:5d} C={res.C:+.4f} B_in={res.B_insample:.4f}")

    pd.DataFrame(decomposition_rows).to_csv(OUT_DIR / "decomposition.csv", index=False)
    pd.DataFrame(frontier_rows).to_csv(OUT_DIR / "frontiers.csv", index=False)
    pd.DataFrame(taxonomy_rows).to_csv(OUT_DIR / "miss_taxonomy.csv", index=False)
    pd.DataFrame(auc_rows).to_csv(OUT_DIR / "auc_table.csv", index=False)

    # ---------------------------------------------------------------- F1 (confirmatory)
    print(f"\n=== F1 (confirmatory, two-sided, n_boot={n_boot_conf}) ===")
    f1_cohorts = mult.by_name("F1_compression_gap").cohorts
    f1_rows = []
    for cohort in f1_cohorts:
        panel = _slice(cohort, "<40")
        probs, y_esc, lesion_ids = _arrays(panel)
        res = paired_bootstrap_C(probs, y_esc, lesion_ids, esc, n_boot=n_boot_conf, seed=seed)
        res.update({"family": "F1_compression_gap", "cohort": cohort, "band": "<40",
                    "n": len(panel), "n_escalating": int(y_esc.sum()),
                    "n_lesions": int(len(np.unique(lesion_ids)))})
        f1_rows.append(res)
        print(f"  {cohort:10s} C={res['C']:+.4f} [{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}] "
              f"p={res['p_two_sided']:.4f}")
    f1_adj = mult.adjust("F1_compression_gap", [r["p_two_sided"] for r in f1_rows])
    for r, p_holm, sig in zip(f1_rows, f1_adj["p_holm"], f1_adj["significant_holm"]):
        r["p_holm"] = p_holm
        r["significant_holm"] = sig
        r["exceeds_mcid"] = bool(abs(r["C"]) >= mcid)

    # ---------------------------------------------------------------- F2 (confirmatory)
    print(f"\n=== F2 (confirmatory, one-sided, HAM-OOF <40, n_boot={n_boot_conf}) ===")
    panel = _slice("ham_oof", "<40")
    panel_full = pd.read_csv(PANEL_DIR / "ham_oof.csv")
    mask = (panel_full["age_band"] == "<40").to_numpy()
    members_u40 = mem.load_member_probs("ham_oof", panel_full["image_id"].to_numpy())[mask]
    res_f2 = dec.decompose_panel(panel, cohort="ham_oof", subgroup="<40",
                                 member_probs=members_u40, run_f2=True,
                                 n_boot=n_boot_conf, seed=seed)
    f2_rows = []
    for t in res_f2.f2_tests:
        f2_rows.append({"family": "F2_ranking_certification", "cohort": "ham_oof",
                        "band": "<40", "score": t["score"], "diff_vs_s": t["point"],
                        "ci_lo": t["ci_lo"], "ci_hi": t["ci_hi"],
                        "p_one_sided": t["p_one_sided"], "p_holm": t["p_holm"],
                        "significant_holm": t["significant_holm"],
                        "n_boot_used": t["n_boot_used"]})
        print(f"  {t['score']:<14s} diff={t['point']:+.4f} [{t['ci_lo']:+.4f},{t['ci_hi']:+.4f}] "
              f"p_holm={t['p_holm']:.4f} {'SIG' if t['significant_holm'] else 'ns'}")
    print(f"  B_certified = {res_f2.B_certified:.4f} -> "
          f"{dec.certification_verdict(res_f2.B_certified, mcid)}")

    pd.DataFrame(f1_rows + f2_rows).to_csv(OUT_DIR / "primary_comparisons.csv", index=False)
    (OUT_DIR / "bootstrap_intervals.json").write_text(json.dumps({
        "session": "S34", "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "plan_logical_sha256": json.loads((OUT_DIR / "analysis_plan.sha256").read_text())["logical_sha256"],
        "n_boot_confirmatory": n_boot_conf, "n_boot_exploratory": n_boot_expl,
        "seed": seed, "mcid": mcid,
        "F1": {"adjustment": f1_adj, "rows": f1_rows},
        "F2": {"rows": f2_rows, "B_certified": res_f2.B_certified,
               "certification": dec.certification_verdict(res_f2.B_certified, mcid)},
    }, indent=2, default=str), encoding="utf-8")

    print(f"\nwrote decomposition.csv, frontiers.csv, miss_taxonomy.csv, auc_table.csv, "
          f"primary_comparisons.csv, bootstrap_intervals.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S34 -- registered pass (frontiers + decomposition)")
    parser.add_argument("--smoke", action="store_true", help="tiny n_boot; validates the pipeline only")
    parser.add_argument("--n-boot-exploratory", type=int, default=500)
    args = parser.parse_args(argv)
    plan = _plan()
    n_conf = 50 if args.smoke else int(plan["n_boot_confirmatory"])
    n_expl = 20 if args.smoke else args.n_boot_exploratory
    if args.smoke:
        print("*** SMOKE RUN -- numbers are not the registered pass ***\n")
    return run(n_conf, n_expl, seed=int(plan["seed"]))


if __name__ == "__main__":
    raise SystemExit(main())
