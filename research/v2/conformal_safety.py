"""S37 -- F3, conformal subgroup safety on HAM-OOF.

**F3, exactly as frozen** (`multiplicity.FAMILIES`): does a class-conditional (Mondrian,
7-way) conformal calibration protect the under-40 escalation event as well as the coarser
band x escalation (bipartite) calibration `research.conformal.hierarchical` was built for
-- measured by False Reassurance Rate (FRR), the fraction of truly-escalating lesions
whose prediction set contains *no* escalating class at all. Two members, matching size=2
in the frozen family: alpha=0.05 and alpha=0.10, the same pair the project's published
conformal work (session 4/6) already reports at. Both two-sided, Holm-adjusted together.

**Which Dirichlet map, and the caveat that comes with it.** The blueprint's section 11
leakage lock is explicit: the **deployed** map only
(`research/selective/results_oof/fit_state.json`), loaded through
`research.external.frozen_params` and never refitted here. An earlier draft of this module
instead refitted Dirichlet on the conformal tuning half, following the published
`run_session4_conformal --fit-split oof` recipe (its L1 fix); the owner resolved the
conflict in favour of the lock, so the refit is gone.

**The cost of that choice, stated rather than buried.** The deployed map was fitted on all
6,981 OOF rows (`fit_state.json:fit_n`), which *includes* the calibration half this module
draws its conformal quantile from. Split conformal's finite-sample guarantee assumes the
calibration scores are exchangeable with what is scored under **one fixed score function**
chosen independently of them; a calibrator that has seen the calibration points weakens
that. So coverage here is **approximate and must be audited empirically, never asserted** --
which is what `frr_by_group.csv` does by measuring achieved marginal coverage against
nominal. This compounds with the caveat the OOF conformal variant already carries (its
scores come from five fold models, not the one model that would make the guarantee a
theorem). Both are reasons to read these numbers as measured coverage, not as a certificate.

**Why HAM-OOF's calibration half rather than val.** The family is declared
`cohorts=["ham_oof"]`, and the calibration half is the only ham_oof-native slice that is
lesion-disjoint from the tuning half and still holds enough under-40 escalating cases
(roughly half of OOF's 64) to make FRR_<40 estimable at all -- val has 22, below the
project's own 30-positive gate. This departs from the published runner's val/test
evaluation split, recorded here rather than silently reproduced.

    $py -m research.v2.conformal_safety --run-f3
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.paths import load_class_mapping, resolve
from research.ablation.bootstrap import lesion_resample_indices
from research.ablation.run_part_a import holm_bonferroni
from research.conformal import calibrate, hierarchical
from research.conformal import scores as conformal_scores
from research.ensembling.data import ARCHS, load_split_matrix
from research.ensembling.methods import soft_vote_arithmetic
from research.external import frozen_params as fp
from research.external.frozen_params import escalating_indices
from research.run_session4_conformal import _tune_raps
from research.v2.panels import _ham_ages_and_bands

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "v2"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"

EPS = 1e-12
SEED = 42
N_BOOT = 2000
ALPHAS = (0.05, 0.10)
BAND_ORDER = ("<40", "40-59", "60+", "unknown")
COHORT = "ham_oof"


def _paired_frr_bootstrap(
    sets_mondrian: np.ndarray, sets_bipartite: np.ndarray,
    y_true: np.ndarray, lesion_ids: np.ndarray, esc: list[int],
    *, n_boot: int = N_BOOT, seed: int = SEED,
) -> dict:
    """Paired lesion-grouped bootstrap of FRR(bipartite) - FRR(mondrian), two-sided.

    Both sets are scored on the identical rows, and one lesion resample drives both
    quantities in every draw -- what makes the interval on the *difference* tighter than
    differencing two independently-bootstrapped FRRs would give.
    """
    is_serious = np.isin(y_true, esc)
    idx = np.flatnonzero(is_serious)
    has_m = sets_mondrian[idx][:, esc].any(axis=1)
    has_b = sets_bipartite[idx][:, esc].any(axis=1)
    reassured_m = (~has_m).astype(np.float64)
    reassured_b = (~has_b).astype(np.float64)
    lesion_serious = lesion_ids[idx]

    point = float(reassured_b.mean() - reassured_m.mean())
    draws = []
    for boot_idx in lesion_resample_indices(lesion_serious, n_boot=n_boot, seed=seed):
        draws.append(float(reassured_b[boot_idx].mean() - reassured_m[boot_idx].mean()))
    draws = np.asarray(draws)
    if len(draws) == 0:
        return {"point": point, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_two_sided": float("nan"), "n_boot_used": 0}
    p_two_sided = min(1.0, 2.0 * min(float((draws <= 0).mean()), float((draws >= 0).mean())))
    return {
        "point": point,
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "p_two_sided": p_two_sided,
        "n_boot_used": int(len(draws)),
        "n_serious": int(len(idx)),
    }


def run_alpha(alpha: float, *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    mapping = load_class_mapping()
    num_classes = mapping.num_classes
    esc = escalating_indices()

    fit = load_split_matrix("train", predictions_dir="research/predictions_oof_tta")
    fit_ens = soft_vote_arithmetic(fit.probs)
    ages, bands = _ham_ages_and_bands(fit.image_ids)

    tune_idx, cal_idx = calibrate.grouped_halves(fit.lesion_ids, seed=seed)
    # Blueprint section 11 leakage lock: the DEPLOYED Dirichlet map only
    # (research/selective/results_oof/fit_state.json), never a per-run refit.
    fit_probs = fp.calibrate(fit_ens)

    rng = np.random.default_rng(seed)
    k_reg, penalty = _tune_raps(fit_probs[tune_idx], fit.y_true[tune_idx], num_classes, alpha, rng)

    score_matrix = conformal_scores.aps_scores(
        fit_probs, np.random.default_rng(seed), penalty=penalty, k_reg=k_reg
    )
    cal_scores = conformal_scores.true_label_scores(score_matrix[cal_idx], fit.y_true[cal_idx])

    mondrian_state = calibrate.fit(cal_scores, fit.y_true[cal_idx], num_classes, alpha, "raps", mondrian=True)
    bipartite_state = hierarchical.fit_bipartite(
        cal_scores, fit.y_true[cal_idx], bands[cal_idx], num_classes, esc, alpha, "raps",
        bands=BAND_ORDER,
    )
    # H4's reference arm (blueprint section 7): the MARGINAL calibrator, whose guarantee is
    # over all rows. H4 asks whether FRR in a subgroup materially exceeds 1 - that marginal
    # coverage -- i.e. whether the guarantee holds while the subgroup is unprotected.
    marginal_state = calibrate.fit(cal_scores, fit.y_true[cal_idx], num_classes, alpha, "raps", mondrian=False)

    eval_score_matrix = score_matrix[cal_idx]
    sets_mondrian = calibrate.prediction_sets(mondrian_state, eval_score_matrix)
    sets_bipartite = hierarchical.prediction_sets_bipartite(bipartite_state, eval_score_matrix, bands[cal_idx])
    sets_marginal = calibrate.prediction_sets(marginal_state, eval_score_matrix)

    y_cal = fit.y_true[cal_idx]
    lesion_cal = fit.lesion_ids[cal_idx]
    bands_cal = bands[cal_idx]
    band_mask = bands_cal == "<40"

    frr_mondrian = hierarchical.false_reassurance(
        sets_mondrian[band_mask], y_cal[band_mask], lesion_cal[band_mask], esc, n_boot=n_boot, seed=seed,
    )
    frr_bipartite = hierarchical.false_reassurance(
        sets_bipartite[band_mask], y_cal[band_mask], lesion_cal[band_mask], esc, n_boot=n_boot, seed=seed,
    )
    diff = _paired_frr_bootstrap(
        sets_mondrian[band_mask], sets_bipartite[band_mask],
        y_cal[band_mask], lesion_cal[band_mask], esc, n_boot=n_boot, seed=seed,
    )

    coverage_mondrian = hierarchical.cell_coverage(sets_mondrian, y_cal, bands_cal, esc, BAND_ORDER)
    coverage_bipartite = hierarchical.cell_coverage(sets_bipartite, y_cal, bands_cal, esc, BAND_ORDER)

    # ---- H4: FRR by group under the marginal calibrator, against 1 - marginal coverage ----
    covered_marginal = sets_marginal[np.arange(len(y_cal)), y_cal]
    marginal_coverage = float(covered_marginal.mean())
    frr_by_group = []
    groups = {"all_escalating": np.ones(len(y_cal), dtype=bool)}
    for band in ("<40", "40-59", "60+"):
        groups[band] = bands_cal == band
    for label, mask in groups.items():
        prop = hierarchical.false_reassurance(
            sets_marginal[mask], y_cal[mask], lesion_cal[mask], esc, n_boot=n_boot, seed=seed,
        )
        frr_by_group.append({
            "alpha": alpha, "calibrator": "raps_marginal", "group": label,
            "frr_k": prop.numerator, "frr_n": prop.denominator, "frr": prop.point,
            "cp_lo": prop.clopper_pearson[0], "cp_hi": prop.clopper_pearson[1],
            "marginal_coverage": marginal_coverage,
            "one_minus_marginal_coverage": 1.0 - marginal_coverage,
            "gap_frr_minus_1mcov": prop.point - (1.0 - marginal_coverage),
            # The project's own gate for trusting a band-conditional estimate
            # (research/selective/fairness.py MIN_GROUP_POSITIVES).
            "powered": bool(prop.denominator >= 30),
        })

    return {
        "frr_by_group": frr_by_group,
        "marginal_coverage": marginal_coverage,
        "alpha": alpha,
        "k_reg": k_reg, "raps_lambda": penalty,
        "n_tune": int(len(tune_idx)), "n_cal": int(len(cal_idx)), "n_cal_under40": int(band_mask.sum()),
        "frr_mondrian": frr_mondrian.as_dict(),
        "frr_bipartite": frr_bipartite.as_dict(),
        "diff_bipartite_minus_mondrian": diff,
        "bipartite_backed_off_cells": [list(c) for c in bipartite_state.backed_off_cells],
        "bipartite_degenerate_cells": [list(c) for c in bipartite_state.degenerate_cells],
        "mondrian_degenerate_classes": list(mondrian_state.degenerate_classes),
        "coverage_mondrian_under40": {
            k: v for k, v in coverage_mondrian.items() if k.startswith("<40")
        },
        "coverage_bipartite_under40": {
            k: v for k, v in coverage_bipartite.items() if k.startswith("<40")
        },
    }


def _append_ledger(rows: list[dict]) -> None:
    session = "v2_s37_conformal"
    ledger_rows = [{
        "timestamp": datetime.now(timezone.utc).isoformat(), "session": session,
        "method": f"F3_conformal[alpha={row['alpha']}]", "split": f"{COHORT}/<40",
        "macro_f1": "", "accuracy": "", "balanced_accuracy": "", "weighted_f1": "",
        "macro_roc_auc": "", "ece": "",
        "escalation_sens": "", "missed_serious": "",
        "p_value_vs_baseline": round(row["p_holm"], 6) if np.isfinite(row["p_holm"]) else "",
        "notes": (f"S37 F3 conformal subgroup safety; FRR<40 mondrian={row['frr_mondrian_point']:.4f} "
                  f"bipartite={row['frr_bipartite_point']:.4f}; diff={row['diff_point']:+.4f} "
                  f"[{row['diff_ci_lo']:+.4f},{row['diff_ci_hi']:+.4f}]; "
                  f"significant_holm={row['significant_holm']}"),
    } for row in rows]
    frame = pd.DataFrame(ledger_rows)
    if LEDGER_PATH.exists():
        existing = pd.read_csv(LEDGER_PATH)
        keep = existing["session"] != session
        frame = pd.concat([existing[keep], frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S37 -- F3 conformal subgroup safety (HAM-OOF)")
    parser.add_argument("--run-f3", action="store_true", required=True)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)

    results = [run_alpha(a, n_boot=args.n_boot) for a in ALPHAS]

    rows = []
    for res in results:
        rows.append({
            "cohort": COHORT, "alpha": res["alpha"], "k_reg": res["k_reg"], "raps_lambda": res["raps_lambda"],
            "n_cal": res["n_cal"], "n_cal_under40": res["n_cal_under40"],
            "frr_mondrian_point": res["frr_mondrian"]["point"],
            "frr_mondrian_k": res["frr_mondrian"]["numerator"], "frr_mondrian_n": res["frr_mondrian"]["denominator"],
            "frr_mondrian_ci_lo": res["frr_mondrian"][res["frr_mondrian"]["primary_interval"] + "_95"][0],
            "frr_mondrian_ci_hi": res["frr_mondrian"][res["frr_mondrian"]["primary_interval"] + "_95"][1],
            "frr_bipartite_point": res["frr_bipartite"]["point"],
            "frr_bipartite_k": res["frr_bipartite"]["numerator"], "frr_bipartite_n": res["frr_bipartite"]["denominator"],
            "frr_bipartite_ci_lo": res["frr_bipartite"][res["frr_bipartite"]["primary_interval"] + "_95"][0],
            "frr_bipartite_ci_hi": res["frr_bipartite"][res["frr_bipartite"]["primary_interval"] + "_95"][1],
            "diff_point": res["diff_bipartite_minus_mondrian"]["point"],
            "diff_ci_lo": res["diff_bipartite_minus_mondrian"]["ci_lo"],
            "diff_ci_hi": res["diff_bipartite_minus_mondrian"]["ci_hi"],
            "p_two_sided": res["diff_bipartite_minus_mondrian"]["p_two_sided"],
            "bipartite_backed_off_cells": ";".join(f"{b}/{g}" for b, g in res["bipartite_backed_off_cells"]),
            "bipartite_degenerate_cells": ";".join(f"{b}/{g}" for b, g in res["bipartite_degenerate_cells"]),
        })

    pvals = [r["p_two_sided"] for r in rows]
    adjusted, reject = holm_bonferroni(pvals)
    for row, p_holm, sig in zip(rows, adjusted, reject):
        row["p_holm"] = float(p_holm)
        row["significant_holm"] = bool(sig)

    print(f"\nF3 conformal subgroup safety, {COHORT} <40, Holm over {len(rows)} alphas:")
    for row in rows:
        print(f"  alpha={row['alpha']:.2f} FRR_mondrian={row['frr_mondrian_point']:.4f} "
              f"({row['frr_mondrian_k']}/{row['frr_mondrian_n']}) "
              f"FRR_bipartite={row['frr_bipartite_point']:.4f} ({row['frr_bipartite_k']}/{row['frr_bipartite_n']}) "
              f"diff={row['diff_point']:+.4f} [{row['diff_ci_lo']:+.4f},{row['diff_ci_hi']:+.4f}] "
              f"p={row['p_two_sided']:.4f} p_holm={row['p_holm']:.4f} "
              f"{'SIG' if row['significant_holm'] else 'ns'}")
        if row["bipartite_backed_off_cells"]:
            print(f"    backed-off cells: {row['bipartite_backed_off_cells']}")
        if row["bipartite_degenerate_cells"]:
            print(f"    degenerate cells: {row['bipartite_degenerate_cells']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    csv_path = OUT_DIR / "conformal_subgroup_safety.csv"
    frame.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path.relative_to(REPO_ROOT)}")

    # H4's artifact, named as the blueprint's section 14 file list requires.
    frr_rows = [r for res in results for r in res["frr_by_group"]]
    frr_frame = pd.DataFrame(frr_rows)
    frr_path = OUT_DIR / "frr_by_group.csv"
    frr_frame.to_csv(frr_path, index=False)
    print(f"wrote {frr_path.relative_to(REPO_ROOT)}")
    print("\nH4 -- FRR by group vs 1 - marginal coverage (marginal RAPS calibrator):")
    for r in frr_rows:
        print(f"  alpha={r['alpha']:.2f} {r['group']:<15s} FRR={r['frr']:.4f} "
              f"({r['frr_k']}/{r['frr_n']}) CP[{r['cp_lo']:.4f},{r['cp_hi']:.4f}]  "
              f"1-cov={r['one_minus_marginal_coverage']:.4f}  gap={r['gap_frr_minus_1mcov']:+.4f}"
              f"{'' if r['powered'] else '  [UNDERPOWERED n<30]'}")

    _append_ledger(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
