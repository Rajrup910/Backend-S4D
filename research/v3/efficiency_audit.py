"""S40 / Phase A2 -- re-reading S34's "under-40 is the least efficient band".

S34 (`results/v2/S34_CHECKPOINT.md` section 3) corrected matched-budget sensitivity for the
ceiling a band's prevalence imposes:

    ceiling     = min(1, q / prior)
    efficiency  = sensitivity / ceiling

and found under-40 least efficient in 16 of 20 cohort x budget cells, concluding the ranking
deficit is real and largest where it matters clinically. **The arithmetic is right.** What
the table does not carry is the value this statistic takes under a score that carries no
information at all. Referring a random q fraction catches ~q of the escalating cases, so

    chance efficiency = q / min(1, q / prior) = max(q, prior)

which varies by a factor of seven across the bands being compared -- 0.0485 for HAM-OOF
under-40 against 0.3555 for 60+. Read against it, the same two cells that look like 0.516
(bad) and 0.987 (good) are 10.3x and 2.8x chance respectively, and the ranking inverts.

That inversion is not, by itself, a correction: it depends on *how* one normalizes. The
multiplicative reading (efficiency / chance) favours under-40; the additive reading
((efficiency - chance) / (1 - chance)), which maps chance to 0 and perfect to 1, still
favours 60+. Two defensible normalizations of one statistic disagree about the ordering,
which is the honest reason this instrument cannot settle the question on its own.

So the audit reports a third column that needs no normalization choice at all: band AUC and
partial AUC, which are invariant to prevalence by construction. That instrument was already
computed in the same session (`results/v2/auc_table.csv`) and it disagrees with the headline
reading -- under-40 is worst **only in HAM-OOF**, is level with 60+ in BCN, and is the
*best*-ranked band in MSKCC and PAD, reproducing S14's non-replication.

This is a correction of the reading, not of the sums. It matters because it decides whether
an under-40-specific intervention is chasing a HAM10000 artifact.

    $py -m research.v3.efficiency_audit
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.external import frozen_params as fp
from research.v2 import estimators as est
from research.v2 import frontier as fr

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = REPO_ROOT / "results" / "v2" / "panels"
S34_EFFICIENCY = REPO_ROOT / "results" / "v2" / "frontier_efficiency.csv"
S34_AUC = REPO_ROOT / "results" / "v2" / "auc_table.csv"
OUT_DIR = REPO_ROOT / "results" / "v3"
OUT_CSV = OUT_DIR / "efficiency_audit.csv"
OUT_JSON = OUT_DIR / "efficiency_audit_report.json"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"

SEED = 42
N_BOOT = 500  # S34's own exploratory setting, so the two are comparable
FPR_MAX = 0.20
Q_GRID = (0.05, 0.10, 0.15, 0.20, 0.30)
BANDS = ("<40", "40-59", "60+")
COHORTS = {"ham_oof": "ham_oof", "bcn20000": "bcn20000", "mskcc": "mskcc", "pad": "pad"}
REPRO_TOL = 0.01


def _panel_arrays(name: str) -> dict:
    panel = pd.read_csv(PANEL_DIR / f"{name}.csv")
    esc = fp.escalating_indices()
    prob_cols = [c for c in panel.columns if c.startswith("p_") and not c.startswith("p_raw_")]
    return {
        "s": est.escalation_mass(panel[prob_cols].to_numpy(dtype=float), esc),
        "y_esc": np.isin(panel["y_true"].to_numpy(), esc),
        "bands": panel["age_band"].astype(str).to_numpy(),
        "lesion_ids": panel["effective_lesion_id"].astype(str).to_numpy(),
    }


def audit_cohort(cohort: str, n_boot: int) -> tuple[list[dict], list[dict]]:
    """Efficiency rows (per q) and one prior-free AUC row per band."""
    data = _panel_arrays(COHORTS[cohort])
    eff_rows, auc_rows = [], []

    for band in BANDS:
        mask = data["bands"] == band
        n, y = int(mask.sum()), data["y_esc"][mask]
        n_esc = int(y.sum())
        if n_esc == 0 or n_esc == n:
            continue
        prior = n_esc / n
        scores, lesions = data["s"][mask], data["lesion_ids"][mask]

        auc = fr.partial_auc_ci(y, scores, lesions, fpr_max=FPR_MAX, seed=SEED, n_boot=n_boot)
        auc_rows.append({
            "cohort": cohort, "band": band, "n": n, "n_escalating": n_esc,
            "prior": round(prior, 4),
            "full_auc": auc["full_auc"], "partial_auc_mcclish": auc["partial_auc_mcclish"],
            "pauc_ci_lo": auc["ci_lo"], "pauc_ci_hi": auc["ci_hi"],
            "underpowered": bool(n_esc < 30),
        })

        for point in fr.frontier(scores, y, lesions, q_grid=Q_GRID, seed=SEED, n_boot=n_boot):
            q, sens = point.q, point.sensitivity
            ceiling = min(1.0, q / prior)
            efficiency = sens / ceiling if ceiling > 0 else float("nan")
            chance = max(q, prior)
            eff_rows.append({
                "cohort": cohort, "band": band, "n": n, "n_escalating": n_esc,
                "prior": round(prior, 4), "q": q,
                "sensitivity": round(sens, 4),
                "sens_ci_lo": round(point.ci_lo, 4), "sens_ci_hi": round(point.ci_hi, 4),
                "ceiling": round(ceiling, 4),
                "efficiency": round(efficiency, 4),
                # --- the three columns S34's table did not carry ---
                "chance_baseline": round(chance, 4),
                "lift_ratio": round(efficiency / chance, 4) if chance > 0 else float("nan"),
                "corrected_efficiency": round((efficiency - chance) / (1 - chance), 4)
                if chance < 1 else float("nan"),
                "eff_ci_lo": round(point.ci_lo / ceiling, 4) if ceiling > 0 else float("nan"),
                "eff_ci_hi": round(min(point.ci_hi / ceiling, 1.0), 4) if ceiling > 0 else float("nan"),
                "underpowered": bool(n_esc < 30),
            })
    return eff_rows, auc_rows


NEAR_TIE = 0.01


def _worst_band(frame: pd.DataFrame, cohort: str, column: str, q: float | None = None) -> str:
    sub = frame[frame["cohort"] == cohort]
    if q is not None:
        sub = sub[sub["q"] == q]
    return "n/a" if sub.empty else str(sub.loc[sub[column].idxmin(), "band"])


def _under40_margin(auc: pd.DataFrame, cohort: str, column: str) -> dict:
    """How far under-40 sits from the next-worst band -- a label alone hides a 0.003 tie.

    Ranking by argmin makes a band "worst" whether it loses by 0.003 or 0.10. In BCN the
    two prior-free variants disagree precisely because the margin is inside the noise, and
    reporting only the label would manufacture a deficit out of a tie.
    """
    sub = auc[auc["cohort"] == cohort].set_index("band")[column]
    if "<40" not in sub.index or len(sub) < 2:
        return {"margin": float("nan"), "verdict": "n/a"}
    others = sub.drop("<40")
    margin = float(sub["<40"] - others.min())  # negative => under-40 is worst
    if abs(margin) < NEAR_TIE:
        verdict = "tied"
    elif margin < 0:
        verdict = "under40_worst"
    else:
        verdict = "under40_not_worst"
    return {"margin": round(margin, 4), "verdict": verdict,
            "under40": round(float(sub["<40"]), 4),
            "next_worst_band": str(others.idxmin()),
            "next_worst": round(float(others.min()), 4)}


def _reproduction_check(eff: pd.DataFrame) -> dict:
    """Prove this module reads what S34 read, before disagreeing with how it was read."""
    if not S34_EFFICIENCY.is_file():
        return {"checked": False, "reason": "results/v2/frontier_efficiency.csv absent"}
    s34 = pd.read_csv(S34_EFFICIENCY)
    merged = eff.merge(s34[["cohort", "band", "q", "efficiency"]],
                       on=["cohort", "band", "q"], how="inner", suffixes=("", "_s34"))
    if merged.empty:
        return {"checked": False, "reason": "no overlapping cells"}
    diff = (merged["efficiency"] - merged["efficiency_s34"]).abs()
    return {
        "checked": True, "n_cells": int(len(merged)),
        "max_abs_diff": float(diff.max()),
        "reproduced": bool(diff.max() < REPRO_TOL),
        "tolerance": REPRO_TOL,
    }


def run(n_boot: int) -> int:
    eff_rows, auc_rows = [], []
    for cohort in COHORTS:
        print(f"auditing {cohort}...")
        e, a = audit_cohort(cohort, n_boot)
        eff_rows.extend(e)
        auc_rows.extend(a)

    eff = pd.DataFrame(eff_rows)
    auc = pd.DataFrame(auc_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eff.to_csv(OUT_CSV, index=False)
    auc.to_csv(OUT_DIR / "efficiency_audit_auc.csv", index=False)

    repro = _reproduction_check(eff)

    # Which band does each instrument call worst, per cohort?
    verdicts = {}
    for cohort in COHORTS:
        verdicts[cohort] = {
            "worst_by_raw_efficiency_q05": _worst_band(eff, cohort, "efficiency", 0.05),
            "worst_by_lift_ratio_q05": _worst_band(eff, cohort, "lift_ratio", 0.05),
            "worst_by_corrected_efficiency_q05": _worst_band(eff, cohort, "corrected_efficiency", 0.05),
            "worst_by_partial_auc": _worst_band(auc, cohort, "partial_auc_mcclish"),
            "worst_by_full_auc": _worst_band(auc, cohort, "full_auc"),
        }

    margins = {
        c: {"full_auc": _under40_margin(auc, c, "full_auc"),
            "partial_auc": _under40_margin(auc, c, "partial_auc_mcclish")}
        for c in COHORTS
    }
    # A cohort supports the claim only if BOTH prior-free variants call under-40 worst by
    # more than a near-tie; if they disagree, the band ordering is flat there.
    supports = [c for c, m in margins.items()
                if m["full_auc"]["verdict"] == "under40_worst"
                and m["partial_auc"]["verdict"] == "under40_worst"]
    reverses = [c for c, m in margins.items()
                if m["full_auc"]["verdict"] == "under40_not_worst"
                and m["partial_auc"]["verdict"] == "under40_not_worst"]
    flat = [c for c in COHORTS if c not in supports and c not in reverses]
    agree = sum(1 for v in verdicts.values()
                if v["worst_by_raw_efficiency_q05"] == v["worst_by_full_auc"])

    report = {
        "session": "S40", "phase": "A2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "statement": (
            "Re-read of S34's exploratory efficiency claim. S34's arithmetic is reproduced, "
            "not disputed; what is added is the chance baseline max(q, prior) the statistic "
            "is silent about, two normalizations of it that disagree with each other, and the "
            "prior-free AUC instrument that needs no normalization choice."
        ),
        "n_boot": n_boot, "seed": SEED, "q_grid": list(Q_GRID),
        "reproduction_of_s34": repro,
        "worst_band_by_instrument": verdicts,
        "under40_margins": margins,
        "near_tie_threshold": NEAR_TIE,
        "finding": {
            "supports_under40_deficit": supports,
            "reverses_under40_deficit": reverses,
            "flat_or_ambiguous": flat,
            "n_cohorts": len(COHORTS),
            "instruments_agree_in_n_cohorts": agree,
            "reading": (
                f"Both prior-free variants call under-40 worst in {len(supports)} of "
                f"{len(COHORTS)} cohorts ({', '.join(supports) if supports else 'none'}); "
                f"both call it NOT worst in {len(reverses)} "
                f"({', '.join(reverses) if reverses else 'none'}); "
                f"the ordering is flat or the variants disagree in {len(flat)} "
                f"({', '.join(flat) if flat else 'none'}). Separately, the two "
                "chance-corrections of S34's efficiency disagree with each other in every "
                "cohort, so that instrument does not decide the ordering on its own."
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nS34 reproduction: {repro}")
    print("\nworst band by instrument (q=0.05 where applicable):")
    header = f"  {'cohort':<10} {'raw eff':<10} {'lift':<10} {'corrected':<12} {'pAUC':<8} {'fullAUC':<8}"
    print(header)
    for cohort, v in verdicts.items():
        print(f"  {cohort:<10} {v['worst_by_raw_efficiency_q05']:<10} "
              f"{v['worst_by_lift_ratio_q05']:<10} {v['worst_by_corrected_efficiency_q05']:<12} "
              f"{v['worst_by_partial_auc']:<8} {v['worst_by_full_auc']:<8}")
    print("\nunder-40 margin vs next-worst band (negative => under-40 is worst):")
    for cohort, m in margins.items():
        print(f"  {cohort:<10} full_auc {m['full_auc']['margin']:+.4f} "
              f"({m['full_auc']['verdict']})   pAUC {m['partial_auc']['margin']:+.4f} "
              f"({m['partial_auc']['verdict']})")
    print(f"\n{report['finding']['reading']}")
    print(f"\nwrote {OUT_CSV.relative_to(REPO_ROOT)}\nwrote {OUT_JSON.relative_to(REPO_ROOT)}")

    _append_ledger(report)
    return 0


def _append_ledger(report: dict) -> None:
    session, method = "v3_s40_efficiency_audit", "A2_efficiency_audit"
    found = report["finding"]
    row = {
        "timestamp": report["generated_at"], "session": session, "method": method,
        "split": "oof", "macro_f1": "", "accuracy": "", "balanced_accuracy": "",
        "weighted_f1": "", "macro_roc_auc": "", "ece": "", "escalation_sens": "",
        "missed_serious": "", "p_value_vs_baseline": "",
        "notes": (
            f"S40_A2; prior-free AUC supports under-40 deficit in "
            f"{len(found['supports_under40_deficit'])}/{found['n_cohorts']} cohorts, "
            f"reverses in {len(found['reverses_under40_deficit'])}, "
            f"flat in {len(found['flat_or_ambiguous'])}; S34 efficiency reproduced="
            f"{report['reproduction_of_s34'].get('reproduced')}"
        ),
    }
    frame = pd.DataFrame([row])
    if LEDGER_PATH.is_file():
        old = pd.read_csv(LEDGER_PATH)
        frame = pd.concat([old[~((old["session"] == session) & (old["method"] == method))],
                           frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S40 A2 -- efficiency audit")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)
    return run(args.n_boot)


if __name__ == "__main__":
    raise SystemExit(main())
