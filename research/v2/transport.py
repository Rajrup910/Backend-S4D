"""S38 -- V6 transport + V7 PAD: does a HAM-OOF-fitted mass threshold transport?

The gap S21 left open (CLAUDE.md's S8b caveat): every prior cross-domain result in this
project either re-tunes a cutoff on the target cohort (an oracle question, "how much
information is in the score here") or applies a rule fitted somewhere else without ever
separating *how much* of the gap is lost sensitivity from *how much* is simply spending
the wrong fraction of referral capacity. `research.v2.estimators.transport_term` (S31)
provides that separation:

    T_matched  = S_oracle(at the REALIZED burden) - S_frozen   -- zero under any monotone
                 recalibration, positive only when the frozen score genuinely reorders
                 cases differently on the target cohort than on HAM-OOF.
    burden_error = realized burden - intended burden           -- what a miscalibrated
                 cutoff actually costs a clinic in referral volume.
    T_intended = S_oracle(at the INTENDED burden) - S_frozen   -- signed: negative means
                 the frozen rule over-referred and bought sensitivity it never budgeted.

This module is pure application: fit the absolute cutoff `tau` on `s(x)` (escalation mass)
at HAM-OOF's own natural argmax burden, freeze it, and apply it UNCHANGED to BCN-20000,
MSKCC, PAD-UFES-20, and (as a same-domain sanity reference, not a target of the transport
claim) HAM-val. Every row is stamped `threshold_type=frozen_deployable`
(`research.v2.policies`); the target cohort's own natural argmax is reported alongside,
never in the same column, per the standing rule that oracle and frozen results never share
a table without that column.

**F5, exploratory only** (multiplicity.py: "oracle-vs-frozen transport magnitude" is
explicitly F5's territory, not a confirmatory family) -- point estimates and lesion-grouped
bootstrap intervals are reported, never a significance claim.

**The PAD caveat, stated before any number below is read.** PAD-UFES-20's escalating
prior is ~77% (inverted relative to HAM's ~19% and BCN/MSKCC's ~19-26%) -- documented
repeatedly in this project (S8b, S12, CLAUDE.md). Any policy that refers a large
*constant* fraction of PAD mechanically achieves high sensitivity simply because most of
PAD is escalating; a frozen HAM cutoff transporting "well" to PAD by this measure is not
evidence the mechanism replicated, only that PAD's prior does most of the work. This is
never cited as mechanism replication -- exactly the S8b/S12 discipline, restated here for
the transport question specifically.

    $py -m research.v2.transport --band ALL
    $py -m research.v2.transport --band "<40"
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.paths import load_class_mapping, resolve
from research.ablation.bootstrap import lesion_resample_indices
from research.external.frozen_params import escalating_indices
from research.stats.intervals import proportion
from research.v2 import estimators as est
from research.v2 import policies as pol

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = REPO_ROOT / "results" / "v2" / "panels"
OUT_DIR = REPO_ROOT / "results" / "v2"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"
CLASS_CODES: tuple[str, ...] = tuple(load_class_mapping().codes)
ESCALATING_CODES = frozenset({"akiec", "bcc", "mel"})

DEV_COHORT = "ham_oof"
TARGET_COHORTS: tuple[str, ...] = ("ham_val", "bcn20000", "mskcc", "pad")
PAD_CAVEAT = (
    "PAD-UFES-20's escalating prior is ~77%, inverted relative to HAM's ~19% -- any "
    "policy referring a large constant fraction of PAD achieves high sensitivity "
    "mechanically. Never cite PAD transport numbers as mechanism replication."
)
MSKCC_CAVEAT = (
    "72% of MSKCC rows have a null lesion_id (S28); effective_lesion_id falls back to a "
    "per-row singleton, so MSKCC's lesion-grouped bootstrap is optimistic -- reported for "
    "completeness (F5), never as confirmatory evidence."
)

SEED = 42
N_BOOT = 2000


def _load_panel(cohort: str) -> pd.DataFrame:
    return pd.read_csv(PANEL_DIR / f"{cohort}.csv")


def _band_slice(panel: pd.DataFrame, band: str) -> pd.DataFrame:
    if band == "ALL":
        return panel.reset_index(drop=True)
    return panel[panel["age_band"] == band].reset_index(drop=True)


def _scores_and_labels(panel: pd.DataFrame, esc: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probs = panel[[f"p_{c}" for c in CLASS_CODES]].to_numpy(dtype=float)
    y_esc = panel["true_code"].isin(ESCALATING_CODES).to_numpy()
    scores = est.escalation_mass(probs, esc)
    lesion_ids = panel["effective_lesion_id"].to_numpy()
    return probs, scores, y_esc, lesion_ids


def transport_bootstrap(
    dev_scores: np.ndarray, target_scores: np.ndarray, target_y_esc: np.ndarray,
    target_lesion_ids: np.ndarray, dev_rate: float, *, n_boot: int = N_BOOT, seed: int = SEED,
) -> dict:
    """Lesion-grouped bootstrap over the TARGET cohort of every `transport_term` output.

    `tau` (fixed on dev, once) is not re-fit inside the resample -- only the target
    cohort's sampling variability is characterised, which is the standard treatment of a
    frozen, already-deployed parameter.
    """
    point = est.transport_term(dev_scores, target_scores, target_y_esc, dev_rate, seed=seed)
    keys = ["S_frozen", "S_oracle_at_realized", "S_oracle_at_intended",
            "T_matched", "T_intended", "burden_realized", "burden_error"]
    draws = {k: [] for k in keys}
    for idx in lesion_resample_indices(target_lesion_ids, n_boot=n_boot, seed=seed):
        sub_y = target_y_esc[idx]
        if sub_y.sum() == 0:
            continue
        res = est.transport_term(dev_scores, target_scores[idx], sub_y, dev_rate, seed=seed)
        for k in keys:
            draws[k].append(res[k])

    out = {"point": point, "n_boot_used": len(draws["S_frozen"])}
    for k in keys:
        arr = np.asarray(draws[k], dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            out[f"{k}_ci_lo"] = float("nan")
            out[f"{k}_ci_hi"] = float("nan")
        else:
            out[f"{k}_ci_lo"] = float(np.percentile(arr, 2.5))
            out[f"{k}_ci_hi"] = float(np.percentile(arr, 97.5))
    return out


def frozen_vs_argmax_bootstrap(
    target_probs: np.ndarray, target_scores: np.ndarray, target_y_esc: np.ndarray,
    target_lesion_ids: np.ndarray, tau: float, esc: list[int],
    *, n_boot: int = N_BOOT, seed: int = SEED,
) -> dict:
    """H3's estimand (blueprint section 7): the sign of `S^{s,frozen} - S_argmax` on the
    target cohort, by paired lesion-grouped bootstrap.

    GO requires the frozen rule to BEAT argmax with a CI excluding zero; the falsifier is
    `frozen <= argmax`. Both arms are evaluated on the identical resampled rows in every
    draw, which is what makes the difference's interval a paired one.
    """
    refers_frozen = est.frozen_threshold_refers(target_scores, tau)
    refers_argmax = est.argmax_refers(target_probs, esc)
    point = (est.sensitivity(refers_frozen, target_y_esc)
             - est.sensitivity(refers_argmax, target_y_esc))

    draws = []
    for idx in lesion_resample_indices(target_lesion_ids, n_boot=n_boot, seed=seed):
        sub_y = target_y_esc[idx]
        if sub_y.sum() == 0:
            continue
        sf = est.sensitivity(refers_frozen[idx], sub_y)
        sa = est.sensitivity(refers_argmax[idx], sub_y)
        if np.isfinite(sf) and np.isfinite(sa):
            draws.append(sf - sa)
    draws = np.asarray(draws, dtype=float)
    if len(draws) == 0:
        return {"point": point, "ci_lo": float("nan"), "ci_hi": float("nan"), "n_boot_used": 0}
    return {
        "point": float(point),
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "n_boot_used": int(len(draws)),
    }


def run_cohort(cohort: str, band: str, esc: list[int], *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    dev_panel = _band_slice(_load_panel(DEV_COHORT), band)
    _, dev_scores, dev_y_esc, _ = _scores_and_labels(dev_panel, esc)
    dev_probs = dev_panel[[f"p_{c}" for c in CLASS_CODES]].to_numpy(dtype=float)
    r_dev = int(est.argmax_refers(dev_probs, esc).sum())
    dev_rate = r_dev / len(dev_panel) if len(dev_panel) else float("nan")

    target_panel = _band_slice(_load_panel(cohort), band)
    target_probs, target_scores, target_y_esc, target_lesion_ids = _scores_and_labels(target_panel, esc)

    boot = transport_bootstrap(dev_scores, target_scores, target_y_esc, target_lesion_ids,
                                dev_rate, n_boot=n_boot, seed=seed)
    point = boot["point"]

    # Target's own natural argmax, for context -- NEVER in the same threshold_type column.
    argmax_row = pol.natural_argmax(target_probs, target_y_esc, esc, eval_cohort=cohort, subgroup=band)
    argmax_flags = est.argmax_refers(target_probs, esc)[target_y_esc]
    argmax_prop = proportion(argmax_flags, target_lesion_ids[target_y_esc],
                              label=f"argmax_natural[{cohort}/{band}]", n_boot=n_boot, seed=seed)

    frozen_flags = est.frozen_threshold_refers(target_scores, est.threshold_for_rate(dev_scores, dev_rate))[target_y_esc]
    frozen_prop = proportion(frozen_flags, target_lesion_ids[target_y_esc],
                              label=f"frozen_transport[{cohort}/{band}]", n_boot=n_boot, seed=seed)

    escalating_prior = float(target_y_esc.mean()) if len(target_y_esc) else float("nan")
    tau = est.threshold_for_rate(dev_scores, dev_rate)
    h3 = frozen_vs_argmax_bootstrap(target_probs, target_scores, target_y_esc,
                                     target_lesion_ids, tau, esc, n_boot=n_boot, seed=seed)

    row = {
        "dev_cohort": DEV_COHORT, "eval_cohort": cohort, "band": band,
        "threshold_type": "frozen_deployable",
        "n_dev": len(dev_panel), "n_target": len(target_panel), "n_escalating_target": int(target_y_esc.sum()),
        "escalating_prior_target": escalating_prior,
        "tau": point["tau"], "r_intended": point["r_intended"], "r_realized": point["r_realized"],
        "burden_intended": point["burden_intended"], "burden_realized": point["burden_realized"],
        "burden_error": point["burden_error"],
        "S_frozen": point["S_frozen"],
        "S_frozen_ci_lo": frozen_prop.interval[0], "S_frozen_ci_hi": frozen_prop.interval[1],
        "S_frozen_ci_kind": frozen_prop.primary,
        "S_oracle_at_realized": point["S_oracle_at_realized"],
        "S_oracle_at_intended": point["S_oracle_at_intended"],
        "T_matched": point["T_matched"], "T_matched_ci_lo": boot["T_matched_ci_lo"], "T_matched_ci_hi": boot["T_matched_ci_hi"],
        "T_intended": point["T_intended"], "T_intended_ci_lo": boot["T_intended_ci_lo"], "T_intended_ci_hi": boot["T_intended_ci_hi"],
        "S_argmax_natural": argmax_row.sensitivity,
        "S_argmax_natural_ci_lo": argmax_prop.interval[0], "S_argmax_natural_ci_hi": argmax_prop.interval[1],
        "S_argmax_natural_ci_kind": argmax_prop.primary,
        "burden_argmax_natural": argmax_row.burden,
        # H3 (blueprint section 7): GO iff frozen beats argmax with CI excluding 0;
        # falsifier is frozen <= argmax.
        "S_frozen_minus_argmax": h3["point"],
        "S_frozen_minus_argmax_ci_lo": h3["ci_lo"],
        "S_frozen_minus_argmax_ci_hi": h3["ci_hi"],
        "n_boot_used": boot["n_boot_used"],
        "caveat": PAD_CAVEAT if cohort == "pad" else (MSKCC_CAVEAT if cohort == "mskcc" else ""),
    }
    return row


def _append_ledger(rows: list[dict], band: str) -> None:
    session = "v2_s38_transport"
    ledger_rows = [{
        "timestamp": datetime.now(timezone.utc).isoformat(), "session": session,
        "method": f"transport[{row['eval_cohort']}]", "split": f"{row['eval_cohort']}/{band}",
        "macro_f1": "", "accuracy": "", "balanced_accuracy": "", "weighted_f1": "",
        "macro_roc_auc": "", "ece": "",
        "escalation_sens": round(row["S_frozen"], 6), "missed_serious": "",
        "p_value_vs_baseline": "",
        "notes": (f"S38 frozen transport; tau={row['tau']:.4f}; burden intended="
                  f"{row['burden_intended']:.4f} realized={row['burden_realized']:.4f}; "
                  f"S_frozen={row['S_frozen']:.4f} S_argmax_natural={row['S_argmax_natural']:.4f}; "
                  f"T_matched={row['T_matched']:+.4f} T_intended={row['T_intended']:+.4f}; "
                  f"escalating_prior={row['escalating_prior_target']:.4f}; "
                  f"threshold_type=frozen_deployable; F5 exploratory, no confirmatory family"
                  + (f"; CAVEAT: {row['caveat']}" if row["caveat"] else "")),
    } for row in rows]
    frame = pd.DataFrame(ledger_rows)
    if LEDGER_PATH.exists():
        existing = pd.read_csv(LEDGER_PATH)
        keep = ~((existing["session"] == session) & (existing["split"].str.endswith(f"/{band}")))
        frame = pd.concat([existing[keep], frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)


def run(band: str, *, n_boot: int, cohorts: tuple[str, ...]) -> int:
    esc = escalating_indices()
    rows = [run_cohort(c, band, esc, n_boot=n_boot) for c in cohorts]

    print(f"\nS38 frozen transport (dev={DEV_COHORT}, band={band}):")
    for row in rows:
        print(f"  {row['eval_cohort']:10s} n={row['n_target']:6d} esc={row['n_escalating_target']:4d} "
              f"prior={row['escalating_prior_target']:.4f} tau={row['tau']:.4f}")
        print(f"    burden  intended={row['burden_intended']:.4f} realized={row['burden_realized']:.4f} "
              f"error={row['burden_error']:+.4f}")
        print(f"    S_frozen={row['S_frozen']:.4f} [{row['S_frozen_ci_lo']:.4f},{row['S_frozen_ci_hi']:.4f}]  "
              f"S_argmax_natural={row['S_argmax_natural']:.4f} [{row['S_argmax_natural_ci_lo']:.4f},{row['S_argmax_natural_ci_hi']:.4f}] "
              f"(burden {row['burden_argmax_natural']:.4f})")
        print(f"    T_matched={row['T_matched']:+.4f} [{row['T_matched_ci_lo']:+.4f},{row['T_matched_ci_hi']:+.4f}]  "
              f"T_intended={row['T_intended']:+.4f} [{row['T_intended_ci_lo']:+.4f},{row['T_intended_ci_hi']:+.4f}]")
        print(f"    H3 S_frozen-S_argmax={row['S_frozen_minus_argmax']:+.4f} "
              f"[{row['S_frozen_minus_argmax_ci_lo']:+.4f},{row['S_frozen_minus_argmax_ci_hi']:+.4f}]")
        if row["caveat"]:
            print(f"    CAVEAT: {row['caveat']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "transport_frozen_vs_oracle.csv"
    frame = pd.DataFrame(rows)
    if csv_path.exists():
        previous = pd.read_csv(csv_path)
        previous = previous[previous["band"] != band]  # this runner prunes its own prior rows
        frame = pd.concat([previous, frame], ignore_index=True)
    frame.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path.relative_to(REPO_ROOT)}")

    _append_ledger(rows, band)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S38 -- frozen mass-threshold transport (F5, exploratory)")
    parser.add_argument("--band", default="ALL", choices=["ALL", "<40", "40-59", "60+"])
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--cohorts", nargs="*", default=list(TARGET_COHORTS))
    args = parser.parse_args(argv)
    return run(args.band, n_boot=args.n_boot, cohorts=tuple(args.cohorts))


if __name__ == "__main__":
    raise SystemExit(main())
