"""S68 -- target-side referral recalibration: spend the budget you declared, not twice it.

The problem S59 left. The deployed stack is S56's per-band selective abstention at a nominal 0.20
referral budget. On the reserved cohort it referred **0.386** -- nearly double -- and the contract's
0.25 limit was missed by more than the floors were. That is not a mechanism failure. S56's
thresholds are quantiles of an uncertainty score computed on **HAM** rows, and reserved is BCN and
MSKCC. Under domain shift the score distribution moves right, so a cut placed at the source's
80th percentile lands far below the target's and refers everything above it.

The fix is the cheapest one available and it changes no model: place the same per-band cuts at the
**target domain's** quantiles instead, estimated on BCN and MSKCC rows from the pooled *training*
split -- same archives as reserved, never read, and now scored by the S72 fold models that never
saw them. The mechanism, the floors and the score are all S56's; only where the quantile is read
from changes.

Two arms, both fitted on the V4 cross-fitted OOF matrix:

    source   thresholds from the quantile of every OOF row, HAM included. The status quo, restated
             on a V4 base so the comparison is like for like.
    target   thresholds from the quantile of the BCN/MSKCC OOF rows alone.

The endpoint declared before either is scored: **|realised referral - nominal budget| on the
BCN/MSKCC rows of V4 val**, lower is better, MCID 0.05. Under-40 system sensitivity is reported
beside it as the cost term, because a rule that hits its budget by referring fewer of the cases
that need referring has not helped. Neither arm may be preferred on a reserved number -- S73 reads
reserved once, as part of the composed stack, and this session does not read it at all.

Depends on S72: it needs `results/v4/kfold/oof_predictions.csv` and
`results/v4/kfold/val_predictions.csv`. Both are written by `scripts/run_s72.ps1`.

    $py -m research.v4.s68_target_thresholds --freeze-plan
    $py -m research.v4.s68_target_thresholds --run
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "v4" / "s68"
PLAN_PATH = REPO_ROOT / "results" / "v4" / "s68_plan.json"
KFOLD_DIR = REPO_ROOT / "results" / "v4" / "kfold"
OOF_PATH = KFOLD_DIR / "oof_predictions.csv"
VAL_PATH = KFOLD_DIR / "val_predictions.csv"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"
SESSION = "v4_s68"

#: The archives reserved is made of. "Target domain" means these, and nothing else.
TARGET_ARCHIVES = ("bcn20000", "mskcc")
BUDGETS = (0.10, 0.15, 0.20, 0.25, 0.30)
MCID_BUDGET_ERROR = 0.05
ARMS = ("source", "target")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(path: Path, what: str) -> None:
    if not path.is_file():
        raise SystemExit(
            f"{path.relative_to(REPO_ROOT)} is missing -- {what}. S68 depends on S72: run\n"
            f"    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\run_s72.ps1")


def load_panel(path: Path, name: str):
    """A `research.v4.s56_abstention.Panel` over a V4 prediction CSV, plus its archive column."""
    from research.v4.s56_abstention import Panel, esc_indices
    from ml.paths import load_class_mapping

    mapping = load_class_mapping()
    frame = pd.read_csv(path, low_memory=False)
    probs = frame[[f"p_{code}" for code in mapping.codes]].to_numpy(float)
    panel = Panel(name, probs, frame["y_true"].to_numpy(), frame["age_band"].to_numpy(),
                  frame["group_id"].astype(str).to_numpy(), esc_indices())
    return panel, frame["archive"].astype(str).to_numpy()


def subset(panel, mask: np.ndarray, name: str):
    from research.v4.s56_abstention import Panel

    return Panel(name, panel.probs[mask], panel.y7[mask], panel.bands[mask],
                 panel.groups[mask], panel.esc_idx)


def fit_arm(arm: str, oof, oof_archives: np.ndarray, score: str, budget: float):
    """S56's per-band fit, run on whichever rows the arm says the quantile comes from."""
    from research.v4 import s56_abstention as s56

    fit_panel = oof if arm == "source" else subset(
        oof, np.isin(oof_archives, TARGET_ARCHIVES), "oof_target")
    return s56.fit_band(fit_panel, score, budget), len(fit_panel)


def apply_policy(policy, panel) -> np.ndarray:
    """S56's deployed rule: refer i iff u_i >= min(tau_band(i), tau_global)."""
    scores = panel.score(policy.score)
    thresholds = np.full(len(panel), policy.tau_global, dtype=float)
    for band, tau in policy.tau_band.items():
        thresholds = np.where(panel.bands == band, np.minimum(tau, policy.tau_global), thresholds)
    return scores >= thresholds


def run() -> int:
    if not PLAN_PATH.is_file():
        print("no frozen plan -- run --freeze-plan first")
        return 2
    _require(OOF_PATH, "S72's cross-fitted OOF matrix has not been assembled")
    _require(VAL_PATH, "V4 val has not been scored with the fold ensemble")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    score = plan["score"]

    oof, oof_archives = load_panel(OOF_PATH, "v4_oof")
    val, val_archives = load_panel(VAL_PATH, "v4_val")
    target_val = np.isin(val_archives, TARGET_ARCHIVES)
    print(f"OOF {len(oof):,} rows ({np.isin(oof_archives, TARGET_ARCHIVES).sum():,} target-domain)")
    print(f"val {len(val):,} rows ({target_val.sum():,} target-domain) -- the evaluation surface")
    if target_val.sum() == 0:
        raise SystemExit("V4 val has no BCN/MSKCC rows; the endpoint is undefined")
    val_target = subset(val, target_val, "v4_val_target")

    rows = []
    for budget in BUDGETS:
        for arm in ARMS:
            policy, n_fit = fit_arm(arm, oof, oof_archives, score, budget)
            referred = apply_policy(policy, val_target)
            realised = float(referred.mean())
            described = _describe(val_target, referred)
            rows.append({
                "arm": arm, "budget": budget, "n_fit_rows": n_fit,
                "realised_referral": realised,
                "budget_error": abs(realised - budget),
                "signed_budget_error": realised - budget,
                "tau_global": policy.tau_global,
                **{f"tau_{band}": tau for band, tau in policy.tau_band.items()},
                **described,
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / "arms_val.csv", index=False)
    print(f"\n{'budget':>7} {'arm':>7} {'referred':>9} {'err':>7} "
          f"{'<40 sys sens':>13} {'macroF1':>8}")
    for _, row in frame.iterrows():
        print(f"{row['budget']:>7.2f} {row['arm']:>7} {row['realised_referral']:>9.3f} "
              f"{row['signed_budget_error']:>+7.3f} {row['sys_sens_<40']:>13.3f} "
              f"{row['retained_macro_f1']:>8.3f}")

    # The declared endpoint, averaged over the budget grid so the verdict does not rest on one
    # operating point, and reported at the deployed 0.20 as well.
    pivot = frame.pivot(index="budget", columns="arm", values="budget_error")
    mean_gain = float((pivot["source"] - pivot["target"]).mean())
    at_deployed = float(pivot.loc[0.20, "source"] - pivot.loc[0.20, "target"])
    sens = frame.pivot(index="budget", columns="arm", values="sys_sens_<40")
    sens_cost = float((sens["target"] - sens["source"]).loc[0.20])
    verdict = ("ADOPT" if mean_gain >= MCID_BUDGET_ERROR and at_deployed > 0 else
               "NULL" if abs(mean_gain) < MCID_BUDGET_ERROR else "REJECT")

    report = {
        "session": "S68",
        "plan_sha256": sha256(PLAN_PATH),
        "inputs": {"oof": str(OOF_PATH.relative_to(REPO_ROOT)), "oof_sha256": sha256(OOF_PATH),
                   "val": str(VAL_PATH.relative_to(REPO_ROOT)), "val_sha256": sha256(VAL_PATH)},
        "score": score,
        "endpoint": "|realised referral - nominal budget| on V4 val's BCN/MSKCC rows",
        "mcid": MCID_BUDGET_ERROR,
        "mean_budget_error_reduction_target_vs_source": mean_gain,
        "budget_error_reduction_at_0.20": at_deployed,
        "under40_system_sensitivity_change_at_0.20": sens_cost,
        "per_budget": frame.to_dict("records"),
        "verdict": verdict,
        "reads": {"reserved": False, "ham_test": False},
        "note": "Fitted and developed only. S73 composes the adopted arm into the stack and reads "
                "reserved once, under its own receipt.",
    }
    (OUT_DIR / "s68_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    write_ledger([{
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(), "session": SESSION,
        "method": f"s68_{row['arm']}_b{int(row['budget'] * 100):02d}", "split": "v4_val_target",
        "macro_f1": row["retained_macro_f1"], "escalation_sens": row["sys_sens_<40"],
        "notes": (f"realised_referral={row['realised_referral']:.4f} "
                  f"budget={row['budget']:.2f} err={row['signed_budget_error']:+.4f} "
                  f"score={score} n_fit={row['n_fit_rows']} plan={sha256(PLAN_PATH)[:16]}"),
    } for _, row in frame.iterrows()])

    print(f"\nmean budget error reduction (target - source): {mean_gain:+.4f} "
          f"(MCID {MCID_BUDGET_ERROR})")
    print(f"at the deployed 0.20 budget: {at_deployed:+.4f}; "
          f"<40 system sensitivity change {sens_cost:+.4f}")
    print(f"VERDICT {verdict}")
    print(f"wrote {(OUT_DIR / 's68_report.json').relative_to(REPO_ROOT)}")
    return 0


def _describe(panel, referred: np.ndarray) -> dict[str, float]:
    from research.v4.s56_abstention import macro_f1

    out: dict[str, float] = {}
    system_escalates = panel.pred_esc | referred
    for band in ("<40", "40-59", "60+"):
        mask = (panel.bands == band) & panel.y_esc
        out[f"sys_sens_{band}"] = (float(system_escalates[mask].mean())
                                   if mask.sum() else float("nan"))
        out[f"n_esc_{band}"] = int(mask.sum())
    kept = ~referred
    out["retained_macro_f1"] = (macro_f1(panel.y7[kept], panel.pred[kept])
                                if kept.sum() else float("nan"))
    out["coverage"] = float(kept.mean())
    return out


def write_ledger(rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    if LEDGER_PATH.is_file():
        old = pd.read_csv(LEDGER_PATH, low_memory=False)
        kept = old[old["session"] != SESSION]
        print(f"ledger: pruned {len(old) - len(kept)} prior {SESSION} rows")
        frame = pd.concat([kept, frame], ignore_index=True)
    frame.to_csv(LEDGER_PATH, index=False)


def freeze_plan() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # The score is pinned to whatever S56 selected on val, not re-selected here: re-picking it on a
    # V4 panel would be a second selection on the same development surface.
    s56_selection = json.loads(
        (REPO_ROOT / "results" / "v4" / "s56" / "selection_val.json").read_text(encoding="utf-8"))
    score = s56_selection.get("selected") or s56_selection.get("score")
    payload = {
        "session": "s68",
        "purpose": "place S56's per-band referral cuts at the target domain's quantiles so the "
                   "realised referral rate matches the declared budget under domain shift",
        "depends_on": ["S71 (fold partition)", "S72 (cross-fitted OOF + val ensemble scores)"],
        "arms": {"source": "quantile over all OOF rows (status quo, restated on a V4 base)",
                 "target": f"quantile over OOF rows whose archive is in {list(TARGET_ARCHIVES)}"},
        "score": score,
        "score_provenance": "inherited from S56's val selection; not re-selected here",
        "budgets": list(BUDGETS),
        "endpoint": "|realised referral - nominal budget| on V4 val's BCN/MSKCC rows",
        "mcid": MCID_BUDGET_ERROR,
        "cost_term": "under-40 system sensitivity at the deployed 0.20 budget, reported beside "
                     "the endpoint; a rule that hits its budget by referring fewer of the cases "
                     "that need referring has not helped",
        "decision_rule": "ADOPT if the mean budget-error reduction reaches the MCID and the "
                         "reduction at 0.20 is positive; NULL if the mean is within the MCID of "
                         "zero; REJECT otherwise",
        "reads": {"reserved": False, "ham_test": False},
        "not_done": ["S68 fits and develops only; the reserved read belongs to S73."],
    }
    PLAN_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {PLAN_PATH.relative_to(REPO_ROOT)}  sha256 {sha256(PLAN_PATH)[:16]}")
    print(f"  score inherited from S56: {score}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-plan", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    return freeze_plan() if args.freeze_plan else run()


if __name__ == "__main__":
    raise SystemExit(main())
