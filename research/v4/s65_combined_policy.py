"""S65 -- one combined per-band triage policy: S55 calibration -> frozen lambda -> S56 abstention.

The three decision-layer tools that measured positive, composed on the frozen V1 ensemble in the
order S59 will use (V4 runbook §6b, S65):

1. **per-band Dirichlet** (S55, `frozen_params.calibrate_by_band`; cross-fitted within band on
   OOF, the deployed per-band maps on val and reserved);
2. **the frozen 3-band lambda rule** (S5, `frozen_params.apply_age_rule`) -- S66 rejected the
   per-hospital lambda, S57b rejected lambda(age), so this is the rule S65 takes. It is applied
   to the per-band-calibrated probabilities although it was fit under the global map: the
   runbook asks for the frozen rule, and refitting it here would be a fourth tool;
3. **per-band CRC-floor abstention** (S56), **refit** on the S55-calibrated scores with the lambda
   rule's decision defining which escalating rows are still silent misses.

Arms (all on one set of rows, so every contrast is paired):

* ``S56``      global map, argmax, S56's band abstention with S56's frozen score -- the primary
  comparator; it must reproduce S56's own frontier exactly, and the run refuses if it does not.
* ``LAM``      global map, frozen lambda, no referral -- the non-inferiority comparator.
* ``COMB``     the combined policy (score re-selected on val by S56's declared rule).
* ``COMB_global``  COMB's decision with one global threshold (S56's cost, restated).
* ``CAL_S56`` / ``LAM_S56``  decomposition: only calibration, or only lambda, added to S56.
* ``S56_matched``  S56 at the budget whose OOF flag rate matches COMB's at R = 0.20 -- the
  lambda rule escalates extra cases outside the abstention budget, so a like-for-like workload
  comparison is declared beside the primary.

Fit on OOF, select on val, evaluate once on reserved under `results/v4/s65/reserved_receipt.json`.
No HAM test read; no image is scored (reserved probabilities are S13/S54's frozen raw soft-votes).

    $py -m research.v4.s65_combined_policy --selftest
    $py -m research.v4.s65_combined_policy --select        # OOF fit + val selection
    $py -m research.v4.s65_combined_policy --freeze-plan   # results/v4/s65_plan.json
    $py -m research.v4.s65_combined_policy --reserved --smoke   # rehearse on val
    $py -m research.v4.s65_combined_policy --reserved      # the one reserved read
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.v4 import s56_abstention as s56

REPO_ROOT = s56.REPO_ROOT
OUT_DIR = REPO_ROOT / "results" / "v4" / "s65"
PLAN_PATH = REPO_ROOT / "results" / "v4" / "s65_plan.json"
SELECTION_PATH = OUT_DIR / "selection_val.json"
POLICY_PATH = OUT_DIR / "policy_oof.json"
RECEIPT_PATH = OUT_DIR / "reserved_receipt.json"
S56_DIR = REPO_ROOT / "results" / "v4" / "s56"
LEDGER_PATH = s56.LEDGER_PATH
SESSION = "v4_s65"

BUDGETS = s56.BUDGETS
PRIMARY_BUDGET = s56.PRIMARY_BUDGET
PRIMARY_BAND = "<40"
NI_BAND = "60+"
REPORT_BANDS = ("ALL", "<40", "40-59", "60+")
MCID = 0.05
NI_MARGIN = 0.05
SPEC_FLOOR = 0.85            # S5's lambda constraint, nominal off HAM (S57a)
MATCH_GRID = np.round(np.arange(PRIMARY_BUDGET, 0.6001, 0.005), 4)
N_BOOT = s56.N_BOOT
BOOT_SEED = s56.BOOT_SEED
REPRO_TOL = 1e-9
VIEWS = ("S56", "LAM", "CAL", "COMB")   # (calibrator, decision) pairs


# ============================================================================ views
@dataclass
class Views:
    """One split under both calibrators; `panel(view)` gives the S56 Panel for a view."""

    name: str
    glob: np.ndarray       # global deployed map (cross-fitted on OOF)
    bandcal: np.ndarray    # S55 per-band maps (cross-fitted within band on OOF)
    y7: np.ndarray
    bands: np.ndarray
    groups: np.ndarray

    def __post_init__(self) -> None:
        from research.external import frozen_params as fp

        self.bands = np.asarray(self.bands).astype(str)
        esc = s56.esc_indices()
        lam = {"glob": fp.apply_age_rule(self.glob, bands=self.bands),
               "bandcal": fp.apply_age_rule(self.bandcal, bands=self.bands)}
        spec = {"S56": ("glob", None), "LAM": ("glob", lam["glob"]),
                "CAL": ("bandcal", None), "COMB": ("bandcal", lam["bandcal"])}
        self.panels = {v: s56.Panel(self.name, getattr(self, cal), self.y7, self.bands,
                                    self.groups, esc, pred=pred)
                       for v, (cal, pred) in spec.items()}

    def panel(self, view: str) -> s56.Panel:
        return self.panels[view]


def oof_views() -> Views:
    from research.agerule import lambda_rule as lr
    from research.external import frozen_params as fp
    from research.multical import groupwise as gw
    from research.run_session55_multical import _fold_column

    raw = fp.load_ham_oof_panel(calibrate_probs=False)
    folds = _fold_column(fp.OOF_PREDICTIONS_DIR, raw.image_ids)
    glob = lr.crossfit_calibration(raw.probs_raw, raw.y_true, folds)
    bandcal = gw.crossfit_group_dirichlet(raw.probs_raw, raw.y_true, raw.bands, folds, glob)
    return Views("oof", glob, bandcal, raw.y_true, raw.bands, raw.lesion_ids.astype(str))


def val_views() -> Views:
    from research.ensembling.data import load_split_matrix
    from research.ensembling.methods import soft_vote_arithmetic
    from research.external import frozen_params as fp

    matrix = load_split_matrix("val", predictions_dir="research/predictions_tta")
    raw = soft_vote_arithmetic(matrix.probs)
    bands = fp.age_bands(fp._ages_for(matrix.image_ids))
    return Views("val", fp.calibrate(raw), fp.calibrate_by_band(raw, bands), matrix.y_true,
                 bands, matrix.lesion_ids.astype(str))


def reserved_views() -> tuple[Views, dict[str, Any]]:
    from research.external import frozen_params as fp

    panel, meta, raw = s56.reserved_panel_with_raw()   # verifies glob == deployed(raw)
    return Views("reserved", panel.probs, fp.calibrate_by_band(raw, panel.bands), panel.y7,
                 panel.bands, panel.groups), meta


# ============================================================================ policies
@dataclass
class ArmSpec:
    arm: str
    view: str
    kind: str              # "band" | "global" | "none"
    budget: float
    score: str | None


def fit_policy(views: Views, spec: ArmSpec) -> s56.Policy | None:
    if spec.kind == "none":
        return None
    fit = s56.fit_band if spec.kind == "band" else s56.fit_global
    return fit(views.panel(spec.view), spec.score, spec.budget)


def referred(views: Views, spec: ArmSpec, policy: s56.Policy | None) -> np.ndarray:
    panel = views.panel(spec.view)
    if policy is None:
        return np.zeros(len(panel), bool)
    return policy.refer(panel.score(policy.score), panel.bands)


def flag_rate(panel: s56.Panel, ref: np.ndarray, mask: np.ndarray) -> float:
    return float((panel.pred_esc | ref)[mask].mean())


def matched_budget(oof: Views, s56_score: str, target: float) -> float:
    """Smallest S56 budget on MATCH_GRID whose OOF all-ages flag rate reaches `target`."""
    panel = oof.panel("S56")
    everyone = np.ones(len(panel), bool)
    for r in MATCH_GRID:
        pol = s56.fit_band(panel, s56_score, float(r))
        if flag_rate(panel, pol.refer(panel.score(s56_score), panel.bands), everyone) >= target:
            return float(r)
    raise SystemExit(f"no S56 budget up to {MATCH_GRID[-1]} reaches OOF flag rate {target:.4f}")


def arm_specs(s56_score: str, score: str, matched: float | None) -> list[ArmSpec]:
    specs = [ArmSpec("S56_none", "S56", "none", 0.0, None),
             ArmSpec("LAM", "LAM", "none", 0.0, None),
             ArmSpec("COMB_none", "COMB", "none", 0.0, None)]
    for r in BUDGETS:
        specs += [ArmSpec("S56", "S56", "band", r, s56_score),
                  ArmSpec("COMB", "COMB", "band", r, score),
                  ArmSpec("COMB_global", "COMB", "global", r, score),
                  ArmSpec("CAL_S56", "CAL", "band", r, score),
                  ArmSpec("LAM_S56", "LAM", "band", r, score)]
    if matched is not None:
        specs.append(ArmSpec("S56_matched", "S56", "band", matched, s56_score))
    return specs


def fit_all(oof: Views, specs: list[ArmSpec]) -> dict[tuple[str, float], tuple[ArmSpec, Any]]:
    return {(s.arm, s.budget): (s, fit_policy(oof, s)) for s in specs}


INTERVAL_ARMS = ("S56_none", "LAM", "COMB_none", "S56", "COMB", "COMB_global", "S56_matched")


def frontier(views: Views, fitted: dict, *, intervals: bool, n_boot: int = N_BOOT) -> pd.DataFrame:
    rows = []
    for (arm, r), (spec, pol) in fitted.items():
        panel = views.panel(spec.view)
        ref = referred(views, spec, pol)
        ci = intervals and arm in INTERVAL_ARMS
        for row in s56.describe(panel, ref, intervals=ci, n_boot=n_boot):
            band = row["band"]
            m = np.ones(len(panel), bool) if band == "ALL" else panel.bands == band
            neg = m & ~panel.y_esc
            rows.append({"arm": arm, "view": spec.view, "kind": spec.kind, "budget": r,
                         "score": spec.score, "floor": pol.floor if pol is not None else np.nan,
                         **row,
                         "flag_rate": flag_rate(panel, ref, m),
                         "decision_esc_rate": float(panel.pred_esc[m].mean()),
                         "decision_specificity": (float(1 - panel.pred_esc[neg].mean())
                                                  if neg.any() else np.nan)})
    return pd.DataFrame(rows)


# ============================================================================ contrasts
def arm_stat(panel: s56.Panel, ref: np.ndarray, idx: np.ndarray, stat: str) -> float:
    if stat == "system_sens":
        pos = idx[panel.y_esc[idx]]
        return float((panel.pred_esc | ref)[pos].mean()) if len(pos) else float("nan")
    if stat == "referral_rate":
        return float(ref[idx].mean())
    if stat == "flag_rate":
        return float((panel.pred_esc | ref)[idx].mean())
    if stat == "retained_macro_f1":
        keep = idx[~ref[idx]]
        return s56.macro_f1(panel.y7[keep], panel.pred[keep])
    raise ValueError(stat)


def paired(views: Views, fitted: dict, a: tuple[str, float], b: tuple[str, float], band: str,
           stat: str, n_boot: int = N_BOOT) -> dict[str, Any]:
    """stat(a) - stat(b) on the same rows, one lesion-grouped resample shared by both arms."""
    from research.stats.calibration_slices import grouped_bootstrap_scalar

    (sa, pa), (sb, pb) = fitted[a], fitted[b]
    panel_a, panel_b = views.panel(sa.view), views.panel(sb.view)
    ref_a, ref_b = referred(views, sa, pa), referred(views, sb, pb)
    m = np.ones(len(views.y7), bool) if band == "ALL" else views.bands == band
    idx = np.flatnonzero(m)

    def delta(sub: np.ndarray) -> float:
        return arm_stat(panel_a, ref_a, sub, stat) - arm_stat(panel_b, ref_b, sub, stat)

    lo, hi = grouped_bootstrap_scalar(lambda local: delta(idx[local]), views.groups[idx],
                                      n_boot=n_boot, seed=BOOT_SEED)
    return {"a": f"{a[0]}@{a[1]:g}", "b": f"{b[0]}@{b[1]:g}", "band": band, "stat": stat,
            "value_a": arm_stat(panel_a, ref_a, idx, stat),
            "value_b": arm_stat(panel_b, ref_b, idx, stat),
            "delta": delta(idx), "ci_lo": float(lo), "ci_hi": float(hi)}


def contrast_plan(matched: float) -> list[tuple[str, tuple, tuple, str, str]]:
    """(label, arm a, arm b, band, stat): the declared contrast family, all descriptive but
    the two that decide (primary, non_inferiority)."""
    out = []
    for r in BUDGETS:
        for band, stat in (("<40", "system_sens"), ("<40", "referral_rate"), ("<40", "flag_rate"),
                           ("40-59", "system_sens"), ("60+", "system_sens"),
                           ("ALL", "system_sens"), ("ALL", "referral_rate"),
                           ("ALL", "flag_rate"), ("ALL", "retained_macro_f1")):
            label = ("primary" if (r == PRIMARY_BUDGET and band == PRIMARY_BAND
                                   and stat == "system_sens") else "comb_vs_s56")
            out.append((label, ("COMB", r), ("S56", r), band, stat))
        out.append(("non_inferiority" if r == PRIMARY_BUDGET else "comb_vs_lam",
                    ("COMB", r), ("LAM", 0.0), NI_BAND, "system_sens"))
        out.append(("comb_vs_lam", ("COMB", r), ("LAM", 0.0), "ALL", "system_sens"))
        out.append(("comb_vs_lam", ("COMB", r), ("LAM", 0.0), "ALL", "flag_rate"))
    r = PRIMARY_BUDGET
    for band in ("<40", "60+"):
        out.append(("band_vs_global", ("COMB", r), ("COMB_global", r), band, "system_sens"))
    out += [("decomp_calibration", ("CAL_S56", r), ("S56", r), "<40", "system_sens"),
            ("decomp_lambda", ("LAM_S56", r), ("S56", r), "<40", "system_sens"),
            ("decomp_calibration_given_lambda", ("COMB", r), ("LAM_S56", r), "<40", "system_sens"),
            ("workload_matched", ("COMB", r), ("S56_matched", matched), "<40", "system_sens"),
            ("workload_matched", ("COMB", r), ("S56_matched", matched), "ALL", "flag_rate"),
            ("workload_matched", ("COMB", r), ("S56_matched", matched), "60+", "system_sens"),
            ("workload_matched", ("COMB", r), ("S56_matched", matched), "ALL", "system_sens")]
    return out


def primary_outcome(d: dict[str, Any]) -> str:
    if d["ci_hi"] < 0:
        return "HARM"
    if d["ci_lo"] > 0:
        return "SUPPORTED" if d["delta"] >= MCID else "POSITIVE_BELOW_MCID"
    return "NOT_RESOLVED"


def ni_outcome(d: dict[str, Any]) -> str:
    if d["ci_lo"] > -NI_MARGIN:
        return "NONINFERIOR"
    if d["ci_hi"] < -NI_MARGIN:
        return "INFERIOR"
    return "INCONCLUSIVE"


def verdict(primary: str, ni: str, matched: dict[str, Any]) -> str:
    if primary != "SUPPORTED":
        return "REJECT-flat" if primary in ("NOT_RESOLVED", "POSITIVE_BELOW_MCID") else "REJECT-harm"
    if ni != "NONINFERIOR":
        return "REJECT-60plus"
    return "ADOPT" if matched["ci_lo"] > 0 else "REJECT-cost"


# ============================================================================ reproduction
def check_s56_reproduced(frame: pd.DataFrame, s56_csv: Path, role_primary: bool) -> float:
    """Our S56 arm must equal S56's published frontier row for row (score, budget, band)."""
    ref = pd.read_csv(s56_csv)
    if role_primary:
        ref = ref[ref["role"] == "primary"]
    ref = ref[ref["arm"] == "band"].set_index(["budget", "band"])
    ours = frame[frame["arm"] == "S56"].set_index(["budget", "band"])
    worst = 0.0
    for col in ("referral_rate", "system_sens", "retained_macro_f1"):
        joined = ours[[col]].join(ref[[col]], rsuffix="_s56", how="inner")
        if len(joined) != len(ref):
            raise SystemExit(f"S56 reproduction: {len(joined)} of {len(ref)} rows matched")
        worst = max(worst, float((joined[col] - joined[f"{col}_s56"]).abs().max()))
    if worst > REPRO_TOL:
        raise SystemExit(f"S56 arm does not reproduce {s56.rel(s56_csv)} (max drift {worst:.2e})")
    return worst


# ============================================================================ plan + receipt
def selection_meta() -> dict[str, Any]:
    return json.loads(SELECTION_PATH.read_text(encoding="utf-8"))


def plan_payload(sel: dict[str, Any]) -> dict[str, Any]:
    return {
        "session": "S65",
        "title": "Combined per-band triage policy: S55 calibration -> frozen lambda -> S56 abstention",
        "model": "frozen V1: 6-CNN uniform soft-vote, 24-view TTA (S54 outcome 4 keeps it)",
        "composition": [
            "per-band Dirichlet: research/multical/results_oof/fit_state.json via "
            "frozen_params.calibrate_by_band (unknown band -> deployed global map); OOF rows "
            "cross-fitted within band (research.multical.groupwise.crossfit_group_dirichlet)",
            "frozen 3-band lambda: research/agerule/results_oof/age_rule_lambda.json via "
            "frozen_params.apply_age_rule; NOT refit on the per-band probabilities (declared "
            "mismatch: fit under the global map)",
            "per-band CRC-floor abstention (research.v4.s56_abstention.fit_band) refit on OOF on "
            "the per-band-calibrated scores, silent misses defined by the lambda decision"],
        "fit_split": "HAM OOF (6,981 rows)",
        "selection_split": "HAM val (1,532 rows); S56's declared score rule applied to COMB",
        "evaluation": "manifest_v4 split=reserved (4,733 images, 104 under-40 escalating "
                      "lesions), S13/S54 frozen raw V1 soft-votes; read once",
        "budgets": list(BUDGETS),
        "primary_budget": PRIMARY_BUDGET,
        "referral_semantics": "R is the abstention budget, as in S56. Cases the lambda rule "
                              "escalates are flagged outside it; flag rate = escalated or "
                              "referred is reported for every arm and band",
        "arms": {
            "S56": "global map, argmax, S56 band abstention, S56's frozen score "
                   f"({sel['s56_score']}); must reproduce results/v4/s56 frontiers exactly",
            "LAM": "global map, frozen lambda, no referral",
            "COMB": f"per-band map, frozen lambda, band abstention, score {sel['selected']}",
            "COMB_global": "COMB decision, one global threshold",
            "CAL_S56": "per-band map, argmax, band abstention (decomposition)",
            "LAM_S56": "global map, frozen lambda, band abstention (decomposition)",
            "S56_matched": f"S56 at budget {sel['matched_budget']}, the smallest on a 0.005 grid "
                           f"whose OOF all-ages flag rate reaches COMB's at R={PRIMARY_BUDGET} "
                           f"({sel['matched_target_flag_rate']:.4f})"},
        "score_selection": {"candidates": list(s56.SCORES), "rule": sel["rule"],
                            "selected": sel["selected"],
                            "selection_file": s56.rel(SELECTION_PATH),
                            "selection_sha256": s56.sha256(SELECTION_PATH)},
        "primary_endpoint": {
            "quantity": f"under-40 system escalation sensitivity, COMB minus S56, R={PRIMARY_BUDGET}, "
                        "reserved",
            "interval": f"lesion-grouped paired bootstrap (group_id), {N_BOOT} draws, seed {BOOT_SEED}",
            "mcid": MCID,
            "outcomes": {"SUPPORTED": "delta >= MCID and CI lower bound > 0",
                         "POSITIVE_BELOW_MCID": "CI lower bound > 0 and delta < MCID",
                         "NOT_RESOLVED": "CI contains 0",
                         "HARM": "CI upper bound < 0"}},
        "non_inferiority": {
            "quantity": f"60+ system sensitivity, COMB at R={PRIMARY_BUDGET} minus LAM (frozen "
                        "lambda rule, full coverage), reserved, same paired bootstrap",
            "margin": -NI_MARGIN,
            "outcomes": {"NONINFERIOR": f"CI lower bound > -{NI_MARGIN}",
                         "INFERIOR": f"CI upper bound < -{NI_MARGIN}",
                         "INCONCLUSIVE": "otherwise"},
            "companion": "COMB band minus COMB_global on 60+ restates S56's -0.100 cost; reported"},
        "declared_cost_table": ["per-band referral rate", "per-band flag rate",
                                "per-band system sensitivity", "all-ages system sensitivity",
                                "per-band decision specificity (0.85 floor, nominal off HAM)",
                                "retained Macro-F1", "NNB at pi 0.01/0.03/0.05 (upper bound)"],
        "verdict_rule": {
            "ADOPT": "primary SUPPORTED, non-inferiority NONINFERIOR, and the workload-matched "
                     "<40 contrast (COMB - S56_matched) has CI lower bound > 0",
            "REJECT-cost": "SUPPORTED and NONINFERIOR, but the gain does not survive matching "
                           "S56's flag rate (it is bought by lambda's extra escalations)",
            "REJECT-60plus": "SUPPORTED but not NONINFERIOR",
            "REJECT-flat": "primary NOT_RESOLVED or POSITIVE_BELOW_MCID",
            "REJECT-harm": "primary HARM"},
        "carry_forward": [
            "nominal budgets do not transfer (S56: 20% nominal -> ~39-40% realised on reserved); "
            "realised referral and flag rates are reported as measured",
            "floors are nominal off HAM: S56 met 0 of 15 cells; the deployed 40-59 lambda gives "
            "specificity 0.748 on reserved (S57a)",
            "S64: calibration is ranking-neutral, so a gain should come from lambda x abstention"],
        "pre_read_observations": [
            "OOF/val only, recorded before the reserved read: the per-band map largely undoes "
            "the frozen <40 lambda (it was fit under the global map). <40 decision sensitivity "
            "LAM -> COMB_none: OOF 0.625 -> 0.438, val 0.591 -> 0.273. COMB's <40 floor is then "
            "bought by abstention: OOF <40 referral 0.608 at R=0.20 (S56: 0.317)",
            "COMB_global refers 0.046 of <40 on OOF: the band arm is what carries <40"],
        "not_done": ["no HAM test read", "no image scored", "lambda not refit",
                     "no per-hospital lambda (S66 REJECT-cost)"],
    }


def freeze_plan() -> int:
    if RECEIPT_PATH.is_file():
        raise SystemExit("reserved already read under a frozen S65 plan; refusing to rewrite it")
    if not SELECTION_PATH.is_file():
        raise SystemExit("run --select first: the plan pins the val selection")
    with PLAN_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(plan_payload(selection_meta()), indent=2, sort_keys=True))
    digest = s56.sha256(PLAN_PATH)
    print(f"wrote {s56.rel(PLAN_PATH)}\nsha256 {digest}")
    sel = selection_meta()
    write_ledger([{"method": "S65_plan", "split": "none",
                   "notes": f"S65 plan frozen before the reserved read; COMB score "
                            f"{sel['selected']} (val-selected), S56 matched budget "
                            f"{sel['matched_budget']}; primary <40 system sensitivity COMB-S56 at "
                            f"R={PRIMARY_BUDGET}, MCID {MCID}; 60+ NI margin -{NI_MARGIN}; "
                            f"sha256 {digest}"}], prune=["S65_plan"])
    return 0


def receipt_begin(rerun_reason: str | None, resume: bool) -> dict[str, Any]:
    from research.v4.s54_guard import git_head, now

    digest = s56.sha256(PLAN_PATH)
    receipt = (json.loads(RECEIPT_PATH.read_text(encoding="utf-8")) if RECEIPT_PATH.is_file()
               else {"cohort": "manifest_v4 split=reserved (S65)", "plan_sha256": digest,
                     "executions": []})
    if receipt["plan_sha256"] != digest:
        raise SystemExit("s65_plan.json changed after the reserved read; refusing")
    last = receipt["executions"][-1] if receipt["executions"] else None
    if resume:
        if last is None or last["status"] != "started":
            raise SystemExit("--resume needs an execution that started and did not complete")
        last.setdefault("resumed_at", []).append(now())
    else:
        if last is not None and last["status"] == "started":
            raise SystemExit("the last S65 reserved execution did not complete: use --resume")
        if receipt["executions"] and not rerun_reason:
            raise SystemExit("S65 has already read the reserved cohort; a repeat needs "
                             "--rerun-reason, which the receipt keeps permanently")
        receipt["executions"].append({"execution": len(receipt["executions"]) + 1,
                                      "status": "started", "started_at": now(),
                                      "rerun_reason": rerun_reason, "git_head": git_head()})
    _write_json(RECEIPT_PATH, receipt)
    return receipt


def receipt_complete(receipt: dict[str, Any], items: list[Path], summary: dict) -> None:
    from research.v4.s54_guard import now

    receipt["executions"][-1].update(status="completed", completed_at=now(), **summary,
                                     items={s56.rel(p): s56.sha256(p) for p in items})
    _write_json(RECEIPT_PATH, receipt)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, default=float))


def write_ledger(rows: list[dict[str, Any]], prune: list[str]) -> None:
    """Prune this session's rows for these methods, then append (the S20 hazard)."""
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    frame = pd.DataFrame([{"timestamp": stamp, "session": SESSION, **r} for r in rows])
    old = pd.read_csv(LEDGER_PATH, low_memory=False)
    kept = old[~((old["session"] == SESSION) & (old["method"].isin(prune)))]
    print(f"ledger: pruned {len(old) - len(kept)} prior {SESSION} row(s), appended {len(frame)}")
    pd.concat([kept, frame], ignore_index=True).to_csv(LEDGER_PATH, index=False)


# ============================================================================ runs
def _cost_table(frame: pd.DataFrame, arms: tuple[str, ...], budget: float) -> pd.DataFrame:
    sub = frame[frame["arm"].isin(arms) & frame["band"].isin(REPORT_BANDS)
                & ((frame["budget"] == budget) | (frame["kind"] == "none")
                   | (frame["arm"] == "S56_matched"))]
    cols = ["arm", "budget", "band", "n_escalating", "referral_rate", "flag_rate", "system_sens",
            "decision_specificity", "retained_macro_f1", "nnb_pi0.03"]
    return sub[cols]


def _print(frame: pd.DataFrame) -> None:
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print(frame.round(4).to_string(index=False))


def run_select() -> int:
    from research import testguard

    testguard.block_test_reads("S65 selection: OOF fit, val selection")
    oof, val = oof_views(), val_views()
    s56_sel = json.loads((S56_DIR / "selection_val.json").read_text(encoding="utf-8"))
    s56_score = s56_sel["selected"]
    selection = s56.select_score(oof.panel("COMB"), val.panel("COMB"))
    comb = s56.fit_band(oof.panel("COMB"), selection["selected"], PRIMARY_BUDGET)
    cp = oof.panel("COMB")
    target = flag_rate(cp, comb.refer(cp.score(comb.score), cp.bands), np.ones(len(cp), bool))
    matched = matched_budget(oof, s56_score, target)
    selection.update(s56_score=s56_score,
                     s56_selection_sha256=s56.sha256(S56_DIR / "selection_val.json"),
                     matched_budget=matched, matched_target_flag_rate=target)
    _write_json(SELECTION_PATH, selection)

    fitted = fit_all(oof, arm_specs(s56_score, selection["selected"], matched))
    _write_json(POLICY_PATH, {
        "lambda_by_band": __import__("research.external.frozen_params",
                                     fromlist=["x"]).load_lambda_by_band(),
        "calibration": {"per_band": "research/multical/results_oof/fit_state.json",
                        "global": "research/selective/results_oof/fit_state.json"},
        "policies": {f"{a}@{r:g}": {"view": s.view, "kind": s.kind,
                                    **(p.as_dict() if p is not None else {})}
                     for (a, r), (s, p) in fitted.items()}})
    val_frame = frontier(val, fitted, intervals=True)
    drift_val = check_s56_reproduced(val_frame, S56_DIR / "frontier_val.csv", role_primary=False)
    val_frame.to_csv(OUT_DIR / "frontier_val.csv", index=False)
    oof_frame = frontier(oof, fitted, intervals=False)
    check_s56_reproduced(oof_frame, S56_DIR / "frontier_oof_insample.csv", role_primary=False)
    oof_frame.to_csv(OUT_DIR / "frontier_oof.csv", index=False)

    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)}
                      for k, v in selection["table"].items()}, indent=1))
    print(f"COMB score: {selection['selected']} (eligible {selection['eligible']}); "
          f"S56 score {s56_score}; matched budget {matched} (OOF flag target {target:.4f}); "
          f"S56 val reproduction drift {drift_val:.1e}")
    for name, frame in (("OOF (in-sample)", oof_frame), ("val", val_frame)):
        print(f"\n--- {name}, R={PRIMARY_BUDGET}")
        _print(_cost_table(frame, ("S56_none", "LAM", "COMB_none", "S56", "COMB", "COMB_global",
                                   "CAL_S56", "LAM_S56", "S56_matched"), PRIMARY_BUDGET))
    rows = []
    for _, g in val_frame[(val_frame["band"] == "ALL")].iterrows():
        u40 = val_frame[(val_frame["arm"] == g["arm"]) & (val_frame["budget"] == g["budget"])
                        & (val_frame["band"] == "<40")].iloc[0]
        rows.append({"method": f"S65_val_{g['arm']}_R{int(round(g['budget'] * 1000))}",
                     "split": "val", "macro_f1": g["retained_macro_f1"],
                     "escalation_sens": g["system_sens"], "notes": _note(g, u40)})
    write_ledger(rows, prune=[r["method"] for r in rows])
    return 0


def _note(g: pd.Series, u40: pd.Series) -> str:
    return (f"S65 {g['arm']} ({g['view']}, {g['kind']}) R={g['budget']:g} score={g['score']}; "
            f"all-ages referral {g['referral_rate']:.4f} flag {g['flag_rate']:.4f} "
            f"system_sens {g['system_sens']:.4f} retained_F1 {g['retained_macro_f1']:.4f}; "
            f"<40 referral {u40['referral_rate']:.4f} flag {u40['flag_rate']:.4f} "
            f"system_sens {u40['system_sens']:.4f} ({u40['n_escalating']} esc)")


def run_reserved(rerun_reason: str | None, n_boot: int, *, smoke: bool, resume: bool) -> int:
    from research import testguard

    testguard.block_test_reads("S65 reserved evaluation: HAM test is never read")
    if not PLAN_PATH.is_file():
        raise SystemExit("plan not frozen: run --freeze-plan")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan["score_selection"]["selection_sha256"] != s56.sha256(SELECTION_PATH):
        raise SystemExit("selection_val.json changed after the plan pinned it")
    sel = selection_meta()
    if sel["s56_selection_sha256"] != s56.sha256(S56_DIR / "selection_val.json"):
        raise SystemExit("S56's selection changed since S65 pinned it")
    oof = oof_views()
    fitted = fit_all(oof, arm_specs(sel["s56_score"], sel["selected"], sel["matched_budget"]))

    out_dir = OUT_DIR / "smoke" if smoke else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        receipt, (views, meta) = None, (val_views(), {"smoke": "HAM val stands in"})
    else:
        receipt = receipt_begin(rerun_reason, resume)
        views, meta = reserved_views()
    frame = frontier(views, fitted, intervals=True, n_boot=n_boot)
    s56_csv = S56_DIR / ("frontier_val.csv" if smoke else "frontier_reserved.csv")
    drift = check_s56_reproduced(frame, s56_csv, role_primary=not smoke)
    frontier_path = out_dir / f"frontier_{'val' if smoke else 'reserved'}.csv"
    frame.to_csv(frontier_path, index=False)

    contrasts = pd.DataFrame([{"label": label, "budget": a[1], **paired(views, fitted, a, b, band,
                                                                         stat, n_boot)}
                              for label, a, b, band, stat in contrast_plan(sel["matched_budget"])])
    contrast_path = out_dir / f"contrasts_{'val' if smoke else 'reserved'}.csv"
    contrasts.to_csv(contrast_path, index=False)

    pick = lambda label, band="<40": contrasts[(contrasts["label"] == label)  # noqa: E731
                                               & (contrasts["band"] == band)].iloc[0].to_dict()
    primary = pick("primary")
    ni = pick("non_inferiority", NI_BAND)
    matched = pick("workload_matched")
    p_out, n_out = primary_outcome(primary), ni_outcome(ni)
    comb_rows = frame[(frame["arm"] == "COMB") & (frame["kind"] == "band")]
    floor_transfer = []
    for r in BUDGETS:
        for b in s56.FLOOR_BANDS:
            row = comb_rows[(comb_rows["budget"] == r) & (comb_rows["band"] == b)].iloc[0]
            floor = row["floor"]
            floor_transfer.append({
                "budget": r, "band": b, "nominal_floor": floor, "realised": row["system_sens"],
                "ci_lo": row["system_sens_ci_lo"], "ci_hi": row["system_sens_ci_hi"],
                "status": ("violated" if row["system_sens_ci_hi"] < floor
                           else "met" if row["system_sens"] >= floor
                           else "below_point_within_ci")})
    spec = frame[frame["arm"].isin(["LAM", "COMB_none"]) & frame["band"].isin(s56.FLOOR_BANDS)]
    report = {
        "plan_sha256": s56.sha256(PLAN_PATH), "split": views.name, "reserved": meta,
        "comb_score": sel["selected"], "s56_score": sel["s56_score"],
        "matched_budget": sel["matched_budget"],
        "s56_reproduction_max_drift": drift,
        "floors_oof": {f"{r:g}": fitted[("COMB", r)][1].floor for r in BUDGETS},
        "s56_floors_oof": {f"{r:g}": fitted[("S56", r)][1].floor for r in BUDGETS},
        "primary": {**primary, "outcome": p_out, "mcid": MCID},
        "non_inferiority": {**ni, "outcome": n_out, "margin": -NI_MARGIN},
        "workload_matched": {**matched, "positive": bool(matched["ci_lo"] > 0)},
        "verdict": verdict(p_out, n_out, matched),
        "cost_table_R20": _cost_table(frame, ("S56_none", "LAM", "COMB_none", "S56", "COMB",
                                              "COMB_global", "S56_matched"),
                                      PRIMARY_BUDGET).to_dict(orient="records"),
        "decision_specificity_vs_floor": [
            {"arm": r["arm"], "band": r["band"], "specificity": r["decision_specificity"],
             "nominal_floor": SPEC_FLOOR, "below": bool(r["decision_specificity"] < SPEC_FLOOR)}
            for _, r in spec.iterrows()],
        "floor_transfer": floor_transfer}
    report_path = out_dir / "s65_report.json"
    _write_json(report_path, report)
    if receipt is not None:
        receipt_complete(receipt, [frontier_path, contrast_path, report_path],
                         {"verdict": report["verdict"]})

    _print(pd.DataFrame(report["cost_table_R20"]))
    _print(contrasts[["label", "a", "b", "band", "stat", "value_a", "value_b", "delta",
                      "ci_lo", "ci_hi"]][contrasts["budget"] == PRIMARY_BUDGET])
    print(json.dumps({k: report[k] for k in ("primary", "non_inferiority", "workload_matched",
                                             "verdict", "s56_reproduction_max_drift")},
                     indent=1, default=float))
    if smoke:
        return 0
    rows = []
    for _, g in frame[frame["band"] == "ALL"].iterrows():
        u40 = frame[(frame["arm"] == g["arm"]) & (frame["budget"] == g["budget"])
                    & (frame["band"] == "<40")].iloc[0]
        rows.append({"method": f"S65_reserved_{g['arm']}_R{int(round(g['budget'] * 1000))}",
                     "split": "reserved", "macro_f1": g["retained_macro_f1"],
                     "escalation_sens": g["system_sens"], "notes": _note(g, u40)})
    rows.append({"method": "S65_verdict", "split": "reserved",
                 "notes": f"S65 primary <40 system sens COMB-S56 at R={PRIMARY_BUDGET} "
                          f"{primary['delta']:+.4f} [{primary['ci_lo']:+.4f}, {primary['ci_hi']:+.4f}]"
                          f" -> {p_out} (MCID {MCID}); 60+ COMB-LAM {ni['delta']:+.4f} "
                          f"[{ni['ci_lo']:+.4f}, {ni['ci_hi']:+.4f}] -> {n_out}; workload-matched "
                          f"{matched['delta']:+.4f} [{matched['ci_lo']:+.4f}, {matched['ci_hi']:+.4f}]"
                          f"; verdict {report['verdict']}; plan {report['plan_sha256'][:16]}"})
    write_ledger(rows, prune=[r["method"] for r in rows])
    return 0


# ============================================================================ self-test
def _synthetic(n: int = 3000, seed: int = 0) -> Views:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 7, n)
    logits = rng.normal(size=(n, 7))
    logits[np.arange(n), y] += rng.normal(1.5, 1.0, n)
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    tilt = np.exp(logits * 1.1) / np.exp(logits * 1.1).sum(1, keepdims=True)
    bands = rng.choice(np.array(s56.BANDS), n, p=[0.2, 0.4, 0.38, 0.02])
    return Views("syn", probs, tilt, y, bands, (np.arange(n) // 2).astype(str))


def selftest() -> int:
    from research.external import frozen_params as fp

    checks = 0
    views = _synthetic()
    esc = s56.esc_indices()

    # 1. an explicit argmax decision is identical to the default Panel
    base = s56.Panel("x", views.glob, views.y7, views.bands, views.groups, esc)
    same = s56.Panel("x", views.glob, views.y7, views.bands, views.groups, esc,
                     pred=views.glob.argmax(1))
    for score in s56.SCORES:
        assert np.array_equal(base.score(score), same.score(score))
    assert s56.fit_band(base, "msp", 0.2).as_dict() == s56.fit_band(same, "msp", 0.2).as_dict()
    checks += 1

    # 2. lambda = 0 in every band is the argmax; the S56 view is the argmax
    zero = {b: 0.0 for b in (*s56.FLOOR_BANDS, "pooled")}
    assert np.array_equal(fp.apply_age_rule(views.glob, bands=views.bands, lam=zero),
                          views.glob.argmax(1))
    assert np.array_equal(views.panel("S56").pred, views.glob.argmax(1))
    checks += 1

    # 3. the lambda views only ever add escalations, and esc_risk refers them last
    lam = views.panel("LAM")
    assert (lam.pred_esc >= views.panel("S56").pred_esc).all()
    assert (lam.score("esc_risk")[lam.pred_esc] == -1).all()
    assert (lam.score("esc_risk")[~lam.pred_esc] >= 0).all()
    checks += 1

    # 4. a paired contrast of an arm with itself is exactly zero; the CRC floor holds in-sample
    fitted = fit_all(views, arm_specs("msp", "msp", 0.25))
    for stat in ("system_sens", "referral_rate", "flag_rate", "retained_macro_f1"):
        d = paired(views, fitted, ("COMB", 0.2), ("COMB", 0.2), "<40", stat, n_boot=30)
        assert d["delta"] == 0 and d["ci_lo"] == 0 and d["ci_hi"] == 0, stat
    spec, pol = fitted[("COMB", 0.2)]
    panel = views.panel("COMB")
    ref = referred(views, spec, pol)
    for b in s56.FLOOR_BANDS:
        m = panel.bands == b
        fn = int((panel.would_miss & ~ref & m).sum())
        assert (fn + 1) / (panel.y_esc[m].sum() + 1) <= 1 - pol.floor + 1e-9
    checks += 1

    # 5. matched budget: first grid point reaching the target (synthetic lambda escalates most
    #    rows, so the target is taken from an S56 fit rather than from COMB)
    p56 = views.panel("S56")

    def fr(budget: float) -> float:
        pol56 = s56.fit_band(p56, "msp", budget)
        return flag_rate(p56, pol56.refer(p56.score("msp"), p56.bands), np.ones(len(p56), bool))

    target = fr(0.3) - 1e-3
    r = matched_budget(views, "msp", target)
    assert r <= 0.3 and fr(r) >= target
    assert r == MATCH_GRID[0] or fr(round(r - 0.005, 4)) < target
    checks += 1

    # 6. outcome, NI and verdict mapping
    assert primary_outcome({"delta": 0.06, "ci_lo": 0.01, "ci_hi": 0.1}) == "SUPPORTED"
    assert primary_outcome({"delta": 0.04, "ci_lo": 0.01, "ci_hi": 0.1}) == "POSITIVE_BELOW_MCID"
    assert primary_outcome({"delta": 0.04, "ci_lo": -0.01, "ci_hi": 0.1}) == "NOT_RESOLVED"
    assert primary_outcome({"delta": -0.04, "ci_lo": -0.1, "ci_hi": -0.01}) == "HARM"
    assert ni_outcome({"ci_lo": -0.049, "ci_hi": 0.1}) == "NONINFERIOR"
    assert ni_outcome({"ci_lo": -0.2, "ci_hi": -0.051}) == "INFERIOR"
    assert ni_outcome({"ci_lo": -0.06, "ci_hi": 0.0}) == "INCONCLUSIVE"
    assert verdict("SUPPORTED", "NONINFERIOR", {"ci_lo": 0.01}) == "ADOPT"
    assert verdict("SUPPORTED", "NONINFERIOR", {"ci_lo": -0.01}) == "REJECT-cost"
    assert verdict("SUPPORTED", "INCONCLUSIVE", {"ci_lo": 0.01}) == "REJECT-60plus"
    assert verdict("NOT_RESOLVED", "NONINFERIOR", {"ci_lo": 0.01}) == "REJECT-flat"
    checks += 1

    # 7. the declared contrast family has exactly one primary and one NI member
    labels = [c[0] for c in contrast_plan(0.25)]
    assert labels.count("primary") == 1 and labels.count("non_inferiority") == 1
    checks += 1

    print(f"selftest: {checks}/{checks} checks passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--select", action="store_true", help="OOF fit + val selection")
    mode.add_argument("--freeze-plan", action="store_true")
    mode.add_argument("--reserved", action="store_true", help="the one reserved read")
    parser.add_argument("--rerun-reason", default=None)
    parser.add_argument("--resume", action="store_true", help="finish a crashed reserved read")
    parser.add_argument("--smoke", action="store_true", help="--reserved path on HAM val")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.selftest:
        return selftest()
    if args.select:
        return run_select()
    if args.freeze_plan:
        return freeze_plan()
    return run_reserved(args.rerun_reason, args.n_boot, smoke=args.smoke, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
