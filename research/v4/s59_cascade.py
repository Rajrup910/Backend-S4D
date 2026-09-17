"""S59 (re-scoped) -- the deployable cascade and its contract, measured end-to-end.

Why re-scoped (V4 runbook §4 S59, CHANGELOG §S58/§S65/§S66/§S57b). The runbook's six-stage cascade
does not survive the verdicts:

* admissibility gate (S58) -- **fails** (PAD reject 0.378, rejects escalating lesions more often);
* domain router (S58) -- **not needed** (routed head lost to the pooled head) and misses its target;
* domain-matched head (S58) -- supported, but on a different base (frozen HAM ConvNeXt features),
  and the base model is undecided until the S53r readout;
* per-band calibration (S55) -- **not stacked**: S65 showed it undoes the frozen under-40 lambda;
* lambda(age) (S57b) and per-hospital lambda (S66) -- **REJECT-cost**, so the frozen 3-band rule.

What is left is the stack this module composes, on the frozen V1 ensemble:

    V1 probabilities (deployed global Dirichlet)
      -> decision: argmax (stack ``S56``) or the frozen 3-band lambda rule (stack ``CASC``)
      -> S56 per-band CRC-floor abstention, refit on OOF under that decision (score pinned to
         S56's val-selected ``msp``; no re-selection)
      -> optional conformal safety net (``+CONF``): RAPS bipartite alpha=0.05 (S6 state,
         reproduced bit-for-bit here) refers any retained, benign-called case whose set
         contains an escalating class.

S65's lesson is built in: a layer that adds escalations outside the abstention budget must beat
S56 **at matched workload** (OOF flag rate), or it is not adopted. Both the lambda layer and the
conformal layer are adopted only by that rule, declared before val is scored and applied on val
(all-ages, because val holds 22 under-40 escalating images -- S48).

The deliverable is the runbook's **contract**, issued on HAM (OOF floor S*, val retained
Macro-F1, val NNB, nominal budget R) and checked end-to-end on the evaluation cohort with
lesion-grouped intervals. The composed guarantee is **measured**, never a union bound.
Expectation, stated before any read: the contract does not transfer (S56 met 0/15 floors on
reserved; the deployed 40-59 lambda breaks its specificity floor there, S57a).

Status (2026-09-17): runner and **draft** plan only. The plan is not frozen and reserved is not
read until the S53r readout confirms the base model (``--freeze-plan`` refuses otherwise).

    $py -m research.v4.s59_cascade --selftest
    $py -m research.v4.s59_cascade --select            # OOF fit + val selection (no ledger)
    $py -m research.v4.s59_cascade --draft-plan        # results/v4/s59_plan.draft.json
    $py -m research.v4.s59_cascade --reserved --smoke  # rehearse on HAM val -> results/v4/s59/smoke/
    # after S53r, owner's call:
    $py -m research.v4.s59_cascade --freeze-plan --confirm-base v1
    $py -m research.v4.s59_cascade --reserved          # the one reserved read (receipt-guarded)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.v4 import s56_abstention as s56

REPO_ROOT = s56.REPO_ROOT
OUT_DIR = REPO_ROOT / "results" / "v4" / "s59"
SMOKE_DIR = OUT_DIR / "smoke"
SELECTION_PATH = OUT_DIR / "selection_val.json"
DRAFT_PLAN_PATH = REPO_ROOT / "results" / "v4" / "s59_plan.draft.json"
PLAN_PATH = REPO_ROOT / "results" / "v4" / "s59_plan.json"
RECEIPT_PATH = OUT_DIR / "reserved_receipt.json"
S56_PLAN_PATH = s56.PLAN_PATH
S53R_REPORT = REPO_ROOT / "results" / "v4" / "s53r" / "s53r_report.json"
CONFORMAL_OOF = REPO_ROOT / "research" / "conformal" / "results_oof" / "fit_state.json"
HIERARCHICAL_OOF = (REPO_ROOT / "research" / "conformal" / "results_hierarchical_oof"
                    / "fit_state.json")
LEDGER_PATH = s56.LEDGER_PATH
SESSION = "v4_s59"

BASE = "v1"
BUDGETS = s56.BUDGETS
FALLBACK_BUDGET = s56.PRIMARY_BUDGET
TARGET_FLOOR = 0.85            # the operating point: smallest budget whose OOF S* reaches this
STACK_MARGIN = 0.01            # a layer is adopted only if it beats matched workload by this (val)
REFERRAL_TOLERANCE = 0.05      # realised referral may exceed nominal R by this before NOT_MET
CONF_ALPHA = 0.05
CONF_TAG = "a05"
CONF_METHOD = "RAPS"
CONF_SEED = 42
BAND_ORDER = ("<40", "40-59", "60+", "unknown")
FLOOR_BANDS = s56.FLOOR_BANDS
REPORT_BANDS = ("ALL",) + FLOOR_BANDS
MATCH_GRID = np.round(np.arange(0.05, 0.9001, 0.005), 4)
NNB_PI = 0.03
N_BOOT = s56.N_BOOT
BOOT_SEED = s56.BOOT_SEED
VIEWS = ("ARG", "LAM")          # decision: argmax | frozen 3-band lambda
STACKS = {"S56": "ARG", "CASC": "LAM"}


# ============================================================================ conformal layer
@dataclass
class ConformalLayer:
    """RAPS bipartite (band x escalation) at alpha 0.05, rebuilt from the frozen S6 recipe.

    Probabilities use the *conformal* Dirichlet map (`frozen_params.load_dirichlet("conformal")`),
    not the deployed one: the quantiles were calibrated under it (runbook, S12). The fit must
    reproduce `results_hierarchical_oof/fit_state.json` cell-for-cell or the layer refuses.
    """

    state: Any
    k_reg: int
    penalty: float
    cal_mask_oof: np.ndarray        # OOF rows in the calibration half (in-sample for the layer)
    reproduced_cells: int = 0

    @classmethod
    def fit(cls) -> "ConformalLayer":
        from research.conformal import calibrate, hierarchical
        from research.conformal import scores as cs
        from research.external import frozen_params as fp

        frozen = json.loads(CONFORMAL_OOF.read_text(encoding="utf-8"))
        hyper = frozen["raps_hyperparameters"][CONF_TAG]
        k_reg, penalty = int(hyper["k_reg"]), float(hyper["penalty"])
        raw = fp.load_ham_oof_panel(calibrate_probs=False)
        _, cal_idx = calibrate.grouped_halves(raw.lesion_ids, seed=int(frozen["seed"]))
        probs = fp.calibrate(raw.probs_raw, fp.load_dirichlet("conformal"))
        matrix = cs.aps_scores(probs, np.random.default_rng(CONF_SEED), penalty=penalty,
                               k_reg=k_reg)
        cal_scores = cs.true_label_scores(matrix[cal_idx], raw.y_true[cal_idx])
        state = hierarchical.fit_bipartite(cal_scores, raw.y_true[cal_idx],
                                           raw.bands[cal_idx], probs.shape[1],
                                           fp.escalating_indices(), CONF_ALPHA, CONF_METHOD,
                                           BAND_ORDER)
        want = json.loads(HIERARCHICAL_OOF.read_text(encoding="utf-8"))["states"][
            f"{CONF_METHOD}_bipartite_{CONF_TAG}"]["cell_quantiles"]
        got = {f"{b}|{hierarchical.GROUP_LABELS[g]}": v
               for (b, g), v in state.cell_quantiles.items()}
        if set(got) != set(want):
            raise AssertionError(f"conformal cells differ: {sorted(got)} vs {sorted(want)}")
        for cell, value in want.items():
            if not np.isclose(float(value), float(got[cell]), rtol=0, atol=1e-12,
                              equal_nan=True):
                raise AssertionError(f"conformal cell {cell}: refit {got[cell]} vs frozen {value}")
        mask = np.zeros(len(raw.y_true), bool)
        mask[cal_idx] = True
        return cls(state, k_reg, penalty, mask, reproduced_cells=len(want))

    def sets(self, raw: np.ndarray, bands: np.ndarray) -> np.ndarray:
        from research.conformal import hierarchical
        from research.conformal import scores as cs
        from research.external import frozen_params as fp

        probs = fp.calibrate(raw, fp.load_dirichlet("conformal"))
        matrix = cs.aps_scores(probs, np.random.default_rng(CONF_SEED), penalty=self.penalty,
                               k_reg=self.k_reg)
        return hierarchical.prediction_sets_bipartite(self.state, matrix, bands)


# ============================================================================ views
@dataclass
class Views:
    """One split: deployed V1 probabilities under both decisions, plus conformal sets."""

    name: str
    probs: np.ndarray            # deployed global map (cross-fitted on OOF)
    raw: np.ndarray              # uncalibrated soft-vote (for the conformal map)
    y7: np.ndarray
    bands: np.ndarray
    groups: np.ndarray
    sets: np.ndarray | None = None
    panels: dict[str, s56.Panel] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from research.external import frozen_params as fp

        self.bands = np.asarray(self.bands).astype(str)
        esc = s56.esc_indices()
        lam = fp.apply_age_rule(self.probs, bands=self.bands)
        self.panels = {"ARG": s56.Panel(self.name, self.probs, self.y7, self.bands, self.groups,
                                        esc),
                       "LAM": s56.Panel(self.name, self.probs, self.y7, self.bands, self.groups,
                                        esc, pred=lam)}

    def attach_conformal(self, layer: ConformalLayer) -> None:
        self.sets = layer.sets(self.raw, self.bands)

    def conf_flag(self, view: str) -> np.ndarray:
        """Retained-and-benign-called rows whose set holds an escalating class."""
        if self.sets is None:
            raise RuntimeError("conformal sets not attached")
        panel = self.panels[view]
        return self.sets[:, panel.esc_idx].any(axis=1) & ~panel.pred_esc


def oof_views() -> Views:
    from research.agerule import lambda_rule as lr
    from research.external import frozen_params as fp
    from research.run_session55_multical import _fold_column

    raw = fp.load_ham_oof_panel(calibrate_probs=False)
    folds = _fold_column(fp.OOF_PREDICTIONS_DIR, raw.image_ids)
    probs = lr.crossfit_calibration(raw.probs_raw, raw.y_true, folds)
    return Views("oof", probs, raw.probs_raw, raw.y_true, raw.bands,
                 raw.lesion_ids.astype(str))


def val_views() -> Views:
    from research.ensembling.data import load_split_matrix
    from research.ensembling.methods import soft_vote_arithmetic
    from research.external import frozen_params as fp

    matrix = load_split_matrix("val", predictions_dir="research/predictions_tta")
    raw = soft_vote_arithmetic(matrix.probs)
    bands = fp.age_bands(fp._ages_for(matrix.image_ids))
    return Views("val", fp.calibrate(raw), raw, matrix.y_true, bands,
                 matrix.lesion_ids.astype(str))


def reserved_views() -> tuple[Views, dict[str, Any]]:
    panel, meta, raw = s56.reserved_panel_with_raw()   # verifies probs == deployed(raw)
    return Views("reserved", panel.probs, raw, panel.y7, panel.bands, panel.groups), meta


# ============================================================================ arms
@dataclass(frozen=True)
class Arm:
    name: str          # e.g. "CASC_CONF"
    view: str          # "ARG" | "LAM"
    budget: float      # 0 = no abstention
    conformal: bool = False

    @property
    def key(self) -> str:
        return f"{self.name}@{self.budget:g}"


def fit_arm(oof: Views, arm: Arm) -> s56.Policy | None:
    if arm.budget <= 0:
        return None
    return s56.fit_band(oof.panels[arm.view], score_name(), arm.budget)


def referred(views: Views, arm: Arm, policy: s56.Policy | None) -> np.ndarray:
    panel = views.panels[arm.view]
    ref = (np.zeros(len(panel), bool) if policy is None
           else policy.refer(panel.score(policy.score), panel.bands))
    if arm.conformal:
        ref = ref | views.conf_flag(arm.view)
    return ref


def flag_rate(panel: s56.Panel, ref: np.ndarray, mask: np.ndarray | None = None) -> float:
    flags = panel.pred_esc | ref
    return float(flags.mean() if mask is None else flags[mask].mean())


_SCORE: dict[str, str] = {}


def score_name() -> str:
    """S56's val-selected score, pinned from its frozen plan -- S59 does not re-select it."""
    if "name" not in _SCORE:
        plan = json.loads(S56_PLAN_PATH.read_text(encoding="utf-8"))
        _SCORE["name"] = plan["score_selection"]["selected"]
    return _SCORE["name"]


def operating_budget(oof: Views, view: str) -> tuple[float, dict[str, float]]:
    """Smallest budget whose OOF nominal floor S* reaches TARGET_FLOOR (fallback 0.20)."""
    floors = {str(r): s56.fit_band(oof.panels[view], score_name(), r).floor for r in BUDGETS}
    for r in BUDGETS:
        if floors[str(r)] is not None and floors[str(r)] >= TARGET_FLOOR - 1e-12:
            return float(r), floors
    return FALLBACK_BUDGET, floors


def matched_budget(oof: Views, view: str, target: float) -> float | None:
    """Smallest band-abstention budget (no conformal) whose OOF flag rate reaches `target`."""
    panel = oof.panels[view]
    for r in MATCH_GRID:
        pol = s56.fit_band(panel, score_name(), float(r))
        if flag_rate(panel, pol.refer(panel.score(pol.score), panel.bands)) >= target - 1e-12:
            return float(r)
    return None


def build_arms(oof: Views, r0: dict[str, float]) -> tuple[list[Arm], dict[str, Any]]:
    """Every arm S59 evaluates; the matched comparators are derived on OOF only."""
    arms = [Arm("V1", "ARG", 0.0), Arm("LAM", "LAM", 0.0)]
    for r in BUDGETS:
        for stack, view in STACKS.items():
            arms += [Arm(stack, view, r), Arm(f"{stack}_CONF", view, r, conformal=True)]
    fitted = {a.key: (a, fit_arm(oof, a)) for a in arms}
    matching: dict[str, Any] = {}

    def add_matched(name: str, view: str, source: Arm) -> None:
        a, pol = fitted[source.key]
        target = flag_rate(oof.panels[a.view], referred(oof, a, pol))
        m = matched_budget(oof, view, target)
        matching[name] = {"source": source.key, "oof_flag_rate_target": target, "budget": m}
        if m is not None:
            arm = Arm(name, view, m)
            fitted[arm.key] = (arm, fit_arm(oof, arm))

    # the lambda layer vs S56 alone at CASC's workload
    add_matched("S56_matched_CASC", "ARG", Arm("CASC", "LAM", r0["CASC"]))
    # the conformal layer vs its own stack at the +CONF workload
    for stack, view in STACKS.items():
        add_matched(f"{stack}_matched_CONF", view,
                    Arm(f"{stack}_CONF", view, r0[stack], conformal=True))
    return [a for a, _ in fitted.values()], {"fitted": fitted, "matching": matching}


# ============================================================================ evaluation
def frontier(views: Views, fitted: dict, *, intervals: bool, n_boot: int) -> pd.DataFrame:
    rows = []
    for key, (arm, pol) in fitted.items():
        panel = views.panels[arm.view]
        ref = referred(views, arm, pol)
        for row in s56.describe(panel, ref, intervals=intervals, n_boot=n_boot):
            m = (np.ones(len(panel), bool) if row["band"] == "ALL"
                 else panel.bands == row["band"])
            rows.append({"arm": arm.name, "key": key, "view": arm.view, "budget": arm.budget,
                         "conformal": arm.conformal,
                         "floor": pol.floor if pol is not None else np.nan, **row,
                         "flag_rate": flag_rate(panel, ref, m)})
    return pd.DataFrame(rows)


def arm_stat(views: Views, arm: Arm, ref: np.ndarray, idx: np.ndarray, stat: str) -> float:
    panel = views.panels[arm.view]
    if stat == "system_sens":
        pos = idx[panel.y_esc[idx]]
        return float((panel.pred_esc | ref)[pos].mean()) if len(pos) else float("nan")
    if stat == "referral_rate":
        return float(ref[idx].mean()) if len(idx) else float("nan")
    if stat == "flag_rate":
        return float((panel.pred_esc | ref)[idx].mean()) if len(idx) else float("nan")
    if stat == "retained_macro_f1":
        keep = idx[~ref[idx]]
        return s56.macro_f1(panel.y7[keep], panel.pred[keep])
    raise ValueError(stat)


def paired(views: Views, fitted: dict, a: str, b: str, band: str, stat: str,
           n_boot: int) -> dict[str, Any]:
    from research.stats.calibration_slices import grouped_bootstrap_scalar

    (arm_a, pol_a), (arm_b, pol_b) = fitted[a], fitted[b]
    ref_a, ref_b = referred(views, arm_a, pol_a), referred(views, arm_b, pol_b)
    m = np.ones(len(views.y7), bool) if band == "ALL" else views.bands == band
    idx = np.flatnonzero(m)

    def delta(sub: np.ndarray) -> float:
        return (arm_stat(views, arm_a, ref_a, sub, stat)
                - arm_stat(views, arm_b, ref_b, sub, stat))

    lo, hi = grouped_bootstrap_scalar(lambda local: delta(idx[local]), views.groups[idx],
                                      n_boot=n_boot, seed=BOOT_SEED)
    return {"a": a, "b": b, "band": band, "stat": stat,
            "value_a": arm_stat(views, arm_a, ref_a, idx, stat),
            "value_b": arm_stat(views, arm_b, ref_b, idx, stat),
            "delta": delta(idx), "ci_lo": float(lo), "ci_hi": float(hi)}


# ============================================================================ selection (val)
def select(oof: Views, val: Views) -> dict[str, Any]:
    """The rule, declared before val is scored (and restated in the plan):

    1. Operating budget per stack: the smallest R in BUDGETS whose OOF nominal floor S* is
       >= TARGET_FLOOR (0.85); 0.20 if none reaches it.
    2. Lambda layer: stack ``CASC`` (lambda + abstention at its R0) is adopted over ``S56``
       only if its **val all-ages** system sensitivity exceeds ``S56_matched_CASC`` (S56 at the
       budget matching CASC's OOF flag rate) by >= STACK_MARGIN. Otherwise ``S56`` -- one layer
       fewer. All-ages because val has 22 under-40 escalating images (S48 bars decisions on
       <= 25 positives); the under-40 value is reported, and decides nothing.
    3. Conformal layer: for the chosen stack, ``+CONF`` at R0 is adopted only if its val
       all-ages system sensitivity exceeds ``{stack}_matched_CONF`` by >= STACK_MARGIN.
       Otherwise conformal sets are an output field only (the S60 payload), not a decision layer.
    4. Contract terms are issued from HAM: S* from the OOF fit, M and N from val.
    """
    r0 = {stack: operating_budget(oof, view)[0] for stack, view in STACKS.items()}
    floors = {stack: operating_budget(oof, view)[1] for stack, view in STACKS.items()}
    _, meta = build_arms(oof, r0)
    fitted, matching = meta["fitted"], meta["matching"]
    table = frontier(val, fitted, intervals=False, n_boot=0)

    def val_value(key: str, band: str = "ALL", col: str = "system_sens") -> float:
        row = table[(table["key"] == key) & (table["band"] == band)]
        return float(row[col].iloc[0]) if len(row) else float("nan")

    decisions: dict[str, Any] = {}
    m_casc = matching["S56_matched_CASC"]["budget"]
    casc_key = f"CASC@{r0['CASC']:g}"
    if m_casc is None:
        lam_adopt, d_lam = False, float("nan")
    else:
        d_lam = val_value(casc_key) - val_value(f"S56_matched_CASC@{m_casc:g}")
        lam_adopt = bool(d_lam >= STACK_MARGIN)
    stack = "CASC" if lam_adopt else "S56"
    decisions["lambda_layer"] = {
        "adopted": lam_adopt, "delta_val_all": d_lam, "margin": STACK_MARGIN,
        "casc": casc_key, "comparator": (None if m_casc is None
                                         else f"S56_matched_CASC@{m_casc:g}"),
        "descriptive_delta_val_u40": (
            float("nan") if m_casc is None
            else val_value(casc_key, "<40") - val_value(f"S56_matched_CASC@{m_casc:g}", "<40"))}

    m_conf = matching[f"{stack}_matched_CONF"]["budget"]
    conf_key = f"{stack}_CONF@{r0[stack]:g}"
    if m_conf is None:
        conf_adopt, d_conf = False, float("nan")
    else:
        comp = f"{stack}_matched_CONF@{m_conf:g}"
        d_conf = val_value(conf_key) - val_value(comp)
        conf_adopt = bool(d_conf >= STACK_MARGIN)
    decisions["conformal_layer"] = {
        "adopted": conf_adopt, "delta_val_all": d_conf, "margin": STACK_MARGIN,
        "arm": conf_key,
        "comparator": None if m_conf is None else f"{stack}_matched_CONF@{m_conf:g}"}

    deployed = conf_key if conf_adopt else f"{stack}@{r0[stack]:g}"
    arm, pol = fitted[deployed]
    contract = {
        "deployed_arm": deployed,
        "nominal_budget_R": arm.budget,
        "floor_S": pol.floor,
        "frr_F": None if pol.floor is None else 1.0 - pol.floor,
        "retained_macro_f1_M": val_value(deployed, "ALL", "retained_macro_f1"),
        "nnb_pi003_N": val_value(deployed, "ALL", f"nnb_pi{NNB_PI:.2f}"),
        "referral_tolerance": REFERRAL_TOLERANCE,
        "val_realised_referral": val_value(deployed, "ALL", "referral_rate"),
    }
    return {"base": BASE, "score": score_name(), "operating_budget": r0,
            "oof_floors": floors, "matching": matching, "decisions": decisions,
            "stack": stack, "contract_terms": contract, "rule": select.__doc__.strip(),
            "val_frontier_rows": int(len(table))}


# ============================================================================ contract
def term_status(lo: float, hi: float, target: float, direction: str) -> str:
    """MET / NOT_MET / UNRESOLVED of an interval against a one-sided target."""
    if any(np.isnan(v) for v in (lo, hi, target)):
        return "UNRESOLVED"
    if direction == ">=":
        return "MET" if lo >= target else ("NOT_MET" if hi < target else "UNRESOLVED")
    return "MET" if hi <= target else ("NOT_MET" if lo > target else "UNRESOLVED")


def joint_status(statuses: list[str]) -> str:
    if any(s == "NOT_MET" for s in statuses):
        return "CONTRACT_FAILS"
    if all(s == "MET" for s in statuses):
        return "CONTRACT_HOLDS"
    return "CONTRACT_UNRESOLVED"


def contract(views: Views, fitted: dict, terms: dict[str, Any], n_boot: int) -> dict[str, Any]:
    """Every term on the same rows, measured end-to-end; a composed claim, not a union bound."""
    from research.stats import intervals as iv
    from research.stats.calibration_slices import grouped_bootstrap_scalar

    arm, pol = fitted[terms["deployed_arm"]]
    panel = views.panels[arm.view]
    ref = referred(views, arm, pol)
    caught = panel.pred_esc | ref
    rows: list[dict[str, Any]] = []
    S = terms["floor_S"]
    for band in FLOOR_BANDS:
        pos = (panel.bands == band) & panel.y_esc
        if not pos.any():
            rows.append({"term": f"sensitivity[{band}]", "status": "UNRESOLVED",
                         "note": "no escalating rows"})
            continue
        prop = iv.proportion(caught[pos], panel.groups[pos], n_boot=n_boot, seed=BOOT_SEED)
        lo, hi = prop.interval
        status = term_status(lo, hi, S, ">=")
        rows.append({"term": f"sensitivity[{band}]", "target": S, "direction": ">=",
                     "value": float(caught[pos].mean()), "ci_lo": lo, "ci_hi": hi,
                     "ci_method": prop.method_short, "status": status,
                     "n_escalating": int(pos.sum()),
                     "n_escalating_groups": int(pd.unique(panel.groups[pos]).size)})
        rows.append({"term": f"frr[{band}]", "target": 1 - S, "direction": "<=",
                     "value": float(1 - caught[pos].mean()), "ci_lo": 1 - hi, "ci_hi": 1 - lo,
                     "status": status, "note": "complement of the sensitivity term"})

    everyone = np.arange(len(panel))
    lo, hi = grouped_bootstrap_scalar(
        lambda idx: arm_stat(views, arm, ref, idx, "retained_macro_f1"), views.groups,
        n_boot=n_boot, seed=BOOT_SEED)
    M = terms["retained_macro_f1_M"]
    rows.append({"term": "retained_macro_f1[ALL]", "target": M, "direction": ">=",
                 "value": arm_stat(views, arm, ref, everyone, "retained_macro_f1"),
                 "ci_lo": float(lo), "ci_hi": float(hi), "status": term_status(lo, hi, M, ">=")})

    ref_prop = iv.proportion(ref, views.groups, n_boot=n_boot, seed=BOOT_SEED)
    R = terms["nominal_budget_R"] + terms["referral_tolerance"]
    rows.append({"term": "referral_rate[ALL]", "target": R, "direction": "<=",
                 "value": float(ref.mean()), "ci_lo": ref_prop.interval[0],
                 "ci_hi": ref_prop.interval[1], "status": term_status(*ref_prop.interval, R, "<="),
                 "note": f"nominal R {terms['nominal_budget_R']:g} + tolerance "
                         f"{terms['referral_tolerance']:g}"})

    nnb = s56.nnb(panel, np.ones(len(panel), bool), ref)[f"nnb_pi{NNB_PI:.2f}"]
    N = terms["nnb_pi003_N"]
    rows.append({"term": f"nnb_pi{NNB_PI:.2f}[ALL]", "target": N, "direction": "<=",
                 "value": nnb, "status": ("UNRESOLVED" if np.isnan(nnb) or np.isnan(N)
                                          else ("MET" if nnb <= N else "NOT_MET")),
                 "note": "point estimate only (upper bound on biopsy burden)"})
    coverage = float(1 - ref.mean())
    statuses = [r["status"] for r in rows if not r["term"].startswith("frr")]
    return {"panel": views.name, "deployed_arm": terms["deployed_arm"], "coverage": coverage,
            "terms": rows, "joint": joint_status(statuses),
            "composition": "measured end-to-end on the same rows with lesion-grouped "
                           "intervals; not a union bound",
            "expectation_declared": "CONTRACT_FAILS (S56: 0/15 floors met on reserved; "
                                    "S57a: 40-59 lambda specificity 0.748 on reserved)"}


def conformal_summary(views: Views, sel: dict[str, Any]) -> dict[str, Any]:
    """Set-level quantities of the conformal layer (payload even when not a decision layer)."""
    esc = s56.esc_indices()
    y_esc = np.isin(views.y7, esc)
    has_esc = views.sets[:, esc].any(axis=1)
    covered = views.sets[np.arange(len(views.y7)), views.y7]
    view = STACKS[sel["stack"]]
    out: dict[str, Any] = {"alpha": CONF_ALPHA, "method": f"{CONF_METHOD} bipartite"}
    for band in REPORT_BANDS:
        m = np.ones(len(views.y7), bool) if band == "ALL" else views.bands == band
        pos = m & y_esc
        out[band] = {"n": int(m.sum()), "coverage": float(covered[m].mean()),
                     "mean_set_size": float(views.sets[m].sum(1).mean()),
                     "set_frr": float(1 - has_esc[pos].mean()) if pos.any() else float("nan"),
                     "extra_flag_rate": float(views.conf_flag(view)[m].mean())}
    return out


def contrast_plan(sel: dict[str, Any]) -> list[tuple[str, str, str, str, str]]:
    r0, match = sel["operating_budget"], sel["matching"]
    dep = sel["contract_terms"]["deployed_arm"]
    out = [("deployed_vs_V1", dep, "V1@0", band, stat)
           for band in REPORT_BANDS for stat in ("system_sens", "referral_rate")]
    out += [("deployed_vs_V1", dep, "V1@0", "ALL", "retained_macro_f1")]
    if match["S56_matched_CASC"]["budget"] is not None:
        mk = f"S56_matched_CASC@{match['S56_matched_CASC']['budget']:g}"
        out += [("lambda_layer_workload_matched", f"CASC@{r0['CASC']:g}", mk, band, stat)
                for band in REPORT_BANDS for stat in ("system_sens",)]
        out += [("lambda_layer_workload_matched", f"CASC@{r0['CASC']:g}", mk, "ALL",
                 "flag_rate")]
    out += [("lambda_layer_same_budget", f"CASC@{r0['CASC']:g}", f"S56@{r0['CASC']:g}", band,
             "system_sens") for band in ("<40", "60+", "ALL")]
    stack = sel["stack"]
    mc = match[f"{stack}_matched_CONF"]["budget"]
    if mc is not None:
        out += [("conformal_layer_workload_matched", f"{stack}_CONF@{r0[stack]:g}",
                 f"{stack}_matched_CONF@{mc:g}", band, "system_sens")
                for band in ("<40", "ALL")]
    return out


# ============================================================================ plan
def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def plan_payload(sel: dict[str, Any], *, draft: bool) -> dict[str, Any]:
    return {
        "session": "S59",
        "status": "DRAFT -- not frozen; base model pending the S53r readout" if draft
                  else "FROZEN before the reserved read",
        "title": "Re-scoped cascade: frozen V1 + frozen 3-band lambda + S56 abstention "
                 "(+ conformal if it composes), and its contract measured end-to-end",
        "base": {"model": "frozen V1: 6-CNN uniform soft-vote, 24-view TTA, deployed HAM-OOF "
                          "Dirichlet (research/selective/results_oof/fit_state.json)",
                 "pending": "S53r may nominate a V4 base. A V4 base has no cross-fitted OOF "
                            "predictions, so the fit split would change -- that is a plan "
                            "revision, not a flag. --freeze-plan accepts only base v1."},
        "stages_excluded": {
            "admissibility_gate": "S58: PAD reject 0.378 (target >= 0.80); escalating lesions "
                                  "rejected more often (0.087 vs 0.067)",
            "domain_router": "S58: routed head not non-inferior to pooled; accuracy 0.932 < 0.95",
            "domain_head": "S58 H1 supported but sits on a different base (frozen HAM "
                           "ConvNeXt features); revisit only if the S53r base decision allows",
            "per_band_calibration": "S65: undoes the frozen under-40 lambda (0.409 -> 0.262 on "
                                    "reserved); stacking needs a lambda refit (a new tool)",
            "lambda_age_and_per_centre": "S57b and S66 REJECT-cost; frozen 3-band rule kept"},
        "stages": ["V1 probabilities (deployed map)",
                   "decision: argmax (S56) or frozen 3-band lambda "
                   "(frozen_params.apply_age_rule) (CASC)",
                   "S56 per-band CRC-floor abstention refit on OOF under that decision; "
                   f"score pinned to S56's val-selected '{sel['score']}'",
                   f"optional conformal safety net: {CONF_METHOD} bipartite alpha {CONF_ALPHA}, "
                   "conformal Dirichlet map, frozen S6 hyperparameters, cell quantiles asserted "
                   "equal to research/conformal/results_hierarchical_oof/fit_state.json; refers "
                   "retained benign-called rows whose set holds an escalating class"],
        "fit_split": "HAM OOF (6,981 rows), cross-fitted deployed Dirichlet; conformal on the "
                     "OOF calibration half (grouped_halves seed 42)",
        "selection_split": "HAM val (1,532 rows)",
        "evaluation": "manifest_v4 split=reserved (4,733 images, 104 under-40 escalating "
                      "lesions); frozen V1 predictions (S13 + S54 top-up); read once under "
                      f"{rel(RECEIPT_PATH)}",
        "budgets": list(BUDGETS),
        "operating_budget_rule": f"smallest R with OOF S* >= {TARGET_FLOOR}; else "
                                 f"{FALLBACK_BUDGET}",
        "selection": {k: sel[k] for k in ("operating_budget", "oof_floors", "matching",
                                          "decisions", "stack", "contract_terms", "rule")},
        "selection_file": rel(SELECTION_PATH),
        "selection_sha256": s56.sha256(SELECTION_PATH),
        "workload_matching": "matched budget = smallest band-abstention budget on a 0.005 grid "
                             "(0.05-0.90) whose OOF all-ages flag rate (decision-escalated or "
                             "referred) reaches the source arm's (S65 precedent)",
        "primary_deliverable": {
            "quantity": "the contract of the deployed arm on reserved: per-band system "
                        "sensitivity >= S and FRR <= 1-S (S = OOF nominal floor), all-ages "
                        "retained Macro-F1 >= M (val), realised referral <= R + "
                        f"{REFERRAL_TOLERANCE}, NNB(pi {NNB_PI}) <= N (val)",
            "term_status": "MET if the interval clears the target, NOT_MET if it lies wholly on "
                           "the wrong side, UNRESOLVED otherwise; NNB by point estimate",
            "joint": "CONTRACT_HOLDS iff every term MET; CONTRACT_FAILS if any NOT_MET; "
                     "else CONTRACT_UNRESOLVED",
            "composition": "measured end-to-end on the same rows; not a union bound",
            "expectation": "CONTRACT_FAILS -- S56 met 0/15 floors on reserved and the deployed "
                           "40-59 lambda breaks its specificity floor there (S57a). A failure is "
                           "the expected transfer finding, not a surprise."},
        "secondary_contrasts": [
            {"label": c[0], "a": c[1], "b": c[2], "band": c[3], "stat": c[4]}
            for c in contrast_plan(sel)],
        "interval": f"lesion-grouped (group_id) paired bootstrap, {N_BOOT} draws, seed "
                    f"{BOOT_SEED}; proportions via research.stats.intervals.proportion",
        "no_p_values": "descriptive contrasts; the contract is the deliverable, no Holm family",
        "caveats": [
            "reserved has been read 7 times in V4 (S54, S56, S57a, S57b, S58, S65, S66); this is "
            "the 8th question on a reused evaluation set",
            "HAM-calibrated budgets and floors do not transfer (S56); the contract is a HAM-issued "
            "promise checked off-distribution",
            "OOF conformal flags are in-sample for the calibration half (used only to set the "
            "matched budget)"],
        "open_decisions_for_owner": [
            "base model after S53r (v1 unless a V4 base is nominated)",
            "whether to add a target-side recalibration arm (thresholds refit on BCN/MSKCC V4 "
            "train rows scored by V1, the S66 fitting data) -- S56 says off-HAM floors need it; "
            "not built here"],
        "not_done": ["no HAM test read", "no reserved read in this session",
                     "no ledger write outside --reserved"],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def load_selection() -> dict[str, Any]:
    if not SELECTION_PATH.is_file():
        raise SystemExit("run --select first")
    return json.loads(SELECTION_PATH.read_text(encoding="utf-8"))


def draft_plan() -> int:
    write_json(DRAFT_PLAN_PATH, plan_payload(load_selection(), draft=True))
    print(f"wrote {rel(DRAFT_PLAN_PATH)} (DRAFT, not frozen)\nsha256 "
          f"{s56.sha256(DRAFT_PLAN_PATH)}")
    return 0


def freeze_plan(confirm_base: str | None) -> int:
    if RECEIPT_PATH.is_file():
        raise SystemExit("reserved already read under a frozen S59 plan; refusing to rewrite it")
    if not S53R_REPORT.is_file():
        raise SystemExit(f"{rel(S53R_REPORT)} missing: the plan waits for the S53r readout")
    if confirm_base != BASE:
        raise SystemExit(f"--confirm-base {BASE} required: the owner confirms the base model "
                         "after reading S53r")
    write_json(PLAN_PATH, plan_payload(load_selection(), draft=False))
    digest = s56.sha256(PLAN_PATH)
    print(f"wrote {rel(PLAN_PATH)}\nsha256 {digest}")
    write_ledger([{"method": "S59_plan", "split": "none",
                   "notes": f"S59 plan frozen before the reserved read; sha256 {digest}"}],
                 prune=["S59_plan"])
    return 0


# ============================================================================ receipt + ledger
def receipt_begin(rerun_reason: str | None, resume: bool) -> dict[str, Any]:
    from research.v4.s54_guard import git_head, now

    if not PLAN_PATH.is_file():
        raise SystemExit("plan not frozen: run --freeze-plan after the S53r readout")
    digest = s56.sha256(PLAN_PATH)
    receipt = (json.loads(RECEIPT_PATH.read_text(encoding="utf-8")) if RECEIPT_PATH.is_file()
               else {"cohort": "manifest_v4 split=reserved (S59)", "plan_sha256": digest,
                     "executions": []})
    if receipt["plan_sha256"] != digest:
        raise SystemExit("s59_plan.json changed after the reserved read; refusing")
    last = receipt["executions"][-1] if receipt["executions"] else None
    if resume:
        if last is None or last["status"] != "started":
            raise SystemExit("--resume needs an execution that started and did not complete")
        last.setdefault("resumed_at", []).append(now())
        write_json(RECEIPT_PATH, receipt)
        return receipt
    if last is not None and last["status"] == "started":
        raise SystemExit("the last S59 reserved execution did not complete: use --resume")
    if receipt["executions"] and not rerun_reason:
        raise SystemExit("S59 has already read the reserved cohort; a repeat needs "
                         "--rerun-reason, which the receipt keeps permanently")
    receipt["executions"].append({"execution": len(receipt["executions"]) + 1,
                                  "status": "started", "started_at": now(),
                                  "rerun_reason": rerun_reason, "git_head": git_head()})
    write_json(RECEIPT_PATH, receipt)
    return receipt


def receipt_complete(receipt: dict[str, Any], items: list[Path], summary: dict) -> None:
    from research.v4.s54_guard import now

    receipt["executions"][-1].update(status="completed", completed_at=now(), summary=summary,
                                     items={rel(p): s56.sha256(p) for p in items})
    write_json(RECEIPT_PATH, receipt)


def write_ledger(rows: list[dict[str, Any]], prune: list[str]) -> None:
    """Prune this session's rows for these methods, then append (the S20 hazard)."""
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    frame = pd.DataFrame([{"timestamp": stamp, "session": SESSION, **r} for r in rows])
    old = pd.read_csv(LEDGER_PATH, low_memory=False)
    kept = old[~((old["session"] == SESSION) & (old["method"].isin(prune)))]
    pd.concat([kept, frame], ignore_index=True).to_csv(LEDGER_PATH, index=False)


# ============================================================================ runners
def run_select() -> int:
    from research import testguard

    testguard.block_test_reads("S59 -- no HAM test read")
    oof, val = oof_views(), val_views()
    layer = ConformalLayer.fit()
    oof.attach_conformal(layer)
    val.attach_conformal(layer)
    sel = select(oof, val)
    sel["conformal"] = {"reproduced_cells": layer.reproduced_cells, "k_reg": layer.k_reg,
                        "penalty": layer.penalty}
    write_json(SELECTION_PATH, sel)
    d = sel["decisions"]
    print(f"score {sel['score']} | R0 {sel['operating_budget']} | stack {sel['stack']}")
    print(f"lambda layer adopted={d['lambda_layer']['adopted']} "
          f"(val all-ages delta {d['lambda_layer']['delta_val_all']:+.4f})")
    print(f"conformal layer adopted={d['conformal_layer']['adopted']} "
          f"(val all-ages delta {d['conformal_layer']['delta_val_all']:+.4f})")
    print(f"deployed arm {sel['contract_terms']['deployed_arm']}; wrote {rel(SELECTION_PATH)}")
    return 0


def run_reserved(rerun_reason: str | None, n_boot: int, *, smoke: bool, resume: bool) -> int:
    from research import testguard

    testguard.block_test_reads("S59 -- no HAM test read")
    sel = load_selection()
    if smoke:
        out = SMOKE_DIR
        receipt = None
    else:
        if not PLAN_PATH.is_file():
            raise SystemExit("plan not frozen: the reserved read waits for the S53r readout")
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        if plan["selection_sha256"] != s56.sha256(SELECTION_PATH):
            raise SystemExit("selection_val.json changed after the plan froze; refusing")
        out = OUT_DIR
        receipt = receipt_begin(rerun_reason, resume)

    oof = oof_views()
    layer = ConformalLayer.fit()
    oof.attach_conformal(layer)
    _, meta = build_arms(oof, sel["operating_budget"])
    fitted = meta["fitted"]
    if meta["matching"] != sel["matching"]:
        raise SystemExit("OOF refit does not reproduce the selection's matched budgets")

    if smoke:
        views, panel_meta = val_views(), {"note": "smoke: HAM val stands in for reserved; "
                                                  "M and N were issued on val, so their terms "
                                                  "are in-sample here"}
    else:
        views, panel_meta = reserved_views()
    views.attach_conformal(layer)

    table = frontier(views, fitted, intervals=True, n_boot=n_boot)
    terms = sel["contract_terms"]
    result = contract(views, fitted, terms, n_boot)
    contrasts = pd.DataFrame([{"label": lab, **paired(views, fitted, a, b, band, stat, n_boot)}
                              for lab, a, b, band, stat in contrast_plan(sel)])
    report = {"session": "S59", "smoke": smoke, "panel": views.name, "base": BASE,
              "plan_sha256": None if smoke else s56.sha256(PLAN_PATH),
              "selection_sha256": s56.sha256(SELECTION_PATH), "n_boot": n_boot,
              "panel_meta": panel_meta, "stack": sel["stack"], "contract": result,
              "conformal": conformal_summary(views, sel), "test_read": False}
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"frontier_{views.name}.csv", out / f"contrasts_{views.name}.csv",
             out / "s59_report.json"]
    table.to_csv(paths[0], index=False)
    contrasts.to_csv(paths[1], index=False)
    write_json(paths[2], report)

    print(f"[{views.name}] deployed {result['deployed_arm']} coverage {result['coverage']:.3f} "
          f"-> {result['joint']}")
    for t in result["terms"]:
        if t["term"].startswith("frr"):
            continue
        ci = (f" [{t['ci_lo']:.3f}, {t['ci_hi']:.3f}]" if "ci_lo" in t else "")
        tgt = t.get("target", float("nan"))
        print(f"  {t['term']:<26} {t.get('value', float('nan')):.3f}{ci} "
              f"{t.get('direction', '')} {tgt:.3f}  {t['status']}")
    for _, c in contrasts.iterrows():
        print(f"  {c['label']:<34} {c['a']:>22} - {c['b']:<24} {c['band']:<6} {c['stat']:<18}"
              f" {c['delta']:+.4f} [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]")

    if not smoke:
        rows = [{"method": f"S59_contract_{t['term']}", "split": "reserved",
                 "notes": f"{t.get('value')} vs {t.get('target')} -> {t['status']}"}
                for t in result["terms"]]
        rows.append({"method": "S59_contract_joint", "split": "reserved",
                     "notes": f"{result['deployed_arm']}: {result['joint']}"})
        write_ledger(rows, prune=[r["method"] for r in rows])
        receipt_complete(receipt, paths, {"joint": result["joint"],
                                          "deployed_arm": result["deployed_arm"]})
    print(f"wrote {', '.join(rel(p) for p in paths)}")
    return 0


# ============================================================================ selftest
def _synthetic(n: int = 2400, seed: int = 0) -> Views:
    rng = np.random.default_rng(seed)
    esc = s56.esc_indices()
    y = rng.integers(0, 7, n)
    logits = rng.normal(size=(n, 7))
    logits[np.arange(n), y] += 1.5
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    bands = rng.choice(["<40", "40-59", "60+"], n)
    views = Views("synthetic", probs, probs, y, bands, np.arange(n).astype(str))
    sets = probs >= 0.15
    views.sets = sets
    assert esc  # escalating classes exist
    return views


def selftest() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            failures.append(name)

    check("1 term_status >=", [term_status(0.8, 0.9, 0.75, ">="), term_status(0.6, 0.7, 0.75, ">="),
                               term_status(0.7, 0.8, 0.75, ">=")] == ["MET", "NOT_MET", "UNRESOLVED"])
    check("2 term_status <=", [term_status(0.1, 0.2, 0.25, "<="), term_status(0.3, 0.4, 0.25, "<="),
                               term_status(0.2, 0.3, 0.25, "<=")] == ["MET", "NOT_MET", "UNRESOLVED"])
    check("3 joint", [joint_status(["MET", "MET"]), joint_status(["MET", "NOT_MET"]),
                      joint_status(["MET", "UNRESOLVED"])]
          == ["CONTRACT_HOLDS", "CONTRACT_FAILS", "CONTRACT_UNRESOLVED"])

    _SCORE["name"] = "msp"
    v = _synthetic()
    flag = v.conf_flag("LAM")
    check("4 conformal flag never re-flags a decision escalation",
          not (flag & v.panels["LAM"].pred_esc).any())
    check("5 lambda decision escalates a superset of argmax",
          bool((v.panels["LAM"].pred_esc | ~v.panels["ARG"].pred_esc).all()))

    arm = Arm("CASC_CONF", "LAM", 0.2, conformal=True)
    pol = fit_arm(v, arm)
    ref = referred(v, arm, pol)
    base = referred(v, Arm("CASC", "LAM", 0.2), pol)
    check("6 +CONF refers a superset of its stack", bool((ref | ~base).all()))

    rates = []
    for r in (0.1, 0.2, 0.3):
        p = fit_arm(v, Arm("S56", "ARG", r))
        rates.append(flag_rate(v.panels["ARG"], referred(v, Arm("S56", "ARG", r), p)))
    check("7 flag rate monotone in budget", rates == sorted(rates), str(np.round(rates, 3)))
    target = rates[1]
    m = matched_budget(v, "ARG", target)
    check("8 matched budget reaches its target",
          m is not None and flag_rate(v.panels["ARG"], referred(
              v, Arm("S56", "ARG", m), fit_arm(v, Arm("S56", "ARG", m)))) >= target - 1e-12,
          f"(m={m})")
    r0, floors = operating_budget(v, "LAM")
    check("9 operating budget obeys its rule",
          r0 == FALLBACK_BUDGET or floors[str(r0)] >= TARGET_FLOOR, f"(R0={r0})")
    _SCORE.clear()
    print(f"S59 selftest: {9 - len(failures)}/9 passed")
    return 1 if failures else 0


# ============================================================================ main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--select", action="store_true", help="OOF fit + val selection")
    mode.add_argument("--draft-plan", action="store_true", help="write the unfrozen plan")
    mode.add_argument("--freeze-plan", action="store_true", help="after S53r only")
    mode.add_argument("--reserved", action="store_true", help="the one reserved read")
    parser.add_argument("--smoke", action="store_true", help="with --reserved: rehearse on val")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-reason")
    parser.add_argument("--confirm-base", choices=[BASE])
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.select:
        return run_select()
    if args.draft_plan:
        return draft_plan()
    if args.freeze_plan:
        return freeze_plan(args.confirm_base)
    return run_reserved(args.rerun_reason, args.n_boot, smoke=args.smoke, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
