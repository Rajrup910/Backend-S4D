"""S56 -- group-conditional selective abstention: one referral threshold per age band.

The problem (V4 runbook §4, S56). Abstention has only ever used **one global threshold** on an
uncertainty score. Under-40 misses are *confidently* wrong, so a global threshold refers few of
them (S4: 11.1% against 33-39% in older bands). Raising the global abstention rate lifts retained
Macro-F1 and leaves the under-40 misses in the retained set.

The mechanism. For a global referral budget ``R`` (a fraction of all cases), fit one score
threshold per age band so that the **system escalation sensitivity** is at least a common floor
``S`` in *every* band, where

    system sensitivity(band) = |{i escalating : pred_i escalates  OR  i is referred}| / |{i escalating}|

A referred case goes to a clinician, so it is not a silent miss. The complement is the
**false-reassurance rate** (FRR): escalating cases that were retained *and* called benign -- the
same event `research/session9/plan.py` defines for conformal sets, restated for abstention.

Fitting, per budget ``R`` (all on OOF, cross-fitted Dirichlet probabilities):

1. For each fitted band ``b`` and floor ``S``, ``k_b(S)`` is the fewest referrals (highest score
   first) such that the **conformal-risk-control bound** holds:
   ``(FN_b(k) + 1) / (P_b + 1) <= 1 - S`` (Angelopoulos et al. 2024; the ``+1`` is the
   finite-sample correction, and it is exact only under exchangeability, which reserved is not --
   so the floor is *nominal* there and its transfer is measured, never asserted).
2. ``S*`` is the largest ``S`` on a 0.005 grid with ``sum_b k_b(S) <= R * N``: the highest floor
   the budget can buy in every band at once (max-min).
3. The unspent budget is allocated by one global threshold over the rows not yet referred, so both
   arms spend the same nominal budget and differ only in *where* they spend it.
4. Deployed rule: refer ``i`` iff ``u_i >= min(tau_band(i), tau_global)``. The ``unknown`` band
   has no floor and uses ``tau_global`` alone (38 OOF rows, too few to certify anything).

The comparator (arm ``global``) is the status quo: one threshold at the OOF ``1 - R`` quantile.

Where things are fit (runbook §10): thresholds on OOF; the uncertainty **score** is selected on
HAM val by a rule declared in the plan before val is scored; the reserved cohort (4,733 BCN +
MSKCC images, 104 under-40 escalating lesions) is evaluation only, read once under
`results/v4/s56/reserved_receipt.json`. The model is the frozen V1 ensemble (S54 outcome 4: no
representation beat it), reserved probabilities are S13/S54's frozen files -- no image is scored.

    $py -m research.v4.s56_abstention --selftest
    $py -m research.v4.s56_abstention --select        # val selection -> results/v4/s56/selection_val.json
    $py -m research.v4.s56_abstention --freeze-plan   # results/v4/s56_plan.json (pins the selection)
    $py -m research.v4.s56_abstention --reserved      # the one reserved read (receipt-guarded)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "v4" / "s56"
PLAN_PATH = REPO_ROOT / "results" / "v4" / "s56_plan.json"
SELECTION_PATH = OUT_DIR / "selection_val.json"
RECEIPT_PATH = OUT_DIR / "reserved_receipt.json"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"
SESSION = "v4_s56"

BUDGETS = (0.10, 0.15, 0.20, 0.25, 0.30)
PRIMARY_BUDGET = 0.20
PRIMARY_BAND = "<40"
FLOOR_BANDS = ("<40", "40-59", "60+")
BANDS = ("<40", "40-59", "60+", "unknown")
SCORES = ("msp", "margin", "entropy", "esc_risk")
FLOOR_GRID = np.round(np.arange(0.0, 1.0001, 0.005), 4)
PREVALENCES = (0.01, 0.03, 0.05)
MCID = 0.10               # S48's primary-endpoint MCID, in the same unit
F1_NONINFERIORITY = 0.02  # selection guard, see `select_score`
SECONDARY_SCORE = "esc_risk"  # declared descriptive arm (see plan)
N_BOOT = 2000
BOOT_SEED = 42
EPS = 1e-12


# ============================================================================ scores
def uncertainty(name: str, probs: np.ndarray, esc_idx: list[int],
                pred_esc: np.ndarray | None = None) -> np.ndarray:
    """Higher = refer first. `esc_risk` is escalation mass on cases the decision calls benign.

    A case the decision already escalates is not a silent miss whatever its score, so `esc_risk`
    gives it -1 (referred last). The decision is the argmax unless `pred_esc` says otherwise
    (S65 passes the lambda rule's). The three classical scores are the S4 candidates.
    """
    from research.selective import scores as sc

    # `research.selective.scores` already returns "higher = less trustworthy"; do not invert
    if name == "msp":
        return sc.max_softmax(probs)
    if name == "margin":
        return sc.top_two_margin(probs)
    if name == "entropy":
        return sc.predictive_entropy(probs)
    if name == "esc_risk":
        mass = probs[:, esc_idx].sum(axis=1)
        if pred_esc is None:
            pred_esc = np.isin(probs.argmax(1), esc_idx)
        return np.where(pred_esc, -1.0, mass)
    raise ValueError(f"unknown score {name!r}")


# ============================================================================ panels
@dataclass
class Panel:
    name: str
    probs: np.ndarray          # calibrated, (N, 7)
    y7: np.ndarray
    bands: np.ndarray          # str
    groups: np.ndarray         # lesion / group id for the bootstrap
    esc_idx: list[int]
    pred: np.ndarray | None = None   # hard decision; None = argmax (S65 passes the lambda rule)

    def __post_init__(self) -> None:
        self.bands = np.asarray(self.bands).astype(str)
        self.pred = self.probs.argmax(1) if self.pred is None else np.asarray(self.pred, int)
        self.y_esc = np.isin(self.y7, self.esc_idx)
        self.pred_esc = np.isin(self.pred, self.esc_idx)
        #: the rows a referral can rescue: escalating, and called benign by the argmax
        self.would_miss = self.y_esc & ~self.pred_esc

    def __len__(self) -> int:
        return len(self.y7)

    def score(self, name: str) -> np.ndarray:
        return uncertainty(name, self.probs, self.esc_idx, self.pred_esc)


def esc_indices() -> list[int]:
    from research.external import frozen_params as fp

    return fp.escalating_indices()


def oof_panel() -> Panel:
    """HAM OOF, cross-fitted Dirichlet: no row is scored by a map that saw it (S5/S55 rule)."""
    from research.agerule import lambda_rule as lr
    from research.external import frozen_params as fp
    from research.run_session55_multical import _fold_column

    raw = fp.load_ham_oof_panel(calibrate_probs=False)
    folds = _fold_column(fp.OOF_PREDICTIONS_DIR, raw.image_ids)
    probs = lr.crossfit_calibration(raw.probs_raw, raw.y_true, folds)
    return Panel("oof", probs, raw.y_true, raw.bands, raw.lesion_ids.astype(str), esc_indices())


def val_panel() -> Panel:
    """HAM val, 24-view TTA soft-vote + the deployed map (fit on OOF, never on val)."""
    from research.ensembling.data import load_split_matrix
    from research.ensembling.methods import soft_vote_arithmetic
    from research.external import frozen_params as fp

    matrix = load_split_matrix("val", predictions_dir="research/predictions_tta")
    probs = fp.calibrate(soft_vote_arithmetic(matrix.probs))
    bands = fp.age_bands(fp._ages_for(matrix.image_ids))
    return Panel("val", probs, matrix.y_true, bands, matrix.lesion_ids.astype(str),
                 esc_indices())


def reserved_panel() -> tuple[Panel, dict[str, Any]]:
    panel, meta, _ = reserved_panel_with_raw()
    return panel, meta


def reserved_panel_with_raw() -> tuple[Panel, dict[str, Any], np.ndarray]:
    """manifest_v4 reserved + the frozen deployed-V1 probabilities S54 gated with.

    Reuses S54's loaders (cohort assertion, label agreement, exactly-once coverage), then checks
    that each stored calibrated row is the deployed map applied to its stored raw soft-vote --
    the files carry both, and a mismatch would mean a different calibrator.
    """
    from research.external import frozen_params as fp
    from research.v4 import s54_gate as g

    panel = g.build_panel("reserved")
    cohort = g.check_cohort(panel)
    probs, covered, sources = g.load_v1_probs(panel, smoke=False)
    assert covered.all()
    codes = g.class_codes()
    raw_parts = []
    for path in sources:
        frame = pd.read_csv(REPO_ROOT / path, usecols=["image_id"] + [f"p_raw_{c}" for c in codes])
        raw_parts.append(frame.assign(image_id=frame["image_id"].astype(str)))
    raw = pd.concat(raw_parts).drop_duplicates("image_id").set_index("image_id")
    raw = raw.loc[panel["image_id"], [f"p_raw_{c}" for c in codes]].to_numpy(float)
    drift = float(np.abs(fp.calibrate(raw) - probs).max())
    if drift > 1e-6:
        raise ValueError(f"reserved V1 probabilities are not the deployed map (max drift {drift:.2e})")
    meta = {"cohort": cohort, "sources": sources, "calibration_max_abs_drift": drift}
    return Panel("reserved", probs, panel["y7"].to_numpy(), panel["age_band"].to_numpy(),
                 panel["group_id"].to_numpy(), esc_indices()), meta, raw


# ============================================================================ fitting
def refer_count_for_floor(u: np.ndarray, would_miss: np.ndarray, n_pos: int,
                          floor: float) -> int:
    """Fewest top-score referrals in one band with (FN + 1) / (P + 1) <= 1 - floor.

    Returns ``len(u) + 1`` when even referring the whole band cannot satisfy the bound (with
    FN = 0 the bound is 1/(P+1), so a floor above P/(P+1) is unreachable at any budget).
    """
    if n_pos == 0:
        return 0
    order = np.argsort(-u, kind="stable")
    # fn_after[k] = misses left in the retained set after referring the top k rows
    fn_after = int(would_miss.sum()) - np.concatenate([[0], np.cumsum(would_miss[order])])
    ok = (fn_after + 1) / (n_pos + 1) <= (1.0 - floor) + 1e-12
    return int(np.argmax(ok)) if ok.any() else len(u) + 1


def threshold_for_count(u: np.ndarray, k: int) -> float:
    """Score cut-off referring (at least) the top k rows: refer iff u >= threshold."""
    if k <= 0:
        return float("inf")
    if k >= len(u):
        return float("-inf")
    return float(np.sort(u)[::-1][k - 1])


@dataclass
class Policy:
    arm: str                   # "band" | "global"
    score: str
    budget: float
    floor: float | None        # S*, nominal (band arm only)
    tau_band: dict[str, float]
    tau_global: float
    fit_counts: dict[str, int]

    def thresholds(self, bands: np.ndarray) -> np.ndarray:
        band_tau = np.array([self.tau_band.get(b, np.inf) for b in bands])
        return np.minimum(band_tau, self.tau_global)

    def refer(self, u: np.ndarray, bands: np.ndarray) -> np.ndarray:
        return u >= self.thresholds(bands)

    def as_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "score": self.score, "budget": self.budget, "floor": self.floor,
                "tau_band": self.tau_band, "tau_global": self.tau_global,
                "fit_counts": self.fit_counts}


def fit_global(panel: Panel, score: str, budget: float) -> Policy:
    u = panel.score(score)
    k = int(round(budget * len(u)))
    return Policy("global", score, budget, None, {}, threshold_for_count(u, k), {"ALL": k})


def fit_band(panel: Panel, score: str, budget: float) -> Policy:
    u = panel.score(score)
    total = int(round(budget * len(u)))
    per_band = {b: panel.bands == b for b in FLOOR_BANDS}

    def counts(floor: float) -> dict[str, int]:
        return {b: refer_count_for_floor(u[m], panel.would_miss[m], int(panel.y_esc[m].sum()), floor)
                for b, m in per_band.items()}

    floor, chosen = 0.0, counts(0.0)
    for s in FLOOR_GRID:             # monotone in s, so the last feasible grid point is S*
        c = counts(float(s))
        if sum(c.values()) > total:
            break
        floor, chosen = float(s), c
    tau_band = {b: threshold_for_count(u[per_band[b]], k) for b, k in chosen.items()}

    # spend what is left with one global threshold over the rows not already referred
    base = u >= np.array([tau_band.get(b, np.inf) for b in panel.bands])
    left = total - int(base.sum())
    rest = u[~base]
    tau_global = threshold_for_count(rest, left) if left > 0 else float("inf")
    return Policy("band", score, budget, floor, tau_band, tau_global,
                  {**chosen, "global_topup": max(left, 0)})


# ============================================================================ evaluation
def macro_f1(y: np.ndarray, pred: np.ndarray, k: int = 7) -> float:
    from research.v4.s54_gate import macro_f1 as mf

    return mf(y, pred, k) if len(y) else float("nan")


def system_pred(panel: Panel, referred: np.ndarray) -> np.ndarray:
    """Hard prediction with every referral treated as an escalation (for NNB's flagged set)."""
    pred = panel.pred.copy()
    pred[referred & ~panel.pred_esc] = panel.esc_idx[0]
    return pred


def nnb(panel: Panel, mask: np.ndarray, referred: np.ndarray) -> dict[str, float]:
    """NNB of the flagged set (argmax-escalated or referred), re-weighted to each prevalence.

    Counting every referral as a biopsy makes this an **upper bound** on biopsy burden.
    """
    from research.session9 import nnb as nb

    if not panel.y_esc[mask].any():
        return {f"nnb_pi{p:.2f}": float("nan") for p in PREVALENCES}
    burden = nb.biopsy_burden(panel.y7[mask], system_pred(panel, referred)[mask], panel.esc_idx,
                              cohort=panel.name, rule="s56")
    return {f"nnb_pi{p:.2f}": nb.nnb_at(burden, p) for p in PREVALENCES}


def describe(panel: Panel, referred: np.ndarray, *, intervals: bool,
             n_boot: int = N_BOOT) -> list[dict[str, Any]]:
    """One row per band and ALL: the S56 deliverable columns."""
    from research.stats import intervals as iv

    rows = []
    for band in ("ALL",) + BANDS:
        m = np.ones(len(panel), bool) if band == "ALL" else panel.bands == band
        if not m.any():
            continue
        pos = m & panel.y_esc
        caught = (panel.pred_esc | referred)[pos]
        keep = m & ~referred
        kept_pos = keep & panel.y_esc
        row = {"panel": panel.name, "band": band, "n": int(m.sum()),
               "n_escalating": int(pos.sum()),
               "n_escalating_groups": int(pd.unique(panel.groups[pos]).size),
               "referral_rate": float(referred[m].mean()),
               "n_referred": int(referred[m].sum()),
               "system_sens": float(caught.mean()) if pos.any() else float("nan"),
               "frr": float(1 - caught.mean()) if pos.any() else float("nan"),
               "n_false_reassured": int((~caught).sum()),
               "retained_sens": (float(panel.pred_esc[kept_pos].mean())
                                 if kept_pos.any() else float("nan")),
               "retained_macro_f1": macro_f1(panel.y7[keep], panel.pred[keep]),
               "rescued_misses": int((panel.would_miss & referred & m).sum()),
               "argmax_misses": int((panel.would_miss & m).sum()),
               **nnb(panel, m, referred)}
        if intervals and pos.any():
            prop = iv.proportion(caught, panel.groups[pos], n_boot=n_boot, seed=BOOT_SEED)
            row.update(system_sens_ci_lo=prop.interval[0], system_sens_ci_hi=prop.interval[1],
                       system_sens_ci_method=prop.method_short)
        rows.append(row)
    return rows


def paired_delta(panel: Panel, a: np.ndarray, b: np.ndarray, band: str,
                 stat: str, n_boot: int = N_BOOT) -> dict[str, Any]:
    """stat(arm a) - stat(arm b) with one lesion-grouped resample shared by both arms."""
    from research.stats.calibration_slices import grouped_bootstrap_scalar

    m = np.ones(len(panel), bool) if band == "ALL" else panel.bands == band
    idx_band = np.flatnonzero(m)

    def value(ref: np.ndarray, idx: np.ndarray) -> float:
        if stat == "system_sens":
            pos = idx[panel.y_esc[idx]]
            return float((panel.pred_esc | ref)[pos].mean()) if len(pos) else float("nan")
        if stat == "referral_rate":
            return float(ref[idx].mean())
        if stat == "retained_macro_f1":
            keep = idx[~ref[idx]]
            return macro_f1(panel.y7[keep], panel.pred[keep])
        raise ValueError(stat)

    point = value(a, idx_band) - value(b, idx_band)
    lo, hi = grouped_bootstrap_scalar(
        lambda local: value(a, idx_band[local]) - value(b, idx_band[local]),
        panel.groups[idx_band], n_boot=n_boot, seed=BOOT_SEED)
    return {"band": band, "stat": stat, "delta": float(point), "ci_lo": float(lo),
            "ci_hi": float(hi)}


def fit_all(fit: Panel, score: str) -> dict[tuple[str, float], Policy]:
    out = {}
    for r in BUDGETS:
        out[("band", r)] = fit_band(fit, score, r)
        out[("global", r)] = fit_global(fit, score, r)
    return out


def frontier(eval_panel: Panel, policies: dict[tuple[str, float], Policy], *,
             intervals: bool, n_boot: int = N_BOOT) -> pd.DataFrame:
    rows = []
    base = np.zeros(len(eval_panel), bool)
    for row in describe(eval_panel, base, intervals=intervals, n_boot=n_boot):
        rows.append({"arm": "none", "budget": 0.0, "floor": np.nan, **row})
    for (arm, r), pol in policies.items():
        ref = pol.refer(eval_panel.score(pol.score), eval_panel.bands)
        for row in describe(eval_panel, ref, intervals=intervals, n_boot=n_boot):
            rows.append({"arm": arm, "budget": r, "floor": pol.floor, "score": pol.score, **row})
    return pd.DataFrame(rows)


# ============================================================================ selection (val)
def select_score(fit: Panel, val: Panel) -> dict[str, Any]:
    """The rule, declared before val was scored (and restated in the plan):

    Among the four scores, under the **band** arm, pick the highest val **all-ages** system
    escalation sensitivity averaged over the five budgets -- all-ages because val has 22
    under-40 escalating cases and S48 retires <=25-positive quantities from any decision --
    among scores whose mean retained Macro-F1 is within ``F1_NONINFERIORITY`` of the best
    score's. Ties (to 1e-9) go to the earlier name in ``SCORES``.
    """
    table = {}
    for score in SCORES:
        pols = fit_all(fit, score)
        frame = frontier(val, {k: v for k, v in pols.items() if k[0] == "band"}, intervals=False)
        all_rows = frame[(frame["arm"] == "band") & (frame["band"] == "ALL")]
        u40 = frame[(frame["arm"] == "band") & (frame["band"] == PRIMARY_BAND)]
        table[score] = {"mean_system_sens_all": float(all_rows["system_sens"].mean()),
                        "mean_retained_macro_f1": float(all_rows["retained_macro_f1"].mean()),
                        "mean_referral_rate": float(all_rows["referral_rate"].mean()),
                        "descriptive_mean_system_sens_u40": float(u40["system_sens"].mean()),
                        "oof_floors": {str(r): pols[("band", r)].floor for r in BUDGETS}}
    best_f1 = max(t["mean_retained_macro_f1"] for t in table.values())
    eligible = [s for s in SCORES
                if table[s]["mean_retained_macro_f1"] >= best_f1 - F1_NONINFERIORITY]
    chosen = max(eligible, key=lambda s: (round(table[s]["mean_system_sens_all"], 9),
                                          -SCORES.index(s)))
    return {"selected": chosen, "eligible": eligible, "table": table,
            "rule": select_score.__doc__.strip()}


# ============================================================================ plan + receipt
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def plan_payload(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "session": "S56",
        "title": "Group-conditional selective abstention (per-age-band referral thresholds)",
        "model": "frozen V1: 6-CNN uniform soft-vote, 24-view TTA, deployed HAM-OOF Dirichlet "
                 "(research/selective/results_oof/fit_state.json); S54 outcome 4 keeps it",
        "fit_split": "HAM OOF (6,981 rows), cross-fitted Dirichlet",
        "selection_split": "HAM val (1,532 rows)",
        "evaluation": "manifest_v4 split=reserved (4,733 images, 104 under-40 escalating "
                      "lesions); frozen V1 predictions (S13 + S54 top-up); read once",
        "budgets": list(BUDGETS),
        "primary_budget": PRIMARY_BUDGET,
        "arms": {"global": "one threshold at the OOF (1-R) score quantile (status quo)",
                 "band": "per-band CRC floor thresholds, max-min S*, leftover spent globally"},
        "referral_semantics": "a referred case is not a silent miss; system sensitivity = "
                              "(argmax-escalated or referred) / escalating; FRR = 1 - that",
        "crc_bound": "(FN_b + 1) / (P_b + 1) <= 1 - S, per band, on OOF",
        "floor_grid_step": 0.005,
        "floor_bands": list(FLOOR_BANDS),
        "unknown_band": "no floor, global threshold only",
        "score_selection": {"candidates": list(SCORES), "rule": selection["rule"],
                            "noninferiority": F1_NONINFERIORITY,
                            "selected": selection["selected"],
                            "selection_file": rel(SELECTION_PATH),
                            "selection_sha256": sha256(SELECTION_PATH)},
        "primary_endpoint": {
            "quantity": "under-40 system escalation sensitivity, band arm minus global arm, "
                        f"at R = {PRIMARY_BUDGET}, same score, reserved",
            "interval": "lesion-grouped paired bootstrap (group_id), one resample for both arms, "
                        f"{N_BOOT} draws, seed {BOOT_SEED}",
            "mcid": MCID,
            "outcomes": {
                "SUPPORTED": "delta >= MCID and CI lower bound > 0",
                "POSITIVE_BELOW_MCID": "CI lower bound > 0 and delta < MCID",
                "NOT_RESOLVED": "CI contains 0",
                "HARM": "CI upper bound < 0"}},
        "cost_reported_with_endpoint": [
            "under-40 referral rate, band - global (paired CI)",
            "all-ages referral rate, band - global (paired CI) -- the nominal budget is fit on "
            "OOF and the realised rate on reserved is measured, not assumed",
            "retained Macro-F1, band - global (paired CI)"],
        "floor_transfer_check": "per fitted band, realised reserved system sensitivity with its "
                                "interval against the nominal OOF floor S*; 'violated' when the "
                                "interval's upper bound is below S*",
        "per_budget_columns": ["retained_macro_f1", "system_sens (per band)", "frr (per band)",
                               "referral_rate (per band)", "nnb at pi 0.01/0.03/0.05 "
                               "(upper bound: every referral counted as a biopsy)"],
        "proportion_intervals": "research.stats.intervals.proportion (Clopper-Pearson below 30 "
                                "events, lesion-grouped bootstrap otherwise)",
        "secondary_descriptive": {
            "score": SECONDARY_SCORE,
            "why": "highest val system sensitivity but fails the Macro-F1 guard, so not "
                   "selected; its reserved frontier and <40 band-global contrast are reported "
                   "beside the primary and decide nothing"},
        "not_done": ["no HAM test read", "no reserved re-scoring of images",
                     "S55 per-band calibration is not adopted here: the deployed global map is "
                     "held fixed so the contrast isolates the abstention rule"],
    }


def freeze_plan() -> int:
    if RECEIPT_PATH.is_file():
        raise SystemExit("reserved already read under a frozen S56 plan; refusing to rewrite it")
    if not SELECTION_PATH.is_file():
        raise SystemExit("run --select first: the plan pins the val selection")
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    with PLAN_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(plan_payload(selection), indent=2, sort_keys=True))
    digest = sha256(PLAN_PATH)
    print(f"wrote {rel(PLAN_PATH)}\nsha256 {digest}")
    write_ledger([{"method": "S56_plan", "split": "none",
                   "notes": f"S56 plan frozen before the reserved read; score "
                            f"{selection['selected']} (val-selected); primary <40 system "
                            f"sensitivity band-global at R={PRIMARY_BUDGET}, MCID {MCID}; "
                            f"sha256 {digest}"}], prune=["S56_plan"])
    return 0


def receipt_begin(rerun_reason: str | None, resume: bool = False) -> dict[str, Any]:
    from research.v4.s54_guard import git_head, now

    if not PLAN_PATH.is_file():
        raise SystemExit("plan not frozen: run --freeze-plan")
    digest = sha256(PLAN_PATH)
    receipt = (json.loads(RECEIPT_PATH.read_text(encoding="utf-8")) if RECEIPT_PATH.is_file()
               else {"cohort": "manifest_v4 split=reserved (S56)", "plan_sha256": digest,
                     "executions": []})
    if receipt["plan_sha256"] != digest:
        raise SystemExit("s56_plan.json changed after the reserved read; refusing")
    last = receipt["executions"][-1] if receipt["executions"] else None
    if resume:
        # a crash before `receipt_complete` printed or wrote nothing; resuming is not a rerun
        if last is None or last["status"] != "started":
            raise SystemExit("--resume needs an execution that started and did not complete")
        last.setdefault("resumed_at", []).append(now())
        _write_receipt(receipt)
        return receipt
    if last is not None and last["status"] == "started":
        raise SystemExit("the last S56 reserved execution did not complete: use --resume")
    if receipt["executions"] and not rerun_reason:
        raise SystemExit("S56 has already read the reserved cohort; a repeat needs "
                         "--rerun-reason, which the receipt keeps permanently")
    entry = {"execution": len(receipt["executions"]) + 1, "status": "started",
             "started_at": now(), "rerun_reason": rerun_reason, "git_head": git_head()}
    receipt["executions"].append(entry)
    _write_receipt(receipt)
    return receipt


def receipt_complete(receipt: dict[str, Any], items: list[Path]) -> None:
    from research.v4.s54_guard import now

    entry = receipt["executions"][-1]
    entry.update(status="completed", completed_at=now(),
                 items={rel(p): sha256(p) for p in items})
    _write_receipt(receipt)


def _write_receipt(receipt: dict[str, Any]) -> None:
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RECEIPT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(receipt, indent=2))


def write_ledger(rows: list[dict[str, Any]], prune: list[str]) -> None:
    """Prune this session's rows for these methods, then append (the S20 hazard)."""
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    frame = pd.DataFrame([{"timestamp": stamp, "session": SESSION, **r} for r in rows])
    old = pd.read_csv(LEDGER_PATH, low_memory=False)
    kept = old[~((old["session"] == SESSION) & (old["method"].isin(prune)))]
    print(f"ledger: pruned {len(old) - len(kept)} prior {SESSION} row(s), appended {len(frame)}")
    pd.concat([kept, frame], ignore_index=True).to_csv(LEDGER_PATH, index=False)


# ============================================================================ runs
def run_select() -> int:
    from research import testguard

    testguard.block_test_reads("S56 selection: OOF fit, val selection")
    fit, val = oof_panel(), val_panel()
    selection = select_score(fit, val)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SELECTION_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(selection, indent=2, sort_keys=True))
    policies = fit_all(fit, selection["selected"])
    (OUT_DIR / "policies_oof.json").write_text(
        json.dumps({f"{a}_{r}": p.as_dict() for (a, r), p in policies.items()}, indent=2,
                   default=float), encoding="utf-8")
    val_frame = frontier(val, policies, intervals=True)
    val_frame.to_csv(OUT_DIR / "frontier_val.csv", index=False)
    frontier(fit, policies, intervals=False).to_csv(OUT_DIR / "frontier_oof_insample.csv",
                                                    index=False)
    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)}
                      for k, v in selection["table"].items()}, indent=1))
    print(f"selected: {selection['selected']} (eligible {selection['eligible']})")
    _print_frontier(val_frame)
    rows = [{"method": f"S56_val_{a}_R{int(r * 100)}", "split": "val",
             "macro_f1": g["retained_macro_f1"], "escalation_sens": g["system_sens"],
             "notes": _note(g, val_frame, a, r)}
            for (a, r), g in _all_rows(val_frame).items()]
    write_ledger(rows, prune=[r["method"] for r in rows])
    return 0


def _all_rows(frame: pd.DataFrame) -> dict[tuple[str, float], pd.Series]:
    sub = frame[(frame["band"] == "ALL") & (frame["arm"] != "none")]
    return {(row["arm"], row["budget"]): row for _, row in sub.iterrows()}


def _note(row: pd.Series, frame: pd.DataFrame, arm: str, budget: float) -> str:
    u40 = frame[(frame["arm"] == arm) & (frame["budget"] == budget) & (frame["band"] == "<40")]
    u40 = u40.iloc[0]
    return (f"S56 {arm} arm R={budget:.2f} score={row['score']} floor={row['floor']}; "
            f"all-ages referral {row['referral_rate']:.4f} system_sens {row['system_sens']:.4f} "
            f"retained_F1 {row['retained_macro_f1']:.4f}; <40 referral {u40['referral_rate']:.4f} "
            f"system_sens {u40['system_sens']:.4f} ({u40['n_escalating']} esc)")


def _print_frontier(frame: pd.DataFrame) -> None:
    cols = ["arm", "budget", "floor", "band", "n_escalating", "referral_rate", "system_sens",
            "retained_macro_f1", "nnb_pi0.03"]
    sub = frame[frame["band"].isin(["ALL", "<40", "40-59", "60+"])][cols]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(sub.round(4).to_string(index=False))


def outcome(delta: dict[str, Any]) -> str:
    if delta["ci_hi"] < 0:
        return "HARM"
    if delta["ci_lo"] > 0:
        return "SUPPORTED" if delta["delta"] >= MCID else "POSITIVE_BELOW_MCID"
    return "NOT_RESOLVED"


def run_reserved(rerun_reason: str | None, n_boot: int, *, smoke: bool = False,
                 resume: bool = False) -> int:
    """The reserved read. `smoke` runs the identical path on HAM val into s56/smoke/ with no
    receipt and no ledger rows, so the one real read cannot die on a code path never executed."""
    from research import testguard

    testguard.block_test_reads("S56 reserved evaluation: HAM test is never read")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8")) if PLAN_PATH.is_file() else None
    if plan is None:
        raise SystemExit("plan not frozen: run --freeze-plan")
    if plan["score_selection"]["selection_sha256"] != sha256(SELECTION_PATH):
        raise SystemExit("selection_val.json changed after the plan pinned it")
    score = plan["score_selection"]["selected"]
    fit = oof_panel()
    policies = fit_all(fit, score)       # deterministic: identical to the --select fit

    out_dir = OUT_DIR / "smoke" if smoke else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        receipt, (reserved, meta) = None, (val_panel(), {"smoke": "HAM val stands in"})
    else:
        receipt = receipt_begin(rerun_reason, resume)
        reserved, meta = reserved_panel()
    frame = frontier(reserved, policies, intervals=True, n_boot=n_boot).assign(role="primary")
    secondary = fit_all(fit, SECONDARY_SCORE) if SECONDARY_SCORE != score else {}
    if secondary:
        extra = frontier(reserved, secondary, intervals=True, n_boot=n_boot)
        frame = pd.concat([frame, extra[extra["arm"] != "none"].assign(role="secondary")],
                          ignore_index=True)
    frontier_path = out_dir / "frontier_reserved.csv"
    frame.to_csv(frontier_path, index=False)

    contrasts = []
    for r in BUDGETS:
        u = reserved.score(score)
        ref_b = policies[("band", r)].refer(u, reserved.bands)
        ref_g = policies[("global", r)].refer(u, reserved.bands)
        for band, stat in ((PRIMARY_BAND, "system_sens"), (PRIMARY_BAND, "referral_rate"),
                           ("40-59", "system_sens"), ("60+", "system_sens"),
                           ("ALL", "system_sens"), ("ALL", "referral_rate"),
                           ("ALL", "retained_macro_f1")):
            contrasts.append({"budget": r, "contrast": "band-global",
                              **paired_delta(reserved, ref_b, ref_g, band, stat, n_boot)})
    for r in BUDGETS if secondary else ():
        u = reserved.score(SECONDARY_SCORE)
        ref_b = secondary[("band", r)].refer(u, reserved.bands)
        ref_g = secondary[("global", r)].refer(u, reserved.bands)
        for band, stat in ((PRIMARY_BAND, "system_sens"), (PRIMARY_BAND, "referral_rate"),
                           ("ALL", "referral_rate"), ("ALL", "retained_macro_f1")):
            contrasts.append({"budget": r, "contrast": f"band-global[{SECONDARY_SCORE}]",
                              **paired_delta(reserved, ref_b, ref_g, band, stat, n_boot)})
    contrast_frame = pd.DataFrame(contrasts)
    contrast_path = out_dir / "contrasts_reserved.csv"
    contrast_frame.to_csv(contrast_path, index=False)

    primary = contrast_frame[(contrast_frame["contrast"] == "band-global")
                             & (contrast_frame["budget"] == PRIMARY_BUDGET)
                             & (contrast_frame["band"] == PRIMARY_BAND)
                             & (contrast_frame["stat"] == "system_sens")].iloc[0].to_dict()
    transfer = []
    band_rows = frame[(frame["arm"] == "band") & (frame["role"] == "primary")]
    for r in BUDGETS:
        floor = policies[("band", r)].floor
        for b in FLOOR_BANDS:
            row = band_rows[(band_rows["budget"] == r) & (band_rows["band"] == b)].iloc[0]
            transfer.append({"budget": r, "band": b, "nominal_floor": floor,
                             "realised": row["system_sens"], "ci_lo": row["system_sens_ci_lo"],
                             "ci_hi": row["system_sens_ci_hi"],
                             "status": ("violated" if row["system_sens_ci_hi"] < floor
                                        else "met" if row["system_sens"] >= floor
                                        else "below_point_within_ci")})
    report = {"plan_sha256": sha256(PLAN_PATH), "score": score, "reserved": meta,
              "floors_oof": {str(r): policies[("band", r)].floor for r in BUDGETS},
              "primary": {**primary, "outcome": outcome(primary), "mcid": MCID},
              "floor_transfer": transfer}
    report_path = out_dir / "s56_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    if receipt is not None:
        receipt_complete(receipt, [frontier_path, contrast_path, report_path])

    _print_frontier(frame[frame["role"] == "primary"])
    print(contrast_frame.round(4).to_string(index=False))
    print(json.dumps(report["primary"], indent=1, default=float))
    if smoke:
        return 0
    rows = [{"method": f"S56_reserved_{a}_R{int(r * 100)}", "split": "reserved",
             "macro_f1": g["retained_macro_f1"], "escalation_sens": g["system_sens"],
             "notes": _note(g, frame, a, r)}
            for (a, r), g in _all_rows(frame[frame["role"] == "primary"]).items()]
    rows.append({"method": "S56_primary", "split": "reserved",
                 "notes": f"S56 primary: <40 system sensitivity band-global at R={PRIMARY_BUDGET} "
                          f"{primary['delta']:+.4f} [{primary['ci_lo']:+.4f}, "
                          f"{primary['ci_hi']:+.4f}] -> {report['primary']['outcome']} "
                          f"(MCID {MCID}); score {score}; plan {report['plan_sha256'][:16]}"})
    write_ledger(rows, prune=[r["method"] for r in rows])
    return 0


# ============================================================================ self-test
def selftest() -> int:
    rng = np.random.default_rng(0)
    checks = 0

    # 1. CRC count: brute force agrees
    for _ in range(200):
        n = int(rng.integers(5, 60))
        u = rng.random(n)
        wm = rng.random(n) < 0.3
        pos = int(wm.sum() + rng.integers(0, 5))
        s = float(rng.choice(FLOOR_GRID))
        k = refer_count_for_floor(u, wm, pos, s)
        order = np.argsort(-u, kind="stable")
        brute = next((j for j in range(n + 1)
                      if pos == 0 or (wm.sum() - wm[order[:j]].sum() + 1) / (pos + 1)
                      <= 1 - s + 1e-12), n + 1)
        assert k == brute, (k, brute)
    checks += 1

    # 2. threshold_for_count refers exactly k rows on distinct scores
    u = rng.random(100)
    for k in (0, 1, 37, 100):
        assert int((u >= threshold_for_count(u, k)).sum()) == k
    checks += 1

    # 3. synthetic panel: band arm meets the CRC floor in-sample, spends the budget exactly,
    #    and the global arm spends it too
    n = 3000
    esc = [0, 1, 4]
    y = rng.integers(0, 7, n)
    logits = rng.normal(size=(n, 7))
    logits[np.arange(n), y] += rng.normal(1.5, 1.0, n)
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    bands = rng.choice(np.array(BANDS), n, p=[0.2, 0.4, 0.38, 0.02])
    panel = Panel("syn", probs, y, bands, np.arange(n).astype(str), esc)
    for score in SCORES:
        for r in BUDGETS:
            pb, pg = fit_band(panel, score, r), fit_global(panel, score, r)
            u = uncertainty(score, probs, esc)
            ref = pb.refer(u, bands)
            assert abs(int(ref.sum()) - round(r * n)) <= 1, (score, r, ref.sum())
            assert int(pg.refer(u, bands).sum()) == round(r * n)
            for b in FLOOR_BANDS:
                m = bands == b
                fn = int((panel.would_miss & ~ref & m).sum())
                assert (fn + 1) / (panel.y_esc[m].sum() + 1) <= 1 - pb.floor + 1e-9
    checks += 1

    # 4. max-min: one grid step above S* is infeasible
    pb = fit_band(panel, "esc_risk", 0.2)
    u = uncertainty("esc_risk", probs, esc)
    nxt = round(pb.floor + 0.005, 4)
    need = sum(refer_count_for_floor(u[bands == b], panel.would_miss[bands == b],
                                     int(panel.y_esc[bands == b].sum()), nxt)
               for b in FLOOR_BANDS)
    assert need > round(0.2 * n)
    checks += 1

    # 5. macro_f1 matches sklearn
    from sklearn.metrics import f1_score

    pred = probs.argmax(1)
    assert abs(macro_f1(y, pred) - f1_score(y, pred, labels=range(7), average="macro",
                                            zero_division=0)) < 1e-12
    checks += 1

    # 6. describe: FRR = 1 - system sensitivity; no referral -> system sens = argmax sens
    rows = describe(panel, np.zeros(n, bool), intervals=False)
    top = rows[0]
    assert abs(top["system_sens"] - panel.pred_esc[panel.y_esc].mean()) < 1e-12
    assert abs(top["frr"] + top["system_sens"] - 1) < 1e-12
    checks += 1

    # 7. paired delta of an arm against itself is exactly zero
    d = paired_delta(panel, ref, ref, "<40", "system_sens", n_boot=50)
    assert d["delta"] == 0 and d["ci_lo"] == 0 and d["ci_hi"] == 0
    checks += 1

    # 8. outcome mapping
    assert outcome({"delta": 0.12, "ci_lo": 0.01, "ci_hi": 0.2}) == "SUPPORTED"
    assert outcome({"delta": 0.05, "ci_lo": 0.01, "ci_hi": 0.1}) == "POSITIVE_BELOW_MCID"
    assert outcome({"delta": 0.05, "ci_lo": -0.01, "ci_hi": 0.1}) == "NOT_RESOLVED"
    assert outcome({"delta": -0.05, "ci_lo": -0.1, "ci_hi": -0.01}) == "HARM"
    checks += 1

    # 9. orientation: an undecided row outranks a confident one under every score (the first
    #    --select run inverted msp/margin and referred the most confident cases first)
    sure = np.array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.94, 0.01]])
    torn = np.array([[0.05, 0.05, 0.05, 0.05, 0.36, 0.39, 0.05]])
    for score in SCORES:
        assert uncertainty(score, torn, esc)[0] > uncertainty(score, sure, esc)[0], score
    checks += 1

    print(f"selftest: {checks}/{checks} checks passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--select", action="store_true", help="OOF fit + val selection (no reserved)")
    mode.add_argument("--freeze-plan", action="store_true")
    mode.add_argument("--reserved", action="store_true", help="the one reserved read")
    parser.add_argument("--rerun-reason", default=None)
    parser.add_argument("--resume", action="store_true", help="finish a crashed reserved read")
    parser.add_argument("--smoke", action="store_true", help="--reserved path on HAM val")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.select:
        return run_select()
    if args.freeze_plan:
        return freeze_plan()
    return run_reserved(args.rerun_reason, args.n_boot, smoke=args.smoke, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
