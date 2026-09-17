"""S57b -- the gate plan, the one reserved read, the verdict, and (only on ADOPT) the freeze.

Order, enforced by the code:

1. `--freeze-plan` writes `results/v4/s57b_plan.json` from development artefacts only (the
   cross-fit audit, the bootstrap, S57a's arm tables), with every input hashed. All five gates,
   the confirmatory family, the Holm correction, the bootstrap and the verdict logic are fixed
   here. Gate 5 and the development half of Gate 2 are *already resolved* in the plan, because
   they are development quantities.
2. `--reserved` reads the S49 reserved cohort **once** (receipt
   `results/v4/s57b/reserved_receipt.json`; a repeat needs `--rerun-reason`), refuses to run if
   the plan or any input changed since it was frozen, evaluates only the quantities the plan
   registers, and writes `results/v4/lambda_verdict.json`, `results/v4/lambda_ablation.csv` and
   `paper/tables/lambda_ablation.tex`. **It cannot fit**: every fitting entry point is replaced
   by a function that raises for the duration of the read. HAM test is never read (testguard).
3. Only on ADOPT: `results/v4/analysis_plan_s57.json` and the frozen curve under
   `research/agerule/results_oof/`, loaded thereafter only through `frozen_params`.

    $py -m research.v4.lambda_freeze --selftest
    $py -m research.v4.lambda_freeze --freeze-plan
    $py -m research.v4.lambda_freeze --reserved --smoke     # rehearsal on HAM val, no receipt
    $py -m research.v4.lambda_freeze --reserved             # the one reserved read
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.v4 import lambda_age as la  # noqa: E402
from research.v4 import lambda_crossfit as lc  # noqa: E402
from research.v4 import lambda_stability as ls  # noqa: E402

PLAN = REPO_ROOT / "results" / "v4" / "s57b_plan.json"
S57B_DIR = lc.S57B_DIR
RECEIPT = S57B_DIR / "reserved_receipt.json"
VERDICT = REPO_ROOT / "results" / "v4" / "lambda_verdict.json"
ABLATION_CSV = REPO_ROOT / "results" / "v4" / "lambda_ablation.csv"
ABLATION_TEX = REPO_ROOT / "paper" / "tables" / "lambda_ablation.tex"
ANALYSIS_PLAN = REPO_ROOT / "results" / "v4" / "analysis_plan_s57.json"
FROZEN_CURVE = REPO_ROOT / "research" / "agerule" / "results_oof" / "lambda_curve.json"
SESSION = "v4_s57b"
ARMS = ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7")
MCID = 0.10                    # S48
ALPHA = 0.05
NI_MARGIN = 0.02               # Macro-F1 / balanced accuracy (S56's guard, same unit)
SPEC_NI_MARGIN = 0.02          # reserved specificity vs A1, per band
REFERRAL_BLOWOUT = 1.5         # under-40 referral ratio candidate / A1
N_BOOT = 2000
BOOT_SEED = 42
VERDICT_ORDER = ("ADOPT", "PROMISING", "REJECT-cost", "REJECT-unstable", "REJECT-flat")


# ============================================================================ plan
def inputs() -> dict[str, Path]:
    from research.agerule import lambda_rule as lr
    from research.external import frozen_params as fp
    from research.v4 import s54_gate as g

    out = {
        "arm_tables": la.CANDIDATES_JSON, "crossfit_audit": lc.AUDIT_JSON, "crossfit_rows": lc.ROWS_CSV,
        "ablation_dev": lc.ABLATION_DEV_CSV, "bootstrap": ls.BOOT_JSON, "bootstrap_draws": ls.DRAWS_NPZ,
        "lambda_curve": ls.CURVE_CSV, "frozen_band_lambda": REPO_ROOT / fp.LAMBDA_STATE,
        "deployed_dirichlet": REPO_ROOT / fp.DIRICHLET_STATE, "lambda_rule_py": Path(lr.__file__),
        "manifest_v4": g.MANIFEST,
    }
    for c in g.V1_COHORTS:
        out[f"v1_{c}"] = g.V1_FROZEN_DIR / f"ensemble_dirichlet_{c}.csv"
    for p in sorted(g.topup_dir(False).glob("ensemble_dirichlet_*.csv")):
        out[f"v1_topup_{p.stem}"] = p
    return out


def hashes() -> dict[str, dict[str, str]]:
    return {k: {"path": la.rel(p), "sha256": la.sha256(p)} for k, p in inputs().items()}


def build_plan() -> dict[str, Any]:
    from research.v4.s54_guard import git_head, now

    audit = json.loads(lc.AUDIT_JSON.read_text(encoding="utf-8"))
    boot = json.loads(ls.BOOT_JSON.read_text(encoding="utf-8"))
    family = audit["confirmatory_family"]
    missing = [c for c in family if c not in boot["gate5"]]
    if missing:
        raise SystemExit(f"bootstrap lacks Gate 5 for {missing}; re-run lambda_stability")
    dev = pd.read_csv(lc.ABLATION_DEV_CSV).set_index("arm")
    counts = pd.read_csv(lc.ROWS_CSV)
    gate2_dev = {}
    from research.stats.intervals import clopper_pearson
    for c in family:
        per_band = {}
        for b in la.FLOOR_BANDS:
            neg = counts[(counts.band == b) & ~counts.y.isin(la.esc_indices())]
            tn = int((neg[f"flag_{c}"] == 0).sum())
            lo, hi = clopper_pearson(tn, len(neg))
            per_band[b] = {"specificity": tn / len(neg), "cp": [lo, hi], "pass": hi >= la.MIN_SPECIFICITY}
        gate2_dev[c] = {"per_band": per_band, "pass": all(v["pass"] for v in per_band.values())}
    return {
        "session": SESSION, "frozen_at": now(), "git_head": git_head(), "bootstrap_seed": BOOT_SEED,
        "question": "does any lambda(age) finer than three bands beat the frozen S5 rule?",
        "evaluation_surface": {"primary": "manifest_v4 split=reserved (4,733 images, 104 under-40 "
                                          "escalating lesions), deployed V1 ensemble, read once",
                               "ham_test": "not read in S57 (runbook §S57b)",
                               "probabilities": "S13/S54 frozen deployed-V1 files; no image scored"},
        "arms": {a: {"source": la.rel(la.CANDIDATES_JSON), "kind": ls.KIND[a]} for a in ARMS},
        "candidate_selection": {"primary": lc.PRIMARY_CANDIDATE, "rule": audit["dev_best_rule"],
                                "dev_best": audit["dev_best"], "confirmatory_family": family,
                                "selected_on": "cross-fitted HAM OOF only -- never a reserved quantity"},
        "hyperparameters": boot["hyperparameters_fixed_at_s57a"],
        "objective": "expected clinical cost (build_cost_matrix) s.t. escalation specificity >= 0.85 per band",
        "primary_endpoint": {"quantity": "under-40 escalation sensitivity (image-level), candidate - A1",
                             "interval": f"lesion-grouped paired percentile bootstrap, {N_BOOT} resamples, "
                                         f"seed {BOOT_SEED}, resampling lesions within the band",
                             "p_value": "two-sided bootstrap p", "multiplicity": f"Holm over {family}"},
        "secondary_endpoints": ["under-40 missed serious", "overall sensitivity", "overall missed serious",
                                "Macro-F1", "balanced accuracy", "specificity per band", "referral per band",
                                "NNB per band (pi 0.01/0.03/0.05)",
                                "ECE and AURC: unchanged by construction (lambda moves neither the "
                                "probabilities nor the confidence ranking) -- reported once, not per arm"],
        "gates": {
            "1_under40": {"rule": f"delta >= {MCID} AND 95% CI lower bound > 0 AND Holm p < {ALPHA}",
                          "directional": f"delta >= {MCID} with CI including 0 (-> PROMISING at most)"},
            "2_specificity": {"development": "cross-fitted OOF: per-band Clopper-Pearson upper bound >= 0.85",
                              "reserved": f"per band, candidate specificity >= A1's - {SPEC_NI_MARGIN} "
                                          "(A1 already misses the 0.85 floor on reserved at 40-59, S57a, "
                                          "known before this plan; the floor is not relaxed, it is reported)",
                              "development_result": gate2_dev},
            "3_noninferiority": {"rule": f"Macro-F1 and balanced accuracy: lower 95% bound of "
                                         f"(candidate - A1) > -{NI_MARGIN}; 40-59 and 60+ sensitivity point "
                                         f"estimate >= A1's Clopper-Pearson lower bound"},
            "4_referral": {"rule": f"per-band delta referral and delta NNB reported; BLOWOUT if under-40 "
                                   f"referral(candidate) / referral(A1) > {REFERRAL_BLOWOUT}"},
            "5_stability": {"rule": boot["gate5_rule"], "result": {c: boot["gate5"][c] for c in family}},
        },
        "verdict_logic": [
            "ADOPT: gate 1 passes and gates 2-5 pass (2 on both halves)",
            "REJECT-cost: gate 1 passes and gate 4 BLOWOUT",
            "PROMISING: gate 1 directional, gates 2, 3, 5 pass, no BLOWOUT (A1 stays deployed)",
            "REJECT-unstable: gate 5 fails",
            "REJECT-flat: otherwise",
            f"overall verdict: the most favourable member verdict in the order {list(VERDICT_ORDER)}; "
            "every member's verdict is reported",
        ],
        "exploratory": {"ablation_steps": [f"{a}->{b} ({w})" for a, b, w in lc.STEPS],
                        "arms_not_in_family": [a for a in ARMS if a not in family],
                        "deviance_diagnostic": audit["deviance_diagnostic"]},
        "dev_summary": {c: {"crossfit_under40_sens": float(dev.loc[c, "sensitivity|<40"]),
                            "crossfit_cost": float(dev.loc[c, "crossfit_cost"])} for c in ARMS},
        "freeze_on_adopt": {"file": la.rel(ANALYSIS_PLAN), "curve": la.rel(FROZEN_CURVE),
                            "loader": "frozen_params.load_lambda_curve()"},
        "transport_after_freeze_only": "lambda_transport.py, non-reserved BCN/MSKCC rows, zero-shot",
        "inputs": hashes(),
    }


def freeze_plan() -> int:
    if PLAN.is_file():
        raise SystemExit(f"{la.rel(PLAN)} exists; the plan is immutable once written")
    plan = build_plan()
    la._dump(PLAN, la._clean(plan))
    print(f"plan frozen: {la.rel(PLAN)} sha256 {la.sha256(PLAN)}")
    print(f"  family {plan['candidate_selection']['confirmatory_family']}; gate 5 "
          + ", ".join(f"{c} {v['pass']}" for c, v in plan["gates"]["5_stability"]["result"].items())
          + "; gate 2 (dev) "
          + ", ".join(f"{c} {v['pass']}" for c, v in plan["gates"]["2_specificity"]["development_result"].items()))
    return 0


# ============================================================================ the read
@contextlib.contextmanager
def no_fitting() -> Iterator[None]:
    """The frozen runner cannot fit: every fitting entry point raises while it runs."""
    from research.agerule import lambda_rule as lr

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("fitting is disabled during the S57b evaluation")

    targets = [(lr, "fit_lambda"), (lr, "fit_lambda_by_group"), (la, "fit_link_curve"),
               (la, "local_fits"), (la, "fit_pooled"), (la, "arm_link"), (la, "refit_a1")]
    saved = [(m, n, getattr(m, n)) for m, n in targets]
    try:
        for m, n in targets:
            setattr(m, n, refuse)
        yield
    finally:
        for m, n, f in saved:
            setattr(m, n, f)


def load_arms() -> dict[str, la.Arm]:
    arms = json.loads(la.CANDIDATES_JSON.read_text(encoding="utf-8"))["arms"]
    return {a: la.Arm(a, arms[a]["kind"], np.array([arms[a]["table"][str(int(k))] for k in la.AGE_KNOTS]),
                      float(arms[a]["missing_age_lambda"])) for a in ARMS}


def reserved_eval_panel() -> tuple[la.Panel, dict[str, Any]]:
    from research.v4 import s54_gate as g
    from research.v4.s56_abstention import reserved_panel

    p, meta = reserved_panel()
    frame = g.build_panel("reserved")
    assert np.array_equal(frame["y7"].to_numpy(), p.y7) and np.array_equal(frame["age_band"].to_numpy(), p.bands)
    ages = frame["age_approx"].to_numpy(dtype=float)
    assert np.array_equal(np.isnan(ages), p.bands == "unknown"), "missing age and 'unknown' band disagree"
    return la.Panel("reserved", p.probs, p.y7, ages, frame["group_id"].to_numpy(), p.bands), meta


def evaluate_panel(panel: la.Panel, plan: dict[str, Any]) -> dict[str, Any]:
    from ml.evaluation.metrics import compute_metrics
    from research.agerule import lambda_rule as lr
    from research.stats.intervals import clopper_pearson
    from sklearn.metrics import balanced_accuracy_score, f1_score

    esc, cm = la.esc_indices(), la.cost_matrix()
    arms = load_arms()
    lam = {a: arms[a].lam_for(panel.ages) for a in ARMS}
    assert all(np.isfinite(v).all() for v in lam.values())
    with no_fitting():
        preds = {}
        for a in ARMS:
            out = np.empty(len(panel), int)
            for v in np.unique(lam[a]):
                m = lam[a] == v
                out[m] = lr.apply_lambda(panel.probs[m], float(v), esc)
            preds[a] = out
        table, abl = lc.ablation(panel, lam, esc, cm, N_BOOT, panel.name)
    evals = abl["evaluations"]
    labels = list(range(panel.probs.shape[1]))
    for a in ARMS:   # the bootstrap's metric must be the reported metric
        mf = f1_score(panel.y, preds[a], labels=labels, average="macro", zero_division=0)
        assert abs(mf - evals[a]["all"]["macro_f1"]) < 1e-12
        assert np.array_equal(np.isin(preds[a], esc), abl["flags"][a])
    te = np.isin(panel.y, esc)
    u40 = np.flatnonzero(panel.bands == "<40")
    everyone = np.arange(len(panel))
    family = plan["candidate_selection"]["confirmatory_family"]
    members = {}
    for i, c in enumerate(family):
        fc, f1_ = abl["flags"][c], abl["flags"]["A1"]
        d_sens = lc.paired_boot(panel.lesion[u40], lambda idx: (
            fc[u40][idx][te[u40][idx]].mean() - f1_[u40][idx][te[u40][idx]].mean())
            if te[u40][idx].any() else np.nan, N_BOOT, BOOT_SEED + i)
        d_mf1 = lc.paired_boot(panel.lesion, lambda idx: (
            f1_score(panel.y[idx], preds[c][idx], labels=labels, average="macro", zero_division=0)
            - f1_score(panel.y[idx], preds["A1"][idx], labels=labels, average="macro", zero_division=0)),
            N_BOOT, BOOT_SEED + 100 + i)
        d_ba = lc.paired_boot(panel.lesion, lambda idx: (
            balanced_accuracy_score(panel.y[idx], preds[c][idx])
            - balanced_accuracy_score(panel.y[idx], preds["A1"][idx])), N_BOOT, BOOT_SEED + 200 + i)
        members[c] = {"under40_sensitivity": d_sens, "macro_f1": d_mf1, "balanced_accuracy": d_ba}
    # Holm over the family on the primary endpoint
    ps = sorted((members[c]["under40_sensitivity"]["p_boot"], c) for c in family)
    running = 0.0
    for rank, (p, c) in enumerate(ps):
        running = max(running, min(1.0, (len(ps) - rank) * p))
        members[c]["under40_sensitivity"]["p_holm"] = running

    results = {}
    a1 = evals["A1"]
    for c in family:
        e = evals[c]
        m = members[c]
        s = m["under40_sensitivity"]
        g1_pass = s["delta"] >= MCID and s["ci"][0] > 0 and s["p_holm"] < ALPHA
        g1_dir = s["delta"] >= MCID and not g1_pass
        spec_res = {b: {"candidate": e[b]["specificity"], "A1": a1[b]["specificity"],
                        "pass": e[b]["specificity"] >= a1[b]["specificity"] - SPEC_NI_MARGIN,
                        "floor_met": e[b]["specificity"] >= la.MIN_SPECIFICITY} for b in la.FLOOR_BANDS}
        g2_dev = plan["gates"]["2_specificity"]["development_result"][c]["pass"]
        g2 = g2_dev and all(v["pass"] for v in spec_res.values())
        older = {}
        for b in ("40-59", "60+"):
            pos = int(a1[b]["n_escalating"])
            caught_a1 = pos - int(a1[b]["missed_serious"])
            lo, _ = clopper_pearson(caught_a1, pos)
            older[b] = {"candidate": e[b]["sensitivity"], "A1": a1[b]["sensitivity"], "A1_cp_lo": lo,
                        "pass": e[b]["sensitivity"] >= lo}
        g3 = (m["macro_f1"]["ci"][0] > -NI_MARGIN and m["balanced_accuracy"]["ci"][0] > -NI_MARGIN
              and all(v["pass"] for v in older.values()))
        ref = {b: {"candidate": e[b]["referral_rate"], "A1": a1[b]["referral_rate"],
                   "delta": e[b]["referral_rate"] - a1[b]["referral_rate"],
                   "delta_nnb_pi0.03": e[b].get("nnb_pi0.03", np.nan) - a1[b].get("nnb_pi0.03", np.nan)}
               for b in ("all",) + la.FLOOR_BANDS}
        ratio = ref["<40"]["candidate"] / ref["<40"]["A1"] if ref["<40"]["A1"] > 0 else np.inf
        blowout = bool(ratio > REFERRAL_BLOWOUT)
        g5 = plan["gates"]["5_stability"]["result"][c]["pass"]
        if g1_pass and g2 and g3 and not blowout and g5:
            v = "ADOPT"
        elif g1_pass and blowout:
            v = "REJECT-cost"
        elif g1_dir and g2 and g3 and g5 and not blowout:
            v = "PROMISING"
        elif not g5:
            v = "REJECT-unstable"
        else:
            v = "REJECT-flat"
        results[c] = {"verdict": v,
                      "gate1": {**s, "pass": bool(g1_pass), "directional": bool(g1_dir), "mcid": MCID},
                      "gate2": {"development_pass": bool(g2_dev), "reserved": spec_res, "pass": bool(g2)},
                      "gate3": {"macro_f1": m["macro_f1"], "balanced_accuracy": m["balanced_accuracy"],
                                "older_bands": older, "pass": bool(g3)},
                      "gate4": {"per_band": ref, "under40_referral_ratio": float(ratio),
                                "blowout": blowout},
                      "gate5": {"pass": bool(g5)}}
    overall = min((r["verdict"] for r in results.values()), key=VERDICT_ORDER.index)
    return {"table": table, "steps": abl["steps"], "evaluations": evals, "members": results,
            "overall": overall}


def write_table(dev: pd.DataFrame, res: pd.DataFrame, family: list[str]) -> pd.DataFrame:
    dev = dev.assign(split="oof_crossfit")
    both = pd.concat([dev, res.assign(crossfit_cost=np.nan)], ignore_index=True)
    both.to_csv(ABLATION_CSV, index=False, lineterminator="\n")
    d, r = dev.set_index("arm"), res.set_index("arm")
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Age-resolution ablation for the escalation bias $\lambda(\mathrm{age})$. "
             r"Development columns are cross-fitted HAM OOF (nested: calibrator, hyperparameters and "
             r"$\lambda$ never see the scored fold); reserved columns are the BCN\,+\,MSKCC cohort, read "
             r"once. Sens.\ = escalation sensitivity; Ref.\ = share referred (called escalating); "
             r"Spec.\ = lowest per-band escalation specificity. $\dagger$ confirmatory family.}",
             r"\label{tab:lambda_ablation}", r"\small",
             r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{Development (cross-fitted OOF)} & \multicolumn{5}{c}{Reserved cohort} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-10}",
             r"Arm & Mechanism & Cost & $<$40 Sens. & $<$40 Ref. & $<$40 Sens. & $<$40 Ref. & All Sens. & Macro-F1 & Spec. \\",
             r"\midrule"]
    names = {"A0": "argmax", "A1": "frozen 3-band", "A2": "5-year, unsmoothed", "A3": "kernel",
             "A4": "linear", "A5": "quadratic", "A6": "P-spline", "A7": "shrunk 5-year"}
    for a in ARMS:
        mark = r"$^\dagger$" if a in family else ""
        spec = min(r.loc[a, f"specificity|{b}"] for b in la.FLOOR_BANDS)
        lines.append(f"{a}{mark} & {names[a]} & {d.loc[a, 'crossfit_cost']:.3f} & {d.loc[a, 'sensitivity|<40']:.3f} & "
                     f"{d.loc[a, 'referral_rate|<40']:.3f} & {r.loc[a, 'sensitivity|<40']:.3f} & "
                     f"{r.loc[a, 'referral_rate|<40']:.3f} & {r.loc[a, 'sensitivity|all']:.3f} & "
                     f"{r.loc[a, 'macro_f1']:.4f} & {spec:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    ABLATION_TEX.parent.mkdir(parents=True, exist_ok=True)
    ABLATION_TEX.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return both


def reserved(smoke: bool, rerun_reason: str | None) -> int:
    from research import testguard
    from research.v4.s54_guard import git_head, now

    testguard.block_test_reads("S57b evaluation: reserved (or HAM val in a smoke), never HAM test")
    if not PLAN.is_file():
        raise SystemExit("freeze the plan first (--freeze-plan)")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    drift = {k: v for k, v in hashes().items() if plan["inputs"].get(k, {}).get("sha256") != v["sha256"]}
    drift.update({k: "missing now" for k in plan["inputs"] if k not in hashes()})
    if drift:
        raise SystemExit(f"inputs changed since the plan was frozen: {sorted(drift)}")
    plan_sha = la.sha256(PLAN)
    out_dir = S57B_DIR / "smoke" if smoke else S57B_DIR
    receipt = None
    if not smoke:
        receipt = (json.loads(RECEIPT.read_text(encoding="utf-8")) if RECEIPT.is_file()
                   else {"cohort": "manifest_v4 split=reserved (S57b gates)", "plan": la.rel(PLAN),
                         "plan_sha256": plan_sha, "executions": []})
        if receipt["plan_sha256"] != plan_sha:
            raise SystemExit("the plan changed after the receipt was opened")
        if receipt["executions"] and not rerun_reason:
            raise SystemExit("S57b has already read reserved; a repeat needs --rerun-reason")
        receipt["executions"].append({"execution": len(receipt["executions"]) + 1, "status": "started",
                                      "started_at": now(), "rerun_reason": rerun_reason, "git_head": git_head()})
        la._dump(RECEIPT, receipt)
        panel, meta = reserved_eval_panel()
    else:
        panel, meta = la.val_panel(), {"smoke": "HAM val stands in for reserved"}
        panel.name = "ham_val_smoke"

    res = evaluate_panel(panel, plan)
    dev = pd.read_csv(lc.ABLATION_DEV_CSV)
    out_dir.mkdir(parents=True, exist_ok=True)
    family = plan["candidate_selection"]["confirmatory_family"]
    verdict = {
        "session": SESSION, "smoke": smoke, "panel": panel.name, "plan": la.rel(PLAN), "plan_sha256": plan_sha,
        "verdict": res["overall"], "members": res["members"],
        "consequence": {"ADOPT": "lambda(age) becomes the deployed decision layer; S57c unlocks",
                        "PROMISING": "retained as exploratory; A1 stays deployed; S57c stays locked",
                        "REJECT-flat": "three bands already capture the recoverable decision-rule benefit; "
                                       "A1 stays deployed; S57c stays locked",
                        "REJECT-unstable": "the data support broad conditioning, not fine resolution, at this "
                                           "sample size; A1 stays deployed; S57c stays locked",
                        "REJECT-cost": "improvement achievable but not clinically efficient; A1 stays "
                                       "deployed; S57c stays locked"}[res["overall"]],
        "ablation_steps_reserved": res["steps"],
        "deviance_diagnostic_dev": plan["exploratory"]["deviance_diagnostic"],
        "panel_meta": meta,
        "not_read": "HAM test",
    }
    if smoke:
        la._dump(out_dir / "lambda_verdict.json", la._clean(verdict))
        res["table"].to_csv(out_dir / "lambda_ablation_reserved.csv", index=False, lineterminator="\n")
    else:
        la._dump(VERDICT, la._clean(verdict))
        res["table"].to_csv(out_dir / "ablation_reserved.csv", index=False, lineterminator="\n")
        write_table(dev, res["table"], family)
        frozen = freeze_if_adopted(verdict, plan)
        verdict["freeze"] = frozen
        la._dump(VERDICT, la._clean(verdict))
    report(res, panel.name)
    if receipt is not None:
        items = {la.rel(p): la.sha256(p) for p in (VERDICT, ABLATION_CSV, ABLATION_TEX, out_dir / "ablation_reserved.csv")}
        receipt["executions"][-1].update(status="completed", completed_at=now(), verdict=res["overall"], items=items)
        la._dump(RECEIPT, receipt)
        ledger(res, plan)
    return 0


def freeze_if_adopted(verdict: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    if verdict["verdict"] != "ADOPT":
        return {"frozen": False, "reason": f"verdict {verdict['verdict']}: nothing is frozen; A1 stays deployed"}
    from research.v4.s54_guard import git_head, now

    adopted = [c for c, r in verdict["members"].items() if r["verdict"] == "ADOPT"]
    c = min(adopted, key=lambda a: plan["dev_summary"][a]["crossfit_cost"])
    arm = load_arms()[c]
    la._dump(FROZEN_CURVE, la._clean({"source_session": SESSION, "arm": c, **arm.as_dict()}))
    la._dump(ANALYSIS_PLAN, la._clean({**plan, "adopted": c, "frozen_curve": la.rel(FROZEN_CURVE),
                                       "frozen_curve_sha256": la.sha256(FROZEN_CURVE),
                                       "git_head_at_freeze": git_head(), "frozen_at_adopt": now()}))
    return {"frozen": True, "arm": c, "curve": la.rel(FROZEN_CURVE), "analysis_plan": la.rel(ANALYSIS_PLAN),
            "next": "add frozen_params.load_lambda_curve() as the only loader, then run lambda_transport"}


def report(res: dict[str, Any], name: str) -> None:
    t = res["table"].set_index("arm")
    print(f"\n[{name}] arm  <40 sens / referral   all sens  MF1     spec per band")
    for a in ARMS:
        print(f"  {a}   {t.loc[a, 'sensitivity|<40']:.3f} / {t.loc[a, 'referral_rate|<40']:.3f}       "
              f"{t.loc[a, 'sensitivity|all']:.3f}    {t.loc[a, 'macro_f1']:.4f}  "
              + " ".join(f"{t.loc[a, f'specificity|{b}']:.3f}" for b in la.FLOOR_BANDS))
    for c, r in res["members"].items():
        g1 = r["gate1"]
        print(f"  {c}: {r['verdict']}  G1 {g1['delta']:+.3f} [{g1['ci'][0]:+.3f}, {g1['ci'][1]:+.3f}] "
              f"Holm p {g1['p_holm']:.3f} pass={g1['pass']}  G2={r['gate2']['pass']}  G3={r['gate3']['pass']}  "
              f"G4 ratio {r['gate4']['under40_referral_ratio']:.2f} blowout={r['gate4']['blowout']}  "
              f"G5={r['gate5']['pass']}")
    print(f"  OVERALL: {res['overall']}")


def ledger(res: dict[str, Any], plan: dict[str, Any]) -> None:
    rows = []
    for a, e in res["evaluations"].items():
        rows.append({"method": f"S57b_reserved_{a}", "split": "reserved", "macro_f1": e["all"]["macro_f1"],
                     "balanced_accuracy": e["all"]["balanced_accuracy"], "escalation_sens": e["all"]["sensitivity"],
                     "missed_serious": e["all"]["missed_serious"],
                     "notes": f"S57b reserved, frozen S57a table {a}: <40 sens {e['<40']['sensitivity']:.4f} "
                              f"referral {e['<40']['referral_rate']:.4f}; min band spec "
                              f"{min(e[b]['specificity'] for b in la.FLOOR_BANDS):.4f}"})
    for c, r in res["members"].items():
        g = r["gate1"]
        rows.append({"method": f"S57b_verdict_{c}", "split": "reserved", "p_value_vs_baseline": g["p_holm"],
                     "notes": f"S57b {c} vs A1 <40 sens {g['delta']:+.4f} [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}] "
                              f"Holm p {g['p_holm']:.4f}; verdict {r['verdict']}; overall {res['overall']}"})
    lc.write_ledger(rows, "S57b_reserved")


# ============================================================================ selftest
def selftest() -> int:
    fails = 0

    def check(name: str, ok: bool) -> None:
        nonlocal fails
        fails += not ok
        print(("PASS " if ok else "FAIL ") + name)

    from research import testguard
    from research.agerule import lambda_rule as lr

    before = lr.fit_lambda
    with no_fitting():
        try:
            lr.fit_lambda(None, None, None, None)
            ok = False
        except RuntimeError:
            ok = True
    check("the frozen runner cannot fit (fit_lambda raises inside no_fitting)", ok)
    check("fitting is restored afterwards", lr.fit_lambda is before)
    testguard.block_test_reads("S57b selftest")
    try:
        testguard.check_split("test", "selftest")
        ok = False
    except testguard.TestSplitLocked:
        ok = True
    check("testguard is armed and refuses the HAM test split", ok)
    head = __import__("subprocess").run(["git", "-C", str(REPO_ROOT), "show", "HEAD:research/agerule/lambda_rule.py"],
                                        capture_output=True).stdout
    check("S5's lambda_rule.py is byte-identical to the committed file",
          head.replace(b"\r\n", b"\n") == Path(lr.__file__).read_bytes().replace(b"\r\n", b"\n"))
    esc = la.esc_indices()
    cm = la.cost_matrix()
    probs = np.tile([[0.05, 0.05, 0.05, 0.05, 0.05, 0.7, 0.05]], (40, 1))
    y = np.array([4] * 20 + [5] * 20)
    try:
        lr.fit_lambda(probs, y, cm, esc, grid=np.array([0.0, 0.7]), min_specificity=0.0, strict=True)
        ok = False
    except lr.GridBoundaryError:
        ok = True
    check("GridBoundaryError still fires at the grid edge", ok)
    arms = load_arms()
    check("arm tables load for all eight arms and are finite",
          set(arms) == set(ARMS) and all(np.isfinite(a.knots).all() for a in arms.values()))
    check("A1 table equals the frozen S5 lambdas", np.allclose(arms["A1"].knots, la.arm_a1().knots))
    # verdict logic on synthetic gate results
    check("verdict order puts ADOPT first", min(["REJECT-flat", "ADOPT"], key=VERDICT_ORDER.index) == "ADOPT")
    if PLAN.is_file():
        again = json.loads(PLAN.read_text(encoding="utf-8"))
        h = {k: v["sha256"] for k, v in again["inputs"].items()}
        now_h = {k: v["sha256"] for k, v in hashes().items()}
        check("plan input hashes reproduce", h == now_h)
    print(f"{fails and 'FAILURES' or 'all passed'} ({fails} failed)")
    return int(fails > 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--freeze-plan", action="store_true")
    ap.add_argument("--reserved", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rerun-reason", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.freeze_plan:
        return freeze_plan()
    if args.reserved:
        return reserved(args.smoke, args.rerun_reason)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
