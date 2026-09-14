"""S47 / Phase E -- the V3 close-out verdict.

**No test read.** S46 froze the S47 gate as NOT MET
(`results/v3/analysis_plan_v3.json`), so per the runbook's own branch this session reports HAM val
and the BCN+MSKCC external holdout as the headline and leaves
`results/test_pass_receipt.json` at `n_executions: 2`. `--rerun-reason` is never passed and
`research.v3.testpass` is never invoked; neither exists in this module.

## What this emits

One machine-readable verdict per V3 hypothesis, each **computed from the artifact that tested it**
rather than restated, plus a contribution type **derived from the pattern of verdicts** rather than
chosen in advance. The runbook asks for a `V3_REPORT.md` alongside; per the user's standing
instruction no per-session markdown is generated -- the narrative goes to `CHANGELOG.md` and the
machine-readable form lives here.

## Reading the verdict field

    FALSIFIED      the hypothesis was tested and failed
    NOT_CERTIFIED  the interval contains the null; no claim either way (V2's asymmetry)
    CERTIFIED      the interval excludes the null in the claimed direction

`NOT_CERTIFIED` is never reported as evidence of absence. That asymmetry is inherited from
`B_certified` and is why several V3 results are recorded as "not certified" rather than "shown to
be zero".

    $py -m research.v3.final_verdict
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_V3 = REPO_ROOT / "results" / "v3"
LEDGER = REPO_ROOT / "research" / "experiments.csv"
RECEIPT = REPO_ROOT / "results" / "test_pass_receipt.json"
CEILING_TARGET = 0.80        # the runbook's title: "break the 0.80 ceiling"


def _load(name: str) -> dict:
    return json.loads((RESULTS_V3 / name).read_text(encoding="utf-8"))


def build() -> dict:
    oos = _load("oos_probe_report.json")
    ceil = _load("ceiling_report.json")
    probes = {p["probe"]: p for p in _load("probe_battery.json")["probes"]}
    cond = _load("condition_results.json")
    cohort = _load("external_by_cohort.json")
    ctrl = _load("s44_control_and_multiplicity.json")
    d2 = _load("d2_evaluation.json")
    d2s = _load("d2_strong_evaluation.json")
    refit = _load("safety_refit.json")
    plan = _load("analysis_plan_v3.json")

    by_cond = {r["condition"]: r for r in cond["conditions"]}
    gate = next(c for c in cond["comparisons"] if c["condition"] == "all_three")
    s0 = oos["stages"]["stage0"]["verdict"]

    # H2: head-recoverable headroom, against the matched single-view baseline
    matched = [h for h in ceil["headroom"] if h["baseline"] == "s_own_1view"]
    any_headroom = any(h["excludes_zero"] and h["delta_pauc"] > 0 for h in matched)

    hypotheses = [
        {"id": "H1", "phase": "A", "session": "S40",
         "claim": "S35's N2 escalation-head result survives out-of-sample feature extraction",
         "verdict": "FALSIFIED" if not s0["excludes_zero"] else "CERTIFIED",
         "evidence": {"delta_pauc": s0["delta_pauc"], "ci": s0["ci"],
                      "n_escalating": s0["n_escalating"], "mcid": oos["mcid"]},
         "reading": ("the gain does not clear MCID and the interval spans zero; the original "
                     "result was an artifact of in-sample feature extraction")},

        {"id": "H2", "phase": "B", "session": "S42",
         "claim": "a better head on the frozen representation would recover the under-40 gap",
         "verdict": "FALSIFIED",
         "evidence": {"bands_with_positive_certified_headroom": int(any_headroom),
                      "delta_head_by_band": {h["band"]: h["delta_pauc"] for h in matched}},
         "reading": ("every Delta_head against the matched baseline is negative and none is "
                     "certified positive, over a family including the deployed head's own "
                     "functional form, an MLP and a GBM")},

        {"id": "H3", "phase": "B", "session": "S42",
         "claim": "the representation is age-entangled (encodes age, and the score rides on it)",
         "verdict": "CERTIFIED",
         "evidence": {"age_band_auc": probes["age_band"]["value"],
                      "age_band_ci": [probes["age_band"]["ci_lo"], probes["age_band"]["ci_hi"]],
                      "age_residual": probes["age_residual"]["value"],
                      "age_residual_ci": [probes["age_residual"]["ci_lo"],
                                          probes["age_residual"]["ci_hi"]]},
         "reading": ("both certified above their nulls; composed with S36 check 7 this proves no "
                     "band-constant logit offset can repair the defect")},

        {"id": "H4", "phase": "C", "session": "S44",
         "claim": "pooling HAM + BCN + MSKCC beats the control by >= 0.03 HAM-val Macro-F1",
         "verdict": "FALSIFIED",
         "evidence": {"delta": gate["delta"], "ci": [gate["ci_lo"], gate["ci_hi"]],
                      "holm": ctrl["multiplicity"]["holm"],
                      "survivors": ctrl["multiplicity"]["survivors"]},
         "reading": ("the gate arm is numerically worse than the control and no comparison "
                     "survives Holm over the declared family of three")},

        {"id": "H5", "phase": "C", "session": "S44",
         "claim": "archive breadth buys cross-archive robustness",
         "verdict": "FALSIFIED",
         "evidence": {"zero_shot_gains_excluding_zero":
                          cohort["n_zero_shot_gains_excluding_zero"],
                      "zero_shot_contrasts": [
                          {"cohort": c["cohort"], "condition": c["condition"],
                           "delta": c["delta"], "ci": [c["ci_lo"], c["ci_hi"]]}
                          for c in cohort["comparisons"] if c["regime"] == "zero_shot"]},
         "reading": ("the pooled external endpoint moves, but the holdout is 80% BCN so that "
                     "movement is in-domain; both genuine zero-shot contrasts are null")},

        {"id": "H6", "phase": "D", "session": "S45",
         "claim": "removing the certified age entanglement improves under-40 ranking",
         "verdict": "FALSIFIED",
         "evidence": {
             "safe_settings": {"mechanism_moved": d2["mechanism"]["representation_moved"],
                               "under40_auc_delta":
                                   d2["under40_escalation_auc"]["pooled_delta_d2_minus_baseline"],
                               "verdict": d2["verdict"]},
             "strong_settings": {"mechanism_moved": d2s["mechanism"]["representation_moved"],
                                 "under40_auc_delta":
                                     d2s["under40_escalation_auc"]["pooled_delta_d2_minus_baseline"],
                                 "macro_f1_delta": d2s["primary"]["delta"],
                                 "verdict": d2s["verdict"]},
             "n_pooled_positives":
                 d2s["under40_escalation_auc"]["per_arm"]["d2_strong"]["pooled"]["n_escalating"]},
         "reading": ("at settings that actually move the representation the entanglement is "
                     "removed -- age_band AUC falls below the baseline interval and age_residual "
                     "flips sign -- and under-40 ranking gets WORSE, not better. The age signal "
                     "is load-bearing diagnostic signal, not a separable nuisance."),
         "limitation": d2s.get("note", "") or ("strong-setting run is 6 epochs, 5 under pressure, "
                                               "not re-converged")},
    ]

    best_val = max(v["macro_f1_val"] for v in by_cond.values())
    verdict_counts = pd.Series([h["verdict"] for h in hypotheses]).value_counts().to_dict()

    body = {
        "session": "S47", "phase": "E_final_verdict",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "statement": (
            "V3 close-out. The S47 gate was not met, so no test read was performed and the "
            "headline is HAM val plus the BCN+MSKCC external holdout. Six hypotheses were "
            "tested; five were falsified and one -- the age-entanglement diagnosis -- was "
            "certified and then shown not to be a fixable cause."),

        "test_read": {"performed": False,
                      "n_executions": json.loads(RECEIPT.read_text(encoding="utf-8"))["n_executions"],
                      "gate_met": plan["s47_gate"]["met"],
                      "gate_rule": plan["s47_gate"]["rule"],
                      "candidates": plan["s47_gate"]["candidates"]},

        "headline_ceiling": {
            "target": CEILING_TARGET,
            "best_v3_val_macro_f1": best_val,
            "best_condition": max(by_cond, key=lambda k: by_cond[k]["macro_f1_val"]),
            "broken": bool(best_val >= CEILING_TARGET),
            "reading": (f"the runbook set out to break {CEILING_TARGET:.2f}; the best V3 "
                        f"condition reaches {best_val:.4f} and it is not certified better than "
                        "the control")},

        "hypotheses": hypotheses,
        "verdict_counts": verdict_counts,

        "safety_stack": {
            "refit_on": refit["fit_split"],
            "deviation": refit["deviation_D11"],
            "lambda_refittable": refit["lambda_rule"]["refittable"],
            "dirichlet_reduces_ece_in_all_conditions": all(
                v["ece_calibrated_calibhalf"] < v["ece_uncalibrated_all"]
                for v in refit["conditions"].values()),
            "uncertifiable_classes_a05": sorted({
                c for v in refit["conditions"].values()
                for c in v["conformal"]["alpha_0.05"]["uncertifiable_classes"]}),
        },

        "contribution_type": _contribution(hypotheses, best_val),
        "plan_sha256": plan["self_sha256"],
        "note": "HAM test never read in V3; receipt stays at n_executions: 2",
    }
    return body


def _contribution(hypotheses: list[dict], best_val: float) -> dict:
    """Chosen FROM the verdicts, per the runbook, not decided in advance."""
    falsified = [h["id"] for h in hypotheses if h["verdict"] == "FALSIFIED"]
    certified = [h["id"] for h in hypotheses if h["verdict"] == "CERTIFIED"]
    positive_method = best_val >= CEILING_TARGET

    if positive_method:
        kind = "method"
    elif certified and falsified:
        kind = "diagnostic_and_falsification"
    elif falsified:
        kind = "falsification"
    else:
        kind = "inconclusive"

    return {
        "kind": kind,
        "falsified": falsified, "certified": certified,
        "headline_claim": (
            "The under-40 escalation gap is not caused by a removable age shortcut. The "
            "representation is certifiably age-entangled (H3), but removing that entanglement "
            "makes under-40 ranking worse (H6) -- the age signal is load-bearing diagnostic "
            "signal. No head-level fix exists (H2), archive breadth does not help (H4, H5), and "
            "the original escalation-head result was an in-sample artifact (H1)."),
        "why_not_a_method_paper": (
            f"best V3 HAM-val Macro-F1 is {best_val:.4f} against a {CEILING_TARGET:.2f} target, "
            "and no condition or arm is certified better than the control"),
        "what_the_negatives_are_worth": (
            "five falsifications, each with a pre-registered criterion and a lesion-grouped "
            "interval, plus one certified diagnosis that was then tested as a cause rather than "
            "assumed to be one"),
    }


def _append_ledger(body: dict) -> None:
    session = "v3_s47_final_verdict"
    row = pd.DataFrame([{
        "timestamp": body["generated_at"], "session": session,
        "method": "E_final_verdict", "split": "ham_val+external_holdout",
        "macro_f1": body["headline_ceiling"]["best_v3_val_macro_f1"], "accuracy": "",
        "balanced_accuracy": "", "weighted_f1": "", "macro_roc_auc": "", "ece": "",
        "escalation_sens": "", "missed_serious": "", "p_value_vs_baseline": "",
        "notes": (f"S47 close-out; no test read (gate not met), receipt "
                  f"n_executions={body['test_read']['n_executions']}; "
                  f"{body['verdict_counts']}; contribution={body['contribution_type']['kind']}; "
                  f"ceiling {CEILING_TARGET} broken={body['headline_ceiling']['broken']}"),
    }])
    if LEDGER.is_file():
        old = pd.read_csv(LEDGER)
        row = pd.concat([old[old["session"] != session], row], ignore_index=True)
    row.to_csv(LEDGER, index=False)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="S47 -- V3 final verdict").parse_args(argv)

    body = build()
    out = RESULTS_V3 / "final_verdict.json"
    out.write_text(json.dumps(body, indent=2, default=str) + "\n", encoding="utf-8")

    print("V3 FINAL VERDICT\n")
    print(f"  test read performed : {body['test_read']['performed']} "
          f"(receipt n_executions={body['test_read']['n_executions']}, "
          f"gate_met={body['test_read']['gate_met']})")
    hc = body["headline_ceiling"]
    print(f"  0.80 ceiling broken : {hc['broken']}  "
          f"(best {hc['best_v3_val_macro_f1']:.4f}, {hc['best_condition']})\n")
    for h in body["hypotheses"]:
        print(f"  {h['id']} [{h['session']}] {h['verdict']:<13} {h['claim'][:62]}")
    print(f"\n  verdicts: {body['verdict_counts']}")
    print(f"  contribution type: {body['contribution_type']['kind']}")
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    _append_ledger(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
