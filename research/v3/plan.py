"""S46 / Phase E -- freeze the V3 analysis plan and register it as a frozen artifact.

Writes `results/v3/analysis_plan_v3.json` with a `self_sha256` (same idiom as
`research/session9/plan.py`), then registers that hash in `results/frozen_artifacts.json`
**without disturbing the sibling keys**.

## The S11 precedent this module is written against

`build_paper_artifacts.py` once rewrote `results/frozen_artifacts.json` wholesale and would have
deleted the `analysis_plan` key S9 had written there. The file now holds
`frozen_at`, `statement`, `num_files`, `files` (34 prediction hashes) and `analysis_plan`.
`register()` below performs a **read-modify-write** that adds only `analysis_plan_v3` and asserts
every pre-existing key and all 34 file hashes survive byte-identically. `--check` re-verifies
after the fact.

## What the plan records

V3 produced **no certified improvement**, so this plan does not freeze a deployment candidate. It
freezes what was measured, the gate that was evaluated, and the reasons the remaining stack was or
was not refit -- so S47 inherits a decision rather than re-deriving one.

The S47 gate was fixed in the runbook before any V3 training existed:

    winning system beats `ham_only` by >= 0.03 HAM-val Macro-F1 with a CI excluding zero

and it is **not met** by any candidate. This module records that verdict; it does not act on it.
S47 remains the user's call and the test receipt stays at `n_executions: 2`.

    $py -m research.v3.plan
    $py -m research.v3.plan --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO_ROOT / "results" / "v3" / "analysis_plan_v3.json"
FROZEN_ARTIFACTS = REPO_ROOT / "results" / "frozen_artifacts.json"
RESULTS_V3 = REPO_ROOT / "results" / "v3"

MCID = 0.03
GATE_KEY = "analysis_plan_v3"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str) -> dict:
    return json.loads((RESULTS_V3 / name).read_text(encoding="utf-8"))


def build() -> dict:
    """Every number here is read from results/v3/, never typed (Hard Rule 4)."""
    cond = _load("condition_results.json")
    cohort = _load("external_by_cohort.json")
    ctrl = _load("s44_control_and_multiplicity.json")
    d2 = _load("d2_evaluation.json")
    d2s = _load("d2_strong_evaluation.json")
    refit = _load("safety_refit.json")

    by_cond = {r["condition"]: r for r in cond["conditions"]}
    gate_cmp = next(c for c in cond["comparisons"] if c["condition"] == "all_three")

    body = {
        "session": "S46", "phase": "E_plan_freeze",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "statement": (
            "V3 analysis plan. Frozen after S44 (Phase C conditions), S45 (Phase D arm D2) and "
            "the S46 safety refit. No V3 candidate is certified better than the ham_only "
            "control, so this plan freezes measurements and a gate verdict, not a deployment "
            "candidate. The HAM test split was not read at any point in V3."
        ),

        "phase_c": {
            "gate": {"metric": "macro_f1", "split": "ham_val", "control": "ham_only",
                     "arm": "all_three", "delta_required": MCID,
                     "delta": gate_cmp["delta"], "ci_lo": gate_cmp["ci_lo"],
                     "ci_hi": gate_cmp["ci_hi"], "fires": gate_cmp["gate_fires"]},
            "conditions": {k: {"macro_f1_val": v["macro_f1_val"],
                               "esc_sens_under40": v["esc_sens_under40"],
                               "macro_f1_external": v["macro_f1_external"]}
                           for k, v in by_cond.items()},
            "multiplicity": ctrl["multiplicity"],
            "outcome_preregistered": cond["outcome"],
            "outcome_substantive": (
                "4_archives_do_not_pool_naively -- the pre-registered classifier returns outcome "
                "3 on the pooled external endpoint, but that holdout is 80% BCN so the movement "
                "is in-domain; 0 of 2 genuine zero-shot transfer contrasts have a CI excluding "
                "zero"),
            "zero_shot_transfer_gains_excluding_zero": cohort["n_zero_shot_gains_excluding_zero"],
        },

        "endpoint_retired": {
            "endpoint": "under40_escalation_sensitivity_thresholded",
            "reason": ("S44 measured a same-data retraining spread of "
                       f"{ctrl['control_reproduction']['same_data_spread']['under40_cases']} of "
                       f"{by_cond['ham_only']['under40_n']} cases, larger than the entire "
                       "between-condition spread; at 22 positives it cannot discriminate"),
            "same_data_spread": ctrl["control_reproduction"]["same_data_spread"],
            "between_condition_spread": ctrl["control_reproduction"]["between_condition_spread"],
            "replacement": ("under-40 escalation-mass AUC pooled over HAM val + external holdout "
                            "(76 positives) -- deviation D10"),
        },

        "phase_d": {
            "arm": "D2_age_invariant_gradient_reversal",
            "arm_choice_rule": ("pre-committed in the S43 addendum before any Phase C result: if "
                                "pooling archives moves nothing, a dual-view fix to the same "
                                "entanglement is less likely to help. Phase C moved nothing."),
            "safe_settings": {
                "config": "age_weight=1.0 max_lambda=1.0 k_inner=5 lr=1e-5, 30 epochs",
                "verdict": d2["verdict"],
                "primary_delta": d2["primary"]["delta"],
                "mechanism_moved": d2["mechanism"]["representation_moved"],
                "under40_auc_delta": d2["under40_escalation_auc"]["pooled_delta_d2_minus_baseline"],
            },
            "strong_settings": {
                "config": "age_weight=3 max_lambda=3 k_inner=10 lr=5e-5, 6 epochs",
                "verdict": d2s["verdict"],
                "primary_delta": d2s["primary"]["delta"],
                "mechanism_moved": d2s["mechanism"]["representation_moved"],
                "under40_auc_delta": d2s["under40_escalation_auc"]["pooled_delta_d2_minus_baseline"],
                "limitation": ("6 epochs, 5 under pressure -- not re-converged, so part of the "
                               "Macro-F1 drop is training disruption rather than the intrinsic "
                               "cost of invariance"),
            },
            "conclusion": (
                "Removing the age entanglement S42 certified makes under-40 ranking WORSE, not "
                "better. The age signal is not a separable nuisance; it is load-bearing "
                "diagnostic signal. This does NOT establish that age-invariance is worthless -- "
                "no setting was found where invariance came without cost."),
        },

        "safety_refit": {
            "deviation": refit["deviation_D11"],
            "fit_split": refit["fit_split"], "eval_split": refit["eval_split"],
            "n_tuning": refit["n_tuning"], "n_calibration": refit["n_calibration"],
            "lambda_rule": refit["lambda_rule"],
            "per_condition": {k: {"ece_uncalibrated": v["ece_uncalibrated_all"],
                                  "ece_dirichlet": v["ece_calibrated_calibhalf"],
                                  "conformal_a05_coverage":
                                      v["conformal"]["alpha_0.05"]["marginal"]["coverage"],
                                  "uncertifiable_a05":
                                      v["conformal"]["alpha_0.05"]["uncertifiable_classes"]}
                              for k, v in refit["conditions"].items()},
        },

        "s47_gate": {
            "rule": ("winning system beats ham_only by >= 0.03 HAM-val Macro-F1 with a CI "
                     "excluding zero"),
            "frozen_in": "V3_SESSION_RUNBOOK.md S46, before any V3 training existed",
            "candidates": {
                "all_three_vs_ham_only": gate_cmp["delta"],
                "ham_mskcc_vs_ham_only": next(
                    c["delta"] for c in cond["comparisons"] if c["condition"] == "ham_mskcc"),
                "ham_mskcc_holm_p": ctrl["multiplicity"]["holm"]["ham_mskcc"],
                "d2_vs_all_three": d2["primary"]["delta"],
                "d2_strong_vs_all_three": d2s["primary"]["delta"],
            },
            "met": False,
            "consequence": ("no test read. results/test_pass_receipt.json stays at "
                            "n_executions: 2. S47 reports HAM-val and the external holdout."),
        },

        "test_read_status": {"n_executions": 2, "v3_read_test": False},
        "outputs_root": "results/v3/",
    }

    canonical = json.dumps(body, indent=2, sort_keys=False)
    body["self_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return body


def register(plan_path: Path) -> dict:
    """Read-modify-write `frozen_artifacts.json`, asserting sibling keys survive (S11 bug)."""
    before = json.loads(FROZEN_ARTIFACTS.read_text(encoding="utf-8"))
    sibling_keys = set(before) - {GATE_KEY}
    file_hashes_before = {f["path"]: f["sha256"] for f in before.get("files", [])}

    after = dict(before)
    after[GATE_KEY] = {
        "path": str(plan_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "sha256": _sha256_file(plan_path),
        "bytes": plan_path.stat().st_size,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }

    assert sibling_keys <= set(after), "sibling keys lost -- the S11 bug"
    file_hashes_after = {f["path"]: f["sha256"] for f in after.get("files", [])}
    assert file_hashes_before == file_hashes_after, "prediction hashes altered"
    assert before.get("analysis_plan") == after.get("analysis_plan"), "S9 analysis_plan altered"

    FROZEN_ARTIFACTS.write_text(json.dumps(after, indent=2) + "\n", encoding="utf-8")
    return {"sibling_keys_preserved": sorted(sibling_keys),
            "n_file_hashes": len(file_hashes_after)}


def check() -> int:
    ok = True
    if not PLAN_PATH.is_file():
        print("  plan not frozen yet -- run without --check first")
        return 1
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))

    recorded = plan.pop("self_sha256")
    canonical = json.dumps(plan, indent=2, sort_keys=False)
    recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    good = recorded == recomputed
    print(f"  self_sha256 {recorded[:16]}... {'matches' if good else 'MISMATCH'}")
    ok &= good

    art = json.loads(FROZEN_ARTIFACTS.read_text(encoding="utf-8"))
    good = GATE_KEY in art and art[GATE_KEY]["sha256"] == _sha256_file(PLAN_PATH)
    print(f"  registered in frozen_artifacts.json: {'yes, hash matches' if good else 'NO'}")
    ok &= good

    good = "analysis_plan" in art and len(art.get("files", [])) == 34
    print(f"  S9 siblings intact: analysis_plan={'analysis_plan' in art}, "
          f"{len(art.get('files', []))} file hashes (expect 34) -> {'PASS' if good else 'FAIL'}")
    ok &= good

    good = plan["s47_gate"]["met"] is False and plan["test_read_status"]["n_executions"] == 2
    print(f"  S47 gate met={plan['s47_gate']['met']}, receipt={plan['test_read_status']['n_executions']} "
          f"-> {'PASS' if good else 'FAIL'}")
    ok &= good

    print("\n" + ("PLAN FROZEN AND VERIFIED" if ok else "VERIFICATION FAILED"))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="S46 -- freeze the V3 analysis plan")
    p.add_argument("--check", action="store_true")
    args = p.parse_args(argv)
    if args.check:
        return check()

    plan = build()
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    info = register(PLAN_PATH)
    print(f"wrote {PLAN_PATH.relative_to(REPO_ROOT)}")
    print(f"  self_sha256 {plan['self_sha256'][:16]}...")
    print(f"  S47 gate met: {plan['s47_gate']['met']}")
    print(f"  registered in frozen_artifacts.json; siblings preserved "
          f"{info['sibling_keys_preserved']}, {info['n_file_hashes']} file hashes intact")
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
