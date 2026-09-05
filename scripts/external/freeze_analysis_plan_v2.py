"""Amend the post-S11 pre-registration to v2 and re-hash the provenance record (S12).

Append-only. `results/external/analysis_plan_post_s11.json` (v1, SHA-256
`e6193e19...`) is left byte-for-byte alone; v2 is a second file that names v1's hash,
declares what changed, and says plainly which changes were made *after* the v1 lock.

The point of an amendment is that it is legible as one. Every entry in `deviations` states
what was declared, what is now true, when the change was made relative to the lock, and how
it affects inference. Two of them are the reason S12 exists at all: E6 and E7 were written
into the battery after the lock and never appeared in v1's endpoint list, and v1's
`frozen_transfer_parameters.dirichlet_map` named a file that is not the map the published
system deploys.

    $py scripts/external/freeze_analysis_plan_v2.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

V1_PATH = Path("results/external/analysis_plan_post_s11.json")
V2_PATH = Path("results/external/analysis_plan_post_s11_v2.json")
PROV_PATH = Path("results/external/post_s11_provenance.json")
AMENDED = "2026-09-05T00:00:00Z"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(v1: dict, v1_hash: str) -> dict:
    plan = json.loads(json.dumps(v1))  # deep copy; v1 is never mutated in place
    plan["version"] = "post_s11_v2"
    plan["supersedes"] = {"version": "post_s11_v1", "file": str(V1_PATH).replace("\\", "/"),
                          "sha256": v1_hash}
    plan["amended_timestamp"] = AMENDED
    plan["amendment_session"] = "S12"

    # --- E0 withdrawn --------------------------------------------------------------
    plan["workstream_endpoints"]["E0_comparison_arm"] = {
        "status": "WITHDRAWN before any result was computed",
        "reason": ("the arm required retraining under a re-weighted age prior; the "
                   "checkpoints are frozen (see the immutability rule) and the two scripts "
                   "that would have produced it were never implemented beyond a stub. "
                   "Withdrawing it is a smaller claim than reporting it, not a larger one."),
        "multiple_comparison_handling": ("retained in the family denominator and entered "
                                         "into Holm-Bonferroni at p = 1.0. The family stays "
                                         "at 5 members; withdrawing a member by shrinking "
                                         "the denominator would inflate every survivor."),
        "withdrawn_endpoints": v1["workstream_endpoints"]["E0_comparison_arm"],
    }

    # --- E1 gains the dose-response endpoint, declared before S13 runs ---------------
    plan["workstream_endpoints"]["E1_age_rule_replication"] = list(
        v1["workstream_endpoints"]["E1_age_rule_replication"]
    ) + ["E1_dose_response_skew_ordering"]
    plan["new_endpoints_declared_in_v2"] = {
        "E1_dose_response_skew_ordering": {
            "declared": AMENDED,
            "declared_before_data": True,
            "rationale": ("the under-40 escalating prior is 4.85% in HAM, 4.8% in MSKCC and "
                          "18.6% in BCN. Ordering under-40 argmax escalation sensitivity by "
                          "that skew tests the decision-rule account as a dose-response "
                          "across three independent centres rather than as a binary "
                          "replication in one."),
            "prediction": ("if the failure is a decision-rule artefact of a low escalating "
                           "prior, sensitivity should order BCN > HAM ~ MSKCC; if the "
                           "escalation-mass AUC is flat across the same ordering, the "
                           "ranking signal survives and only the operating point moves."),
            "direction": "two-sided; a null or reversed ordering is reported as measured",
            "status": "declared before any BCN/MSK image was scored (S13 has not run)",
        }
    }

    # --- E6 / E7 admitted as post-hoc -----------------------------------------------
    plan["workstream_endpoints"]["E6_decision_curves"] = {
        "status": "POST-HOC SECONDARY -- added after the v1 lock",
        "declared": AMENDED,
        "honest_note": ("E6 does not appear in v1's endpoint list. It was written into the "
                        "battery after the SHA-256 freeze and is therefore not "
                        "pre-registered. It is reported as exploratory, is excluded from "
                        "the Holm family, and carries no confirmatory claim."),
        "endpoints": ["net_benefit_lambda_rule_vs_argmax",
                      "net_benefit_risk_model_s_esc_threshold",
                      "delta_net_benefit_within_under40_band"],
        "panel": "research/predictions_oof_tta (24-view TTA) + deployed Dirichlet map",
        "test_anchor": ("net benefit on HAM test reconstructed from TP, FP and N in "
                        "results/session9/agerule_test.csv -- a frozen S9 artifact. "
                        "No new test read."),
    }
    plan["workstream_endpoints"]["E7_case_atlas"] = {
        "status": "POST-HOC SECONDARY -- added after the v1 lock; illustrative only",
        "declared": AMENDED,
        "honest_note": ("a figure, not an endpoint. No statistic is drawn from it and it "
                        "enters no family. Exemplars are selected by running the frozen "
                        "rule, and a panel with no qualifying case is shown as empty."),
        "endpoints": [],
    }

    # --- the corrected parameter pointers -------------------------------------------
    plan["frozen_transfer_parameters"] = {
        "dirichlet_map": "research/selective/results_oof/fit_state.json",
        "dirichlet_map_note": ("v1 named research/calibration/results_oof/fit_state.json. "
                               "That is a different map (max |dW| = 0.18). The file named "
                               "here is bit-identical to the calibrator run_session5_agerule "
                               "fits in process and the one research/session9/testpass.py "
                               "scores the age rule with, so it is the map under which the "
                               "frozen lambdas and the S9 test anchor mean what they say. "
                               "Asserted by research.external.frozen_params --selftest."),
        "conformal_map": ("research/conformal/results_oof/fit_state.json -- the conformal "
                          "quantiles were calibrated under their own tuning-half Dirichlet "
                          "map and must be applied under it, not under the deployed map."),
        "age_rule_lambda": "research/agerule/results_oof/age_rule_lambda.json",
        "conformal_quantiles": "research/conformal/results_oof/fit_state.json",
        "mahalanobis_centroids": "fitted on HAM train features, zero target fitting",
        "loader": "research/external/frozen_params.py (single entry point; nothing is typed)",
    }

    plan["deviations"] = [
        {
            "id": "D1", "when": "after the v1 lock", "session": "S12",
            "declared": "v1 lists E0-E5 only",
            "actual": "E6 (decision curves) and E7 (case atlas) were also built",
            "resolution": "declared post-hoc secondary; excluded from the Holm family",
            "affects_inference": "no confirmatory claim rests on either",
        },
        {
            "id": "D2", "when": "after the v1 lock", "session": "S12",
            "declared": "dirichlet_map = research/calibration/results_oof/fit_state.json",
            "actual": ("the deployed map is research/selective/results_oof/fit_state.json; "
                       "the two differ by max |dW| = 0.18 and give val Macro-F1 0.7713 vs "
                       "0.7715 for the rule"),
            "resolution": "pointer corrected; frozen_params is now the only loader",
            "affects_inference": ("small in magnitude, but they are different systems and "
                                  "the frozen lambdas were fitted under the deployed one. "
                                  "research/xdomain/run_session8b.py still loads the "
                                  "calibration-module map, so its published PAD figures are "
                                  "not directly comparable to this battery's PAD panels "
                                  "until it is re-run."),
        },
        {
            "id": "D3", "when": "before any result", "session": "S12",
            "declared": "E0_comparison_arm as a family member",
            "actual": "withdrawn; the arm requires retraining and the checkpoints are frozen",
            "resolution": "Holm entry at p = 1.0, family denominator held at 5",
            "affects_inference": "conservative; no survivor gains power from the withdrawal",
        },
        {
            "id": "D4", "when": "before S13 scores any image", "session": "S12",
            "declared": "E1 had three endpoints",
            "actual": "E1_dose_response_skew_ordering added as a fourth",
            "resolution": "declared here, ahead of the data",
            "affects_inference": "pre-registered in the sense that matters: nothing was seen",
        },
        {
            "id": "D5", "when": "after the v1 lock", "session": "S12",
            "declared": "Level_0_InDistribution = HAM10000 Test (N=1502)",
            "actual": ("five scripts read research/predictions (plain, 1-view) on the test "
                       "split outside the single pre-registered S9 pass, which the receipt "
                       "records as 2 executions covering 19 quantities, none of them E3, "
                       "E4, E6 or E7"),
            "resolution": ("every HAM panel repointed to the out-of-fold 24-view TTA matrix "
                           "(N=6,981); where a table wants a test column it is reconstructed "
                           "from frozen S9 artifacts"),
            "affects_inference": ("the Level-0 figures change. They are now out-of-fold "
                                  "rather than held-out-test, which is the correct anchor "
                                  "for anything fitted on OOF and the only one available "
                                  "without a second test read."),
        },
        {
            "id": "D6", "when": "after the v1 lock", "session": "S12",
            "declared": "mahalanobis_distance_monotonicity_L0_L1_L2",
            "actual": "the L0 arm read HAM test features (convnext_tiny_test.npz)",
            "resolution": "L0 is HAM val, as in S8b; the test arm is deleted",
            "affects_inference": ("AUROC val-vs-PAD 0.913, separation ratio 18x -- the S8b "
                                  "figures, unchanged. Nothing rested on the test arm."),
        },
        {
            "id": "D7", "when": "after the v1 lock", "session": "S12",
            "declared": "conformal_set_size_widening_audit",
            "actual": ("APS was reimplemented inline as a deterministic cumulative sum while "
                       "the stored quantiles were calibrated with the project's randomised "
                       "APS/RAPS score, so APS marginal coverage read 0.36 against a nominal "
                       "0.95"),
            "resolution": ("research.conformal.scores is imported; the in-distribution "
                           "reference is the OOF tuning half, scored under the conformal "
                           "Dirichlet map the quantiles were calibrated with"),
            "affects_inference": ("APS mondrian a05 in-distribution coverage 0.36 -> 0.947. "
                                  "The tuning half is not fully held out (the Dirichlet map "
                                  "and RAPS hyperparameters were fitted on it); the audit "
                                  "carries the ID-to-shift contrast, not an absolute HAM "
                                  "coverage claim, which S9 owns on test."),
        },
    ]

    family = dict(v1["multiple_comparison_family"])
    family["denominator_note"] = ("held at 5 members. E0 is withdrawn and enters at p = 1.0 "
                                  "rather than being removed; E6 and E7 are post-hoc and "
                                  "enter no family.")
    plan["multiple_comparison_family"] = family
    return plan


def main() -> int:
    v1_hash = sha256(V1_PATH)
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    payload = json.dumps(build(v1, v1_hash), indent=2).encode("utf-8")
    V2_PATH.write_bytes(payload)
    v2_hash = hashlib.sha256(payload).hexdigest()

    PROV_PATH.write_text(json.dumps({
        "analysis_plan_file": str(V2_PATH).replace("\\", "/"),
        "analysis_plan_sha256": v2_hash,
        "version": "post_s11_v2",
        "frozen_timestamp": AMENDED,
        "pre_registered": True,
        "status": "LOCKED",
        "supersedes": {
            "analysis_plan_file": str(V1_PATH).replace("\\", "/"),
            "analysis_plan_sha256": v1_hash,
            "version": "post_s11_v1",
            "frozen_timestamp": "2026-09-05T05:33:00Z",
            "retained": "byte-for-byte; amendments are append-only",
        },
        "deviation_note": (
            "Amended 2026-09-05 in Session 12. Seven deviations are recorded in the v2 "
            "plan's `deviations` array. Two were made before any corresponding data existed "
            "(E0 withdrawn, E1 dose-response added ahead of S13) and are pre-registered in "
            "the sense that matters. Five correct work that had already been done under the "
            "v1 lock -- E6/E7 admitted as post-hoc, the Dirichlet pointer corrected, the "
            "Level-0 panel moved off unregistered test reads, the Mahalanobis test arm "
            "deleted, and the conformal score function repaired. Those five are amendments "
            "to a broken record, not pre-registration, and are labelled as such wherever "
            "their numbers are reported."
        ),
    }, indent=2), encoding="utf-8")

    print(f"v1 retained  sha256 {v1_hash[:16]}...")
    print(f"v2 written   sha256 {v2_hash[:16]}...  -> {V2_PATH}")
    print(f"provenance re-hashed with both -> {PROV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
