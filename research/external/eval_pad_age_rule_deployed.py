"""Session 16: recompute the PAD-UFES-20 age-rule probe under the *deployed* Dirichlet map.

S8b evaluated the frozen age rule on PAD through `research/xdomain/run_session8b.py`, which
loads `research/calibration/results_oof/fit_state.json`. That is not the map the published
system deploys: S12 established that three OOF Dirichlet fits are in circulation and that the
deployed one is `research/selective/results_oof/fit_state.json` (see DATASET_REFINING.md
Sec. 2a). The two differ by `max|dW| = 0.18`, which on PAD moves the calibrated probabilities
by up to 0.097 and the rule's escalation sensitivity by 0.017.

That was tolerable while the S8b numbers stood alone. It stopped being tolerable in S16, when
the composite validity table began reporting the same PAD quantities under the deployed map
in the same paper: the manuscript would have said the Dirichlet arm misses 1,264 serious cases
in prose and 1,268 in the adjacent table. This module recomputes the probe under the deployed
map so both agree, and records the superseded arm alongside it rather than quietly discarding
it.

Nothing here is new evidence and nothing is fitted on PAD. The same six frozen HAM-only
checkpoints, the same soft-vote, the same frozen per-band lambda; only the calibrator changes,
and it changes to the one every other number in the paper uses. **No test split is read.**

The rule is applied through `frozen_params.apply_age_rule` -- the single loader S12 created --
and cross-checked against `run_session8b.apply_age_rule`, so that the two implementations are
shown to agree rather than assumed to.

Usage:
    python -m research.external.eval_pad_age_rule_deployed [--no-ledger]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.evaluation.metrics import compute_metrics
from ml.paths import resolve
from research.calibration.methods import apply_calibration
from research.experiment_log import log_experiment
from research.external import frozen_params as fp
from research.xdomain import run_session8b as s8

SESSION = "session_post_s11"
WORKSTREAM = "E2b_pad_age_rule_deployed_map"
OUT_PATH = Path("results/external/pad_age_rule_deployed.json")

DEPLOYED_MAP = "research/selective/results_oof/fit_state.json"
SUPERSEDED_MAP = "research/calibration/results_oof/fit_state.json"


def _summary(y_true: np.ndarray, pred: np.ndarray) -> dict:
    metrics = compute_metrics(y_true, pred, None)
    clinical = metrics["clinical"]
    return {
        "macro_f1": round(float(metrics["macro_f1"]), 4),
        "balanced_accuracy": round(float(metrics["balanced_accuracy"]), 4),
        "accuracy": round(float(metrics["accuracy"]), 4),
        "escalation_sens": round(float(clinical["binary_sensitivity"]), 4),
        "mel_recall": round(float(metrics["per_class"]["mel"]["recall"]), 4),
        "missed_serious": int(clinical["missed_serious_cases"]),
    }


def evaluate() -> dict:
    image_ids, y_true, probs = s8.load_pad_matrix()
    vote = probs.mean(axis=1)

    manifest = pd.read_csv(resolve(s8.MANIFEST_PAD)).set_index("image_id")
    ages = manifest.loc[image_ids, "age"].to_numpy()
    lam = fp.load_lambda_by_band()

    deployed = fp.calibrate(vote)
    superseded = apply_calibration(s8.load_dirichlet(),
                                   np.log(np.clip(vote, fp.EPS, None)))

    # The canonical loader and S8b's local copy must implement the same rule. If this ever
    # fails, one of them has drifted and no number below is trustworthy (audit finding A2).
    rule_canonical = fp.apply_age_rule(deployed, ages)
    rule_s8b_impl = s8.apply_age_rule(deployed, ages, s8.load_lambdas())
    agree = bool(np.array_equal(rule_canonical, rule_s8b_impl))
    if not agree:
        raise AssertionError(
            "frozen_params.apply_age_rule and run_session8b.apply_age_rule disagree on "
            f"{int((rule_canonical != rule_s8b_impl).sum())} of {len(rule_canonical)} PAD "
            "predictions; the rule form has drifted (see audit finding A2)")

    return {
        "session": "S16",
        "workstream": WORKSTREAM,
        "role": "directional probe, not a pre-registered endpoint; no Holm family",
        "test_read": False,
        "cohort": f"PAD-UFES-20, all {len(image_ids)} images, no weights retrained",
        "panel": "6-CNN uniform soft-vote over the frozen HAM-only checkpoints (no TTA)",
        "rule": "argmax_c ( p_c + lambda_band * 1[c escalates] )",
        "lambda_by_band": lam,
        "rule_implementations_agree": agree,
        "deployed_map": DEPLOYED_MAP,
        "superseded_map": SUPERSEDED_MAP,
        "max_abs_probability_difference_between_maps": round(
            float(np.abs(deployed - superseded).max()), 6),
        "deployed": {
            "raw_soft_vote": _summary(y_true, vote.argmax(axis=1)),
            "dirichlet": _summary(y_true, deployed.argmax(axis=1)),
            "dirichlet_plus_age_rule": _summary(y_true, rule_canonical),
        },
        "superseded_s8b": {
            "raw_soft_vote": _summary(y_true, vote.argmax(axis=1)),
            "dirichlet": _summary(y_true, superseded.argmax(axis=1)),
            "dirichlet_plus_age_rule": _summary(
                y_true, fp.apply_age_rule(superseded, ages)),
        },
        "reading": (
            "the probe's direction is unchanged by the calibrator: the frozen rule still "
            "raises escalation sensitivity and melanoma recall on PAD at the cost of "
            "specificity. It remains a directional probe and not a replication, because "
            "PAD's escalating prevalence is the inverse of HAM10000's, so a constant "
            "escalating-class bonus must help mechanically under this prior regardless of "
            "whether the under-40 mechanism is what fires. The same-modality evidence is "
            "Workstream E1."),
    }


def write_ledger(report: dict) -> int:
    written = 0
    for arm, variants in (("deployed", report["deployed"]),
                          ("superseded_s8b", report["superseded_s8b"])):
        for variant, summary in variants.items():
            log_experiment({
                "session": SESSION,
                "method": f"E2b_pad_age_rule[{arm}/{variant}]",
                "split": "pad",
                "macro_f1": summary["macro_f1"],
                "accuracy": summary["accuracy"],
                "balanced_accuracy": summary["balanced_accuracy"],
                "escalation_sens": summary["escalation_sens"],
                "missed_serious": summary["missed_serious"],
                "notes": (f"{WORKSTREAM}; dirichlet_map="
                          f"{report['deployed_map'] if arm == 'deployed' else report['superseded_map']}"
                          f"; mel_recall={summary['mel_recall']}; frozen per-band lambda, "
                          f"nothing fitted on PAD; no test read"),
            })
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-ledger", action="store_true",
                        help="skip research/experiments.csv (rehearsal runs only)")
    args = parser.parse_args(argv)

    report = evaluate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH.as_posix()}")
    print(f"  maps differ by at most {report['max_abs_probability_difference_between_maps']} "
          f"in calibrated probability")
    for arm in ("deployed", "superseded_s8b"):
        for variant, summary in report[arm].items():
            print(f"  {arm:15s} {variant:24s} macro-F1 {summary['macro_f1']:.4f}  "
                  f"esc.sens {summary['escalation_sens']:.4f}  "
                  f"missed {summary['missed_serious']}")
    if not args.no_ledger:
        print(f"  {write_ledger(report)} rows logged to research/experiments.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
