"""Session 2 (non-TTA part): calibration, cost-sensitive thresholds, and DCA.

Operates on the Session-1 winning ensemble (uniform soft-vote across all 6 backbones),
reconstructed here from the same `research/predictions/` matrices rather than re-run --
combining already-computed val/test probabilities is free, so there is no reason to touch
the GPU again for this half of Phase 2.

Pipeline, all fit on the fit split / applied to test exactly once:
  1. Fit temperature scaling, matrix scaling, and Dirichlet calibration on the ensemble's
     fit-split probabilities (as log-probabilities / pseudo-logits); compare ECE on test.
  2. Build the clinical cost matrix and optimize per-class thresholds on the
     best-calibrated fit-split probabilities; compare expected clinical cost and escalation
     sensitivity/specificity against uncalibrated argmax, on test.
  3. Decision Curve Analysis: net benefit of the calibrated + thresholded ensemble vs.
     Treat All / Treat None / the ConvNeXt-Tiny baseline, over p_t in [0.01, 0.50].

Since session 6 the fit split is selectable (`research.fitsplit`). The default reproduces
the published run exactly. Under `--fit-split oof` the calibrator coefficients and the
decision thresholds come from the 6,981 out-of-fold training rows instead, while the
**choice of calibrator family stays on validation** -- which is the whole point, since
validation is no longer also the fitting set and is finally free to select.

Usage:
    python -m research.run_session2_calibration
    python -m research.run_session2_calibration --fit-split oof --no-test
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from ml.evaluation.metrics import compute_metrics, expected_calibration_error, reliability_bins
from ml.evaluation.plots import plot_reliability_diagram
from ml.paths import REPO_ROOT, load_class_mapping, resolve
from research import fitsplit
from research.calibration.methods import (
    apply_calibration,
    fit_dirichlet_calibration,
    fit_matrix_scaling,
    fit_temperature_scaling,
)
from research.dca.decision_curve import net_benefit, net_benefit_treat_all, plot_decision_curves
from research.ensembling.data import load_split_matrix
from research.ensembling.methods import soft_vote_arithmetic
from research.experiment_log import log_experiment
from research.thresholds.cost_matrix import build_cost_matrix, expected_cost
from research.thresholds.optimize import apply_thresholds, optimize_thresholds

BASELINE_ARCH = "convnext_tiny"
EPS = 1e-12
PUBLISHED_OUT_DIR = "research/calibration/results"
PUBLISHED_DCA_DIR = "research/dca/results"
PUBLISHED_SESSION = "session2"


def ece_of(y_true: np.ndarray, probs: np.ndarray, num_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    return expected_calibration_error(confidences, correct, num_bins=num_bins)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions-dir", default="research/predictions",
                        help="Frozen prediction matrices for val (selection) and test.")
    parser.add_argument("--out-dir", default=None,
                        help=f"Report directory. Defaults to {PUBLISHED_OUT_DIR} for a "
                             "published val-fitted run, and to a suffixed sibling otherwise.")
    parser.add_argument("--dca-out-dir", default=None,
                        help=f"Decision-curve figure directory. Defaults alongside "
                             f"{PUBLISHED_DCA_DIR} using the same suffix rule.")
    fitsplit.add_fit_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = fitsplit.resolve_fit(
        args, published_out_dir=PUBLISHED_OUT_DIR, published_session=PUBLISHED_SESSION
    )
    dca_dir_name = args.dca_out_dir or plan.sibling_out_dir(PUBLISHED_DCA_DIR)
    print(f"Calibration run: {plan.describe()}")

    mapping = load_class_mapping()

    # The fit matrix supplies every fitted parameter; `val` supplies every *selection*.
    # Under --fit-split val these are the same object, which is the published behaviour.
    fit = load_split_matrix(plan.matrix_split, predictions_dir=plan.fit_predictions_dir)
    if plan.is_oof or plan.fit_predictions_dir != plan.val_predictions_dir:
        val = load_split_matrix("val", predictions_dir=plan.val_predictions_dir)
    else:
        val = fit

    fit_ens = soft_vote_arithmetic(fit.probs)
    fit_log = np.log(np.clip(fit_ens, EPS, None))
    val_ens = fit_ens if val is fit else soft_vote_arithmetic(val.probs)
    val_log = fit_log if val is fit else np.log(np.clip(val_ens, EPS, None))

    test = test_ens = test_log = None
    if plan.read_test:
        test = load_split_matrix("test", predictions_dir=plan.val_predictions_dir)
        test_ens = soft_vote_arithmetic(test.probs)
        test_log = np.log(np.clip(test_ens, EPS, None))
        print(f"Uncalibrated ensemble: val ECE={ece_of(val.y_true, val_ens):.4f}  "
              f"test ECE={ece_of(test.y_true, test_ens):.4f}")
    else:
        print(f"Uncalibrated ensemble: fit ECE={ece_of(fit.y_true, fit_ens):.4f}  "
              f"val ECE={ece_of(val.y_true, val_ens):.4f}  (test locked)")

    # --- 1. Calibration -----------------------------------------------------------------
    calibrators = {
        "temperature": fit_temperature_scaling(fit_log, fit.y_true),
        "matrix_scaling": fit_matrix_scaling(fit_log, fit.y_true),
        "dirichlet": fit_dirichlet_calibration(fit_ens, fit.y_true),
    }

    calibration_results = {"uncalibrated": {"val_probs": val_ens, "test_probs": test_ens}}
    for name, state in calibrators.items():
        calibration_results[name] = {
            "val_probs": apply_calibration(state, val_log),
            "test_probs": apply_calibration(state, test_log) if plan.read_test else None,
        }

    out_dir = resolve(plan.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_rows = []
    for name, res in calibration_results.items():
        val_ece = ece_of(val.y_true, res["val_probs"])
        if plan.read_test:
            test_ece = ece_of(test.y_true, res["test_probs"])
            test_metrics = compute_metrics(
                test.y_true, res["test_probs"].argmax(axis=1), res["test_probs"]
            )
            calib_rows.append((name, val_ece, test_ece, test_metrics))
            print(f"[{name:14s}] val ECE={val_ece:.4f}  test ECE={test_ece:.4f}  "
                  f"test macro_f1={test_metrics['macro_f1']:.4f}")
            log_experiment({
                "session": plan.session, "method": f"calibration_{name}", "split": "test",
                "macro_f1": test_metrics["macro_f1"], "ece": test_ece,
                "escalation_sens": test_metrics["clinical"]["binary_sensitivity"],
                "missed_serious": test_metrics["clinical"]["missed_serious_cases"],
                "notes": f"val_ece={val_ece:.4f}",
            })
            confidences = res["test_probs"].max(axis=1)
            correct = (res["test_probs"].argmax(axis=1) == test.y_true).astype(float)
            plot_reliability_diagram(
                reliability_bins(confidences, correct, num_bins=15),
                out_dir / f"reliability_{name}.png",
                title=f"{name} calibration (test ECE={test_ece:.3f})",
            )
        else:
            val_metrics = compute_metrics(
                val.y_true, res["val_probs"].argmax(axis=1), res["val_probs"]
            )
            calib_rows.append((name, val_ece, None, val_metrics))
            print(f"[{name:14s}] val ECE={val_ece:.4f}  val macro_f1={val_metrics['macro_f1']:.4f}")
            log_experiment({
                "session": plan.session, "method": f"calibration_{name}", "split": "val",
                "macro_f1": val_metrics["macro_f1"], "ece": val_ece,
                "escalation_sens": val_metrics["clinical"]["binary_sensitivity"],
                "missed_serious": val_metrics["clinical"]["missed_serious_cases"],
                "notes": f"fit_split={plan.fit_split}; fit-only run, test not read",
            })
            confidences = res["val_probs"].max(axis=1)
            correct = (res["val_probs"].argmax(axis=1) == val.y_true).astype(float)
            plot_reliability_diagram(
                reliability_bins(confidences, correct, num_bins=15),
                out_dir / f"reliability_{name}_val.png",
                title=f"{name} calibration (val ECE={val_ece:.3f}, {plan.fit_split}-fitted)",
            )

    # Pick the calibration method with the lowest val ECE -- selection stays on val, under
    # every fit split. Selecting on the OOF rows the coefficients were fitted on would
    # relocate the in-sample problem rather than remove it.
    best_name = min(
        calibrators.keys(),
        key=lambda k: ece_of(val.y_true, calibration_results[k]["val_probs"]),
    )
    print(f"\nBest calibration by val ECE: {best_name}")
    best_fit_probs = apply_calibration(calibrators[best_name], fit_log)
    best_val_probs = calibration_results[best_name]["val_probs"]
    best_test_probs = calibration_results[best_name]["test_probs"] if plan.read_test else None

    # --- 2. Cost-sensitive thresholds ----------------------------------------------------
    cost_matrix = build_cost_matrix(mapping)
    threshold_state = optimize_thresholds(best_fit_probs, fit.y_true, cost_matrix, mapping)

    uncal_cost = thresholded_cost = None
    uncal_metrics = thresholded_metrics = None
    if plan.read_test:
        uncal_argmax_test = test_ens.argmax(axis=1)
        uncal_cost = expected_cost(test.y_true, uncal_argmax_test, cost_matrix)
        uncal_metrics = compute_metrics(test.y_true, uncal_argmax_test, test_ens)

        thresholded_test_preds = apply_thresholds(best_test_probs, threshold_state.thresholds)
        thresholded_cost = expected_cost(test.y_true, thresholded_test_preds, cost_matrix)
        thresholded_metrics = compute_metrics(test.y_true, thresholded_test_preds, best_test_probs)

        print(f"\nCost-sensitive thresholds (fit on {plan.fit_split}, {best_name}-calibrated):")
        print(f"  thresholds = {np.round(threshold_state.thresholds, 3).tolist()}")
        print(f"  uncalibrated argmax  : test cost={uncal_cost:.4f}  "
              f"escalation sens={uncal_metrics['clinical']['binary_sensitivity']:.4f}")
        print(f"  cost-sensitive       : test cost={thresholded_cost:.4f}  "
              f"escalation sens={thresholded_metrics['clinical']['binary_sensitivity']:.4f}  "
              f"specificity({plan.fit_split})={threshold_state.fit_specificity:.4f}")

        log_experiment({
            "session": plan.session, "method": "cost_sensitive_thresholds", "split": "test",
            "macro_f1": thresholded_metrics["macro_f1"],
            "escalation_sens": thresholded_metrics["clinical"]["binary_sensitivity"],
            "missed_serious": thresholded_metrics["clinical"]["missed_serious_cases"],
            "notes": f"thresholds={np.round(threshold_state.thresholds, 3).tolist()}; "
                     f"test_cost={thresholded_cost:.4f} vs uncalibrated_cost={uncal_cost:.4f}",
        })
    else:
        print(f"\nCost-sensitive thresholds (fit on {plan.fit_split}, {best_name}-calibrated):")
        print(f"  thresholds = {np.round(threshold_state.thresholds, 3).tolist()}")
        print(f"  fit cost={threshold_state.fit_cost:.4f}  "
              f"fit specificity={threshold_state.fit_specificity:.4f}")
        log_experiment({
            "session": plan.session, "method": "cost_sensitive_thresholds",
            "split": plan.matrix_split,
            "notes": f"fit_split={plan.fit_split}; thresholds="
                     f"{np.round(threshold_state.thresholds, 3).tolist()}; "
                     f"fit_cost={threshold_state.fit_cost:.4f}; "
                     f"fit_specificity={threshold_state.fit_specificity:.4f}; test not read",
        })

    # --- 3. Decision Curve Analysis --------------------------------------------------------
    escalating_idx = [c.index for c in mapping.classes if c.needs_escalation]
    dca_note = "not computed -- fit-only run, test not read"
    if plan.read_test:
        y_test_escalate = np.isin(test.y_true, escalating_idx)
        thresholds_grid = np.linspace(0.01, 0.50, 50)
        ensemble_score = best_test_probs[:, escalating_idx].sum(axis=1)
        baseline_score = test.probs_for(BASELINE_ARCH)[:, escalating_idx].sum(axis=1)

        curves = {
            f"Calibrated ensemble ({best_name})": net_benefit(y_test_escalate, ensemble_score, thresholds_grid),
            f"{BASELINE_ARCH} baseline": net_benefit(y_test_escalate, baseline_score, thresholds_grid),
            "Treat All": net_benefit_treat_all(y_test_escalate, thresholds_grid),
            "Treat None": np.zeros_like(thresholds_grid),
        }
        dca_dir = resolve(dca_dir_name)
        plot_decision_curves(thresholds_grid, curves, dca_dir / "decision_curve.png")
        print(f"\nDCA plot written to {(dca_dir / 'decision_curve.png').relative_to(REPO_ROOT)}")

        dca_note = (
            f"net_benefit@0.1: ensemble={net_benefit(y_test_escalate, ensemble_score, np.array([0.1]))[0]:.4f} "
            f"baseline={net_benefit(y_test_escalate, baseline_score, np.array([0.1]))[0]:.4f} "
            f"treat_all={net_benefit_treat_all(y_test_escalate, np.array([0.1]))[0]:.4f}"
        )
        log_experiment({
            "session": plan.session, "method": "dca_summary", "split": "test",
            "notes": dca_note,
        })
    else:
        print("\nDCA skipped: net benefit is a test-set quantity and test is locked.")

    # --- fitted state, for the S9 single test pass and the OOF-vs-val comparison ----------
    state_path = fitsplit.write_fit_state(plan, {
        "selected_calibrator": best_name,
        "selection_split": "val",
        "selection_metric": "ece",
        "val_ece": {name: ece_of(val.y_true, res["val_probs"])
                    for name, res in calibration_results.items()},
        "calibrators": {
            name: {"method": state.method, "weight": state.weight,
                   "bias": state.bias, "temperature": state.temperature}
            for name, state in calibrators.items()
        },
        "cost_sensitive_thresholds": threshold_state.thresholds,
        "cost_sensitive_fit_cost": threshold_state.fit_cost,
        "cost_sensitive_fit_specificity": threshold_state.fit_specificity,
        "fit_n": int(fit.num_samples),
        "val_n": int(val.num_samples),
    })

    # --- report ----------------------------------------------------------------------------
    fit_desc = (
        "val-fit"
        if not plan.is_oof
        else f"OOF-fit over {fit.num_samples} train rows ({plan.fit_predictions_dir})"
    )
    lines = [
        "# Phase 2 — Calibration, Cost-Sensitive Thresholds & DCA (Session 2)",
        "",
        f"## Calibration (winning Session-1 ensemble: uniform soft-vote, {fit_desc}, "
        + ("test single-read)" if plan.read_test else "test not read)"),
        "",
        ("| Method | val ECE | test ECE | test Macro-F1 | test Escalation Sens. |"
         if plan.read_test else
         "| Method | val ECE | — | val Macro-F1 | val Escalation Sens. |"),
        "|---|---:|---:|---:|---:|",
    ]
    for name, val_ece, test_ece, metrics in calib_rows:
        marker = " **(best by val ECE)**" if name == best_name else ""
        second = f"{test_ece:.4f}" if test_ece is not None else "—"
        lines.append(f"| {name}{marker} | {val_ece:.4f} | {second} | {metrics['macro_f1']:.4f} | "
                     f"{metrics['clinical']['binary_sensitivity']:.4f} |")

    lines += [
        "",
        f"## Cost-sensitive thresholds (fit on {best_name}-calibrated {plan.fit_split} probabilities)",
        "",
        f"- Thresholds: `{np.round(threshold_state.thresholds, 3).tolist()}`",
        f"- {'Val' if not plan.is_oof else 'OOF'} specificity at fit time: "
        f"{threshold_state.fit_specificity:.4f} (floor: 0.85)",
        "",
    ]
    if plan.read_test:
        lines += [
            "| Decision rule | test expected cost | test Macro-F1 | test Escalation Sens. | test Missed Serious |",
            "|---|---:|---:|---:|---:|",
            f"| Uncalibrated argmax | {uncal_cost:.4f} | {uncal_metrics['macro_f1']:.4f} | "
            f"{uncal_metrics['clinical']['binary_sensitivity']:.4f} | "
            f"{uncal_metrics['clinical']['missed_serious_cases']} |",
            f"| Cost-sensitive thresholds | {thresholded_cost:.4f} | {thresholded_metrics['macro_f1']:.4f} | "
            f"{thresholded_metrics['clinical']['binary_sensitivity']:.4f} | "
            f"{thresholded_metrics['clinical']['missed_serious_cases']} |",
            "",
            "## Decision Curve Analysis",
            "",
            f"See `{dca_dir_name}/decision_curve.png`. Net benefit at p_t=0.10: "
            f"ensemble={net_benefit(y_test_escalate, ensemble_score, np.array([0.1]))[0]:.4f}, "
            f"{BASELINE_ARCH} baseline={net_benefit(y_test_escalate, baseline_score, np.array([0.1]))[0]:.4f}, "
            f"Treat All={net_benefit_treat_all(y_test_escalate, np.array([0.1]))[0]:.4f}.",
            "",
        ]
    else:
        lines += [
            f"- Expected clinical cost on the fit split: {threshold_state.fit_cost:.4f}",
            "",
            "## Test-set quantities",
            "",
            "Not computed. This is a fit-only run (`--no-test`): the test split is locked by "
            "`research.testguard` so that every test quantity in this round is emitted by the "
            "single pre-registered pass, not accumulated incrementally. Decision Curve "
            "Analysis is a test-set quantity and is therefore also deferred.",
            "",
        ]

    # Only non-published runs carry this section, so re-running the default regenerates
    # `research/calibration/results/session2_report.md` byte-identically.
    if plan.out_dir != PUBLISHED_OUT_DIR:
        lines += [
            "## Fit and selection splits",
            "",
            f"Coefficients and thresholds fitted on **{plan.fit_split}** "
            f"(`{plan.fit_predictions_dir}`, n={fit.num_samples}); the calibrator *family* was "
            f"selected by ECE on **validation** (n={val.num_samples}), which under "
            "`--fit-split oof` is data no coefficient here has seen. Selecting the family on "
            "the same OOF rows the coefficients were fitted on would relocate the in-sample "
            "problem rather than remove it.",
            "",
            "## Fitted state",
            "",
            f"`{state_path.relative_to(REPO_ROOT).as_posix()}` — the calibrator coefficients and "
            "decision thresholds this run produced, in the form the single test pass consumes.",
            "",
        ]

    report_path = out_dir / "session2_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {report_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
