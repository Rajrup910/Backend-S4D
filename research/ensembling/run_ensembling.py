"""Session 1: fit and evaluate the Phase-1 ensembling algorithms.

For each method:
  1. An honest "val_oof" macro-F1 is computed. Parameter-free methods (soft-vote,
     rank-average) need no fitting, so this is just their direct val score. Fitted
     methods (Nelder-Mead weights, stacking, Caruana greedy) are scored via 5-fold
     stratified lesion-grouped CV *within* val, so the score used for method selection
     never comes from a model that saw the row it's scored on.
  2. The method is then refit once on the *full* val split and applied to test exactly
     once -- the single read the master spec requires.

Every val_oof and test result is logged to research/experiments.csv. A consolidated
report (diversity audit, per-method scores, bootstrap CI + McNemar vs. the ConvNeXt-Tiny
baseline for the winning method) is written to research/ensembling/report.md.

Usage:
    python -m research.ensembling.run_ensembling
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ml.evaluation.metrics import compute_metrics
from ml.paths import REPO_ROOT, resolve
from research.ensembling.data import ARCHS, load_split_matrix
from research.ensembling.diversity import diversity_report
from research.ensembling.methods import (
    apply_caruana,
    apply_nelder_mead,
    apply_stacking,
    fit_caruana_greedy,
    fit_nelder_mead_weights,
    fit_stacking,
    rank_average,
    soft_vote_arithmetic,
    soft_vote_geometric,
)
from research.ensembling.oof import oof_scores
from research.ensembling.stats import bootstrap_macro_f1_ci, mcnemar_test
from research.experiment_log import log_experiment

BASELINE_ARCH = "convnext_tiny"  # current single-model leader, Macro-F1 0.746 (CLAUDE.md)


def _metrics_row(y_true: np.ndarray, scores: np.ndarray) -> dict:
    preds = scores.argmax(axis=1)
    row_sums = scores.sum(axis=1, keepdims=True)
    probs = scores / np.where(row_sums == 0, 1.0, row_sums)
    return compute_metrics(y_true, preds, probs)


def evaluate_parameter_free(name: str, val, test, combine_fn) -> dict:
    val_scores = combine_fn(val.probs)
    val_metrics = _metrics_row(val.y_true, val_scores)
    test_scores = combine_fn(test.probs)
    test_metrics = _metrics_row(test.y_true, test_scores)
    return {
        "name": name,
        "val_oof_macro_f1": val_metrics["macro_f1"],  # no fitting => val score is already honest
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_preds": test_scores.argmax(axis=1),
        "notes": "no fitting; val_oof == direct val score",
    }


def evaluate_nelder_mead(val, test) -> dict:
    oof = oof_scores(
        val.probs, val.y_true, val.lesion_ids,
        fit_fn=fit_nelder_mead_weights, predict_fn=apply_nelder_mead,
    )
    val_oof_metrics = _metrics_row(val.y_true, oof)

    final_state = fit_nelder_mead_weights(val.probs, val.y_true)
    test_scores = apply_nelder_mead(final_state, test.probs)
    test_metrics = _metrics_row(test.y_true, test_scores)
    return {
        "name": "nelder_mead_simplex",
        "val_oof_macro_f1": val_oof_metrics["macro_f1"],
        "val_metrics": val_oof_metrics,
        "test_metrics": test_metrics,
        "test_preds": test_scores.argmax(axis=1),
        "notes": f"weights={np.round(final_state.weights, 3).tolist()}",
    }


def evaluate_stacking(val, test) -> dict:
    oof = oof_scores(
        val.logits, val.y_true, val.lesion_ids,
        fit_fn=fit_stacking, predict_fn=apply_stacking,
    )
    val_oof_metrics = _metrics_row(val.y_true, oof)

    final_state = fit_stacking(val.logits, val.y_true)
    test_scores = apply_stacking(final_state, test.logits)
    test_metrics = _metrics_row(test.y_true, test_scores)
    return {
        "name": "nonneg_stacking_ridge",
        "val_oof_macro_f1": val_oof_metrics["macro_f1"],
        "val_metrics": val_oof_metrics,
        "test_metrics": test_metrics,
        "test_preds": test_scores.argmax(axis=1),
        "notes": "one-vs-rest Ridge(positive=True) on K*C flattened logits, alpha=1.0",
    }


def evaluate_caruana(val, test, max_size: int = 20) -> dict:
    oof = oof_scores(
        val.probs, val.y_true, val.lesion_ids,
        fit_fn=lambda feat, y: fit_caruana_greedy(feat, y, max_size=max_size),
        predict_fn=apply_caruana,
    )
    val_oof_metrics = _metrics_row(val.y_true, oof)

    final_state = fit_caruana_greedy(val.probs, val.y_true, max_size=max_size)
    test_scores = apply_caruana(final_state, test.probs)
    test_metrics = _metrics_row(test.y_true, test_scores)
    best_size = int(np.argmax(final_state.curve)) + 1
    picks = [val.archs[i] for i in final_state.selected[:best_size]]
    return {
        "name": "caruana_greedy",
        "val_oof_macro_f1": val_oof_metrics["macro_f1"],
        "val_metrics": val_oof_metrics,
        "test_metrics": test_metrics,
        "test_preds": test_scores.argmax(axis=1),
        "notes": f"best_size={best_size} picks={picks}",
        "curve": final_state.curve,
    }


def evaluate_baseline(val, test, arch: str) -> dict:
    val_probs = val.probs_for(arch)
    val_metrics = compute_metrics(val.y_true, val_probs.argmax(axis=1), val_probs)
    test_probs = test.probs_for(arch)
    test_metrics = compute_metrics(test.y_true, test_probs.argmax(axis=1), test_probs)
    return {
        "name": f"{arch}_baseline",
        "val_oof_macro_f1": val_metrics["macro_f1"],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_preds": test_probs.argmax(axis=1),
        "notes": "single frozen model, no ensembling",
    }


def render_report(results: list[dict], diversity: dict, baseline_name: str,
                   winner: dict, bootstrap: dict, mcnemar: dict) -> str:
    lines = [
        "# Phase 1 — Ensembling Results (Session 1)",
        "",
        "## Diversity audit (val split)",
        "",
        f"- Mean pairwise disagreement: **{diversity['mean_disagreement']:.4f}**",
        f"- Mean Yule's Q: **{diversity['mean_yules_q']:.4f}** (lower / negative = more complementary)",
        f"- Mean double-fault ratio: **{diversity['mean_double_fault']:.4f}**",
        "",
        "## Method comparison",
        "",
        "| Method | val_oof Macro-F1 | test Macro-F1 | test Balanced Acc | test Escalation Sens. | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in sorted(results, key=lambda x: -x["test_metrics"]["macro_f1"]):
        tm = r["test_metrics"]
        lines.append(
            f"| {r['name']} | {r['val_oof_macro_f1']:.4f} | **{tm['macro_f1']:.4f}** | "
            f"{tm['balanced_accuracy']:.4f} | {tm['clinical']['binary_sensitivity']:.4f} | {r['notes']} |"
        )

    lines += [
        "",
        f"## Winner: `{winner['name']}` vs. `{baseline_name}` baseline (test split, single read)",
        "",
        f"- Bootstrap 95% CI (N=1000) for winner's test Macro-F1: "
        f"**{bootstrap['mean']:.4f}** [{bootstrap['ci_low']:.4f}, {bootstrap['ci_high']:.4f}]",
        f"- McNemar's test (winner vs. baseline, paired on test images): "
        f"chi2={mcnemar['statistic']:.3f}, p={mcnemar['p_value']:.4g} "
        f"(only-winner-correct={mcnemar['only_a_correct']}, only-baseline-correct={mcnemar['only_b_correct']})",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default="session1")
    args = parser.parse_args(argv)

    val = load_split_matrix("val")
    test = load_split_matrix("test")
    print(f"Loaded val ({val.num_samples} images) and test ({test.num_samples} images) "
          f"across {val.num_archs} architectures: {val.archs}")

    div = diversity_report(val)
    out_dir = resolve("research/ensembling/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    div["disagreement"].to_csv(out_dir / "diversity_disagreement.csv")
    div["yules_q"].to_csv(out_dir / "diversity_yules_q.csv")
    div["double_fault"].to_csv(out_dir / "diversity_double_fault.csv")
    print(f"Diversity: mean disagreement={div['mean_disagreement']:.4f}, "
          f"mean Yule's Q={div['mean_yules_q']:.4f}, mean double-fault={div['mean_double_fault']:.4f}")

    results = [
        evaluate_baseline(val, test, BASELINE_ARCH),
        evaluate_parameter_free("soft_vote_arithmetic", val, test, soft_vote_arithmetic),
        evaluate_parameter_free("soft_vote_geometric", val, test, soft_vote_geometric),
        evaluate_parameter_free("rank_average", val, test, rank_average),
        evaluate_nelder_mead(val, test),
        evaluate_stacking(val, test),
        evaluate_caruana(val, test),
    ]

    for r in results:
        print(f"[{r['name']:24s}] val_oof macro_f1={r['val_oof_macro_f1']:.4f}  "
              f"test macro_f1={r['test_metrics']['macro_f1']:.4f}")

        log_experiment({
            "session": args.session, "method": r["name"], "split": "val_oof",
            "macro_f1": r["val_oof_macro_f1"],
            "notes": r["notes"],
        })
        tm = r["test_metrics"]
        log_experiment({
            "session": args.session, "method": r["name"], "split": "test",
            "macro_f1": tm["macro_f1"], "accuracy": tm["accuracy"],
            "balanced_accuracy": tm["balanced_accuracy"], "weighted_f1": tm["weighted_f1"],
            "macro_roc_auc": tm.get("roc_auc_macro", ""), "ece": tm.get("expected_calibration_error", ""),
            "escalation_sens": tm["clinical"]["binary_sensitivity"],
            "missed_serious": tm["clinical"]["missed_serious_cases"],
            "notes": r["notes"],
        })

    baseline = next(r for r in results if r["name"] == f"{BASELINE_ARCH}_baseline")
    ensemble_candidates = [r for r in results if r is not baseline]
    winner = max(ensemble_candidates, key=lambda r: r["val_oof_macro_f1"])

    bootstrap = bootstrap_macro_f1_ci(test.y_true, winner["test_preds"])
    mcnemar = mcnemar_test(test.y_true, winner["test_preds"], baseline["test_preds"])
    p_value = mcnemar["p_value"]

    log_experiment({
        "session": args.session, "method": f"{winner['name']}_vs_{BASELINE_ARCH}", "split": "test",
        "macro_f1": winner["test_metrics"]["macro_f1"], "p_value_vs_baseline": p_value,
        "notes": f"McNemar chi2={mcnemar['statistic']:.3f}; bootstrap 95% CI "
                 f"[{bootstrap['ci_low']:.4f}, {bootstrap['ci_high']:.4f}]",
    })

    report = render_report(results, div, f"{BASELINE_ARCH}_baseline", winner, bootstrap, mcnemar)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"\nWinner (selected on val_oof, applied to test once): {winner['name']}")
    print(f"  test macro_f1={winner['test_metrics']['macro_f1']:.4f}  vs baseline "
          f"{baseline['test_metrics']['macro_f1']:.4f}  (McNemar p={p_value:.4g})")
    print(f"\nReport written to {(out_dir / 'report.md').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
