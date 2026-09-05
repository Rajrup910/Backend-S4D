"""Compare TTA against the non-TTA baseline and Session-1 winning ensemble, test split once.

Two comparisons, both single-read on test:
  1. ConvNeXt-Tiny alone: TTA-pooled probabilities vs. the plain Session-0 predictions.
  2. The Session-1 winner (uniform soft-vote across all 6 backbones): rebuilt from the
     TTA-pooled probabilities of all 6, vs. the original non-TTA ensemble.

Usage:
    python -m research.tta.evaluate_tta
"""

from __future__ import annotations

from ml.evaluation.metrics import compute_metrics
from ml.paths import REPO_ROOT, resolve
from research.ensembling.data import load_split_matrix
from research.ensembling.methods import soft_vote_arithmetic
from research.experiment_log import log_experiment

BASELINE_ARCH = "convnext_tiny"


def main() -> int:
    test_plain = load_split_matrix("test", predictions_dir="research/predictions")
    test_tta = load_split_matrix("test", predictions_dir="research/predictions_tta")

    baseline_plain = test_plain.probs_for(BASELINE_ARCH)
    baseline_tta = test_tta.probs_for(BASELINE_ARCH)
    ensemble_plain = soft_vote_arithmetic(test_plain.probs)
    ensemble_tta = soft_vote_arithmetic(test_tta.probs)

    rows = [
        (f"{BASELINE_ARCH}_no_tta", baseline_plain),
        (f"{BASELINE_ARCH}_tta", baseline_tta),
        ("ensemble_no_tta", ensemble_plain),
        ("ensemble_tta", ensemble_tta),
    ]

    lines = [
        "# Phase 2 — TTA Results (Session 2)",
        "",
        "24-view (8-fold dihedral x 3 scales) TTA with entropy-weighted pooling, test split, single read.",
        "",
        "| Config | Macro-F1 | Balanced Acc | Escalation Sens. | Missed Serious | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, probs in rows:
        preds = probs.argmax(axis=1)
        metrics = compute_metrics(test_plain.y_true, preds, probs)
        lines.append(
            f"| {name} | {metrics['macro_f1']:.4f} | {metrics['balanced_accuracy']:.4f} | "
            f"{metrics['clinical']['binary_sensitivity']:.4f} | {metrics['clinical']['missed_serious_cases']} | "
            f"{metrics['expected_calibration_error']:.4f} |"
        )
        log_experiment({
            "session": "session2", "method": name, "split": "test",
            "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["balanced_accuracy"],
            "weighted_f1": metrics["weighted_f1"], "macro_roc_auc": metrics.get("roc_auc_macro", ""),
            "ece": metrics["expected_calibration_error"],
            "escalation_sens": metrics["clinical"]["binary_sensitivity"],
            "missed_serious": metrics["clinical"]["missed_serious_cases"],
            "notes": "24-view TTA, entropy-weighted pooling" if "tta" in name and "no_tta" not in name else "no TTA",
        })
        print(f"[{name:20s}] macro_f1={metrics['macro_f1']:.4f}  "
              f"escalation_sens={metrics['clinical']['binary_sensitivity']:.4f}  "
              f"ece={metrics['expected_calibration_error']:.4f}")

    out_dir = resolve("research/tta/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {(out_dir / 'report.md').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
