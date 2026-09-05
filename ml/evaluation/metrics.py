"""Metric computation shared by training and evaluation.

Accuracy is deliberately *not* the headline number anywhere in this project. With `nv`
at roughly two thirds of HAM10000, a model that predicts "mole" for every image scores
about 67% accuracy while being clinically useless -- it would miss every melanoma. The
metrics that matter here are macro F1, balanced accuracy, and per-class recall on the
malignant classes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ml.paths import load_class_mapping


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """Full metric set for a set of predictions.

    Args:
        y_true: integer labels, shape (N,)
        y_pred: predicted integer labels, shape (N,)
        y_prob: softmax probabilities, shape (N, C). Needed for ROC-AUC and calibration.
    """
    mapping = load_class_mapping()
    labels = list(range(mapping.num_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = {}
    for skin_class in mapping.classes:
        i = skin_class.index
        per_class[skin_class.code] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "malignancy": skin_class.malignancy,
        }

    # `labels=` is passed explicitly so the macro average runs over all declared classes,
    # matching `per_class` below. Without it sklearn averages only over classes present in
    # y_true or y_pred, which makes macro F1 incomparable between two models that happen to
    # predict different subsets of the classes -- exactly the comparison this project makes.
    results: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_codes": list(mapping.codes),
        "num_samples": int(len(y_true)),
    }

    results["clinical"] = clinical_summary(y_true, y_pred, per_class)

    if y_prob is not None:
        results.update(probability_metrics(y_true, y_prob, labels))

    return results


def clinical_summary(
    y_true: np.ndarray, y_pred: np.ndarray, per_class: dict[str, Any]
) -> dict[str, Any]:
    """Screening-oriented view: how often did we miss something that needed attention?

    A false negative -- calling a melanoma a mole -- is the failure mode that matters in a
    screening tool. This collapses the 7-way problem into "needs escalation vs does not"
    and reports sensitivity on that, alongside per-class recall for the malignant classes.
    """
    mapping = load_class_mapping()
    escalating = {c.index for c in mapping.classes if c.needs_escalation}

    true_serious = np.isin(y_true, list(escalating))
    pred_serious = np.isin(y_pred, list(escalating))

    true_positive = int(np.sum(true_serious & pred_serious))
    false_negative = int(np.sum(true_serious & ~pred_serious))
    false_positive = int(np.sum(~true_serious & pred_serious))
    true_negative = int(np.sum(~true_serious & ~pred_serious))

    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) else 0.0

    return {
        "escalating_classes": sorted(mapping.by_index(i).code for i in escalating),
        "binary_sensitivity": float(sensitivity),
        "binary_specificity": float(specificity),
        "missed_serious_cases": false_negative,
        "false_alarms": false_positive,
        "malignant_recall": {
            c.code: per_class[c.code]["recall"] for c in mapping.classes if c.is_malignant
        },
    }


def probability_metrics(y_true: np.ndarray, y_prob: np.ndarray, labels: list[int]) -> dict[str, Any]:
    """ROC-AUC and calibration, which need probabilities rather than hard predictions."""
    results: dict[str, Any] = {}

    present = np.unique(y_true)
    if len(present) < 2:
        results["roc_auc_macro"] = None
        results["roc_auc_note"] = "undefined: fewer than two classes present"
    else:
        try:
            results["roc_auc_macro"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
            )
            results["roc_auc_weighted"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted", labels=labels)
            )
        except ValueError as exc:
            results["roc_auc_macro"] = None
            results["roc_auc_note"] = f"undefined: {exc}"

    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    results["expected_calibration_error"] = float(expected_calibration_error(confidences, correct))
    results["mean_confidence"] = float(confidences.mean())
    results["mean_accuracy"] = float(correct.mean())
    # A positive gap means the model is overconfident: it claims more certainty than it earns.
    results["confidence_gap"] = float(confidences.mean() - correct.mean())
    results["reliability_bins"] = reliability_bins(confidences, correct)

    return results


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, num_bins: int = 15
) -> float:
    """ECE: average gap between confidence and accuracy, weighted by bin population.

    0 means perfectly calibrated -- among images the model called 80% confident, exactly
    80% were right. Neural networks are typically overconfident, so expect a positive value.
    """
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidences > lower) & (confidences <= upper)
        if not in_bin.any():
            continue
        weight = in_bin.mean()
        error += weight * abs(correct[in_bin].mean() - confidences[in_bin].mean())
    return error


def reliability_bins(
    confidences: np.ndarray, correct: np.ndarray, num_bins: int = 10
) -> list[dict[str, float]]:
    """Data for a reliability diagram (master spec, section 16)."""
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidences > lower) & (confidences <= upper)
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(in_bin.sum()),
                "mean_confidence": float(confidences[in_bin].mean()) if in_bin.any() else 0.0,
                "accuracy": float(correct[in_bin].mean()) if in_bin.any() else 0.0,
            }
        )
    return bins


def format_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """sklearn's text classification report, with our class codes as labels."""
    mapping = load_class_mapping()
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(mapping.num_classes)),
        target_names=list(mapping.codes),
        zero_division=0,
        digits=4,
    )


def summarise(metrics: dict[str, Any]) -> str:
    """Compact console summary. Accuracy is shown last, on purpose."""
    clinical = metrics["clinical"]
    lines = [
        f"  macro F1            {metrics['macro_f1']:.4f}",
        f"  balanced accuracy   {metrics['balanced_accuracy']:.4f}",
        f"  weighted F1         {metrics['weighted_f1']:.4f}",
        f"  accuracy            {metrics['accuracy']:.4f}   <- not the headline number",
        "",
        f"  sensitivity to lesions needing escalation  {clinical['binary_sensitivity']:.4f}",
        f"  missed serious cases                       {clinical['missed_serious_cases']}",
    ]
    for code, recall in clinical["malignant_recall"].items():
        lines.append(f"  recall on {code:<5}                            {recall:.4f}")
    if metrics.get("roc_auc_macro") is not None:
        lines.append(f"  macro ROC-AUC                              {metrics['roc_auc_macro']:.4f}")
    if "expected_calibration_error" in metrics:
        lines.append(f"  expected calibration error                 {metrics['expected_calibration_error']:.4f}")
    return "\n".join(lines)
