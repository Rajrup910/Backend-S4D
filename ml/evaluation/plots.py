"""Figures for the report.

Matplotlib only, no seaborn styling tricks -- these end up in a PDF report and need to
stay legible in greyscale print.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display on a training box / CI

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    matrix: list[list[int]] | np.ndarray,
    class_codes: list[str],
    output_path: str | Path,
    title: str = "Confusion matrix",
) -> Path:
    """Two panels: raw counts, and row-normalised (per-class recall on the diagonal).

    The normalised panel is the one to put in the report. Raw counts are dominated by
    `nv` and hide how badly the rare classes do.
    """
    counts = np.asarray(matrix, dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    normalised = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    for ax, data, subtitle, fmt in (
        (axes[0], counts, "Counts", "{:.0f}"),
        (axes[1], normalised, "Row-normalised (recall on the diagonal)", "{:.2f}"),
    ):
        image = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max() if data.max() else 1)
        ax.set_xticks(range(len(class_codes)), class_codes, rotation=45, ha="right")
        ax.set_yticks(range(len(class_codes)), class_codes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(subtitle)

        threshold = data.max() / 2 if data.max() else 0.5
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j,
                    i,
                    fmt.format(data[i, j]),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if data[i, j] > threshold else "black",
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_reliability_diagram(
    bins: list[dict[str, Any]],
    output_path: str | Path,
    title: str = "Reliability diagram",
) -> Path:
    """Confidence vs accuracy. Bars below the diagonal mean overconfidence."""
    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(6.5, 7), height_ratios=[3, 1], sharex=True
    )

    centres = [(b["lower"] + b["upper"]) / 2 for b in bins]
    accuracies = [b["accuracy"] for b in bins]
    counts = [b["count"] for b in bins]
    width = (bins[0]["upper"] - bins[0]["lower"]) * 0.9 if bins else 0.1

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    ax.bar(centres, accuracies, width=width, edgecolor="black", alpha=0.75, label="observed accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    ax_hist.bar(centres, counts, width=width, edgecolor="black", alpha=0.75, color="grey")
    ax_hist.set_xlabel("Model confidence")
    ax_hist.set_ylabel("Images")
    ax_hist.set_xlim(0, 1)
    ax_hist.grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, output_path)


def plot_training_curves(
    history: list[dict[str, Any]],
    output_path: str | Path,
    title: str = "Training",
) -> Path:
    """Loss and validation macro-F1 per epoch, with the stage transition marked."""
    if not history:
        raise ValueError("history is empty")

    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(13, 5))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], marker="o", ms=3, label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], marker="s", ms=3, label="validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_metric.plot(epochs, [h["val_macro_f1"] for h in history], marker="o", ms=3, label="val macro F1")
    ax_metric.plot(
        epochs, [h["val_balanced_accuracy"] for h in history], marker="s", ms=3, label="val balanced acc"
    )
    ax_metric.plot(
        epochs,
        [h["val_accuracy"] for h in history],
        marker="^",
        ms=3,
        linestyle=":",
        label="val accuracy",
    )
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Score")
    ax_metric.set_title("Validation metrics")
    ax_metric.legend()
    ax_metric.grid(alpha=0.3)

    # Mark where the backbone was unfrozen -- the loss usually jumps there, and an
    # examiner will ask what the discontinuity is.
    stages = [h.get("stage") for h in history]
    if "head" in stages and "finetune" in stages:
        transition = epochs[stages.index("finetune")]
        for ax in (ax_loss, ax_metric):
            ax.axvline(transition - 0.5, color="red", linestyle="--", alpha=0.6, linewidth=1)
            ax.text(
                transition - 0.4,
                ax.get_ylim()[1] * 0.97,
                "backbone unfrozen",
                fontsize=8,
                color="red",
                va="top",
            )

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_model_comparison(
    results: dict[str, dict[str, Any]],
    output_path: str | Path,
    title: str = "ResNet-50 vs EfficientNet-B0",
) -> Path:
    """Side-by-side quality and cost, which is the whole point of the comparison."""
    archs = list(results)
    fig, (ax_quality, ax_cost) = plt.subplots(1, 2, figsize=(13, 5))

    metric_names = ["macro_f1", "balanced_accuracy", "weighted_f1", "accuracy"]
    labels = ["Macro F1", "Balanced acc", "Weighted F1", "Accuracy"]
    positions = np.arange(len(metric_names))
    width = 0.8 / max(len(archs), 1)

    for i, arch in enumerate(archs):
        metrics = results[arch]["metrics"]
        values = [metrics[name] for name in metric_names]
        offset = (i - (len(archs) - 1) / 2) * width
        bars = ax_quality.bar(positions + offset, values, width, label=arch)
        ax_quality.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

    ax_quality.set_xticks(positions, labels, rotation=15)
    ax_quality.set_ylim(0, 1.05)
    ax_quality.set_ylabel("Score")
    ax_quality.set_title("Quality")
    ax_quality.legend()
    ax_quality.grid(alpha=0.3, axis="y")

    cost_names = ["params_millions", "model_size_mb", "latency_ms"]
    cost_labels = ["Parameters (M)", "Size (MB)", "Latency (ms)"]
    positions = np.arange(len(cost_names))
    for i, arch in enumerate(archs):
        meta = results[arch]["meta"]
        values = [
            meta["params"] / 1e6,
            meta["model_size_mb"],
            meta["latency"]["mean_ms"],
        ]
        offset = (i - (len(archs) - 1) / 2) * width
        bars = ax_cost.bar(positions + offset, values, width, label=arch)
        ax_cost.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)

    ax_cost.set_xticks(positions, cost_labels, rotation=15)
    ax_cost.set_ylabel("Lower is better")
    ax_cost.set_title("Computational cost")
    ax_cost.legend()
    ax_cost.grid(alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, output_path)


def _save(fig: plt.Figure, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
