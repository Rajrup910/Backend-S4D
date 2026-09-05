"""Figures for conformal prediction: per-class coverage against the target."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a training box / CI

import matplotlib.pyplot as plt
import numpy as np

from ml.evaluation.plots import _save
from ml.paths import load_class_mapping
from research.conformal.metrics import SetMetrics


def plot_class_coverage(
    variants: dict[str, SetMetrics], target: float, output_path: str | Path
) -> Path:
    """Per-class coverage bars for each calibration variant, with the target line.

    The point the figure has to make: a marginal guarantee can sit on the target line
    overall while individual classes fall well below it, and class-conditional
    calibration is what pulls them back up — visibly, per class, rather than on average.
    """
    mapping = load_class_mapping()
    codes = [c.code for c in mapping.classes if c.code in next(iter(variants.values())).per_class]
    escalating = {c.code for c in mapping.classes if c.needs_escalation}

    x = np.arange(len(codes))
    width = 0.8 / max(1, len(variants))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, metrics) in enumerate(variants.items()):
        values = [metrics.per_class[code]["coverage"] for code in codes]
        ax.bar(x + i * width - 0.4 + width / 2, values, width=width, label=name, edgecolor="black")

    ax.axhline(target, color="black", linestyle="--", linewidth=1.2, label=f"target {target:.0%}")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{code}*" if code in escalating else code for code in codes]
    )
    ax.set_ylabel("Coverage (truth in the prediction set)")
    ax.set_xlabel("True class  (* = escalating)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class coverage: marginal vs class-conditional calibration")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return _save(fig, output_path)
