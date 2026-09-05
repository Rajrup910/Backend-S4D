"""Figures for Phase 4: risk-coverage curves and the clinical abstention trade-off."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a training box / CI

import matplotlib.pyplot as plt
import numpy as np

from ml.evaluation.plots import _save
from research.selective.risk_coverage import SelectivePoint, risk_coverage_curve


def plot_risk_coverage(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: str | Path,
    title: str = "Risk-coverage",
    min_coverage: float = 0.5,
) -> Path:
    """Selective risk against coverage for each uncertainty score.

    A score that ranks errors well drops steeply as coverage falls; a useless score
    traces a flat line at the model's overall error rate. Only coverage above
    `min_coverage` is drawn — the low-coverage tail is computed from a handful of images
    and its wild swings are sampling noise, not a property worth reading.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (coverage, risk) in curves.items():
        visible = coverage >= min_coverage
        ax.plot(coverage[visible], risk[visible], linewidth=1.6, label=name)

    ax.set_xlabel("Coverage (fraction of cases the system answers)")
    ax.set_ylabel("Selective risk (error rate on answered cases)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_clinical_tradeoff(
    points: list[SelectivePoint], output_path: str | Path, title: str = "Abstention trade-off"
) -> Path:
    """Macro-F1 and missed serious cases as the referral rate rises.

    The two axes are the trade the clinic actually makes: every point moved right is more
    patients told to come in, bought with fewer missed malignancies on the left axis.
    """
    rates = [p.abstention_rate * 100 for p in points]
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(rates, [p.macro_f1 for p in points], "o-", color="tab:blue", label="Macro-F1 (retained)")
    ax.plot(
        rates,
        [p.escalation_sensitivity for p in points],
        "s-",
        color="tab:green",
        label="Escalation sensitivity (retained)",
    )
    ax.set_xlabel("Target abstention rate (%), threshold fitted on validation")
    ax.set_ylabel("Score on retained cases")
    ax.grid(alpha=0.3)

    ax_right = ax.twinx()
    ax_right.bar(
        rates,
        [p.missed_serious for p in points],
        width=1.6,
        alpha=0.25,
        color="tab:red",
        label="Missed serious (retained)",
    )
    ax_right.set_ylabel("Missed serious cases among retained")

    handles, labels = ax.get_legend_handles_labels()
    bars, bar_labels = ax_right.get_legend_handles_labels()
    ax.legend(handles + bars, labels + bar_labels, loc="lower left", fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, output_path)


def build_curves(
    y_true: np.ndarray, y_pred: np.ndarray, scores: dict[str, np.ndarray]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {name: risk_coverage_curve(y_true, y_pred, value) for name, value in scores.items()}
