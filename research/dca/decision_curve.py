"""Decision Curve Analysis on the binary "needs escalation" screening view.

Net Benefit(p_t) = TP/N - FP/N * (p_t / (1 - p_t))

`p_t` is the threshold probability at which a clinician would act (e.g. 0.10 means "refer
if I think there's a >=10% chance this needs escalation"). The `p_t/(1-p_t)` term is the
exchange rate between one false positive (an unnecessary referral) and one true positive
at that risk tolerance. A model beats "Treat All" only where its curve sits above it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def net_benefit(y_true_escalate: np.ndarray, escalation_score: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n = len(y_true_escalate)
    benefits = np.empty(len(thresholds))
    for i, p_t in enumerate(thresholds):
        predicted_positive = escalation_score > p_t
        true_positive = np.sum(predicted_positive & y_true_escalate)
        false_positive = np.sum(predicted_positive & ~y_true_escalate)
        exchange_rate = p_t / (1.0 - p_t)
        benefits[i] = true_positive / n - (false_positive / n) * exchange_rate
    return benefits


def net_benefit_treat_all(y_true_escalate: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n = len(y_true_escalate)
    prevalence = np.sum(y_true_escalate) / n
    exchange_rate = thresholds / (1.0 - thresholds)
    return prevalence - (1.0 - prevalence) * exchange_rate


def plot_decision_curves(
    thresholds: np.ndarray,
    curves: dict[str, np.ndarray],
    output_path: str | Path,
    title: str = "Decision Curve Analysis — needs-escalation screening",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, values in curves.items():
        style = "--" if label in ("Treat All", "Treat None") else "-"
        ax.plot(thresholds, values, style, label=label, linewidth=2 if style == "-" else 1.5)

    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Threshold probability $p_t$")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(bottom=min(-0.05, float(min(v.min() for v in curves.values()))))
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
