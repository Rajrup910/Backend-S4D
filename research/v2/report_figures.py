"""S39 -- report figures, built from CSVs/JSON only (no new computation).

Two figures for `V2_REPORT.md`: F1's compression gap C by cohort (the refuted central
hypothesis), and F3's under-40 False Reassurance Rate under Mondrian vs bipartite
conformal calibration (the one confirmed positive result). Both read already-frozen
artifacts and compute nothing new.

    $py -m research.v2.report_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "v2"
FIG_DIR = REPO_ROOT / "paper" / "v2" / "figures"


def figure_f1_compression_gap() -> Path:
    boot = json.loads((RESULTS_DIR / "bootstrap_intervals.json").read_text(encoding="utf-8"))
    rows = boot["F1"]["rows"]
    cohorts = [r["cohort"] for r in rows]
    c_vals = [r["C"] for r in rows]
    lo = [r["C"] - r["ci_lo"] for r in rows]
    hi = [r["ci_hi"] - r["C"] for r in rows]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.05, color="grey", linewidth=0.8, linestyle="--", label="MCID (0.05)")
    ax.axhline(-0.05, color="grey", linewidth=0.8, linestyle="--")
    ax.errorbar(cohorts, c_vals, yerr=[lo, hi], fmt="o", capsize=4, color="tab:blue")
    ax.set_ylabel("C (signed compression gap)")
    ax.set_title("F1: decision compression gap, under-40 band (NO-GO — S34/S39)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "f1_compression_gap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_f3_frr() -> Path:
    df = pd.read_csv(RESULTS_DIR / "conformal_subgroup_safety.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    width = 0.35
    x = range(len(df))
    ax.bar([i - width / 2 for i in x], df["frr_mondrian_point"], width, label="RAPS-Mondrian (7-class)")
    ax.bar([i + width / 2 for i in x], df["frr_bipartite_point"], width, label="RAPS-bipartite (band x escalation)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"alpha={a}" for a in df["alpha"]])
    ax.set_ylabel("FRR, under-40 band")
    ax.set_title("F3: False Reassurance Rate, HAM-OOF calibration half (GO at alpha=0.10)")
    for i, row in df.iterrows():
        marker = "*" if row["significant_holm"] else ""
        ax.text(i, max(row["frr_mondrian_point"], row["frr_bipartite_point"]) + 0.01,
                marker, ha="center", fontsize=14)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "f3_frr_under40.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    p1 = figure_f1_compression_gap()
    p2 = figure_f3_frr()
    print(f"wrote {p1.relative_to(REPO_ROOT)}")
    print(f"wrote {p2.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
