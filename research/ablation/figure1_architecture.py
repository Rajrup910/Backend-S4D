"""Session 5, Part A (figures): Figure 1, the pipeline architecture diagram.

This is a schematic of the deployed pipeline, not a generated result -- there is no
`results/` artifact to regenerate it from. It exists here (rather than being drawn once by
hand) so it stays reproducible and gets fixed alongside the pipeline it describes: if a
stage's name or count changes (e.g. archs added to the ensemble), this file changes with it.

Usage:
    python -m research.ablation.figure1_architecture
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ml.paths import REPO_ROOT, resolve

BACKBONES = ["ConvNeXt-T", "ConvNeXt-S", "DenseNet-121", "EffNet-B0", "EffNet-B3", "ResNet-50"]

STAGE_COLOR = "#4C72B0"
INPUT_COLOR = "#55A868"
DECISION_COLOR = "#C44E52"
BOX_TEXT_COLOR = "white"


def _box(ax, xy, w, h, text, color=STAGE_COLOR, fontsize=10, text_color=BOX_TEXT_COLOR):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor="black", facecolor=color, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=text_color, zorder=3, wrap=True)
    return patch


def _arrow(ax, start, end, **kwargs):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.3, color="black", zorder=1, **kwargs,
    )
    ax.add_patch(arrow)


def build_figure() -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # --- Inputs -----------------------------------------------------------------------
    _box(ax, (0.3, 6.6), 2.0, 0.9, "Dermoscopy /\nclinical image", color=INPUT_COLOR)
    _box(ax, (0.3, 5.2), 2.0, 0.9, "Metadata\n(age, sex, site)", color=INPUT_COLOR)

    # --- Backbone ensemble (6 frozen CNNs) ---------------------------------------------
    n = len(BACKBONES)
    bb_w, bb_h = 1.75, 0.55
    bb_x0 = 3.1
    bb_y_top = 7.3
    gap = 0.12
    for i, name in enumerate(BACKBONES):
        y = bb_y_top - i * (bb_h + gap)
        _box(ax, (bb_x0, y), bb_w, bb_h, name, fontsize=8.5)
        _arrow(ax, (2.3, 7.05), (bb_x0, y + bb_h / 2))

    # --- Metadata / fusion side path (fed only into the fusion comparison, not the
    #     deployed ensemble -- see CLAUDE.md: ARCHS is a fixed 6-CNN tuple) -------------
    fusion_x = bb_x0
    fusion_y = 0.4
    _box(ax, (fusion_x, fusion_y), bb_w, bb_h, "Gated fusion\n(ConvNeXt-T + meta)",
         color="#8172B2", fontsize=8)
    _arrow(ax, (2.3, 5.65), (fusion_x, fusion_y + bb_h / 2))

    # --- Soft-vote ensemble --------------------------------------------------------------
    vote_x = 5.6
    _box(ax, (vote_x, 3.7), 1.9, 1.1, "Uniform\nsoft-vote\n(K=6)", color=STAGE_COLOR)
    for i in range(n):
        y = bb_y_top - i * (bb_h + gap) + bb_h / 2
        _arrow(ax, (bb_x0 + bb_w, y), (vote_x, 4.25))

    # --- TTA ------------------------------------------------------------------------------
    tta_x = 8.0
    _box(ax, (tta_x, 3.7), 1.7, 1.1, "24-view\ndihedral x\nscale TTA", color=STAGE_COLOR)
    _arrow(ax, (vote_x + 1.9, 4.25), (tta_x, 4.25))

    # --- Dirichlet calibration -------------------------------------------------------------
    cal_x = 10.1
    _box(ax, (cal_x, 3.7), 1.7, 1.1, "Dirichlet\ncalibration\n(fit on val)", color=STAGE_COLOR)
    _arrow(ax, (tta_x + 1.7, 4.25), (cal_x, 4.25))

    # --- Downstream decision layers (two independent policies off calibration, not a
    #     sequential chain -- routed with a curved connector so the arrow to the lower box
    #     visibly bows around the upper one instead of appearing to cut through it) ---------
    dec_x = 10.1
    _box(ax, (dec_x, 2.7), 1.7, 1.0, "Margin-based\nselective\nabstention", color=DECISION_COLOR)
    _box(ax, (dec_x, 0.6), 1.7, 1.0, "Class-conditional\nconformal sets\n(LAC / APS / RAPS)", color=DECISION_COLOR)
    _arrow(ax, (cal_x + 0.4, 3.7), (dec_x + 0.4, 3.2))
    _arrow(ax, (cal_x + 1.3, 3.7), (dec_x + 1.3, 1.1), connectionstyle="arc3,rad=0.35")

    ax.text(12.85, 4.25, "Diagnosis\n(7-class)", ha="left", va="center", fontsize=10)
    _arrow(ax, (11.8, 4.25), (12.75, 4.25))
    ax.text(12.85, 3.2, "Refer for\nbiopsy", ha="left", va="center", fontsize=10)
    _arrow(ax, (11.8, 3.2), (12.75, 3.2))
    ax.text(12.85, 1.1, "Prediction\nset", ha="left", va="center", fontsize=10)
    _arrow(ax, (11.8, 1.1), (12.75, 1.1))

    legend_handles = [
        mpatches.Patch(color=INPUT_COLOR, label="Input"),
        mpatches.Patch(color=STAGE_COLOR, label="Pipeline stage"),
        mpatches.Patch(color="#8172B2", label="Standalone comparison (not in deployed ensemble)"),
        mpatches.Patch(color=DECISION_COLOR, label="Clinical decision layer"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.0, -0.02),
              fontsize=8.5, frameon=False, ncol=1)

    ax.set_title(
        "Cross-paradigmatic ensemble pipeline: 6 frozen CNN backbones -> soft-vote -> "
        "24-view TTA -> Dirichlet calibration -> selective abstention / conformal sets",
        fontsize=10.5, pad=14,
    )

    out_path = resolve("paper/figures/figure1_architecture.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_figure()
