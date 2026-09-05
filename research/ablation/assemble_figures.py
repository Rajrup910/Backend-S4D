"""Session 5, Part A (figures): assemble the manuscript's Figure 2 and copy every other
already-generated plot into `paper/figures/`, per Finding L4 (top-level `results/` /
`paper/figures/` were empty despite hard rule 4).

Figure 2 (reliability before/after Dirichlet) is the only composite -- everything else
(DCA, risk-coverage, class-conditional conformal coverage, Grad-CAM overlays) already
exists as a standalone plot from an earlier session and just needs to live under
`paper/figures/` instead of scattered across `research/*/results/`.

Usage:
    python -m research.ablation.assemble_figures
"""

from __future__ import annotations

import shutil

from PIL import Image, ImageDraw, ImageFont

from ml.paths import REPO_ROOT, resolve


def build_figure2_reliability() -> None:
    """Side-by-side reliability diagram: uncalibrated vs Dirichlet-calibrated (val-fitted)."""
    left = Image.open(resolve("research/calibration/results/reliability_uncalibrated.png"))
    right = Image.open(resolve("research/calibration/results/reliability_dirichlet.png"))

    label_h = 50
    gap = 20
    width = left.width + right.width + gap
    height = max(left.height, right.height) + label_h

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()

    draw.text((left.width // 2 - 90, 10), "(a) Uncalibrated", fill="black", font=font)
    draw.text((left.width + gap + right.width // 2 - 130, 10), "(b) Dirichlet-calibrated", fill="black", font=font)

    canvas.paste(left, (0, label_h))
    canvas.paste(right, (left.width + gap, label_h))

    out_path = resolve("paper/figures/figure2_reliability.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Wrote {out_path.relative_to(REPO_ROOT)} ({canvas.size[0]}x{canvas.size[1]})")


def build_figure4_selective() -> None:
    """Session 15: risk-coverage and the abstention trade-off as one two-panel float.

    These were Figures 4 and 6 of the S10 draft and they tell one story twice -- the same
    sweep over the same abstention thresholds, read once as risk against coverage and once
    as the safety/workload trade. Two single-column floats cost roughly half a page more
    than one full-width float, and the paper is over its page budget, so they are merged
    here rather than dropped: no panel loses a curve.
    """
    left = Image.open(resolve("paper/figures/figure4_risk_coverage.png"))
    right = Image.open(resolve("paper/figures/figure6_abstention_tradeoff.png"))

    label_h = 50
    gap = 20
    width = left.width + right.width + gap
    height = max(left.height, right.height) + label_h

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()

    draw.text((left.width // 2 - 130, 10), "(a) Risk-coverage", fill="black", font=font)
    draw.text((left.width + gap + right.width // 2 - 170, 10),
              "(b) Safety / workload trade-off", fill="black", font=font)

    canvas.paste(left, (0, label_h))
    canvas.paste(right, (left.width + gap, label_h))

    out_path = resolve("paper/figures/figure4_selective.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Wrote {out_path.relative_to(REPO_ROOT)} ({canvas.size[0]}x{canvas.size[1]})")


def build_figure7_gradcam() -> None:
    """2x4 Grad-CAM montage: a confident correct case per row-1 class, a characteristic error below.

    Panels are chosen deterministically from `gradcam_stats.csv` -- highest-confidence correct
    case for the top row, highest-confidence error for the bottom row -- so the figure is a
    reproducible artifact rather than a hand-picked selection. The bottom row is deliberately
    the model's *confident* mistakes: those are the ones abstention cannot catch (Sec. IV-E).
    """
    import pandas as pd

    gradcam_dir = resolve("paper/figures/gradcam")
    stats = pd.read_csv(gradcam_dir / "gradcam_stats.csv")

    def pick(true_code: str, want_correct: bool) -> pd.Series | None:
        subset = stats[(stats["true_code"] == true_code) & (stats["correct"] == want_correct)]
        if subset.empty:
            return None
        return subset.sort_values("confidence", ascending=False).iloc[0]

    top_classes = ["akiec", "nv", "vasc", "df"]      # classes the model handles well
    bottom_classes = ["mel", "bkl", "bcc", "df"]      # classes it confuses
    selection = [pick(c, True) for c in top_classes] + [pick(c, False) for c in bottom_classes]

    def overlay_path(row: pd.Series):
        tag = "ok" if row["correct"] else "ERR"
        name = (f"{row['true_code']}_{row['image_id']}_pred-{row['pred_code']}_"
                f"{row['confidence']:.2f}_{tag}.png")
        return gradcam_dir / name

    tiles, captions = [], []
    for row in selection:
        if row is None:
            continue
        path = overlay_path(row)
        if not path.exists():
            print(f"  Figure 7: missing overlay {path.name}, panel skipped")
            continue
        tiles.append(Image.open(path).convert("RGB"))
        captions.append(f"{row['true_code']} -> {row['pred_code']} ({row['confidence']:.2f})")

    if not tiles:
        print("  Figure 7: no overlays available, skipped")
        return

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    tile_w = min(t.width for t in tiles)
    tile_h = min(t.height for t in tiles)
    caption_h = 34
    pad = 8

    canvas = Image.new(
        "RGB",
        (cols * tile_w + (cols + 1) * pad, rows * (tile_h + caption_h) + (rows + 1) * pad),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    for index, (tile, caption) in enumerate(zip(tiles, captions)):
        r, c = divmod(index, cols)
        x = pad + c * (tile_w + pad)
        y = pad + r * (tile_h + caption_h + pad)
        canvas.paste(tile.resize((tile_w, tile_h)), (x, y))
        draw.text((x + 4, y + tile_h + 6), caption, fill="black", font=font)

    out_path = resolve("paper/figures/figure7_gradcam.png")
    canvas.save(out_path)
    print(f"Wrote {out_path.relative_to(REPO_ROOT)} ({canvas.size[0]}x{canvas.size[1]}, "
          f"{len(tiles)} panels)")


COPY_MAP = {
    "research/dca/results/decision_curve.png": "paper/figures/figure3_decision_curve_analysis.png",
    "research/selective/results/risk_coverage.png": "paper/figures/figure4_risk_coverage.png",
    "research/selective/results/abstention_tradeoff.png": "paper/figures/figure6_abstention_tradeoff.png",
    "research/conformal/results/class_conditional_coverage.png": "paper/figures/figure5_conformal_coverage.png",
}


def copy_existing_figures() -> None:
    for src, dst in COPY_MAP.items():
        src_path = resolve(src)
        dst_path = resolve(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        print(f"Copied {src} -> {dst_path.relative_to(REPO_ROOT)}")


def main() -> int:
    build_figure2_reliability()
    copy_existing_figures()          # the two selective panels must exist before the merge
    build_figure4_selective()
    build_figure7_gradcam()
    print(
        "\nNote: Figure 1 (architecture diagram) is drawn by "
        "research/ablation/figure1_architecture.py, not by this script."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
