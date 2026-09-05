"""Generate and audit Grad-CAM heatmaps for a trained model.

Section 18 of the master specification is explicit: producing a heatmap and calling it
explainability is not enough. This script samples images from the test split, writes the
overlays, and computes aggregate statistics that flag heatmaps landing on the border
rather than on the lesion -- the signature of a model keying on vignetting, hairs, rulers
or dermatoscope framing instead of the lesion itself.

The output is a folder of overlays to inspect **by eye**, plus a report summarising where
the model tends to look. Neither proves the model reasons like a dermatologist. They
identify the cases worth examining, which is the honest claim.

Usage:
    python -m ml.explainability.generate_gradcam --checkpoint ml/checkpoints/efficientnet_b0_best.pt
    python -m ml.explainability.generate_gradcam --checkpoint ... --per-class 8 --only-errors
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm

from ml.explainability.gradcam import GradCAM, cam_statistics, describe_focus, overlay_heatmap
from ml.paths import REPO_ROOT, load_class_mapping, load_training_config, resolve
from ml.preprocessing.transforms import build_eval_transform
from ml.training.common import build_model_from_checkpoint, gradcam_target_layer, resolve_device

BORDER_MASS_WARNING = 0.5


def select_samples(
    manifest: pd.DataFrame,
    splits: pd.DataFrame,
    split: str,
    per_class: int,
    seed: int,
) -> pd.DataFrame:
    frame = manifest.merge(splits[["image_id", "split"]], on="image_id", validate="one_to_one")
    frame = frame[frame["split"] == split]
    if frame.empty:
        raise ValueError(f"no images in split {split!r}")
    # pandas >=2.2 excludes the grouping column from the frame passed into `.apply` unless
    # it's re-selected explicitly -- without `[frame.columns]` here, `class_code` silently
    # disappears from every group and the `sample.class_code` lookup below raises.
    return (
        frame.groupby("class_code", group_keys=False)[frame.columns]
        .apply(lambda group: group.sample(min(len(group), per_class), random_state=seed))
        .reset_index(drop=True)
    )


def render_report(
    rows: list[dict[str, Any]], meta: dict[str, Any], by_class: dict[str, list[dict[str, Any]]]
) -> str:
    frame = pd.DataFrame(rows)
    suspicious = frame[frame["border_mass_fraction"] > BORDER_MASS_WARNING]

    lines = [
        "# Grad-CAM Validation",
        "",
        f"**Generated:** {date.today().isoformat()}  ",
        f"**Model:** {meta['arch']} — `{meta['checkpoint']}`  ",
        f"**Target layer:** `{meta['target_layer']}` (last convolutional block)  ",
        f"**Split:** {meta['split']}  ",
        f"**Images:** {len(frame)}",
        "",
        "## Method",
        "",
        "Gradients of the predicted-class score are taken with respect to the feature maps of",
        "the last convolutional block, averaged spatially into per-channel importance weights,",
        "used to weight and sum those feature maps, passed through ReLU, normalised, and",
        "resized over the original image.",
        "",
        "## Where the model looks",
        "",
        "`border_mass_fraction` is the share of heatmap activation falling in the outer 15% frame",
        f"of the image. Values above **{BORDER_MASS_WARNING:.0%}** are flagged: lesions in this dataset",
        "are roughly centred, so a border-dominated heatmap suggests the model is responding to",
        "framing artefacts rather than the lesion.",
        "",
        "| Statistic | Value |",
        "|---|---:|",
        f"| Mean border mass fraction | {frame['border_mass_fraction'].mean():.3f} |",
        f"| Median border mass fraction | {frame['border_mass_fraction'].median():.3f} |",
        f"| Images flagged (> {BORDER_MASS_WARNING:.0%}) | **{len(suspicious)} / {len(frame)}** |",
        f"| Mean peak offset from centre | {frame['peak_offset_from_centre'].mean():.3f} |",
        f"| Degenerate (all-zero) maps | {int(frame['is_degenerate'].sum())} |",
        "",
        "## Per class",
        "",
        "| Class | Images | Mean border mass | Flagged | Mean confidence |",
        "|---|---:|---:|---:|---:|",
    ]
    for code, entries in by_class.items():
        sub = pd.DataFrame(entries)
        flagged = int((sub["border_mass_fraction"] > BORDER_MASS_WARNING).sum())
        lines.append(
            f"| `{code}` | {len(sub)} | {sub['border_mass_fraction'].mean():.3f} | "
            f"{flagged} | {sub['confidence'].mean():.3f} |"
        )

    if len(suspicious):
        lines += [
            "",
            "## Flagged images — inspect these by eye",
            "",
            "| Image | True | Predicted | Confidence | Border mass |",
            "|---|---|---|---:|---:|",
        ]
        for row in suspicious.sort_values("border_mass_fraction", ascending=False).head(20).itertuples():
            lines.append(
                f"| `{row.image_id}` | {row.true_code} | {row.pred_code} | "
                f"{row.confidence:.3f} | {row.border_mass_fraction:.3f} |"
            )

    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Grad-CAM shows which regions influenced the score. It does **not** show that the model",
        "  used medically correct features, and it must not be presented as proof that it did.",
        "- The map is computed at the last conv layer's resolution (typically 7x7 for a 224x224",
        "  input) and upsampled. The apparent precision of the overlay is interpolation, not",
        "  fine-grained localisation.",
        "- A convincing-looking heatmap on a wrong prediction is common. Always read the heatmap",
        "  together with the predicted class and its confidence.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    config = load_training_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap opacity in the overlay.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="")
    parser.add_argument(
        "--only-errors",
        action="store_true",
        help="Keep only misclassified images -- the most informative ones to inspect.",
    )
    args = parser.parse_args(argv)

    checkpoint_path = resolve(args.checkpoint)
    if not checkpoint_path.is_file():
        print(f"ERROR: no checkpoint at {checkpoint_path}", file=sys.stderr)
        return 1

    manifest_path = resolve(config["data"]["manifest"])
    splits_path = resolve(config["data"]["splits"])
    if not (manifest_path.is_file() and splits_path.is_file()):
        print("ERROR: run the dataset pipeline first (prepare_dataset, split_dataset).", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    model, payload = build_model_from_checkpoint(checkpoint_path, device)
    arch = payload["arch"]
    image_size = payload.get("image_size", config["data"]["image_size"])
    transform = build_eval_transform(image_size)
    mapping = load_class_mapping()

    samples = select_samples(
        pd.read_csv(manifest_path),
        pd.read_csv(splits_path),
        args.split,
        args.per_class,
        config["seed"],
    )
    print(f"Selected {len(samples)} images from the {args.split} split ({args.per_class} per class)")

    out_dir = resolve(args.out_dir) if args.out_dir else resolve(config["paths"]["results_dir"]) / arch / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_layer = gradcam_target_layer(model, arch)
    rows: list[dict[str, Any]] = []
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with GradCAM(model, target_layer) as cam_engine:
        for sample in tqdm(samples.itertuples(index=False), total=len(samples), desc="grad-cam"):
            image = Image.open(REPO_ROOT / sample.path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

            cam, class_index, probabilities = cam_engine(tensor)
            predicted = mapping.by_index(class_index)
            correct = predicted.code == sample.class_code

            if args.only_errors and correct:
                continue

            stats = cam_statistics(cam)
            row = {
                "image_id": sample.image_id,
                "true_code": sample.class_code,
                "pred_code": predicted.code,
                "correct": correct,
                "confidence": float(probabilities[class_index]),
                "focus": describe_focus(stats),
                **{k: v for k, v in stats.items() if k not in ("resolution", "peak_position")},
            }
            rows.append(row)
            by_class[sample.class_code].append(row)

            status = "ok" if correct else "ERR"
            filename = (
                f"{sample.class_code}_{sample.image_id}_pred-{predicted.code}"
                f"_{probabilities[class_index]:.2f}_{status}.png"
            )
            overlay_heatmap(image, cam, alpha=args.alpha).save(out_dir / filename)

    if not rows:
        print("No images matched the selection (did --only-errors exclude everything?)")
        return 0

    meta = {
        "arch": arch,
        "checkpoint": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
        "target_layer": type(target_layer).__name__,
        "split": args.split,
    }
    pd.DataFrame(rows).to_csv(out_dir / "gradcam_stats.csv", index=False)
    (out_dir / "gradcam_report.md").write_text(
        render_report(rows, meta, dict(by_class)), encoding="utf-8"
    )
    (out_dir / "gradcam_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    frame = pd.DataFrame(rows)
    flagged = int((frame["border_mass_fraction"] > BORDER_MASS_WARNING).sum())
    print(f"\nWrote {len(rows)} overlays to {out_dir.relative_to(REPO_ROOT)}/")
    print(f"Mean border mass fraction: {frame['border_mass_fraction'].mean():.3f}")
    print(f"Flagged (border-dominated): {flagged}/{len(frame)}")
    if flagged:
        print(
            "\nInspect the flagged overlays by eye. Persistent border focus points at dataset "
            "bias, cropping, or artefacts -- investigate before reporting Grad-CAM as evidence."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
