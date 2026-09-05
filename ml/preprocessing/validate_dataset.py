"""Validate the prepared dataset before any training happens.

This is the "inspect the data before you trust it" step from section 5 of the master
specification. It checks the things that silently ruin dermatology projects:

  * corrupt or unreadable images
  * images too small to be useful at the training resolution
  * exact duplicate files (same bytes, different image_id)
  * class distribution and imbalance ratio
  * how many images share a lesion_id  -- this is what makes a random split leak
  * lesions carrying more than one diagnosis label (would break grouped splitting)
  * missing metadata fields

It writes a machine-readable report to ml/results/dataset_report.json and a
human-readable summary to ml/results/dataset_report.md, and it fails loudly on
problems that would invalidate training.

Usage:
    python -m ml.preprocessing.validate_dataset
    python -m ml.preprocessing.validate_dataset --skip-hash    # faster, skips duplicate detection
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm

from ml.paths import REPO_ROOT, load_class_mapping, load_training_config, resolve

Image.MAX_IMAGE_PIXELS = 200_000_000  # guard against decompression bombs, allow big dermoscopy


def file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def inspect_images(manifest: pd.DataFrame, min_side: int, compute_hashes: bool) -> dict[str, Any]:
    corrupt: list[str] = []
    too_small: list[dict[str, Any]] = []
    sizes: Counter[tuple[int, int]] = Counter()
    modes: Counter[str] = Counter()
    digests: dict[str, list[str]] = {}

    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="images", unit="img"):
        path = REPO_ROOT / row.path
        if not path.is_file():
            corrupt.append(f"{row.image_id}: file missing at {row.path}")
            continue
        try:
            with Image.open(path) as img:
                img.verify()  # cheap structural check
            with Image.open(path) as img:
                width, height = img.size
                modes[img.mode] += 1
        except Exception as exc:  # noqa: BLE001 - any decode failure is a data problem
            corrupt.append(f"{row.image_id}: {type(exc).__name__}: {exc}")
            continue

        sizes[(width, height)] += 1
        if min(width, height) < min_side:
            too_small.append({"image_id": row.image_id, "size": [width, height]})

        if compute_hashes:
            digests.setdefault(file_digest(path), []).append(row.image_id)

    duplicate_groups = [ids for ids in digests.values() if len(ids) > 1] if compute_hashes else []

    return {
        "corrupt": corrupt,
        "too_small": too_small,
        "distinct_resolutions": len(sizes),
        "most_common_resolutions": [
            {"size": list(size), "count": count} for size, count in sizes.most_common(5)
        ],
        "colour_modes": dict(modes),
        "duplicate_groups": duplicate_groups,
        "duplicate_image_count": sum(len(g) - 1 for g in duplicate_groups),
        "hashing_performed": compute_hashes,
    }


def inspect_labels(manifest: pd.DataFrame) -> dict[str, Any]:
    mapping = load_class_mapping()

    counts = manifest["class_code"].value_counts()
    distribution = {
        c.code: {
            "count": int(counts.get(c.code, 0)),
            "share": float(counts.get(c.code, 0) / len(manifest)),
            "malignancy": c.malignancy,
        }
        for c in mapping.classes
    }
    non_zero = [v["count"] for v in distribution.values() if v["count"] > 0]

    return {
        "num_images": int(len(manifest)),
        "num_classes_declared": mapping.num_classes,
        "num_classes_present": len(non_zero),
        "distribution": distribution,
        "largest_class": counts.idxmax(),
        "smallest_present_class": counts.idxmin(),
        "imbalance_ratio": float(max(non_zero) / min(non_zero)) if non_zero else 0.0,
    }


def inspect_grouping(manifest: pd.DataFrame) -> dict[str, Any]:
    per_lesion = manifest.groupby("lesion_id")
    image_counts = per_lesion.size()

    # A lesion must carry exactly one diagnosis, otherwise grouped splitting is ill-defined.
    conflicting = per_lesion["class_code"].nunique()
    conflicted_lesions = conflicting[conflicting > 1].index.tolist()

    multi = image_counts[image_counts > 1]
    return {
        "num_lesions": int(image_counts.size),
        "num_images": int(len(manifest)),
        "lesions_with_multiple_images": int(multi.size),
        "images_in_multi_image_lesions": int(multi.sum()),
        "max_images_per_lesion": int(image_counts.max()),
        "mean_images_per_lesion": float(image_counts.mean()),
        "leakage_exposure": float(multi.sum() / len(manifest)),
        "lesions_with_conflicting_labels": conflicted_lesions,
    }


def inspect_metadata(manifest: pd.DataFrame) -> dict[str, Any]:
    optional = [c for c in ("age", "sex", "localization", "dx_type") if c in manifest.columns]
    return {
        column: {
            "missing": int(manifest[column].isna().sum()),
            "distinct": int(manifest[column].nunique(dropna=True)),
        }
        for column in optional
    }


def render_markdown(report: dict[str, Any]) -> str:
    labels, grouping, images = report["labels"], report["grouping"], report["images"]

    lines = [
        "# Dataset Validation Report",
        "",
        f"**Generated:** {report['generated']}  ",
        f"**Manifest:** `{report['manifest']}`  ",
        f"**Class mapping version:** {report['class_mapping_version']}",
        "",
        "## Summary",
        "",
        f"- Images: **{labels['num_images']:,}**",
        f"- Lesions: **{grouping['num_lesions']:,}**",
        f"- Classes present: **{labels['num_classes_present']} / {labels['num_classes_declared']}**",
        f"- Imbalance ratio (largest : smallest): **{labels['imbalance_ratio']:.1f} : 1**",
        f"- Corrupt / unreadable images: **{len(images['corrupt'])}**",
        f"- Exact duplicate files: **{images['duplicate_image_count']}**"
        + ("" if images["hashing_performed"] else " _(hashing skipped)_"),
        "",
        "## Class distribution",
        "",
        "| Class | Malignancy | Images | Share |",
        "|---|---|---:|---:|",
    ]
    for code, stats in labels["distribution"].items():
        lines.append(
            f"| `{code}` | {stats['malignancy']} | {stats['count']:,} | {stats['share']:.1%} |"
        )

    lines += [
        "",
        "## Data leakage exposure",
        "",
        f"- Lesions with more than one image: **{grouping['lesions_with_multiple_images']:,}**",
        f"- Images belonging to such lesions: **{grouping['images_in_multi_image_lesions']:,}** "
        f"(**{grouping['leakage_exposure']:.1%}** of the dataset)",
        f"- Most images of a single lesion: **{grouping['max_images_per_lesion']}**",
        "",
        "> This is why the split is grouped by `lesion_id`. Splitting these images at random",
        "> would put near-identical photographs of the same lesion into both the training and",
        "> test sets, and the resulting test score would measure memorisation, not generalisation.",
        "",
        "## Image properties",
        "",
        f"- Distinct resolutions: {images['distinct_resolutions']}",
        f"- Colour modes: {images['colour_modes']}",
        f"- Images smaller than the training resolution: {len(images['too_small'])}",
        "",
        "| Resolution | Count |",
        "|---|---:|",
    ]
    for entry in images["most_common_resolutions"]:
        lines.append(f"| {entry['size'][0]} x {entry['size'][1]} | {entry['count']:,} |")

    if images["corrupt"]:
        lines += ["", "## Corrupt images", ""]
        lines += [f"- {item}" for item in images["corrupt"][:25]]
        if len(images["corrupt"]) > 25:
            lines.append(f"- ... and {len(images['corrupt']) - 25} more")

    lines += ["", "## Metadata completeness", "", "| Field | Missing | Distinct |", "|---|---:|---:|"]
    for field, stats in report["metadata"].items():
        lines.append(f"| `{field}` | {stats['missing']:,} | {stats['distinct']:,} |")

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    config = load_training_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=config["data"]["manifest"])
    parser.add_argument("--min-side", type=int, default=config["data"]["image_size"])
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Skip byte-level duplicate detection (faster, but duplicates go unreported).",
    )
    args = parser.parse_args(argv)

    manifest_path = resolve(args.manifest)
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        print("Run: python -m ml.preprocessing.prepare_dataset", file=sys.stderr)
        return 1

    manifest = pd.read_csv(manifest_path)
    print(f"Validating {len(manifest):,} rows from {manifest_path.relative_to(REPO_ROOT)}\n")

    report = {
        "generated": date.today().isoformat(),
        "manifest": manifest_path.relative_to(REPO_ROOT).as_posix(),
        "class_mapping_version": load_class_mapping().version,
        "labels": inspect_labels(manifest),
        "grouping": inspect_grouping(manifest),
        "metadata": inspect_metadata(manifest),
        "images": inspect_images(manifest, args.min_side, not args.skip_hash),
    }

    results_dir = resolve(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (results_dir / "dataset_report.md").write_text(render_markdown(report), encoding="utf-8")

    labels, grouping, images = report["labels"], report["grouping"], report["images"]
    print("\n" + "=" * 64)
    print(f"Images                 {labels['num_images']:,}")
    print(f"Lesions                {grouping['num_lesions']:,}")
    print(f"Classes present        {labels['num_classes_present']}/{labels['num_classes_declared']}")
    print(f"Imbalance ratio        {labels['imbalance_ratio']:.1f} : 1"
          f"  ({labels['largest_class']} vs {labels['smallest_present_class']})")
    print(f"Leakage exposure       {grouping['leakage_exposure']:.1%} of images share a lesion")
    print(f"Corrupt images         {len(images['corrupt'])}")
    print(f"Duplicate files        {images['duplicate_image_count']}")
    print("=" * 64)
    print(f"Reports written to {results_dir.relative_to(REPO_ROOT)}/dataset_report.{{json,md}}")

    # --- blocking conditions ---
    problems = []
    if images["corrupt"]:
        problems.append(f"{len(images['corrupt'])} corrupt or missing image(s)")
    if grouping["lesions_with_conflicting_labels"]:
        problems.append(
            f"{len(grouping['lesions_with_conflicting_labels'])} lesion(s) carry more than one "
            f"diagnosis label, which makes grouped splitting ill-defined"
        )
    if labels["num_classes_present"] < labels["num_classes_declared"]:
        problems.append("some declared classes have no images in this dataset")

    if problems:
        print("\nBLOCKING PROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nFix these before training. Details in dataset_report.json.")
        return 1

    print("\nValidation passed. Next: python -m ml.preprocessing.split_dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
