"""Create a leakage-free train / validation / test split.

**The single most important file in the ML pipeline.**

HAM10000 contains multiple photographs of the *same physical lesion* -- roughly a
fifth of the dataset. Those images are near-duplicates: same patient, same lesion,
same session, slightly different angle or magnification. If they are split at
random, the test set contains images whose twins the model already memorised, and
the reported accuracy is inflated by several points for no reason other than a bug.

So the split is performed over **lesions**, not images:

    1. Reduce the manifest to one row per lesion_id (each lesion has one diagnosis).
    2. Stratify those lesions by diagnosis, so every split keeps the class balance.
    3. Assign every image to the split its lesion landed in.
    4. Assert afterwards that no lesion_id appears in two splits.

Step 4 is not decoration. It is the check that proves the claim we make in the report.

The result is written to a version-controlled CSV so the exact split can be
reproduced (or audited) even if scikit-learn's RNG behaviour changes in a future
release.

Usage:
    python -m ml.preprocessing.split_dataset
    python -m ml.preprocessing.split_dataset --seed 7 --out ml/configs/splits/split_v2.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.paths import REPO_ROOT, load_training_config, resolve

SPLIT_NAMES = ("train", "val", "test")


def lesion_table(manifest: pd.DataFrame, group_column: str, stratify_column: str) -> pd.DataFrame:
    """One row per lesion, carrying that lesion's diagnosis."""
    labels_per_lesion = manifest.groupby(group_column)[stratify_column].nunique()
    conflicted = labels_per_lesion[labels_per_lesion > 1]
    if not conflicted.empty:
        raise ValueError(
            f"{len(conflicted)} lesion(s) carry more than one value of '{stratify_column}', "
            f"so they cannot be assigned to a single split: {conflicted.index[:5].tolist()}.\n"
            f"Run validate_dataset.py and resolve the label conflict first."
        )
    return (
        manifest.groupby(group_column, as_index=False)[stratify_column]
        .first()
        .sort_values(group_column)
        .reset_index(drop=True)
    )


def check_stratification_feasible(lesions: pd.DataFrame, stratify_column: str) -> None:
    """A three-way stratified split needs at least 3 lesions in every class."""
    counts = lesions[stratify_column].value_counts()
    too_rare = counts[counts < 3]
    if not too_rare.empty:
        raise ValueError(
            "Cannot build a stratified three-way split: class(es) "
            f"{too_rare.to_dict()} have fewer than 3 lesions.\n"
            "Options: merge the class with justification, drop it explicitly, or use "
            "cross-validation instead of a fixed split. Do not let it silently land in one split."
        )


def split_lesions(
    lesions: pd.DataFrame,
    stratify_column: str,
    fractions: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    """Stratified three-way split over lesions."""
    total = sum(fractions[name] for name in SPLIT_NAMES)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total} from {fractions}")

    strata = lesions[stratify_column]

    # First cut: train vs (val + test).
    train_lesions, holdout_lesions = train_test_split(
        lesions,
        train_size=fractions["train"],
        stratify=strata,
        random_state=seed,
        shuffle=True,
    )

    # Second cut: split the holdout into val and test, preserving their relative sizes.
    holdout_fraction = fractions["val"] + fractions["test"]
    val_share_of_holdout = fractions["val"] / holdout_fraction
    val_lesions, test_lesions = train_test_split(
        holdout_lesions,
        train_size=val_share_of_holdout,
        stratify=holdout_lesions[stratify_column],
        random_state=seed,
        shuffle=True,
    )

    assignments = pd.concat(
        [
            train_lesions.assign(split="train"),
            val_lesions.assign(split="val"),
            test_lesions.assign(split="test"),
        ]
    )
    return assignments


def assert_no_leakage(images: pd.DataFrame, group_column: str) -> None:
    """Prove that no lesion crosses a split boundary."""
    splits_per_lesion = images.groupby(group_column)["split"].nunique()
    offenders = splits_per_lesion[splits_per_lesion > 1]
    if not offenders.empty:
        raise AssertionError(
            f"LEAKAGE: {len(offenders)} lesion(s) appear in more than one split: "
            f"{offenders.index[:5].tolist()}"
        )

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(images.loc[images["split"] == a, group_column]) & set(
            images.loc[images["split"] == b, group_column]
        )
        if overlap:
            raise AssertionError(f"LEAKAGE: {len(overlap)} lesion(s) shared between {a} and {b}")


def distribution_table(images: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(images["class_code"], images["split"])
    for name in SPLIT_NAMES:
        if name not in counts.columns:
            counts[name] = 0
    return counts[list(SPLIT_NAMES)]


def render_report(
    images: pd.DataFrame,
    lesions: pd.DataFrame,
    seed: int,
    fractions: dict[str, float],
    output: Path,
) -> str:
    image_counts = images["split"].value_counts()
    lesion_counts = lesions["split"].value_counts()
    table = distribution_table(images)

    lines = [
        "# Train / Validation / Test Split",
        "",
        f"**Generated:** {date.today().isoformat()}  ",
        f"**Seed:** {seed}  ",
        f"**Split file:** `{output.relative_to(REPO_ROOT).as_posix()}`  ",
        f"**Requested lesion fractions:** train {fractions['train']:.0%} / "
        f"val {fractions['val']:.0%} / test {fractions['test']:.0%}",
        "",
        "## Method",
        "",
        "Images are **not** split directly. HAM10000 contains multiple photographs of the same",
        "physical lesion, identified by `lesion_id`. The split is performed over lesions,",
        "stratified by diagnosis, and every image inherits its lesion's split. This guarantees",
        "that no two images of the same lesion can appear on both sides of the train/test",
        "boundary, so the test score measures generalisation to unseen lesions.",
        "",
        "This is verified programmatically after assignment (`assert_no_leakage`), not assumed.",
        "",
        "## Sizes",
        "",
        "| Split | Lesions | Images | Image share |",
        "|---|---:|---:|---:|",
    ]
    for name in SPLIT_NAMES:
        n_img = int(image_counts.get(name, 0))
        lines.append(
            f"| {name} | {int(lesion_counts.get(name, 0)):,} | {n_img:,} | "
            f"{n_img / len(images):.1%} |"
        )

    lines += [
        "",
        "> Image shares deviate slightly from the requested lesion fractions because lesions",
        "> contribute different numbers of images. This is expected and correct -- the alternative",
        "> (forcing exact image proportions) would require breaking lesion groups.",
        "",
        "## Class distribution per split",
        "",
        "| Class | " + " | ".join(SPLIT_NAMES) + " | total |",
        "|---|" + "---:|" * (len(SPLIT_NAMES) + 1),
    ]
    for code, row in table.iterrows():
        total = int(row.sum())
        cells = " | ".join(f"{int(row[name]):,}" for name in SPLIT_NAMES)
        lines.append(f"| `{code}` | {cells} | {total:,} |")

    totals = " | ".join(f"{int(table[name].sum()):,}" for name in SPLIT_NAMES)
    lines.append(f"| **total** | {totals} | {len(images):,} |")
    lines += [
        "",
        "## Reproducibility",
        "",
        "The split is committed to version control as a CSV of "
        "`image_id, lesion_id, class_code, split`. Training reads that file rather than",
        "re-deriving the split, so a future scikit-learn release cannot silently change which",
        "images are in the test set.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    config = load_training_config()
    split_cfg = config["split"]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=config["data"]["manifest"])
    parser.add_argument("--out", default=config["data"]["splits"])
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument("--train-frac", type=float, default=split_cfg["train"])
    parser.add_argument("--val-frac", type=float, default=split_cfg["val"])
    parser.add_argument("--test-frac", type=float, default=split_cfg["test"])
    parser.add_argument(
        "--source-views",
        action="store_true",
        help="Also write per-source view files (<out>.<source>.csv) using the SAME "
             "assignment, so a model trained on the combined split can be scored on each "
             "dataset separately. Requires a 'source' column in the manifest.",
    )
    args = parser.parse_args(argv)

    manifest_path = resolve(args.manifest)
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        print("Run: python -m ml.preprocessing.prepare_dataset", file=sys.stderr)
        return 1

    manifest = pd.read_csv(manifest_path)
    group_column = split_cfg["group_column"]
    stratify_column = split_cfg["stratify_column"]
    fractions = {"train": args.train_frac, "val": args.val_frac, "test": args.test_frac}

    print(f"Manifest: {len(manifest):,} images")

    lesions = lesion_table(manifest, group_column, stratify_column)
    print(f"Lesions:  {len(lesions):,}")
    check_stratification_feasible(lesions, stratify_column)

    assigned_lesions = split_lesions(lesions, stratify_column, fractions, args.seed)
    images = manifest.merge(
        assigned_lesions[[group_column, "split"]], on=group_column, how="left", validate="many_to_one"
    )

    if images["split"].isna().any():
        raise AssertionError(f"{images['split'].isna().sum()} image(s) were not assigned a split")

    assert_no_leakage(images, group_column)
    print("Leakage check: PASSED (no lesion appears in more than one split)")

    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["image_id", group_column, "class_code", "class_index", "split"]
    images[columns].sort_values("image_id").to_csv(out_path, index=False)

    # Optional per-source views: identical assignment, filtered to one dataset each, so the
    # combined model can be evaluated on "phone photos only" vs "dermoscopy only".
    if args.source_views:
        if "source" not in images.columns:
            print("WARNING: --source-views requested but the manifest has no 'source' column; "
                  "skipping.", file=sys.stderr)
        else:
            for source, group in images.groupby("source"):
                view_path = out_path.with_suffix(f".{source}.csv")
                group[columns].sort_values("image_id").to_csv(view_path, index=False)
                per = group["split"].value_counts()
                print(f"  view {view_path.name}: "
                      f"{int(per.get('train', 0))} train / {int(per.get('val', 0))} val / "
                      f"{int(per.get('test', 0))} test")

    results_dir = resolve(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    report = render_report(images, assigned_lesions, args.seed, fractions, out_path)
    (results_dir / "split_report.md").write_text(report, encoding="utf-8")

    print()
    print(distribution_table(images).to_string())
    print()
    for name in SPLIT_NAMES:
        subset = images[images["split"] == name]
        print(
            f"{name:<6} {len(subset):>6,} images  "
            f"{subset[group_column].nunique():>6,} lesions  "
            f"({len(subset) / len(images):.1%})"
        )
    print(f"\nSplit written to {out_path.relative_to(REPO_ROOT)}")
    print(f"Report written to {(results_dir / 'split_report.md').relative_to(REPO_ROOT)}")
    print("\nNext: python -m ml.training.train --arch resnet50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
