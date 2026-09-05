"""Turn a raw HAM10000 download into a canonical manifest.

HAM10000 is distributed in at least two different shapes and it is not worth
reorganising 10,000 files by hand to satisfy a hardcoded path:

    Kaggle (kmader/skin-cancer-mnist-ham10000):
        HAM10000_metadata.csv
        HAM10000_images_part_1/*.jpg
        HAM10000_images_part_2/*.jpg
        hmnist_*.csv                      <- 28x28 downsamples, unused here

    Harvard Dataverse (doi:10.7910/DVN/DBW86T):
        HAM10000_metadata.tab / .csv
        HAM10000_images_part_1.zip -> extracted somewhere
        HAM10000_images_part_2.zip -> extracted somewhere

So this script does not care about layout. It searches the dataset root
recursively for the metadata file and for every image, joins them on `image_id`,
and writes one manifest that the rest of the pipeline reads:

    ml/data/manifest.csv
        image_id, lesion_id, dx, class_index, class_code, dx_type,
        age, sex, localization, path

`path` is relative to the repository root, so the manifest stays valid as long as
the dataset is not moved.

Usage:
    python -m ml.preprocessing.prepare_dataset
    python -m ml.preprocessing.prepare_dataset --dataset-root D:/datasets/ham10000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ml.paths import REPO_ROOT, load_class_mapping, load_training_config, resolve

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
METADATA_NAMES = ("HAM10000_metadata.csv", "HAM10000_metadata.tab", "HAM10000_metadata")
REQUIRED_COLUMNS = {"lesion_id", "image_id", "dx"}


def find_metadata(root: Path) -> Path:
    """Locate the HAM10000 metadata table anywhere under `root`."""
    for name in METADATA_NAMES:
        direct = root / name
        if direct.is_file():
            return direct

    candidates = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.name.lower().startswith("ham10000_metadata")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No HAM10000 metadata file found under {root}.\n"
            f"Expected one of {METADATA_NAMES}. Did the archive finish extracting?"
        )
    # Prefer the shallowest match, so a stray copy in a subfolder does not win.
    return min(candidates, key=lambda p: len(p.relative_to(root).parts))


def read_metadata(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix == ".tab" else ","
    df = pd.read_csv(path, sep=separator)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required column(s): {sorted(missing)}.\n"
            f"Found columns: {list(df.columns)}"
        )
    return df


def index_images(root: Path) -> dict[str, Path]:
    """Map image_id -> file path for every image under `root`.

    The `hmnist_*.csv` downsample files are ignored automatically because they are
    not images. Duplicate stems (the same image_id in two folders) keep the first
    match and are reported by the caller.
    """
    index: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            continue
        stem = path.stem
        if stem in index:
            duplicates.append(stem)
            continue
        index[stem] = path
    if duplicates:
        print(
            f"  note: {len(duplicates)} image_id(s) appeared in more than one folder; "
            f"kept the first occurrence (e.g. {duplicates[:3]})"
        )
    return index


def build_manifest(dataset_root: Path) -> pd.DataFrame:
    mapping = load_class_mapping()

    metadata_path = find_metadata(dataset_root)
    print(f"  metadata: {metadata_path.relative_to(dataset_root)}")
    meta = read_metadata(metadata_path)
    print(f"  metadata rows: {len(meta):,}")

    images = index_images(dataset_root)
    print(f"  images found on disk: {len(images):,}")
    if not images:
        raise FileNotFoundError(
            f"No .jpg/.png files under {dataset_root}. Extract the image archives there."
        )

    # --- label mapping ---
    known_dx = set(mapping.codes)
    found_dx = set(meta["dx"].unique())
    unknown = found_dx - known_dx
    if unknown:
        raise ValueError(
            f"Metadata contains dx values not present in class_mapping.json: {sorted(unknown)}.\n"
            f"Mapped classes are {sorted(known_dx)}. Update the mapping deliberately -- do not "
            f"silently drop labels."
        )
    unused = known_dx - found_dx
    if unused:
        print(f"  warning: mapping declares classes absent from this dataset: {sorted(unused)}")

    dx_to_index = {c.code: c.index for c in mapping.classes}
    meta["class_index"] = meta["dx"].map(dx_to_index)
    meta["class_code"] = meta["dx"]

    # --- join to files on disk ---
    meta["path"] = meta["image_id"].map(images)
    missing = meta["path"].isna()
    if missing.any():
        examples = meta.loc[missing, "image_id"].head(5).tolist()
        print(
            f"  warning: {missing.sum():,} metadata rows have no image file "
            f"(e.g. {examples}); dropping them"
        )
        meta = meta.loc[~missing].copy()

    orphans = set(images) - set(meta["image_id"])
    if orphans:
        print(f"  note: {len(orphans):,} image file(s) on disk have no metadata row; ignoring them")

    meta["path"] = meta["path"].map(lambda p: Path(p).resolve().relative_to(REPO_ROOT).as_posix())

    columns = [
        "image_id",
        "lesion_id",
        "dx",
        "class_index",
        "class_code",
        "dx_type",
        "age",
        "sex",
        "localization",
        "path",
    ]
    return meta[[c for c in columns if c in meta.columns]].sort_values("image_id").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    config = load_training_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset-root",
        default=config["data"]["dataset_root"],
        help="Folder containing the extracted HAM10000 download.",
    )
    parser.add_argument(
        "--out",
        default=config["data"]["manifest"],
        help="Where to write the manifest CSV.",
    )
    args = parser.parse_args(argv)

    dataset_root = resolve(args.dataset_root)
    if not dataset_root.is_dir():
        print(f"ERROR: dataset root does not exist: {dataset_root}", file=sys.stderr)
        print(
            "\nDownload HAM10000 and extract it there. See ml/datasets/README.md.",
            file=sys.stderr,
        )
        return 1

    print(f"Preparing dataset from {dataset_root}")
    manifest = build_manifest(dataset_root)

    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)

    print(f"\nWrote {len(manifest):,} rows -> {out_path.relative_to(REPO_ROOT)}")
    print(f"Unique lesions: {manifest['lesion_id'].nunique():,}")
    print("\nClass distribution:")
    counts = manifest["class_code"].value_counts()
    for code, count in counts.items():
        print(f"  {code:<6} {count:>6,}  ({count / len(manifest):>5.1%})")
    print("\nNext: python -m ml.preprocessing.validate_dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
