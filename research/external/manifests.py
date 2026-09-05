"""Adapt the raw ISIC-2019 cohort manifests (BCN-20000, MSKCC) into the
image_id/path/class_index schema `ml.preprocessing.dataset.LesionDataset` expects.

Mirrors `research/xdomain/extract_pad.py`'s `make_pad_all_split`: every row is external
to HAM10000 training, so every row is labelled `test`. Two cohorts, so two adapters
(`build_bcn20000_manifest`, `build_mskcc_manifest`) sharing one implementation.

Rows whose `class_code` has no slot in the frozen 7-class HAM mapping (`scc`, `unk`) are
dropped -- the checkpoints were never trained on those classes, so there is nothing a
`class_index` for them could mean. `class_mapping.json`'s `planned_extension` documents
this explicitly: adding `scc` requires retraining, not a manifest trick.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.paths import REPO_ROOT, load_class_mapping, resolve

RAW_MANIFEST_TEMPLATE = "data/external/manifest_{cohort}.csv"
ADAPTED_MANIFEST_TEMPLATE = "data/external/manifest_{cohort}_adapted.csv"
SPLIT_TEMPLATE = "ml/configs/splits/split_{cohort}_all.csv"
IMAGES_DIR = "data/external/isic2019_images/ISIC_2019_Training_Input"


def _adapt_cohort_manifest(cohort: str) -> tuple[Path, Path]:
    """Build the adapted manifest + all-test split for one ISIC-2019 cohort.

    Idempotent: rewritten from the raw cohort manifest on every call.
    Returns (manifest_path, split_path).
    """
    raw_path = resolve(RAW_MANIFEST_TEMPLATE.format(cohort=cohort))
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"no raw manifest at {raw_path} -- run scripts/external/prepare_isic2019_external.py first"
        )

    mapping = load_class_mapping()
    raw = pd.read_csv(raw_path)

    dropped = raw[~raw["class_code"].isin(mapping.codes)]
    frame = raw[raw["class_code"].isin(mapping.codes)].copy()

    frame["image_id"] = frame["image"].astype(str)
    frame["path"] = frame["image_id"].apply(lambda img: f"{IMAGES_DIR}/{img}.jpg")
    frame["class_index"] = frame["class_code"].map(mapping.dx_to_index())
    # MSKCC rows are ~72% missing lesion_id in the source ISIC-2019 metadata (only
    # BCN-prefixed lesion_ids are populated) -- preserve NaN rather than stringifying it.
    frame["lesion_id"] = frame["lesion_id"].where(frame["lesion_id"].isna(), frame["lesion_id"].astype(str))

    manifest = frame[
        ["image_id", "lesion_id", "class_code", "class_index", "path", "age_approx", "sex"]
    ].reset_index(drop=True)

    split = pd.DataFrame(
        {
            "image_id": manifest["image_id"],
            "lesion_id": manifest["lesion_id"],
            "class_code": manifest["class_code"],
            "class_index": manifest["class_index"],
            "split": "test",
        }
    )

    manifest_path = resolve(ADAPTED_MANIFEST_TEMPLATE.format(cohort=cohort))
    split_path = resolve(SPLIT_TEMPLATE.format(cohort=cohort))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    split.to_csv(split_path, index=False)

    print(
        f"[{cohort}] {len(manifest)} images adapted "
        f"({len(dropped)} dropped: {sorted(dropped['class_code'].unique().tolist())}) "
        f"-> {manifest_path.relative_to(REPO_ROOT)}"
    )
    return manifest_path, split_path


def build_bcn20000_manifest() -> tuple[Path, Path]:
    return _adapt_cohort_manifest("bcn20000")


def build_mskcc_manifest() -> tuple[Path, Path]:
    return _adapt_cohort_manifest("mskcc")


COHORT_ADAPTERS = {
    "bcn20000": build_bcn20000_manifest,
    "mskcc": build_mskcc_manifest,
}
