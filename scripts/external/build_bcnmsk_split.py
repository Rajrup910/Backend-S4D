"""Build lesion-grouped 70/15/15 split on BCN-20000 + MSKCC for the Comparison Arm (E0).

Discipline:
  - 7 classes only (dropping SCC and UNK to maintain class-space parity with HAM10000).
  - Lesion-grouped splitting (assert_no_leakage). Where lesion_id is null, image is its own group.
  - Stratified across all 7 classes.
  - Seed 42 for split construction. Test split is touched once.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CLASS_CODES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_CODES)}
ESCALATING = ["akiec", "bcc", "mel"]


def main():
    print("=== Building BCN+MSK Lesion-Grouped Split (Workstream E0) ===")
    manifest_path = Path("data/external/manifest_isic2019_nonham.csv")
    assert manifest_path.is_file(), f"Missing {manifest_path}"
    
    df = pd.read_csv(manifest_path)
    total_initial = len(df)
    print(f"Total non-HAM images in ISIC 2019: {total_initial}")
    
    # 1. Drop SCC and UNK
    scc_count = (df["class_code"] == "scc").sum()
    unk_count = (df["class_code"] == "unk").sum()
    print(f"Dropping SCC images: {scc_count} (reported explicitly per E0.3/E1.4)")
    print(f"Dropping UNK images: {unk_count}")
    
    df_7c = df[df["class_code"].isin(CLASS_CODES)].copy()
    print(f"Retained 7-class images: {len(df_7c)}")
    
    # 2. Lesion grouping field
    # Where lesion_id is null, treat each image as its own unique lesion (Hard Rule 6)
    null_lesions = df_7c["lesion_id"].isna().sum()
    df_7c["effective_lesion_id"] = df_7c["lesion_id"].fillna("no_lesion_" + df_7c["image"])
    print(f"Images with null lesion_id treated as unique lesions: {null_lesions} / {len(df_7c)}")
    
    # 3. Add class_index and image path
    df_7c["class_index"] = df_7c["class_code"].map(CLASS_TO_IDX)
    df_7c["path"] = "data/external/isic2019_images/" + df_7c["image"] + ".jpg"
    
    # 4. Collapse to unique lesions for stratified splitting
    lesions = (
        df_7c.groupby("effective_lesion_id", as_index=False)
        .first()[["effective_lesion_id", "class_code", "class_index"]]
        .sort_values("effective_lesion_id")
        .reset_index(drop=True)
    )
    print(f"Unique lesion clusters: {len(lesions)}")
    
    # 5. Stratified 70/15/15 split
    seed = 42
    train_les, holdout_les = train_test_split(
        lesions, train_size=0.70, stratify=lesions["class_index"], random_state=seed, shuffle=True
    )
    val_les, test_les = train_test_split(
        holdout_les, train_size=0.50, stratify=holdout_les["class_index"], random_state=seed, shuffle=True
    )
    
    train_les = train_les.assign(split="train")
    val_les = val_les.assign(split="val")
    test_les = test_les.assign(split="test")
    
    split_map = pd.concat([train_les, val_les, test_les]).set_index("effective_lesion_id")["split"].to_dict()
    df_7c["split"] = df_7c["effective_lesion_id"].map(split_map)
    
    # 6. Leakage assertion
    splits_per_lesion = df_7c.groupby("effective_lesion_id")["split"].nunique()
    leaked = splits_per_lesion[splits_per_lesion > 1]
    assert leaked.empty, f"LEAKAGE DETECTED: {len(leaked)} lesions cross split boundaries!"
    print("Leakage assertion passed: exactly 0 lesions cross split boundaries.")
    
    # 7. Check split statistics
    split_counts = df_7c["split"].value_counts().to_dict()
    print(f"Split image counts: {split_counts}")
    
    # Check age band statistics in test split
    test_df = df_7c[df_7c["split"] == "test"].copy()
    test_age = test_df[test_df["age_approx"].notna()]
    test_under40 = test_age[test_age["age_approx"] < 40]
    test_under40_escal = test_under40[test_under40["class_code"].isin(ESCALATING)]
    
    print(f"Test split total: {len(test_df)} images")
    print(f"Test split <40 with age: {len(test_under40)} images")
    print(f"Test split <40 escalating lesions: {len(test_under40_escal)} (HAM had 21)")
    
    # Save split
    out_path = Path("ml/configs/splits/split_bcnmsk.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_7c.to_csv(out_path, index=False)
    print(f"Saved split to {out_path}")

if __name__ == "__main__":
    main()
