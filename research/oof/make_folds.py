"""Partition the HAM10000 training split into 5 stratified, lesion-grouped folds for OOF.

Produces:
    ml/configs/splits/split_v1.folds.csv        (train split rows only, image_id -> fold index)
    ml/configs/splits/oof/split_v1.fold{k}.csv   (materialized split files for k in 0..4)
    ml/results/oof_fold_report.md                (fold composition and power audit report)

Constraints enforced:
    - StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on lesion_id & class_index
    - assert_no_leakage() passes for every fold file
    - Global validation split is preserved as 'val' in each fold file (for identical early stopping)
    - Real test split images are strictly excluded
    - Yields audited: df, vasc, and escalating lesions in patients <40
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ml.paths import REPO_ROOT, load_class_mapping, load_training_config, resolve
from ml.preprocessing.split_dataset import assert_no_leakage

ESCALATING_CLASSES = {"akiec", "bcc", "mel"}


def build_folds(
    splits_path: Path,
    manifest_path: Path,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict]:
    splits = pd.read_csv(splits_path)
    manifest = pd.read_csv(manifest_path).set_index("image_id")
    
    train_mask = splits["split"] == "train"
    val_mask = splits["split"] == "val"
    train_df = splits[train_mask].copy().reset_index(drop=True)
    val_df = splits[val_mask].copy().reset_index(drop=True)
    
    # 1. StratifiedGroupKFold on train lesions
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(train_df, y=train_df["class_index"], groups=train_df["lesion_id"])
    ):
        train_df.loc[val_idx, "fold"] = fold_idx
    
    assert (train_df["fold"] >= 0).all(), "Some train images were not assigned to a fold"
    
    # Check no lesion appears in multiple folds
    lesion_folds = train_df.groupby("lesion_id")["fold"].nunique()
    multi_fold_lesions = lesion_folds[lesion_folds > 1]
    if not multi_fold_lesions.empty:
        raise AssertionError(f"Leaked lesions across folds: {len(multi_fold_lesions)}")
    
    # 2. Materialize fold files
    fold_dfs: list[pd.DataFrame] = []
    oof_dir = splits_path.parent / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    
    for k in range(n_splits):
        fold_train = train_df[train_df["fold"] != k][["image_id", "lesion_id", "class_code", "class_index"]].copy()
        fold_train["split"] = "train"
        
        # Real validation split is kept as 'val' for uniform early stopping
        fold_val = val_df[["image_id", "lesion_id", "class_code", "class_index"]].copy()
        fold_val["split"] = "val"
        
        # Fold k's holdout from train is marked as 'test' (the OOF partition)
        fold_oof = train_df[train_df["fold"] == k][["image_id", "lesion_id", "class_code", "class_index"]].copy()
        fold_oof["split"] = "test"
        
        fold_combined = pd.concat([fold_train, fold_val, fold_oof], ignore_index=True)
        
        # Rigorous leakage verification
        assert_no_leakage(fold_combined, "lesion_id")
        
        # Ensure no real test images are present
        real_test_ids = set(splits[splits["split"] == "test"]["image_id"])
        leaked_real_test = set(fold_combined["image_id"]) & real_test_ids
        if leaked_real_test:
            raise AssertionError(f"Real test images leaked into fold {k}: {len(leaked_real_test)}")
        
        fold_dfs.append(fold_combined)
        fold_combined.to_csv(oof_dir / f"split_v1.fold{k}.csv", index=False)

    # Save summary mapping table (train rows only, NO 'split' column)
    folds_summary = train_df[["image_id", "lesion_id", "class_code", "class_index", "fold"]].copy()
    folds_summary_path = splits_path.parent / "split_v1.folds.csv"
    folds_summary.to_csv(folds_summary_path, index=False)
    
    # 3. Compute audit statistics
    stats: dict = {
        "n_splits": n_splits,
        "total_train_images": len(train_df),
        "total_train_lesions": train_df["lesion_id"].nunique(),
        "folds": {},
    }
    
    # Attach ages from manifest
    train_df["age"] = manifest.loc[train_df["image_id"], "age"].values
    train_df["is_under40"] = train_df["age"] < 40
    train_df["is_escalating"] = train_df["class_code"].isin(ESCALATING_CLASSES)
    train_df["under40_escalating"] = train_df["is_under40"] & train_df["is_escalating"]
    
    for k in range(n_splits):
        fold_k = train_df[train_df["fold"] == k]
        stats["folds"][k] = {
            "images": len(fold_k),
            "lesions": fold_k["lesion_id"].nunique(),
            "class_counts": fold_k["class_code"].value_counts().to_dict(),
            "under40_escalating": int(fold_k["under40_escalating"].sum()),
            "under40_total": int(fold_k["is_under40"].sum()),
        }
        
    stats["total_df"] = int((train_df["class_code"] == "df").sum())
    stats["total_vasc"] = int((train_df["class_code"] == "vasc").sum())
    stats["total_under40_escalating"] = int(train_df["under40_escalating"].sum())
    
    return folds_summary, fold_dfs, stats


def generate_report(stats: dict, output_path: Path) -> None:
    n_splits = stats["n_splits"]
    classes = sorted(list(stats["folds"][0]["class_counts"].keys()))
    
    lines = [
        "# Stratified Lesion-Grouped K-Fold Report (Session 6 OOF)",
        "",
        f"**Generated:** {datetime.now().isoformat(timespec='seconds')}  ",
        f"**Folds:** {n_splits}  ",
        f"**Total Train Images Partitioned:** {stats['total_train_images']}  ",
        f"**Total Train Lesions Partitioned:** {stats['total_train_lesions']}  ",
        "",
        "## 1. Per-Fold Composition",
        "",
        "| Fold | Holdout Images | Holdout Lesions | <40 Escalating | <40 Total |",
        "|---|---:|---:|---:|---:|",
    ]
    
    for k in range(n_splits):
        f = stats["folds"][k]
        lines.append(
            f"| Fold {k} | {f['images']} | {f['lesions']} | {f['under40_escalating']} | {f['under40_total']} |"
        )
    
    lines.extend([
        "",
        "## 2. Per-Class Distribution Across Folds",
        "",
        "| Class | " + " | ".join([f"Fold {k}" for k in range(n_splits)]) + " | Total |",
        "|---| " + " | ".join(["---:" for _ in range(n_splits)]) + " | ---:|",
    ])
    
    for c in classes:
        row = [f"**{c}**"]
        total_c = 0
        for k in range(n_splits):
            count = stats["folds"][k]["class_counts"].get(c, 0)
            row.append(str(count))
            total_c += count
        row.append(str(total_c))
        lines.append("| " + " | ".join(row) + " |")
        
    lines.extend([
        "",
        "## 3. Statistical Power Audit",
        "",
        f"- **`df` Total Yield:** {stats['total_df']} (target: ≥70; validation baseline was 24)",
        f"- **`vasc` Total Yield:** {stats['total_vasc']} (target: ≥95; validation baseline was 22)",
        f"- **`<40` Escalating Total Yield:** {stats['total_under40_escalating']} (target: ≥64; validation baseline was 22, test was 21)",
        "",
        "> **Methodological Guarantee**: Every fold holdout was validated with `assert_no_leakage()`. "
        "No patient lesion crosses train/test boundaries, and real test-split images are completely excluded.",
    ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {output_path.relative_to(REPO_ROOT)}")


def main():
    config = load_training_config()
    splits_path = resolve(config["data"]["splits"])
    manifest_path = resolve(config["data"]["manifest"])
    report_path = resolve("ml/results/oof_fold_report.md")
    
    print(f"Loading split from {splits_path.relative_to(REPO_ROOT)}...")
    _, _, stats = build_folds(splits_path, manifest_path)
    
    generate_report(stats, report_path)
    print("\nFold partitioning complete and verified.")
    print(f"  Total train images: {stats['total_train_images']}")
    print(f"  Rare class df yield: {stats['total_df']}")
    print(f"  Rare class vasc yield: {stats['total_vasc']}")
    print(f"  Under-40 escalating yield: {stats['total_under40_escalating']}")


if __name__ == "__main__":
    main()
