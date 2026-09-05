"""Partition ISIC 2019 into BCN-20000 and MSKCC, excluding HAM10000."""
import pandas as pd
from pathlib import Path

data_dir = Path("data/external")
meta_path = data_dir / "ISIC_2019_Training_Metadata.csv"
gt_path = data_dir / "ISIC_2019_Training_GroundTruth.csv"
ham_path = Path("data/ham10000/HAM10000_metadata.csv")

assert meta_path.is_file(), f"Missing {meta_path}"
assert gt_path.is_file(), f"Missing {gt_path}"
assert ham_path.is_file(), f"Missing {ham_path}"

meta = pd.read_csv(meta_path)
gt = pd.read_csv(gt_path)
ham = pd.read_csv(ham_path)

# Merge metadata with ground truth
merged = pd.merge(meta, gt, on="image")

# Map one-hot ground truth to primary class code
classes_isic = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "VASC", "SCC", "UNK"]
code_mapping = {
    "AK": "akiec",
    "BCC": "bcc",
    "BKL": "bkl",
    "DF": "df",
    "MEL": "mel",
    "NV": "nv",
    "VASC": "vasc",
    "SCC": "scc",
    "UNK": "unk"
}

def get_class(row):
    for c in classes_isic:
        if row[c] == 1.0:
            return code_mapping[c]
    return "unk"

merged["class_code"] = merged.apply(get_class, axis=1)

# Verify HAM10000 overlap
ham_images = set(ham["image_id"])
isic_images = set(merged["image"])
overlap = ham_images.intersection(isic_images)
print(f"HAM10000 overlap check: {len(overlap)} / {len(ham_images)} images match exactly.")
assert len(overlap) == 10015, f"Expected 10,015 HAM images in ISIC 2019, found {len(overlap)}"

# Exclude HAM10000
non_ham = merged[~merged["image"].isin(ham_images)].copy()
print(f"Non-HAM images in ISIC 2019: {len(non_ham)}")
assert len(non_ham) == 15316, f"Expected 15,316 non-HAM images, found {len(non_ham)}"

# Partition into BCN-20000 and MSKCC
# BCN: lesion_id starts with BCN (12,413 images)
# MSKCC: lesion_id starts with MSK or is NaN (2,903 images)
is_bcn = non_ham["lesion_id"].fillna("").str.startswith("BCN")
bcn_df = non_ham[is_bcn].copy()
mskcc_df = non_ham[~is_bcn].copy()

bcn_df["source"] = "bcn20000"
mskcc_df["source"] = "mskcc"
non_ham["source"] = non_ham.apply(lambda r: "bcn20000" if str(r["lesion_id"]).startswith("BCN") else "mskcc", axis=1)

print(f"BCN-20000 images: {len(bcn_df)}")
print(f"MSKCC images: {len(mskcc_df)}")
assert len(bcn_df) == 12413, f"Expected 12,413 BCN images, found {len(bcn_df)}"
assert len(mskcc_df) == 2903, f"Expected 2,903 MSKCC images, found {len(mskcc_df)}"

# Save manifests
bcn_df.to_csv(data_dir / "manifest_bcn20000.csv", index=False)
mskcc_df.to_csv(data_dir / "manifest_mskcc.csv", index=False)
non_ham.to_csv(data_dir / "manifest_isic2019_nonham.csv", index=False)

print("Manifests generated successfully in data/external/")
