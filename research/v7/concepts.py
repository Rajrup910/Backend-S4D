"""Build B2's concept targets: ISIC 2018 Task 2 dermoscopic attribute masks, already on disk.

    python -m research.v7.concepts        # writes results/v7/concept_targets.csv and results/v7/concept_premise.json

`data/external/isic2018_task2/ISIC2018_Task2_Training_GroundTruth_v3.zip` holds expert masks for globules, streaks,
pigment network, negative network and milia-like cysts on 2,594 images; 2,593 are rows of our manifest, all MSKCC. Only
DEVELOPMENT rows are opened (the zip also covers 772 reserved and 380 val rows; their masks are concept labels and stay
unread). A development image keeps its own fold, so the concept teacher for fold f never sees fold f's masks.

The premise check is descriptive, on development rows: are globules and streaks (the report's young-melanoma features;
streaks are the masks closest to pseudopods) more common in under-40 melanoma than in under-40 nevi here? If they are not,
build B2 loses its stated rationale before any GPU time is spent.
"""

from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats as sps

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research import testguard  # noqa: E402

ZIP = REPO / "data" / "external" / "isic2018_task2" / "ISIC2018_Task2_Training_GroundTruth_v3.zip"
MANIFEST = REPO / "ml" / "data" / "manifest_v4.csv"
ATTRIBUTES = ("globules", "streaks", "pigment_network", "negative_network", "milia_like_cyst")
TARGETS = REPO / "results" / "v7" / "concept_targets.csv"
PREMISE = REPO / "results" / "v7" / "concept_premise.json"
_NAME = re.compile(r"(ISIC_\d{7})_attribute_([a-z_]+)\.png$")


def development_ids() -> set[str]:
    frame = pd.read_csv(MANIFEST, usecols=["image_id", "split"], dtype=str)
    dev = frame.loc[frame["split"] == "train", "image_id"]
    return set(dev.str.replace("_downsampled", "", regex=False))


def extract(zip_path: Path = ZIP, allowed: set[str] | None = None) -> pd.DataFrame:
    """Area fraction of each attribute mask per image, for allowed ids only (masks of other ids are never decoded)."""
    rows: dict[str, dict[str, float]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            match = _NAME.search(name)
            if not match:
                continue
            image_id, attribute = match.groups()
            if allowed is not None and image_id not in allowed:
                continue
            mask = np.asarray(Image.open(io.BytesIO(archive.read(name))).convert("L"))
            rows.setdefault(image_id, {})[f"{attribute}_frac"] = float((mask > 127).mean())
    frame = pd.DataFrame.from_dict(rows, orient="index").rename_axis("isic_id").reset_index()
    for attribute in ATTRIBUTES:
        frame[f"{attribute}_present"] = frame[f"{attribute}_frac"] > 0
    return frame


def premise(targets: pd.DataFrame) -> dict:
    """Prevalence of each attribute in under-40 melanoma vs under-40 nevus (development rows), Fisher exact test."""
    labels = pd.read_csv(MANIFEST, usecols=["image_id", "split", "class_7", "age_band", "group_id"], dtype=str)
    labels = labels[labels["split"] == "train"].assign(isic_id=lambda f: f["image_id"].str.replace("_downsampled", "", regex=False))
    frame = targets.merge(labels, on="isic_id")
    out: dict = {"rows_with_masks": int(len(frame)), "by_age_class": {}}
    for band in ("<40", "40-59", "60+"):
        part = frame[frame["age_band"] == band]
        cell = {}
        for attribute in ATTRIBUTES:
            mel = part.loc[part["class_7"] == "mel", f"{attribute}_present"]
            nv = part.loc[part["class_7"] == "nv", f"{attribute}_present"]
            if len(mel) and len(nv):
                table = [[int(mel.sum()), int((~mel).sum())], [int(nv.sum()), int((~nv).sum())]]
                odds, p = sps.fisher_exact(table)
                cell[attribute] = {"melanoma": f"{int(mel.sum())}/{len(mel)}", "nevus": f"{int(nv.sum())}/{len(nv)}",
                                   "odds_ratio": float(odds), "fisher_p": float(p)}
        out["by_age_class"][band] = {"melanoma_images": int((part["class_7"] == "mel").sum()),
                                     "melanoma_groups": int(part.loc[part["class_7"] == "mel", "group_id"].nunique()),
                                     "nevus_images": int((part["class_7"] == "nv").sum()), "attributes": cell}
    out["unit_note"] = "images, not lesions; MSKCC rows only; descriptive, no gate"
    return out


def main() -> int:
    testguard.block_test_reads("V7 concept targets: development rows only")
    allowed = development_ids()
    targets = extract(ZIP, allowed)
    TARGETS.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(TARGETS, index=False)
    report = {"source": str(ZIP.relative_to(REPO)).replace("\\", "/"), "development_images_with_masks": int(len(targets)),
              "reserved_or_val_masks_read": 0, "premise": premise(targets)}
    PREMISE.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["premise"]["by_age_class"]["<40"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
