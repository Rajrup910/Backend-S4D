"""Prepare MRA-MIDAS for V7's single sealed external read. Labels are written to a restricted file and never printed.

    python -m research.v7.midas_prepare      # writes data/external/midas/sealed/ and data/external/midas/labels_restricted/

Cohort (fixed before any label is joined to anything): dermoscopic images (`midas_distance == "dscope"`) of biopsied lesions
(`midas_iscontrol == "no"`, as the MIDAS paper restricts its model analysis), whose pathology category maps to one of the 7
classes by MAPPING below. MAPPING was drafted from the 15 category NAMES alone (no counts, no images) on 10 Nov 2026 and follows
V6's rules (squamous cell carcinoma and SCC in situ -> akiec, as in V6's 7-class scheme). Intermediate melanocytic tumours
(severe atypia, Spitz, AIMP) are not in the primary read; their image ids go to a restricted secondary list, read once with the
primary, to report whether V7 called them escalating or referred them.

Clustering unit: `midas_record_id` (733 records, up to 42 images each, so patient-level), stricter than lesion-level.
Printed: only image counts, record counts and age-band counts (age is metadata). No class counts, no per-image label.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MIDAS = REPO / "data" / "external" / "midas"
CLINICAL = MIDAS / "clinical.csv"
FILES = MIDAS / "files.csv"
SEALED = MIDAS / "sealed"
RESTRICTED = MIDAS / "labels_restricted"
COHORT = SEALED / "cohort.csv"
LABELS = RESTRICTED / "labels.csv"
INTERMEDIATE = RESTRICTED / "intermediate_secondary.csv"
RECEIPT = SEALED / "prepare_receipt.json"
EXCLUDED = None
MAPPING = {
    "malignant- melanoma": "mel", "malignant- bcc": "bcc", "malignant- ak": "akiec", "malignant- scc": "akiec",
    "malignant- sccis": "akiec", "benign-melanocytic nevus": "nv", "benign-seborrheic keratosis": "bkl",
    "benign-dermatofibroma": "df", "benign-hemangioma": "vasc",
    "benign-fibrous papule": EXCLUDED, "benign-other": EXCLUDED, "malignant- other": EXCLUDED,
    "other- non-neoplastic, inflammatory, infectious": EXCLUDED,
    "melanocytic tumor, possible re-excision (severe, spitz, aimp)": "INTERMEDIATE",
    "other- melanocytic lesion, possible re-excision (severe, spitz, aimp)": "INTERMEDIATE",
}
SAFE = ["midas_record_id", "midas_file_name", "midas_iscontrol", "midas_distance", "midas_age", "midas_fitzpatrick", "midas_location"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def age_band(age: float) -> str:
    return "unknown" if pd.isna(age) else "<40" if age < 40 else "40-59" if age < 60 else "60+"


def main() -> int:
    if LABELS.exists() or COHORT.exists():
        raise SystemExit("MIDAS is already prepared; preparation runs once (delete nothing, check the receipt)")
    meta = pd.read_csv(CLINICAL, usecols=SAFE, dtype=str)
    eligible = (meta["midas_distance"] == "dscope") & (meta["midas_iscontrol"] == "no")
    files = pd.read_csv(FILES, dtype=str)
    real = set(files.loc[~files["file_name"].str.contains("/"), "file_name"])
    if not meta.loc[eligible, "midas_file_name"].isin(real).all():
        raise SystemExit("an eligible dermoscopic row has no image file")
    # The one label read: the pathology column, mapped in memory, written straight to the restricted files.
    path = pd.read_csv(CLINICAL, usecols=["midas_file_name", "midas_path"], dtype=str)
    names = path["midas_path"].fillna("").str.strip()
    unknown = sorted(set(names[eligible]) - set(MAPPING))
    if unknown:
        raise SystemExit(f"{len(unknown)} pathology categories are not in the declared mapping; extend MAPPING first")
    mapped = names.map(MAPPING)
    primary = eligible & mapped.notna() & (mapped != "INTERMEDIATE")
    secondary = eligible & (mapped == "INTERMEDIATE")
    image_id = "MIDAS:" + meta["midas_file_name"]
    group = "midas_record:" + meta["midas_record_id"]
    age = pd.to_numeric(meta["midas_age"], errors="coerce")
    cohort = pd.DataFrame({"image_id": image_id, "group_id": group, "age_band": age.map(age_band),
                           "fitzpatrick": meta["midas_fitzpatrick"], "location": meta["midas_location"],
                           "file_name": meta["midas_file_name"]})
    SEALED.mkdir(parents=True, exist_ok=True)
    RESTRICTED.mkdir(parents=True, exist_ok=True)
    cohort[primary].to_csv(COHORT, index=False)
    pd.DataFrame({"image_id": image_id[primary], "label": mapped[primary]}).to_csv(LABELS, index=False)
    pd.DataFrame({"image_id": image_id[secondary], "group_id": group[secondary], "age_band": age[secondary].map(age_band)}).to_csv(INTERMEDIATE, index=False)
    receipt = {"prepared": time.strftime("%Y-%m-%dT%H:%M:%S"), "mapping": MAPPING,
               "rule": "dermoscopic (dscope) and biopsied (iscontrol == no); category mapped to 7 classes; intermediate tumours secondary",
               "primary_images": int(primary.sum()), "primary_records": int(group[primary].nunique()),
               "primary_by_age_band": cohort.loc[primary, "age_band"].value_counts().to_dict(),
               "secondary_intermediate_images": int(secondary.sum()),
               "eligible_dermoscopic_biopsied": int(eligible.sum()),
               "sha256": {"clinical_csv": sha(CLINICAL), "files_csv": sha(FILES), "cohort_csv": sha(COHORT),
                          "labels_csv": sha(LABELS), "intermediate_csv": sha(INTERMEDIATE)},
               "printed": "counts and age bands only; no class counts, no per-image labels"}
    RECEIPT.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("eligible_dermoscopic_biopsied", "primary_images", "primary_records",
                                              "primary_by_age_band", "secondary_intermediate_images")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
