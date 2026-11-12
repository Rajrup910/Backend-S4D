"""HIBA becomes training-only from V7 (owner decision, 10 Nov 2026). MRA-MIDAS becomes the sole sealed external read.

    python -m research.v7.hiba_training      # writes results/v7/hiba_train/ (the pool itself: research/v7/pool_v7.py)

Integrity conditions, fixed here before any V7 training run:
1. V6's HIBA result (`results/v6/hiba/hiba_report.json`, receipt `results/v6/hiba_receipt.json`: one execution, labels read
   once on 7 Nov) stays exactly as read. It is V6's external check and was computed before HIBA touched any training.
2. No V7 system is ever scored or compared on HIBA. HIBA is not evidence for V7 in any form.
3. The reason is recorded before training: HIBA adds young escalating lesions, the scarcest data in the project. The switch
   is not a response to a bad result: V6 caught 18 of 20 young escalating HIBA lesions.
4. HIBA rows go through the V7 data gate as a training-only split (`train_hiba`): a HIBA row linked to any development,
   spent or MIDAS row is removed from training.

Labels: V6's class rules find melanoma and nevus in `diagnosis_2`, which V6's local HIBA cache does not hold (it keeps only
`diagnosis_3`), so building from that cache would silently drop every melanoma and nevus. Instead the four diagnosis fields
are fetched once from the public ISIC archive, exactly as V6's HIBA read did (`research/v6/hiba_read.py::read_labels`), and
cached with a timestamp (`results/v7/hiba_train/hiba_metadata_api.csv`). The 8-class label uses V6's added-pool function
(`research.v6.added_pool.map_class_8`); the 7-class label is checked against V6's HIBA mapping (`s84_milk10k.map_class`) and
any disagreement stops the build. Rows that map to no class are excluded, as V6 excluded them. Cohort and images: the 1,199-lesion V6 cohort, one image per
lesion (`results/v6/hiba/cohort.csv`, `data/external/hiba/`).
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

from research.v5 import young_data  # noqa: E402
from research.v5.s84_milk10k import map_class as v6_hiba_map_7  # noqa: E402
from research.v6 import head8  # noqa: E402
from research.v6.added_pool import map_class_8  # noqa: E402

COHORT = REPO / "results" / "v6" / "hiba" / "cohort.csv"
CACHE = REPO / "results" / "v6" / "archive_meta"
COLLECTIONS = ("175", "176", "251")
IMAGES = "data/external/hiba"
V6_ADDED = REPO / "results" / "v6" / "added_pool" / "added_train.csv"
OUT_DIR = REPO / "results" / "v7" / "hiba_train"
HIBA_TRAIN = OUT_DIR / "hiba_train.csv"
RECEIPT = OUT_DIR / "hiba_training_receipt.json"
V6_RECEIPT = REPO / "results" / "v6" / "hiba_receipt.json"
ESCALATING_7 = ("akiec", "bcc", "mel")
API_CACHE = OUT_DIR / "hiba_metadata_api.csv"
DIAGNOSES = ("diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4")


def fetch_metadata() -> pd.DataFrame:
    """The four diagnosis fields per HIBA image, from the public ISIC archive (as V6's read did); cached after one fetch."""
    if API_CACHE.is_file():
        return pd.read_csv(API_CACHE, dtype=str)
    rows = []
    for collection in COLLECTIONS:
        page = young_data._get({"collections": collection, "limit": 100})
        while True:
            for item in page["results"]:
                clinical = item["metadata"].get("clinical", {})
                rows.append({"isic_id": item["isic_id"], "confirm": clinical.get("diagnosis_confirm_type", ""),
                             **{d: clinical.get(d, "") for d in DIAGNOSES}})
            if not page.get("next"):
                break
            page = young_data._get(url=page["next"])
    frame = pd.DataFrame(rows).drop_duplicates("isic_id").assign(fetched=time.strftime("%Y-%m-%dT%H:%M:%S"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(API_CACHE, index=False)
    return frame.astype(str)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> pd.DataFrame:
    cohort = pd.read_csv(COHORT, dtype=str)
    cohort = cohort[cohort["excluded"].fillna("") == ""]
    meta = fetch_metadata().fillna("")
    frame = cohort.merge(meta, on="isic_id", how="left")
    if frame["diagnosis_3"].isna().any():
        raise SystemExit(f"{int(frame['diagnosis_3'].isna().sum())} cohort images have no archive metadata")
    records = frame[["confirm", *DIAGNOSES]].fillna("").to_dict("records")
    frame["class_8"] = [map_class_8(r) for r in records]
    frame["v6_class_7"] = [v6_hiba_map_7(pd.Series(r)) for r in records]
    frame = frame[frame["class_8"].notna()].copy()
    frame["class_index_8"] = frame["class_8"].map(head8.CLASSES_8.index)
    frame["class_index_7"] = head8.collapse_labels(frame["class_index_8"].to_numpy())
    frame["class_7"] = [head8.CLASSES_7[i] for i in frame["class_index_7"]]
    disagree = frame[frame["class_7"] != frame["v6_class_7"]]
    if len(disagree):
        raise SystemExit(f"{len(disagree)} rows: 8-class map collapses to a different class than V6's HIBA map, "
                         f"e.g. {disagree[['class_8', 'v6_class_7']].head(3).to_dict('records')}")
    lesion = frame["lesion_key"].fillna(frame["isic_id"])
    out = pd.DataFrame({
        "image_id": "HIBA:" + frame["isic_id"],             # the same id the V7 master table uses, so gate removals match
        "group_id": "hiba:" + lesion, "effective_lesion_id": "hiba:" + lesion,
        "age_band": frame["age_band"].fillna("unknown"), "archive": "hiba",
        "class_7": frame["class_7"], "class_8": frame["class_8"],
        "class_index_7": frame["class_index_7"].astype(int), "class_index_8": frame["class_index_8"].astype(int),
        "escalating_7": frame["class_7"].isin(ESCALATING_7), "image_path": [f"{IMAGES}/{i}.jpg" for i in frame["isic_id"]],
        "histopathology": frame["confirm"].fillna("").str.lower().str.contains("histopathology"),
    })
    missing = [p for p in out["image_path"] if not (REPO / p).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} HIBA images missing, e.g. {missing[0]}")
    return out.reset_index(drop=True)


def main() -> int:
    v6_receipt = json.loads(V6_RECEIPT.read_text(encoding="utf-8"))
    if v6_receipt.get("label_read_count") != 1 or v6_receipt.get("n_executions") != 1:
        raise SystemExit("V6's HIBA read is not in its expected final state (one execution, one label read); stop and check")
    hiba = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hiba.to_csv(HIBA_TRAIN, index=False)
    young_esc = hiba[(hiba["age_band"] == "<40") & hiba["escalating_7"]]
    receipt = {
        "decision": "HIBA is training-only from V7; MRA-MIDAS is the sole sealed external read (owner, 10 Nov 2026)",
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "conditions": ["V6's HIBA result stays as read (receipt below, unchanged)", "no V7 system is ever scored or compared on HIBA",
                       "reason recorded before training: young escalating lesions are the scarcest data", "HIBA rows pass the V7 data gate as split train_hiba"],
        "v6_hiba_receipt_sha256": sha(V6_RECEIPT), "v6_hiba_label_read_count": v6_receipt["label_read_count"],
        "label_source": "ISIC archive API, collections 175/176/251, diagnosis_1-4 (cached in results/v7/hiba_train/hiba_metadata_api.csv); "
                        "8-class by research.v6.added_pool.map_class_8, checked equal to V6's HIBA 7-class map",
        "rows": int(len(hiba)), "lesions": int(hiba["group_id"].nunique()), "by_class_7": hiba["class_7"].value_counts().to_dict(),
        "by_age_band": hiba["age_band"].value_counts().to_dict(), "under40_escalating_lesions": int(young_esc["group_id"].nunique()),
        "under40_escalating_by_class": young_esc["class_7"].value_counts().to_dict(), "histopathology_share": round(float(hiba["histopathology"].mean()), 3),
        "outputs": {"hiba_train": str(HIBA_TRAIN.relative_to(REPO)), "hiba_train_sha256": sha(HIBA_TRAIN),
                    "pool": "the V7 training pool is built from this file by research/v7/pool_v7.py",
                    "v6_added_train_sha256_unchanged": sha(V6_ADDED)},
    }
    RECEIPT.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("rows", "lesions", "by_class_7", "by_age_band", "under40_escalating_lesions", "under40_escalating_by_class", "histopathology_share")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
