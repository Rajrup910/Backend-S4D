"""The V7 training pool beyond the development folds (owner decision, 10 Nov 2026): the biggest pool that keeps integrity.

    python -m research.v7.pool_v7       # writes results/v7/added_pool_v7/added_train_v7.csv and pool_v7_receipt.json

Sources, each with its own row in `pool_source`:
* `v6_added`       V6's 25,471 added rows (ISIC 2019 Test Barcelona part, ISIC 2020, MILK10k, DERM12345), unchanged.
* `hiba`           1,199 HIBA lesions, training-only from V7 (`research/v7/hiba_training.py`).
* `retired_split`  the manifest's `reserved`, `val` and `ham_val` splits (8,535 images). Their evaluation roles ended in
                   V1-V4 (`reserved` was read once at S63, closed; `val` and `ham_val` were selection sets). Labels come from the
                   manifest unchanged. `ham_test` (V1's published test set) stays OUT, and so do the development folds' own
                   evaluation roles and MIDAS.
Kept separate, never trained on: development fold 0 (selection) and folds 1-4 (the one in-domain read), `ham_test`, MIDAS.

Duplicates: image ids are unique across the pool; byte-identical images (same SHA-256 in the V7 master table) are kept once,
the first in source order. Links to evaluation rows are not handled here: the data gate removes the training-side row at train
time (`research/data/overlap_guard.py`), and the trainer refuses to start without the gate's receipt.
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

V6_ADDED = REPO / "results" / "v6" / "added_pool" / "added_train.csv"
HIBA_TRAIN = REPO / "results" / "v7" / "hiba_train" / "hiba_train.csv"
MANIFEST = REPO / "ml" / "data" / "manifest_v4.csv"
MASTER = REPO / "ml" / "data" / "v7" / "master_v7.csv"
OUT = REPO / "results" / "v7" / "added_pool_v7" / "added_train_v7.csv"
RECEIPT = OUT.with_name("pool_v7_receipt.json")
RETIRED_SPLITS = ("reserved", "val", "ham_val")
KEPT_OUT = ("ham_test",)
IMAGE_DIR = "data/external/isic2019_images/ISIC_2019_Training_Input"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retired_rows(columns: list[str]) -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    rows = manifest[manifest["split"].isin(RETIRED_SPLITS)].copy()
    dev_groups = set(manifest.loc[manifest["split"] == "train", "group_id"])
    held_groups = set(manifest.loc[manifest["split"].isin(KEPT_OUT), "group_id"])
    if set(rows["group_id"]) & (dev_groups | held_groups):
        raise SystemExit("a retired-split lesion group is shared with the development rows or ham_test")
    rows["image_path"] = [f"{IMAGE_DIR}/{i}.jpg" for i in rows["image_id"].astype(str)]
    return rows.reindex(columns=columns)


def build() -> tuple[pd.DataFrame, dict]:
    added = pd.read_csv(V6_ADDED, low_memory=False)
    columns = list(added.columns)
    hiba = pd.read_csv(HIBA_TRAIN, low_memory=False)
    parts = {"v6_added": added, "hiba": hiba.reindex(columns=columns), "retired_split": retired_rows(columns)}
    pool = pd.concat([p.assign(pool_source=name) for name, p in parts.items()], ignore_index=True)
    if pool["image_id"].astype(str).duplicated().any():
        raise SystemExit(f"{int(pool['image_id'].astype(str).duplicated().sum())} image ids repeat across pool sources")
    master = pd.read_csv(MASTER, usecols=["image_id", "sha256"], dtype=str).set_index("image_id")["sha256"]
    pool["sha256"] = pool["image_id"].astype(str).map(master)
    if pool["sha256"].isna().any():
        raise SystemExit(f"{int(pool['sha256'].isna().sum())} pool rows are not in the V7 master table; rerun the data gate first")
    duplicate = pool["sha256"].duplicated(keep="first")
    report = {"rows_by_source_before": pool["pool_source"].value_counts().to_dict(),
              "byte_identical_dropped": int(duplicate.sum()),
              "byte_identical_dropped_by_source": pool.loc[duplicate, "pool_source"].value_counts().to_dict()}
    pool = pool[~duplicate].drop(columns="sha256").reset_index(drop=True)
    for column in ("image_id", "group_id", "effective_lesion_id", "age_band", "archive", "class_index_7", "class_index_8", "escalating_7", "image_path"):
        if pool[column].isna().any():
            raise SystemExit(f"pool column {column} has empty cells")
    missing = [p for p in pool["image_path"] if not (REPO / p).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} pool images missing, e.g. {missing[0]}")
    report["rows_by_source_after"] = pool["pool_source"].value_counts().to_dict()
    report["rows"] = int(len(pool))
    report["under40_rows_by_source"] = pool[pool["age_band"] == "<40"]["pool_source"].value_counts().to_dict()
    report["rows_by_archive"] = pool["archive"].value_counts().to_dict()
    return pool, report


def main() -> int:
    pool, report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(OUT, index=False)
    receipt = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "decision": "owner, 10 Nov 2026: biggest training pool that keeps integrity",
               "trained_on": ["development folds other than the held-out one", "young extras", *report["rows_by_source_after"]],
               "never_trained_on": ["the held-out development fold", "ham_test", "midas"],
               "inputs_sha256": {p.relative_to(REPO).as_posix(): sha(p) for p in (V6_ADDED, HIBA_TRAIN, MANIFEST, MASTER)},
               "output_sha256": sha(OUT), **report}
    RECEIPT.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
