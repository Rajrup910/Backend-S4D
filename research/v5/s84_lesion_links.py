"""S84 lesion-level exclusion list (deviation D6): MILK10k lesions already published in ISIC 2019 or used in V5 training.

    python -m research.v5.s84_lesion_links                    # fetch ISIC 2019 lesion ids from the ISIC API, write the list
    python -m research.v5.s84_lesion_links --cache-dir DIR    # use an earlier API pull (DIR/all_65.csv, DIR/all_72.csv)

LABEL-FREE: reads isic_id / lesion_id / image_type only. No diagnosis column of MILK10k, ISIC 2019 or the young extras.

Why: S75/S84 exclude a MILK10k lesion only if its image matches an ISIC 2019 image (ISIC id, or dHash AND pHash within 3).
A lesion photographed again under a new ISIC id passes that filter, yet the V5 composite trained on the same lesion's
ISIC 2019 image (development rows) and V1 on its HAM image. The ISIC Archive records lesion identity as `lesion_id`
(the IL_ namespace), so the link is checked there: a MILK10k lesion is excluded if its lesion_id appears in ISIC 2019
Training (collection 65), ISIC 2019 Test (collection 72) or the V5 young extras (results/v5/young_data/extra_train.csv).
Writes
  results/v5/isic2019_archive_lesion_ids.csv   isic_id, lesion_id, collection (the API snapshot, so the list re-derives offline)
  results/v5/s84_lesion_links.json             rule, sources with sha256, counts, linked_lesion_ids (sorted)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
V5 = REPO_ROOT / "results" / "v5"
MILK_META = REPO_ROOT / "data" / "external" / "milk10k" / "milk10k.csv"
EXTRAS = V5 / "young_data" / "extra_train.csv"
SNAPSHOT = V5 / "isic2019_archive_lesion_ids.csv"
LINKS = V5 / "s84_lesion_links.json"
API = "https://api.isic-archive.com/api/v2/images/search/?limit=100&collections={}"
COLLECTIONS = {65: "ISIC 2019 Training", 72: "ISIC 2019 Test"}
EXPECTED_SIZES = {65: 25_331, 72: 8_238}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch_collection(collection: int) -> pd.DataFrame:
    """Page through the ISIC v2 search API for one collection; isic_id and lesion_id only."""
    url, rows = API.format(collection), []
    while url:
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    page = json.load(response)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(3)
        rows += [{"isic_id": r["isic_id"], "lesion_id": (r.get("metadata", {}).get("clinical", {}) or {}).get("lesion_id")}
                 for r in page["results"]]
        url = page.get("next")
    return pd.DataFrame(rows)


def load_snapshot(cache_dir: Path | None) -> pd.DataFrame:
    frames = []
    for collection in COLLECTIONS:
        frame = (pd.read_csv(cache_dir / f"all_{collection}.csv", usecols=["isic_id", "lesion_id"])
                 if cache_dir else fetch_collection(collection))
        if len(frame) != EXPECTED_SIZES[collection]:
            raise SystemExit(f"collection {collection}: {len(frame)} images, expected {EXPECTED_SIZES[collection]}")
        frames.append(frame.assign(collection=collection))
    return pd.concat(frames, ignore_index=True)


def linked_lesions(milk_lesions: pd.Series, isic2019_lesions: pd.Series, extras_lesions: pd.Series) -> dict[str, set]:
    """MILK10k lesion ids that also appear in ISIC 2019 or the extras, by source. Pure, for the tests."""
    milk = set(milk_lesions.dropna().astype(str))
    return {"isic2019": milk & set(isic2019_lesions.dropna().astype(str)),
            "young_extras": milk & set(extras_lesions.dropna().astype(str))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="read DIR/all_65.csv and DIR/all_72.csv from an earlier ISIC API pull instead of fetching")
    args = parser.parse_args()

    snapshot = load_snapshot(args.cache_dir)
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    snapshot.sort_values(["collection", "isic_id"]).to_csv(SNAPSHOT, index=False)

    meta = pd.read_csv(MILK_META, low_memory=False, usecols=["isic_id", "lesion_id", "image_type"])
    derm = meta[meta["image_type"].astype(str).str.lower() == "dermoscopic"].drop_duplicates("lesion_id")
    extras = pd.read_csv(EXTRAS, usecols=["image_id", "lesion_id"])
    by_source = {"training": linked_lesions(derm["lesion_id"], snapshot.loc[snapshot.collection == 65, "lesion_id"], extras["lesion_id"]),
                 "test": linked_lesions(derm["lesion_id"], snapshot.loc[snapshot.collection == 72, "lesion_id"], extras["lesion_id"])}
    linked = by_source["training"]["isic2019"] | by_source["test"]["isic2019"] | by_source["training"]["young_extras"]
    blob = {
        "deviation": "D6 (results/v5/protocol_deviations.json)",
        "rule": "exclude a MILK10k dermoscopic lesion whose ISIC Archive lesion_id appears in ISIC 2019 Training "
                "(collection 65), ISIC 2019 Test (collection 72) or the V5 young extras; label-free",
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "snapshot_source": f"cache {args.cache_dir}" if args.cache_dir else "ISIC API v2 /images/search, live",
        "inputs_sha256": {"milk10k_meta": sha256(MILK_META), "young_extras": sha256(EXTRAS), "isic2019_snapshot": sha256(SNAPSHOT)},
        "counts": {
            "milk10k_dermoscopic_lesions": int(derm["lesion_id"].nunique()),
            "linked_isic2019_training": len(by_source["training"]["isic2019"]),
            "linked_isic2019_test": len(by_source["test"]["isic2019"]),
            "linked_young_extras": len(by_source["training"]["young_extras"]),
            "linked_total_union": len(linked),
            "isic_id_shared_with_isic2019": int(derm["isic_id"].isin(snapshot["isic_id"]).sum()),
        },
        "linked_lesion_ids": sorted(linked),
    }
    LINKS.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(json.dumps(blob["counts"], indent=2))
    print(f"-> {SNAPSHOT.relative_to(REPO_ROOT).as_posix()}, {LINKS.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
