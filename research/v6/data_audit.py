"""V6-0 data allocation audit (docs/V6_RUNSHEET.md section 0.3, section 0.8 item 3). Metadata only: no image is read.

    python -m research.v6.data_audit --pull      # fetch any collection not yet cached, then audit
    python -m research.v6.data_audit             # audit from the cache

For every source V6 would add to training, counts its identifier coverage and its links, by lesion id and by patient
id, to every row V6 must stay clear of: the manifest (development, reserved, HAM val/test), the V5 young extras, and
the HIBA confirmation cohort. Writes results/v6/data_allocation.json. Hash matching needs the images, so it is not
done here; the JSON says so. Diagnoses are only counted, never used to choose anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "results" / "v6" / "archive_meta"
OUT = REPO / "results" / "v6" / "data_allocation.json"
MANIFEST = REPO / "ml" / "data" / "manifest_v4.csv"
EXTRAS = REPO / "results" / "v5" / "young_data" / "extra_train.csv"
CANDIDATES = REPO / "results" / "v5" / "young_data" / "candidates.csv"
LINKS = REPO / "results" / "v5" / "s84_lesion_links.json"
API = "https://api.isic-archive.com/api/v2/images/search/?collections={c}&limit=100"
FIELDS = ("isic_id", "lesion_id", "patient_id", "age", "diagnosis_3", "confirm", "image_type")
#: ISIC collection ids. HIBA spans three collections (section 0.8b).
SOURCES = {"isic2019_train": "65", "isic2019_test": "72", "isic2020": "70", "milk10k": "425", "derm12345": "399",
           "hiba_175": "175", "hiba_176": "176", "hiba_251": "251"}
ADDED = ("isic2019_test", "isic2020", "milk10k", "derm12345")


def pull(collection: str, path: Path) -> int:
    """Walk the archive's cursor pagination for one collection and cache the identifier columns."""
    url, rows = API.format(c=collection), []
    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            page = json.load(response)
        for item in page["results"]:
            clinical = item["metadata"].get("clinical", {})
            acquisition = item["metadata"].get("acquisition", {})
            rows.append({"isic_id": item["isic_id"], "lesion_id": clinical.get("lesion_id", ""),
                         "patient_id": clinical.get("patient_id", ""), "age": clinical.get("age_approx", ""),
                         "diagnosis_3": clinical.get("diagnosis_3", ""),
                         "confirm": clinical.get("diagnosis_confirm_type", ""),
                         "image_type": acquisition.get("image_type", "")})
        url = page.get("next")
        time.sleep(0.2)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ids(rows: list[dict], column: str) -> set[str]:
    return {r[column] for r in rows if r.get(column)}


def archive_id(image_id: str) -> str:
    return image_id.removesuffix("_downsampled")


def is_young_melanoma(row: dict) -> bool:
    """Under 40 by the archive's approximate age; primary melanoma only (metastases are not a lesion V6 screens)."""
    diagnosis = row["diagnosis_3"].lower()
    return bool(row["age"]) and float(row["age"]) < 40 and "melanoma" in diagnosis and "metasta" not in diagnosis


def net_contribution(meta: dict[str, list[dict]], extras_arch: list[dict]) -> dict:
    """What each source adds beyond V5's pool, before any class map: labelled, dermoscopic, D6-clean rows that are not
    exact copies of an extras image. Only descriptive of the plan; no training decision is read from a diagnosis."""
    extras = {r["isic_id"] for r in extras_arch}
    extras_patients = ids(extras_arch, "patient_id")
    d6 = set(json.loads(LINKS.read_text(encoding="utf-8"))["linked_lesion_ids"])
    out = {}
    for name in ADDED:
        labelled = [r for r in meta[name] if r["diagnosis_3"]
                    and (name != "milk10k" or (r["image_type"] == "dermoscopic" and r["lesion_id"] not in d6))]
        new = [r for r in labelled if r["isic_id"] not in extras]
        with_age = [r for r in labelled if r["age"]]
        # What a patient-level drop against the extras would cost (rejected in section 0.3: both sides are training-only).
        lost = [r for r in new if r["patient_id"] in extras_patients]
        out[name] = {"labelled_rows": len(labelled), "rows_after_extras_images": len(new),
                     "rows_lost_to_extras_patient_drop": len(lost),
                     "young_rows_lost_to_extras_patient_drop": sum(bool(r["age"]) and float(r["age"]) < 40 for r in lost),
                     "rows_with_age": len(with_age),
                     "young_rows_after_extras_images": sum(bool(r["age"]) and float(r["age"]) < 40 for r in new),
                     "young_melanoma_lesions_labelled": len({r["lesion_id"] or r["isic_id"]
                                                             for r in labelled if is_young_melanoma(r)}),
                     "young_melanoma_lesions_new": len({r["lesion_id"] or r["isic_id"]
                                                        for r in new if is_young_melanoma(r)})}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull", action="store_true", help="fetch collections missing from the cache")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    meta: dict[str, list[dict]] = {}
    hashes: dict[str, str] = {}
    for name, collection in SOURCES.items():
        path = CACHE / f"{name}.csv"
        if not path.exists():
            if not args.pull:
                raise SystemExit(f"{path} is missing; run with --pull")
            print(f"pulled {name}: {pull(collection, path)} rows", flush=True)
        meta[name] = read(path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    hiba = list({r["isic_id"]: r for name in ("hiba_175", "hiba_176", "hiba_251") for r in meta[name]}.values())
    arch = {r["isic_id"]: r for name in ("isic2019_train", "isic2019_test", "isic2020", "milk10k") for r in meta[name]}
    manifest = read(MANIFEST)
    extras = read(EXTRAS)
    # Manifest and extras carry their own lesion ids, so archive ids come through the image id. MSKCC manifest ids carry
    # a `_downsampled` suffix the archive does not, and the extras' ids come from the V5 candidate pull (all collections).
    manifest_arch = [arch[archive_id(r["image_id"])] for r in manifest if archive_id(r["image_id"]) in arch]
    candidates = {r["image_id"]: {**r, "isic_id": r["image_id"]} for r in read(CANDIDATES)}
    extras_arch = [candidates[r["image_id"]] for r in extras if r["image_id"] in candidates]
    guarded = {"manifest": manifest_arch, "young_extras": extras_arch, "hiba": hiba}

    report: dict = {"note": "metadata links only; hash matching needs the images and is still to do",
                    "inputs_sha256": hashes, "image_hash_audit_done": False,
                    "manifest_rows_with_archive_id": len(manifest_arch), "manifest_rows": len(manifest),
                    "extras_rows_with_archive_id": len(extras_arch), "extras_rows": len(extras),
                    "extras_with_lesion_id": sum(bool(r["lesion_id"]) for r in extras_arch),
                    "extras_with_patient_id": sum(bool(r["patient_id"]) for r in extras_arch),
                    "hiba": {"images": len(hiba), "lesions": len(ids(hiba, "lesion_id")),
                             "patients": len(ids(hiba, "patient_id"))}, "sources": {}}
    for name in ADDED:
        rows = meta[name]
        entry: dict = {"images": len(rows), "with_lesion_id": sum(bool(r["lesion_id"]) for r in rows),
                       "with_patient_id": sum(bool(r["patient_id"]) for r in rows),
                       "with_diagnosis_3": sum(bool(r["diagnosis_3"]) for r in rows),
                       "dedup_level": "patient" if any(r["patient_id"] for r in rows) else "lesion, patient unknown",
                       "links": {}}
        own = {r["isic_id"] for r in rows}
        dropped: set[str] = set()
        for target, target_rows in guarded.items():
            images, lesions = own & {r["isic_id"] for r in target_rows}, ids(rows, "lesion_id") & ids(target_rows, "lesion_id")
            patients = ids(rows, "patient_id") & ids(target_rows, "patient_id")
            linked = {r["isic_id"] for r in rows
                      if r["isic_id"] in images or r["lesion_id"] in lesions or r["patient_id"] in patients}
            dropped |= linked
            entry["links"][target] = {"shared_images": len(images), "shared_lesions": len(lesions),
                                      "shared_patients": len(patients), "source_images_linked": len(linked)}
        entry["images_dropped_by_links"] = len(dropped)
        report["sources"][name] = entry
    # Added sources against each other: not leakage, but one lesion must not count twice in training.
    report["between_added"] = {
        f"{a}|{b}": {"shared_images": len({r["isic_id"] for r in meta[a]} & {r["isic_id"] for r in meta[b]}),
                     "shared_lesions": len(ids(meta[a], "lesion_id") & ids(meta[b], "lesion_id")),
                     "shared_patients": len(ids(meta[a], "patient_id") & ids(meta[b], "patient_id"))}
        for i, a in enumerate(ADDED) for b in ADDED[i + 1:]}
    # A patient-id link can only be found where both sides have the id, so the manifest's coverage bounds what
    # "0 shared patients" means.
    coverage: dict[str, list[int]] = {}
    for row, archived in zip(manifest, manifest_arch):
        counts = coverage.setdefault(row["archive"], [0, 0])
        counts[0] += bool(archived["patient_id"])
        counts[1] += 1
    report["manifest_patient_id_coverage"] = {k: {"with_patient_id": v[0], "rows": v[1]} for k, v in coverage.items()}
    report["manifest_with_lesion_id"] = sum(bool(r["lesion_id"]) for r in manifest_arch)
    report["net_training_contribution"] = net_contribution(meta, extras_arch)
    # The young share V6-2's sampler arm holds (section 0.4), on the whole development pool; the sampler uses each fold's own.
    dev = [r for r in manifest if r["split"] == "train"]
    young = sum(r["age_band"] == "<40" for r in dev) + len(extras)
    rows = len(dev) + len(extras)
    added = report["net_training_contribution"].values()
    new_rows, new_young = sum(v["rows_after_extras_images"] for v in added), sum(v["young_rows_after_extras_images"] for v in added)
    report["young_share"] = {"v5_pool": {"young_rows": young, "rows": rows, "share": young / rows},
                             "expanded_pool": {"young_rows": young + new_young, "rows": rows + new_rows,
                                               "share": (young + new_young) / (rows + new_rows)},
                             "note": "before class maps; DERM12345 has no age, so it counts as not young"}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["sources"], indent=1))
    print(json.dumps(report["between_added"], indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
