"""V6 expanded pool: class maps, link exclusions, resized cache and image-hash audit (docs/V6_RUNSHEET.md section 0.3, 0.8 item 3).

    python -m research.v6.added_pool --pull            # archive metadata per source; no image
    python -m research.v6.added_pool --select          # class map and id links; writes candidates.csv; no image
    python -m research.v6.added_pool --download        # resumable: full image -> hashes -> resized cache
    python -m research.v6.added_pool --finalize        # hash audit -> added_train.csv, audit.json, data_allocation.json

`train_v6 --added-train results/v6/added_pool/added_train.csv` reads the output. Rows are training-only and never enter a fold's
held-out rows. Nothing here reads a test, reserved or external cohort's labels or scores anything.

**Class map, declared for every added source.** V5's `young_data.CLASS_RULES` (the map the young extras used), split to 8 classes:
SCC, its in-situ form and keratoacanthoma are `scc`; actinic keratosis is `akiec_ak`. Keratoacanthoma follows S84's rule
(`s84_milk10k.EXTRA_RULES`). Melanoma metastasis is excluded: it is not a lesion the system screens (`data_audit.is_young_melanoma`).
Anything the rules do not match (scar, Kaposi sarcoma, Paget disease, indeterminate lesions, a missing diagnosis) is excluded and
counted, so no row enters by a guess.

**ISIC 2019 Test is its Barcelona part only** (section 0.3; 1,592 otherwise usable rows are attributed to the MILK study team).

**Links, section 0.3.** Dropped: any link by image, lesion or patient id to a manifest row or to HIBA; MILK10k's D6 lesions; an image
already in the young extras. Kept: a lesion or patient shared with the young extras only, since both sides train. Manifest lesion ids are
not ISIC ids, so manifest rows are compared through their archive records, as `data_audit.py` does. **Hashes:** dHash AND pHash within
radius 3 (the S49 rule, `research/v4/dedupe.py`) against the 27,629 corpus images, the young extras, HIBA and earlier added rows. A copy of a
corpus or HIBA image also removes the other added images of its lesion and patient.

**HIBA is hashed and thrown away:** the confirmation cohort's pixels are never stored and its diagnoses are never used here, so its one
receipted read (V6-6) is untouched. **Cache:** shorter side 512 px, which covers the 384 px member's 439 px eval resize and its training
crops without upsampling; smaller images are not enlarged. Hashes come from the full image, before resizing. JPEG quality 95.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import time
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from research import testguard
from research.v4 import dedupe
from research.v5 import young_data
from research.v6 import data_audit, head8

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "v6" / "added_pool"
CANDIDATES = OUT_DIR / "candidates.csv"
SELECTION = OUT_DIR / "selection.json"
HASHES = OUT_DIR / "hashes.csv"
HIBA_HASHES = OUT_DIR / "hiba_hashes.csv"
ADDED_TRAIN = OUT_DIR / "added_train.csv"
AUDIT = OUT_DIR / "audit.json"
CACHE = REPO / "data" / "external" / "v6_added"
MILK_ZIP = REPO / "data" / "external" / "milk10k" / "milk10k.zip"
FULL_URL = "https://isic-archive.s3.amazonaws.com/images/{}.jpg"
SOURCES = {name: data_audit.SOURCES[name] for name in data_audit.ADDED}
HIBA = ("hiba_175", "hiba_176", "hiba_251")
SHORT_SIDE = 512
JPEG_QUALITY = 95
DUP_RADIUS = dedupe.CLUSTER_RADIUS
MIN_FREE_GB = 10.0
ESCALATING_8 = {"akiec_ak", "bcc", "mel", "scc"}
META_FIELDS = ("isic_id", "image_type", "attribution", "license", "lesion_id", "patient_id", "age_approx", "sex", "anatom_site_general",
               "diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4", "confirm", "url", "bytes", "pixels_x", "pixels_y")


# ------------------------------------------------------------------ class map
def map_class_8(clinical: dict) -> str | None:
    """One of `head8.CLASSES_8`, or None to exclude the row."""
    d3, d4 = str(clinical.get("diagnosis_3", "") or "").lower(), str(clinical.get("diagnosis_4", "") or "").lower()
    if "metasta" in d3 or "metasta" in d4:
        return None
    if "keratoacanthoma" in d3 or "keratoacanthoma" in d4 or "squamous cell carcinoma" in d3:
        return "scc"
    clean = {k: ("" if pd.isna(v) else v) for k, v in clinical.items()}
    code = young_data.map_class(clean)
    return "akiec_ak" if code == "akiec" else code


def age_band(age: float | None) -> str:
    if age is None or pd.isna(age):
        return "unknown"
    return "<40" if age < 40 else "40-59" if age < 60 else "60+"


# ------------------------------------------------------------------ metadata
def meta_path(source: str) -> Path:
    return OUT_DIR / f"meta_{source}.csv"


def pull() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for source, collection in SOURCES.items():
        if meta_path(source).exists():
            continue
        rows, page = [], young_data._get({"collections": collection, "limit": 100})
        while True:
            for item in page["results"]:
                clinical, acquisition = item["metadata"].get("clinical", {}), item["metadata"].get("acquisition", {})
                rows.append({"isic_id": item["isic_id"], "image_type": acquisition.get("image_type", ""),
                             "attribution": item.get("attribution", ""), "license": item.get("copyright_license", ""),
                             **{k: clinical.get(k, "") for k in ("lesion_id", "patient_id", "age_approx", "sex", "anatom_site_general",
                                                                 "diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4")},
                             "confirm": clinical.get("diagnosis_confirm_type", ""), "url": item["files"]["full"]["url"],
                             "bytes": item["files"]["full"]["size"], "pixels_x": acquisition.get("pixels_x", ""),
                             "pixels_y": acquisition.get("pixels_y", "")})
            if not page.get("next"):
                break
            page = young_data._get(url=page["next"])
        part = meta_path(source).with_suffix(".part")
        with part.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=META_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        part.replace(meta_path(source))
        print(f"pulled {source}: {len(rows):,} rows", flush=True)
    return 0


def guarded_ids() -> dict[str, set[str]]:
    """Image, lesion and patient ids of the manifest's archive records and of HIBA, plus the extras' images, from the audit's cache."""
    cache = {n: data_audit.read(data_audit.CACHE / f"{n}.csv") for n in data_audit.SOURCES}
    arch = {r["isic_id"]: r for n in ("isic2019_train", "isic2019_test", "isic2020", "milk10k") for r in cache[n]}
    manifest = [arch[i] for i in (data_audit.archive_id(r["image_id"]) for r in data_audit.read(data_audit.MANIFEST)) if i in arch]
    hiba = [r for n in HIBA for r in cache[n]]
    out = {"extras_images": {r["image_id"] for r in data_audit.read(data_audit.EXTRAS)}}
    for name, rows in (("manifest", manifest), ("hiba", hiba)):
        out[f"{name}_images"] = {r["isic_id"] for r in rows}
        out[f"{name}_lesions"] = data_audit.ids(rows, "lesion_id")
        out[f"{name}_patients"] = data_audit.ids(rows, "patient_id")
    return out


def exclusion_reason(row: pd.Series, source: str, guarded: dict[str, set[str]], d6: set[str]) -> str | None:
    """Why a row is out, in a fixed order, so every count in `selection.json` is unambiguous."""
    if row["image_type"] != "dermoscopic":
        return "not_dermoscopic"
    if source == "isic2019_test" and "barcelona" not in str(row["attribution"]).lower():
        return "isic2019_test_not_the_bcn_part"
    image, lesion, patient = str(row["isic_id"]), str(row["lesion_id"] or ""), str(row["patient_id"] or "")
    for target in ("manifest", "hiba"):
        if image in guarded[f"{target}_images"] or lesion in guarded[f"{target}_lesions"] or patient in guarded[f"{target}_patients"]:
            return f"linked_to_{target}"
    if source == "milk10k" and lesion in d6:
        return "milk10k_d6"
    if image in guarded["extras_images"]:
        return "already_in_young_extras"
    if map_class_8(row.to_dict()) is None:
        return "no_class_on_the_head"
    return None


def select() -> int:
    testguard.block_test_reads("V6 added pool: training-only sources")
    guarded = guarded_ids()
    guarded = {k: v - {""} for k, v in guarded.items()}
    d6 = set(json.loads(data_audit.LINKS.read_text(encoding="utf-8"))["linked_lesion_ids"])
    kept, report = [], {}
    for source in SOURCES:
        frame = pd.read_csv(meta_path(source), low_memory=False, dtype={"lesion_id": str, "patient_id": str}).fillna({"lesion_id": "", "patient_id": ""})
        reasons = [exclusion_reason(row, source, guarded, d6) for _, row in frame.iterrows()]
        frame["reason"] = reasons
        out = frame[frame["reason"].isna()].drop(columns="reason").assign(source=source)
        out["class_8"] = [map_class_8(r) for r in out.to_dict("records")]
        unmapped = frame.loc[frame["reason"] == "no_class_on_the_head", "diagnosis_3"].fillna("(none)")
        report[source] = {"archive_rows": len(frame), "kept": len(out), "excluded": frame["reason"].value_counts().to_dict(),
                          "kept_by_class_8": out["class_8"].value_counts().to_dict(),
                          "unmapped_diagnosis_3": unmapped.value_counts().to_dict(),
                          "kept_by_attribution": out["attribution"].value_counts().head(12).to_dict(),
                          "kept_by_license": out["license"].value_counts().to_dict(),
                          "kept_young": int((out["age_approx"] < 40).sum()), "kept_no_age": int(out["age_approx"].isna().sum()),
                          "full_download_gb": round(float(out["bytes"].sum()) / 1e9, 2)}
        kept.append(out)
    candidates = pd.concat(kept, ignore_index=True)
    lesions = candidates.loc[candidates["lesion_id"] != "", ["lesion_id", "source"]].drop_duplicates()
    between = lesions["lesion_id"].duplicated(keep=False)
    report["lesions_in_two_added_sources"] = int(lesions.loc[between, "lesion_id"].nunique())
    candidates.to_csv(CANDIDATES, index=False)
    SELECTION.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


# ------------------------------------------------------------------ images
def hash_bytes(data: bytes) -> tuple[int, int]:
    """`dedupe.hash_one` on bytes, so the full image never has to be written to disk."""
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_REDUCED_COLOR_8)
    if image is None:
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        return 0, 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return int(dedupe._dhash(gray)), int(dedupe._phash(gray))


def resized_jpeg(data: bytes, short_side: int = SHORT_SIDE) -> tuple[bytes, int, int]:
    """Shorter side `short_side`; a smaller image is re-encoded at its own size, never enlarged."""
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        scale = short_side / min(width, height)
        if scale < 1:
            size = (round(width * scale), round(height * scale))
            image.draft("RGB", (size[0] * 2, size[1] * 2))
            image = image.convert("RGB").resize(size, Image.LANCZOS)
        else:
            image = image.convert("RGB")
        out = io.BytesIO()
        image.save(out, "JPEG", quality=JPEG_QUALITY)
    return out.getvalue(), width, height


def fetch(url: str, attempts: int = 5) -> bytes:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                length = response.headers.get("Content-Length")
                data = response.read()
            if length is not None and len(data) != int(length):
                raise OSError(f"{len(data)} bytes, Content-Length {length}")
            return data
        except Exception:  # noqa: BLE001 -- transient network errors; retried, then re-raised
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def cache_path(source: str, image_id: str) -> Path:
    return CACHE / source / f"{image_id}.jpg"


def done_ids(path: Path) -> set[str]:
    return set(pd.read_csv(path, usecols=["image_id"])["image_id"].astype(str)) if path.exists() else set()


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def work_added(row: dict, milk: zipfile.ZipFile | None) -> dict:
    image_id, source = str(row["isic_id"]), row["source"]
    name = f"images/{image_id}.jpg"
    data = milk.read(name) if source == "milk10k" and milk is not None and name in milk.NameToInfo else fetch(row["url"])
    dhash, phash = hash_bytes(data)
    if dhash == 0 and phash == 0:
        raise OSError("does not decode")
    small, width, height = resized_jpeg(data)
    path = cache_path(source, image_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    part.write_bytes(small)
    part.replace(path)
    return {"image_id": image_id, "source": source, "dhash": dhash, "phash": phash, "full_width": width, "full_height": height,
            "full_bytes": len(data), "cache_bytes": len(small)}


def work_hiba(image_id: str) -> dict:
    data = fetch(FULL_URL.format(image_id))
    dhash, phash = hash_bytes(data)
    if dhash == 0 and phash == 0:
        raise OSError("does not decode")
    return {"image_id": image_id, "dhash": dhash, "phash": phash}


def hiba_ids() -> list[str]:
    return sorted({r["isic_id"] for n in HIBA for r in data_audit.read(data_audit.CACHE / f"{n}.csv")})


def download(workers: int) -> int:
    testguard.block_test_reads("V6 added pool: training-only sources; HIBA hashed, not stored")
    candidates = pd.read_csv(CANDIDATES, low_memory=False)
    failures: list[tuple[str, str]] = []
    started = time.time()
    milk = zipfile.ZipFile(MILK_ZIP) if MILK_ZIP.exists() else None

    def run(tasks: list, work, ledger: Path, label: str) -> None:
        with ThreadPoolExecutor(workers) as pool:
            futures = {pool.submit(work, task): task for task in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    append_row(ledger, future.result())
                except Exception as error:  # noqa: BLE001 -- recorded; the next run retries it
                    failures.append((str(futures[future]), repr(error)[:160]))
                if i % 250 == 0 or i == len(tasks):
                    free = shutil.disk_usage(REPO).free / 1e9
                    print(f"{label}: {i:,}/{len(tasks):,}  failed {len(failures)}  C: {free:.1f} GB free  {(time.time() - started) / 60:.1f} min", flush=True)
                    if free < MIN_FREE_GB:
                        pool.shutdown(cancel_futures=True)
                        raise SystemExit(f"stopping: C: has {free:.1f} GB free, floor {MIN_FREE_GB}. Rerun to resume.")

    done = done_ids(HASHES)
    tasks = [r for r in candidates.to_dict("records") if str(r["isic_id"]) not in done]
    run(tasks, lambda row: work_added(row, milk), HASHES, "added")
    done = done_ids(HIBA_HASHES)
    run([i for i in hiba_ids() if i not in done], work_hiba, HIBA_HASHES, "hiba hash")
    print(f"finished in {(time.time() - started) / 60:.1f} min; {len(failures)} failed (rerun to retry)")
    for task, error in failures[:10]:
        print(" ", task[:90], error)
    return 1 if failures else 0


# ------------------------------------------------------------------ audit
def within(dhash: int, phash: int, d: np.ndarray, p: np.ndarray, radius: int = DUP_RADIUS) -> bool:
    if not len(d):
        return False
    return bool(((dedupe._popcount64(d ^ np.uint64(dhash)) <= radius) & (dedupe._popcount64(p ^ np.uint64(phash)) <= radius)).any())


def hash_audit(rows: pd.DataFrame, targets: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.Series:
    """First reason each row is a copy, or None. Fixed targets first, then earlier kept rows, so one image is kept once."""
    reasons: list[str | None] = [None] * len(rows)
    d_all, p_all = rows["dhash"].to_numpy(np.uint64), rows["phash"].to_numpy(np.uint64)
    keep = np.zeros(len(rows), dtype=bool)
    for i in range(len(rows)):
        dhash, phash = int(d_all[i]), int(p_all[i])
        for name, (d, p) in targets.items():
            if within(dhash, phash, d, p):
                reasons[i] = f"hash_copy_of_{name}"
                break
        if reasons[i] is None and within(dhash, phash, d_all[:i][keep[:i]], p_all[:i][keep[:i]]):
            reasons[i] = "hash_copy_of_earlier_added"
        keep[i] = reasons[i] is None
    return pd.Series(reasons, index=rows.index, dtype=object)


def linked_to_copies(rows: pd.DataFrame) -> pd.Series:
    """`hash_reason` extended by id: a copy of a corpus or HIBA image puts its lesion and patient in that cohort, so their other
    added images go too (section 0.3). Copies of the young extras or of other added rows are double counting only, so they do not spread."""
    out = rows["hash_reason"].copy()
    hit = rows["hash_reason"].isin(["hash_copy_of_corpus", "hash_copy_of_hiba"])
    for column in ("lesion_id", "patient_id"):
        values = rows[column].fillna("").astype(str)
        linked = set(values[hit]) - {""}
        out[out.isna() & values.isin(linked)] = f"same_{column}_as_a_hash_copy"
    return out


def build_rows(kept: pd.DataFrame) -> pd.DataFrame:
    """`manifest_v4.csv` columns plus `image_path`. No lesion id (DERM12345) falls back to the patient id, then the image id."""
    columns = list(pd.read_csv(data_audit.MANIFEST, nrows=0).columns)
    lesion = kept["lesion_id"].fillna("").astype(str)
    patient = kept["patient_id"].fillna("").astype(str)
    anchor = [le or pa or im for le, pa, im in zip(lesion, patient, kept["isic_id"].astype(str))]
    group = [f"__v6__{a}" for a in anchor]
    index_8 = {code: i for i, code in enumerate(head8.CLASSES_8)}
    labels_8 = kept["class_8"].map(index_8).to_numpy(int)
    labels_7 = head8.collapse_labels(labels_8)
    out = pd.DataFrame({
        "image_id": kept["isic_id"].astype(str).to_numpy(), "class_8": kept["class_8"].to_numpy(),
        "class_7": [head8.CLASSES_7[i] for i in labels_7], "class_index_8": labels_8, "class_index_7": labels_7,
        "escalating_8": kept["class_8"].isin(ESCALATING_8).to_numpy(), "escalating_7": kept["class_8"].isin(ESCALATING_8).to_numpy(),
        "age_approx": kept["age_approx"].to_numpy(), "age_band": [age_band(a) for a in kept["age_approx"]], "sex": kept["sex"].to_numpy(),
        "anatom_site_general": kept["anatom_site_general"].to_numpy(), "archive": kept["source"].to_numpy(), "in_ham": False,
        "lesion_id": lesion.to_numpy(), "effective_lesion_id": group, "lesion_id_was_null": (lesion == "").to_numpy(),
        "dup_cluster": 0, "group_id": group, "ham_split": np.nan, "split": "train_added",
        "image_path": [cache_path(s, i).relative_to(REPO).as_posix() for s, i in zip(kept["source"], kept["isic_id"].astype(str))],
    })
    return out[[*columns, "image_path"]]


def finalize() -> int:
    testguard.block_test_reads("V6 added pool: hash audit; HIBA hashes only")
    candidates = pd.read_csv(CANDIDATES, low_memory=False, dtype={"lesion_id": str, "patient_id": str})
    hashes = pd.read_csv(HASHES, dtype={"image_id": str}).drop_duplicates("image_id")
    missing = set(candidates["isic_id"].astype(str)) - set(hashes["image_id"])
    if missing:
        raise SystemExit(f"{len(missing)} candidates have no hash yet; rerun --download")
    hiba = pd.read_csv(HIBA_HASHES)
    if set(hiba_ids()) - set(hiba["image_id"].astype(str)):
        raise SystemExit("HIBA is not fully hashed; rerun --download")
    corpus = dedupe.load_hashes()
    extras_files = sorted(young_data.IMAGE_DIR.glob("*.jpg"))
    with ThreadPoolExecutor(4) as pool:
        extras_hashes = list(pool.map(lambda p: dedupe.hash_one(str(p)), extras_files))
    arrays = lambda values: np.array(values, dtype=np.uint64)  # noqa: E731
    targets = {"corpus": (corpus["dhash"].to_numpy(np.uint64), corpus["phash"].to_numpy(np.uint64)),
               "hiba": (arrays(hiba["dhash"].tolist()), arrays(hiba["phash"].tolist())),
               "young_extras": (arrays([h[0] for h in extras_hashes]), arrays([h[1] for h in extras_hashes]))}
    rows = candidates.assign(isic_id=candidates["isic_id"].astype(str)).merge(hashes, left_on="isic_id", right_on="image_id", validate="one_to_one",
                                                                              suffixes=("", "_hash"))
    rows["hash_reason"] = linked_to_copies(rows.assign(hash_reason=hash_audit(rows, targets)))
    kept = rows[rows["hash_reason"].isna()].reset_index(drop=True)
    out = build_rows(kept)
    out.to_csv(ADDED_TRAIN, index=False)
    per_source = {s: {"candidates": int((rows["source"] == s).sum()), "dropped": rows.loc[(rows["source"] == s) & rows["hash_reason"].notna(), "hash_reason"].value_counts().to_dict(),
                      "kept": int((out["archive"] == s).sum()), "kept_young": int(((out["archive"] == s) & (out["age_band"] == "<40")).sum()),
                      "kept_by_class_8": out.loc[out["archive"] == s, "class_8"].value_counts().to_dict()} for s in SOURCES}
    pool_rows = data_audit.read(data_audit.MANIFEST)
    dev = sum(r["split"] == "train" for r in pool_rows)
    dev_young = sum(r["split"] == "train" and r["age_band"] == "<40" for r in pool_rows)
    n_extras = len(data_audit.read(data_audit.EXTRAS))
    young_v5, rows_v5 = dev_young + n_extras, dev + n_extras
    report = {"rule": f"dHash AND pHash within radius {DUP_RADIUS}", "corpus_images": len(corpus), "extras_images": len(extras_files), "hiba_images_hashed": len(hiba),
              "hiba_pixels_stored": False, "sources": per_source, "added_rows": len(out), "added_young": int((out["age_band"] == "<40").sum()),
              "added_no_age": int((out["age_band"] == "unknown").sum()), "kept_by_class_8": out["class_8"].value_counts().to_dict(),
              "cache_gb": round(float(kept["cache_bytes"].sum()) / 1e9, 2), "cache_mean_kb": round(float(kept["cache_bytes"].mean()) / 1e3, 1),
              "full_gb_read": round(float(rows["full_bytes"].sum()) / 1e9, 2), "short_side": SHORT_SIDE, "jpeg_quality": JPEG_QUALITY,
              "young_share": {"v5_pool": young_v5 / rows_v5, "expanded_pool": (young_v5 + int((out["age_band"] == "<40").sum())) / (rows_v5 + len(out))},
              "added_train": ADDED_TRAIN.relative_to(REPO).as_posix()}
    AUDIT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    allocation = json.loads(data_audit.OUT.read_text(encoding="utf-8"))
    allocation.update({"note": "metadata links, plus the image-hash audit in results/v6/added_pool/audit.json", "image_hash_audit_done": True, "added_pool_audit": report})
    data_audit.OUT.write_text(json.dumps(allocation, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    for flag in ("--pull", "--select", "--download", "--finalize"):
        action.add_argument(flag, action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    if args.pull:
        return pull()
    if args.select:
        return select()
    if args.download:
        return download(args.workers)
    return finalize()


if __name__ == "__main__":
    raise SystemExit(main())
