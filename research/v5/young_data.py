"""Young, histology-confirmed dermoscopy for the `youngdata` arm (audit AU24, runsheet section 4).

    python -m research.v5.young_data --count          # metadata only, no image downloaded
    python -m research.v5.young_data --download       # owner-approved, after --count

Pool: ISIC Archive images with age_approx <= 35 (the corpus's "<40" band), diagnosis confirmed by
histopathology, dermoscopic. **Excluded before anything is fetched:**
  * every image of the V4 corpus (`manifest_v4.csv`, all 25,331 ISIC-2019 rows, test and reserved
    included);
  * the MILK10k collections (V5's S84 cohort) and the HIBA collections (V6's confirmation cohort);
  * PAD-UFES-20 (smartphone, V6 routing);
  * diagnoses that do not map to the 7 classes.
**Excluded after download:** any image that is a perceptual duplicate of one of the 27,629 hashed
corpus images (`research/v4/dedupe.py`: dHash AND pHash within radius 3, the S49 rule).

Rows are **train-only**: `train_v5 --extra-train` appends them to the training frame and refuses
any that share an image or lesion group with the development partition. No row is ever held out,
so every paired fold-0 comparison keeps identical evaluation rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from research.v4.recipe import MANIFEST, REPO_ROOT

API = "https://api.isic-archive.com/api/v2/images/search/"
QUERY = 'age_approx:[0 TO 35] AND diagnosis_confirm_type:"histopathology" AND image_type:"dermoscopic"'
#: ISIC collection ids, read from /api/v2/collections/ on 2026-09-30.
EXCLUDED_COLLECTIONS = {425: "MILK10k", 424: "MILK10k Benchmark",
                        251: "HIBA 2019-2022", 176: "HIBA Skin Lesions (Hospital Italiano)",
                        175: "HIBA Skin Lesions", 406: "PAD-UFES-20"}
OUT_DIR = REPO_ROOT / "results" / "v5" / "young_data"
IMAGE_DIR = REPO_ROOT / "data" / "external" / "isic_young_v5"
DUP_RADIUS = 3  # research/v4/dedupe.py CLUSTER_RADIUS (S49)

#: ISIC hierarchical diagnosis -> the 7 V4 classes. Matched on diagnosis_2, then diagnosis_3;
#: anything else is excluded and listed in the count report for review.
CLASS_RULES: tuple[tuple[str, str, str], ...] = (
    ("diagnosis_2", "Malignant melanocytic proliferations", "mel"),
    ("diagnosis_3", "Basal cell carcinoma", "bcc"),
    ("diagnosis_3", "Squamous cell carcinoma", "akiec"),
    ("diagnosis_3", "Solar or actinic keratosis", "akiec"),
    ("diagnosis_3", "Actinic keratosis", "akiec"),
    ("diagnosis_2", "Benign melanocytic proliferations", "nv"),
    ("diagnosis_3", "Seborrheic keratosis", "bkl"),
    ("diagnosis_3", "Pigmented benign keratosis", "bkl"),
    ("diagnosis_3", "Solar lentigo", "bkl"),
    ("diagnosis_3", "Lichen planus like keratosis", "bkl"),
    ("diagnosis_3", "Dermatofibroma", "df"),
    ("diagnosis_2", "Benign soft tissue proliferations - Vascular", "vasc"),
)
CLASS_7 = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
ESCALATING = {"mel", "bcc", "akiec"}


def _get(params: dict | None = None, url: str | None = None) -> dict:
    target = url or API + "?" + urllib.parse.urlencode(params or {})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(target, timeout=90) as response:
                return json.load(response)
        except Exception:  # noqa: BLE001 -- transient API errors; retried, then re-raised
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _all_ids(params: dict) -> list[dict]:
    rows, page = [], _get({**params, "limit": 100})
    while True:
        rows.extend(page["results"])
        if not page.get("next"):
            return rows
        page = _get(url=page["next"])


def map_class(clinical: dict) -> str | None:
    for field, needle, code in CLASS_RULES:
        if needle.lower() in str(clinical.get(field, "")).lower():
            return code
    return None


def count() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus = set(pd.read_csv(MANIFEST, usecols=["image_id"])["image_id"].astype(str))
    excluded: dict[str, int] = {}
    blocked: set[str] = set()
    for cid, name in EXCLUDED_COLLECTIONS.items():
        ids = {r["isic_id"] for r in _all_ids({"query": QUERY, "collections": cid})}
        excluded[name] = len(ids)
        blocked |= ids
    pool = _all_ids({"query": QUERY})
    rows, unmapped = [], Counter()
    for r in pool:
        clinical = r["metadata"].get("clinical", {})
        iid = r["isic_id"]
        if iid in corpus or iid in blocked:
            continue
        code = map_class(clinical)
        if code is None:
            unmapped[f"{clinical.get('diagnosis_2')} | {clinical.get('diagnosis_3')}"] += 1
            continue
        rows.append({"image_id": iid, "class_7": code, "age_approx": clinical.get("age_approx"),
                     "sex": clinical.get("sex"),
                     "anatom_site_general": clinical.get("anatom_site_general",
                                                         clinical.get("anatom_site_1")),
                     "lesion_id": clinical.get("lesion_id"), "patient_id": clinical.get("patient_id"),
                     "url": r["files"]["full"]["url"], "bytes": r["files"]["full"]["size"],
                     "license": r.get("copyright_license"), "attribution": r.get("attribution")})
    frame = pd.DataFrame(rows)
    # Lesion-level exclusion. The corpus's lesion ids (HAM_*, MSK4_*, BCN_*) are not ISIC `IL_*`
    # ids, so an id join proves nothing. Instead fetch every archive image of each candidate
    # lesion; if any of them is a corpus image or in an excluded collection, drop the lesion.
    lesion_hits: dict[str, int] = {}
    for lesion in sorted(frame["lesion_id"].dropna().unique()):
        ids = {r["isic_id"] for r in _all_ids({"query": f'lesion_id:"{lesion}"'})}
        hit = len(ids & corpus) + len(ids & blocked)
        if hit:
            lesion_hits[lesion] = hit
    leaked = frame["lesion_id"].isin(lesion_hits)
    frame = frame[~leaked].reset_index(drop=True)
    frame.to_csv(OUT_DIR / "candidates.csv", index=False)
    report = {
        "query": QUERY, "api": API, "pool_total": len(pool),
        "excluded_in_v4_corpus": sum(r["isic_id"] in corpus for r in pool),
        "excluded_collections": excluded,
        "excluded_unmapped_diagnosis": dict(unmapped.most_common()),
        "excluded_lesion_seen_in_corpus_or_blocked": {"lesions": len(lesion_hits),
                                                      "images": int(leaked.sum())},
        "candidates_without_lesion_id": int(frame["lesion_id"].isna().sum()),
        "candidates": len(frame),
        "by_class": frame["class_7"].value_counts().to_dict() if len(frame) else {},
        "escalating": int(frame["class_7"].isin(ESCALATING).sum()) if len(frame) else 0,
        "unique_lesion_ids": int(frame["lesion_id"].nunique()) if len(frame) else 0,
        "licenses": frame["license"].value_counts().to_dict() if len(frame) else {},
        "download_bytes": int(frame["bytes"].sum()) if len(frame) else 0,
        "images_downloaded": 0, "test_read": False, "reserved_read": False,
        "note": "Metadata only. Perceptual-duplicate removal happens after the approved download.",
    }
    out = OUT_DIR / "count.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"pool {report['pool_total']:,}; candidates {report['candidates']:,} "
          f"({report['escalating']} escalating); by class {report['by_class']}; "
          f"download {report['download_bytes'] / 1e9:.2f} GB")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return out


def _is_duplicate(dh: int, ph: int, corpus_d: np.ndarray, corpus_p: np.ndarray) -> bool:
    from research.v4.dedupe import _popcount64

    d = _popcount64(corpus_d ^ np.uint64(dh))
    p = _popcount64(corpus_p ^ np.uint64(ph))
    return bool(((d <= DUP_RADIUS) & (p <= DUP_RADIUS)).any())


def _decodes(path: Path) -> bool:
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
        return True
    except Exception:  # noqa: BLE001 -- any decode failure means re-fetch
        return False


def _fetch(url: str, path: Path, attempts: int = 5) -> None:
    """Download to a .part file, check it against the server's Content-Length and that it
    decodes, then rename -- a dropped connection never leaves a truncated jpg (30 Sep
    IncompleteRead). The archive's recorded `size` is not used: it can be stale."""
    part = path.with_suffix(".part")
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                length = response.headers.get("Content-Length")
                data = response.read()
            if length is not None and len(data) != int(length):
                raise IOError(f"{len(data)} bytes, Content-Length {length}")
            part.write_bytes(data)
            if not _decodes(part):
                raise IOError("downloaded file does not decode")
            part.replace(path)
            return
        except Exception:  # noqa: BLE001 -- transient network errors; retried, then re-raised
            part.unlink(missing_ok=True)
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))


def download() -> Path:
    from research.v4.dedupe import hash_one, load_hashes

    candidates = OUT_DIR / "candidates.csv"
    if not candidates.is_file():
        raise SystemExit("run --count first; the owner approves the download from count.json")
    frame = pd.read_csv(candidates, low_memory=False)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    hashes = load_hashes()
    corpus_d = hashes["dhash"].to_numpy(np.uint64)
    corpus_p = hashes["phash"].to_numpy(np.uint64)
    kept, dropped = [], {"duplicate_of_corpus": 0, "unreadable": 0}
    for i, row in enumerate(frame.itertuples(index=False), 1):
        path = IMAGE_DIR / f"{row.image_id}.jpg"
        if not path.is_file() or not _decodes(path):
            _fetch(row.url, path)
        dh, ph = hash_one(str(path))
        if dh == 0 and ph == 0:
            dropped["unreadable"] += 1
            continue
        if _is_duplicate(dh, ph, corpus_d, corpus_p):
            dropped["duplicate_of_corpus"] += 1
            path.unlink()
            continue
        kept.append(row._asdict())
        if i % 200 == 0:
            print(f"  {i:,}/{len(frame):,}  kept {len(kept):,}")
    out = build_manifest_rows(pd.DataFrame(kept))
    report = json.loads((OUT_DIR / "count.json").read_text(encoding="utf-8"))
    report.update({"images_downloaded": len(frame), "dropped": dropped, "kept": len(out),
                   "kept_by_class": out["class_7"].value_counts().to_dict(),
                   "extra_train_sha256": hashlib.sha256(
                       (OUT_DIR / "extra_train.csv").read_bytes()).hexdigest()})
    (OUT_DIR / "count.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"kept {len(out):,} rows; dropped {dropped}; wrote results/v5/young_data/extra_train.csv")
    return OUT_DIR / "extra_train.csv"


def build_manifest_rows(kept: pd.DataFrame) -> pd.DataFrame:
    """`manifest_v4.csv` columns for the kept rows, so train_v5 can append them unchanged."""
    columns = pd.read_csv(MANIFEST, nrows=0).columns
    lesion = kept["lesion_id"].where(kept["lesion_id"].notna(), None)
    group = [f"__young__{l}" if isinstance(l, str) and l else f"__young__{i}"
             for l, i in zip(lesion, kept["image_id"])]
    code_index = {c: i for i, c in enumerate(CLASS_7)}
    out = pd.DataFrame({
        "image_id": kept["image_id"], "class_8": kept["class_7"], "class_7": kept["class_7"],
        "class_index_8": kept["class_7"].map(code_index), "class_index_7": kept["class_7"].map(code_index),
        "escalating_8": kept["class_7"].isin(ESCALATING), "escalating_7": kept["class_7"].isin(ESCALATING),
        "age_approx": kept["age_approx"], "age_band": "<40", "sex": kept["sex"],
        "anatom_site_general": kept["anatom_site_general"], "archive": "isic_young",
        "in_ham": False, "lesion_id": lesion, "effective_lesion_id": group,
        "lesion_id_was_null": lesion.isna(), "dup_cluster": 0, "group_id": group,
        "ham_split": np.nan, "split": "train_extra",
    })
    out = out[list(columns)]
    out.to_csv(OUT_DIR / "extra_train.csv", index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--count", action="store_true", help="metadata only (the default)")
    mode.add_argument("--download", action="store_true", help="owner-approved image download")
    args = parser.parse_args(argv)
    if args.download:
        download()
    else:
        count()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
