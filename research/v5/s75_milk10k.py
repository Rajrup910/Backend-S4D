# ruff: noqa: I001  -- precheck_common must be imported first: it hides the GPU before torch loads
"""S75 -- MILK10k eligibility, counts only (runsheet section 9; pre-registered before download).

    python -m research.v5.s75_milk10k

No model is run and no prediction is made. Metadata and the dermoscopic images only:
  1. one dermoscopic image per lesion (`image_type == dermoscopic`); clinical images are not used;
  2. diagnoses -> the 7 classes with the young-data rules (`research.v5.young_data.CLASS_RULES`,
     plus keratoacanthoma -> akiec, runsheet: "akiec = AK + SCC/KA"); anything unmapped
     (inflammatory / infectious / other) goes to the out-of-scope stratum, counted separately;
  3. a lesion is EXCLUDED if its ISIC id is one of the 25,331 ISIC-2019 images, or its image's
     perceptual hashes match any of them under the S49 rule (`research/v4/dedupe.py`: dHash AND
     pHash both within Hamming radius 3, same hash functions, same cache);
  4. cohort label: "cohort-external, partly same-institution" (Vienna and MSKCC contribute);
  5. counts by class x age band, including the under-40 escalating lesion count.
Output: results/v5/s75_milk10k_eligibility.json. The S84 read (one execution) is separate.
"""

from __future__ import annotations

from research.v5 import precheck_common as pc  # first: hides the GPU

import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research.v4 import dedupe  # noqa: E402
from research.v5.young_data import CLASS_7, CLASS_RULES, ESCALATING  # noqa: E402

MILK_DIR = pc.REPO_ROOT / "data" / "external" / "milk10k"
MILK_ZIP = MILK_DIR / "milk10k.zip"
MILK_META = MILK_DIR / "milk10k.csv"
OUT = pc.REPO_ROOT / "results" / "v5" / "s75_milk10k_eligibility.json"
RADIUS = dedupe.CLUSTER_RADIUS  # 3 (S49)
EXTRA_RULES = (("diagnosis_3", "Keratoacanthoma", "akiec"), ("diagnosis_4", "Keratoacanthoma", "akiec"))
BANDS = ("<40", "40-59", "60+", "unknown")
COHORT_LABEL = "cohort-external, partly same-institution"


def map_class(row: pd.Series) -> str | None:
    for field, needle, code in CLASS_RULES + EXTRA_RULES:
        if needle.lower() in str(row.get(field, "") or "").lower():
            return code
    return None


def age_band(age) -> str:
    if pd.isna(age):
        return "unknown"
    age = float(age)
    return "<40" if age < 40 else ("40-59" if age < 60 else "60+")


def popcount(x: np.ndarray) -> np.ndarray:
    return dedupe._popcount64(x)


def hash_bytes(data: bytes) -> tuple[int, int]:
    """dedupe.hash_one on in-memory bytes (same reduced decode, same two hashes)."""
    import cv2

    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_8)
    if img is None:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return (0, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (int(dedupe._dhash(gray)), int(dedupe._phash(gray)))


def main() -> int:
    context = pc.start("S75")
    meta = pd.read_csv(MILK_META, low_memory=False)
    derm = meta[meta["image_type"].astype(str).str.lower() == "dermoscopic"].copy()
    n_lesions_all = int(meta["lesion_id"].nunique())
    derm = derm.drop_duplicates("lesion_id", keep="first").reset_index(drop=True)

    hashes = dedupe.load_hashes()
    isic = hashes[hashes["archive"] == "isic2019"]
    if len(isic) != 25_331:
        raise SystemExit(f"expected 25,331 ISIC-2019 hashes, found {len(isic)}")
    isic_ids = set(isic["image_id"].astype(str))
    d_ref = isic["dhash"].to_numpy(dtype=np.uint64)
    p_ref = isic["phash"].to_numpy(dtype=np.uint64)

    started = time.time()
    id_match, hash_match, unreadable = [], [], 0
    with zipfile.ZipFile(MILK_ZIP) as z:
        names = {n.rsplit("/", 1)[-1].rsplit(".", 1)[0]: n for n in z.namelist()
                 if n.lower().endswith(".jpg")}
        for i, iid in enumerate(derm["isic_id"].astype(str)):
            id_match.append(iid in isic_ids)
            if iid not in names:
                hash_match.append(False)
                unreadable += 1
                continue
            dh, ph = hash_bytes(z.read(names[iid]))
            if dh == 0 and ph == 0:
                unreadable += 1
                hash_match.append(False)
                continue
            both = ((popcount(d_ref ^ np.uint64(dh)) <= RADIUS)
                    & (popcount(p_ref ^ np.uint64(ph)) <= RADIUS))
            hash_match.append(bool(both.any()))
            if (i + 1) % 1000 == 0:
                print(f"  hashed {i + 1:,}/{len(derm):,}  {time.time() - started:.0f}s")
    derm["id_match"] = id_match
    derm["hash_match"] = hash_match
    derm["excluded"] = derm["id_match"] | derm["hash_match"]
    derm["class_7"] = derm.apply(map_class, axis=1)
    derm["age_band"] = derm["age_approx"].map(age_band)

    eligible = derm[~derm["excluded"]]
    in_scope = eligible[eligible["class_7"].notna()]
    out_scope = eligible[eligible["class_7"].isna()]
    table = {c: {b: int(((in_scope["class_7"] == c) & (in_scope["age_band"] == b)).sum())
                 for b in BANDS} for c in CLASS_7}
    esc = in_scope["class_7"].isin(ESCALATING)
    unmapped = Counter(f"{r.diagnosis_2} | {r.diagnosis_3}" for r in out_scope.itertuples())
    report = {
        "cohort_label": COHORT_LABEL,
        "source": {"zip": MILK_ZIP.name,
                   "zip_sha256": hashlib.sha256(MILK_ZIP.read_bytes()).hexdigest()},
        "lesions_total": n_lesions_all,
        "lesions_with_dermoscopic_image": int(len(derm)),
        "excluded": {"total": int(derm["excluded"].sum()), "isic_id_match": int(derm["id_match"].sum()),
                     "perceptual_hash_match": int(derm["hash_match"].sum()),
                     "rule": f"S49: dHash AND pHash within Hamming {RADIUS} of any ISIC-2019 image"},
        "unreadable_images": unreadable,
        "eligible_lesions": int(len(eligible)),
        "in_scope_lesions": int(len(in_scope)),
        "out_of_scope_lesions": int(len(out_scope)),
        "out_of_scope_diagnoses": dict(unmapped.most_common()),
        "counts_class_by_age_band": table,
        "escalating_by_age_band": {b: int((esc & (in_scope["age_band"] == b)).sum()) for b in BANDS},
        "under40_escalating_lesions": int((esc & (in_scope["age_band"] == "<40")).sum()),
        "under40_in_scope_lesions": int((in_scope["age_band"] == "<40").sum()),
        "histopathology_confirmed_in_scope": int(
            (in_scope["diagnosis_confirm_type"].astype(str) == "histopathology").sum()),
        "counts_only": True, "predictions_made": False, "test_read": False,
        "reserved_read": False, "context": context,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("lesions_with_dermoscopic_image", "excluded",
                                              "eligible_lesions", "in_scope_lesions",
                                              "out_of_scope_lesions", "escalating_by_age_band",
                                              "under40_escalating_lesions")}, indent=2))
    print(f"wrote {OUT.relative_to(pc.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
