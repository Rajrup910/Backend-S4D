"""Independent data checks, no repo write. (1) Class map against known labels: the V6 map applied to the ISIC 2019 Training archive records of the
manifest's own rows. (2) Age bands likewise. (3) Leak links recomputed from a fresh pull, separate from added_pool. (4) Every cached image decodes.
(5) Hash sensitivity at wider radii against development rows."""
import json, sys, urllib.request, time, csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from research import testguard
from research.v4 import dedupe
from research.v5 import young_data
from research.v6 import added_pool, data_audit, head8

SCRATCH = Path(sys.argv[1]); META = SCRATCH / "isic2019_train_full.csv"

def pull():
    if META.exists(): return
    rows, page = [], young_data._get({"collections": "65", "limit": 100})
    while True:
        for item in page["results"]:
            c = item["metadata"].get("clinical", {})
            rows.append({"isic_id": item["isic_id"], **{k: c.get(k, "") for k in ("lesion_id", "patient_id", "age_approx", "diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4")}})
        if not page.get("next"): break
        page = young_data._get(url=page["next"])
    pd.DataFrame(rows).to_csv(META, index=False)

if __name__ == "__main__":
    testguard.block_test_reads("V6 data audit")
    a = pd.read_csv(added_pool.ADDED_TRAIN, low_memory=False, dtype={"image_id": str})
    m = pd.read_csv(data_audit.MANIFEST, low_memory=False, dtype={"image_id": str})
    # (4) decode
    t = time.time()
    def ok(p):
        try:
            with Image.open(p) as im:
                im.load(); return im.size, im.mode
        except Exception as e:
            return repr(e)[:80]
    with ThreadPoolExecutor(6) as pool:
        res = list(pool.map(ok, a["image_path"]))
    bad = [(p, r) for p, r in zip(a["image_path"], res) if isinstance(r, str)]
    sizes = np.array([r[0] for r in res if not isinstance(r, str)]); modes = pd.Series([r[1] for r in res if not isinstance(r, str)]).value_counts().to_dict()
    print(f"DECODE {len(a):,} cached images in {(time.time()-t)/60:.1f} min: bad {len(bad)} {bad[:3]} modes {modes} short side min/median {sizes.min(1).min()}/{np.median(sizes.min(1))} long max {sizes.max()}", flush=True)
    # (5) hash sensitivity
    corpus = dedupe.load_hashes(); h = pd.read_csv(added_pool.HASHES, dtype={"image_id": str}).drop_duplicates("image_id")
    h = h[h.image_id.isin(a.image_id)]
    ids = corpus["image_id"].astype(str).str.replace("_downsampled", "", regex=False)
    dev = set(m[m.split == "train"].image_id.str.replace("_downsampled", "", regex=False))
    is_dev = ids.isin(dev).to_numpy()
    d, p = corpus["dhash"].to_numpy(np.uint64), corpus["phash"].to_numpy(np.uint64)
    print("corpus rows", len(d), "dev rows among them", int(is_dev.sum()), flush=True)
    out = {r: {"any": 0, "dev": 0} for r in (3, 4, 5, 6, 8)}
    byr = {r: [] for r in out}
    for row in h.itertuples():
        dd = dedupe._popcount64(d ^ np.uint64(int(row.dhash))); pp = dedupe._popcount64(p ^ np.uint64(int(row.phash)))
        for r in out:
            hit = (dd <= r) & (pp <= r)
            if hit.any():
                out[r]["any"] += 1
                if (hit & is_dev).any():
                    out[r]["dev"] += 1; byr[r].append(row.source)
    print("HASH wider radius, kept added rows near any corpus image / a development row:", out, flush=True)
    for r in (5, 6, 8): print(" radius", r, "dev hits by source", pd.Series(byr[r]).value_counts().to_dict())
    # (1)(2)(3) fresh metadata
    pull()
    meta = pd.read_csv(META, low_memory=False, dtype=str).fillna("")
    mm = m.assign(arch=m.image_id.str.replace("_downsampled", "", regex=False)).merge(meta, left_on="arch", right_on="isic_id", how="left", suffixes=("", "_arch"))
    have = mm[mm.isic_id.notna()]
    print(f"MANIFEST rows {len(m):,}, with a fresh ISIC 2019 record {len(have):,}; without: {mm[mm.isic_id.isna()].archive.value_counts().to_dict()}", flush=True)
    mapped = [added_pool.map_class_8(r) for r in have[["diagnosis_2", "diagnosis_3", "diagnosis_4"]].to_dict("records")]
    have = have.assign(mapped=mapped)
    print("CLASS MAP vs manifest class_8 (rows):")
    print(pd.crosstab(have["class_8"], have["mapped"].fillna("(excluded)")).to_string(), flush=True)
    diff = have[have["mapped"].notna() & (have["mapped"] != have["class_8"])]
    print("disagreements", len(diff), "of", int(have["mapped"].notna().sum()))
    print(diff.groupby(["class_8", "diagnosis_3", "mapped"]).size().sort_values(ascending=False).head(25).to_string(), flush=True)
    ex = have[have["mapped"].isna()]
    print("excluded by map though the manifest labels them:", len(ex)); print(ex.groupby(["class_8", "diagnosis_3"]).size().sort_values(ascending=False).head(15).to_string(), flush=True)
    age = pd.to_numeric(have.age_approx_arch if "age_approx_arch" in have else have.age_approx, errors="coerce")
    bands = [added_pool.age_band(x) for x in age]
    print("AGE BAND disagreements with manifest:", int((np.array(bands) != have.age_band.to_numpy()).sum()), "of", len(have), flush=True)
    # (3) leak recheck by lesion and patient on fresh metadata
    src = pd.concat([pd.read_csv(added_pool.meta_path(s), low_memory=False, dtype=str).fillna("").assign(source=s) for s in added_pool.SOURCES])
    kept = src[src.isic_id.isin(a.image_id)]
    man_les, man_pat = set(have.lesion_id_arch if "lesion_id_arch" in have else have.lesion_id) - {""}, set(have.patient_id) - {""}
    dev_rows = have[have.split == "train"]
    print("LEAK fresh metadata, kept added rows: shared lesions with ISIC 2019 Training records", int(kept.lesion_id.isin(man_les).sum()), "shared patients", int(kept.patient_id.isin(man_pat).sum()), "rows", len(kept), "unmatched", len(a) - len(kept), flush=True)
    d6 = set(json.loads(data_audit.LINKS.read_text(encoding="utf-8"))["linked_lesion_ids"])
    print(" MILK rows in D6", int(kept[kept.source == "milk10k"].lesion_id.isin(d6).sum()))
    hib = pd.concat([pd.read_csv(data_audit.CACHE / f"{n}.csv", usecols=["isic_id", "lesion_id", "patient_id"], dtype=str).fillna("") for n in added_pool.HIBA])
    print(" HIBA shared images", int(kept.isic_id.isin(hib.isic_id).sum()), "lesions", int(kept.lesion_id.isin(set(hib.lesion_id) - {""}).sum()), "patients", int(kept.patient_id.isin(set(hib.patient_id) - {""}).sum()))
    print(" manifest image ids in added", int(a.image_id.isin(set(m.image_id.str.replace("_downsampled", "", regex=False))).sum()), "| extras in added", int(a.image_id.isin(set(pd.read_csv(data_audit.EXTRAS, dtype=str).image_id)).sum()))
    print(" same lesion across added sources:", int(kept[kept.lesion_id != ""].groupby("lesion_id").source.nunique().gt(1).sum()))
    # class mix
    pool = pd.concat([m[m.split == "train"][["class_8", "age_band"]], pd.read_csv(data_audit.EXTRAS, usecols=["class_8", "age_band"]), a[["class_8", "age_band"]]])
    print("EXPANDED class counts", pool.class_8.value_counts().to_dict())
