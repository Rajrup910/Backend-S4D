"""Independent V7 data-gate audit (12 Nov 2026). Read-only: writes only into this folder.

Re-derives, from the files on disk rather than from the gate's own report:
  A. receipt hashes (master, links, gate code) against the bytes now on disk
  B. per-role ledger: images, groups, lesion ids, patient ids, class and age-band counts
  C. fold isolation of development rows by master_group, group_id, src_lesion_id, src_patient_id
  D. group sharing between training roles and the two protected roles (ham_test, midas)
  E. identical-bytes (sha256) and identical-pHash clusters spanning folds or roles
  F. confirmed / at-risk links spanning folds or protected roles
  G. files on disk; sha256 of a random sample recomputed from bytes; decode of the sample
  H. label conflicts inside one master_group

Never reads MIDAS labels or ham_test predictions: only ids, hashes and paths of those rows.
Run: python -I results/v7/audit_2026-11-12/independent_data_audit.py   (from the repo root)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "independent_data_audit.json"
PROTECTED = ("ham_test", "midas")
TRAIN_ROLES = ("train", "train_added", "train_extra", "train_hiba", "reserved", "val", "ham_val")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    res: dict = {}
    receipt = json.loads((REPO / "results/v7/data_gate_receipt.json").read_text(encoding="utf-8"))
    master_p, links_p = REPO / "ml/data/v7/master_v7.csv", REPO / "ml/data/v7/links_v7.csv"
    code = {k: sha(REPO / "research/data" / k) for k in receipt["code_sha256"]}
    res["A_receipt"] = {
        "master_matches": sha(master_p) == receipt["master_sha256"],
        "links_matches": sha(links_p) == receipt["links_sha256"],
        "code_mismatches": [k for k, v in code.items() if v != receipt["code_sha256"][k]],
        "receipt_written": receipt["written"],
    }

    m = pd.read_csv(master_p, low_memory=False)
    m["image_id"] = m["image_id"].astype(str)
    # labels and age: manifest_v4 (ISIC-2019 rows) + V7 pool table (added/extra) + HIBA training table
    man = pd.read_csv(REPO / "ml/data/manifest_v4.csv", low_memory=False)
    pool = pd.read_csv(REPO / "results/v7/added_pool_v7/added_train_v7.csv", low_memory=False)
    hiba = pd.read_csv(REPO / "results/v7/hiba_train/hiba_train.csv", low_memory=False)
    lab_cols = ["image_id", "class_7", "age_band", "age_approx"]
    labels = pd.concat([d[[c for c in lab_cols if c in d.columns]] for d in (man, pool, hiba)], ignore_index=True)
    labels["image_id"] = labels["image_id"].astype(str)
    labels = labels.drop_duplicates("image_id")
    mm = m.merge(labels, on="image_id", how="left")
    is_midas = mm["frozen_split"].eq("midas")
    mm.loc[is_midas, ["class_7", "age_band", "age_approx"]] = None  # sealed: never summarised

    ledger = {}
    for role, g in mm.groupby("frozen_split"):
        row = {
            "images": int(len(g)),
            "kept": int(g["status"].eq("kept").sum()),
            "master_groups": int(g["master_group"].nunique()),
            "group_ids": int(g["group_id"].nunique()),
            "src_lesion_ids": int(g["src_lesion_id"].nunique()),
            "rows_with_lesion_id": int(g["src_lesion_id"].notna().sum()),
            "src_patient_ids": int(g["src_patient_id"].nunique()),
            "rows_with_patient_id": int(g["src_patient_id"].notna().sum()),
            "archives": g["archive"].value_counts().to_dict(),
        }
        if role != "midas":
            row["label_missing"] = int(g["class_7"].isna().sum())
            row["class_7"] = g["class_7"].value_counts().to_dict()
            row["age_band"] = g["age_band"].fillna("MISSING").value_counts().to_dict()
            esc = g["class_7"].isin(["mel", "bcc", "akiec"])
            u40 = g["age_band"].eq("<40")
            row["under40_escalating_images"] = int((esc & u40).sum())
            row["under40_escalating_groups"] = int(g.loc[esc & u40, "master_group"].nunique())
        ledger[role] = row
    res["B_ledger"] = ledger

    dev = mm[mm["frozen_split"].eq("train")]
    iso = {}
    for key in ("master_group", "group_id", "src_lesion_id", "src_patient_id"):
        d = dev.dropna(subset=[key])
        n = d.groupby(key)["fold"].nunique()
        iso[key] = {"keys": int(len(n)), "keys_in_more_than_one_fold": int((n > 1).sum()),
                    "examples": n[n > 1].index[:5].tolist()}
    iso["fold_sizes"] = dev["fold"].value_counts().sort_index().to_dict()
    iso["dev_rows_without_fold"] = int(dev["fold"].isna().sum())
    res["C_fold_isolation"] = iso

    kept = mm[mm["status"].eq("kept")]
    share = {}
    for key in ("master_group", "src_lesion_id", "src_patient_id"):
        d = kept.dropna(subset=[key])
        roles = d.groupby(key)["frozen_split"].agg(lambda s: set(s))
        bad = roles[roles.apply(lambda s: bool(s & set(PROTECTED)) and bool(s & set(TRAIN_ROLES)))]
        share[key] = {"keys_shared_protected_and_training_kept": int(len(bad)), "examples": bad.index[:5].tolist()}
    res["D_protected_sharing"] = share

    def cross(col: str) -> dict:
        d = kept.dropna(subset=[col])
        dup = d[d.duplicated(col, keep=False)]
        out = {"rows_in_clusters": int(len(dup)), "clusters": int(dup[col].nunique())}
        dd = dup[dup["frozen_split"].eq("train")]
        nf = dd.groupby(col)["fold"].nunique()
        out["clusters_spanning_dev_folds"] = int((nf > 1).sum())
        ng = dup.groupby(col)["master_group"].nunique()
        out["clusters_spanning_master_groups"] = int((ng > 1).sum())
        roles = dup.groupby(col)["frozen_split"].agg(lambda s: set(s))
        out["clusters_touching_protected_and_training"] = int(
            roles.apply(lambda s: bool(s & set(PROTECTED)) and bool(s & set(TRAIN_ROLES))).sum())
        return out
    res["E_exact_duplicates"] = {"sha256": cross("sha256"), "phash_exact": cross("phash")}

    links = pd.read_csv(links_p, low_memory=False)
    # links name rows by master image_id (HIBA/MIDAS ids carry an "ARCHIVE:" prefix there)
    res["F_links_image_id_unique"] = bool(mm["image_id"].is_unique)
    idx = pd.Series(range(len(mm)), index=mm["image_id"]).groupby(level=0).first()

    def locate(series: pd.Series) -> pd.Series:
        return series.astype(str).map(idx)
    la, lb = locate(links["a"]), locate(links["b"])
    res["F_links"] = {"rows": int(len(links)), "tiers": links["tier"].value_counts().to_dict(),
                      "unresolved_endpoints": int(la.isna().sum() + lb.isna().sum())}
    ok = la.notna() & lb.notna()
    L = links[ok].copy()
    A, B = mm.iloc[la[ok].astype(int)].reset_index(drop=True), mm.iloc[lb[ok].astype(int)].reset_index(drop=True)
    L = L.reset_index(drop=True)
    both_kept = A["status"].eq("kept") & B["status"].eq("kept")
    same_group = A["master_group"].values == B["master_group"].values
    both_dev = A["frozen_split"].eq("train") & B["frozen_split"].eq("train")
    diff_fold = both_dev & (A["fold"].values != B["fold"].values)
    prot = (A["frozen_split"].isin(PROTECTED) & B["frozen_split"].isin(TRAIN_ROLES)) | \
           (B["frozen_split"].isin(PROTECTED) & A["frozen_split"].isin(TRAIN_ROLES))
    for tier in sorted(L["tier"].unique()):
        t = L["tier"].eq(tier)
        res["F_links"][f"{tier}"] = {
            "links": int(t.sum()),
            "cross_master_group": int((t & ~same_group).sum()),
            "dev_cross_fold_both_kept": int((t & diff_fold & both_kept).sum()),
            "protected_to_training_both_kept": int((t & prot & both_kept).sum()),
        }

    exists = mm["path"].map(lambda p: os.path.isfile(p))
    res["G_files"] = {"missing": int((~exists).sum()), "missing_by_role": mm.loc[~exists, "frozen_split"].value_counts().to_dict()}
    random.seed(20261112)
    sample = random.sample(range(len(mm)), 1500)
    bad_hash, bad_decode = [], []
    try:
        from PIL import Image
    except ImportError:
        Image = None
    for i in sample:
        p = Path(mm.at[i, "path"])
        if not p.is_file():
            continue
        if sha(p) != mm.at[i, "sha256"]:
            bad_hash.append(mm.at[i, "image_id"])
        if Image is not None:
            try:
                with Image.open(p) as im:
                    im.convert("RGB").load()
            except Exception:
                bad_decode.append(mm.at[i, "image_id"])
    res["G_files"].update({"sample": len(sample), "sha256_mismatch": bad_hash[:20], "n_sha256_mismatch": len(bad_hash),
                           "decode_failures": bad_decode[:20], "n_decode_failures": len(bad_decode),
                           "decoder": "PIL" if Image else "unavailable"})

    lab = mm[~is_midas].dropna(subset=["class_7"])
    nc = lab.groupby("master_group")["class_7"].nunique()
    conflicted = nc[nc > 1]
    res["H_label_conflicts"] = {
        "master_groups_with_more_than_one_class": int(len(conflicted)),
        "rows_in_them": int(lab["master_group"].isin(conflicted.index).sum()),
        "in_dev": int(lab[lab["frozen_split"].eq("train") & lab["master_group"].isin(conflicted.index)]["master_group"].nunique()),
        "examples": conflicted.index[:8].tolist(),
    }

    OUT.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
