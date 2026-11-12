"""Per-run isolation check, re-derived without calling the gate's own `removals()` or `training_rows()`.

For each V7 run form (fold 0-4, archive holdout BCN/HAM/MSKCC) take the training rows the receipt allows
(everything on the training side, minus the receipt's global and per-run removal lists, minus the evaluated rows)
and count every remaining connection to an evaluated row: shared master_group, shared group_id, shared src_lesion_id,
shared src_patient_id, identical sha256, identical pHash, confirmed link, at-risk link. Also counts connections from
any training row of any run to ham_test or MIDAS.
Run from the repo root: python -I results/v7/audit_2026-11-12/independent_fold_isolation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "independent_fold_isolation.json"
TRAINING_ONLY = ("train_added", "train_extra", "train_hiba", "reserved", "val", "ham_val")
SPENT = ("ham_test", "midas")


#: Patient ids read straight from the archives' own metadata, not from the builder's `src_patient_id` (added 13 Nov 2026:
#: the builder had left the young extras and ISIC 2020 without ids, and a check on its own column could not see that).
ARCHIVE_META = ("results/v6/archive_meta/isic2019_train.csv", "results/v6/archive_meta/isic2020.csv",
                "results/v6/archive_meta/derm12345.csv", "results/v5/young_data/candidates.csv")


def archive_patients() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ARCHIVE_META:
        meta = pd.read_csv(REPO / name, dtype=str)
        key = "isic_id" if "isic_id" in meta.columns else "image_id"
        meta = meta.dropna(subset=["patient_id"])
        out.update(zip(meta[key].str.replace("_downsampled", "", regex=False), meta["patient_id"].str.strip()))
    return out


def compute() -> dict:
    m = pd.read_csv(REPO / "ml/data/v7/master_v7.csv", low_memory=False)
    m["image_id"] = m["image_id"].astype(str)
    m["archive_patient"] = m["archive_id"].astype(str).str.replace("_downsampled", "", regex=False).map(archive_patients())
    links = pd.read_csv(REPO / "ml/data/v7/links_v7.csv", low_memory=False)
    rec = json.loads((REPO / "results/v7/data_gate_receipt.json").read_text(encoding="utf-8"))
    rem = rec["report"]["removal_rows"]
    glob = set(rem["global"])
    split = m["frozen_split"].astype(str)
    side = split.eq("train") | split.isin(TRAINING_ONLY)

    def connections(train: pd.DataFrame, evald: pd.DataFrame) -> dict:
        out = {}
        for key in ("master_group", "group_id", "src_lesion_id", "src_patient_id", "archive_patient", "sha256", "phash"):
            a = set(train[key].dropna().astype(str))
            b = set(evald[key].dropna().astype(str))
            out[key] = len(a & b)
        t, e = set(train["image_id"]), set(evald["image_id"])
        for tier in ("confirmed", "at_risk"):
            L = links[links["tier"].eq(tier)]
            ab = L["a"].astype(str).isin(t) & L["b"].astype(str).isin(e)
            ba = L["b"].astype(str).isin(t) & L["a"].astype(str).isin(e)
            out[f"{tier}_links"] = int((ab | ba).sum())
        return out

    res = {"runs": {}}
    for f in range(5):
        drop = glob | set(rem["by_fold"][str(f)])
        evald = m[split.eq("train") & m["fold"].eq(f)]
        train = m[side & ~(split.eq("train") & m["fold"].eq(f)) & ~m["image_id"].isin(drop)]
        res["runs"][f"fold{f}"] = {"train_rows": int(len(train)), "eval_rows": int(len(evald)),
                                   "removed_for_run": len(drop), **connections(train, evald)}
    for arch in ("bcn20000", "ham", "mskcc"):
        drop = glob | set(rem["by_holdout"][arch])
        evald = m[split.eq("train") & m["archive"].eq(arch)]
        train = m[side & ~m["archive"].eq(arch) & ~m["image_id"].isin(drop)]
        res["runs"][f"holdout_{arch}"] = {"train_rows": int(len(train)), "eval_rows": int(len(evald)),
                                          "removed_for_run": len(drop), **connections(train, evald)}
    any_train = m[side & ~m["image_id"].isin(glob)]
    res["any_training_vs_spent"] = connections(any_train, m[split.isin(SPENT)])
    return res


def main() -> int:
    res = compute()
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
