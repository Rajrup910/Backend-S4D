"""Leak audit by identifiers: lesion and patient ids across development folds, young extras, the added pool and HIBA's metadata. Metadata only: no image, no prediction, no HIBA label.

    python scripts/audit_leak_ids.py

Writes results/v6/screens/leak_audit_ids.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import testguard  # noqa: E402
from research.v4.s71_kfold import load_assignments  # noqa: E402
from research.v6 import added_pool, data_audit  # noqa: E402

OUT = REPO / "results/v6/screens/leak_audit_ids.json"
NULL = {"", "nan", "none"}


def clean(values) -> set[str]:
    return {str(v) for v in values if str(v).strip().lower() not in NULL}


def main() -> int:
    testguard.block_test_reads("leak audit by identifiers: metadata only")
    cache = {n: data_audit.read(data_audit.CACHE / f"{n}.csv") for n in data_audit.SOURCES}
    meta = {r["isic_id"]: r for n in ("isic2019_train", "isic2019_test", "isic2020", "milk10k") for r in cache[n]}

    folds = load_assignments()[["image_id", "fold"]].copy()
    folds["image_id"] = folds["image_id"].astype(str)
    manifest = pd.read_csv(data_audit.MANIFEST, low_memory=False, dtype=str)[["image_id", "archive"]]
    dev = folds.merge(manifest, on="image_id", how="left")
    dev["isic"] = [data_audit.archive_id(i) for i in dev["image_id"]]
    dev["patient"] = [meta.get(i, {}).get("patient_id", "") for i in dev["isic"]]
    dev["lesion"] = [meta.get(i, {}).get("lesion_id", "") for i in dev["isic"]]
    has_patient = ~dev["patient"].astype(str).str.strip().str.lower().isin(NULL)
    report: dict = {"dev_rows": len(dev), "dev_rows_with_patient_id": int(has_patient.sum()),
                    "dev_patient_id_share_by_archive": {str(k): float(v) for k, v in has_patient.groupby(dev["archive"]).mean().items()}}

    # across folds: lesion grouping does not promise patient grouping
    known = dev[has_patient]
    per_patient = known.groupby("patient")["fold"].nunique()
    spread = set(per_patient[per_patient > 1].index)
    report["patients_with_id"] = int(per_patient.size)
    report["patients_in_more_than_one_fold"] = len(spread)
    report["fold0_rows_whose_patient_is_also_in_training_folds"] = int((known["fold"].eq(0) & known["patient"].isin(spread)).sum())
    report["fold0_rows"] = int(dev["fold"].eq(0).sum())
    report["rows_whose_patient_spans_folds_share"] = float(known["patient"].isin(spread).mean()) if len(known) else None
    lesions = dev[~dev["lesion"].astype(str).str.strip().str.lower().isin(NULL)].groupby("lesion")["fold"].nunique()
    report["lesions_in_more_than_one_fold"] = int((lesions > 1).sum())

    # the added pool and extras against development and HIBA
    guarded = added_pool.guarded_ids()
    added = pd.read_csv(REPO / "results/v6/added_pool/added_train.csv", low_memory=False, dtype=str)
    source_meta = {r["isic_id"]: r for rows in cache.values() for r in rows if "isic_id" in r}
    candidates = pd.read_csv(REPO / "results/v6/added_pool/candidates.csv", low_memory=False, dtype=str).set_index("isic_id")
    ids = added["image_id"].astype(str)
    kept = candidates.reindex(ids)
    added_patients, added_lesions = clean(kept["patient_id"]), clean(kept["lesion_id"])
    has = ~kept["patient_id"].fillna("").str.strip().str.lower().isin(NULL)
    report["added_rows"] = len(added)
    report["added_rows_with_patient_id"] = int(has.sum())
    report["added_patient_id_share_by_archive"] = {str(k): float(v) for k, v in has.groupby(added["archive"].to_numpy()).mean().items()}
    dev_patients, dev_lesions = clean(dev["patient"]), clean(dev["lesion"])
    report["added_vs_dev"] = {"shared_images": len(set(ids) & set(dev["image_id"])), "shared_patients": len(added_patients & dev_patients), "shared_lesions": len(added_lesions & dev_lesions)}
    report["added_vs_guarded_manifest"] = {"shared_patients": len(added_patients & guarded["manifest_patients"]), "shared_lesions": len(added_lesions & guarded["manifest_lesions"])}
    report["added_vs_hiba"] = {"shared_images": len(set(ids) & guarded["hiba_images"]), "shared_patients": len(added_patients & guarded["hiba_patients"]), "shared_lesions": len(added_lesions & guarded["hiba_lesions"])}
    extras = pd.read_csv(data_audit.EXTRAS, low_memory=False, dtype=str)
    extra_patients = clean(extras["patient_id"]) if "patient_id" in extras else set()
    report["extras_vs_hiba"] = {"shared_images": len(set(extras["image_id"].astype(str)) & guarded["hiba_images"]), "shared_patients": len(extra_patients & guarded["hiba_patients"])}
    report["extras_vs_dev"] = {"shared_images": len(set(extras["image_id"].astype(str)) & set(dev["image_id"])), "shared_patients": len(extra_patients & dev_patients)}
    report["dev_vs_hiba"] = {"shared_images": len(set(dev["isic"].astype(str)) & guarded["hiba_images"]), "shared_patients": len(dev_patients & guarded["hiba_patients"]), "shared_lesions": len(dev_lesions & guarded["hiba_lesions"])}
    OUT.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
