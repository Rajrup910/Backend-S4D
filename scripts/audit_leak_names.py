"""Leak audit by normalised names: one ISIC image under two names, such as ISIC_0012503_downsampled in the manifest and ISIC_0012503 in a training source. Metadata and banked predictions only.

    python scripts/audit_leak_names.py

Exact string checks cannot see these. Counts each development row whose base name is also a young extra or an added row, by fold, archive, band and class, then re-reads what those rows do to
reads already made. Writes results/v6/screens/leak_audit_names.json. No test or reserved row, and no V6 prediction on folds 1-4.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import testguard  # noqa: E402
from research.v5.train_v5 import fold_frames  # noqa: E402
from research.v6 import gate, stage_rules as st, system_reads as sr  # noqa: E402
from research.v6.train_v6 import MANIFEST, YOUNG_EXTRAS  # noqa: E402

OUT = REPO / "results/v6/screens/leak_audit_names.json"
ESC = {"akiec", "bcc", "mel"}


def base(name: str) -> str:
    return re.sub(r"_(downsampled|resized|cropped)$", "", str(name))


def table(hits: pd.DataFrame) -> dict:
    return {"rows": len(hits), "by_fold": hits["fold"].value_counts().sort_index().to_dict(), "by_archive": hits["archive"].value_counts().to_dict(), "by_band": hits["age_band"].value_counts().to_dict(),
            "by_class": hits["class_7"].value_counts().to_dict(), "same_class": float((hits["class_7"] == hits["class_other"]).mean()) if len(hits) else None,
            "under40_escalating_rows": int(((hits["age_band"] == "<40") & hits["class_7"].isin(ESC)).sum()), "under40_escalating_groups": int(hits.loc[(hits["age_band"] == "<40") & hits["class_7"].isin(ESC), "group_id"].nunique()),
            "same_lesion_id": int((hits["effective_lesion_id"].astype(str) == hits["lesion_other"].astype(str)).sum()), "image_ids": hits["image_id"].tolist()}


def effect(named: set[str]) -> dict:
    """V5's Gate A compares a system that trained on the extras with a control that did not, so the copies favour V5 there. V6 against V5 trains on them on both sides. Point estimates, on the rows each read used."""
    rows, models = sr.load_matrix((sr.V5_SYSTEM, sr.V5_NO_TTA, sr.V5_CONTROL))
    drop = np.isin(rows.image_id, list(named))

    def read(a: str, b: str, where: np.ndarray) -> dict:
        young = where & rows.u40
        out = {"rows": int(where.sum()), "u40_rows": int(young.sum()), "u40_escalating_rows": int((young & rows.esc).sum()), "u40_escalating_groups": len(set(rows.group[young & rows.esc]))}
        for name, mask in (("pauc_u40", young), ("pauc_all", where)):
            out[name] = sr._pauc(rows.esc[mask], models[a][mask]) - sr._pauc(rows.esc[mask], models[b][mask])
        out["macro_f1"] = sr._mf1(rows.y[where], models[a][where]) - sr._mf1(rows.y[where], models[b][where])
        return out

    folds_1_4 = rows.fold != 0
    v5 = {"as_read": read(sr.V5_NO_TTA, sr.V5_CONTROL, folds_1_4), "without_the_copies": read(sr.V5_NO_TTA, sr.V5_CONTROL, folds_1_4 & ~drop), "copies_in_folds_1_4": int((folds_1_4 & drop).sum()),
          "copies_escalating_in_folds_1_4": int((folds_1_4 & drop & rows.esc).sum())}
    members = {n: st.fold0_read(gate.pred_path(n, 42, t, sr.V6_TTA_DIR, checkpoint="swad_tta")) for n, t in {"anchor_h7": "v62y", "panderm_b_h7": "v63", "dinov2_b_h7": "v63", "anchor384_h7": "v63"}.items()}
    first = members["anchor_h7"]
    position = {image: i for i, image in enumerate(rows.image_id)}
    where = np.array([position[i] for i in first["image_id"]])
    v6, v5_p = np.mean([m["p"] for m in members.values()], axis=0), models[sr.V5_SYSTEM][where]
    fold0, keep = {}, ~np.isin(first["image_id"], list(named))
    for name, mask in (("as_read", np.ones(len(keep), bool)), ("without_the_copies", keep)):
        young = mask & first["young"]
        fold0[name] = {"rows": int(mask.sum()), "u40_escalating_rows": int((young & first["esc"]).sum()), "macro_f1": sr._mf1(first["y"][mask], v6[mask]) - sr._mf1(first["y"][mask], v5_p[mask]),
                       "pauc_all": sr._pauc(first["esc"][mask], v6[mask]) - sr._pauc(first["esc"][mask], v5_p[mask]), "pauc_u40": sr._pauc(first["esc"][young], v6[young]) - sr._pauc(first["esc"][young], v5_p[young])}
    return {"v5_composite_minus_control_folds_1_4_no_tta": v5, "v6_vote_minus_v5_fold_0_tta": fold0}


def main() -> int:
    testguard.block_test_reads("leak audit by names: metadata and banked predictions only")
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    parts = []
    for fold in range(5):
        train, val = fold_frames(manifest, fold)
        parts.append(val.assign(fold=fold))
    dev = pd.concat(parts, ignore_index=True)
    dev["image_id"] = dev["image_id"].astype(str)
    dev["base"] = dev["image_id"].map(base)
    extras = pd.read_csv(YOUNG_EXTRAS, low_memory=False, dtype={"image_id": str})
    added = pd.read_csv(REPO / "results/v6/added_pool/added_train.csv", low_memory=False, dtype={"image_id": str})
    report: dict = {"dev_rows": len(dev), "dev_names_changed_by_normalising": int((dev["image_id"] != dev["base"]).sum()), "dev_rows_by_suffix_and_archive": dev.loc[dev["image_id"] != dev["base"], "archive"].value_counts().to_dict(),
                    "duplicate_base_names_inside_dev": int(dev["base"].duplicated().sum())}
    named: set[str] = set()
    for name, source in (("young_extras", extras), ("added_pool", added)):
        other = source.assign(base=source["image_id"].map(base))[["base", "class_7", "effective_lesion_id", "image_id", "group_id"]].rename(columns={"class_7": "class_other", "effective_lesion_id": "lesion_other", "image_id": "id_other", "group_id": "group_other"})
        hits = dev.merge(other, on="base")
        report[name] = {"source_rows": len(source), **table(hits)}
        named |= set(hits["image_id"])
    report["effect"] = effect(named)
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    shown = {k: v for k, v in report.items() if k != "young_extras"}
    print(json.dumps({**shown, "young_extras": {k: v for k, v in report["young_extras"].items() if k != "image_ids"}}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
