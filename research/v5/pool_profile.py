"""Training-pool profile: where the under-40 escalating evidence lives, and how hard each archive is.

    python -m research.v5.pool_profile

Input to the V6 pool-design discussion (V6_RUNSHEET §A10.4). Development rows only
(`manifest_v4.csv` split == "train", i.e. folds 0-4) plus the young-data extra rows; the LOAO
readout uses the stored V5 control LOAO predictions (held-out archive inside the development
partition). No test / reserved row is read. Writes `results/v5/pool_profile.json`.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from research import testguard
from research.v4.recipe import REPO_ROOT
from research.v5.screen_gate import pauc
from research.v5.train_v5 import MANIFEST

EXTRA = REPO_ROOT / "results" / "v5" / "young_data" / "extra_train.csv"
YOUNG_COUNT = REPO_ROOT / "results" / "v5" / "young_data" / "count.json"
PRED_DIR = REPO_ROOT / "results" / "v5" / "preds"
OUT = REPO_ROOT / "results" / "v5" / "pool_profile.json"
ARCHIVES = ("ham", "bcn20000", "mskcc")


def cells(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for (archive, band), sub in frame.groupby(["archive", "age_band"]):
        esc = sub[sub["escalating_7"] == 1]
        rows.append({"archive": archive, "age_band": band, "images": int(len(sub)),
                     "lesions": int(sub["group_id"].nunique()), "esc_images": int(len(esc)),
                     "esc_lesions": int(esc["group_id"].nunique()),
                     "mel_lesions": int(sub.loc[sub["class_7"] == "mel", "group_id"].nunique()),
                     "esc_prevalence": round(float(len(esc) / len(sub)), 4)})
    return rows


def loao() -> list[dict]:
    rows = []
    for archive in ARCHIVES:
        f = pd.read_csv(PRED_DIR / f"control_loao-{archive}_s42_in22k_v5scr.csv", low_memory=False)
        y, p = f["y_true"].to_numpy(), f["pred_index"].to_numpy()
        present = sorted(np.unique(y))
        esc = f["y_esc"].astype(bool).to_numpy()
        u40 = (f["age_band"] == "<40").to_numpy()
        s = f["declared_score"].to_numpy(dtype=float)
        rows.append({"held_out": archive, "n": int(len(f)), "classes_present": len(present),
                     "macro_f1_present": float(f1_score(y, p, labels=present, average="macro",
                                                        zero_division=0)),
                     "balanced_accuracy": float(balanced_accuracy_score(y, p)),
                     "pauc_all": pauc(esc, s), "pauc_u40": pauc(esc[u40], s[u40]),
                     "n_esc_u40_images": int((esc & u40).sum())})
    return rows


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    testguard.block_test_reads("pool profile: development rows and young-data extra rows only")
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    dev = manifest[manifest["split"] == "train"]
    extra = pd.read_csv(EXTRA, low_memory=False).assign(archive="isic_young_extra")
    young_esc = dev[(dev["age_band"] == "<40") & (dev["escalating_7"] == 1)]
    count = json.loads(YOUNG_COUNT.read_text(encoding="utf-8"))
    out = {
        "source": "research/v5/pool_profile.py",
        "dev_rows": int(len(dev)),
        "cells": cells(pd.concat([dev, extra], ignore_index=True)),
        "u40_escalating_lesions_dev_by_archive": {
            a: int(young_esc.loc[young_esc["archive"] == a, "group_id"].nunique()) for a in ARCHIVES},
        "u40_escalating_lesions_dev_total": int(young_esc["group_id"].nunique()),
        "age60_escalating_lesions_dev_total": int(
            dev[(dev["age_band"] == "60+") & (dev["escalating_7"] == 1)]["group_id"].nunique()),
        "young_extra": {"images": int(len(extra)), "lesions": int(extra["group_id"].nunique()),
                        "esc_lesions": int(extra.loc[extra["escalating_7"] == 1, "group_id"].nunique()),
                        "by_class": extra["class_7"].value_counts().to_dict()},
        "young_archive_pool": {k: count[k] for k in ("query", "pool_total", "excluded_in_v4_corpus",
                                                      "candidates", "candidates_without_lesion_id")},
        "loao_control_in22k_s42": loao(),
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}: u40 escalating lesions {out['u40_escalating_lesions_dev_by_archive']} "
          f"(total {out['u40_escalating_lesions_dev_total']}) vs 60+ {out['age60_escalating_lesions_dev_total']}; "
          f"young extra +{out['young_extra']['esc_lesions']} escalating lesions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
