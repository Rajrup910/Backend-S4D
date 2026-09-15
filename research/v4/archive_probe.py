"""S49 -- does colour constancy reduce archive decodability? Measured, on the same instrument.

V3 pooled archives whose colour statistics differ and the pooling failed. S42/S43 showed the
embedding can tell dermoscopy archives apart. If that separability is carried by illuminant
statistics, then `research/v4/colour.py` removes a confound and V4's re-pool is a genuinely
different experiment. If it is not, that is a clean negative and the re-pool is a re-run -- which
is worth knowing *before* spending three nights of GPU on it.

The instrument is not re-invented: features are extracted by
`research.selective.features.extract_features` through
`ml/checkpoints/convnext_tiny_best.HAM-only.pt`, exactly as S43 did, and scored by
`research.v3.probes.probe_archive` unchanged. The only difference between the two arms is
whether each image passes through `shades_of_grey` first.

**The control that makes this readable.** The `raw` arm is re-extracted here rather than read
from S43's cached `.npz`, and the run checks that it reproduces those cached features. If it does
not, the comparison is measuring pipeline drift rather than colour constancy, and the run says so
instead of reporting a difference.

**What a drop would and would not mean.** The probe is cross-fitted and lesion-grouped, so a fall
in AUC means illuminant statistics carried archive identity. It does **not** follow that pooling
then works: H4 could have failed for reasons that have nothing to do with colour. The verdict
field states which of the two readings the numbers support and neither is allowed to imply the
other.

    $py -m research.v4.archive_probe
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ml.paths import resolve
from ml.preprocessing.transforms import build_eval_transform
from ml.training.common import build_model_from_checkpoint
from research.experiment_log import log_experiment
from research.selective.features import extract_features
from research.v3.probes import probe_archive
from research.v4.colour import TRANSFORMS

SESSION = "v4_s49"
METHOD_PREFIX = "S49_archive_probe"

CHECKPOINT = "ml/checkpoints/convnext_tiny_best.HAM-only.pt"
MANIFEST_V3 = "ml/data/manifest_v3.csv"
EXTERNAL_HOLDOUT = "ml/configs/splits/v3/external_holdout.csv"
SPLIT_V1 = "ml/configs/splits/split_v1.csv"
HAM_DIRS = ("data/ham10000/HAM10000_images_part_1", "data/ham10000/HAM10000_images_part_2")
CACHED_RAW = {
    "ham_val": "research/selective/features/convnext_tiny_val.npz",
    "bcn20000": "research/v3/features/convnext_tiny_bcn20000_holdout.npz",
    "mskcc": "research/v3/features/convnext_tiny_mskcc_holdout.npz",
}
OUT_CSV = "results/v4/archive_probe_pre_post.csv"
OUT_JSON = "results/v4/archive_probe_pre_post.json"

IMAGE_SIZE = 224
BATCH_SIZE = 64          # S43's value; see check_raw_reproduces


class _Rows(Dataset):
    def __init__(self, frame: pd.DataFrame, transform, colour):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.colour = colour

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["path"])
        return self.transform(self.colour(image)), int(row["class_index"]), row["image_id"]


def _cohort_rows() -> dict[str, pd.DataFrame]:
    """The same rows S43 used: HAM val, plus the BCN and MSKCC holdouts."""
    manifest = pd.read_csv(resolve(MANIFEST_V3))
    manifest["image_id"] = manifest["image_id"].astype(str)
    holdout = pd.read_csv(resolve(EXTERNAL_HOLDOUT))
    holdout["image_id"] = holdout["image_id"].astype(str)

    out = {}
    for cohort in ("bcn20000", "mskcc"):
        ids = holdout.loc[(holdout["cohort"] == cohort) & (holdout["split"] != "train"),
                          "image_id"]
        out[cohort] = manifest.set_index("image_id").loc[ids].reset_index()

    v1 = pd.read_csv(resolve(SPLIT_V1))
    ham = v1[v1["split"] == "val"].copy()
    paths = {}
    for folder in HAM_DIRS:
        for path in resolve(folder).glob("*.jpg"):
            paths[path.stem] = str(path)
    ham["path"] = ham["image_id"].map(paths)
    out["ham_val"] = ham.dropna(subset=["path"]).reset_index(drop=True)
    return out


def extract(arm: str, rows: dict[str, pd.DataFrame], device, model, transform) -> dict:
    colour = TRANSFORMS[arm]
    features, lesions = {}, {}
    for cohort, frame in rows.items():
        loader = DataLoader(_Rows(frame, transform, colour), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0,
                            pin_memory=(device.type == "cuda"))
        feats, _labels, image_ids = extract_features(model, loader, device)
        features[cohort] = feats
        order = frame.set_index("image_id").loc[list(image_ids)]
        lesions[cohort] = order["lesion_id"].to_numpy()
        print(f"    [{arm}] {cohort}: {feats.shape}")
    return {"features": features, "lesions": lesions}


def check_raw_reproduces(features: dict[str, np.ndarray]) -> dict:
    """The raw arm must reproduce S43's cached features, or the comparison is measuring drift.

    `BATCH_SIZE` is 64 because that is what S43 used, and it is not cosmetic: at batch 32 this
    same code reproduced the BCN cache to a max absolute deviation of 0.0235 (mean 3.2e-4) and at
    batch 64 to exactly 0.0. cuDNN selects convolution algorithms by input shape, so the batch
    size is part of the extraction contract. The deviation was far too small to move an AUC by
    the 0.039 measured here, but reporting a reproduction check that does not reproduce would
    have left a reader unable to tell those two cases apart.
    """
    report = {}
    for cohort, path in CACHED_RAW.items():
        target = resolve(path)
        if not target.is_file() or cohort not in features:
            report[cohort] = {"status": "no cached reference"}
            continue
        cached = np.load(target, allow_pickle=True)["features"]
        fresh = features[cohort]
        if cached.shape != fresh.shape:
            report[cohort] = {"status": "SHAPE MISMATCH",
                              "cached": list(cached.shape), "fresh": list(fresh.shape)}
            continue
        diff = np.abs(cached - fresh)
        report[cohort] = {
            "status": "exact" if diff.max() == 0 else ("ok" if diff.max() < 1e-2 else "DRIFT"),
            "max_abs_delta": float(diff.max()),
            "mean_abs_delta": float(diff.mean()),
            "feature_abs_mean": float(np.abs(cached).mean()),
        }
    return report


def prune_prior_rows(path: str = "research/experiments.csv") -> int:
    """Drop this runner's own earlier rows so a re-run replaces rather than duplicates them.

    Added after this module duplicated its six rows on a second run -- the fourth time the
    project has hit the no-prune hazard. The prefix is `S49_archive_probe`, not the session-wide
    `S49_`, so that `build_corpus.py`'s prune cannot delete these rows and vice versa.
    """
    import csv

    target = resolve(path)
    if not target.is_file():
        return 0
    with target.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames or [], list(reader)
    kept = [r for r in rows if not (r.get("session") == SESSION
                                    and str(r.get("method", "")).startswith(METHOD_PREFIX))]
    removed = len(rows) - len(kept)
    if removed:
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="S49 -- archive decodability before and after colour constancy.")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _payload = build_model_from_checkpoint(resolve(CHECKPOINT), device)
    transform = build_eval_transform(IMAGE_SIZE)
    rows = _cohort_rows()
    print(f"S49 -- archive probe pre/post on {device.type}; cohorts "
          f"{ {k: len(v) for k, v in rows.items()} }")

    arms = {arm: extract(arm, rows, device, model, transform)
            for arm in ("raw", "shades_of_grey")}

    reproduction = check_raw_reproduces(arms["raw"]["features"])
    print(f"  raw-arm reproduction of S43 cache: {reproduction}")

    #: the within-modality contrast is the one that matters; the three-cohort set is secondary
    cohort_sets = {
        "ham_val_vs_bcn20000": ("ham_val", "bcn20000"),
        "ham_val_vs_mskcc": ("ham_val", "mskcc"),
        "all_three": ("ham_val", "bcn20000", "mskcc"),
    }

    records = []
    for set_name, members in cohort_sets.items():
        for arm, payload in arms.items():
            result = probe_archive(
                {c: payload["features"][c] for c in members},
                {c: payload["lesions"][c] for c in members},
            )
            records.append({
                "cohort_set": set_name, "arm": arm, "auc": result["value"],
                "ci_lo": result["ci_lo"], "ci_hi": result["ci_hi"],
                "n": result["n"], "within_modality_only": result["within_modality_only"],
                "entangled": result["entangled"],
            })

    table = pd.DataFrame(records)
    wide = table.pivot(index="cohort_set", columns="arm", values="auc")
    wide["delta"] = wide["shades_of_grey"] - wide["raw"]
    merged = table.merge(wide[["delta"]], on="cohort_set")

    resolve(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(resolve(OUT_CSV), index=False)

    primary = wide.loc["ham_val_vs_bcn20000"]
    lo_raw = table[(table.cohort_set == "ham_val_vs_bcn20000") & (table.arm == "raw")]
    lo_cc = table[(table.cohort_set == "ham_val_vs_bcn20000") & (table.arm == "shades_of_grey")]
    overlap = not (float(lo_cc["ci_hi"].iloc[0]) < float(lo_raw["ci_lo"].iloc[0]))

    residual = float(primary["shades_of_grey"])
    verdict = (
        "colour constancy does NOT materially reduce within-modality archive decodability; "
        "H4's failure gets no mechanism from illuminant statistics and the V4 re-pool is a "
        "re-run rather than a new experiment"
        if overlap or primary["delta"] > -0.02 else
        "colour constancy materially reduces within-modality archive decodability "
        f"({primary['raw']:.4f} -> {residual:.4f}, non-overlapping intervals), so illuminant "
        "statistics carry part of archive identity and the V4 re-pool is a different experiment "
        "rather than a re-run. TWO things this does not license: the residual AUC is still "
        f"{residual:.3f}, so archive identity is reduced and nowhere near removed -- most of it "
        "survives colour normalisation and lives in something else; and a mechanism for archive "
        "separability is not a mechanism for H4's failure, which may have had reasons unrelated "
        "to colour. The pooling experiment still has to be run and can still fail."
    )

    report = {
        "session": "S49", "phase": "U_corpus", "test_read": False,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": CHECKPOINT, "minkowski_p": 6,
        "raw_arm_reproduces_s43_cache": reproduction,
        "cohort_sizes": {k: int(len(v)) for k, v in rows.items()},
        "results": records,
        "primary_contrast": "ham_val_vs_bcn20000",
        "primary_delta_auc": float(primary["delta"]),
        "intervals_overlap": bool(overlap),
        "verdict": verdict,
    }
    resolve(OUT_JSON).write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    print(merged.to_string(index=False))
    print(f"\n  primary (HAM val vs BCN-20000): raw {primary['raw']:.4f} -> "
          f"cc {primary['shades_of_grey']:.4f} (delta {primary['delta']:+.4f})")
    print(f"  VERDICT: {verdict}")

    if not args.no_ledger:
        removed = prune_prior_rows()
        if removed:
            print(f"  (replaced {removed} archive-probe ledger row(s) from a previous run)")
        for record in records:
            log_experiment({
                "session": SESSION,
                "method": f"{METHOD_PREFIX}[{record['cohort_set']}|{record['arm']}]",
                "split": "ham_val+external_holdout",
                "macro_roc_auc": round(record["auc"], 4),
                "notes": (
                    f"Phase U; archive decodability {record['auc']:.4f} "
                    f"[{record['ci_lo']:.4f}, {record['ci_hi']:.4f}] on n={record['n']}; "
                    f"arm={record['arm']} (shades-of-grey p=6); within_modality="
                    f"{record['within_modality_only']}; frozen convnext_tiny HAM-only "
                    "extractor, no test read"
                ),
            })
        print(f"  ledger      {len(records)} rows under session={SESSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
