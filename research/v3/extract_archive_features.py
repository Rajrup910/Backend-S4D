"""S43 support -- penultimate features for the external archives, extractor-matched to HAM.

S42's `archive` probe returned AUC 1.000 and was **uninformative**, because the only external
features this repository had were PAD-UFES-20: dermoscopy against smartphone clinical
photography, so perfect separation is a statement about imaging *modality*, not about
acquisition entanglement within a modality. The probe said so itself
(`within_modality_only: false`) and D1 was recorded as unevaluated rather than refuted.

The informative contrast is **HAM against BCN-20000** -- both dermoscopy, different hospitals --
which is exactly what Phase C pools over. This module produces it.

## The one thing that makes the comparison valid

Features are extracted with **`ml/checkpoints/convnext_tiny_best.HAM-only.pt`**, the same
checkpoint that produced every `research/selective/features/*.npz`. That is deliberate, and it
is the same constraint S42 hit when choosing a transport pair: `convnext_tiny_oof.npz` comes
from the five fold checkpoints, so mixing it with a full-train embedding reports **extractor
mismatch** as a finding. The archive probe must compare like with like, so the reference side
is `convnext_tiny_val.npz` (HAM val, same checkpoint, out-of-sample for it) and the new side is
written here under the same contract.

`extract_features` is imported from `research.selective.features` **unchanged**, so the
forward-hook definition of "penultimate" is identical to every cached feature file in the repo.
Only the checkpoint's inputs differ.

## Which rows

The external **holdout** rows only (`split != "train"` in
`ml/configs/splits/v3/external_holdout.csv`). Using training rows would make the probe's
"can you tell the archives apart" question partly a question about memorisation, and after
Phase C those rows are in some conditions' training sets. The holdout is never trained on by
any condition, so the probe stays honest whenever it is run.

    $py -m research.v3.extract_archive_features                  # bcn20000 + mskcc
    $py -m research.v3.extract_archive_features --cohorts bcn20000
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ml.preprocessing.transforms import build_eval_transform
from ml.training.common import build_model_from_checkpoint
from research.selective.features import extract_features

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_V3 = REPO_ROOT / "ml" / "data" / "manifest_v3.csv"
EXTERNAL_HOLDOUT = REPO_ROOT / "ml" / "configs" / "splits" / "v3" / "external_holdout.csv"
CHECKPOINT = REPO_ROOT / "ml" / "checkpoints" / "convnext_tiny_best.HAM-only.pt"
OUT_DIR = REPO_ROOT / "research" / "v3" / "features"
IMAGE_SIZE = 224
BATCH_SIZE = 64


class _RowDataset(Dataset):
    """Explicit manifest rows, yielding LesionDataset's (tensor, label, image_id) contract."""

    def __init__(self, frame: pd.DataFrame,
                 transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self._paths = frame["path"].tolist()
        self._labels = frame["class_index"].astype(int).tolist()
        self._image_ids = frame["image_id"].astype(str).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        with Image.open(REPO_ROOT / self._paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self._labels[index], self._image_ids[index]


def run(cohorts: list[str]) -> int:
    for path in (MANIFEST_V3, EXTERNAL_HOLDOUT):
        if not path.is_file():
            raise FileNotFoundError(f"{path.relative_to(REPO_ROOT)} missing -- "
                                    f"run `$py -m research.v3.build_multiarchive` first")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"no checkpoint at {CHECKPOINT.relative_to(REPO_ROOT)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = build_model_from_checkpoint(CHECKPOINT, device)
    transform = build_eval_transform(IMAGE_SIZE)
    print(f"Device: {device}  arch={payload['arch']}  "
          f"checkpoint={CHECKPOINT.name} (matches research/selective/features/*.npz)\n")

    manifest = pd.read_csv(MANIFEST_V3)
    manifest["image_id"] = manifest["image_id"].astype(str)
    holdout = pd.read_csv(EXTERNAL_HOLDOUT)
    holdout["image_id"] = holdout["image_id"].astype(str)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cohort in cohorts:
        ids = holdout.loc[(holdout["cohort"] == cohort) & (holdout["split"] != "train"),
                          "image_id"]
        if ids.empty:
            print(f"[{cohort}] no holdout rows -- skipped")
            continue
        frame = manifest.set_index("image_id").loc[ids].reset_index()
        loader = DataLoader(_RowDataset(frame, transform), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0,
                            pin_memory=(device.type == "cuda"))
        features, labels, image_ids = extract_features(model, loader, device)
        out = OUT_DIR / f"convnext_tiny_{cohort}_holdout.npz"
        np.savez(out, features=features, labels=labels,
                 image_ids=np.asarray(image_ids, dtype=object).astype(str))
        print(f"[{cohort}] {features.shape} -> {out.relative_to(REPO_ROOT)}")

    print("\nNext: re-run the archive probe with a within-modality cohort set "
          "(HAM val vs bcn20000 holdout), which is the contrast D1 actually needs.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="S43 -- extractor-matched features for the external archive holdouts")
    parser.add_argument("--cohorts", nargs="+", default=["bcn20000", "mskcc"],
                        choices=["bcn20000", "mskcc"])
    args = parser.parse_args(argv)
    return run(args.cohorts)


if __name__ == "__main__":
    raise SystemExit(main())
