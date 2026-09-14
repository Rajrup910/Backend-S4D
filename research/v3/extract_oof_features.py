"""S40 / Phase A1 Stage 1 -- the honest OOF feature panel.

`research/selective/features/convnext_tiny_train.npz` is in-sample: it was extracted by
`research/selective/features.py` with `ml/checkpoints/convnext_tiny_best.HAM-only.pt`, the
model fitted on all 6,981 train images, run over those same images. Every probe built on it
inherits that leak (see `oos_probe.py`).

This module builds the replacement. For each fold `f in 0..4` it loads
`ml/checkpoints/oof/convnext_tiny-oof_f{f}_best.pt` and extracts features for exactly the
rows that fold held out, so no row is ever scored by a model that trained on it. The result
is aligned to `results/v2/panels/ham_oof.csv` by `image_id` -- the same 6,981 rows, now
cross-fitted on both sides.

**Why this does not route through `features.py`'s CLI.** `extract_one` builds a
`LesionDataset(split=...)`, and the OOF fold files mark each fold's held-out rows as
`split="test"` (verified: `split_v1.fold0.csv` has 5,584 train / 1,532 val / 1,397 test).
Asking `LesionDataset` for those rows means asking for `split="test"`, which trips
`testguard` and -- worse -- reads as a test-split access in every log and audit, when the
rows are actually held-out *training* folds. Extraction is therefore driven from the `fold`
column of `ml/configs/splits/split_v1.folds.csv`, and `_FoldDataset` below selects rows by
explicit `image_id`. The real HAM test split is never touched.

`extract_features` itself is imported from `research.selective.features` **unchanged**, so
the forward-hook definition of "penultimate" is identical to the one that produced every
cached feature file in the repository. Only the checkpoint and the row selection differ.

    $py -m research.v3.extract_oof_features
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

from ml.paths import REPO_ROOT, load_training_config, resolve
from ml.preprocessing.transforms import build_eval_transform
from ml.training.common import build_model_from_checkpoint, resolve_device
from research.selective.features import extract_features

FOLDS_PATH = REPO_ROOT / "ml" / "configs" / "splits" / "split_v1.folds.csv"
FOLD_SPLIT_TEMPLATE = REPO_ROOT / "ml" / "configs" / "splits" / "oof" / "split_v1.fold{fold}.csv"
CHECKPOINT_TEMPLATE = "ml/checkpoints/oof/{arch}-oof_f{fold}_best.pt"
PANEL_PATH = REPO_ROOT / "results" / "v2" / "panels" / "ham_oof.csv"
OUT_DIR = REPO_ROOT / "research" / "v3" / "features"
N_FOLDS = 5


class _FoldDataset(Dataset):
    """Manifest rows for an explicit `image_id` list, yielding LesionDataset's contract.

    Deliberately not `LesionDataset`: that class selects by the `split` column, and the rows
    wanted here are one cross-validation fold, which no split column names without calling
    held-out training data "test".
    """

    def __init__(self, manifest: pd.DataFrame, image_ids: list[str],
                 transform: Callable[[Image.Image], torch.Tensor]) -> None:
        frame = manifest.set_index("image_id").loc[image_ids].reset_index()
        self._paths = frame["path"].tolist()
        self._labels = frame["class_index"].astype(int).tolist()
        self._image_ids = frame["image_id"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path = REPO_ROOT / self._paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), self._labels[index], self._image_ids[index]


def _assert_no_fold_leak(fold: int, held_ids: set[str]) -> int:
    """Set intersection against the fold model's own training rows -- checked, not inferred.

    The runbook's gate for this stage: *every row from a model that did not train on it*.
    """
    split_path = Path(str(FOLD_SPLIT_TEMPLATE).format(fold=fold))
    frame = pd.read_csv(split_path)
    trained = set(frame.loc[frame["split"] == "train", "image_id"].astype(str))
    declared_held = set(frame.loc[frame["split"] == "test", "image_id"].astype(str))

    overlap = held_ids & trained
    if overlap:
        raise AssertionError(
            f"fold {fold}: {len(overlap)} rows would be scored by a model that trained on them "
            f"(e.g. {sorted(overlap)[:3]})"
        )
    if held_ids != declared_held:
        raise AssertionError(
            f"fold {fold}: the `fold` column and {split_path.name}'s held-out rows disagree "
            f"({len(held_ids ^ declared_held)} rows differ)"
        )
    return len(trained)


def run(arch: str, batch_size: int, workers: int, device_arg: str) -> int:
    config = load_training_config()
    device = resolve_device(device_arg)
    manifest = pd.read_csv(resolve(config["data"]["manifest"]))
    folds = pd.read_csv(FOLDS_PATH)
    panel_ids = pd.read_csv(PANEL_PATH, usecols=["image_id"])["image_id"].astype(str).tolist()

    print(f"Device: {device}  arch={arch}  folds={N_FOLDS}")
    print(f"{len(folds)} OOF rows across {folds['fold'].nunique()} folds\n")

    all_features, all_ids = [], []
    for fold in range(N_FOLDS):
        held = folds.loc[folds["fold"] == fold, "image_id"].astype(str).tolist()
        n_trained = _assert_no_fold_leak(fold, set(held))

        checkpoint = resolve(CHECKPOINT_TEMPLATE.format(arch=arch, fold=fold))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"no fold checkpoint at {checkpoint}")
        model, payload = build_model_from_checkpoint(checkpoint, device)
        image_size = payload.get("image_size", config["data"]["image_size"])

        dataset = _FoldDataset(manifest, held, build_eval_transform(image_size))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=device.type == "cuda")

        print(f"[fold {fold}] {len(dataset)} held-out rows, model trained on {n_trained} others")
        features, _labels, image_ids = extract_features(model, loader, device)
        all_features.append(features)
        all_ids.extend(image_ids)
        print(f"  -> {features.shape[0]} x {features.shape[1]}")

    features = np.concatenate(all_features, axis=0)
    ids = np.asarray(all_ids, dtype=str)

    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate image_id across folds -- the folds are not a partition")
    order = pd.Index(ids).get_indexer(panel_ids)
    if (order < 0).any():
        raise AssertionError(f"{int((order < 0).sum())} panel rows have no extracted feature")
    features, ids = features[order], ids[order]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{arch}_oof.npz"
    np.savez_compressed(out_path, features=features.astype(np.float32),
                        image_ids=ids, labels=np.zeros(len(ids), dtype=np.int64))
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}  {features.shape[0]} x {features.shape[1]}")
    print("aligned to results/v2/panels/ham_oof.csv by image_id; no row scored by its own model")
    return 0


def main(argv: list[str] | None = None) -> int:
    config = load_training_config()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default="convnext_tiny")
    parser.add_argument("--batch-size", type=int, default=config["training"]["batch_size"])
    parser.add_argument("--num-workers", type=int, default=config["data"]["num_workers"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    return run(args.arch, args.batch_size, args.num_workers, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
