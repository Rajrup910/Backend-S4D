"""Extract out-of-fold (OOF) prediction matrices from the 5-fold checkpoints trained in S2.

For each arch, for each fold k: load `ml/checkpoints/oof/{arch}-oof_f{k}_best.pt` and run
inference over that fold's held-out partition -- the rows `ml/configs/splits/oof/split_v1.fold{k}.csv`
labels `split="test"` (make_folds.py's convention: fold k's holdout stands in for "test" so the
existing train/val/test-only LesionDataset needs no change). Every train image is scored by
exactly one fold model, the one that never saw it during training -- that is what makes the
resulting predictions out-of-fold.

Two modes, mirroring the two existing extractors:
    --mode plain   single deterministic view (mirrors research/extract_predictions.py). Fast;
                   used for the early diagnose_shift sanity check before committing to TTA time.
    --mode tta     24-view uncertainty-weighted TTA (drives research.tta.extract_tta_predictions
                   .extract_one directly -- one TTA implementation, not a second one). This is
                   the production OOF matrix every OOF-fitted downstream script should read.

Per-fold predictions are staged under `<out-dir>/_folds/fold{k}/{arch}_train.csv`, then assembled
into a single `<out-dir>/{arch}_train.csv` covering the full train split exactly once. That file
loads with **zero loader changes** via
    load_split_matrix("train", predictions_dir="research/predictions_oof")
because `research/ensembling/data.py:_lesion_lookup()` reads `ml/configs/splits/split_v1.csv`,
which has an image_id -> lesion_id row for every train image already.

Default output directories are NEW (`research/predictions_oof(_tta)`), never the frozen
`research/predictions(_tta)` directories `results/frozen_artifacts.json` hashes -- dropping a
`{arch}_train.csv` into those would silently change the declared frozen set (session 5, Sec III-G).

Assertions on assembly, per arch (fail loudly, do not silently drop rows):
    - exactly 6,981 rows (the frozen train split's image count)
    - set(image_id) equals the train split's image_id set exactly, no duplicates
    - each row's class probabilities sum to 1 within 1e-4

Writes `results/oof_provenance.json`: per (mode, arch, fold), the checkpoint path, its SHA256,
the row count extracted, the fold's validation macro-F1 (from `ml/results/experiments.csv`, the
per-run training ledger -- NOT `research/experiments.csv`'s "val_oof" naming, which means
something different there), and epochs trained. This file is intentionally kept separate from
`frozen_artifacts.json`: it documents fold-checkpoint provenance, not a frozen result set.

Usage:
    python -m research.oof.extract_oof --mode plain
    python -m research.oof.extract_oof --mode tta
    python -m research.oof.extract_oof --mode plain --archs convnext_tiny --folds 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from ml.evaluation.evaluate import predict_split
from ml.paths import REPO_ROOT, load_class_mapping, load_training_config, resolve
from ml.preprocessing.dataset import LesionDataset
from ml.preprocessing.transforms import build_eval_transform
from ml.training.common import build_model_from_checkpoint, resolve_device
from research.ensembling.data import ARCHS
from research.tta.extract_tta_predictions import extract_one as extract_one_tta

FOLD_HOLDOUT_SPLIT = "test"  # make_folds.py's label for a fold's own held-out partition
OUT_DIRS = {"plain": "research/predictions_oof", "tta": "research/predictions_oof_tta"}


def _fold_split_path(splits_dir: Path, fold: int) -> Path:
    path = splits_dir / f"split_v1.fold{fold}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing fold split file {path}. Run research.oof.make_folds first.")
    return path


def _fold_checkpoint_path(checkpoints_dir: Path, arch: str, fold: int) -> Path:
    path = checkpoints_dir / f"{arch}-oof_f{fold}_best.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing fold checkpoint {path}. Run research.oof.train_folds for "
            f"arch={arch} fold={fold} first."
        )
    return path


def extract_one_plain(
    arch: str,
    fold: int,
    checkpoints_dir: Path,
    splits_dir: Path,
    out_dir: Path,
    device,
    config: dict,
    batch_size: int,
    num_workers: int,
) -> Path:
    """Single deterministic-view extraction, mirroring research/extract_predictions.py but
    pointed at a fold's own split file and holdout partition."""
    checkpoint_path = _fold_checkpoint_path(checkpoints_dir, arch, fold)
    splits_path = _fold_split_path(splits_dir, fold)

    mapping = load_class_mapping()
    model, payload = build_model_from_checkpoint(checkpoint_path, device)
    image_size = payload.get("image_size", config["data"]["image_size"])

    dataset = LesionDataset(
        manifest_path=config["data"]["manifest"],
        splits_path=splits_path,
        split=FOLD_HOLDOUT_SPLIT,
        transform=build_eval_transform(image_size),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"[{arch} fold{fold}] predicting on OOF holdout ({len(dataset)} images, plain)...")
    output = predict_split(model, loader, device, temperature=1.0)

    frame = _build_frame(output, mapping, arch, fold, checkpoint_path, has_logits=True)

    fold_dir = out_dir / "_folds" / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    out_path = fold_dir / f"{arch}_train.csv"
    frame.to_csv(out_path, index=False)
    print(f"  -> {out_path.relative_to(REPO_ROOT)} ({len(frame)} rows)")
    return out_path


def extract_one_tta_fold(
    arch: str,
    fold: int,
    checkpoints_dir: Path,
    splits_dir: Path,
    out_dir: Path,
    device,
    config: dict,
    scales: tuple[float, ...],
    batch_size: int,
    num_workers: int,
) -> Path:
    """Thin wrapper around the existing, already-tested TTA extractor -- one TTA
    implementation shared by the Session-2 val/test extraction and this OOF extraction."""
    splits_path = _fold_split_path(splits_dir, fold)
    fold_dir = out_dir / "_folds" / f"fold{fold}"
    out_name = f"{arch}_train.csv"

    extract_one_tta(
        arch=arch,
        split=FOLD_HOLDOUT_SPLIT,
        checkpoints_dir=checkpoints_dir,
        out_dir=fold_dir,
        device=device,
        config=config,
        scales=scales,
        batch_size=batch_size,
        num_workers=num_workers,
        limit=None,
        splits_file=splits_path,
        checkpoint_template="{arch}-oof_f{fold}_best.pt",
        fold=fold,
        out_name=out_name,
    )
    out_path = fold_dir / out_name

    # extract_one_tta writes frame["split"] = FOLD_HOLDOUT_SPLIT ("test"); relabel to "train"
    # so the published file is honest about which overall split these images belong to. The
    # "split" column is not read by load_split_matrix, but it is read by a human auditing the file.
    frame = pd.read_csv(out_path)
    frame["split"] = "train"
    frame["fold"] = fold
    frame.to_csv(out_path, index=False)
    return out_path


def _build_frame(output: dict, mapping, arch: str, fold: int, checkpoint_path: Path, has_logits: bool) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "image_id": output["image_ids"],
            "true_index": output["labels"],
            "true_code": [mapping.by_index(i).code for i in output["labels"]],
            "pred_index": output["predictions"],
            "pred_code": [mapping.by_index(i).code for i in output["predictions"]],
        }
    )
    for skin_class in mapping.classes:
        frame[f"p_{skin_class.code}"] = output["probabilities"][:, skin_class.index]
    if has_logits:
        for skin_class in mapping.classes:
            frame[f"logit_{skin_class.code}"] = output["logits"][:, skin_class.index]
    frame["arch"] = arch
    frame["split"] = "train"
    frame["fold"] = fold
    frame["checkpoint"] = checkpoint_path.relative_to(REPO_ROOT).as_posix()
    return frame


def assemble(arch: str, out_dir: Path, folds: list[int], train_ids: set[str], class_codes: tuple[str, ...]) -> Path:
    frames = []
    for fold in folds:
        path = out_dir / "_folds" / f"fold{fold}" / f"{arch}_train.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing staged OOF predictions for {arch} fold {fold}: {path}. "
                f"Extract every requested fold before assembling."
            )
        frames.append(pd.read_csv(path))

    combined = pd.concat(frames, ignore_index=True).sort_values("image_id").reset_index(drop=True)

    dupes = combined["image_id"][combined["image_id"].duplicated()]
    if not dupes.empty:
        raise AssertionError(
            f"{arch}: {dupes.nunique()} image_id(s) appear in more than one fold's OOF "
            f"predictions -- fold assignment is supposed to be a partition. e.g. {dupes.unique()[:3].tolist()}"
        )

    got_ids = set(combined["image_id"])
    if got_ids != train_ids:
        missing = train_ids - got_ids
        extra = got_ids - train_ids
        raise AssertionError(
            f"{arch}: assembled OOF predictions do not match the train split exactly. "
            f"missing={len(missing)} (e.g. {sorted(missing)[:3]}), extra={len(extra)} (e.g. {sorted(extra)[:3]})"
        )
    if len(combined) != len(train_ids):
        raise AssertionError(f"{arch}: expected {len(train_ids)} rows, got {len(combined)}")

    prob_cols = [f"p_{c}" for c in class_codes]
    row_sums = combined[prob_cols].sum(axis=1).to_numpy()
    bad = np.abs(row_sums - 1.0) > 1e-4
    if bad.any():
        raise AssertionError(
            f"{arch}: {bad.sum()} row(s) have class probabilities that do not sum to 1 "
            f"(max deviation {np.abs(row_sums - 1.0).max():.2e})"
        )

    out_path = out_dir / f"{arch}_train.csv"
    combined.to_csv(out_path, index=False)
    print(f"[{arch}] assembled {len(combined)} OOF rows -> {out_path.relative_to(REPO_ROOT)}")
    return out_path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fold_training_metadata(ledger: pd.DataFrame, arch: str, fold: int) -> dict:
    """Pull epochs/val-macro-F1 for one fold checkpoint from ml/results/experiments.csv, the
    per-run training ledger that ml/training/train.py writes automatically -- distinct from
    research/experiments.csv, the research-phase ledger train_folds.py separately logs to."""
    run_name = f"{arch}_oof_f{fold}"
    rows = ledger[ledger["run_name"] == run_name]
    if rows.empty:
        return {"epochs_run": None, "val_macro_f1": None}
    row = rows.iloc[-1]  # latest, in case of a re-run
    return {"epochs_run": int(row["epochs_run"]), "val_macro_f1": float(row["best_val_metric"])}


def update_provenance(mode: str, archs: list[str], folds: list[int], checkpoints_dir: Path, out_dir: Path) -> Path:
    provenance_path = resolve("results/oof_provenance.json")
    provenance: dict = {}
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    ledger_path = resolve("ml/results/experiments.csv")
    ledger = pd.read_csv(ledger_path) if ledger_path.is_file() else pd.DataFrame(columns=["run_name"])

    mode_entry = provenance.setdefault(mode, {})
    for arch in archs:
        arch_entry = mode_entry.setdefault(arch, {})
        for fold in folds:
            checkpoint_path = _fold_checkpoint_path(checkpoints_dir, arch, fold)
            staged = out_dir / "_folds" / f"fold{fold}" / f"{arch}_train.csv"
            n = int(pd.read_csv(staged).shape[0]) if staged.is_file() else None
            meta = _fold_training_metadata(ledger, arch, fold)
            arch_entry[str(fold)] = {
                "checkpoint": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(checkpoint_path),
                "n": n,
                "val_macro_f1": meta["val_macro_f1"],
                "epochs_run": meta["epochs_run"],
            }

    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Provenance written to {provenance_path.relative_to(REPO_ROOT)}")
    return provenance_path


def main(argv: list[str] | None = None) -> int:
    config = load_training_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["plain", "tta"])
    parser.add_argument("--archs", nargs="+", default=list(ARCHS))
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--checkpoints-dir", default="ml/checkpoints/oof")
    parser.add_argument("--splits-dir", default="ml/configs/splits/oof")
    parser.add_argument("--out-dir", default=None, help="Defaults to research/predictions_oof(_tta) by --mode.")
    parser.add_argument("--scales", nargs="+", type=float, default=[0.9, 1.0, 1.1], help="tta mode only")
    parser.add_argument("--batch-size", type=int, default=config["training"]["batch_size"])
    parser.add_argument("--tta-batch-size", type=int, default=4, help="tta mode: images/batch, each expands to 24 views")
    parser.add_argument("--num-workers", type=int, default=config["data"]["num_workers"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-assemble", action="store_true", help="stage per-fold files only, do not assemble/assert")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    checkpoints_dir = resolve(args.checkpoints_dir)
    splits_dir = resolve(args.splits_dir)
    out_dir = resolve(args.out_dir) if args.out_dir else resolve(OUT_DIRS[args.mode])

    mapping = load_class_mapping()
    train_split = pd.read_csv(resolve(config["data"]["splits"]))
    train_ids = set(train_split.loc[train_split["split"] == "train", "image_id"])

    print(f"Mode: {args.mode}  Archs: {args.archs}  Folds: {args.folds}  Device: {device}")
    print(f"Checkpoints: {checkpoints_dir.relative_to(REPO_ROOT)}  Out: {out_dir.relative_to(REPO_ROOT)}\n")

    for arch in args.archs:
        for fold in args.folds:
            if args.mode == "plain":
                extract_one_plain(
                    arch=arch, fold=fold, checkpoints_dir=checkpoints_dir, splits_dir=splits_dir,
                    out_dir=out_dir, device=device, config=config,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                )
            else:
                extract_one_tta_fold(
                    arch=arch, fold=fold, checkpoints_dir=checkpoints_dir, splits_dir=splits_dir,
                    out_dir=out_dir, device=device, config=config, scales=tuple(args.scales),
                    batch_size=args.tta_batch_size, num_workers=args.num_workers,
                )

        if not args.skip_assemble:
            assemble(arch, out_dir, args.folds, train_ids, mapping.codes)

    update_provenance(args.mode, args.archs, args.folds, checkpoints_dir, out_dir)

    print(f"\nDone. {len(args.archs)} arch(es) x {len(args.folds)} fold(s) extracted in {args.mode} mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
