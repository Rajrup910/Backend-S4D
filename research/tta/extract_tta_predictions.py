"""Extract TTA-pooled prediction matrices, mirroring research/extract_predictions.py.

Writes research/predictions_tta/<arch>_<split>.csv in the same column layout as the
Session-0 files (image_id, true/pred index+code, p_<code>, logit_<code>, arch, split,
checkpoint) so `research/ensembling/data.py` can load either set interchangeably. The
"logit" columns here are log(pooled probability), not raw model logits -- TTA pooling
happens in probability space (entropy-weighted average across 24 views), so there is no
single pre-softmax vector per image to report; the column is kept for schema parity with
non-TTA predictions, not for a second calibration pass.

Usage:
    python -m research.tta.extract_tta_predictions
    python -m research.tta.extract_tta_predictions --limit 32       # smoke test
    python -m research.tta.extract_tta_predictions --archs convnext_tiny --splits val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Subset

from ml.paths import REPO_ROOT, load_class_mapping, load_training_config, resolve
from ml.preprocessing.dataset import LesionDataset
from ml.training.common import build_model_from_checkpoint, resolve_device
from research.ensembling.data import ARCHS
from research.tta.predict import predict_tta


def extract_one(
    arch: str,
    split: str,
    checkpoints_dir: Path,
    out_dir: Path,
    device,
    config: dict,
    scales: tuple[float, ...],
    batch_size: int,
    num_workers: int,
    limit: int | None,
    splits_file: str | Path | None = None,
    checkpoint_template: str | None = None,
    fold: int | None = None,
    out_name: str | None = None,
) -> None:
    if checkpoint_template:
        checkpoint_path = checkpoints_dir / checkpoint_template.format(arch=arch, fold=fold)
    else:
        checkpoint_path = checkpoints_dir / f"{arch}_best.HAM-only.pt"
        if not checkpoint_path.is_file():
            checkpoint_path = checkpoints_dir / f"{arch}_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"no checkpoint at {checkpoint_path}"
        )

    mapping = load_class_mapping()
    model, payload = build_model_from_checkpoint(checkpoint_path, device)
    image_size = payload.get("image_size", config["data"]["image_size"])

    splits_path = splits_file if splits_file is not None else config["data"]["splits"]
    dataset = LesionDataset(
        manifest_path=config["data"]["manifest"],
        splits_path=splits_path,
        split=split,
        transform=None,
    )
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))

    output = predict_tta(
        model=model,
        base_dataset=dataset,
        device=device,
        image_size=image_size,
        scales=scales,
        batch_size=batch_size,
        num_workers=num_workers,
        description=f"{arch:<16} [{split:<4}]",
    )

    frame = pd.DataFrame()
    frame["image_id"] = output["image_ids"]
    frame["true_index"] = output["labels"]
    frame["true_code"] = [mapping.codes[i] for i in output["labels"]]
    frame["pred_index"] = output["probabilities"].argmax(axis=1)
    frame["pred_code"] = [mapping.codes[i] for i in frame["pred_index"]]

    eps = 1e-12
    for skin_class in mapping.classes:
        frame[f"p_{skin_class.code}"] = output["probabilities"][:, skin_class.index]
        frame[f"logit_{skin_class.code}"] = np.log(
            np.clip(output["probabilities"][:, skin_class.index], eps, None)
        )
    frame["arch"] = arch
    frame["split"] = split
    frame["checkpoint"] = checkpoint_path.relative_to(REPO_ROOT).as_posix()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (out_name if out_name else f"{arch}_{split}.csv")
    frame.to_csv(out_path, index=False)
    print(f"  -> {out_path.relative_to(REPO_ROOT)} ({len(frame)} rows)")


def main(argv: list[str] | None = None) -> int:
    config = load_training_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archs", nargs="+", default=list(ARCHS))
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    parser.add_argument("--checkpoints-dir", default="ml/checkpoints")
    parser.add_argument("--out-dir", default="research/predictions_tta")
    parser.add_argument("--out-name", default=None, help="Explicit output filename override.")
    parser.add_argument("--splits-file", default=None, help="Explicit split CSV override.")
    parser.add_argument("--checkpoint-template", default=None, help="Template e.g. '{arch}-oof_f{fold}_best.pt'.")
    parser.add_argument("--fold", type=int, default=None, help="Fold index when using --checkpoint-template.")
    parser.add_argument("--scales", nargs="+", type=float, default=[0.9, 1.0, 1.1])
    parser.add_argument("--batch-size", type=int, default=4, help="images per batch (each expands to 24 views)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="cap images per split, for smoke testing")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    checkpoints_dir = resolve(args.checkpoints_dir)
    out_dir = resolve(args.out_dir)
    scales = tuple(args.scales)

    print(f"Archs: {args.archs}  Splits: {args.splits}  Scales: {scales}  Device: {device}\n")
    for arch in args.archs:
        for split in args.splits:
            extract_one(
                arch=arch,
                split=split,
                checkpoints_dir=checkpoints_dir,
                out_dir=out_dir,
                device=device,
                config=config,
                scales=scales,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                limit=args.limit,
                splits_file=args.splits_file,
                checkpoint_template=args.checkpoint_template,
                fold=args.fold,
                out_name=args.out_name,
            )

    print(f"\nDone. {len(args.archs) * len(args.splits)} TTA prediction file(s) written to "
          f"{out_dir.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
