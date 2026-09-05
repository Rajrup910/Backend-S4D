"""Session 0: extract standardized prediction matrices for every trained checkpoint.

Runs each of the 6 HAM-only checkpoints once on `val` and once on `test`, and writes
the full per-class probability + logit vectors (not just argmax) to
research/predictions/<arch>_<split>.csv. Phase 1 ensembling (soft-vote, rank-average,
Nelder-Mead, stacking, Caruana greedy) reads these matrices directly instead of
re-running inference.

No temperature scaling or calibration is applied here (temperature=1.0) -- that is
Phase 2's job, fitted on top of these raw logits. Test-split predictions are extracted
now (single read, per the master spec) but must not be looked at again until final
reporting; only val predictions may be used to fit anything in later phases.

Usage:
    python -m research.extract_predictions
    python -m research.extract_predictions --splits val          # val only
    python -m research.extract_predictions --checkpoints-glob "*.HAM-only.pt"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from ml.evaluation.evaluate import predict_split
from ml.paths import REPO_ROOT, load_class_mapping, load_training_config, resolve
from ml.preprocessing.dataset import LesionDataset
from ml.preprocessing.transforms import build_eval_transform
from ml.training.common import build_model_from_checkpoint, resolve_device


def extract_one(
    checkpoint_path: Path,
    split: str,
    out_dir: Path,
    device,
    batch_size: int,
    num_workers: int,
    config: dict,
) -> None:
    mapping = load_class_mapping()
    model, payload = build_model_from_checkpoint(checkpoint_path, device)
    arch = payload["arch"]
    image_size = payload.get("image_size", config["data"]["image_size"])

    dataset = LesionDataset(
        manifest_path=config["data"]["manifest"],
        splits_path=config["data"]["splits"],
        split=split,
        transform=build_eval_transform(image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"[{arch}] predicting on {split} split ({len(dataset)} images)...")
    output = predict_split(model, loader, device, temperature=1.0)

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
    for skin_class in mapping.classes:
        frame[f"logit_{skin_class.code}"] = output["logits"][:, skin_class.index]
    frame["arch"] = arch
    frame["split"] = split
    frame["checkpoint"] = checkpoint_path.relative_to(REPO_ROOT).as_posix()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{arch}_{split}.csv"
    frame.to_csv(out_path, index=False)
    print(f"  -> {out_path.relative_to(REPO_ROOT)} ({len(frame)} rows)")


def main(argv: list[str] | None = None) -> int:
    config = load_training_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints-dir", default="ml/checkpoints")
    parser.add_argument("--checkpoints-glob", default="*.HAM-only.pt")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    parser.add_argument("--out-dir", default="research/predictions")
    parser.add_argument("--batch-size", type=int, default=config["training"]["batch_size"])
    parser.add_argument("--num-workers", type=int, default=config["data"]["num_workers"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    checkpoints = sorted(resolve(args.checkpoints_dir).glob(args.checkpoints_glob))
    if not checkpoints:
        print(f"ERROR: no checkpoints matching {args.checkpoints_glob!r} in {args.checkpoints_dir}", file=sys.stderr)
        return 1

    out_dir = resolve(args.out_dir)
    print(f"Found {len(checkpoints)} checkpoint(s): {[c.name for c in checkpoints]}")
    print(f"Splits: {args.splits}  Device: {device}\n")

    for checkpoint_path in checkpoints:
        for split in args.splits:
            extract_one(checkpoint_path, split, out_dir, device, args.batch_size, args.num_workers, config)

    print(f"\nDone. {len(checkpoints) * len(args.splits)} prediction file(s) written to {out_dir.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
