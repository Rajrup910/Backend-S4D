"""S8b GPU pass: score the six frozen HAM-only CNNs on the *full* PAD-UFES-20 cohort.

This is the only GPU step in S8b. It produces, for all 2,106 PAD images:

  * `research/predictions_pad/<arch>.csv` -- per-class probabilities + logits for each of
    the six published HAM-only checkpoints, same schema as `research/predictions/`.
  * `research/selective/features/convnext_tiny_pad.npz` -- ConvNeXt-Tiny penultimate
    features, for the Mahalanobis-under-shift test in `run_session8b.py`.

Why the full cohort and not the 314-row PAD test split already on disk: the existing
`ml/results/eval_*_HAM-on-PAD/` predictions cover only `split_pad_only.csv`'s 314 test
rows, which holds ~200 Fitzpatrick-labelled images and 8 melanomas -- far too few to
power a Fitzpatrick fairness slice or a melanoma-recall claim. Because these are
HAM-trained models applied cross-domain, *no* PAD image was ever used to fit anything, so
scoring all 2,106 is leak-free by construction (there is no PAD training split to respect
here -- Hard Rule 1's grouped-split discipline governs the HAM pipeline, not this external
read).

This writes to NEW directories only. It never touches `research/predictions{,_tta}/` or
the frozen `convnext_tiny_{train,val,test}.npz` feature cache, so
`results/frozen_artifacts.json` is unaffected.

    $py -m research.xdomain.extract_pad                      # all 6 archs + convnext features
    $py -m research.xdomain.extract_pad --archs convnext_tiny  # smoke test one arch
"""

from __future__ import annotations

import argparse
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
from research.selective.features import extract_features

ARCHS = (
    "convnext_tiny",
    "convnext_small",
    "densenet121",
    "efficientnet_b0",
    "efficientnet_b3",
    "resnet50",
)
CHECKPOINT_TEMPLATE = "ml/checkpoints/{arch}_best.HAM-only.pt"
PAD_MANIFEST = "ml/data/manifest_pad.csv"
PAD_ALL_SPLIT = "ml/configs/splits/split_pad_all.csv"
FEATURE_ARCH = "convnext_tiny"


def make_pad_all_split(manifest_path: Path, out_path: Path) -> int:
    """Write a split file marking every PAD row `test` so LesionDataset loads all of it.

    `ml/preprocessing/dataset.py` restricts `split` to train/val/test; labelling every
    external row `test` is the honest choice -- for a HAM-trained model, every PAD image
    is held out. Idempotent: rewritten on each run from the current manifest.
    """
    manifest = pd.read_csv(manifest_path)
    frame = pd.DataFrame(
        {
            "image_id": manifest["image_id"].astype(str),
            "lesion_id": manifest["lesion_id"].astype(str),
            "class_code": manifest["class_code"].astype(str),
            "class_index": manifest["class_index"].astype(int),
            "split": "test",
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return len(frame)


def _pad_config(config: dict) -> dict:
    """A deep-ish copy of the training config pointed at the PAD manifest + all-split."""
    import copy

    cfg = copy.deepcopy(config)
    cfg["data"]["manifest"] = PAD_MANIFEST
    cfg["data"]["splits"] = PAD_ALL_SPLIT
    return cfg


def extract_predictions(arch: str, out_dir: Path, device, batch_size: int, workers: int, config: dict) -> None:
    mapping = load_class_mapping()
    checkpoint = resolve(CHECKPOINT_TEMPLATE.format(arch=arch))
    model, payload = build_model_from_checkpoint(checkpoint, device)
    image_size = payload.get("image_size", config["data"]["image_size"])

    dataset = LesionDataset(
        manifest_path=PAD_MANIFEST,
        splits_path=PAD_ALL_SPLIT,
        split="test",
        transform=build_eval_transform(image_size),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")

    print(f"[{arch}] predicting on PAD ({len(dataset)} images)...")
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
    for c in mapping.classes:
        frame[f"p_{c.code}"] = output["probabilities"][:, c.index]
    for c in mapping.classes:
        frame[f"logit_{c.code}"] = output["logits"][:, c.index]
    frame["arch"] = arch
    frame["split"] = "pad"
    frame["checkpoint"] = checkpoint.relative_to(REPO_ROOT).as_posix()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{arch}.csv"
    frame.to_csv(out_path, index=False)
    print(f"  -> {out_path.relative_to(REPO_ROOT)} ({len(frame)} rows)")


def extract_convnext_features(out_dir: Path, device, batch_size: int, workers: int, config: dict) -> None:
    checkpoint = resolve(CHECKPOINT_TEMPLATE.format(arch=FEATURE_ARCH))
    model, payload = build_model_from_checkpoint(checkpoint, device)
    image_size = payload.get("image_size", config["data"]["image_size"])

    dataset = LesionDataset(
        manifest_path=PAD_MANIFEST,
        splits_path=PAD_ALL_SPLIT,
        split="test",
        transform=build_eval_transform(image_size),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")

    print(f"[{FEATURE_ARCH}] extracting features on PAD ({len(dataset)} images)...")
    features, labels, image_ids = extract_features(model, loader, device)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{FEATURE_ARCH}_pad.npz"
    np.savez_compressed(
        out_path,
        features=features.astype(np.float32),
        labels=labels.astype(np.int64),
        image_ids=np.asarray(image_ids, dtype=object).astype(str),
    )
    print(f"  -> {out_path.relative_to(REPO_ROOT)}  {features.shape[0]} x {features.shape[1]}")


def main(argv: list[str] | None = None) -> int:
    config = load_training_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archs", nargs="+", default=list(ARCHS))
    parser.add_argument("--pred-out-dir", default="research/predictions_pad")
    parser.add_argument("--feature-out-dir", default="research/selective/features")
    parser.add_argument("--batch-size", type=int, default=config["training"]["batch_size"])
    parser.add_argument("--num-workers", type=int, default=config["data"]["num_workers"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-features", action="store_true", help="skip ConvNeXt feature extraction")
    args = parser.parse_args(argv)

    manifest_path = resolve(PAD_MANIFEST)
    if not manifest_path.is_file():
        print(f"ERROR: {PAD_MANIFEST} missing. Run scripts.download_pad_ufes first.", file=sys.stderr)
        return 1
    n = make_pad_all_split(manifest_path, resolve(PAD_ALL_SPLIT))
    print(f"PAD-all split: {n} rows -> {PAD_ALL_SPLIT}\n")

    device = resolve_device(args.device)
    print(f"Device: {device}  archs={args.archs}\n")
    pad_cfg = _pad_config(config)

    for arch in args.archs:
        checkpoint = resolve(CHECKPOINT_TEMPLATE.format(arch=arch))
        if not checkpoint.is_file():
            print(f"ERROR: no checkpoint at {checkpoint}", file=sys.stderr)
            return 1
        extract_predictions(arch, resolve(args.pred_out_dir), device,
                            args.batch_size, args.num_workers, pad_cfg)

    if not args.no_features and FEATURE_ARCH in args.archs:
        extract_convnext_features(resolve(args.feature_out_dir), device,
                                  args.batch_size, args.num_workers, pad_cfg)

    print("\nDone. Now run:  $py -m research.xdomain.run_session8b")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
