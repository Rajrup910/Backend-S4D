"""Extract penultimate-layer features for the Mahalanobis distance score.

    python -m research.selective.features                     # convnext_tiny, all 3 splits
    python -m research.selective.features --archs convnext_tiny resnet50
    python -m research.selective.features --splits train val  # skip test until needed

This is the only part of Phase 4 that touches the GPU. Everything else reads the cached
probability matrices in `research/predictions_tta/`, but a feature-space OOD score needs
the activations feeding the classifier head, and those were never written to disk.

Features are taken as the **input to the final Linear layer**, captured with a forward
hook rather than by surgery on the model. Every supported backbone — the four
torchvision CNNs, the two ConvNeXts, and the timm transformers — ends in a single Linear
head, so hooking it is the one definition of "penultimate" that holds across all of them
and cannot drift from what the checkpoint actually computes.

The train split is included deliberately: the Gaussians in `mahalanobis.py` must be
fitted on training data only, never on the validation cases whose scores set the
abstention threshold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.paths import REPO_ROOT, load_training_config, resolve
from ml.preprocessing.dataset import LesionDataset
from ml.preprocessing.transforms import build_eval_transform
from ml.training.common import build_model_from_checkpoint, resolve_device
from research import testguard

DEFAULT_ARCHS = ("convnext_tiny",)
DEFAULT_SPLITS = ("train", "val", "test")
CHECKPOINT_TEMPLATE = "ml/checkpoints/{arch}_best.HAM-only.pt"


def _final_linear(model: nn.Module) -> nn.Linear:
    """The classifier head: the last nn.Linear in registration order."""
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("model has no nn.Linear layer to hook")
    return linears[-1]


@torch.no_grad()
def extract_features(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run the loader through the model, returning (features, labels, image_ids)."""
    captured: list[torch.Tensor] = []

    def hook(_module, inputs, _output) -> None:
        captured.append(inputs[0].detach().float().cpu())

    handle = _final_linear(model).register_forward_hook(hook)
    model.eval()

    features, labels, image_ids = [], [], []
    try:
        for images, batch_labels, batch_ids in loader:
            captured.clear()
            model(images.to(device, non_blocking=True))
            if len(captured) != 1:
                raise RuntimeError(f"expected 1 head activation per batch, got {len(captured)}")
            features.append(captured[0].numpy())
            labels.append(np.asarray(batch_labels))
            image_ids.extend(batch_ids)
    finally:
        handle.remove()

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0), image_ids


def feature_path(arch: str, split: str, out_dir: str = "research/selective/features") -> Path:
    return resolve(out_dir) / f"{arch}_{split}.npz"


def load_features(arch: str, split: str, out_dir: str = "research/selective/features") -> dict:
    """Read a cached npz, with an actionable message if the GPU pass has not been run."""
    testguard.check_split(split, "research.selective.features.load_features")
    path = feature_path(arch, split, out_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path.relative_to(REPO_ROOT)}. Run "
            f"`python -m research.selective.features --archs {arch}` first."
        )
    data = np.load(path, allow_pickle=False)
    return {"features": data["features"], "labels": data["labels"], "image_ids": data["image_ids"]}


def extract_one(arch: str, split: str, out_dir: Path, device, batch_size: int, workers: int, config: dict) -> None:
    checkpoint = resolve(CHECKPOINT_TEMPLATE.format(arch=arch))
    model, payload = build_model_from_checkpoint(checkpoint, device)
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
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    print(f"[{arch}] extracting features on {split} ({len(dataset)} images)...")
    features, labels, image_ids = extract_features(model, loader, device)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{arch}_{split}.npz"
    np.savez_compressed(
        out_path,
        features=features.astype(np.float32),
        labels=labels.astype(np.int64),
        image_ids=np.asarray(image_ids, dtype=object).astype(str),
    )
    print(f"  -> {out_path.relative_to(REPO_ROOT)}  {features.shape[0]} x {features.shape[1]}")


def main(argv: list[str] | None = None) -> int:
    config = load_training_config()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--archs", nargs="+", default=list(DEFAULT_ARCHS))
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS),
                        choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default="research/selective/features")
    parser.add_argument("--batch-size", type=int, default=config["training"]["batch_size"])
    parser.add_argument("--num-workers", type=int, default=config["data"]["num_workers"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    out_dir = resolve(args.out_dir)
    print(f"Device: {device}  archs={args.archs}  splits={args.splits}\n")

    for arch in args.archs:
        checkpoint = resolve(CHECKPOINT_TEMPLATE.format(arch=arch))
        if not checkpoint.is_file():
            print(f"ERROR: no checkpoint at {checkpoint}", file=sys.stderr)
            return 1
        for split in args.splits:
            extract_one(arch, split, out_dir, device, args.batch_size, args.num_workers, config)

    print(f"\nDone. Features in {out_dir.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
