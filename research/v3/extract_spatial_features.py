"""S41 / Phase B2 support -- mask-interior vs mask-exterior pooled features.

`probe_lesion_vs_context` needs two feature vectors per image: one pooled over the lesion and
one pooled over everything else. The cached files cannot supply them --
`research/selective/features.py` hooks the final `Linear`, whose input is already
globally pooled and has no spatial extent left. This module hooks one layer earlier, at the
input to `AdaptiveAvgPool2d` (verified `(B, 768, 7, 7)` for ConvNeXt-Tiny at 224), and does the
pooling itself against the Tschandl segmentation masks.

**The mask is put through the image's own geometry.** `build_eval_transform` is
`Resize(224 * RESIZE_RATIO)` then `CenterCrop(224)`; applying anything else to the mask would
misalign interior and exterior by a few pixels everywhere and quietly contaminate both pools.
The same `Resize`/`CenterCrop` pair is therefore reused here, with nearest-neighbour
interpolation so the mask stays binary, and *without* `ToTensor`/`Normalize`, which are
photometric and have no meaning for a mask.

**Pooling is soft, not thresholded.** At 7x7 each cell covers a 32x32 patch of the input, so a
hard in/out label per cell would be badly quantized along the lesion boundary. The binary mask
is average-pooled to the feature grid to give each cell an interior *fraction* `w`, and the two
vectors are the `w`- and `(1-w)`-weighted means of the feature map. Cells straddling the border
contribute to both, in proportion.

Features come from the **fold** checkpoints, matching `extract_oof_features.py`, so nothing
here is scored by a model that trained on it.

⚠️ One documented mismatch with `convnext_tiny_oof.npz`: torchvision's ConvNeXt applies
`LayerNorm2d` *after* `avgpool`, so the vectors here are pre-norm while the cached penultimate
features are post-norm. The interior/exterior comparison is internally valid because both sides
get identical treatment, but these vectors are not interchangeable with the cached ones.

    $py -m research.v3.extract_spatial_features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ml.paths import REPO_ROOT, load_training_config, resolve
from ml.preprocessing.transforms import RESIZE_RATIO, build_eval_transform
from ml.training.common import build_model_from_checkpoint, resolve_device

FOLDS_PATH = REPO_ROOT / "ml" / "configs" / "splits" / "split_v1.folds.csv"
CHECKPOINT_TEMPLATE = "ml/checkpoints/oof/{arch}-oof_f{fold}_best.pt"
MASK_DIR = (REPO_ROOT / "data" / "ham10000" / "HAM10000_segmentations_lesion_tschandl"
            / "HAM10000_segmentations_lesion_tschandl")
MASK_SUFFIX = "_segmentation.png"  # matches research/session9/attribution.py
OUT_DIR = REPO_ROOT / "research" / "v3" / "features"
N_FOLDS = 5
MIN_WEIGHT = 1e-3  # a pool with less total weight than this is not a real region


class _MaskedDataset(Dataset):
    """Yields `(image_tensor, mask_tensor, image_id)` with the mask under the image's geometry."""

    def __init__(self, manifest: pd.DataFrame, image_ids: list[str], image_size: int) -> None:
        frame = manifest.set_index("image_id").loc[image_ids].reset_index()
        self._paths = frame["path"].tolist()
        self._image_ids = frame["image_id"].tolist()
        self.transform = build_eval_transform(image_size)
        self.mask_transform = transforms.Compose([
            transforms.Resize(int(round(image_size * RESIZE_RATIO)),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size),
        ])

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int):
        image_id = self._image_ids[index]
        with Image.open(REPO_ROOT / self._paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        with Image.open(MASK_DIR / f"{image_id}{MASK_SUFFIX}") as raw:
            mask = self.mask_transform(raw.convert("L"))
        mask_t = (torch.from_numpy(np.asarray(mask, dtype=np.float32)) > 127).float()
        return tensor, mask_t, image_id


def _pool_layer(model: nn.Module) -> nn.Module:
    pools = [m for m in model.modules() if isinstance(m, nn.AdaptiveAvgPool2d)]
    if not pools:
        raise ValueError("no AdaptiveAvgPool2d to hook -- cannot locate the spatial feature map")
    return pools[-1]


@torch.no_grad()
def extract_masked(model: nn.Module, loader: DataLoader, device: torch.device):
    """Interior- and exterior-weighted means of the pre-pool feature map."""
    captured: list[torch.Tensor] = []
    handle = _pool_layer(model).register_forward_hook(
        lambda _m, inputs, _o: captured.append(inputs[0].detach().float()))
    model.eval()

    interior, exterior, ids, degenerate = [], [], [], []
    try:
        for images, masks, batch_ids in loader:
            captured.clear()
            model(images.to(device, non_blocking=True))
            feats = captured[0]                                    # (B, C, H, W)
            grid = feats.shape[-2:]
            w = torch.nn.functional.adaptive_avg_pool2d(
                masks.unsqueeze(1).to(device), grid)               # (B, 1, H, W) interior fraction

            for weight, sink in ((w, interior), (1.0 - w, exterior)):
                # total must be (B, 1) to divide a (B, C) pool: a (B, 1, 1) divisor
                # broadcasts to (B, B, C) instead, silently, which the smoke test caught.
                total = weight.sum(dim=(1, 2, 3)).clamp_min(MIN_WEIGHT).unsqueeze(1)
                pooled = (feats * weight).sum(dim=(2, 3)) / total
                sink.append(pooled.cpu().numpy())

            flat_w = w.sum(dim=(1, 2, 3))
            cells = float(grid[0] * grid[1])
            for i, image_id in enumerate(batch_ids):
                if flat_w[i] < MIN_WEIGHT or (cells - flat_w[i]) < MIN_WEIGHT:
                    degenerate.append(image_id)
            ids.extend(batch_ids)
    finally:
        handle.remove()

    return (np.concatenate(interior), np.concatenate(exterior),
            np.asarray(ids, dtype=str), degenerate)


def run(arch: str, batch_size: int, workers: int, device_arg: str) -> int:
    if not MASK_DIR.is_dir():
        raise FileNotFoundError(f"no Tschandl masks at {MASK_DIR}")
    config = load_training_config()
    device = resolve_device(device_arg)
    manifest = pd.read_csv(resolve(config["data"]["manifest"]))
    folds = pd.read_csv(FOLDS_PATH)

    have_mask = {p.name[: -len(MASK_SUFFIX)] for p in MASK_DIR.glob(f"*{MASK_SUFFIX}")}
    usable = folds[folds["image_id"].astype(str).isin(have_mask)]
    missing = len(folds) - len(usable)
    print(f"Device: {device}  arch={arch}")
    print(f"{len(usable)} of {len(folds)} OOF rows have a mask ({missing} missing)\n")

    all_in, all_out, all_ids, all_degenerate = [], [], [], []
    for fold in range(N_FOLDS):
        held = usable.loc[usable["fold"] == fold, "image_id"].astype(str).tolist()
        if not held:
            continue
        checkpoint = resolve(CHECKPOINT_TEMPLATE.format(arch=arch, fold=fold))
        model, payload = build_model_from_checkpoint(checkpoint, device)
        image_size = payload.get("image_size", config["data"]["image_size"])
        loader = DataLoader(_MaskedDataset(manifest, held, image_size), batch_size=batch_size,
                            shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
        print(f"[fold {fold}] {len(held)} held-out rows with masks")
        inside, outside, ids, degenerate = extract_masked(model, loader, device)
        all_in.append(inside)
        all_out.append(outside)
        all_ids.extend(ids)
        all_degenerate.extend(degenerate)
        print(f"  -> interior {inside.shape}  exterior {outside.shape}"
              + (f"  ({len(degenerate)} degenerate)" if degenerate else ""))

    interior = np.concatenate(all_in).astype(np.float32)
    exterior = np.concatenate(all_out).astype(np.float32)
    ids = np.asarray(all_ids, dtype=str)

    keep = ~np.isin(ids, np.asarray(all_degenerate, dtype=str)) if all_degenerate else np.ones(len(ids), bool)
    if all_degenerate:
        print(f"\ndropping {len(all_degenerate)} images whose mask leaves one region empty")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{arch}_oof_spatial.npz"
    np.savez_compressed(out_path, interior=interior[keep], exterior=exterior[keep],
                        image_ids=ids[keep])
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}  {int(keep.sum())} x {interior.shape[1]} per region")
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
