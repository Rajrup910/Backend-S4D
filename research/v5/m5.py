"""M5 -- expert structure supervision from ISIC-2018 Task 2 (Amendment 02 M5; runsheet section 7).

    python -m research.v5.m5 --overlap        # Q4 pre-flight: results/v5/diagnostics/m5_overlap.json

Written 30 Sep as a NEW file while the night queue was running; it is wired into `train_v5` /
`arms.py` only after the queue ends (the queue re-imports those files for every run). Wiring
notes, in order:
  1. `train_v5.run` refuses "m5" today -- remove it from the guard set once the smoke passes.
  2. `build_arm`: `ArmNet` gets `self.m5_head = M5Head()` when `spec.has("m5")`, and its forward
     returns `out["m5_logits"] = self.m5_head(f3)`.
  3. Rows: `Task2Masks.for_frame(train_frame)` gives `has_attr` and masks; the paired transform
     `PairedTrainTransform` replaces `build_train_transform` for `m5` (images and masks share
     every geometric transform; colour transforms touch the image only).
  4. Loss: `m5_loss(out["m5_logits"], masks, has_attr)` x M5_WEIGHT, added in `compute_loss`.
  Masks of val / reserved rows are never loaded (`Task2Masks` only serves ids the caller passes
  from the *training* frame, and `for_frame` asserts `split == "train"`).

Fixed values (runsheet section 7): Conv3x3(384->128) -> GELU -> Conv1x1(128->5); BCE + soft Dice,
total weight 0.20; targets area-downsampled to the stride-16 map; sum over labelled rows divided
by max(1, sum(has_attr)).
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.v4.recipe import MANIFEST, REPO_ROOT

TASK2_ZIP = REPO_ROOT / "data" / "external" / "isic2018_task2" / "ISIC2018_Task2_Training_GroundTruth_v3.zip"
OVERLAP_JSON = REPO_ROOT / "results" / "v5" / "diagnostics" / "m5_overlap.json"
ATTRIBUTES = ("pigment_network", "negative_network", "streaks", "globules", "milia_like_cyst")
M5_WEIGHT = 0.20
M5_HIDDEN = 128
MIN_TRAIN_ROWS = 1000  # Q4 pass rule


class M5Head(nn.Module):
    """Conv3x3(384->128) -> GELU -> Conv1x1(128->5) on the stride-16 map F3 (about 0.44 M params)."""

    def __init__(self, in_channels: int = 384) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, M5_HIDDEN, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(M5_HIDDEN, len(ATTRIBUTES), 1))

    def forward(self, f3: torch.Tensor) -> torch.Tensor:
        return self.body(f3)


def downsample_masks(masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """(B, 5, H, W) in {0,1} -> (B, 5, h, w) soft targets in [0,1] by area averaging, so thin
    streaks survive (nearest-neighbour would drop them)."""
    return F.adaptive_avg_pool2d(masks.float(), size)


def m5_loss(logits: torch.Tensor, masks: torch.Tensor, has_attr: torch.Tensor,
            eps: float = 1.0) -> torch.Tensor:
    """BCE + soft Dice per attribute, summed over labelled rows and divided by max(1, sum has_attr).

    An attribute absent from a labelled image is a valid all-zero target (has_attr = 1); an image
    not in Task 2 is unlabelled (has_attr = 0) and contributes exactly 0. `masks` must already be
    at the logit resolution (use `downsample_masks`).
    """
    logits = logits.float()
    target = masks.float()
    w = has_attr.float().view(-1, 1, 1, 1)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * w).mean(dim=(2, 3)).sum(dim=1)  # (B,)
    prob = torch.sigmoid(logits)
    inter = (prob * target * w).sum(dim=(2, 3))
    denom = (prob * w).sum(dim=(2, 3)) + (target * w).sum(dim=(2, 3))
    dice = 1.0 - (2.0 * inter + eps) / (denom + eps)  # (B, 5)
    per_row = (bce + dice.sum(dim=1) * has_attr.float())
    per_row = per_row * has_attr.float()
    return M5_WEIGHT * per_row.sum() / has_attr.float().sum().clamp(min=1.0)


class Task2Masks:
    """Reads expert masks straight from the ground-truth zip (no extraction, ~143 MB on disk)."""

    def __init__(self, path: Path = TASK2_ZIP) -> None:
        self.path = path
        self._zip: zipfile.ZipFile | None = None
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
        self.prefix = names[0].split("/")[0] + "/"
        ids = {n.split("/")[-1].rsplit("_attribute_", 1)[0] for n in names if "_attribute_" in n}
        self.ids = frozenset(ids)

    def __len__(self) -> int:
        return len(self.ids)

    def has(self, image_id: str) -> bool:
        return image_id in self.ids

    def load(self, image_id: str) -> np.ndarray | None:
        """(5, H, W) uint8 in {0,1} in ATTRIBUTES order, or None when the image is not in Task 2."""
        if image_id not in self.ids:
            return None
        from PIL import Image

        if self._zip is None:
            self._zip = zipfile.ZipFile(self.path)
        planes = []
        for attr in ATTRIBUTES:
            with self._zip.open(f"{self.prefix}{image_id}_attribute_{attr}.png") as fh:
                planes.append((np.asarray(Image.open(io.BytesIO(fh.read())).convert("L")) > 127)
                              .astype(np.uint8))
        return np.stack(planes)

    def for_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """has_attr (N,) for a TRAINING frame; asserts no val/reserved row is passed."""
        if "split" in frame.columns and (frame["split"] != "train").any():
            raise AssertionError("M5 masks may only be requested for training rows")
        return np.array([self.has(str(i)) for i in frame["image_id"]], dtype=np.float32)


class PairedTrainTransform:
    """Geometric transforms applied identically to the image and its 5 masks (transforms.v2 +
    tv_tensors.Mask); colour transforms touch the image only. Unlabelled rows pass zero masks."""

    def __init__(self, image_size: int) -> None:
        from torchvision.transforms import v2

        self.size = image_size
        self.geo = v2.Compose([
            v2.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.111), antialias=True),
            v2.RandomHorizontalFlip(0.5), v2.RandomVerticalFlip(0.5),
            v2.RandomRotation(degrees=20, fill=0)])
        self.colour = v2.Compose([v2.ColorJitter(0.15, 0.15, 0.10, 0.02)])
        self.norm = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    def __call__(self, image, masks: np.ndarray | None):
        from torchvision import tv_tensors
        from torchvision.transforms import v2

        img = tv_tensors.Image(v2.functional.pil_to_tensor(image))
        if masks is None:
            masks = np.zeros((len(ATTRIBUTES), img.shape[-2], img.shape[-1]), dtype=np.uint8)
        mk = tv_tensors.Mask(torch.from_numpy(masks))
        img, mk = self.geo(img, mk)
        img = self.colour(img)
        return self.norm(img), mk.float()


def fit_masks_to_image(masks: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Task-2 masks at the corpus image's size. Same size -> unchanged; same aspect ratio (within
    1%) -> nearest-neighbour resize; otherwise None (the row is treated as unlabelled)."""
    if masks.shape[1:] == (height, width):
        return masks
    if abs(masks.shape[2] / masks.shape[1] - width / height) > 0.01 * (width / height):
        return None
    from PIL import Image

    return np.stack([(np.asarray(Image.fromarray(m * 255).resize((width, height), Image.NEAREST))
                      > 127).astype(np.uint8) for m in masks])


class M5TrainDataset(torch.utils.data.Dataset):
    """Training rows of the `m5` arm: (image, label, extras, masks (5, S, S), has_attr).

    Same rows, labels and extras as `V4Dataset`; the image transform is `PairedTrainTransform`
    (the V4 train augmentation, with the masks carried through every geometric step). Only the
    TRAINING frame may be passed (asserted), so no val / reserved mask is ever read."""

    def __init__(self, frame: pd.DataFrame, image_dir: Path, image_size: int,
                 tabular: np.ndarray | None = None, masks: Task2Masks | None = None) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self._ids = self.frame["image_id"].astype(str).tolist()
        self._labels = self.frame["class_index_7"].astype(int).tolist()
        self.tabular = tabular
        self.transform = PairedTrainTransform(image_size)
        self.masks = masks or Task2Masks()
        self.has_attr = self.masks.for_frame(self.frame)

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def labels(self) -> list[int]:
        return list(self._labels)

    def class_counts(self, num_classes: int) -> list[int]:
        return np.bincount(np.asarray(self._labels), minlength=num_classes).tolist()

    def __getitem__(self, index: int):
        from PIL import Image

        image_id = self._ids[index]
        with Image.open(self.image_dir / f"{image_id}.jpg") as image:
            image = image.convert("RGB")
        masks, has = None, 0.0
        if self.has_attr[index]:
            loaded = self.masks.load(image_id)
            masks = fit_masks_to_image(loaded, image.width, image.height) if loaded is not None else None
            has = 1.0 if masks is not None else 0.0
        x, mk = self.transform(image, masks)
        row = self.tabular[index] if self.tabular is not None else np.zeros(0, dtype=np.float32)
        return (x, self._labels[index], torch.from_numpy(np.asarray(row, dtype=np.float32)),
                mk, torch.tensor(has, dtype=torch.float32))


def overlap_report() -> dict:
    """Q4: Task-2 ids against the manifest by split. Pass rule: >= 1,000 train rows and no
    val/reserved row used. The 12 under-40 MSKCC melanomas are reported, never a gate."""
    masks = Task2Masks()
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    manifest["in_task2"] = manifest["image_id"].astype(str).isin(masks.ids)
    by_split = manifest.groupby("split")["in_task2"].sum().astype(int).to_dict()
    train = manifest[manifest["split"] == "train"]
    u40_mel = train[(train["archive"] == "mskcc") & (train["age_band"] == "<40")
                    & (train["class_7"] == "mel")]
    # Pixel agreement (runsheet section 4, Q4): the Task-2 masks must sit on the corpus image's
    # pixel grid -- same size, or the same aspect ratio (nearest-neighbour resize). 50 train
    # overlaps drawn with a fixed seed; images are opened for their size only.
    from PIL import Image

    from research.v4.recipe import IMAGE_DIR

    sample = train[train["in_task2"]].sample(n=min(50, int(train["in_task2"].sum())),
                                             random_state=20260930)
    agree = {"same_size": 0, "same_aspect_resized": 0, "mismatch": 0, "nonempty_masks": 0}
    for image_id in sample["image_id"].astype(str):
        with Image.open(IMAGE_DIR / f"{image_id}.jpg") as im:
            width, height = im.size
        raw = masks.load(image_id)
        fitted = fit_masks_to_image(raw, width, height)
        if fitted is None:
            agree["mismatch"] += 1
        elif raw.shape[1:] == (height, width):
            agree["same_size"] += 1
        else:
            agree["same_aspect_resized"] += 1
        agree["nonempty_masks"] += int(raw.any())
    pixels_agree = agree["mismatch"] == 0 and len(sample) > 0
    report = {
        "task2_images": len(masks), "overlap_by_split": by_split,
        "overlap_train_rows": int(by_split.get("train", 0)),
        "train_overlap_by_archive": train[train["in_task2"]].groupby("archive").size().to_dict(),
        "val_or_reserved_rows_used": 0,  # M5 loads masks for training rows only (Task2Masks.for_frame)
        "task2_ids_in_val_or_reserved_not_used": int(sum(v for k, v in by_split.items() if k != "train")),
        "under40_mskcc_mel_train": int(len(u40_mel)),
        "under40_mskcc_mel_with_masks": int(u40_mel["in_task2"].sum()),
        "pixel_agreement_sample": {"n": int(len(sample)), **agree, "pass": bool(pixels_agree)},
        "pass_rule": (f">= {MIN_TRAIN_ROWS} train rows overlap, 0 val/reserved rows used, and "
                      f"the 50-image sample's masks sit on the image grid"),
        "pass": bool(by_split.get("train", 0) >= MIN_TRAIN_ROWS and pixels_agree),
        "licence_note": "ISIC 2018 Task 2 ground truth: CC-BY-NC (per ATTRIBUTION.txt in the zip)",
        "test_read": False, "reserved_read": False,
    }
    OVERLAP_JSON.parent.mkdir(parents=True, exist_ok=True)
    OVERLAP_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--overlap", action="store_true")
    args = parser.parse_args(argv)
    if args.overlap:
        r = overlap_report()
        print(json.dumps({k: r[k] for k in ("task2_images", "overlap_by_split", "overlap_train_rows",
                                             "under40_mskcc_mel_with_masks",
                                             "pixel_agreement_sample", "pass")}, indent=2))
        print(f"wrote {OVERLAP_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
