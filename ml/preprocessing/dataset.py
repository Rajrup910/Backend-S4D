"""PyTorch dataset backed by the prepared manifest and the committed split file.

The dataset never derives a split itself. It reads `ml/configs/splits/split_v1.csv`,
which was produced once by `split_dataset.py` and committed. That indirection is what
makes "the test set is genuinely unseen" a checkable claim rather than a promise.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ml.paths import REPO_ROOT, load_class_mapping, resolve


class LesionDataset(Dataset):
    """Images for one split, with integer class labels.

    Yields `(image_tensor, label, image_id)`. `image_id` is carried through so that
    evaluation can write per-image predictions and Grad-CAM can find specific cases.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        splits_path: str | Path,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")

        manifest = pd.read_csv(resolve(manifest_path))
        splits = pd.read_csv(resolve(splits_path))

        missing = set(splits["image_id"]) - set(manifest["image_id"])
        if missing:
            raise ValueError(
                f"{len(missing)} image_id(s) in the split file are absent from the manifest "
                f"(e.g. {sorted(missing)[:3]}). The split was built against a different manifest -- "
                f"re-run split_dataset.py."
            )

        frame = manifest.merge(
            splits[["image_id", "split"]], on="image_id", how="inner", validate="one_to_one"
        )
        frame = frame[frame["split"] == split].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"split {split!r} contains no images")

        self.split = split
        self.transform = transform
        self._frame = frame
        self._paths = frame["path"].tolist()
        self._labels = frame["class_index"].astype(int).tolist()
        self._image_ids = frame["image_id"].tolist()

        mapping = load_class_mapping()
        self.class_codes = mapping.codes
        self.num_classes = mapping.num_classes

        out_of_range = [label for label in self._labels if not 0 <= label < self.num_classes]
        if out_of_range:
            raise ValueError(
                f"manifest contains class_index values outside 0..{self.num_classes - 1}: "
                f"{sorted(set(out_of_range))[:5]}"
            )

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path = REPO_ROOT / self._paths[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
        except Exception as exc:  # noqa: BLE001 - surface the offending file, not a bare stack
            raise RuntimeError(f"failed to read image {path}: {exc}") from exc

        tensor = self.transform(image) if self.transform else image
        return tensor, self._labels[index], self._image_ids[index]

    # --- helpers used by training and evaluation ---

    @property
    def labels(self) -> list[int]:
        """All labels in dataset order -- used for class weights and samplers."""
        return list(self._labels)

    @property
    def image_ids(self) -> list[str]:
        return list(self._image_ids)

    def class_counts(self) -> list[int]:
        counts = [0] * self.num_classes
        for label in self._labels:
            counts[label] += 1
        return counts

    def describe(self) -> dict[str, Any]:
        counts = self.class_counts()
        return {
            "split": self.split,
            "images": len(self),
            "lesions": int(self._frame["lesion_id"].nunique()),
            "per_class": dict(zip(self.class_codes, counts, strict=True)),
        }
