"""Image + tabular-metadata dataset, built on the same manifest/split files as `LesionDataset`.

Kept as a separate class rather than extending `LesionDataset` because the fusion model's
`forward(image, tabular)` needs a different item shape (image, tabular, label, image_id)
-- reusing `LesionDataset`'s loading logic here would mean overriding `__getitem__`
anyway, and duplicating the ~15 lines of manifest/split merge is clearer than subclassing
around a return-shape mismatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ml.paths import REPO_ROOT, load_class_mapping, resolve
from research.fusion.tabular import TabularEncoder


class FusionLesionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        splits_path: str | Path,
        split: str,
        encoder: TabularEncoder,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")

        manifest = pd.read_csv(resolve(manifest_path))
        splits = pd.read_csv(resolve(splits_path))
        frame = manifest.merge(
            splits[["image_id", "split"]], on="image_id", how="inner", validate="one_to_one"
        )
        frame = frame[frame["split"] == split].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"split {split!r} contains no images")

        self.split = split
        self.transform = transform
        self._paths = frame["path"].tolist()
        self._labels = frame["class_index"].astype(int).tolist()
        self._image_ids = frame["image_id"].tolist()
        self._tabular = encoder.transform(frame).astype(np.float32)

        mapping = load_class_mapping()
        self.num_classes = mapping.num_classes
        self.tabular_dim = encoder.output_dim

    def __len__(self) -> int:
        return len(self._paths)

    def class_counts(self) -> list[int]:
        counts = [0] * self.num_classes
        for label in self._labels:
            counts[label] += 1
        return counts

    def __getitem__(self, index: int):
        path = REPO_ROOT / self._paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        tensor = self.transform(image) if self.transform else image
        tabular = torch.from_numpy(self._tabular[index])
        return tensor, tabular, self._labels[index], self._image_ids[index]
