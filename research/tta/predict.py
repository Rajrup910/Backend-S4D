"""Run a frozen model over the 24 TTA views and pool with uncertainty-weighted averaging.

Views the model is confident about (low predictive entropy) count for more than views it
finds ambiguous -- a blurry off-center crop shouldn't drag down an otherwise-clear view of
the lesion.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from research.tta.transforms import TTAView, build_tta_views


class _RawImageDataset(Dataset):
    """Wraps a LesionDataset-like source but yields the untransformed PIL image."""

    def __init__(self, base_dataset) -> None:
        self.base = base_dataset
        # `--limit` wraps the LesionDataset in a torch Subset (possibly more than once),
        # which carries no `.transform`; unwrap before asserting so the documented
        # smoke-test path works rather than raising AttributeError.
        inner = base_dataset
        while isinstance(inner, Subset):
            inner = inner.dataset
        assert inner.transform is None, "pass a LesionDataset with transform=None to TTA"

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label, image_id = self.base[index]
        return image, label, image_id


def _collate_raw(batch):
    images, labels, ids = zip(*batch, strict=True)
    return list(images), torch.tensor(labels), list(ids)


@torch.no_grad()
def predict_tta(
    model: nn.Module,
    base_dataset,
    device: torch.device,
    image_size: int,
    scales: tuple[float, ...] = (0.9, 1.0, 1.1),
    batch_size: int = 4,
    num_workers: int = 2,
    eps: float = 1e-8,
    description: str = "TTA predicting",
) -> dict[str, np.ndarray]:
    model.eval()
    views: list[TTAView] = build_tta_views(image_size, scales)
    num_views = len(views)

    dataset = _RawImageDataset(base_dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate_raw
    )

    probs_all, labels_all, ids_all = [], [], []

    for images, labels, image_ids in tqdm(loader, desc=description, unit="batch"):
        # (batch * num_views, 3, H, W), image-major then view-major so a reshape recovers (batch, views, C)
        view_tensors = torch.stack(
            [view.transform(image) for image in images for view in views], dim=0
        ).to(device, non_blocking=True)

        logits = model(view_tensors).float()
        batch_n = len(images)
        logits = logits.view(batch_n, num_views, -1)
        probs = torch.softmax(logits, dim=-1)  # (batch, views, C)

        entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)  # (batch, views)
        weights = 1.0 / (entropy + eps)
        weights = weights / weights.sum(dim=1, keepdim=True)

        pooled = (probs * weights.unsqueeze(-1)).sum(dim=1)  # (batch, C)

        probs_all.append(pooled.cpu())
        labels_all.append(labels)
        ids_all.extend(image_ids)

    probabilities = torch.cat(probs_all).numpy()
    return {
        "probabilities": probabilities,
        "predictions": probabilities.argmax(axis=1),
        "labels": torch.cat(labels_all).numpy(),
        "image_ids": np.array(ids_all),
    }
