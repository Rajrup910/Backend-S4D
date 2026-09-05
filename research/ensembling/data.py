"""Load and align the standardized prediction matrices from `research/predictions/`.

Every ensembling algorithm needs the same aligned tensor: N images x K architectures x C
classes, plus the ground truth and lesion IDs (for group-aware CV). This module is the
single place that reads those CSVs so alignment is checked once, not once per method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.paths import load_class_mapping, load_training_config, resolve
from research import testguard

ARCHS = (
    "convnext_tiny",
    "convnext_small",
    "densenet121",
    "efficientnet_b0",
    "efficientnet_b3",
    "resnet50",
)


@dataclass(frozen=True)
class PredictionMatrix:
    split: str
    archs: tuple[str, ...]
    class_codes: tuple[str, ...]
    image_ids: np.ndarray  # (N,)
    lesion_ids: np.ndarray  # (N,)
    y_true: np.ndarray  # (N,) int
    probs: np.ndarray  # (N, K, C) float
    logits: np.ndarray  # (N, K, C) float

    @property
    def num_samples(self) -> int:
        return len(self.image_ids)

    @property
    def num_archs(self) -> int:
        return len(self.archs)

    @property
    def num_classes(self) -> int:
        return len(self.class_codes)

    def probs_for(self, arch: str) -> np.ndarray:
        return self.probs[:, self.archs.index(arch), :]

    def preds_for(self, arch: str) -> np.ndarray:
        return self.probs_for(arch).argmax(axis=1)


def _lesion_lookup() -> pd.Series:
    """image_id -> lesion_id, from the split file (needed for group-aware inner CV)."""
    config = load_training_config()
    splits = pd.read_csv(resolve(config["data"]["splits"]))
    return splits.set_index("image_id")["lesion_id"]


def load_split_matrix(
    split: str,
    archs: tuple[str, ...] = ARCHS,
    predictions_dir: str = "research/predictions",
) -> PredictionMatrix:
    testguard.check_split(split, "research.ensembling.data.load_split_matrix")
    mapping = load_class_mapping()
    class_codes = mapping.codes
    pred_dir = resolve(predictions_dir)

    frames = {}
    for arch in archs:
        path = pred_dir / f"{arch}_{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing prediction file {path}. Run "
                f"`python -m research.extract_predictions` first (Session 0)."
            )
        frames[arch] = pd.read_csv(path).sort_values("image_id").reset_index(drop=True)

    reference = frames[archs[0]]
    image_ids = reference["image_id"].to_numpy()
    y_true = reference["true_index"].to_numpy()

    for arch, frame in frames.items():
        if not np.array_equal(frame["image_id"].to_numpy(), image_ids):
            raise ValueError(f"image_id mismatch between {archs[0]} and {arch} on split={split!r}")
        if not np.array_equal(frame["true_index"].to_numpy(), y_true):
            raise ValueError(f"true label mismatch between {archs[0]} and {arch} on split={split!r}")

    probs = np.stack(
        [frames[arch][[f"p_{c}" for c in class_codes]].to_numpy() for arch in archs], axis=1
    )
    logits = np.stack(
        [frames[arch][[f"logit_{c}" for c in class_codes]].to_numpy() for arch in archs], axis=1
    )

    lesion_lookup = _lesion_lookup()
    lesion_ids = lesion_lookup.loc[image_ids].to_numpy()

    return PredictionMatrix(
        split=split,
        archs=archs,
        class_codes=class_codes,
        image_ids=image_ids,
        lesion_ids=lesion_ids,
        y_true=y_true,
        probs=probs,
        logits=logits,
    )
