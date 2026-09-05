"""Load per-image predictions for any ladder rung into a common (y_true, probs, lesion_ids)
shape, regardless of which script produced the CSV.

`research/predictions/`, `research/predictions_tta/` and the two one-off dumps
(`research/predictions/gated_fusion_convnext_tiny_*.csv`, `ml/results/<arch>/predictions.csv`)
use three different column layouts. This module is the single place that normalizes them so
`research/ablation/stats.py` and `run_part_a.py` never special-case a file format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.paths import load_class_mapping, load_training_config, resolve


@dataclass(frozen=True)
class Predictions:
    name: str
    image_ids: np.ndarray
    lesion_ids: np.ndarray
    y_true: np.ndarray
    probs: np.ndarray  # (N, C)

    @property
    def y_pred(self) -> np.ndarray:
        return self.probs.argmax(axis=1)

    @property
    def num_samples(self) -> int:
        return len(self.image_ids)


def _lesion_lookup() -> pd.Series:
    config = load_training_config()
    splits = pd.read_csv(resolve(config["data"]["splits"]))
    return splits.set_index("image_id")["lesion_id"]


def load_predictions(csv_path: str | Path, name: str | None = None) -> Predictions:
    """Load any `research/predictions*/`-style or `ml/results/<arch>/predictions.csv` file.

    Both layouts carry `image_id`, a ground-truth column (`true_index` or `true_code`) and
    `p_<code>` probability columns; that overlap is all this function relies on.
    """
    path = resolve(csv_path)
    frame = pd.read_csv(path)
    mapping = load_class_mapping()
    codes = list(mapping.codes)

    prob_cols = [f"p_{c}" for c in codes]
    missing = [c for c in prob_cols if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing probability columns {missing}")
    probs = frame[prob_cols].to_numpy(dtype=float)

    if "true_index" in frame.columns:
        y_true = frame["true_index"].to_numpy(dtype=int)
    elif "true_code" in frame.columns:
        code_to_index = {c.code: c.index for c in mapping.classes}
        y_true = frame["true_code"].map(code_to_index).to_numpy(dtype=int)
    else:
        raise ValueError(f"{path} has neither true_index nor true_code")

    image_ids = frame["image_id"].to_numpy()
    lesion_lookup = _lesion_lookup()
    lesion_ids = lesion_lookup.loc[image_ids].to_numpy()

    return Predictions(
        name=name or path.stem,
        image_ids=image_ids,
        lesion_ids=lesion_ids,
        y_true=y_true,
        probs=probs,
    )


def align(a: Predictions, b: Predictions) -> tuple[Predictions, Predictions]:
    """Reorder `b` to match `a`'s image_id order; raises if the id sets differ."""
    if set(a.image_ids) != set(b.image_ids):
        raise ValueError(f"{a.name} and {b.name} cover different image sets")
    order = pd.Index(b.image_ids).get_indexer(a.image_ids)
    if (order < 0).any():
        raise ValueError(f"could not align {b.name} to {a.name}")
    return a, Predictions(
        name=b.name,
        image_ids=b.image_ids[order],
        lesion_ids=b.lesion_ids[order],
        y_true=b.y_true[order],
        probs=b.probs[order],
    )
