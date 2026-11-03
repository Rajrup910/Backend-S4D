"""8-class head (V6-1 arm T): AK and SCC as separate classes, read out as the 7-class problem by summing them.

Index order is `ml/data/manifest_v4.csv`'s `class_index_8`: akiec_ak 0, bcc 1, bkl 2, df 3, mel 4, nv 5, scc 6, vasc 7.
The 7-class order is `ml/configs/class_mapping.json`: akiec 0, bcc 1, bkl 2, df 3, mel 4, nv 5, vasc 6, where `akiec` is the
HAM class that already holds AK plus intraepithelial carcinoma. Summing probabilities (not logits) keeps a proper simplex.
"""

from __future__ import annotations

import numpy as np

CLASSES_8 = ("akiec_ak", "bcc", "bkl", "df", "mel", "nv", "scc", "vasc")
CLASSES_7 = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
#: 7-class index -> the 8-class indices it sums.
GROUPS = ((0, 6), (1,), (2,), (3,), (4,), (5,), (7,))


def collapse_probs(probs8: np.ndarray) -> np.ndarray:
    probs8 = np.asarray(probs8)
    if probs8.shape[-1] != len(CLASSES_8):
        raise ValueError(f"expected {len(CLASSES_8)} columns, got {probs8.shape[-1]}")
    return np.stack([probs8[..., list(group)].sum(axis=-1) for group in GROUPS], axis=-1)


def collapse_labels(labels8: np.ndarray) -> np.ndarray:
    lookup = np.empty(len(CLASSES_8), dtype=np.int64)
    for seven, group in enumerate(GROUPS):
        lookup[list(group)] = seven
    return lookup[np.asarray(labels8, dtype=np.int64)]
