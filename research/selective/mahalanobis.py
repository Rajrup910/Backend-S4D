"""Class-conditional Mahalanobis distance in feature space (Lee et al., NeurIPS 2018).

One Gaussian per class with a **single tied covariance** shared across classes, fitted on
the training split's penultimate features. The uncertainty score for an image is the
smallest squared Mahalanobis distance to any class centroid: large when the embedding
sits nowhere near the training manifold, which is exactly the case where a softmax
probability is confident and wrong.

Why tied rather than per-class: with 7 classes, 768 feature dimensions and as few as ~80
training images in `df`, a per-class covariance is rank-deficient and its inverse is
noise. Pooling gives every class the same well-conditioned shape estimate, and the
Ledoit-Wolf shrinkage on top guarantees invertibility without hand-tuned ridge terms.

The fit sees training data only. Validation features set the abstention threshold and
test features are scored once — so nothing the threshold depends on has been fitted on
the split it will be applied to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class MahalanobisState:
    """Fitted centroids and the shared precision matrix."""

    means: np.ndarray       # (C, D)
    precision: np.ndarray   # (D, D)
    class_indices: tuple[int, ...]
    shrinkage: float
    num_fit_samples: int


def fit(features: np.ndarray, labels: np.ndarray, num_classes: int) -> MahalanobisState:
    """Fit class means and one shrunk tied covariance on training features."""
    if features.ndim != 2:
        raise ValueError(f"features must be (N, D), got {features.shape}")
    if len(features) != len(labels):
        raise ValueError(f"features/labels length mismatch: {len(features)} vs {len(labels)}")

    features = features.astype(np.float64)
    present = [c for c in range(num_classes) if np.any(labels == c)]
    if len(present) < num_classes:
        missing = sorted(set(range(num_classes)) - set(present))
        raise ValueError(f"training split has no examples of class index/indices {missing}")

    means = np.stack([features[labels == c].mean(axis=0) for c in present], axis=0)

    # Centre every sample on its own class mean, then pool: this is the tied
    # within-class scatter, the same quantity LDA estimates.
    centred = features - means[[present.index(int(c)) for c in labels]]
    estimator = LedoitWolf(assume_centered=True).fit(centred)

    return MahalanobisState(
        means=means,
        precision=estimator.precision_,
        class_indices=tuple(present),
        shrinkage=float(estimator.shrinkage_),
        num_fit_samples=int(len(features)),
    )


def score(state: MahalanobisState, features: np.ndarray) -> np.ndarray:
    """Minimum squared Mahalanobis distance to any class centroid. Higher = more OOD."""
    x = features.astype(np.float64)
    distances = np.empty((len(x), len(state.means)), dtype=np.float64)
    for i, mean in enumerate(state.means):
        delta = x - mean
        distances[:, i] = np.einsum("ij,jk,ik->i", delta, state.precision, delta)
    return distances.min(axis=1)


def align_to(image_ids: np.ndarray, feature_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Reorder per-image `values` from feature order into prediction-matrix order.

    The feature cache is written in dataset order while the prediction matrices are
    sorted by image_id, so the two must be joined explicitly rather than assumed aligned.
    """
    lookup = {str(image_id): i for i, image_id in enumerate(feature_ids)}
    missing = [str(i) for i in image_ids if str(i) not in lookup]
    if missing:
        raise KeyError(f"{len(missing)} image(s) have no cached features, e.g. {missing[:3]}")
    return values[[lookup[str(i)] for i in image_ids]]
