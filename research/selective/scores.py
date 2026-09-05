"""Uncertainty scores for selective classification.

Every function returns a score where **higher means less trustworthy**, so a single
convention holds throughout Phase 4: abstain on the highest-scoring cases.

Two families are computed:

  * *Predictive* (aleatoric + epistemic mixed) — read off the final probability vector:
    max-softmax-probability, predictive entropy, and the top-two margin. These describe
    how undecided the model is between the classes it knows about.
  * *Epistemic* (disagreement) — only available because the system is an ensemble:
    mutual information (the BALD decomposition) and mean per-class variance across the
    K members. These describe how much the members disagree, which is the part of the
    uncertainty extra data would remove, and it is the signal that fires on inputs
    unlike anything in training.

The feature-space score (Mahalanobis distance) lives in `mahalanobis.py` because it
needs penultimate activations rather than probabilities.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def max_softmax(probs: np.ndarray) -> np.ndarray:
    """1 - max_c p_c. The standard confidence baseline (Hendrycks & Gimpel)."""
    return 1.0 - probs.max(axis=1)


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """H(p) = -sum_c p_c log p_c, in nats."""
    p = np.clip(probs, EPS, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def top_two_margin(probs: np.ndarray) -> np.ndarray:
    """1 - (p_(1) - p_(2)): high when the top two classes are neck and neck."""
    ordered = np.sort(probs, axis=1)
    return 1.0 - (ordered[:, -1] - ordered[:, -2])


def mutual_information(member_probs: np.ndarray) -> np.ndarray:
    """BALD: H(mean_k p_k) - mean_k H(p_k), over a (N, K, C) member tensor.

    The epistemic half of the decomposition. Zero when every member outputs the same
    distribution however unsure that distribution is, so unlike entropy it does not
    penalise a lesion that is genuinely ambiguous to all six backbones alike.
    """
    mean_probs = member_probs.mean(axis=1)
    total = predictive_entropy(mean_probs)
    aleatoric = np.stack(
        [predictive_entropy(member_probs[:, k, :]) for k in range(member_probs.shape[1])], axis=1
    ).mean(axis=1)
    return total - aleatoric


def ensemble_variance(member_probs: np.ndarray) -> np.ndarray:
    """Mean over classes of the across-member variance of p_c, from a (N, K, C) tensor."""
    return member_probs.var(axis=1).mean(axis=1)


def zscore(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Standardise `values` using the mean/std of `reference` (always the val split).

    Combining scores that live on different units — nats, probabilities, squared
    Mahalanobis distances — requires a common scale, and that scale has to be fitted on
    validation data only. Fitting it on the split being scored would let the test set
    influence the abstention rule.
    """
    mean = float(np.mean(reference))
    std = float(np.std(reference))
    if std < EPS:
        return np.zeros_like(values, dtype=np.float64)
    return (values - mean) / std


def combine(components: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> np.ndarray:
    """Sum of val-standardised components — the dual entropy + Mahalanobis policy.

    Args:
        components: score name -> values for the split being scored.
        reference: the same score names -> their values on the validation split.
    """
    missing = set(components) - set(reference)
    if missing:
        raise KeyError(f"no validation reference for component(s): {sorted(missing)}")
    return np.sum([zscore(components[k], reference[k]) for k in sorted(components)], axis=0)


def predictive_scores(mean_probs: np.ndarray, member_probs: np.ndarray) -> dict[str, np.ndarray]:
    """All probability-based scores in one dict, keyed by the name used in reports."""
    return {
        "msp": max_softmax(mean_probs),
        "entropy": predictive_entropy(mean_probs),
        "margin": top_two_margin(mean_probs),
        "mutual_information": mutual_information(member_probs),
        "ensemble_variance": ensemble_variance(member_probs),
    }
