"""Non-conformity scores for split conformal prediction.

Every score here is computed as a full **(N, C) matrix** — a score for each candidate
class of each image — rather than only for the true label. Calibration then reads the
true-label column, and prediction-set construction thresholds the whole matrix. Writing
it once this way is what keeps the two halves consistent: a set is exactly the classes
whose score would have fallen below the calibration quantile, which is the definition the
coverage guarantee rests on.

Three scores, in increasing order of how much they care about the tail:

  * **LAC** (`1 - p_y`, Sadinle et al. 2019) produces the smallest possible sets at a
    given coverage, but distributes its errors unevenly: it will happily return an empty
    set for a hard image, which is useless to a clinician.
  * **APS** (Romano et al. 2020) accumulates probability mass down the sorted ranking
    until the true class is reached, so a class only enters the set once everything more
    likely is already in it. Sets are larger but adapt to difficulty, and are never empty.
  * **RAPS** (Angelopoulos et al. 2021) is APS plus a penalty on deep ranks, which stops
    the long flat tail of a 7-class softmax from dragging six classes into every set.

The randomisation term `u` in APS/RAPS is what makes coverage *exact* rather than merely
conservative; it is drawn from a seeded generator so a rerun reproduces the same sets.
"""

from __future__ import annotations

import numpy as np


def lac_scores(probs: np.ndarray) -> np.ndarray:
    """(N, C) matrix of `1 - p_c`. Higher = less conforming."""
    return 1.0 - probs


def aps_scores(
    probs: np.ndarray,
    rng: np.random.Generator,
    randomized: bool = True,
    penalty: float = 0.0,
    k_reg: int = 0,
) -> np.ndarray:
    """(N, C) adaptive prediction-set scores; RAPS when `penalty` > 0.

    For class c the score is the probability mass ranked strictly above c, plus a
    randomised fraction of c's own mass:

        s(x, c) = sum_{j : p_j > p_c} p_j + u * p_c  +  penalty * max(0, rank(c) - k_reg)

    with rank 1-indexed. Setting `penalty` to zero recovers plain APS.

    Args:
        probs: (N, C) calibrated probabilities.
        rng: seeded generator for the randomisation term.
        randomized: False replaces `u` with 1, giving conservative (over-)coverage.
        penalty: RAPS lambda — cost added per rank beyond `k_reg`.
        k_reg: ranks up to and including this are penalty-free.
    """
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumulative = np.cumsum(sorted_probs, axis=1)

    # ranks[i, c] = 0-indexed position of class c in image i's descending ordering
    ranks = np.argsort(order, axis=1, kind="stable")
    cumulative_including = np.take_along_axis(cumulative, ranks, axis=1)

    u = rng.uniform(size=(len(probs), 1)) if randomized else 1.0
    scores = cumulative_including - probs + u * probs

    if penalty > 0.0:
        scores = scores + penalty * np.maximum(0, (ranks + 1) - k_reg)
    return scores


def true_label_scores(score_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The calibration scores: each image's score for the class it actually is."""
    return score_matrix[np.arange(len(labels)), labels]
