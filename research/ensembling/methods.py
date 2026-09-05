"""The five Phase-1 ensembling algorithms, each operating on aligned (N, K, C) tensors.

Parameter-free methods (soft-vote, rank-average) take no fitting step. Fitted methods
(Nelder-Mead weights, non-negative stacking, Caruana greedy selection) expose a
`fit(...)` that returns a small state object and an `apply(state, probs_or_logits)` that
produces (N, C) scores -- this shape lets `research/ensembling/oof.py` reuse the same
functions for honest out-of-fold scoring and for the final val-fit / test-apply pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from ml.evaluation.metrics import compute_metrics


# ---------------------------------------------------------------------------------------
# Parameter-free: soft-voting
# ---------------------------------------------------------------------------------------

def soft_vote_arithmetic(probs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Arithmetic mean of per-class probabilities across architectures. probs: (N, K, C)."""
    w = _normalize_weights(weights, probs.shape[1])
    return np.tensordot(probs, w, axes=([1], [0]))


def soft_vote_geometric(probs: np.ndarray, weights: np.ndarray | None = None, eps: float = 1e-12) -> np.ndarray:
    """Weighted geometric mean (log-space average), renormalized to sum to 1 per row."""
    w = _normalize_weights(weights, probs.shape[1])
    log_probs = np.log(np.clip(probs, eps, 1.0))
    weighted_log = np.tensordot(log_probs, w, axes=([1], [0]))
    combined = np.exp(weighted_log)
    return combined / combined.sum(axis=1, keepdims=True)


def _normalize_weights(weights: np.ndarray | None, k: int) -> np.ndarray:
    if weights is None:
        return np.full(k, 1.0 / k)
    w = np.asarray(weights, dtype=float)
    return w / w.sum()


# ---------------------------------------------------------------------------------------
# Parameter-free: rank averaging
# ---------------------------------------------------------------------------------------

def rank_average(probs: np.ndarray) -> np.ndarray:
    """Per-model, per-class rank transform (over samples) averaged across models.

    Mitigates a single overconfident model dominating the arithmetic mean: every model
    contributes ranks in [0, 1] regardless of how peaked its softmax output is.
    """
    n, k, c = probs.shape
    ranks = np.empty_like(probs)
    for model_idx in range(k):
        for class_idx in range(c):
            column = probs[:, model_idx, class_idx]
            order = column.argsort()
            rank_of = np.empty(n)
            rank_of[order] = np.arange(n)
            ranks[:, model_idx, class_idx] = rank_of / (n - 1) if n > 1 else 0.0
    combined = ranks.mean(axis=1)
    row_sums = combined.sum(axis=1, keepdims=True)
    return combined / np.where(row_sums == 0, 1.0, row_sums)


# ---------------------------------------------------------------------------------------
# Fitted: Nelder-Mead simplex weight optimization
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class NelderMeadState:
    weights: np.ndarray  # (K,), on the simplex


def fit_nelder_mead_weights(probs: np.ndarray, y_true: np.ndarray, seed: int = 42) -> NelderMeadState:
    """Optimize architecture weights w in the simplex to directly maximize val macro-F1.

    Weights are reparameterized as softmax(theta) so the unconstrained Nelder-Mead search
    automatically stays on the simplex (w_k >= 0, sum w_k = 1) without explicit constraints.
    """
    k = probs.shape[1]
    rng = np.random.default_rng(seed)

    def negative_macro_f1(theta: np.ndarray) -> float:
        w = _softmax(theta)
        combined = np.tensordot(probs, w, axes=([1], [0]))
        preds = combined.argmax(axis=1)
        return -compute_metrics(y_true, preds)["macro_f1"]

    theta0 = np.zeros(k)  # uniform weights as the starting point
    result = minimize(negative_macro_f1, theta0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 2000})
    best_theta, best_score = result.x, result.fun

    # Nelder-Mead can land in a local optimum; a few random restarts cheaply cover more
    # of the simplex given only K=6 architectures.
    for _ in range(5):
        theta0 = rng.normal(scale=1.5, size=k)
        result = minimize(negative_macro_f1, theta0, method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 2000})
        if result.fun < best_score:
            best_theta, best_score = result.x, result.fun

    return NelderMeadState(weights=_softmax(best_theta))


def apply_nelder_mead(state: NelderMeadState, probs: np.ndarray) -> np.ndarray:
    return np.tensordot(probs, state.weights, axes=([1], [0]))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


# ---------------------------------------------------------------------------------------
# Fitted: non-negative stacking meta-learner
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class StackingState:
    models: list[Ridge]  # one non-negative Ridge regressor per class, one-vs-rest


def fit_stacking(logits: np.ndarray, y_true: np.ndarray, alpha: float = 1.0) -> StackingState:
    """One-vs-rest non-negative Ridge regression on flattened (K*C) logit features.

    `positive=True` enforces w_k >= 0 per the roadmap's non-negativity constraint --
    a base model is only ever allowed to push a class score up, never down, which keeps
    the meta-learner from learning adversarial cancellations that would not generalize.
    """
    n, k, c = logits.shape
    flat = logits.reshape(n, k * c)
    models = []
    for class_idx in range(c):
        target = (y_true == class_idx).astype(float)
        model = Ridge(alpha=alpha, positive=True)
        model.fit(flat, target)
        models.append(model)
    return StackingState(models=models)


def apply_stacking(state: StackingState, logits: np.ndarray) -> np.ndarray:
    n, k, c = logits.shape
    flat = logits.reshape(n, k * c)
    scores = np.stack([model.predict(flat) for model in state.models], axis=1)
    # The intercept is unconstrained (only w_k >= 0 is enforced), so a class the
    # ensemble is confident *against* can still score slightly negative; clip so the
    # scores stay usable as pseudo-probabilities downstream.
    return np.clip(scores, 0.0, None)


# ---------------------------------------------------------------------------------------
# Fitted: Caruana greedy ensemble selection
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CaruanaState:
    selected: tuple[int, ...]  # architecture indices, with repeats, in pick order
    curve: tuple[float, ...]  # val macro-F1 after each pick, len == len(selected)


def fit_caruana_greedy(probs: np.ndarray, y_true: np.ndarray, max_size: int = 20) -> CaruanaState:
    """Iterative forward selection *with replacement*: repeatedly add whichever member
    (by index, repeats allowed) most improves the running ensemble's macro-F1.
    """
    k = probs.shape[1]
    max_size = min(max_size, k * 4)  # cap the search; k=6 architectures makes 20 picks mostly repeats
    selected: list[int] = []
    running_sum = np.zeros((probs.shape[0], probs.shape[2]))
    curve: list[float] = []

    for _ in range(max_size):
        best_idx, best_score, best_sum = None, -np.inf, None
        for candidate in range(k):
            trial_sum = running_sum + probs[:, candidate, :]
            trial_mean = trial_sum / (len(selected) + 1)
            preds = trial_mean.argmax(axis=1)
            score = compute_metrics(y_true, preds)["macro_f1"]
            if score > best_score:
                best_idx, best_score, best_sum = candidate, score, trial_sum
        selected.append(best_idx)
        running_sum = best_sum
        curve.append(best_score)

    return CaruanaState(selected=tuple(selected), curve=tuple(curve))


def apply_caruana(state: CaruanaState, probs: np.ndarray, size: int | None = None) -> np.ndarray:
    """Apply the ensemble as it stood after `size` picks (default: best point on the curve)."""
    if size is None:
        size = int(np.argmax(state.curve)) + 1
    picks = state.selected[:size]
    return probs[:, picks, :].mean(axis=1)
