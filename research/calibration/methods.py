"""Multi-class probability calibration: (scalar) temperature, matrix scaling, Dirichlet.

All three fit a mapping from the ensemble's *raw* scores to *calibrated* probabilities on
validation data only (never test), via `torch.optim.LBFGS` minimizing NLL, matching the
temperature-scaling implementation already used in `ml/evaluation/evaluate.py`.

- **Temperature scaling** (Guo et al., 2017): one scalar T. Cannot change argmax, so it
  never touches accuracy/F1 -- purely a confidence-honesty fix.
- **Matrix scaling** (Guo et al., 2017): a full C x C affine map on logits, L2-regularized
  (it has C^2 + C parameters against a ~1500-row validation set, so regularization is not
  optional -- unregularized matrix scaling reliably overfits on splits this size).
- **Dirichlet calibration** (Kull et al., NeurIPS 2019): the same affine map, but applied
  to *log-probabilities* rather than raw logits, with ODIR regularization (the L2 penalty
  applies only to the off-diagonal weights, leaving the diagonal and bias free) -- this is
  what makes it a calibration method distinct from plain matrix scaling rather than a
  reparameterization of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class CalibrationState:
    method: str
    weight: np.ndarray  # (C, C), identity for temperature scaling
    bias: np.ndarray  # (C,)
    temperature: float  # only meaningful for method="temperature"


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32).copy())


def fit_temperature_scaling(logits: np.ndarray, y_true: np.ndarray, max_iter: int = 200) -> CalibrationState:
    num_classes = logits.shape[1]
    logits_t, y_t = _to_tensor(logits), torch.from_numpy(y_true).long()
    log_temperature = torch.zeros(1, requires_grad=True)

    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits_t / log_temperature.exp(), y_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.exp().item())
    return CalibrationState(
        method="temperature",
        weight=np.eye(num_classes, dtype=np.float32) / temperature,
        bias=np.zeros(num_classes, dtype=np.float32),
        temperature=temperature,
    )


def _fit_affine_map(
    features: np.ndarray,
    y_true: np.ndarray,
    method: str,
    l2_lambda: float,
    off_diagonal_only: bool,
    max_iter: int = 300,
    lr: float = 0.01,
) -> CalibrationState:
    """Shared fitting routine for matrix scaling (logit features) and Dirichlet (log-prob features)."""
    n, c = features.shape
    x = _to_tensor(features)
    y = torch.from_numpy(y_true).long()

    weight = nn.Parameter(torch.eye(c))
    bias = nn.Parameter(torch.zeros(c))
    optimizer = torch.optim.LBFGS([weight, bias], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    criterion = nn.CrossEntropyLoss()

    off_diag_mask = 1.0 - torch.eye(c)

    def closure():
        optimizer.zero_grad()
        calibrated_logits = x @ weight.T + bias
        loss = criterion(calibrated_logits, y)
        if off_diagonal_only:
            reg = l2_lambda * ((weight - torch.eye(c)) * off_diag_mask).pow(2).sum()
        else:
            reg = l2_lambda * weight.pow(2).sum()
        total = loss + reg
        total.backward()
        return total

    optimizer.step(closure)
    return CalibrationState(
        method=method,
        weight=weight.detach().numpy(),
        bias=bias.detach().numpy(),
        temperature=float("nan"),
    )


def fit_matrix_scaling(logits: np.ndarray, y_true: np.ndarray, l2_lambda: float = 0.01) -> CalibrationState:
    return _fit_affine_map(logits, y_true, method="matrix_scaling", l2_lambda=l2_lambda, off_diagonal_only=False)


def fit_dirichlet_calibration(
    probabilities: np.ndarray, y_true: np.ndarray, l2_lambda: float = 0.1, eps: float = 1e-12
) -> CalibrationState:
    log_probs = np.log(np.clip(probabilities, eps, None))
    return _fit_affine_map(log_probs, y_true, method="dirichlet", l2_lambda=l2_lambda, off_diagonal_only=True)


def apply_calibration(state: CalibrationState, features: np.ndarray) -> np.ndarray:
    """features: logits for temperature/matrix scaling, log-probabilities for Dirichlet."""
    calibrated_logits = features @ state.weight.T + state.bias
    calibrated_logits = calibrated_logits - calibrated_logits.max(axis=1, keepdims=True)
    exp = np.exp(calibrated_logits)
    return exp / exp.sum(axis=1, keepdims=True)
