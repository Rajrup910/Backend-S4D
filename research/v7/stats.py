"""V7 statistical instruments: the added-score test and the paired lesion-clustered bootstrap.

Added-score test (owner's solution report, "How each build must be tested"): does a new score s add anything beyond the
deployed ensemble? Fit a logistic model with the ensemble's escalation logit as an OFFSET (coefficient fixed at 1),
logit P(escalating) = offset + b0 + b1 * z(s), and test b1 = 0. Two statistics:

* the likelihood-ratio test (assumes independent rows; reported, not used for the decision);
* the Wald test with a cluster-robust (sandwich) variance, clustered by lesion group, because images of one lesion are not
  independent. This is the decision statistic. DeLong is not used: it is invalid for nested models (Demler et al. 2012).

The bootstrap resamples lesion groups with replacement and scores both systems on the same resample (as V6 did).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats as sps

MAX_ITER = 100
TOL = 1e-10


def _fit_logistic(x: np.ndarray, y: np.ndarray, offset: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Newton-Raphson for logistic regression with an offset. Returns (beta, log-likelihood, Hessian of -ll)."""
    beta = np.zeros(x.shape[1])
    for _ in range(MAX_ITER):
        eta = offset + x @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1 - p)
        grad = x.T @ (y - p)
        hess = (x * w[:, None]).T @ x
        step = np.linalg.solve(hess + 1e-12 * np.eye(len(beta)), grad)
        beta += step
        if np.max(np.abs(step)) < TOL:
            break
    eta = offset + x @ beta
    ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    p = 1.0 / (1.0 + np.exp(-eta))
    hess = (x * (p * (1 - p))[:, None]).T @ x
    return beta, ll, hess


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(q / (1 - q))


def added_score_test(y: np.ndarray, ensemble_prob: np.ndarray, score: np.ndarray, groups: np.ndarray) -> dict:
    """Test whether `score` adds to the ensemble's escalation probability, with lesion-clustered inference.

    y: 1 = escalating. ensemble_prob: the ensemble's escalation mass (offset = its logit). score: the added score, any
    scale (standardised here). groups: lesion group per row."""
    y = np.asarray(y, dtype=np.float64)
    if len(np.unique(y)) != 2:
        raise ValueError("the added-score test needs both escalating and non-escalating rows")
    offset = logit(ensemble_prob)
    s = np.asarray(score, dtype=np.float64)
    if not np.isfinite(s).all() or s.std() == 0:
        raise ValueError("the added score must be finite and not constant")
    z = (s - s.mean()) / s.std()
    x0 = np.ones((len(y), 1))
    x1 = np.column_stack([x0, z])
    _, ll0, _ = _fit_logistic(x0, y, offset)
    beta, ll1, hess = _fit_logistic(x1, y, offset)
    p = 1.0 / (1.0 + np.exp(-(offset + x1 @ beta)))
    scores = x1 * (y - p)[:, None]
    labels, inverse = np.unique(np.asarray(groups), return_inverse=True)
    g = len(labels)
    per_cluster = np.zeros((g, x1.shape[1]))
    np.add.at(per_cluster, inverse, scores)
    bread = np.linalg.inv(hess)
    meat = per_cluster.T @ per_cluster
    robust = bread @ meat @ bread * (g / (g - 1))
    naive_se = float(np.sqrt(bread[1, 1]))
    robust_se = float(np.sqrt(robust[1, 1]))
    lr = max(0.0, 2 * (ll1 - ll0))
    wald_z = float(beta[1] / robust_se)
    return {"n_rows": int(len(y)), "n_clusters": int(g), "n_escalating": int(y.sum()),
            "beta_per_sd": float(beta[1]), "odds_ratio_per_sd": float(np.exp(beta[1])),
            "se_naive": naive_se, "se_cluster_robust": robust_se,
            "lr_statistic": float(lr), "lr_p": float(sps.chi2.sf(lr, 1)),
            "wald_z_cluster": wald_z, "wald_p_cluster": float(2 * sps.norm.sf(abs(wald_z))),
            "ci95_beta_cluster": [float(beta[1] - 1.959964 * robust_se), float(beta[1] + 1.959964 * robust_se)],
            "decision_statistic": "wald_p_cluster", "adds_signal": bool(beta[1] > 0 and 2 * sps.norm.sf(abs(wald_z)) < 0.05)}


def cluster_index(groups: np.ndarray) -> list[np.ndarray]:
    labels, inverse = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.cumsum(np.bincount(inverse, minlength=len(labels)))[:-1]
    return np.split(order, bounds)


def paired_cluster_bootstrap(metric: Callable[[np.ndarray], float], a_metric: Callable[[np.ndarray], float],
                             groups: np.ndarray, n_boot: int = 2000, seed: int = 20261110) -> dict:
    """Delta metric(rows) - a_metric(rows) on lesion-group resamples. `metric` and `a_metric` take an index array and score
    the candidate and the comparator on exactly those rows, so the pair is always scored together."""
    clusters = cluster_index(groups)
    all_rows = np.concatenate(clusters)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        pick = np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
        deltas[i] = metric(pick) - a_metric(pick)
    point = metric(all_rows) - a_metric(all_rows)
    finite = deltas[np.isfinite(deltas)]
    tail = min((np.sum(finite <= 0) + 1) / (len(finite) + 1), (np.sum(finite >= 0) + 1) / (len(finite) + 1)) if len(finite) else float("nan")
    return {"delta": float(point), "ci95": [float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5))],
            "sd": float(np.nanstd(deltas)), "n_boot": int(n_boot), "n_clusters": int(len(clusters)), "seed": int(seed),
            "p_boot": float(min(1.0, 2 * tail)), "replicates_not_estimable": int(n_boot - len(finite))}


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values for one declared family. A member without a p-value (NaN) stays NaN and still counts in the
    family size: the family is what was declared, not what happened to be estimable (the conservative reading)."""
    m = len(p_values)
    known = sorted((p, k) for k, p in p_values.items() if np.isfinite(p))
    out, running = {k: float("nan") for k in p_values}, 0.0
    for rank, (p, k) in enumerate(known):
        running = max(running, min(1.0, (m - rank) * p))
        out[k] = float(running)
    return out
