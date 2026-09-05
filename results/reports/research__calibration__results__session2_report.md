# Phase 2 — Calibration, Cost-Sensitive Thresholds & DCA (Session 2)

## Calibration (winning Session-1 ensemble: uniform soft-vote, val-fit, test single-read)

| Method | val ECE | test ECE | test Macro-F1 | test Escalation Sens. |
|---|---:|---:|---:|---:|
| uncalibrated | 0.1547 | 0.1575 | 0.7718 | 0.7828 |
| temperature | 0.0338 | 0.0322 | 0.7718 | 0.7828 |
| matrix_scaling | 0.0263 | 0.0279 | 0.7417 | 0.6862 |
| dirichlet **(best by val ECE)** | 0.0201 | 0.0206 | 0.7907 | 0.7034 |

## Cost-sensitive thresholds (fit on dirichlet-calibrated val probabilities)

- Thresholds: `[-0.3, -0.25, 0.25, 0.0, -0.3, 0.25, -0.3]`
- Val specificity at fit time: 0.8775 (floor: 0.85)

| Decision rule | test expected cost | test Macro-F1 | test Escalation Sens. | test Missed Serious |
|---|---:|---:|---:|---:|
| Uncalibrated argmax | 0.4660 | 0.7718 | 0.7828 | 63 |
| Cost-sensitive thresholds | 0.4171 | 0.7490 | 0.8448 | 45 |

## Decision Curve Analysis

See `research/dca/results/decision_curve.png`. Net benefit at p_t=0.10: ensemble=0.1577, convnext_tiny baseline=0.1559, Treat All=0.1034.
