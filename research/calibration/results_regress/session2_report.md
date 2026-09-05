# Phase 2 — Calibration, Cost-Sensitive Thresholds & DCA (Session 2)

## Calibration (winning Session-1 ensemble: uniform soft-vote, val-fit, test not read)

| Method | val ECE | — | val Macro-F1 | val Escalation Sens. |
|---|---:|---:|---:|---:|
| uncalibrated | 0.1547 | — | 0.7911 | 0.7760 |
| temperature | 0.0338 | — | 0.7911 | 0.7760 |
| matrix_scaling | 0.0263 | — | 0.7554 | 0.7110 |
| dirichlet **(best by val ECE)** | 0.0201 | — | 0.7825 | 0.7208 |

## Cost-sensitive thresholds (fit on dirichlet-calibrated val probabilities)

- Thresholds: `[-0.3, -0.25, 0.25, 0.0, -0.3, 0.25, -0.3]`
- Val specificity at fit time: 0.8775 (floor: 0.85)

- Expected clinical cost on the fit split: 0.3489

## Test-set quantities

Not computed. This is a fit-only run (`--no-test`): the test split is locked by `research.testguard` so that every test quantity in this round is emitted by the single pre-registered pass, not accumulated incrementally. Decision Curve Analysis is a test-set quantity and is therefore also deferred.

## Fit and selection splits

Coefficients and thresholds fitted on **val** (`research/predictions`, n=1532); the calibrator *family* was selected by ECE on **validation** (n=1532), which under `--fit-split oof` is data no coefficient here has seen. Selecting the family on the same OOF rows the coefficients were fitted on would relocate the in-sample problem rather than remove it.

## Fitted state

`research/calibration/results_regress/fit_state.json` — the calibrator coefficients and decision thresholds this run produced, in the form the single test pass consumes.
