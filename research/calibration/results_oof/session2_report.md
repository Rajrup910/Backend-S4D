# Phase 2 — Calibration, Cost-Sensitive Thresholds & DCA (Session 2)

## Calibration (winning Session-1 ensemble: uniform soft-vote, OOF-fit over 6981 train rows (research/predictions_oof), test not read)

| Method | val ECE | — | val Macro-F1 | val Escalation Sens. |
|---|---:|---:|---:|---:|
| uncalibrated | 0.1547 | — | 0.7911 | 0.7760 |
| temperature | 0.0336 | — | 0.7911 | 0.7760 |
| matrix_scaling | 0.0241 | — | 0.7016 | 0.6851 |
| dirichlet **(best by val ECE)** | 0.0210 | — | 0.7719 | 0.7143 |

## Cost-sensitive thresholds (fit on dirichlet-calibrated oof probabilities)

- Thresholds: `[-0.3, -0.3, 0.3, 0.3, -0.3, 0.3, -0.25]`
- OOF specificity at fit time: 0.8626 (floor: 0.85)

- Expected clinical cost on the fit split: 0.3601

## Test-set quantities

Not computed. This is a fit-only run (`--no-test`): the test split is locked by `research.testguard` so that every test quantity in this round is emitted by the single pre-registered pass, not accumulated incrementally. Decision Curve Analysis is a test-set quantity and is therefore also deferred.

## Fit and selection splits

Coefficients and thresholds fitted on **oof** (`research/predictions_oof`, n=6981); the calibrator *family* was selected by ECE on **validation** (n=1532), which under `--fit-split oof` is data no coefficient here has seen. Selecting the family on the same OOF rows the coefficients were fitted on would relocate the in-sample problem rather than remove it.

## Fitted state

`research/calibration/results_oof/fit_state.json` — the calibrator coefficients and decision thresholds this run produced, in the form the single test pass consumes.
