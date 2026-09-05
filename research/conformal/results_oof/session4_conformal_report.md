# Phase 4 — Conformal Prediction Sets (Session 4)

Base system: 24-view TTA, uniform soft-vote over 6 backbones, Dirichlet calibration fitted on the OOF tuning half.

Out-of-fold predictions over the 6981 training images (`research/predictions_oof_tta`) split into a tuning half (3513) and a calibration half (3468), grouped by `lesion_id` so no lesion appears in both. RAPS's `k_reg` and `lambda` are chosen on the tuning half only; all three methods take their conformal quantile from the calibration half. Test is locked by `research.testguard`; every number below is measured on val (n=1532).

**Reading the table.** *Marginal coverage* is the guarantee as usually quoted — over all cases. *Coverage on serious* restricts it to lesions that are actually akiec, bcc or mel. *Worst class* is the lowest per-class coverage, which is what a marginal guarantee is free to sacrifice. *False reassurance* counts malignant lesions whose prediction set contained no escalating class at all — the failure mode that matters clinically, and the one an average cannot show.

**Exchangeability caveat — this variant is approximate, not exact.** Split conformal's finite-sample guarantee requires the calibration and test scores to be exchangeable under *one fixed* score function. These quantiles come from out-of-fold scores produced by five different fold models, none of which is the full-train model that scores test, so the guarantee does not transfer as a theorem. What it buys is calibration sample size for the rare classes, which is the binding constraint at alpha=0.05. Achieved coverage must therefore be audited empirically against nominal, and it is reported above rather than asserted. The val-fitted variant keeps the exact guarantee at n=24/22; CV+ / cross-conformal would restore a (1-2*alpha) guarantee here but requires scoring test with all five fold models — a second test read, out of scope under Hard Rule 2 and noted as the rigorous follow-up.

## alpha = 0.10 (target coverage 90%)

RAPS hyperparameters from the tuning half: `k_reg=1`, `lambda=0.2`.

| Method | Calibration | Marginal coverage | Coverage on serious | Worst class | Mean set size | Singletons | Empty | False reassurance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LAC | marginal | 0.9047 | 0.7630 | 0.6818 (vasc) | 1.086 | 91.5% | 0.0% | 68 |
| LAC | class-conditional | 0.8760 | 0.9188 | 0.7727 (vasc) | 1.259 | 79.2% | 0.2% | 20 |
| APS | marginal | 0.8936 | 0.8409 | 0.6667 (df) | 1.268 | 69.6% | 5.0% | 43 |
| APS | class-conditional | 0.8864 | 0.9026 | 0.8750 (df) | 1.435 | 58.6% | 6.7% | 26 |
| RAPS | marginal | 0.9034 | 0.7630 | 0.7083 (df) | 1.110 | 88.8% | 0.1% | 64 |
| RAPS | class-conditional | 0.8884 | 0.9058 | 0.8182 (vasc) | 1.383 | 58.9% | 5.1% | 21 |

Class-conditional calibration re-fitted on the **full** OOF split (hyperparameter-free methods only, so nothing was tuned on it):

| Method | Calibration | Marginal coverage | Coverage on serious | Worst class | Mean set size | Singletons | Empty | False reassurance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LAC | class-conditional | 0.8884 | 0.9123 | 0.6818 (vasc) | 1.272 | 77.7% | 0.0% | 21 |
| APS | class-conditional | 0.8845 | 0.8961 | 0.7273 (vasc) | 1.403 | 58.7% | 6.8% | 27 |

Per-class coverage and set size, class-conditional calibration:

| Class | n | LAC coverage | LAC size | APS coverage | APS size | RAPS coverage | RAPS size |
|---|---:|---:|---:|---:|---:|---:|---:|
| akiec (escalating) | 53 | 0.9811 | 1.72 | 0.9811 | 2.08 | 0.9623 | 1.75 |
| bcc (escalating) | 82 | 0.9146 | 1.65 | 0.9024 | 1.98 | 0.8902 | 1.79 |
| bkl | 160 | 0.8875 | 1.52 | 0.8875 | 1.84 | 0.8875 | 1.73 |
| df | 24 | 0.8750 | 1.42 | 0.8750 | 1.71 | 0.8750 | 1.50 |
| mel (escalating) | 173 | 0.9017 | 1.39 | 0.8786 | 1.72 | 0.8960 | 1.60 |
| nv | 1018 | 0.8635 | 1.13 | 0.8811 | 1.22 | 0.8851 | 1.23 |
| vasc | 22 | 0.7727 | 1.73 | 0.9091 | 2.18 | 0.8182 | 1.68 |

## alpha = 0.05 (target coverage 95%)

RAPS hyperparameters from the tuning half: `k_reg=1`, `lambda=0.05`.

| Method | Calibration | Marginal coverage | Coverage on serious | Worst class | Mean set size | Singletons | Empty | False reassurance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LAC | marginal | 0.9550 | 0.8799 | 0.7500 (df) | 1.315 | 74.0% | 0.0% | 33 |
| LAC | class-conditional | 0.9380 | 0.9545 | 0.9091 (vasc) | 2.229 | 16.5% | 0.0% | 10 |
| APS | marginal | 0.9497 | 0.9058 | 0.7500 (df) | 1.473 | 62.9% | 1.3% | 25 |
| APS | class-conditional | 0.9458 | 0.9545 | 0.9091 (vasc) | 2.258 | 25.4% | 1.6% | 11 |
| RAPS | marginal | 0.9478 | 0.8734 | 0.7500 (df) | 1.359 | 68.6% | 0.0% | 33 |
| RAPS | class-conditional | 0.9491 | 0.9513 | 0.9091 (vasc) | 2.355 | 20.5% | 1.2% | 10 |

Class-conditional calibration re-fitted on the **full** OOF split (hyperparameter-free methods only, so nothing was tuned on it):

| Method | Calibration | Marginal coverage | Coverage on serious | Worst class | Mean set size | Singletons | Empty | False reassurance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LAC | class-conditional | 0.9452 | 0.9675 | 0.9091 (vasc) | 2.251 | 16.8% | 0.0% | 6 |
| APS | class-conditional | 0.9478 | 0.9643 | 0.9091 (vasc) | 2.213 | 29.2% | 1.6% | 9 |

Per-class coverage and set size, class-conditional calibration:

| Class | n | LAC coverage | LAC size | APS coverage | APS size | RAPS coverage | RAPS size |
|---|---:|---:|---:|---:|---:|---:|---:|
| akiec (escalating) | 53 | 0.9811 | 2.15 | 0.9811 | 2.47 | 0.9811 | 2.51 |
| bcc (escalating) | 82 | 0.9512 | 2.18 | 0.9390 | 2.41 | 0.9268 | 2.37 |
| bkl | 160 | 0.9500 | 2.27 | 0.9375 | 2.50 | 0.9375 | 2.44 |
| df | 24 | 0.9583 | 1.75 | 0.9583 | 2.04 | 0.9583 | 1.83 |
| mel (escalating) | 173 | 0.9480 | 2.18 | 0.9538 | 2.35 | 0.9538 | 2.35 |
| nv | 1018 | 0.9312 | 2.24 | 0.9450 | 2.17 | 0.9509 | 2.34 |
| vasc | 22 | 0.9091 | 2.50 | 0.9091 | 2.91 | 0.9091 | 2.50 |

## What the numbers say

**A marginal guarantee does not protect the patients it needs to.** At alpha=0.10, LAC's marginal calibration lands on 90.5% coverage overall — the guarantee holds — while covering only 76.3% of genuinely malignant lesions, and leaving 68 of them with a prediction set containing no escalating class at all. The average is kept afloat by the 67% of cases that are moles. Class-conditional calibration raises coverage on serious cases to 91.9% and cuts false reassurance to 20.

**The price is shortlist length.** The same switch takes LAC's mean set size from 1.09 to 1.26 and its singleton rate from 91.5% to 79.2% — that is, the system commits to a single diagnosis far less often. That is the real trade this phase buys, and it is a trade worth making: a two-class shortlist that contains the melanoma is clinically useful, a confident singleton that does not is not.

**Best configuration by the clinical metric** is `LAC` with class-conditional calibration at alpha=0.10 (20 false reassurances, 91.9% coverage on serious cases, mean set size 1.26).

**What the extra calibration data did and did not fix.** The rare-class scarcity that caps the val-fitted alpha=0.05 numbers is relieved here: the OOF training split carries 71 `df` and 99 `vasc` images against validation's 24 and 22, which is past the n>=19 a class-conditional threshold needs at alpha=0.05. It does not extend to alpha=0.01, which needs n>=99 per class: `df` still fails there even pooling OOF and validation (95 < 99). The remaining set size at alpha=0.05 is therefore model uncertainty rather than calibration scarcity — and the price is the exchangeability caveat above.

## Figure

`class_conditional_coverage.png` — per-class coverage under marginal vs class-conditional calibration at alpha=0.10, against the 90% target line.

## Fitted state

`research/conformal/results_oof/fit_state.json` — every conformal quantile, its per-class calibration count and the RAPS hyperparameters, in the form the single test pass consumes.
