# Phase 4 — Conformal Prediction Sets (Session 4)

Base system: 24-view TTA, uniform soft-vote over 6 backbones, Dirichlet calibration fitted on val — identical to the selective-classification half of this session, so the two sets of numbers describe the same deployed model.

Validation split into a tuning half (770 images) and a calibration half (762), grouped by `lesion_id` so no lesion appears in both. RAPS's `k_reg` and `lambda` are chosen on the tuning half only; all three methods take their conformal quantile from the calibration half. Test is locked by `research.testguard`; every number below is measured on val (n=1532).

**Reading the table.** *Marginal coverage* is the guarantee as usually quoted — over all cases. *Coverage on serious* restricts it to lesions that are actually akiec, bcc or mel. *Worst class* is the lowest per-class coverage, which is what a marginal guarantee is free to sacrifice. *False reassurance* counts malignant lesions whose prediction set contained no escalating class at all — the failure mode that matters clinically, and the one an average cannot show.

## alpha = 0.10 (target coverage 90%)

RAPS hyperparameters from the tuning half: `k_reg=1`, `lambda=0.2`.

| Method | Calibration | Marginal coverage | Coverage on serious | Worst class | Mean set size | Singletons | Empty | False reassurance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LAC | marginal | 0.9034 | 0.7338 | 0.6604 (akiec) | 1.074 | 92.6% | 0.0% | 75 |
| LAC | class-conditional | 0.9060 | 0.9448 | 0.8861 (nv) | 2.084 | 23.7% | 0.0% | 16 |
| APS | marginal | 0.9145 | 0.8669 | 0.7500 (df) | 1.326 | 67.8% | 3.9% | 37 |
| APS | class-conditional | 0.9236 | 0.9383 | 0.9017 (mel) | 1.995 | 36.4% | 4.8% | 18 |
| RAPS | marginal | 0.9080 | 0.7500 | 0.6763 (mel) | 1.105 | 88.6% | 0.5% | 72 |
| RAPS | class-conditional | 0.9230 | 0.9545 | 0.9096 (nv) | 2.608 | 13.1% | 0.2% | 11 |

Class-conditional calibration re-fitted on the **full** validation split (hyperparameter-free methods only, so nothing was tuned on it):

| Method | Calibration | Marginal coverage | Coverage on serious | Worst class | Mean set size | Singletons | Empty | False reassurance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LAC | class-conditional | 0.9106 | 0.9221 | 0.9028 (nv) | 2.830 | 5.8% | 0.0% | 20 |
| APS | class-conditional | 0.9106 | 0.9221 | 0.9028 (nv) | 2.733 | 9.9% | 0.2% | 21 |

Per-class coverage and set size, class-conditional calibration:

| Class | n | LAC coverage | LAC size | APS coverage | APS size | RAPS coverage | RAPS size |
|---|---:|---:|---:|---:|---:|---:|---:|
| akiec (escalating) | 53 | 0.9623 | 2.47 | 0.9811 | 2.62 | 0.9811 | 2.42 |
| bcc (escalating) | 82 | 0.9878 | 1.54 | 0.9878 | 1.70 | 0.9878 | 2.06 |
| bkl | 160 | 0.9375 | 2.26 | 0.9187 | 2.56 | 0.9313 | 2.49 |
| df | 24 | 0.9583 | 2.29 | 0.9583 | 2.79 | 0.9583 | 2.38 |
| mel (escalating) | 173 | 0.9191 | 2.02 | 0.9017 | 2.25 | 0.9306 | 2.45 |
| nv | 1018 | 0.8861 | 2.10 | 0.9175 | 1.83 | 0.9096 | 2.73 |
| vasc | 22 | 1.0000 | 1.64 | 1.0000 | 2.05 | 1.0000 | 1.86 |

## alpha = 0.05 (target coverage 95%)

RAPS hyperparameters from the tuning half: `k_reg=1`, `lambda=0.05`.

| Method | Calibration | Marginal coverage | Coverage on serious | Worst class | Mean set size | Singletons | Empty | False reassurance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LAC | marginal | 0.9510 | 0.8636 | 0.7083 (df) | 1.289 | 75.7% | 0.0% | 38 |
| LAC | class-conditional | 0.9523 | 0.9805 | 0.9381 (nv) | 3.574 | 0.0% | 0.0% | 4 |
| APS | marginal | 0.9537 | 0.9026 | 0.7500 (df) | 1.513 | 61.6% | 1.3% | 27 |
| APS | class-conditional | 0.9700 | 0.9805 | 0.9627 (nv) | 3.846 | 0.0% | 0.0% | 5 |
| RAPS | marginal | 0.9530 | 0.8701 | 0.7083 (df) | 1.376 | 67.5% | 0.0% | 34 |
| RAPS | class-conditional | 0.9667 | 0.9805 | 0.9587 (nv) | 3.852 | 0.0% | 0.0% | 5 |

Class-conditional calibration re-fitted on the **full** validation split (hyperparameter-free methods only, so nothing was tuned on it):

| Method | Calibration | Marginal coverage | Coverage on serious | Worst class | Mean set size | Singletons | Empty | False reassurance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LAC | class-conditional | 0.9595 | 0.9740 | 0.9528 (nv) | 4.012 | 3.2% | 0.0% | 0 |
| APS | class-conditional | 0.9595 | 0.9740 | 0.9528 (nv) | 3.888 | 2.7% | 0.0% | 0 |

Per-class coverage and set size, class-conditional calibration:

| Class | n | LAC coverage | LAC size | APS coverage | APS size | RAPS coverage | RAPS size |
|---|---:|---:|---:|---:|---:|---:|---:|
| akiec (escalating) | 53 | 0.9623 | 4.30 | 0.9811 | 4.57 | 0.9811 | 4.30 |
| bcc (escalating) | 82 | 1.0000 | 3.46 | 1.0000 | 3.85 | 1.0000 | 3.98 |
| bkl | 160 | 0.9750 | 3.85 | 0.9875 | 4.38 | 0.9812 | 4.09 |
| df | 24 | 1.0000 | 3.71 | 1.0000 | 4.12 | 1.0000 | 3.62 |
| mel (escalating) | 173 | 0.9769 | 3.97 | 0.9711 | 4.37 | 0.9711 | 4.22 |
| nv | 1018 | 0.9381 | 3.45 | 0.9627 | 3.65 | 0.9587 | 3.75 |
| vasc | 22 | 1.0000 | 2.82 | 1.0000 | 3.05 | 1.0000 | 2.77 |

## What the numbers say

**A marginal guarantee does not protect the patients it needs to.** At alpha=0.10, LAC's marginal calibration lands on 90.3% coverage overall — the guarantee holds — while covering only 73.4% of genuinely malignant lesions, and leaving 75 of them with a prediction set containing no escalating class at all. The average is kept afloat by the 67% of cases that are moles. Class-conditional calibration raises coverage on serious cases to 94.5% and cuts false reassurance to 16.

**The price is shortlist length.** The same switch takes LAC's mean set size from 1.07 to 2.08 and its singleton rate from 92.6% to 23.7% — that is, the system commits to a single diagnosis far less often. That is the real trade this phase buys, and it is a trade worth making: a two-class shortlist that contains the melanoma is clinically useful, a confident singleton that does not is not.

**Best configuration by the clinical metric** is `RAPS` with class-conditional calibration at alpha=0.10 (11 false reassurances, 95.5% coverage on serious cases, mean set size 2.61).

**Caveat that limits how far the alpha=0.05 numbers can be pushed.** `df` and `vasc` hold only ~11 images in the calibration half and ~22 in full validation, which is at or below the point where a class-conditional threshold can be certified at all. Their thresholds therefore sit at or near the maximum observed score, so they enter almost every prediction set regardless of what the model believes. Part of the set size at alpha=0.05 is that scarcity, not genuine model uncertainty — the honest fix is more calibration data for the rare classes (OOF predictions over the training split, `--fit-split oof`, raise `df` to 71 and `vasc` to 99), not a smaller alpha.

## Classes the calibration set cannot certify

A class-conditional threshold needs `ceil((n+1)(1-alpha)) <= n` calibration points of that class. Below that no finite threshold carries the guarantee, so the class is always included in the set — a conservative, honest fallback rather than a silently narrower claim:

* alpha=0.05 LAC class-conditional: df, vasc (n=[11, 11]) cannot certify a finite threshold — those classes are always included.
* alpha=0.05 APS class-conditional: df, vasc (n=[11, 11]) cannot certify a finite threshold — those classes are always included.
* alpha=0.05 RAPS class-conditional: df, vasc (n=[11, 11]) cannot certify a finite threshold — those classes are always included.

## Figure

`class_conditional_coverage.png` — per-class coverage under marginal vs class-conditional calibration at alpha=0.10, against the 90% target line.

## Fitted state

`research/conformal/results_valfit/fit_state.json` — every conformal quantile, its per-class calibration count and the RAPS hyperparameters, in the form the single test pass consumes.
