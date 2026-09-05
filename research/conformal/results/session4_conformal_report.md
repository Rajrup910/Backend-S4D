# Phase 4 — Conformal Prediction Sets (Session 4)

Base system: 24-view TTA, uniform soft-vote over 6 backbones, Dirichlet calibration fitted on val — identical to the selective-classification half of this session, so the two sets of numbers describe the same deployed model.

Validation split into a tuning half (770 images) and a calibration half (762), grouped by `lesion_id` so no lesion appears in both. RAPS's `k_reg` and `lambda` are chosen on the tuning half only; all three methods take their conformal quantile from the calibration half, so the comparison is like-for-like. Test (n=1502) is scored once.

**Reading the table.** *Marginal coverage* is the guarantee as usually quoted — over all cases. *Coverage on serious* restricts it to lesions that are actually akiec, bcc or mel. *Worst class* is the lowest per-class coverage, which is what a marginal guarantee is free to sacrifice. *False reassurance* counts malignant lesions whose prediction set contained no escalating class at all — the failure mode that matters clinically, and the one an average cannot show.

## alpha = 0.10 (target coverage 90%)

RAPS hyperparameters from the tuning half: `k_reg=1`, `lambda=0.2`.

| Method | Calibration | Marginal coverage | Coverage on serious | Worst class | Mean set size | Singletons | Empty | False reassurance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LAC | marginal | 0.9041 | 0.7483 | 0.6707 (mel) | 1.082 | 91.5% | 0.1% | 63 |
| LAC | class-conditional | 0.8995 | 0.9138 | 0.8846 (akiec) | 2.158 | 18.6% | 0.0% | 17 |
| APS | marginal | 0.9188 | 0.8379 | 0.7964 (mel) | 1.340 | 67.8% | 3.6% | 39 |
| APS | class-conditional | 0.9314 | 0.9448 | 0.9102 (bkl) | 2.058 | 36.0% | 4.2% | 11 |
| RAPS | marginal | 0.8995 | 0.7517 | 0.6647 (mel) | 1.115 | 87.4% | 0.6% | 63 |
| RAPS | class-conditional | 0.9268 | 0.9414 | 0.9193 (nv) | 2.680 | 11.5% | 0.1% | 6 |

Class-conditional calibration re-fitted on the **full** validation split (hyperparameter-free methods only, so nothing was tuned on it):

| Method | Calibration | Marginal coverage | Coverage on serious | Worst class | Mean set size | Singletons | Empty | False reassurance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LAC | class-conditional | 0.9055 | 0.8966 | 0.8846 (akiec) | 2.902 | 5.6% | 0.0% | 20 |
| APS | class-conditional | 0.9201 | 0.9310 | 0.9102 (bkl) | 2.780 | 8.7% | 0.1% | 13 |

Per-class coverage and set size, class-conditional calibration:

| Class | n | LAC coverage | LAC size | APS coverage | APS size | RAPS coverage | RAPS size |
|---|---:|---:|---:|---:|---:|---:|---:|
| akiec (escalating) | 52 | 0.8846 | 2.02 | 0.9808 | 2.42 | 0.9615 | 2.19 |
| bcc (escalating) | 71 | 0.9859 | 1.75 | 0.9859 | 2.03 | 0.9718 | 2.17 |
| bkl | 167 | 0.9222 | 2.57 | 0.9102 | 2.85 | 0.9281 | 2.77 |
| df | 20 | 1.0000 | 2.15 | 1.0000 | 2.45 | 1.0000 | 2.15 |
| mel (escalating) | 167 | 0.8922 | 2.28 | 0.9162 | 2.59 | 0.9222 | 2.74 |
| nv | 1004 | 0.8875 | 2.12 | 0.9283 | 1.82 | 0.9193 | 2.75 |
| vasc | 21 | 1.0000 | 1.38 | 1.0000 | 1.52 | 1.0000 | 1.48 |

## alpha = 0.05 (target coverage 95%)

RAPS hyperparameters from the tuning half: `k_reg=1`, `lambda=0.05`.

| Method | Calibration | Marginal coverage | Coverage on serious | Worst class | Mean set size | Singletons | Empty | False reassurance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LAC | marginal | 0.9481 | 0.8483 | 0.7964 (mel) | 1.301 | 74.9% | 0.0% | 35 |
| LAC | class-conditional | 0.9461 | 0.9517 | 0.8846 (akiec) | 3.593 | 0.0% | 0.0% | 5 |
| APS | marginal | 0.9541 | 0.9000 | 0.8623 (mel) | 1.523 | 61.7% | 1.4% | 23 |
| APS | class-conditional | 0.9674 | 0.9724 | 0.9622 (nv) | 3.887 | 0.0% | 0.0% | 5 |
| RAPS | marginal | 0.9581 | 0.8828 | 0.8383 (mel) | 1.397 | 65.7% | 0.0% | 24 |
| RAPS | class-conditional | 0.9627 | 0.9621 | 0.9521 (mel) | 3.871 | 0.0% | 0.0% | 4 |

Class-conditional calibration re-fitted on the **full** validation split (hyperparameter-free methods only, so nothing was tuned on it):

| Method | Calibration | Marginal coverage | Coverage on serious | Worst class | Mean set size | Singletons | Empty | False reassurance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LAC | class-conditional | 0.9594 | 0.9586 | 0.9341 (mel) | 4.073 | 3.6% | 0.0% | 0 |
| APS | class-conditional | 0.9634 | 0.9690 | 0.9521 (mel) | 3.933 | 2.0% | 0.0% | 0 |

Per-class coverage and set size, class-conditional calibration:

| Class | n | LAC coverage | LAC size | APS coverage | APS size | RAPS coverage | RAPS size |
|---|---:|---:|---:|---:|---:|---:|---:|
| akiec (escalating) | 52 | 0.8846 | 4.12 | 0.9808 | 4.56 | 0.9615 | 4.27 |
| bcc (escalating) | 71 | 0.9859 | 3.59 | 0.9859 | 3.89 | 0.9859 | 3.89 |
| bkl | 167 | 0.9760 | 4.05 | 0.9820 | 4.65 | 0.9760 | 4.20 |
| df | 20 | 1.0000 | 3.15 | 1.0000 | 3.50 | 1.0000 | 3.15 |
| mel (escalating) | 167 | 0.9581 | 4.08 | 0.9641 | 4.63 | 0.9521 | 4.41 |
| nv | 1004 | 0.9373 | 3.44 | 0.9622 | 3.64 | 0.9592 | 3.75 |
| vasc | 21 | 1.0000 | 2.48 | 1.0000 | 2.67 | 1.0000 | 2.48 |

## What the numbers say

**A marginal guarantee does not protect the patients it needs to.** At alpha=0.10, LAC's marginal calibration lands on 90.4% coverage overall — the guarantee holds — while covering only 74.8% of genuinely malignant lesions, and leaving 63 of them with a prediction set containing no escalating class at all. The average is kept afloat by the 67% of cases that are moles. Class-conditional calibration raises coverage on serious cases to 91.4% and cuts false reassurance to 17.

**The price is shortlist length.** The same switch takes LAC's mean set size from 1.08 to 2.16 and its singleton rate from 91.5% to 18.6% — that is, the system commits to a single diagnosis far less often. That is the real trade this phase buys, and it is a trade worth making: a two-class shortlist that contains the melanoma is clinically useful, a confident singleton that does not is not.

**Best configuration by the clinical metric** is `RAPS` with class-conditional calibration at alpha=0.10 (6 false reassurances, 94.1% coverage on serious cases, mean set size 2.68).

**Caveat that limits how far the alpha=0.05 numbers can be pushed.** `df` and `vasc` hold only ~11 images in the calibration half and ~22 in full validation, which is at or below the point where a class-conditional threshold can be certified at all. Their thresholds therefore sit at or near the maximum observed score, so they enter almost every prediction set regardless of what the model believes. Part of the set size at alpha=0.05 is that scarcity, not genuine model uncertainty — the honest fix is more calibration data for the rare classes (OOF predictions over the training split would roughly quintuple it), not a smaller alpha.

## Classes the calibration set cannot certify

A class-conditional threshold needs `ceil((n+1)(1-alpha)) <= n` calibration points of that class. Below that no finite threshold carries the guarantee, so the class is always included in the set — a conservative, honest fallback rather than a silently narrower claim:

* alpha=0.05 LAC class-conditional: df, vasc (n=[11, 11]) cannot certify a finite threshold — those classes are always included.
* alpha=0.05 APS class-conditional: df, vasc (n=[11, 11]) cannot certify a finite threshold — those classes are always included.
* alpha=0.05 RAPS class-conditional: df, vasc (n=[11, 11]) cannot certify a finite threshold — those classes are always included.

## Figure

`class_conditional_coverage.png` — per-class coverage under marginal vs class-conditional calibration at alpha=0.10, against the 90% target line.
