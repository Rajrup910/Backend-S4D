# Session 5 — age-conditional escalation rule (lambda)

`fit on oof (train split, research/predictions_oof_tta, TTA=True); selection on val (research/predictions_tta); test=LOCKED; out=research/agerule/results_oof; session=session6_oof`

## 1. Is this a decision-rule failure or a representation failure?

Escalation-mass AUC is threshold-free: it asks only whether the model ranks
escalating lesions above benign ones inside a band. A high AUC beside a low argmax
sensitivity means the separation exists and the decision rule is discarding it.

| band | n | n_escalating | escalating_prior | argmax_sens | missed | escalation_mass_auc | referral_rate | auc_ci_lo | auc_ci_hi |
|---|---|---|---|---|---|---|---|---|---|
| <40 | 1319 | 64 | 0.0485 | 0.5469 | 29 | 0.8893 | 0.0523 | 0.8305 | 0.9491 |
| 40-59 | 3112 | 397 | 0.1276 | 0.6121 | 154 | 0.9525 | 0.1019 | 0.9362 | 0.9652 |
| 60+ | 2512 | 893 | 0.3555 | 0.7335 | 238 | 0.9335 | 0.3177 | 0.9223 | 0.9451 |
| unknown | 38 | 2 | 0.0526 | 1.0000 | 0 | 1.0000 | 0.1053 | 1.0000 | 1.0000 |
| ALL | 6981 | 1356 | 0.1942 | 0.6895 | 421 | 0.9487 | 0.1702 | 0.9405 | 0.9564 |

The <40 band has the lowest argmax sensitivity (0.547 against 0.612-0.733) **and** the lowest escalation-mass AUC (0.889, 95% CI 0.831-0.949). Both effects are present, so this is **not** a pure decision-rule failure.

Part of the gap is a threshold that a lambda can move, and part of it is genuinely weaker separation in this band, which no threshold can recover. The AUC interval overlaps 40-59, 60+, so the ranking difference is suggestive rather than established on 64 escalating cases.

This bounds what section 4 can claim: the rule should help in <40 and should help *less* than in bands where the ranking is stronger.

## 2. Fitted lambda per band

Pooled lambda = **0.65**. Fitted on cross-fitted calibrated
probabilities; the same fit on in-sample calibrated probabilities gives
0.65, which is the size of the optimism cross-fitting removes.

| group | lam | n | n_positive | fit_cost | fit_sensitivity | fit_specificity | fit_referral_rate | baseline_cost | baseline_sensitivity | baseline_specificity | baseline_referral_rate | fitted | lambda_if_not_crossfitted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 40-59 | 0.7400 | 3112 | 397 | 0.2522 | 0.8967 | 0.8707 | 0.2272 | 0.5215 | 0.6121 | 0.9727 | 0.1019 | True | 0.7300 |
| 60+ | 0.3300 | 2512 | 893 | 0.6636 | 0.8331 | 0.8511 | 0.3921 | 0.9335 | 0.7335 | 0.9117 | 0.3177 | True | 0.3300 |
| <40 | 0.2600 | 1319 | 64 | 0.2324 | 0.6250 | 0.9522 | 0.0758 | 0.2483 | 0.5469 | 0.9729 | 0.0523 | True | 0.2400 |
| unknown | 0.6500 | 38 | 2 | 0.1053 | 1.0000 | 0.8889 | 0.1579 | 0.0526 | 1.0000 | 0.9444 | 0.1053 | False | 0.6500 |

## 3. One parameter, not seven

| band | lambda | boot_mean | boot_std | ci_lo | ci_hi | share_at_zero | n_boot |
|---|---|---|---|---|---|---|---|
| 40-59 | 0.7400 | 0.7471 | 0.0344 | 0.6700 | 0.8200 | 0.0000 | 400.0000 |
| 60+ | 0.3300 | 0.3314 | 0.0614 | 0.2200 | 0.4400 | 0.0000 | 400.0000 |
| <40 | 0.2600 | 0.2397 | 0.1531 | 0.0000 | 0.6100 | 0.1025 | 400.0000 |

**The lambda for the band this session exists for is the least certain one.** lambda(<40) = 0.26 with a lesion-grouped 95% interval of [0.00, 0.61], and 10.2% of resamples select lambda = 0, i.e. no rule at all. The older bands' intervals exclude zero. This is a direct consequence of 64 escalating cases, and it is the single most important caveat on anything S9 measures from this parameter: an effect that fails to appear on test is as consistent with this interval as one that does.

Refit on 100 lesion resamples of the <40 band, the 7-parameter
threshold vector has a bootstrap standard deviation of 0.1373 on its escalating entries against
0.1499 for the single lambda. **Those two numbers are not
comparable as they stand**: theta is searched over a range of 0.6 and lambda over 1.2. As a share of the range each parameter was
actually searched over, the vector's spread is **0.229** against
**0.125** for lambda -- roughly 1.8x
wider, from seven free parameters instead of one, on the same 64 escalating cases.

The honest reading is that this is a difference of degree, not a disqualification:
at this sample size *both* instruments are unstable, and the case for the scalar
rests as much on it being a single pre-registerable operating point with a
clinical reading as on the spread measured here.

## 4. Held-out application on validation

The band lambdas are frozen from the fit split and applied to validation, which the
fit never saw. Referral rate is reported beside sensitivity: a rule that catches more
melanoma by referring far more patients has moved cost onto that band, not removed it.

| band | n | n_escalating | lambda_applied | base_sens | base_missed | base_referral | rule_sens | rule_missed | rule_referral |
|---|---|---|---|---|---|---|---|---|---|
| <40 | 267 | 22 | 0.2600 | 0.5455 | 10 | 0.0637 | 0.5909 | 9 | 0.0974 |
| 40-59 | 697 | 60 | 0.7400 | 0.5500 | 27 | 0.0803 | 0.8167 | 11 | 0.2023 |
| 60+ | 558 | 226 | 0.3300 | 0.7743 | 51 | 0.3781 | 0.8805 | 27 | 0.4480 |
| unknown | 10 | 0 | 0.6500 | - | 0 | 0.0000 | - | 0 | 0.0000 |
| ALL | 1532 | 308 | - | 0.7143 | 88 | 0.1854 | 0.8474 | 47 | 0.2722 |

Validation Macro-F1: 0.7638 (argmax) -> 0.7713 (lambda rule).

## 5. Does abstention already cover these misses?

Uncertainty score `msp` at threshold 0.3855, from `research/selective/results_oof/fit_state.json`.

| band | missed | missed_and_referred | share_referred | band_abstention_rate |
|---|---|---|---|---|
| <40 | 10 | 1 | 0.1000 | 0.0637 |
| 40-59 | 27 | 6 | 0.2222 | 0.0717 |
| 60+ | 51 | 21 | 0.4118 | 0.1720 |
| ALL | 88 | 28 | 0.3182 | 0.1064 |

## 6. Test reads

None. This session produces fitted parameters only; the single pre-registered test
pass belongs to S9, which applies `age_rule_lambda.json` without refitting it.

