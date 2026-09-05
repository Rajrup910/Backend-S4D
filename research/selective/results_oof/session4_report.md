# Phase 4 — Selective Classification & Subgroup Fairness (Session 4)

System: uniform soft-vote over 6 backbones with 24-view TTA, Dirichlet calibration fitted on oof (6981 out-of-fold training rows).
Fit-only run: test is locked by `research.testguard` and every number below is measured on validation (n=1532), which is held out from every fitted quantity here. Full-coverage val Macro-F1 0.7638, escalation sensitivity 0.7143, 88 missed serious cases.

## Uncertainty scores

Mahalanobis source: disabled on the OOF fit split — the Gaussians are fitted on train features from the full-train checkpoints, so scoring OOF training rows with them is in-sample and the resulting quantiles would not transfer.

| Score | val AURC (selection) | val AURC | val E-AURC |
|---|---:|---:|---:|
| msp **(selected on val)** | 0.03082 | 0.03082 | 0.02120 |
| margin | 0.03086 | 0.03086 | 0.02125 |
| entropy | 0.03128 | 0.03128 | 0.02167 |
| mutual_information | 0.03329 | 0.03329 | 0.02368 |
| ensemble_variance | 0.03451 | 0.03451 | 0.02490 |

## Risk-coverage, argmax decision rule, `msp` abstention

| Target abstention | Achieved coverage | Kept | Macro-F1 | Bal. Acc | Escalation Sens. | Missed serious (retained) | Serious referred | of which model got wrong |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100.0% | 1532 | 0.7638 | 0.7384 | 0.7143 | 88 | 0 | 0 |
| 5% | 94.4% | 1446 | 0.8169 | 0.7975 | 0.7283 | 75 | 32 | 17 |
| 10% | 89.4% | 1369 | 0.8489 | 0.8318 | 0.7551 | 60 | 63 | 35 |
| 15% | 83.9% | 1286 | 0.8878 | 0.8703 | 0.7972 | 43 | 96 | 52 |
| 20% | 79.8% | 1223 | 0.9128 | 0.9014 | 0.7979 | 39 | 115 | 57 |

## Risk-coverage, cost-sensitive decision rule (thresholds `[-0.3, -0.3, 0.3, 0.2, -0.3, 0.3, -0.1]`)

| Target abstention | Achieved coverage | Kept | Macro-F1 | Bal. Acc | Escalation Sens. | Missed serious (retained) | Serious referred | of which model got wrong |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100.0% | 1532 | 0.7559 | 0.7566 | 0.8669 | 41 | 0 | 0 |
| 5% | 94.4% | 1446 | 0.8163 | 0.8322 | 0.8551 | 40 | 32 | 9 |
| 10% | 89.4% | 1369 | 0.8432 | 0.8531 | 0.8367 | 40 | 63 | 13 |
| 15% | 83.9% | 1286 | 0.8683 | 0.8653 | 0.8113 | 40 | 96 | 14 |
| 20% | 79.8% | 1223 | 0.9066 | 0.9010 | 0.8031 | 38 | 115 | 17 |

## Subgroup fairness at 10% abstention (achieved coverage 89.4%)

Gaps are max-minus-min across groups with at least 30 val images. Sensitivity and FPR gaps additionally require at least 10 true escalating cases in the group — a group with a handful of malignancies cannot estimate a sensitivity, and including it would report sampling error as a disparity. Groups below either bar are still listed, marked in the last two columns.

### sex

| Group | n (retained) | Serious cases | Macro-F1 | Escalation Sens. | Missed serious | Referred % | n>=30 | positives>=10 |
|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| male | 724 | 157 | 0.8407 | 0.7898 | 33 | 11.2% | yes | yes |
| female | 635 | 88 | 0.8389 | 0.6932 | 27 | 10.2% | yes | yes |
| unknown | 10 | 0 | 0.2857 | nan | 0 | 0.0% | no | no |

Gaps: `demographic_parity_gap`=0.0827, `macro_f1_gap`=0.0018, `equalized_odds_tpr_gap`=0.0966, `equalized_odds_fpr_gap`=0.0116, `referral_burden_gap`=0.0098

### age_band

| Group | n (retained) | Serious cases | Macro-F1 | Escalation Sens. | Missed serious | Referred % | n>=30 | positives>=10 |
|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| 40-59 | 647 | 50 | 0.8018 | 0.5800 | 21 | 7.2% | yes | yes |
| 60+ | 462 | 177 | 0.8268 | 0.8305 | 30 | 17.2% | yes | yes |
| <40 | 250 | 18 | 0.7890 | 0.5000 | 9 | 6.4% | yes | yes |
| unknown | 10 | 0 | 0.2857 | nan | 0 | 0.0% | no | no |

Gaps: `demographic_parity_gap`=0.3030, `macro_f1_gap`=0.0378, `equalized_odds_tpr_gap`=0.3305, `equalized_odds_fpr_gap`=0.0462, `referral_burden_gap`=0.1084

### localization

| Group | n (retained) | Serious cases | Macro-F1 | Escalation Sens. | Missed serious | Referred % | n>=30 | positives>=10 |
|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| back | 280 | 71 | 0.5779 | 0.8169 | 13 | 10.0% | yes | yes |
| lower extremity | 278 | 36 | 0.6072 | 0.6667 | 12 | 11.7% | yes | yes |
| trunk | 212 | 9 | 0.6741 | 0.7778 | 2 | 2.8% | yes | no |
| abdomen | 155 | 14 | 0.5731 | 0.2143 | 11 | 4.9% | yes | yes |
| upper extremity | 139 | 31 | 0.7443 | 0.8387 | 5 | 17.3% | yes | yes |
| face | 96 | 39 | 0.7525 | 0.8718 | 5 | 20.0% | yes | yes |
| chest | 67 | 24 | 0.5655 | 0.7083 | 7 | 15.2% | yes | yes |
| foot | 55 | 7 | 0.3512 | 0.7143 | 2 | 8.3% | yes | no |
| unknown | 32 | 3 | 0.4000 | 0.6667 | 1 | 5.9% | yes | no |
| scalp | 21 | 5 | 0.5440 | 0.8000 | 1 | 8.7% | no | no |
| neck | 13 | 2 | 0.4821 | 1.0000 | 0 | 23.5% | no | no |
| ear | 8 | 2 | 0.3680 | 0.5000 | 1 | 27.3% | no | no |
| hand | 7 | 2 | 0.2857 | 1.0000 | 0 | 0.0% | no | no |
| genital | 6 | 0 | 0.1429 | nan | 0 | 0.0% | no | no |

Gaps: `demographic_parity_gap`=0.3869, `macro_f1_gap`=0.4013, `equalized_odds_tpr_gap`=0.6575, `equalized_odds_fpr_gap`=0.0877, `referral_burden_gap`=0.1725

## Does abstention rescue the misses it should?

Sensitivity below is at **full coverage**, so it describes the classifier itself rather than the retained subset. `Rescued` counts how many of the serious cases the classifier gets wrong are referred rather than answered incorrectly, at the 10% operating point. A group with a low rescue rate is one the model is *confidently* wrong about, where an uncertainty-based safety net does not deploy.

### sex

| Group | Serious cases | Escalation Sens. (full coverage) | Would be missed | Rescued by referral | Rescue rate | Group referral rate |
|---|---:|---:|---:|---:|---:|---:|
| male | 197 | 0.7462 | 50 | 17 | 34.0% | 11.2% |
| female | 111 | 0.6577 | 38 | 11 | 28.9% | 10.2% |

### age_band

| Group | Serious cases | Escalation Sens. (full coverage) | Would be missed | Rescued by referral | Rescue rate | Group referral rate |
|---|---:|---:|---:|---:|---:|---:|
| 60+ | 226 | 0.7743 | 51 | 21 | 41.2% | 17.2% |
| 40-59 | 60 | 0.5500 | 27 | 6 | 22.2% | 7.2% |
| <40 | 22 | 0.5455 | 10 | 1 | 10.0% | 6.4% |

## Skin tone

Not answered by this experiment, and deliberately not approximated. HAM10000 carries no Fitzpatrick labels; `ml/data/manifest_pad.csv` carries them for 1302 PAD-UFES-20 rows, but the PAD images are not present under `data/` in this repository, so no predictions exist to slice. The ITA image proxy in `ml/ood/skin_tone_slice.py` is documented in its own header as invalid on dermoscopy — vignetting and erythema dominate the angle — and is not used here. Answering the question needs Fitzpatrick-labelled data with images: PAD-UFES-20 restored locally, or Fitzpatrick17k.

## Fit and selection splits

Abstention quantiles, the Dirichlet map and the cost-sensitive thresholds were fitted on **oof** (`research/predictions_oof_tta`, n=6981); the policy score was selected by AURC on **validation** (n=1532). Fitted state: `research/selective/results_oof/fit_state.json`.
