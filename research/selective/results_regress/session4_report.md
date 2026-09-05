# Phase 4 — Selective Classification & Subgroup Fairness (Session 4)

System: uniform soft-vote over 6 backbones with 24-view TTA, Dirichlet calibration fitted on val.
Fit-only run: test is locked by `research.testguard` and every number below is measured on validation (n=1532), which is **in-sample**. Full-coverage val Macro-F1 0.7969, escalation sensitivity 0.7305, 83 missed serious cases.

## Uncertainty scores

Mahalanobis source: convnext_tiny penultimate features, shrinkage=0.031.

| Score | val AURC (selection) | val AURC | val E-AURC |
|---|---:|---:|---:|
| margin **(selected on val)** | 0.02563 | 0.02563 | 0.01799 |
| msp | 0.02671 | 0.02671 | 0.01906 |
| entropy | 0.02714 | 0.02714 | 0.01950 |
| entropy+mahalanobis | 0.02795 | 0.02795 | 0.02031 |
| mutual_information | 0.03226 | 0.03226 | 0.02462 |
| ensemble_variance | 0.03371 | 0.03371 | 0.02606 |
| mahalanobis | 0.03558 | 0.03558 | 0.02793 |

The feature-space and disagreement scores rank errors **worse** than the plain probability scores, and combining Mahalanobis with entropy is worse than entropy alone. This is the expected result rather than a disappointing one: Mahalanobis distance answers "is this input unlike anything in training?", and every test case here is HAM10000 dermoscopy drawn from the same acquisition process as the training set, so there is no distribution shift for it to detect. What is being ranked instead is *in-distribution* difficulty — genuinely ambiguous lesions that sit near a decision boundary while sitting comfortably inside the feature manifold — and the probability vector is the direct measurement of exactly that. The value of the Mahalanobis score is for out-of-distribution inputs a deployed system would actually meet (a smartphone photo, a non-lesion image, another clinic's scope) and that this test split by construction contains none of; PAD-UFES-20 would be the honest place to test it.

## Risk-coverage, argmax decision rule, `margin` abstention

| Target abstention | Achieved coverage | Kept | Macro-F1 | Bal. Acc | Escalation Sens. | Missed serious (retained) | Serious referred | of which model got wrong |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100.0% | 1532 | 0.7969 | 0.7737 | 0.7305 | 83 | 0 | 0 |
| 5% | 95.0% | 1455 | 0.8269 | 0.8022 | 0.7518 | 69 | 30 | 16 |
| 10% | 89.9% | 1378 | 0.8631 | 0.8486 | 0.7773 | 55 | 61 | 33 |
| 15% | 85.0% | 1302 | 0.8836 | 0.8701 | 0.8045 | 43 | 88 | 45 |
| 20% | 80.0% | 1225 | 0.9154 | 0.9028 | 0.8144 | 36 | 114 | 54 |

## Risk-coverage, cost-sensitive decision rule (thresholds `[-0.3, -0.3, 0.3, -0.3, -0.3, 0.3, -0.3]`)

| Target abstention | Achieved coverage | Kept | Macro-F1 | Bal. Acc | Escalation Sens. | Missed serious (retained) | Serious referred | of which model got wrong |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100.0% | 1532 | 0.7764 | 0.8052 | 0.8831 | 36 | 0 | 0 |
| 5% | 95.0% | 1455 | 0.8035 | 0.8291 | 0.8705 | 36 | 30 | 4 |
| 10% | 89.9% | 1378 | 0.8458 | 0.8629 | 0.8543 | 36 | 61 | 8 |
| 15% | 85.0% | 1302 | 0.8721 | 0.8758 | 0.8364 | 36 | 88 | 8 |
| 20% | 80.0% | 1225 | 0.9154 | 0.9028 | 0.8144 | 36 | 114 | 11 |

## Subgroup fairness at 10% abstention (achieved coverage 89.9%)

Gaps are max-minus-min across groups with at least 30 val images. Sensitivity and FPR gaps additionally require at least 10 true escalating cases in the group — a group with a handful of malignancies cannot estimate a sensitivity, and including it would report sampling error as a disparity. Groups below either bar are still listed, marked in the last two columns.

### sex

| Group | n (retained) | Serious cases | Macro-F1 | Escalation Sens. | Missed serious | Referred % | n>=30 | positives>=10 |
|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| male | 727 | 161 | 0.8693 | 0.8261 | 28 | 10.8% | yes | yes |
| female | 641 | 86 | 0.8284 | 0.6860 | 27 | 9.3% | yes | yes |
| unknown | 10 | 0 | 0.2857 | nan | 0 | 0.0% | no | no |

Gaps: `demographic_parity_gap`=0.0987, `macro_f1_gap`=0.0410, `equalized_odds_tpr_gap`=0.1400, `equalized_odds_fpr_gap`=0.0120, `referral_burden_gap`=0.0146

### age_band

| Group | n (retained) | Serious cases | Macro-F1 | Escalation Sens. | Missed serious | Referred % | n>=30 | positives>=10 |
|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| 40-59 | 655 | 47 | 0.8166 | 0.6170 | 18 | 6.0% | yes | yes |
| 60+ | 467 | 185 | 0.8538 | 0.8486 | 28 | 16.3% | yes | yes |
| <40 | 246 | 15 | 0.7788 | 0.4000 | 9 | 7.9% | yes | yes |
| unknown | 10 | 0 | 0.2857 | nan | 0 | 0.0% | no | no |

Gaps: `demographic_parity_gap`=0.3379, `macro_f1_gap`=0.0750, `equalized_odds_tpr_gap`=0.4486, `equalized_odds_fpr_gap`=0.0481, `referral_burden_gap`=0.1028

### localization

| Group | n (retained) | Serious cases | Macro-F1 | Escalation Sens. | Missed serious | Referred % | n>=30 | positives>=10 |
|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| back | 289 | 75 | 0.5638 | 0.8267 | 13 | 7.1% | yes | yes |
| lower extremity | 281 | 34 | 0.7325 | 0.6471 | 12 | 10.8% | yes | yes |
| trunk | 210 | 10 | 0.6664 | 0.7000 | 3 | 3.7% | yes | yes |
| abdomen | 154 | 11 | 0.5890 | 0.2727 | 8 | 5.5% | yes | yes |
| upper extremity | 136 | 29 | 0.7640 | 0.8621 | 4 | 19.0% | yes | yes |
| face | 97 | 43 | 0.7144 | 0.8837 | 5 | 19.2% | yes | yes |
| chest | 68 | 24 | 0.6158 | 0.7917 | 5 | 13.9% | yes | yes |
| foot | 55 | 7 | 0.3629 | 0.7143 | 2 | 8.3% | yes | no |
| unknown | 33 | 3 | 0.4000 | 0.6667 | 1 | 2.9% | yes | no |
| scalp | 21 | 5 | 0.5446 | 0.8000 | 1 | 8.7% | no | no |
| neck | 13 | 2 | 0.5714 | 1.0000 | 0 | 23.5% | no | no |
| ear | 8 | 2 | 0.3680 | 0.5000 | 1 | 27.3% | no | no |
| hand | 7 | 2 | 0.2857 | 1.0000 | 0 | 0.0% | no | no |
| genital | 6 | 0 | 0.1429 | nan | 0 | 0.0% | no | no |

Gaps: `demographic_parity_gap`=0.4238, `macro_f1_gap`=0.4011, `equalized_odds_tpr_gap`=0.6110, `equalized_odds_fpr_gap`=0.0926, `referral_burden_gap`=0.1623

## Does abstention rescue the misses it should?

Sensitivity below is at **full coverage**, so it describes the classifier itself rather than the retained subset. `Rescued` counts how many of the serious cases the classifier gets wrong are referred rather than answered incorrectly, at the 10% operating point. A group with a low rescue rate is one the model is *confidently* wrong about, where an uncertainty-based safety net does not deploy.

### sex

| Group | Serious cases | Escalation Sens. (full coverage) | Would be missed | Rescued by referral | Rescue rate | Group referral rate |
|---|---:|---:|---:|---:|---:|---:|
| male | 197 | 0.7665 | 46 | 18 | 39.1% | 10.8% |
| female | 111 | 0.6667 | 37 | 10 | 27.0% | 9.3% |

### age_band

| Group | Serious cases | Escalation Sens. (full coverage) | Would be missed | Rescued by referral | Rescue rate | Group referral rate |
|---|---:|---:|---:|---:|---:|---:|
| 60+ | 226 | 0.8053 | 44 | 16 | 36.4% | 16.3% |
| 40-59 | 60 | 0.5500 | 27 | 9 | 33.3% | 6.0% |
| <40 | 22 | 0.4545 | 12 | 3 | 25.0% | 7.9% |

## Skin tone

Not answered by this experiment, and deliberately not approximated. HAM10000 carries no Fitzpatrick labels; `ml/data/manifest_pad.csv` carries them for 1302 PAD-UFES-20 rows, but the PAD images are not present under `data/` in this repository, so no predictions exist to slice. The ITA image proxy in `ml/ood/skin_tone_slice.py` is documented in its own header as invalid on dermoscopy — vignetting and erythema dominate the angle — and is not used here. Answering the question needs Fitzpatrick-labelled data with images: PAD-UFES-20 restored locally, or Fitzpatrick17k.

## Fit and selection splits

Abstention quantiles, the Dirichlet map and the cost-sensitive thresholds were fitted on **val** (`research/predictions_tta`, n=1532); the policy score was selected by AURC on **validation** (n=1532). Fitted state: `research/selective/results_regress/fit_state.json`.
