# Phase 4 — Selective Classification & Subgroup Fairness (Session 4)

System: uniform soft-vote over 6 backbones with 24-view TTA, Dirichlet calibration fitted on val.
Test read once, n=1502. Full-coverage test Macro-F1 0.8047, escalation sensitivity 0.7310, 78 missed serious cases.

## Uncertainty scores

Mahalanobis source: convnext_tiny penultimate features, shrinkage=0.031.

| Score | val AURC (selection) | test AURC | test E-AURC |
|---|---:|---:|---:|
| margin **(selected on val)** | 0.02563 | 0.03109 | 0.02242 |
| msp | 0.02671 | 0.03084 | 0.02217 |
| entropy | 0.02714 | 0.03109 | 0.02242 |
| entropy+mahalanobis | 0.02795 | 0.03245 | 0.02378 |
| mutual_information | 0.03226 | 0.03395 | 0.02527 |
| ensemble_variance | 0.03371 | 0.03502 | 0.02634 |
| mahalanobis | 0.03558 | 0.03965 | 0.03097 |

The feature-space and disagreement scores rank errors **worse** than the plain probability scores, and combining Mahalanobis with entropy is worse than entropy alone. This is the expected result rather than a disappointing one: Mahalanobis distance answers "is this input unlike anything in training?", and every test case here is HAM10000 dermoscopy drawn from the same acquisition process as the training set, so there is no distribution shift for it to detect. What is being ranked instead is *in-distribution* difficulty — genuinely ambiguous lesions that sit near a decision boundary while sitting comfortably inside the feature manifold — and the probability vector is the direct measurement of exactly that. The value of the Mahalanobis score is for out-of-distribution inputs a deployed system would actually meet (a smartphone photo, a non-lesion image, another clinic's scope) and that this test split by construction contains none of; PAD-UFES-20 would be the honest place to test it.

## Risk-coverage, argmax decision rule, `margin` abstention

| Target abstention | Achieved coverage | Kept | Macro-F1 | Bal. Acc | Escalation Sens. | Missed serious (retained) | Serious referred | of which model got wrong |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100.0% | 1502 | 0.8047 | 0.7939 | 0.7310 | 78 | 0 | 0 |
| 5% | 95.4% | 1433 | 0.8298 | 0.8143 | 0.7519 | 65 | 28 | 15 |
| 10% | 89.2% | 1340 | 0.8577 | 0.8448 | 0.7621 | 54 | 63 | 29 |
| 15% | 83.6% | 1255 | 0.8696 | 0.8636 | 0.7750 | 45 | 90 | 40 |
| 20% | 77.9% | 1170 | 0.8957 | 0.8806 | 0.7919 | 36 | 117 | 50 |

## Risk-coverage, cost-sensitive decision rule (thresholds `[-0.3, -0.3, 0.3, -0.3, -0.3, 0.3, -0.3]`)

| Target abstention | Achieved coverage | Kept | Macro-F1 | Bal. Acc | Escalation Sens. | Missed serious (retained) | Serious referred | of which model got wrong |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100.0% | 1502 | 0.7429 | 0.7990 | 0.8621 | 40 | 0 | 0 |
| 5% | 95.4% | 1433 | 0.7770 | 0.8123 | 0.8511 | 39 | 28 | 5 |
| 10% | 89.2% | 1340 | 0.8142 | 0.8343 | 0.8282 | 39 | 63 | 8 |
| 15% | 83.6% | 1255 | 0.8573 | 0.8642 | 0.8050 | 39 | 90 | 12 |
| 20% | 77.9% | 1170 | 0.8957 | 0.8806 | 0.7919 | 36 | 117 | 18 |

## Subgroup fairness at 10% abstention (achieved coverage 89.2%)

Gaps are max-minus-min across groups with at least 30 test images. Sensitivity and FPR gaps additionally require at least 10 true escalating cases in the group — a group with a handful of malignancies cannot estimate a sensitivity, and including it would report sampling error as a disparity. Groups below either bar are still listed, marked in the last two columns.

### sex

| Group | n (retained) | Serious cases | Macro-F1 | Escalation Sens. | Missed serious | Referred % | n>=30 | positives>=10 |
|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| male | 690 | 137 | 0.8699 | 0.7956 | 28 | 13.3% | yes | yes |
| female | 638 | 90 | 0.8389 | 0.7111 | 26 | 8.1% | yes | yes |
| unknown | 12 | 0 | 0.2857 | nan | 0 | 0.0% | no | no |

Gaps: `demographic_parity_gap`=0.0743, `macro_f1_gap`=0.0310, `equalized_odds_tpr_gap`=0.0845, `equalized_odds_fpr_gap`=0.0216, `referral_burden_gap`=0.0525

### age_band

| Group | n (retained) | Serious cases | Macro-F1 | Escalation Sens. | Missed serious | Referred % | n>=30 | positives>=10 |
|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| 40-59 | 626 | 50 | 0.8781 | 0.8000 | 10 | 6.7% | yes | yes |
| 60+ | 437 | 160 | 0.8431 | 0.8250 | 28 | 17.9% | yes | yes |
| <40 | 268 | 17 | 0.3828 | 0.0588 | 16 | 7.6% | yes | yes |
| unknown | 9 | 0 | 0.2857 | nan | 0 | 0.0% | no | no |

Gaps: `demographic_parity_gap`=0.3289, `macro_f1_gap`=0.4953, `equalized_odds_tpr_gap`=0.7662, `equalized_odds_fpr_gap`=0.0502, `referral_burden_gap`=0.1115

### localization

| Group | n (retained) | Serious cases | Macro-F1 | Escalation Sens. | Missed serious | Referred % | n>=30 | positives>=10 |
|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| back | 290 | 60 | 0.7192 | 0.7000 | 18 | 12.9% | yes | yes |
| lower extremity | 289 | 33 | 0.8884 | 0.7576 | 8 | 9.4% | yes | yes |
| trunk | 194 | 5 | 0.4461 | 0.2000 | 4 | 5.8% | yes | no |
| upper extremity | 159 | 38 | 0.8329 | 0.8684 | 5 | 9.7% | yes | yes |
| abdomen | 143 | 18 | 0.6512 | 0.7778 | 4 | 8.3% | yes | yes |
| face | 78 | 32 | 0.7470 | 0.9375 | 2 | 13.3% | yes | yes |
| foot | 46 | 6 | 0.4692 | 0.3333 | 4 | 4.2% | yes | no |
| chest | 38 | 7 | 0.5992 | 1.0000 | 0 | 22.4% | yes | no |
| unknown | 38 | 1 | 0.4261 | 1.0000 | 0 | 5.0% | yes | no |
| neck | 23 | 9 | 0.6190 | 0.5556 | 4 | 11.5% | no | no |
| scalp | 18 | 12 | 0.4670 | 0.6667 | 4 | 33.3% | no | yes |
| hand | 14 | 2 | 0.3449 | 0.5000 | 1 | 12.5% | no | no |
| ear | 7 | 4 | 0.4286 | 1.0000 | 0 | 46.2% | no | no |
| genital | 3 | 0 | 0.1429 | nan | 0 | 0.0% | no | no |

Gaps: `demographic_parity_gap`=0.4897, `macro_f1_gap`=0.4623, `equalized_odds_tpr_gap`=0.2375, `equalized_odds_fpr_gap`=0.1957, `referral_burden_gap`=0.1828

## Does abstention rescue the misses it should?

Sensitivity below is at **full coverage**, so it describes the classifier itself rather than the retained subset. `Rescued` counts how many of the serious cases the classifier gets wrong are referred rather than answered incorrectly, at the 10% operating point. A group with a low rescue rate is one the model is *confidently* wrong about, where an uncertainty-based safety net does not deploy.

### sex

| Group | Serious cases | Escalation Sens. (full coverage) | Would be missed | Rescued by referral | Rescue rate | Group referral rate |
|---|---:|---:|---:|---:|---:|---:|
| male | 179 | 0.7430 | 46 | 18 | 39.1% | 13.3% |
| female | 111 | 0.7117 | 32 | 6 | 18.8% | 8.1% |

### age_band

| Group | Serious cases | Escalation Sens. (full coverage) | Would be missed | Rescued by referral | Rescue rate | Group referral rate |
|---|---:|---:|---:|---:|---:|---:|
| 60+ | 199 | 0.7739 | 45 | 17 | 37.8% | 17.9% |
| 40-59 | 70 | 0.7857 | 15 | 5 | 33.3% | 6.7% |
| <40 | 21 | 0.1429 | 18 | 2 | 11.1% | 7.6% |

## Skin tone

Not answered by this experiment, and deliberately not approximated. HAM10000 carries no Fitzpatrick labels; `ml/data/manifest_pad.csv` carries them for 1302 PAD-UFES-20 rows, but the PAD images are not present under `data/` in this repository, so no predictions exist to slice. The ITA image proxy in `ml/ood/skin_tone_slice.py` is documented in its own header as invalid on dermoscopy — vignetting and erythema dominate the angle — and is not used here. Answering the question needs Fitzpatrick-labelled data with images: PAD-UFES-20 restored locally, or Fitzpatrick17k.
