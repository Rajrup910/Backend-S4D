# Session 8b — cross-domain evaluation on PAD-UFES-20 (external, inference-only)

HAM10000-trained models applied to all **2106** PAD-UFES-20 images. Nothing fitted on PAD; every parameter frozen from the HAM OOF fits. HAM test not read (Hard Rule 2); the Mahalanobis in-distribution reference is HAM **val**.

## 1. The ensemble on PAD (first time)

Individual members span Macro-F1 0.124–0.188 (cf. the repo's prior 0.113–0.172). The uniform 6-CNN soft-vote reaches **0.167** Macro-F1, escalation sensitivity **0.356**, melanoma recall **0.115**.

| session | method | split | macro_f1 | balanced_accuracy | accuracy | escalation_sens | mel_recall | missed_serious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| session8b | pad_member_convnext_tiny | pad | 0.1717 | 0.3016 | 0.2555 | 0.3313 | 0.1154 | 1088 |
| session8b | pad_member_convnext_small | pad | 0.1881 | 0.3209 | 0.2526 | 0.362 | 0.1923 | 1038 |
| session8b | pad_member_densenet121 | pad | 0.161 | 0.3045 | 0.2697 | 0.4278 | 0.1154 | 931 |
| session8b | pad_member_efficientnet_b0 | pad | 0.1443 | 0.2915 | 0.2536 | 0.3534 | 0.1154 | 1052 |
| session8b | pad_member_efficientnet_b3 | pad | 0.1239 | 0.2673 | 0.2241 | 0.279 | 0.0577 | 1173 |
| session8b | pad_member_resnet50 | pad | 0.1717 | 0.3093 | 0.2569 | 0.4014 | 0.1731 | 974 |
| session8b | pad_ensemble_softvote | pad | 0.1669 | 0.3097 | 0.2626 | 0.3565 | 0.1154 | 1047 |
| session8b | pad_ensemble_dirichlet | pad | 0.1303 | 0.2931 | 0.2137 | 0.2207 | 0.1154 | 1268 |
| session8b | pad_ensemble_dirichlet_agerule | pad | 0.1723 | 0.3423 | 0.2754 | 0.5679 | 0.3269 | 703 |

The Dirichlet map and age-rule rows show whether the frozen HAM operating point transfers; PAD's escalating prevalence (~77%) is the inverse of HAM's (~19%), so the escalation-biasing lambda rule behaves very differently here — reported as measured.

## 2. Fitzpatrick skin-tone fairness (deployed Dirichlet ensemble)

PAD carries Fitzpatrick for 1,302 of 2,106 rows. Cohort is Fitzpatrick I–III dominated; V/VI are below the power gate and reported suppressed, not dropped.

| group | n | n_escalating | macro_f1 | escalation_sensitivity | escalation_fpr | adequately_powered | positives_powered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| I | 137 | 129 | 0.0887 | 0.3488 | 0.0 | True | True |
| II | 750 | 693 | 0.0967 | 0.2872 | 0.0351 | True | True |
| III | 347 | 316 | 0.1104 | 0.2468 | 0.0323 | True | True |
| IV | 59 | 36 | 0.1522 | 0.2778 | 0.0 | True | True |
| V | 8 | 6 | 0.0476 | 0.5 | 0.0 | False | False |
| VI | 1 | 0 | 0.0 | nan | 0.0 | False | False |
| unknown | 804 | 447 | 0.1034 | 0.0537 | 0.0196 | True | True |

Gaps across powered groups: `{"demographic_parity_gap": 0.28990993935432324, "macro_f1_gap": 0.06354623620278849, "equalized_odds_tpr_gap": 0.2951459341345404, "equalized_odds_fpr_gap": 0.03508771929824561}`

## 3. Mahalanobis distance under real shift

Fitted on HAM train features, the class-conditional Mahalanobis score separates PAD (shift) from HAM val (in-distribution) with **AUROC 0.913** (median score 485 ID vs 8717 shift, 18.0× separation). This redeems its last-place Session-4 ranking, which was on same-distribution HAM test where there was no shift to detect.
