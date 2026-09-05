# Session 7 — statistical hardening (D, G.1, G.2, G.4)

Fit split: **oof** (train split, `research/predictions_oof_tta`). Validation read from `research/predictions_tta`. **Test not read** — `research.testguard` is armed unconditionally in this runner, and Table IV's test column is a placeholder for the S9 single pre-registered pass.

## 1. The interval, not just the estimate (G.1)

Escalation sensitivity is a proportion, and at these counts the choice of interval changes the claim. Both kinds are carried in `age_gap_intervals.csv`; the exact Clopper–Pearson interval leads wherever the numerator is small.

| band | n_escalating | n_caught | sensitivity | ci_lo | ci_hi | interval_method |
|---|---|---|---|---|---|---|
| <40 | 64 | 35 | 0.5469 | 0.3818 | 0.7164 | grouped_bootstrap |
| 40-59 | 397 | 243 | 0.6121 | 0.5468 | 0.6744 | grouped_bootstrap |
| 60+ | 893 | 655 | 0.7335 | 0.6966 | 0.7679 | grouped_bootstrap |
| unknown | 2 | 2 | 1.0000 | 0.1581 | 1.0000 | clopper_pearson |
| ALL | 1356 | 935 | 0.6895 | 0.6568 | 0.7207 | grouped_bootstrap |

The under-40 band holds **64** escalating cases out-of-fold against **22** on validation — the whole reason the age-conditional rule is fitted OOF. Point estimates are 0.547 (OOF) and 0.545 (val); these are different splits scored by different models and are **not** a like-for-like replication.

### The one confirmatory comparison

Pre-specified before any interval above was computed: `<40` versus `60+` escalation sensitivity, sole member of the `age_gap_confirmatory` family.

- difference **-0.187** [-0.370, -0.016], unpaired lesion-grouped bootstrap
- raw p = 0.036; Holm-adjusted p = 0.036 (family of one, so no penalty — which is what pre-specifying buys)
- validation arm, reported for direction only: -0.229 [-0.597, 0.078]

## 2. Which classes actually drag Macro-F1 down (D)

Limitations attributes the ladder's unresolvable upper rungs to `df`/`vasc` scarcity. Per-class F1 on the fit split says otherwise:

| class_code | support | f1 | ci_lo | ci_hi | ci_width |
|---|---|---|---|---|---|
| mel | 773 | 0.6067 | 0.5674 | 0.6449 | 0.0775 |
| akiec | 222 | 0.7000 | 0.6372 | 0.7560 | 0.1188 |
| df | 71 | 0.7087 | 0.5833 | 0.8125 | 0.2292 |
| bkl | 772 | 0.7675 | 0.7355 | 0.7946 | 0.0591 |
| bcc | 361 | 0.8163 | 0.7771 | 0.8540 | 0.0769 |
| vasc | 99 | 0.9082 | 0.8543 | 0.9482 | 0.0939 |
| nv | 4683 | 0.9360 | 0.9294 | 0.9427 | 0.0133 |

Rare-class *interval width* is real — a twenty-image class cannot have a narrow interval — but width and level are different complaints, and the level is where the Macro-F1 is lost. The manuscript sentence must name the classes this table puts at the bottom.

## 3. Under-confidence is not uniform across bands (G.2)

Each band appears twice: on the raw soft-vote and after the global Dirichlet map. If one map could serve every band the two blocks would show the same spread; whether they do is the argument for or against group-wise calibration (Hébert-Johnson et al. 2018).

| source | group | n | accuracy | mean_confidence | signed_gap | ece | ece_ci_lo | ece_ci_hi |
|---|---|---|---|---|---|---|---|---|
| uncalibrated | 40-59 | 3112 | 0.9001 | 0.7099 | -0.1902 | 0.1909 | 0.1807 | 0.2027 |
| uncalibrated | 60+ | 2512 | 0.7906 | 0.6715 | -0.1191 | 0.1201 | 0.1040 | 0.1375 |
| uncalibrated | <40 | 1319 | 0.9303 | 0.7013 | -0.2290 | 0.2290 | 0.2138 | 0.2465 |
| uncalibrated | unknown | 38 | 0.8947 | 0.7213 | -0.1734 | 0.2573 | 0.2359 | 0.2968 |
| uncalibrated | ALL | 6981 | 0.8664 | 0.6945 | -0.1718 | 0.1721 | 0.1640 | 0.1810 |
| dirichlet | 40-59 | 3112 | 0.9039 | 0.9084 | 0.0045 | 0.0260 | 0.0197 | 0.0378 |
| dirichlet | 60+ | 2512 | 0.7894 | 0.8304 | 0.0410 | 0.0410 | 0.0301 | 0.0610 |
| dirichlet | <40 | 1319 | 0.9348 | 0.9079 | -0.0269 | 0.0276 | 0.0219 | 0.0451 |
| dirichlet | unknown | 38 | 0.9474 | 0.9173 | -0.0300 | 0.0813 | 0.0406 | 0.1388 |
| dirichlet | ALL | 6981 | 0.8688 | 0.8803 | 0.0115 | 0.0246 | 0.0179 | 0.0327 |

Spreads across powered bands (max − min):

```json
{
  "oof/uncalibrated": {
    "ece_gap": 0.10892049304406591,
    "signed_gap_spread": 0.10988421910348745,
    "accuracy_gap": 0.13964509399612723,
    "n_groups": 3.0,
    "groups": [
      "40-59",
      "60+",
      "<40"
    ],
    "excluded_groups": [
      "unknown"
    ]
  },
  "oof/dirichlet": {
    "ece_gap": 0.01499214998012733,
    "signed_gap_spread": 0.06792282952331818,
    "accuracy_gap": 0.14538826219438583,
    "n_groups": 3.0,
    "groups": [
      "40-59",
      "60+",
      "<40"
    ],
    "excluded_groups": [
      "unknown"
    ]
  },
  "val/uncalibrated": {
    "ece_gap": 0.06391211939267114,
    "signed_gap_spread": 0.06712993001393763,
    "accuracy_gap": 0.09665337682735287,
    "n_groups": 3.0,
    "groups": [
      "40-59",
      "60+",
      "<40"
    ],
    "excluded_groups": [
      "unknown"
    ]
  },
  "val/dirichlet": {
    "ece_gap": 0.007948392610650987,
    "signed_gap_spread": 0.0627972404173488,
    "accuracy_gap": 0.13509323023639797,
    "n_groups": 3.0,
    "groups": [
      "40-59",
      "60+",
      "<40"
    ],
    "excluded_groups": [
      "unknown"
    ]
  }
}
```

## 4. Intersectional age × sex (D)

OOF-only by design: validation and test cells hold roughly 10–11 escalating cases, at or below the `MIN_POSITIVES=10` gate `research.selective.fairness` already imposes. Suppressed cells are listed with their reason rather than dropped — a table that hides its underpowered cells reads as though those patients were fine.

| group | n | n_escalating | escalation_sensitivity | sens_ci_lo | sens_ci_hi | sens_interval_method | suppressed | suppression_reason |
|---|---|---|---|---|---|---|---|---|
| 40-59 x female | 1542 | 173 | 0.5665 | 0.4706 | 0.6667 | boot | False |  |
| 40-59 x male | 1570 | 224 | 0.6473 | 0.5662 | 0.7281 | boot | False |  |
| 60+ x female | 887 | 294 | 0.7041 | 0.6393 | 0.7687 | boot | False |  |
| 60+ x male | 1625 | 599 | 0.7479 | 0.7047 | 0.7919 | boot | False |  |
| <40 x female | 718 | 38 | 0.5000 | 0.3338 | 0.6662 | CP | False |  |
| <40 x male | 596 | 26 | 0.6154 | 0.4057 | 0.7977 | CP | False |  |
| <40 x unknown | 5 | 0 | - | - | - |  | True | n=5 < MIN_GROUP_SIZE=30; escalating=0 < MIN_POSITIVES=10; group contains an 'unknown' attribute level |
| unknown x female | 4 | 0 | - | - | - |  | True | n=4 < MIN_GROUP_SIZE=30; escalating=0 < MIN_POSITIVES=10; group contains an 'unknown' attribute level |
| unknown x male | 4 | 2 | 1.0000 | 0.1581 | 1.0000 | CP | True | n=4 < MIN_GROUP_SIZE=30; escalating=2 < MIN_POSITIVES=10; group contains an 'unknown' attribute level |
| unknown x unknown | 30 | 0 | - | - | - |  | True | escalating=0 < MIN_POSITIVES=10; group contains an 'unknown' attribute level |

```json
{
  "n_usable_cells": 6.0,
  "equalized_odds_tpr_gap": 0.24791318864774625,
  "demographic_parity_gap": 0.29126976644525393
}
```

## 5. The comparison family grew, and it is declared (G.4)

`results/comparison_families.json` enumerates every family before its members are run. Confirmatory families are Holm-corrected within family; exploratory sets carry no significance claim and are reported as intervals only.

The six ladder McNemar tests were reported unadjusted in S5 while the 42 DeLong tests were corrected. They are a family by the same argument, and are corrected here from the frozen `results/mcnemar_delong.json` — a re-derivation, not a new test read:

| comparison | p_raw | p_holm | significant |
|---|---|---|---|
| A2_convnext_tiny vs A1_resnet50 | 0.4249 | 1.0000 | False |
| A3_swinv2_tiny vs A2_convnext_tiny | 0.3776 | 1.0000 | False |
| A4_gated_fusion vs A2_convnext_tiny | 0.1687 | 0.6747 | False |
| A5_soft_vote_6cnn vs A2_convnext_tiny | 0.0000 | 0.0002 | True |
| A6_soft_vote_6cnn_tta vs A5_soft_vote_6cnn | 0.1048 | 0.5242 | False |
| A7_tta_dirichlet vs A6_soft_vote_6cnn_tta | 0.6583 | 1.0000 | False |

## What S9 and S10 inherit

- Table IV (`paper/tables/table4_agegap.tex`) is written with validation and OOF columns populated and the **test column pending**. S9 fills it from the same generator during the single pre-registered pass.
- `audit_manuscript.py` should gain checks for: the interval method name (Clopper–Pearson, not bootstrap) in the Table IV caption, the per-band ECE values, the corrected rare-class attribution sentence, and the Holm-adjusted McNemar family.
- The Limitations paragraph on rare-class support needs rewriting against `per_class_f1.csv`, not against class counts.

