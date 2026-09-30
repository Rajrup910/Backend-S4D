# Specification: V5 Common Out-of-Fold Matrix (`V5-OOF-COMMON-MATRIX`)

**Document Version:** 1.0.0 (Pre-Registration Specification)  
**Partition Authority:** S71 Frozen 5-Fold Partition (`results/v4/kfold/fold_assignments.csv`)  
**Total Images:** 15,294 rows  
**Total Lesions:** 8,748 unique lesions (8,734 lesion groups, strictly disjoint across folds per `s71_plan.json`)  
**Target Consumer:** Stage 6 Ensembling (`V5-S17-ENSEMBLE`) & Multi-Calibration (`V5-S18-TTA-CAL`)

---

## 1. Scientific & Methodological Rationale

Ensemble model selection, calibration fitting, and diversity auditing in V1–V4 suffered whenever models were evaluated on differing data subsets or mixed validation/OOF splits. 

To eliminate selection degrees of freedom and guarantee mathematical validity:
> **Every candidate model considered for ensemble inclusion in Stage 6 MUST generate its Out-of-Fold (OOF) cross-validated predictions on the exact same common rows defined by the frozen S71 partition.**

No model may be added to the ensemble matrix using held-out validation predictions ($N=2,270$) or unaligned cross-validation splits.

---

## 2. Table Schema & Column Definitions

The artifact `results/v5/v5_oof_common_matrix.csv` must conform to the following schema:

| Column Name | Data Type | Description | Permitted Values / Range |
|---|---|---|---|
| `image_id` | `string` | Unique dermoscopic image identifier | Matches S71 manifest exactly |
| `lesion_id` | `string` | Unique patient/lesion grouping cluster | Verified 0 overlap across folds |
| `fold` | `integer` | Cross-validation fold index | $\{0, 1, 2, 3, 4\}$ |
| `true_label` | `integer` | Ground-truth 7-class diagnostic label | $\{0, 1, 2, 3, 4, 5, 6\}$ (akiec, bcc, bkl, df, mel, nv, vasc) |
| `true_dx` | `string` | Human-readable 3-letter diagnosis code | `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc` |
| `age_band` | `string` | Stratified demographic age stratum | `<40`, `40-59`, `60+` |
| `escalation_status` | `integer` | Clinical binary escalation ground truth | `1` if mel/bcc/akiec; `0` if nv/bkl/df/vasc |
| `prob_{model_id}_{dx}` | `float` | Predicted calibrated probability for class `dx` | $[0.0, 1.0]$, summing to $1.0 \pm 10^{-5}$ per model |
| `logit_{model_id}_{dx}` | `float` | Raw uncalibrated model logit for class `dx` | Real-valued scalar |

---

## 3. Automated Integrity & Pre-Flight Assertions

Stage 6 (`V5-S17-ENSEMBLE`) cannot begin execution until the matrix passes the following automated unit tests:

1. **Row Count Assertion:** Exactly 15,294 rows.
2. **Key Uniqueness:** `image_id` is unique and matches S71 `fold_assignments.csv` row-for-row.
3. **Lesion Disjointness:** Zero `lesion_id` overlap between any fold $k$ and remaining folds $j \ne k$.
4. **Zero Missing Values:** Zero `NaN`, `null`, or infinite values in probability or logit columns.
5. **Probability Normalization:** For every model $m$ and row $i$, $\sum_{c=0}^6 p_{m, i, c} = 1.0 \pm 10^{-5}$.
6. **No Test Contamination:** Zero rows from the held-out HAM test split ($N=1,502$), reserved cohort ($N=4,733$), or validation split ($N=2,270$).

---

## 4. Downstream Metric Pipeline

When `V5-S17-ENSEMBLE` reads this matrix, it computes:
- Pairwise disagreement rate $D_{m1, m2}$;
- Pairwise Yule's $Q$ statistic:
  \[
  Q = \frac{N_{11}N_{00} - N_{01}N_{10}}{N_{11}N_{00} + N_{01}N_{10}}
  \]
  where $N_{ab}$ indicates correct/incorrect status of models $m_1$ and $m_2$;
- Pairwise double-fault measure $DF = \frac{N_{00}}{N}$;
- Stratified under-40 partial AUC ($pAUC_{FPR \le 0.20}$) via lesion-grouped bootstrap.
