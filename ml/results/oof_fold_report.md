# Stratified Lesion-Grouped K-Fold Report (Session 6 OOF)

**Generated:** 2026-09-04T04:01:50  
**Folds:** 5  
**Total Train Images Partitioned:** 6981  
**Total Train Lesions Partitioned:** 5229  

## 1. Per-Fold Composition

| Fold | Holdout Images | Holdout Lesions | <40 Escalating | <40 Total |
|---|---:|---:|---:|---:|
| Fold 0 | 1397 | 1047 | 15 | 287 |
| Fold 1 | 1396 | 1044 | 18 | 263 |
| Fold 2 | 1396 | 1046 | 12 | 257 |
| Fold 3 | 1396 | 1047 | 13 | 268 |
| Fold 4 | 1396 | 1045 | 6 | 244 |

## 2. Per-Class Distribution Across Folds

| Class | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Total |
|---| ---: | ---: | ---: | ---: | ---: | ---:|
| **akiec** | 44 | 44 | 45 | 45 | 44 | 222 |
| **bcc** | 73 | 72 | 72 | 72 | 72 | 361 |
| **bkl** | 155 | 154 | 154 | 155 | 154 | 772 |
| **df** | 14 | 14 | 14 | 14 | 15 | 71 |
| **mel** | 154 | 155 | 155 | 154 | 155 | 773 |
| **nv** | 937 | 937 | 937 | 936 | 936 | 4683 |
| **vasc** | 20 | 20 | 19 | 20 | 20 | 99 |

## 3. Statistical Power Audit

- **`df` Total Yield:** 71 (target: ≥70; validation baseline was 24)
- **`vasc` Total Yield:** 99 (target: ≥95; validation baseline was 22)
- **`<40` Escalating Total Yield:** 64 (target: ≥64; validation baseline was 22, test was 21)

> **Methodological Guarantee**: Every fold holdout was validated with `assert_no_leakage()`. No patient lesion crosses train/test boundaries, and real test-split images are completely excluded.
