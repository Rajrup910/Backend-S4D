# Phase 2 — TTA Results (Session 2)

24-view (8-fold dihedral x 3 scales) TTA with entropy-weighted pooling, test split, single read.

| Config | Macro-F1 | Balanced Acc | Escalation Sens. | Missed Serious | ECE |
|---|---:|---:|---:|---:|---:|
| convnext_tiny_no_tta | 0.7459 | 0.7792 | 0.7759 | 65 | 0.0997 |
| convnext_tiny_tta | 0.7564 | 0.7773 | 0.7931 | 60 | 0.1065 |
| ensemble_no_tta | 0.7718 | 0.7923 | 0.7828 | 63 | 0.1575 |
| ensemble_tta | 0.7859 | 0.8102 | 0.7862 | 62 | 0.1583 |
