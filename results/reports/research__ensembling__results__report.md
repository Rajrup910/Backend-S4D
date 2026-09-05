# Phase 1 — Ensembling Results (Session 1)

## Diversity audit (val split)

- Mean pairwise disagreement: **0.1483**
- Mean Yule's Q: **0.8986** (lower / negative = more complementary)
- Mean double-fault ratio: **0.0989**

## Method comparison

| Method | val_oof Macro-F1 | test Macro-F1 | test Balanced Acc | test Escalation Sens. | Notes |
|---|---:|---:|---:|---:|---|
| nonneg_stacking_ridge | 0.7583 | **0.7815** | 0.7480 | 0.6828 | one-vs-rest Ridge(positive=True) on K*C flattened logits, alpha=1.0 |
| soft_vote_arithmetic | 0.7911 | **0.7718** | 0.7923 | 0.7828 | no fitting; val_oof == direct val score |
| nelder_mead_simplex | 0.7911 | **0.7718** | 0.7923 | 0.7828 | weights=[0.167, 0.167, 0.167, 0.167, 0.167, 0.167] |
| soft_vote_geometric | 0.7850 | **0.7715** | 0.7929 | 0.7862 | no fitting; val_oof == direct val score |
| caruana_greedy | 0.7798 | **0.7688** | 0.7853 | 0.7655 | best_size=16 picks=['convnext_tiny', 'densenet121', 'convnext_small', 'efficientnet_b0', 'efficientnet_b3', 'efficientnet_b3', 'convnext_small', 'densenet121', 'convnext_tiny', 'efficientnet_b3', 'resnet50', 'convnext_small', 'efficientnet_b0', 'efficientnet_b3', 'convnext_small', 'efficientnet_b0'] |
| convnext_tiny_baseline | 0.7482 | **0.7459** | 0.7792 | 0.7759 | single frozen model, no ensembling |
| rank_average | 0.5047 | **0.5074** | 0.7909 | 0.8966 | no fitting; val_oof == direct val score |

## Winner: `soft_vote_arithmetic` vs. `convnext_tiny_baseline` baseline (test split, single read)

- Bootstrap 95% CI (N=1000) for winner's test Macro-F1: **0.7697** [0.7319, 0.8073]
- McNemar's test (winner vs. baseline, paired on test images): chi2=17.203, p=3.359e-05 (only-winner-correct=85, only-baseline-correct=38)
