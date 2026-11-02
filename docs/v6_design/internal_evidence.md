# Internal evidence ledger V1–V5 (for the V6 redesign; every number with its source)

## Levers that moved numbers
| Lever | Effect | Protocol | Source |
|---|---|---|---|
| Pooled multi-archive training | reserved BCN+MSKCC Macro-F1 0.411 → 0.603 [0.588, 0.611] | V1 HAM-only vs V4 R0, 3 seeds | results/v5/same_footing.md B |
| Ensembling (6-CNN soft vote) | HAM test 0.746 → 0.772 Macro-F1, McNemar p 3.4e-05 | HAM test | same_footing A; results/mcnemar_delong.json |
| Ensembling (3-seed vote, one arch) | IN-22k fold 0: 0.662 → 0.682 Macro-F1, BA 0.651 → 0.668, pAUC all 0.814 → 0.831 | fold 0 (selection), descriptive | same_footing C |
| 24-view TTA | +0.014 Macro-F1 | HAM, A5→A6 | results/ablation_table.csv |
| IN-22k trunk | fold 0 +0.028 MF1; folds 1–4 +0.0082 [−0.0045, +0.0191] MF1, u40 pAUC +0.0081 [−0.0127, +0.0301] | Q11, 3 seeds × 4 folds | results/v5/q11_decomposition.json trunk_effect |
| Modules + young data (twostep + m4 + youngdata) | u40 pAUC +0.0270 [+0.0045, +0.0488]; all-age +0.0011 ns; MF1 +0.0025 ns | Q11, same IN-22k trunk | q11_decomposition.json module_effect |
| Composite vs in1k control | u40 +0.0351 [+0.0125, +0.0596]; MF1 +0.0107 [−0.0015, +0.0225]; all-age +0.0020 ns | folds 1–4, 3 seeds | results/v5/confirm_s42_s43_s44.json |
| Young data alone (screen) | u40 pAUC +0.0225, MF1 retained | fold 0, 3 seeds | gate_youngdata_in22k_vs_control.json |
| m4 head | all-age +0.0152 on own head; −0.0042 on escalation mass | folds 1–4, seed 42 | confirm_s42_s43_s44.json secondary_D1 |

## Absolute level (folds 1–4, 12,235 rows, IN-22k composite)
Macro-F1 ≈ 0.675, BA ≈ 0.666, all-age escalation pAUC ≈ 0.82, u40 pAUC ≈ 0.77 (q11 per-seed "a").

## Failure anatomy
- 81 u40 escalating lesions in dev (HAM 34, BCN 35, MSKCC 12) vs 1,435 at 60+; u40 escalating prevalence 2.9% (MSKCC) – 11.9% (BCN) vs 31–75% at 60+ (pool_profile.json).
- At FPR 0.20, 33/81 u40 escalating lesions missed (32 mel); misses predicted nv (51 images), bkl 4; melanin tercile high 19 / mid 11 / low 3 (b1_hard_core.json).
- LOAO (IN-22k control): HAM held out BA 0.519, BCN held out BA 0.391, MSKCC 0.577 (pool_profile.json).
- Young extra pool: 2,078 images, 291 escalating lesions (236 mel), from an ISIC archive query of 5,790 (2,562 already in corpus, 747 candidates lacking lesion ID).

## Measured failures (do not repeat without a stated reason it was unfair)
- Frozen FM probes: PanDerm −0.036, DINOv2 −0.073 (u40), S51.
- Unfrozen DINOv3-ConvNeXt-T without LLRD: Macro-F1 0.38–0.44.
- Loss swaps: LDAM-DRW 0.7256, ASL 0.7305 < CE 0.7459 (HAM).
- 384 px single model: pooled +0.004 (HAM rows gain, BCN not), S01; 224-pretrained weights; FixRes test-size re-score no gain.
- V5 modules: look, geometry, zoom, clues, gem, memory, m7, structure, m5, logic — failed gate or falsifier.
- Adversarial age removal: u40 AUC −0.1049 (V3 H6).
- Decision layers: λ(age), per-band offsets, combined policies — reject-cost (S57b/S65/S66).
- Caruana selection over 8 members overfit 1.5k val (0.7640/0.7839 vs 0.7945 uniform).
- Frozen-feature probes / heads for u40 (S67 stage 1 STAGE2_NO_GO); S64 decision-rule ceiling.
- NR-FP frequency features and Betti curves decode archive (0.960 / 0.816), add nothing.

## Rubric used for the 6/10 (keep identical for the re-score)
Rigour ≈ 9, strategy/focus ≈ 4, feasibility on this hardware ≈ 4; overall weighted to expected value per GPU hour.
