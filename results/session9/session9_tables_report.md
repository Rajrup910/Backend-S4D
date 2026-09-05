# Session 9 -- single pre-registered test pass (`tables` stage)

Plan `results/analysis_plan.json` frozen 2026-09-04T16:23:17+00:00, sha256 `5c9bebcf255a87b7136ce75918454e9d...`.

This stage emitted **18 of 19** pre-registered quantities. Nothing outside the plan was computed from test; `research.session9.plan.quantity` raises on an unregistered id, so that is enforced rather than asserted.

## Not emitted

- **`attribution.lesion_interior`** — stage 'attribution' has not been run; it is a separate GPU stage governed by the same plan and receipt.

## Ladder

| Rung | Macro-F1 [95% CI] | Bal. Acc. | Esc. sens. [95% CI] | Missed |
|---|---|---|---|---|
| `A7_val` | 0.8047 [0.7548, 0.8420] | 0.7939 | 0.7310 [0.6642, 0.7968] | 78 |
| `A7_oof` | 0.7871 [0.7315, 0.8295] | 0.7771 | 0.7310 [0.6629, 0.7986] | 78 |
| `A8` | 0.7810 [0.7207, 0.8258] | 0.7604 | 0.7345 [0.6655, 0.8014] | 77 |

A7-val reproduces the published `results/ablation_table.csv` macro-F1 (0.8047) to 0.00e+00, inside the 0.0005 tolerance.

### Family `s9_new_rungs` (confirmatory, Holm-corrected)

| Comparison | ΔMacro-F1 [95% CI] | p raw | p Holm | significant |
|---|---|---|---|---|
| A7_oof vs A7_val | -0.0176 [-0.0451, +0.0054] | 0.1500 | 0.2420 | no |
| A8 vs A7_val | -0.0237 [-0.0625, +0.0043] | 0.1210 | 0.2420 | no |

## Age bands (Table IV test column)

| Band | n | Esc. | Sensitivity [95% CI] | Interval | Esc.-mass AUC |
|---|---|---|---|---|---|
| <40 | 290 | 21 | 0.143 [0.030, 0.363] | clopper_pearson | 0.810 |
| 40-59 | 671 | 70 | 0.814 [0.694, 0.918] | grouped_bootstrap | 0.975 |
| 60+ | 532 | 199 | 0.764 [0.683, 0.836] | grouped_bootstrap | 0.933 |
| ALL | 1502 | 290 | 0.731 [0.665, 0.793] | grouped_bootstrap | 0.950 |

Confirmatory `<40` vs `60+`: -0.621 [-0.786, -0.386], two-sided p=0.000, Holm p=0.000 (significant at alpha=0.05, sole member of its family).

## Conformal

Coverage is audited, not guaranteed: Split conformal's finite-sample guarantee needs calibration and test scores to be exchangeable under one fixed score function. The OOF calibration scores come from five fold models and the test scores from the full-train ensemble, so no exact guarantee transfers. Coverage is therefore audited empirically against nominal 1-alpha and never asserted. The validation-fitted variant keeps the exact guarantee at n=24/22 per class; CV+/cross-conformal would restore a (1-2alpha) guarantee but needs test scored by all five fold models, which is a second test read and is out of scope here.

| alpha | Method | Calibrator | Coverage | Serious cov. | FRR [95% CI] | Set size |
|---|---|---|---|---|---|---|
| 0.05 | APS | bipartite | 0.9554 | 0.9517 | 0.0241 [0.0098, 0.0491] | 1.75 |
| 0.05 | APS | class_conditional | 0.9541 | 0.9552 | 0.0276 [0.0120, 0.0536] | 2.32 |
| 0.05 | APS | marginal | 0.9521 | 0.9000 | 0.0759 [0.0482, 0.1126] | 1.49 |
| 0.05 | LAC | bipartite | 0.9421 | 0.9483 | 0.0276 [0.0120, 0.0536] | 1.51 |
| 0.05 | LAC | class_conditional | 0.9407 | 0.9448 | 0.0345 [0.0167, 0.0625] | 2.27 |
| 0.05 | LAC | marginal | 0.9501 | 0.8621 | 0.1034 [0.0594, 0.1504] | 1.32 |
| 0.05 | RAPS | bipartite | 0.9541 | 0.9552 | 0.0138 [0.0038, 0.0349] | 1.87 |
| 0.05 | RAPS | class_conditional | 0.9501 | 0.9483 | 0.0207 [0.0076, 0.0445] | 2.40 |
| 0.05 | RAPS | marginal | 0.9527 | 0.8759 | 0.0897 [0.0594, 0.1286] | 1.36 |
| 0.10 | APS | bipartite | 0.8975 | 0.9000 | 0.0690 [0.0426, 0.1045] | 1.38 |
| 0.10 | APS | class_conditional | 0.8961 | 0.8897 | 0.0793 [0.0509, 0.1166] | 1.44 |
| 0.10 | APS | marginal | 0.9008 | 0.8241 | 0.1483 [0.0993, 0.2000] | 1.26 |
| 0.10 | LAC | bipartite | 0.8842 | 0.9069 | 0.0586 [0.0345, 0.0922] | 1.26 |
| 0.10 | LAC | class_conditional | 0.8755 | 0.8862 | 0.0793 [0.0509, 0.1166] | 1.28 |
| 0.10 | LAC | marginal | 0.9008 | 0.7345 | 0.2276 [0.1690, 0.2877] | 1.08 |
| 0.10 | RAPS | bipartite | 0.8928 | 0.8897 | 0.0621 [0.0372, 0.0963] | 1.30 |
| 0.10 | RAPS | class_conditional | 0.9001 | 0.9034 | 0.0517 [0.0292, 0.0839] | 1.40 |
| 0.10 | RAPS | marginal | 0.9021 | 0.7517 | 0.2138 [0.1555, 0.2741] | 1.12 |

## Clinical utility

NNB is prevalence-dependent. HAM10000's escalating prevalence is roughly 20%, against 1-5% in primary-care screening, so the observed NNB must never be compared to the 8-15 dermatologist range. The reported figure is re-weighted to pi=0.03 by importance-weighting the benign class, which assumes the class-conditional score distributions transfer -- an optimistic assumption, stated rather than hidden.

| Cohort | Rule | Sens. | Referral | NNB (observed) | NNB (pi=0.03) |
|---|---|---|---|---|---|
| <40 | `argmax` | 0.143 | 0.028 | 2.67 | 5.2 |
| <40 | `lambda_rule` | 0.238 | 0.066 | 3.80 | 8.1 |
| 40-59 | `argmax` | 0.814 | 0.109 | 1.28 | 2.1 |
| 40-59 | `lambda_rule` | 0.971 | 0.234 | 2.31 | 5.9 |
| 60+ | `argmax` | 0.764 | 0.350 | 1.22 | 5.3 |
| 60+ | `lambda_rule` | 0.844 | 0.423 | 1.34 | 7.6 |
| ALL | `argmax` | 0.731 | 0.178 | 1.26 | 3.0 |
| ALL | `lambda_rule` | 0.831 | 0.268 | 1.67 | 6.2 |

## Files written

- `results/session9/ladder.csv`
- `results/session9/new_rung_comparisons.json`
- `results/session9/age_gap_test.csv`
- `results/session9/age_gap_confirmatory_test.json`
- `results/session9/agerule_test.csv`
- `results/session9/agerule_summary_test.json`
- `results/session9/nnb_test.csv`
- `results/session9/orthogonality_test.csv`
- `results/session9/conformal_test.csv`
- `results/session9/conformal_cells_test.csv`
- `results/session9/frr_sweep_test.csv`
- `results/session9/frr_bounds_test.csv`
- `results/session9/selective_test.csv`
- `results/session9/band_calibration_test.csv`
- `results/session9/band_calibration_gaps_test.json`
- `results/session9/per_class_f1_test.csv`
- `results/session9/intersectional_test.csv`
- `results/session9/intersectional_disparities_test.json`

Receipt: `results/test_pass_receipt.json`.
