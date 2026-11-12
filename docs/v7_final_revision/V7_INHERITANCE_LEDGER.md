# V7 inheritance ledger: what V1 to V6 established, and what V7 does with it

> **Precedence (11 Nov 2026).** `V7_FINAL_MASTER_RESEARCH_PLAN.md` **section 14 controls** wherever this document differs. In particular: the promotion rule is P1-P3 (plan 10.4, 14.3) and Gate A2 is a secondary read; MIDAS is a sealed **retrospective** external biopsy cohort, all ages only, and no prospective or consecutive external series exists (14.5); HIBA and the retired splits are training-only; nothing here is "certified", "frozen" or "validated" until the owner signs Gate 0 and the runs exist; the data gate is audited with stated lower bounds, never leak-free.

Written 10 Nov 2026. Every row names the file its claim comes from; `scripts/audit_v7_final_verification.py` fails if a cited
file is missing. Status words: **CORE** (kept unchanged), **CORE_GATE** (kept as a rule every V7 read must obey),
**CORE_SAFETY** (kept in the deployed safety stack), **RETEST** (kept, with a declared V7 test), **DROPPED** (measured and
not carried in THE FORM TESTED; a different form is a new idea, not a re-run), **OPEN** (a standing problem; see
`V7_STANDING_PROBLEMS.md`). Pre-tests run on folds 1-4 during the rebuild are exploratory: they may move a build's priority,
never drop it, because the owner's drop rules apply to a full build's frozen read (correction of 10 Nov, rows 34-36).

## V1 to V3: baselines, ensembling, calibration, safety

| # | What was established | Evidence | V7 status |
|:-:|---|---|:-:|
| 1 | Ensembling is the only rung that clears significance (A5 vs A2, McNemar p 3.4e-05; 1 of 42 DeLong tests survives Holm) | `results/mcnemar_delong.json` | CORE |
| 2 | Uniform soft vote, not ridge stacking: ridge had the best test and the worst val (test-selection trap) | `results/ablation_table.csv` | CORE_GATE |
| 3 | 24-view TTA is the better base | `research/predictions_tta/` (CLAUDE.md file map) | CORE (pooled by inverse entropy since V5) |
| 4 | The ensemble is UNDER-confident; Dirichlet beats temperature scaling because the distortion is class-dependent | `research/selective/results_oof/fit_state.json` | CORE_SAFETY |
| 5 | Marginal conformal coverage misleads (74.8% on malignant at 90.4% marginal); Mondrian class-conditional fixes it | `research/conformal/results/session4_conformal_report.md` | CORE_SAFETY |
| 6 | Selective abstention: margin and MSP tie; S56 abstention alone is the deployed stack | `research/selective/results/session4_report.md`, `research/v4/s56_abstention.py` | CORE_SAFETY |
| 7 | Bootstrap resamples lesions, not images; exact Clopper-Pearson for small counts | `research/stats/intervals.py`, `results/bootstrap_cis.json` | CORE_GATE |
| 8 | Test lock is mechanical, not promised | `research/testguard.py` | CORE_GATE |
| 9 | Mahalanobis OOD fails without shift and works with it (HAM vs PAD AUROC 0.913) | `research/xdomain/results/mahalanobis_shift.json` | CORE (OOD flag; must be re-fitted on V7 features) |
| 10 | Under-40 blind spot is confidently wrong: abstention refers only about 11% of the misses | `research/selective/results/session4_report.md` | OPEN (P1) |
| 11 | Calibration error flips sign across age bands after the global Dirichlet map (<40 -0.027, 60+ +0.041) | `research/stats/results_oof/band_calibration.csv` | OPEN (P4) |
| 12 | Age gap established, magnitude not (OOF 35/64 vs older bands; <40 vs 60+ -0.187, Holm p 0.036) | `research/stats/results_oof/age_gap_intervals.csv` | OPEN (P1) |
| 13 | The 3-band lambda age rule helps every band and the young least; it loses at matched workload (S65) | `research/agerule/results_oof/age_rule_lambda.json`, `results/v4/s65/s65_report.json` | DROPPED |
| 14 | ITA skin-tone proxy is invalid on dermoscopy; PAD Fitzpatrick I-IV spread 0.093, non-monotonic | `research/xdomain/results/fitzpatrick_gaps.json` | DROPPED (proxy); OPEN (P9, skin tone) |
| 15 | HAM-trained ensemble does not transfer to PAD smartphone photos (Macro-F1 0.124-0.188) | `research/xdomain/results/session8b_report.md` | OPEN (P6, separate modality track) |

## V4: in-domain training and its limit

| # | What was established | Evidence | V7 status |
|:-:|---|---|:-:|
| 16 | In-domain training fixes general accuracy, not under-40 ranking (S51, S54, S58, S64, S67: not a head, weighting or decision-rule problem) | `docs/RESEARCH_LOG.md` | OPEN (P1); reweighting, hard-case upweighting and decision rules DROPPED |
| 17 | The OOF-fitted Dirichlet buys calibration sample size, not accuracy | `results/oof_vs_val_comparison.csv` | CORE_SAFETY |
| 18 | ConvNeXt at 384 px: +0.0297 on HAM val (V4 S53r) | `docs/RESEARCH_LOG.md` | RETEST (B1) |

## V5: composite system, young data, the leak correction

| # | What was established | Evidence | V7 status |
|:-:|---|---|:-:|
| 19 | ConvNeXt 384 on 224-pretrained weights: pooled +0.0034 [-0.016, +0.025], under-40 pAUC -0.019 | `results/v5/s01_decision.json` | RETEST (B1, on high-resolution-pretrained ViTs) |
| 20 | V5's under-40 gain is +0.0136 (p 0.228) without copies, not +0.0351 | `results/v5/confirm_s42_s43_s44_without_copies.json` | CORE_GATE (every gain read with and without copies) |
| 21 | 33 of 81 under-40 escalating lesions missed at FPR 0.20, all called nevus, 19 of 33 in the high-melanin tercile | `results/v5/diagnostics/b1_hard_core.json` | OPEN (P2) |
| 22 | Inverse-entropy pooling of the 24 views | `research/v5/tta_v5.py` | CORE |
| 23 | Whole-image chromophore jitter (M7) lost 0.063 AUROC on held-out BCN: one seed, one holdout, a binary model with SWAD | `results/v6/m7_loao_binary.json`, `research/v5/m7.py` | Weak evidence against (one run); not prioritised, not dropped. B5 is a different transform (lesion-only, wide range) |

## V6: the deployed system

| # | What was established | Evidence | V7 status |
|:-:|---|---|:-:|
| 24 | V6 vote replaces V5: Macro-F1 0.7433 vs 0.7005 (+0.0427 [+0.0245, +0.0618]); all-age pAUC +0.0337 | `results/v6/e_primary.json` | CORE (`V6_REFERENCE` = banked files, `results/v7/v6_reference_manifest.json`) |
| 25 | Gate A fails: under-40 pAUC +0.0381 [+0.0107, +0.0711] against +0.050 | `results/v6/e_primary.json`, `results/v6/gate_a_power.json` | OPEN (P1) |
| 26 | Added pool (25,471 training-only rows), young sampler, 7-class head, per-epoch SWA | `results/v6/added_pool/audit.json`, `research/v6/train_v6.py` | CORE |
| 27 | Leak rule: every read passes on all rows, without the 380 and without the 426 at-risk rows | `results/v6/leak_sensitivity_read.json` | CORE_GATE (V7 adds a fourth set: rows the V7 data gate links) |
| 28 | Nested selection +0.0088; uniform vote kept | `results/v6/nested_selection.json` | CORE (uniform); B6 tested and DROPPED (row 35) |
| 29 | Claim 3 failed: PanDerm does not help under 40 more than at all ages | `results/v6/claim3.json` | OPEN (P8) |
| 30 | HIBA single read: under 40 V6 18/20, V5 19/20; flagged share V6 49.9% | `results/v6/hiba/hiba_report.json`, `results/v7/denominators.json` | CORE_GATE (one receipted read); HIBA spent |
| 31 | LR audit: PanDerm blocks 0-5 moved under 1.4% (layer decay 0.65); DINOv2 still rising at epoch 10; anchor WD 1e-4 | `results/v6/screens/lr_audit_weight_movement.txt` | RETEST (R1-R4) |
| 32 | **Dark-lesion subheadline (owner, 7 Nov):** missed young melanomas by melanin tercile, V5 control 3/13, 8/20, 10/15; V6 vote 2/13, 7/20, 5/15; within-archive missed melanomas still darker (p 0.040) on 48 young melanoma lesions; the frozen-DINOv2 dark specialist adds nothing | `results/v6/screens/dark_v6_oof_read.json`, `results/v6/screens/dark_specialist_probe.json` | **CORE starting line for P2**; required read for every candidate (`research/v7/dark_read.py`, reproduces V6's table exactly) |

## Measured during the V7 rebuild (10 Nov 2026)

| # | Finding | Evidence | Consequence |
|:-:|---|---|---|
| 33 | Adding any one member to a three-member V6 vote moves all-age pAUC by at most +0.004 and Macro-F1 by at most +0.013; adding V5's seeds to V6 lowers Macro-F1 by 0.011 | `results/v7/member_contribution.json` | Members like V6's own add little at the P1 bar (+0.0082). This measures members similar to the existing four; a member with different errors (B3's objective, B5's augmentation) is not measured by it. It lowers, but does not decide, P1's odds |
| 34 | B4 reduced pre-test (z-scores only, V6 logits, folds already read): OR 0.92 per SD [0.73, 1.16] under 40; minimum detectable OR 1.39; melanin rank alone OR 0.82 [0.58, 1.16]. **First written as "dropped"; corrected on recheck** | `results/v7/b4_cohort_relative.json`, `results/v7/recheck_dropped.json` | B4 NOT dropped: modest effects not ruled out; full form (neighbour cohort, fitted stage, V7 logits) untested |
| 35 | B6 reduced pre-test (V6's four members, no checkpoints): same subset in 3 of 4 folds, held-out worst-stratum pAUC -0.017 vs uniform. **First written as "dropped"; corrected on recheck** | `results/v7/b6_worst_group.json`, `results/v7/recheck_dropped.json` | B6 NOT dropped: not useful among V6's members; untested on V7 candidates |
| 36 | ISIC 2018 Task 2 masks on 1,441 development images; under 40, globules 5/11 melanoma vs 127/389 nevus (p 0.52). Fisher at this size detects OR 2 only 20% of the time | `results/v7/concept_premise.json`, `results/v7/recheck_dropped.json` | B2 READY: the premise check could not refute it; data on disk |
| 37 | Perceptual hashes: no different-lesion match at distance 4 in 5.8M pairs; all eight planted families found; only 3.9% of same-lesion re-photographs within 10 | `results/v7/hash_radius_probe.json` | Data gate tier 1; re-photographs need SIFT and the learned matcher |
| 38 | The added "ISIC 2019 Test" rows are its Barcelona part | `docs/V6_RUNSHEET.md` (line 523), `research/data/overlap_guard.py` | Dropped from training in every BCN-held-out run |
| 39 | Class-prior correction does not move BCN (0.638 to 0.635 with the true prior) | `results/v7/prior_shift_diagnostic.json` | DROPPED as a cross-site lever |
| 40 | P4 on the V6 vote (cross-fitted by fold): the global Dirichlet map still leaves the sign flip (<40 -0.015, 60+ +0.031); a band-wise map cuts the 60+ gap by 0.015 [0.009, 0.021], <40 by 0.006 (CI spans 0), 40-59 +0.005 (CI spans 0); Macro-F1 0.7513 vs 0.7443 | `results/v7/p4_band_calibration.json`, `research/v7/band_calibration.py` | Band-wise calibration meets its pre-declared endpoint (exploratory): carried to the V7 safety stack for the frozen read |
| 41 | Registry audit: lineage ledger files all exist; 17 of 32 point estimates found in their receipt JSON, 2 more in `results/ablation_table.csv`, 13 not traced; model registry: 2 of 17 checkpoints are local and both SHA-256 match; licences unverified offline | `results/v7/registry_audit.json`, `scripts/audit_v7_registries.py` | The 13 untraced registry numbers and all licence cells are UNVERIFIED; never cite them without their receipt |
| 42 | HIBA becomes training-only from V7 (owner, 10 Nov): 1,199 lesions (nv 552, bcc 223, mel 159, akiec 125, bkl 61, vasc 40, df 39), 20 under-40 escalating (14 melanoma), 57.5% histopathology | `results/v7/hiba_train/hiba_training_receipt.json`, `research/v7/hiba_training.py` | V6's HIBA read stays V6's; V7 is never scored on HIBA; MIDAS is V7's only external read (plan 10.11) |
| 43 | Retired evaluation splits join training (owner, 10 Nov): V7 pool 35,183 rows = V6 added 25,471 + retired `reserved`/`val`/`ham_val` 8,513 (22 byte-identical copies dropped) + HIBA 1,199; fold 0 trains on 14,018 core + 34,798 added rows (V6: 14,313 + 25,471). `ham_test` and MIDAS never trained on | `results/v7/added_pool_v7/pool_v7_receipt.json`, `research/v7/pool_v7.py`, `results/v7/data_gate_report.json` | Gate passes on 54,079 rows; holdout removals BCN 9,484, HAM 1,476, MSKCC 1,255 (a BCN-holdout smoke caught retired BCN rows left in training; fixed, with a consistency check) |
| 44 | Second-chance candidates (owner request): EfficientNet-B5 NS 456 px (C1), MaxViT-Tiny (C2), EVA-02-Small (C3); V1-V3 CNNs and MaxViT were never retrained with V6's recipe; SigLIP2 already had a fair V6 fold-0 screen (single Macro-F1 0.650, dropped) | `research/v7/members_v7.py`, `results/v6/benchmarks/`, `results/v6/screens/decision_v63.json`, `results/v7/logs/smokes/C2.log` | Same recipe, LR range check, and fold-0 vote rule as every member; no exception for pedigree |
| 45 | `pauc_histo` was a V5 and V6 endpoint (`research/v5/confirmation.py`, `research/v6/gate.py`) and was missing from the V7 plan. On the V6 vote: pAUC 0.865 on histopathology-confirmed negatives vs 0.957 on others; BCN minus HAM -0.053 [-0.083, -0.024] on all negatives, +0.024 [-0.008, +0.056] on confirmed; under-40 band gaps span zero on both (exploratory) | `results/v7/verification_strata.json`, `scripts/v7_verification_strata.py` | RESTORED as declared secondary endpoint E-histo (plan 11.2); MIDAS, a biopsied-only cohort, is compared with 0.865 / 0.814, never 0.889 / 0.833 |
| 46 | The V6 vote with only its two ImageNet-22k ConvNeXt members: Macro-F1 0.7377 vs 0.7433 (-0.0055 [-0.0158, +0.0049]); pAUC -0.0089 [-0.0126, -0.0053] (exploratory) | `results/v7/pretraining_controlled_vote.json`, `scripts/v7_pretraining_controlled_vote.py` | V6's result does not rest on members whose pretraining corpus cannot be audited; V7 repeats every promotion read on this row (plan 11.3, PROPOSED) |
| 47 | B2's student could not start loader workers on Windows (dataset class defined inside a function; then a V6 instance pickled under a patched name). Caught by the GPU smoke, not by the unit tests | `results/v7/logs/smokes/summary_gaps.txt`, `tests/test_v7_concepts.py` | Fixed; a test now pickles the dataset while the trainer seam is patched. All four gap smokes pass |
| 48 | Shortcut chain on the V6 vote (exploratory): the melanin index depends on the source (HAM follow-up nevi 0.19, HAM excised 0.35-0.43, BCN 0.48-0.50); under 40, dark excised nevi are not scored lower (rho +0.137) while dark melanomas are (rho -0.252, p 0.084, 48 lesions); missed young melanomas 2/13, 5/18, 2/17 against all negatives and 2/13, 6/18, 4/17 against excised negatives; BCN's melanoma F1 on confirmed rows is higher than HAM's (0.760 vs 0.668), and df plus vasc (62 BCN images) carry 0.096 of BCN's 0.131 Macro-F1 gap | `results/v7/shortcut_chain.json`, `scripts/v7_shortcut_chain.py` | Row 32's hypothesis is amended, not replaced: lost signal is now the better-supported form. T-CF decides between B5 and B8; S1 and S2 added (plan 12) |
| 49 | `reserved` holds 104 under-40 escalating lesion groups (BCN 80, MSKCC 24; 56 melanoma) against 65 in folds 1-4; V6 never trained on it | `ml/data/manifest_v4.csv` | D2 (plan 12.3, owner decision): cross-fitting it would give 169 groups for a paired read, at the cost of a fifth of its training rows per fold model |
