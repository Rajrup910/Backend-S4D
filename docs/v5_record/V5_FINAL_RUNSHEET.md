# V5 Master Execution Runsheet

**Repository:** https://github.com/Rajrup910/Backend-S4D  
**Document Version:** 2.0.0 (Execution-Ready Final Freeze)  
**Target Hardware:** Single NVIDIA RTX 5050 Laptop GPU (8.55 GB VRAM, torch 2.11+cu128, AMP on)  
**Execution Shell:** Windows PowerShell / Git Bash (`--num-workers 2` mandatory to prevent Windows OS error 1455)  
**Authoritative Rule:** Strictly staged 3-tier execution; no GPU compute allocated without passing preceding gate; planning and pre-registration only.

---

## 1. Hardware Calibration & Timing Foundations

All runtime figures explicitly distinguish between empirically measured runtimes and projected runtimes:

### A. Empirically Measured Benchmarks (Repository Ground Truth)
- **Stage 0 (CPU Manifest / Mask Audit):** $2.5\text{ min}$ (Measured in V4 audit passes).
- **224px 5-Fold Training (S72 Baseline):** $41.0\text{ min}$ per fold ($N=12,235$, 30 epochs, effective batch 32, 2 workers) $\to 4.1\text{ h}$ total 5-fold runtime (Measured in S72).
- **384px vs 224px Scaling Lever (S53r Benchmark):** 384px training scales runtime by $2.26\times$ relative to 224px on identical ConvNeXt architecture ($33.5\text{ min}$ vs $14.8\text{ min}$ on $N=6,981$) (Measured in S53r).
- **S69 Modality Gate Fitting:** $1.8\text{ min}$ on CPU (Measured in S69).

### B. Repository-Derived Extrapolations (Labeled EXTRAPOLATED)
- **384px Single-Fold Screening Run (1 seed, 30 epochs):** $41.0\text{ min} \times 2.26 \approx 92.6\text{ min}$ (EXTRAPOLATED).
- **384px Multi-Seed Baseline (3 seeds, 30 epochs):** $3 \times 92.6\text{ min} \approx 4.6\text{ h}$ (EXTRAPOLATED).
- **384px 5-Fold Cross-Validation (Finalists):** $4.1\text{ h} \times 2.26 \approx 9.3\text{ h}$ (EXTRAPOLATED).
- **24-View Dihedral TTA Inference (384px):** Scaled from 224px ($1.75\text{ h}$) by $2.26\times \approx 3.9\text{ h}$ across full ensemble (EXTRAPOLATED).

---

## 2. Master Execution Runsheet

*Execution follows the Staged Anti-Explosion Framework: Tier 1 (Default Screening on Seed 42) $\to$ Tier 2 (Limited 1-D Local Ablation on Seed 42) $\to$ Tier 3 (Multi-Seed / 5-Fold Confirmation).*  
*Methodological Scope Invariant: S01 answers whether 384px is reproducibly promising on the frozen Fold-0 screening split ($N_{\text{val}}=3,059$). The three-seed Fold-0 result is an initial reproducibility screen, NOT a full cross-validated estimate. Final performance claims and ensemble selection strictly require the complete $N=15,294$ cross-fitted OOF matrix (Stage 5/6).*


| Run ID | Stage | Experiment & Architecture | Required Input | Model Details | Resolution | Seed Policy | GPU Tier | Expected Time | Gate & Verification | Success Condition | Failure Action | Output Artifact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V5-S00-INTEGRITY** | 0 | Split, manifest, mask & registry audit | Repo tree, manifests, HAM masks | None (CPU) | N/A | None | CPU Tier (0 GB VRAM) | 2.5 min *(Measured)* | Stage 0 Gate | Clean git tree; **0 detected patient/lesion overlap using all available identifiers, with strict lesion-group disjointness**; $\ge 90\%$ valid masks; receipts locked (test $\le 2$, reserved $\le 8$) | Split leakage or manifest corruption: **ABORT ENTIRE PIPELINE**; Mask validity $<90\%$: block mask-dependent runs (S05, S06, S06b); Group-DRO cell size $<30$: mark S80 BLOCKED/DEFERRED while core pipeline proceeds | `results/v5/s00_integrity_report.json` |
| **V5-S00b-MORPH-AUDIT** | 0 | Descriptive age-stratified morphology audit (V5-B1) | HAM images + metadata | Feature extractors (CPU) | Native | None | CPU Tier (0 GB VRAM) | 15 min *(EXTRAPOLATED)* | Hypothesis Gate | Area fraction, asymmetry, network regularity extracted across age bands; cell counts verified | Informational only; generates baseline morphology tables for B11 | `results/v5/s00b_morphology_audit.json` |
| **V5-S01-384-CONTROL** | 1 | 384px global resolution baseline confirmation (V5-A1) | Frozen S71 partition (Fold 0, $N_{\text{train}}=12,235, N_{\text{val}}=3,059$) | ConvNeXt-Tiny | 384×384 | 42, 43, 44 (3 seeds) | GPU Tier 1 (~3.4 GB VRAM) | 4.6 h (3 seeds) *(EXTRAPOLATED)* | Gate D & S01 Two-Hurdle Gate | **Two-Hurdle Rule:** 95% bootstrap CI lower bound of paired $\Delta \text{Macro-F1} > 0.000$ AND mean paired $\Delta \text{Macro-F1} \ge +0.015$ (MCID); positive direction across $\ge 2/3$ seeds with 3-seed mean $\ge +0.015$ and min seed $\ge -0.010$; per-class safety (MEL $\Delta \text{F1} \ge -0.020$, BCC $\Delta \text{Recall} \ge -0.020$). *(5-fold OOF $N=15,294$ reserved for finalists)* | Re-verify recipe and augmentations; if paired gain fails, 384px is unconfirmed—stop pipeline | `results/v5/s01_control_384_report.json` |
| **V5-S02-DUALSTREAM** | 2 | Dual-stream global + lesion crop (V5-A2) | S01 checkpoint & crop pipeline | Dual ConvNeXt-Tiny + Capacity-Matched Single Stream Control | 384×384 | Tier 1 (Seed 42) | GPU Tier 2 (~6.2 GB VRAM) | 92 min *(EXTRAPOLATED)* | Gate A & Gate B | Under-40 $\Delta pAUC \ge +0.050$ vs S01 AND outperforms capacity-matched single-stream trunk | Demote dual-stream; log parameter-confound report; proceed to S03 | `results/v5/s02_dualstream_report.json` |
| **V5-S03-ESCALATION** | 2 | Trainable multi-task escalation head (V5-B2) | S01 base architecture | ConvNeXt-Tiny + binary escalation head | 384×384 | Tier 1 (Seed 42, $\lambda=0.50$) | GPU Tier 1 (~3.5 GB VRAM) | 80 min *(EXTRAPOLATED)* | Gate A & Gate B | Under-40 $\Delta pAUC \ge +0.050$ with Macro-F1 $\Delta \ge -0.010$. *(If pass, trigger Tier 2 $\lambda \in \{0.25, 1.0\}$)* | Fix $\lambda_{esc}=0.0$ (drop head); verify head gradient norms | `results/v5/s03_escalation_head_report.json` |
| **V5-S04-HIERARCHICAL** | 2 | Hierarchical diagnosis tree (V5-B3) | S01 base architecture | ConvNeXt-Tiny + hierarchical tree head | 384×384 | Tier 1 (Seed 42) | GPU Tier 1 (~3.5 GB VRAM) | 82 min *(EXTRAPOLATED)* | Gate A & Gate B | Under-40 $\Delta pAUC \ge +0.050$ AND malignant recall $\ge 0.85$ | Retain flat 7-class head; log hierarchical error confusion matrix | `results/v5/s04_hierarchical_report.json` |
| **V5-S05-LOCALPATCH** | 2 | Native-resolution micro-patch attention (V5-B4) | S01 base + native crops | ConvNeXt-Tiny + micro-patch encoder (128px) | 384×384 + 128×128 | Tier 1 (Seed 42) | GPU Tier 2 (~5.8 GB VRAM) | 95 min *(EXTRAPOLATED)* | Gate A & Data-Path QC | **QC:** Crops verified from native resolution prior to downsampling; **Gate:** Under-40 $\Delta pAUC \ge +0.050$, Macro $\Delta \ge -0.010$ vs capacity control | Demote local patches; proceed with strongest surviving Stage 2 representation | `results/v5/s05_localpatch_report.json` |
| **V5-S78-T3-CONFIRM** | 2 | Stage-2 survivor multi-seed confirmation | Surviving Stage 2 candidate(s) from S02–S05 | Top qualifying Stage 2 architecture | 384×384 | 42, 43, 44 (2 additional seeds: 43, 44) | GPU Tier 1/2 (~6.2 GB max) | 3.1 h (2 seeds) *(EXTRAPOLATED)* | Gate A, Gate B & Multi-Seed Consistency | Confirms mean paired gain $\Delta pAUC_{0.20} \ge +0.050$ and $\Delta \text{Macro} \ge -0.010$ across 3 seeds; direction consistent ($\ge 2/3$ seeds positive); reported dispersion $s_{\text{seed}}$ | If fails confirmation, fallback to S01 384px single-stream control trunk; proceed to Stage 3 | `results/v5/s78_t3_survivor_confirm_report.json` |
| **V5-S06-CONTEXT** | 3 | Soft boundary / interior / context decomposition (V5-B5) | Stage 2 survivor + HAM masks | Best Stage 2 trunk + alpha-context | 384×384 | Tier 1 (Seed 42, $\alpha=0.25$) | GPU Tier 2 (~4.5 GB VRAM) | 85 min *(EXTRAPOLATED)* | Gate A & Gate C | Optimal $\alpha \in [0.1, 0.5]$ strictly outperforms $\alpha=1.0$ (full context) & $\alpha=0.0$ on under-40 pAUC | Retain full context ($\alpha=1.0$); log context-shortcut audit | `results/v5/s06_context_report.json` |
| **V5-S06b-RADIAL-POLAR** | 3 | Radial/polar coordinate representation (V5-B9) | Stage 2 survivor + HAM masks | Polar coordinate transformed stream | 384×384 | Tier 1 (Seed 42) *(Exploratory)* | GPU Tier 2 (~4.5 GB VRAM) | 80 min *(EXTRAPOLATED)* | Gate A (Exploratory) | Polar stream improves peripheral streak/pseudopod detection by $\Delta pAUC \ge +0.030$ | Demote radial/polar branch; maintain Cartesian stream | `results/v5/s06b_radial_polar_report.json` |
| **V5-S07-HARDNEG** | 3 | Development OOF hard-negative mining / JTT | Stage 2 OOF errors | Best Stage 2 trunk | 384×384 | Tier 1 (Seed 42) | GPU Tier 1 (~3.5 GB VRAM) | 85 min *(EXTRAPOLATED)* | Gate A & Gate B | Under-40 MEL recall improves $\ge +0.06$ without degrading NV F1 by $>0.02$ | Revert to standard empirical risk minimization (ERM) | `results/v5/s07_hardneg_report.json` |
| **V5-S08-SUPCON** | 3 | Supervised contrastive representation learning (V5-B7) | Best Stage 2 architecture | Best Stage 2 trunk + projection head | 384×384 | Tier 1 (Seed 42, $\tau=0.07$) | GPU Tier 2 (~5.2 GB VRAM) | 90 min *(EXTRAPOLATED)* | Gate A & Gate B | Latent cluster separation $MEL \leftrightarrow NV$ increases; $\Delta pAUC \ge +0.050$ | Demote SupCon; retain cross-entropy + escalation loss | `results/v5/s08_supcon_report.json` |
| **V5-S09-MELNV-RANK** | 3 | Pairwise melanoma-vs-nevus ranking loss (V5-B12) | Best Stage 2 architecture | Best Stage 2 trunk + ranking loss $L_{rank}$ | 384×384 | Tier 1 (Seed 42, $m=0.20$) | GPU Tier 1 (~3.6 GB VRAM) | 82 min *(EXTRAPOLATED)* | Gate A & Gate B | Under-40 MEL vs NV rank inversion drops $\ge 15\%$; $\Delta pAUC \ge +0.050$ | Demote $L_{rank}$; test Tier 2 margin $m \in \{0.10, 0.50\}$ | `results/v5/s09_melnv_rank_report.json` |
| **V5-S09b-MELNV-BRANCH** | 3 | Trainable end-to-end MEL-vs-NV auxiliary branch (V5-B6) | Best Stage 2 architecture | Joint visual encoder + dedicated MEL-NV head | 384×384 | Tier 1 (Seed 42) | GPU Tier 1 (~3.6 GB VRAM) | 82 min *(EXTRAPOLATED)* | Gate A & Gate B | Dedicated branch improves binary MEL-NV separation by $\ge +0.040$ AUC without Macro loss | Demote dedicated branch; rely on 7-class + ranking head | `results/v5/s09b_melnv_branch_report.json` |
| **V5-S10-DWT** | 3 | Discrete Wavelet Transform (Haar) high-frequency (V5-B8) | S01 / Stage 2 trunk | Trunk + DWT LH/HL/HH sub-bands | 384×384 | Tier 1 (Seed 42) | GPU Tier 2 (~5.5 GB VRAM) | 88 min *(EXTRAPOLATED)* | Gate A & Capacity Control | Outperforms capacity-matched CNN baseline by $\Delta pAUC \ge +0.035$ | Demote DWT branch; record parameter-matched null result | `results/v5/s10_dwt_branch_report.json` |
| **V5-S11-MORPH-ATTR** | 3 | Auxiliary morphology attribute supervision (V5-MORPH-ATTR) | Expert attribute annotations (Derm7pt) | Best Stage 2 trunk + attribute heads | 384×384 | Tier 1 (Seed 42) *(Gated)* | GPU Tier 1 (~3.8 GB VRAM) | 85 min *(EXTRAPOLATED)* | Gate A & Attribute Gate | Attribute prediction AUROC $\ge 0.80$ and transfers $\Delta pAUC \ge +0.030$ | If external annotations fail provenance check, mark DEFERRED | `results/v5/s11_morph_attribute_report.json` |
| **V5-S12-GEM** | 3 | Adaptive Generalized Mean (GeM) pooling (V5-B11) | Best Stage 2 trunk | Trunk with GeM pooling ($p=3$) | 384×384 | Tier 1 (Seed 42) | GPU Tier 1 (~3.4 GB VRAM) | 75 min *(EXTRAPOLATED)* | Gate B & Gate A | GeM outperforms standard GAP on Macro-F1 ($\ge +0.005$) and under-40 pAUC | Retain standard Global Average Pooling (GAP) | `results/v5/s12_gem_pooling_report.json` |
| **V5-S13-GROUPDRO** | 4 | Subgroup-robust Group-DRO on best representation (V5-A3) | Single best representation from Stage 3 | Surviving architecture | 384×384 | 42, 43, 44 (3 seeds) | GPU Tier 2 (~4.5 GB VRAM) | 4.6 h (3 seeds) *(EXTRAPOLATED)* | Gate A & Gate C | Worst-group risk ($<40 \times esc$) improves $\ge 10\%$ without collapsing $60+$ recall; seed stability diagnostic (warning threshold $s_{\text{seed}} > 0.015$) | Revert to ERM training; log minimax optimization curves; pipeline proceeds | `results/v5/s13_group_dro_report.json` |
| **V5-S14-CONVNEXTV2** | 5 | ConvNeXt-V2 with GRN (V5-A4) | Stage 3/4 optimal recipe | `convnextv2_tiny.fcmae_ft_in22k_in1k_384` (28.6M) | 384×384 | 42, 43, 44 (3 seeds) | GPU Tier 1 (~3.6 GB VRAM) | 4.6 h (3 seeds) *(EXTRAPOLATED)* | Gate B & Gate D | Macro-F1 $\ge 0.7700$, validated error diversity vs ConvNeXt-V1; seed stability diagnostic (warning $s_{\text{seed}} > 0.015$) | Disqualify ConvNeXt-V2 from finalist pool | `results/v5/s14_convnext_v2_report.json` |
| **V5-S15-EFFNETV2** | 5 | EfficientNetV2-M compound scaling (V5-A4) | Stage 3/4 optimal recipe | `tf_efficientnetv2_m.in21k_ft_in1k` (54.1M) | 384×384 | 42, 43, 44 (3 seeds) | GPU Tier 2 (~5.4 GB VRAM) | 5.5 h (3 seeds) *(EXTRAPOLATED)* | Gate B & Gate D | Macro-F1 $\ge 0.7700$, validated error diversity vs ConvNeXt family; seed stability diagnostic (warning $s_{\text{seed}} > 0.015$) | Disqualify EfficientNetV2-M; inspect batch size | `results/v5/s15_effnet_v2_report.json` |
| **V5-S16-SWINV2** | 5 | SwinV2-Tiny hierarchical self-attention (V5-A4) | Stage 3/4 optimal recipe | `swinv2_tiny_window12to16_192to384.ms_in22k_ft_in1k` (28.35M) | 384×384 | 42, 43, 44 (3 seeds) | GPU Tier 2 (~6.4 GB VRAM) | 6.0 h (3 seeds) *(EXTRAPOLATED)* | Gate B & Gate D | Macro-F1 $\ge 0.7650$, high visual complementarity ($Q < 0.70$); seed stability diagnostic (warning $s_{\text{seed}} > 0.015$) | Disqualify SwinV2; log self-attention failure modes | `results/v5/s16_swin_v2_report.json` |
| **V5-S81-T3-FINALIST-OOF** | 5 | Finalist 5-fold cross-fitted OOF generation | S71 frozen 5-fold partition ($N=15,294$) | Promoted finalist trunk(s) from Stage 5 / S01 | 384×384 | 5 folds per finalist | GPU Tier 1/2 (~6.4 GB max) | Conditional: 7.7 h (1 model), 16.9 h (2 models), 26.9 h (3 models) *(EXTRAPOLATED)* | OOF Alignment & Gate B | 100% complete OOF predictions on all 15,294 rows; identical fold mapping; no test or reserved rows | Re-train corrupted fold; if persistent fold failure, disqualify model from ensemble consideration | `results/v5/s81_t3_oof_predictions_<model>.csv` |
| **V5-S17-ENSEMBLE** | 6 | Error-diverse compact soft-voting ensemble | `V5-OOF-COMMON-MATRIX` ($N=15,294$) | Top surviving heterogeneous models | 384×384 | Pooled OOF | CPU / GPU Tier 1 (~2.0 GB VRAM) | 45 min *(EXTRAPOLATED)* | Gate E & Matrix Prerequisite | **Matrix Check:** 100% row alignment on S71 folds; **Ensemble:** Macro-F1 $\ge 0.7950$, $\Delta pAUC \ge +0.070$ over single best | Retain minimal 2-model ensemble; rule out high-correlation members | `results/v5/s17_ensemble_report.json` |
| **V5-S17b-MORPH-GATE** | 6 | Morphology-gated ensemble routing (V5-B10) | OOF common matrix + morphology predictions | Gating network on top of ensemble | 384×384 | Pooled OOF *(Conditional)* | CPU / GPU Tier 1 (~2.0 GB VRAM) | 30 min *(EXTRAPOLATED)* | Gate E (Gating) | Morphology gate improves young melanoma recall by $\ge +0.03$ over uniform soft-vote; input from validated morphology signal (B1, B5, B6, V5-MORPH-ATTR); simple shallow gate; no raw age input | Retain uniform soft-voting; discard high-capacity gating | `results/v5/s17b_morph_gated_ensemble_report.json` |
| **V5-S18-TTA-CAL** | 7 | 24-view TTA & S55 groupwise multi-calibration | S17 ensemble checkpoints | Final ensemble | 384×384 | Evaluated | GPU Tier 2 (~4.0 GB VRAM) | 3.9 h *(EXTRAPOLATED)* | Safety Gate | Macro-F1 $\ge 0.8050$, signed calibration gap across all age strata $\le 0.015$ | If TTA bottleneck occurs, adopt reduced-view dev policy; recalibrate | `results/v5/s18_tta_calibration_report.json` |
| **V5-S19-SAFETY** | 7 | Conformal prediction safety net & modality gate | Calibrated ensemble + S69 modality classifier | Final safety pipeline | 384×384 | Evaluated | CPU / GPU Tier 1 (~1.5 GB VRAM) | 20 min *(EXTRAPOLATED)* | Deployment Safety Gate | Evaluates predefined calibration, conformal coverage, and modality-admissibility safety properties (100% non-dermoscopy rejection, $<40$ malignant coverage $\ge 95\%$, $FRR \le 0.02$) | Retain conservative threshold fallback; log non-conformal edge cases | `results/v5/s19_conformal_safety_report.json` |
| **V5-S20-EXTERNAL** | 8 | Locked confirmatory read on fresh external cohort (V5-A5) | S75 fresh external dataset + frozen code | Frozen V5 ensemble + safety stack | 384×384 | Locked Single Read | GPU Tier 2 (~4.5 GB VRAM) | 65 min *(EXTRAPOLATED)* | Gate F (Confirmatory) | **Hierarchical Test:** Step 1: External Macro-F1 $\ge 0.7500$ ($p < 0.05$); Step 2: Under-40 sensitivity $\ge 0.700$ vs $H_0 \le 0.500$ ($p < 0.05$, $N_{\text{esc,<40}} \ge 40$ lesions floor for 80.7% exact power; target $N \ge 50$ for 85.9% power) | **EXECUTE PRE-REGISTERED NULL PATHWAY (§26)**; report transport boundaries | `results/v5/s20_confirmatory_external_verdict.json` |


---

## 3. Secondary & Deferred Exploratory Queue

*These runs are formally classified as **NOT CORE V5**. They are executed strictly on development justification and never block the core pipeline progression.*

| Run ID | Stage | Experiment | Required Input | Model | Resolution | Seed | GPU Tier | Expected Time | Gate | Success Condition | Failure Action | Output Artifact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V5-X01-MIXUP-CUTMIX** | Exploratory | 384px Mixup/CutMix sensitivity (V5-X1) | Stage 1 baseline | ConvNeXt-Tiny | 384×384 | 42 | GPU Tier 1 (~3.5 GB VRAM) | 80 min *(EXTRAPOLATED)* | Exploratory Gate | Statistically significant Macro-F1 improvement ($\ge +0.015$) with colour-safe $\alpha \le 0.2$ | Maintain pure geometric augmentation; discard sample blending | `results/v5/x01_mixup_cutmix_report.json` |
| **V5-X02-SWA** | Deferred | Optional SWA schedule ablation (V5-X2) | Stage 1 / Stage 5 checkpoints | Target architecture | 384×384 | 42 | GPU Tier 1 (~3.5 GB VRAM) | 45 min *(EXTRAPOLATED)* | Stability Gate | Reduces validation loss variance across final 10 epochs without Macro-F1 drop | Retain standard best-checkpoint selection; discard weight averaging | `results/v5/x02_swa_schedule_report.json` |
| **V5-X03-FOCAL-LOSS** | Deferred | Exploratory imbalance loss sensitivity (V5-X3) | Stage 1 baseline | ConvNeXt-Tiny | 384×384 | 42 | GPU Tier 1 (~3.4 GB VRAM) | 78 min *(EXTRAPOLATED)* | Screening Gate | Outperforms standard Cross-Entropy on under-40 pAUC by $\ge +0.030$ | Permanently archive class-imbalance loss tweaks; focus on representation | `results/v5/x03_focal_loss_report.json` |
| **V5-X04-BALANCED-SOFTMAX** | Deferred | Balanced Softmax logit adjustment probe (V5-X4) | Stage 1 baseline | ConvNeXt-Tiny | 384×384 | 42 | GPU Tier 1 (~3.4 GB VRAM) | 75 min *(EXTRAPOLATED)* | Screening Gate | Under-40 $\Delta pAUC \ge +0.030$ without overall Macro-F1 drop | Mark permanently null; retain standard cross-entropy | `results/v5/x04_balanced_softmax_report.json` |
| **V5-X05-FOUNDATION-DISTILL** | Deferred | Foundation model feature distillation probe (V5-X5) | PanDerm / DINOv2 teacher + S01 student | ConvNeXt-Tiny student | 384×384 | 42 | GPU Tier 2 (~6.0 GB VRAM) | 110 min *(EXTRAPOLATED)* | Distillation Gate | Distilled student outperforms pure visual S01 student by $\Delta pAUC \ge +0.040$ | Discard foundation distillation; rely on task-specific high-res learning | `results/v5/x05_foundation_distill_report.json` |

---

## 4. Compute Budget & Path Analysis

All compute allocations are derived from empirical benchmarks on the single NVIDIA RTX 5050 Laptop GPU (8.55 GB VRAM):

### A. Minimum Compute Path (Early-Stopping / Fallback Path)
*Assumes Stage 1 confirms 384px baseline; all Stage 2 visual representations fail screening gates; pipeline falls back to S01 trunk; 1 backbone screened and evaluated for OOF ensembling:*
- Stage 0: 0.3 h (CPU)
- Stage 1 (3 seeds baseline): 4.6 h
- Stage 2 (4 screening runs): 5.8 h
- Stage 2 fallback to S01 single-stream trunk: 0 h
- Stage 3: Bypassed (0 h)
- Stage 4: Bypassed / ERM control (0 h)
- Stage 5 (1 modern backbone screened, 3 seeds): 4.6 h
- Stage 5 Finalist 5-Fold OOF (S81-T3, 1 finalist): 7.7 h
- Stage 6 (Ensemble selection + common OOF matrix): 0.8 h
- Stage 7 (TTA + calibration + safety stack): 4.2 h
- Stage 8 (External evaluation read): 1.1 h
- **Total Minimum Compute:** **$\approx 29.1\text{ GPU hours}$**.

### B. Conditional Expected Compute Path (Surviving Representations + Finalist Pipeline)
*Assumes Stage 1 confirms; 1 Stage 2 candidate passes and is confirmed via S78-T3 (2 seeds); 2 Stage 3 biological refinements pass screening/ablation and top survivor is confirmed via S79-T3; Stage 4 Group-DRO runs; Stage 5 screens 3 backbones and generates full 5-fold OOF for 2 finalists via S81-T3; ensemble and safety stack complete:*
- Stage 0: 0.3 h (CPU)
- Stage 1 (3 seeds baseline): 4.6 h
- Stage 2 (4 screening runs): 5.8 h
- Stage 2 survivor confirmation (S78-T3, 2 seeds): 3.1 h
- Stage 3 (passing biological refinements Tier 1/2): 5.6 h
- Stage 3 biological survivor confirmation (S79-T3, 2 seeds): 2.8 h
- Stage 4 (Group-DRO, 3 seeds): 4.6 h
- Stage 5 (ConvNeXt-V2, EffNetV2, SwinV2, 3 seeds each): 16.1 h
- Stage 5 Finalist 5-Fold OOF (S81-T3, 2 finalists): 16.9 h
- Stage 6 (Ensemble selection + common OOF matrix verification): 0.8 h
- Stage 7 (TTA + calibration + safety stack): 4.2 h
- Stage 8 (External evaluation read): 1.1 h
- **Total Expected Compute:** **$\approx 65.9\text{ GPU hours}$** (Scheduled across 8 nightly execution blocks of $\le 9.5\text{ h}$).

### C. Worst-Case Compute Path (Full Screening + All Tier 2 Ablations + 3 Finalists 5-Fold + Exploratory Queue)
*Assumes every Stage 2 & 3 candidate passes to Tier 2 local ablations and multi-seed confirmations, full 5-fold cross-fitting for 3 finalists via S81-T3, and all exploratory runs X1-X5:*
- Stage 0: 0.3 h (CPU)
- Stage 1 (3 seeds baseline): 4.6 h
- Stage 2 (4 screening runs + S78-T3 confirmation): 8.9 h
- Stage 3 (Full 9 screening runs + Tier 2 local ablations + S79-T3): 15.2 h
- Stage 4 (Group-DRO, 3 seeds): 4.6 h
- Stage 5 (ConvNeXt-V2, EffNetV2, SwinV2, 3 seeds each): 16.1 h
- Stage 5 Finalist 5-Fold OOF (S81-T3, 3 finalists): 26.9 h
- Stage 6 (Ensemble selection + common OOF matrix + morphology gate): 1.3 h
- Stage 7 (TTA + calibration + safety stack): 4.2 h
- Stage 8 (External evaluation read): 1.1 h
- Secondary / Exploratory queue (V5-X01 through V5-X05): 6.5 h
- **Total Worst-Case Compute:** **$\approx 98.5\text{ GPU hours}$** (Scheduled across 11 nightly execution blocks).

---

## 5. Non-Negotiable Invariants

1. **Strict Gating:** Never launch multi-seed or 5-fold jobs for an experiment that has not passed its single-seed Tier 1 screening gate.
2. **Memory Protection:** Maximum batch size 16 with gradient accumulation 2 is mandatory for SwinV2-384 and dual-stream models to guarantee peak VRAM remains $\le 6.4\text{ GB}$ (leaving $\ge 2.15\text{ GB}$ headroom).
3. **No Reserved Cohort Reads:** The reserved cohort (read 8 times in V4) is permanently barred from V5 execution.
4. **No Test Snooping:** Neither the HAM locked test set nor the S75 external cohort may be accessed during development or model selection.
