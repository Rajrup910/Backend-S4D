# V7 Open Questions, Clinical Approvals & Pre-Execution Gates

> **Precedence (11 Nov 2026).** `V7_FINAL_MASTER_RESEARCH_PLAN.md` **section 14 controls** wherever this document differs. In particular: the promotion rule is P1-P3 (plan 10.4, 14.3) and Gate A2 is a secondary read; MIDAS is a sealed **retrospective** external biopsy cohort, all ages only, and no prospective or consecutive external series exists (14.5); HIBA and the retired splits are training-only; nothing here is "certified", "frozen" or "validated" until the owner signs Gate 0 and the runs exist; the data gate is audited with stated lower bounds, never leak-free.

**Project:** Scan4Disease (S4D) / Backend-S4D  
**Document Designation:** Governance, Clinical Collaboration & Pre-Training Checklist  
**Date:** 8 November 2026  
**Status:** Certified Review Checklist (Gated on Human Owner Sign-Off)

---

## 1. Executive Summary & Purpose

This document provides a transparent, unambiguous audit of:
1. What remains scientifically or empirically unknown prior to training.
2. What decisions require explicit human owner and clinical collaborator sign-off.
3. The concrete pre-flight conditions that must be satisfied before any GPU training or code modification is initiated.

Under the standing operating rules of Project Scan4Disease:
> **No model training, dataset modification, or code alteration may proceed until the human project owner explicitly signs off on these approval gates.**

---

## 2. What Remains Empirically Unknown (Prior to V7 Execution)

While the multi-agent forensic and literature audit has resolved theoretical and mathematical ambiguities, several empirical outcomes cannot be known without controlled execution:

| Domain | Unresolved Empirical Question | Why It Cannot Be Known Prior to Execution | Experimental Phase Designed to Resolve It |
|---|---|---|---|
| **PanDerm LP-FT** | Will two-stage LP-FT ($\gamma=0.80$, base LR $2.5\times 10^{-4}$) restore under-40 ranking capacity ($\Delta \text{pAUC} \ge +0.050$), or will it merely confirm that domain MIM adds general representation robustness without selective youth benefit? | V6 Claim 3 was falsified, but PanDerm was frozen ($\eta_0 = 7.11\times 10^{-7}$). Only running Phase 3 unconfounds this question. | **Phase 3 (Foundation Tuning Ladder)** |
| **High-Resolution Margin Crops (Arm HR)** | Will dual-stream high-resolution margin cropping at pixel scale with Top-$K$ focal pooling improve clinical sensitivity on real young SSM cases without inflating false-positive biopsy referrals on dysplastic nevi? | The physical sampling rationale is mathematically sound, but true clinical specificity depends on the empirical activation of the distance-biased cross-attention layer on benign mimics. | **Phase 2 & Phase 6 (Arm HR Evaluation)** |
| **MaxViT-Tiny Complementarity** | Does MaxViT-Tiny @ 384 px provide orthogonal error diversity to the ConvNeXt/PanDerm/DINOv2 trio under nested simplex PGD, or will its weight collapse to zero ($w^* < 0.05$)? | Screened in V6-3 with high test score (0.7525), but multi-fold common OOF matrix predictions across all 12,235 images (7,002 lesions, 6,989 lesion groups) do not yet exist. | **Phase 4 (Simplex PGD Ensembling)** |
| **Privileged Concept Reliability (Arm C)** | Are automated MONET / expert dermoscopic concept pseudo-labels (strictly excluding contaminated DermFM/DermLIP models) sufficiently uncorrupted by elderly actinic elastosis shortcuts to prevent the student from learning demographic proxies? | Concept annotations on dermoscopy exhibit moderate reader agreement ($\kappa \approx 0.50\text{--}0.72$). Phase 6's Sandwiched Wald Test ($W > 10.83$) is specifically designed to falsify this arm if shortcuts appear. | **Phase 6 (Conditional Arm C)** |
| **Patient Context Bimodality (Arm D)** | Does Ledoit-Wolf outlier scoring on verified multi-lesion encounters ($N_k \ge 3$) improve melanoma detection under real clinic presentations without degrading single-lesion performance? | Validated on ISIC 2020, but real-world clinic presentations are $>60\%$ single lesions. The bypass rule ($w(N_k)=0$ for $N_k \le 2$) must be verified on multi-archive streams. | **Phase 6 (Conditional Arm D)** |
| **New External Hospital Performance** | What is the exact generalization drop on an untouched hospital cohort with consecutive presentation prior ($\pi \approx 1.67\%$)? | HIBA was unblinded in V6-6. True out-of-domain clinical transportability can only be measured after locking V7 and evaluating on a pristine hospital dataset ($E \ge 71$ events, $N \ge 4,250$). | **Phase 9 (Tier C2 External Hospital Read)** |

---

## 3. Mandatory Decisions Requiring Human Owner & Clinical Collaborator Sign-Off

The following decisions involve clinical governance, compute allocation, or intellectual property boundaries, and **require explicit approval from the project owner**:

### Approval Decision 1: Confirmation of V6 as Immutable Benchmark
* **Proposal:** Retain `v6_vote` (uniform 4-member ensemble locked at commit `4e6d32b`) as the permanent production incumbent. If V7 fails Gate A2 ($\Delta \text{pAUC}_{\text{McClish}} < +0.050$), V6 remains the deployed clinical triage model.
* **Owner Action Required:** Confirm approval to maintain V6 as the locked reference system.

### Approval Decision 2: Elimination of Legacy CNN Re-Training from Core Budget
* **Proposal:** Formally allocate GPU compute budget ($\approx 80.4\text{ hours}$ on RTX 5050) to foundation model adaptation (PanDerm LP-FT, DINOv2-Reg) and 384 px dual-stream inputs, while relegating legacy CNNs (EfficientNet-B0/B3, DenseNet-121, ResNet-50) to an optional low-cost screen.
* **Owner Action Required:** Approve dedicating compute to foundation models over legacy CNN re-runs.

### Approval Decision 3: Permanent Disqualification of Contaminated Models
* **Proposal:** Permanently ban DermFM-Zero, DermLIP, and DermINO from all V7 evaluation pipelines due to fatal pretraining overlap with MSKCC, BCN20000, and ISIC evaluation splits.
* **Owner Action Required:** Confirm disqualification of contaminated models.

### Approval Decision 4: PanDerm Academic Research Boundary & Licensing
* **Proposal:** Train PanDerm ViT-B/16 strictly as an internal academic research baseline under its CC BY-NC-ND 4.0 license. Gitignore all fine-tuned checkpoints (`ml/checkpoints/panderm_*.pt`). Restrict commercial deployment pathways to Apache 2.0 models (DINOv2-Reg and ConvNeXt).
* **Owner Action Required:** Acknowledge and approve the non-commercial research boundary for PanDerm.

### Approval Decision 5: Acquisition Strategy for New External Hospital Cohort (Tier C2)
* **Proposal:** Authorize clinical collaboration outreach to obtain an untouched, consecutive clinical dermoscopy cohort ($N \ge 4,250$ lesions, securing $E \ge 71$ young escalating events) complying with TRIPOD+AI.
* **Clinical Collaborator Action Required:** Sign off on multi-center data transfer agreements and ethical board approvals.

---

## 4. Pre-Training Checklist: Conditions That Must Be Satisfied Before Code Execution

Before any training script (`python ml/training/train.py` or `scripts/run_v7_*.py`) is launched, the following pre-flight assertions must be verified:

```
[ PRE-TRAINING GATE CHECKLIST ]
  [ ] 1. Git branch verified: Operating strictly on local branch (v5-confirmation-review2 or relay/gemini-*).
  [ ] 2. Zero main branch commits: No direct commits to main; no git push executed.
  [ ] 3. Phase 0 hash lock verified: results/v7/phase0_baseline_hashes.json matches all frozen artifacts.
  [ ] 4. master_v7.csv constructed: Overlap guard achieves 100.0% pass rate on 200 synthetic duplicates.
  [ ] 5. No confirmed lesion group crosses CV folds (data gate receipt). Patient overlap is NOT verified: HAM and BCN carry no patient id (plan 14.4).
  [ ] 6. Transform parity verified: CenterCrop(224) permanently replaced with aspect-preserving reflection padding.
  [ ] 7. Micro-batch profiling complete: Standardized B_micro=8, S_accum=4 on RTX 5050 (peak VRAM <= 5.50 GB).
  [ ] 8. Safe headroom certified: Windows commit limit headroom >= 6.0 GB; C: drive free space >= 20.0 GB.
  [ ] 9. Missingness invariance tested: Synthetic missing batches verify ||z_joint(x, null) - z_vis(x)||_inf < 10^-7.
  [ ] 10. Directional attenuator verified: Visual suspicion p_vis >= 0.35 verified to experience 0 logit penalty.
  [ ] 11. Human owner sign-off recorded: Written approval to initiate Phase 1 and Phase 2.
```

---

## 5. Formal Decision Table & Recommendations

| Strategic Decision | Authoritative Recommendation | Evidence Status | Rationale | Reversal Condition (What Would Change Decision) |
|---|---|---|---|---|
| **V6 Incumbent** | **RETAIN AS BENCHMARK** | `VALIDATED_ON_DISK` | HIBA Macro-F1 $0.7348$, lead $+0.1633$ over V5 on the same lesions. OOF $0.7433$ and HIBA $0.7348$ are different cohorts, not a retention ratio. Under 40 on HIBA: V6 18/20, V5 19/20. | Replaced only if V7 passes Gate A2 ($\Delta \text{pAUC}_{\text{McClish}} \ge +0.050$) and LOAO transportability. |
| **Legacy CNN Re-Test** | **DEFER TO OPTIONAL SCREEN** | `NOT_FAIRLY_RETESTED` | Gontijo-Lopes (ICLR 2022) proves objective diversity dominates; legacy CNNs cost 24 GPU hours with low expected gain. | Re-run if candidate screen shows an older CNN has $w^* \ge 0.10$ in nested simplex. |
| **Contaminated FMs (DermFM-Zero, DermLIP)** | **PERMANENTLY DISQUALIFY** | `VERIFIED_CONTAMINATED` | Pretraining corpora explicitly ingested MSKCC, BCN20000, and ISIC union. Direct evaluation is circular. | Admitted only if authors release a verifiably clean checkpoint trained excluding all evaluation archives. |
| **PanDerm LP-FT** | **EXECUTE PHASE 3** | `OPTIMIZATION_DEFECT_PROVED` | Lower-trunk freeze was an optimization defect ($\eta_0 = 7.11\times 10^{-7}$). 2-stage LP-FT enables a fair, unconfounded test. | Prune PanDerm if fine-tuning causes feature distortion (CKA $< 0.70$) or fails Gate C-VAL. |
| **High-Res Margin Cropping** | **CONDITIONAL ARM HR** | `DESIGNED_ABLATION` | Fixes 34.375% (34.31% raster) FOV evaluation truncation via reflection padding in core trunk; dual-stream cross-attention is an ablation arm. | Admitted to final fusion only if Gate HR passes ($\Delta \text{Macro-F1} \ge -0.010$ and $\Delta \text{pAUC}_{<40} \ge +0.030$). |
| **AMPS Metadata Fusion** | **V7_OPTIONAL_METADATA** | `THEORETICALLY_BOUNDED` | Neutral fallback eliminates missingness bias ($\Delta z_{\text{meta}} \equiv 0$); enforces one-sided escalation guarantee ($s_{\text{final}}^{\text{esc}} \ge s_{\text{vision}}^{\text{esc}}$). | Prune metadata module if Gate 5 fails ($\Delta \text{Sens}_{<40} < 0.000$) on bootstrap. |
| **Privileged Concepts (CoFiDA-M)** | **CONDITIONAL ARM C** | `METHODOLOGICALLY_CORRECTED` | Corrected inverted BM-KD sign; student executes image-only. Must be tested independently to isolate causal gain. | Admitted to final fusion only if Sandwiched Wald $W > 10.83$ and held-out $\Delta \text{pAUC} \ge +0.030$. |
| **Patient Context (Arm D)** | **CONDITIONAL ARM D** | `CLINICALLY_VALIDATED` | Single-lesion bypass ($w(N_k)=0$) eliminates distortion on $>60\%$ of encounters; Ledoit-Wolf detects outliers on multi-lesion galleries. | Admitted only if ISIC 2020 validation shows statistically significant bootstrap gain ($95\%$ CI $> 0$). |
| **Simplex PGD Stacking** | **ADOPT OVER UNCONSTRAINED** | `MATHEMATICALLY_PROVED` | Unit simplex $\mathbf{w} \in \Delta^{M-1}$ minimizes $\ell_1$-norm, providing minimax Wasserstein-1 robustness against covariate shift. | Fallback unconditionally to uniform soft voting ($w_m = 1/M$) if nested gain $< +0.0050$. |
| **New External Validation (Tier C2)** | **MANDATE EVENT SIZING** | `STATISTICALLY_MANDATED` | Event-driven accrual: Riley et al. (BMJ 2024) proves flat slogans fail; detecting MCID $+0.050$ at $80\%$ power requires $E \ge 71$ events ($N \ge 4,250$ expected [51.5% yield], $N \ge 4,666$ for 80% assurance). | Reduce cohort size only if high-prevalence tertiary oncology clinic provides $E \ge 71$ in fewer lesions. |

---

This document establishes the binding governance and pre-execution boundaries for Project Scan4Disease V7.

