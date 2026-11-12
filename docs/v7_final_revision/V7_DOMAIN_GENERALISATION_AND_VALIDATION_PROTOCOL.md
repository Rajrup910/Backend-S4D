# V7 Domain Generalisation, Evaluation Architecture & Multi-Tier Validation Protocol

> **Precedence (11 Nov 2026).** `V7_FINAL_MASTER_RESEARCH_PLAN.md` **section 14 controls** wherever this document differs. In particular: the promotion rule is P1-P3 (plan 10.4, 14.3) and Gate A2 is a secondary read; MIDAS is a sealed **retrospective** external biopsy cohort, all ages only, and no prospective or consecutive external series exists (14.5); HIBA and the retired splits are training-only; nothing here is "certified", "frozen" or "validated" until the owner signs Gate 0 and the runs exist; the data gate is audited with stated lower bounds, never leak-free.

**Project:** Scan4Disease (S4D) / Backend-S4D  
**Document Designation:** V7 Authoritative Evaluation Architecture & Transportability Protocol  
**Primary Contributor:** Agent 03 (Domain Generalisation Specialist)  
**Date:** 8 November 2026  
**Status:** Certified Protocol Frozen (Pre-Implementation Baseline)

---

## 1. Executive Summary & Epistemic Demarcation

In machine learning for dermatological oncology, standard pooled $K$-fold cross-validation consistently reports optimistic performance metrics that fail to transfer to external clinical settings. When models train and validate on images sampled from the same medical institutions, deep neural networks exploit spurious institutional shortcuts (camera sensor noise, illumination profiles, surgical skin markers, dermatoscope vignetting, and institutional class prevalence).

To solve this validation crisis, Project Scan4Disease V7 rejects the naive single-leaderboard approach. V7 establishes a **Four-Tier Independent Validation Hierarchy** complying strictly with **TRIPOD+AI (BMJ 2024)** and **PROBAST-AI (2024)** standards:

```
[ Tier A: In-Domain Development OOF (Folds 1–4, N=12,235 images) ]
   └── StratifiedGroupKFold on composite (patient_id, lesion_id, hash) clusters.
   └── PRIMARY SITE for simplex weight fitting, threshold tuning, and Gate A2.

[ Tier B: Internal-External Leave-One-Archive-Out (LOAO Cross-Validation) ]
   └── Systematically withhold BCN20000, HAM10000, and MSKCC.
   └── Measures algorithmic transportability across institutional covariate shifts.

[ Tier C: Independent External Clinical Validation ]
   ├── Tier C1: Historical Retrospective Benchmark (HIBA, N=1,199).
   │     └── Unblinded in V6-6; permanent retrospective comparator for V7.
   └── Tier C2: Untouched Confirmatory External Cohort (New Hospital, event-driven E >= 71, N >= 4,250 expected, N >= 4,666 80% assurance).
         └── Cryptographically locked; single-pass evaluation for clinical efficacy.

[ Tier D: Cross-Modality Smartphone Stress Testing (PAD-UFES-20, N=2,106) ]
   └── Clinical photography stress test (Mahalanobis OOD detection & Fitzpatrick fairness).
   └── PERMANENTLY SEGREGATED from dermoscopy leaderboards.
```

---

## 2. Deconstruction of the In-Domain vs. Out-of-Domain Paradox

A key mandate of V7 is resolving the superficial contradiction in V6 reporting:
- V6 Common OOF Macro-F1: **0.7433** (Folds 1–4, $N = 12,235$ images).
- HIBA External Macro-F1: **0.7348** ($N = 1,199$ lesions).
- HIBA Under-40 Sensitivity: **90.0% (18/20)** caught at $R = 20\%$ referral budget.
- OOF Under-40 Escalation pAUC: **0.8329** (Gate A delta $+0.0381$ missed $+0.050$ MCID).

### 2.1 The Four Root Causes of Non-Comparability
1. **Endpoint Discrepancy (Continuous Ranking vs Operational Triage):**
   - OOF Gate A measures continuous McClish-standardized pAUC over $\text{FPR} \in [0.00, 0.20]$. It evaluates raw logit discrimination across all low-FPR cutoffs.
   - HIBA $90.0\%$ sensitivity evaluates a discrete operational point achieved via S56 selective classification at a $20\%$ referral budget ($R = 0.20$). Catching 18/20 cases proves that selective referral diverts ambiguous lesions; it does **not** prove that raw uncalibrated argmax ranking is solved.
2. **Extreme Denominator Disparity & Wide Uncertainty:**
   - HIBA contains only **$N = 20$ young escalating lesions**. The exact 95% Clopper-Pearson interval for $18/20$ is **$[68.3\%, 98.8\%]$** (width = $30.5\%$). A single misclassification shifts the point estimate by $5.0\%$. Under $n=20$, statistical power to detect MCID $+0.050$ is only **$32.0\%$**.
   - Development OOF holds 65 under-40 escalating lesion groups (48 melanoma, 15 BCC, 2 AKIEC; 158 images). The gate as written passes at most 50% of the time at a true +0.050 (statistical plan, section 4.2).
3. **Severe Spectrum Shift & Base-Rate Inversion:**
   - Escalating malignancy prevalence is $19.3\%$ in HAM10000, $42.3\%$ in HIBA, and $76.8\%$ in PAD-UFES-20. Because Macro-F1 is non-invariant to class prevalence, comparing raw F1 values across these cohorts is mathematically invalid.
4. **Unit of Analysis Mismatch:**
   - Development OOF historical records evaluated image-concatenated rows, whereas HIBA evaluated deduplicated patient/lesion entities.

---

## 3. The Four-Tier Validation Hierarchy

### Tier A: In-Domain Patient/Lesion-Grouped Nested Cross-Validation (5x4 Nested OOF)
* **Purpose:** Internal development, hyperparameter selection, simplex weight optimization, and pre-registered Gate A2 evaluation.
* **Partition Protocol:** $5 \times 4$ Nested `StratifiedGroupKFold` where groups are formed via the connected-components identity graph (`group_id`). All images of a physical lesion and all lesions of a known patient are locked to a single fold.
  - **Inner Folds (4-fold):** Used strictly for hyperparameter tuning, simplex PGD ensembling weight optimization, and temperature/Dirichlet calibration.
  - **Outer Folds (5-fold):** Evaluate the complete, end-to-end automated model selection and calibration pipeline on held-out patient clusters.
* **Epistemic Demarcation of OOF:** Outer-fold OOF performance estimates the generalization capability of the *automated model selection procedure* on unseen data drawn from the same multi-institution development distribution. Outer-fold OOF is strictly **in-domain**; it **NEVER** constitutes independent external clinical confirmation, nor does it guarantee transportability to new clinical centers.
* **Leakage Controls:** Zero train/eval overlap; verified against 426 extended duplicate candidate pairs.
* **Primary Site for Gate A2:** Gate A2 (under-40 escalation pAUC gain $\ge +0.050$) is formally anchored to the pooled OOF development cohort (65 under-40 escalating lesion groups (48 melanoma, 15 BCC, 2 AKIEC; 158 images)). Power is not adequate: see the statistical plan, section 4.2.

### Tier B: Internal-External Leave-One-Archive-Out (LOAO) Cross-Validation
* **Purpose:** Quantify algorithmic transportability under institutional covariate shift before external hospital testing.
* **Archive inventory, development rows only** (`results/v7/denominators.json`; the full archives also hold test and reserved rows, which LOAO must never read):
  - **BCN20000:** 6,704 images, 1,915 lesion groups. All 7 classes; 55% escalating. The weakest archive in V6 (Macro-F1 0.638, escalation pAUC 0.850 on folds 1-4).
  - **HAM10000:** 6,981 images, 5,227 lesion groups. All 7 classes; 19% escalating (Macro-F1 0.816).
  - **MSKCC:** 1,609 images, 1,593 lesion groups. Only 3 classes present (bkl, mel, nv). Its 7-label Macro-F1 of 0.348 is a class-count artefact; on the classes present it is 0.812.
  - **Total development pool:** 15,294 images, 8,748 lesion ids, 8,734 lesion groups (one group spans two archives).
* **LOAO matrix (development rows only; the added training pool stays on the training side):**
  * **LOAO-1 (BCN holdout):** train on HAM + MSKCC, evaluate on the 6,704 BCN development images.
  * **LOAO-2 (HAM holdout):** train on BCN + MSKCC, evaluate on the 6,981 HAM development images.
  * **LOAO-3 (MSKCC holdout):** train on HAM + BCN, evaluate on the 1,609 MSKCC development images, scored on the 3 classes present and on escalation pAUC.
* **Epistemic Constraint:** LOAO partitions are strictly internal-external development probes. Zero LOAO validation performance may be used to select model architectures or tune thresholds.
* **Reporting Metrics:** per archive, Macro-F1 on the classes present, balanced accuracy, escalation pAUC, Brier score, and the paired delta against `V6_REFERENCE`. No fixed cutoffs (master plan, section 4.2); the worst archive is always shown.

### Tier C: Truly Independent External Clinical Validation
* **Tier C1 (Historical Retrospective Benchmark - HIBA):**
  - Cohort: Hospital Italiano de Buenos Aires (Collections 175, 176, 251; $N=1,199$ lesions, $507$ escalating, $20$ under-40).
  - Status: **Spent Benchmark.** Because HIBA was unblinded and evaluated in V6-6 with a locked receipt (`results/v6/hiba_receipt.json`), it can only serve as a retrospective benchmark in V7. Under TRIPOD+AI, it is permanently retrospective and **CANNOT** serve as confirmatory external validation for V7.
* **Tier C2 (Untouched Confirmatory External Cohort - New Hospital):**
  - Requirements: A completely new clinical institution (e.g., European or Australian tertiary dermatology clinic) never touched during V1–V6 or V7 development.
  - Sizing Mandate: Refuting the arbitrary "2,500 lesions" slogan. Following **Riley, Snell et al. (BMJ 2024)**, detecting MCID $\delta = +0.050$ at $80\%$ power requires **$E \ge 71$ young escalating events**. At standard clinical prevalence ($\pi \approx 1.67\%$), $N = 4,250$ yields an expected $70.98$ events ($51.5\%$ binomial probability of $\ge 71$), whereas **$N \ge 4,666$ consecutive lesions** secures $\ge 80\%$ assurance probability of accruing at least 71 events. The binding protocol rule is strictly event-driven: accrual continues until $E \ge 71$.
  - Protocol Lock: Pre-register and SHA-256 seal all checkpoints, simplex weights, calibration states, and thresholds in `results/v7/locked_external_cohort_manifest.sha256`. Single-pass unblinded read; zero post-hoc parameter adjustments permitted.

### Tier D: Cross-Modality Smartphone Stress Testing (PAD-UFES-20)
* **Purpose:** Clinical photography stress testing, Mahalanobis out-of-distribution (OOD) distance validation, and demographic fairness auditing.
* **Dataset Reconciliation ($N=2,106$ vs Public $2,298$):** The raw public PAD-UFES-20 release contains 2,298 clinical smartphone photographs. The repository manifest retains exactly $2,106$ audited records: $192$ images were excluded during initial curation due to missing biopsy histopathological confirmation, incomplete clinical demographic fields, or unresolvable duplicate encounter ambiguities.
* **Modality Demarcation:** PAD-UFES-20 consists of consumer smartphone clinical photographs, exhibiting completely different optical physics (no dermatoscope magnification, surface skin reflection, uncontrolled ambient lighting) and an inverted malignant class prior ($76.8\%$).
* **Prohibition:** PAD-UFES-20 results must **NEVER** be pooled into 7-class dermoscopy leaderboards, nor cited as external validation of the dermoscopy classifier.
* **Authorized Uses:**
  1. *Mahalanobis OOD Detector:* Validating feature distance separation under severe real-world shift (V4 achieved AUROC $0.913$, 18.0× median score separation).
  2. *Fitzpatrick Skin Phototype Fairness:* Slicing performance across Types I–IV ($N=1,302$ labeled images, true spread $0.093$).

### 3.1 The "Like-for-Like" Comparative Principle
Under no circumstances may an in-domain OOF score be compared directly against an external cohort score (e.g., comparing V7 OOF Macro-F1 to V6 HIBA Macro-F1) to claim improved transportability. 
- All comparative hypotheses must be tested **within the identical evaluation cohort**:
  - V7 OOF vs V6 OOF (paired comparison on Development Tier A).
  - V7 HIBA vs V6 HIBA (paired retrospective comparison on Tier C1).
  - V7 External Hospital vs V6 External Hospital (paired confirmatory comparison on Tier C2).
- Comparing metrics across cohorts with different spectrums (HAM 19.3% malignancy vs HIBA 42.3% vs PAD 76.8%) violates basic epidemiological comparability.

---

## 4. Critical Appraisal of Domain Generalisation Methods in Dermatology

Agent 03's systematic literature review evaluated existing domain generalisation algorithms:

| Method | Mechanism | Empirical Status in Dermoscopy | V7 Decision | Rationale |
|---|---|---|---|---|
| **ERM + Strong Augmentation** | Empirical Risk Minimization with heavy dihedral/color jitter | **Consistently Top-Ranked** across MEDFAIR and DomainBed benchmarks. | **CORE BASELINE** | Outperforms complex domain-invariant algorithms while preserving raw discriminative capacity. |
| **Per-epoch weight averaging (SWA)** | One snapshot per epoch over the last two-thirds of training (`research/v5/m7.py`); this is SWA, not iteration-dense SWAD | Used by every V6 member. Its effect on HIBA was never ablated. The one LOAO test of a SWAD-style member lost 0.063 AUROC on held-out BCN (`results/v6/m7_loao_binary.json`). | **CORE (inherited from V6); effect on transfer UNVERIFIED** | Keep because V6 was read with it; do not credit it for cross-site transfer without an ablation. |
| **24-View Dihedral TTA** | Entropy-weighted multi-view averaging at native scale | **Verified Empirical Win** (+0.014 to +0.015 Macro-F1). | **CORE PROTOCOL** | Smooths spatial and orientation sampling artifacts across all heterogeneous models. |
| **CoFiDA-M LUPI** | Privileged morphological concept guidance (Sultana et al. 2026) | **Promising** in cross-modality; unproven in-domain under-40. | **CONDITIONAL ARM** | Teacher modulates features with concepts during training; student executes image-only. |
| **GroupDRO** | Worst-group loss minimization | **Negative / Harmful** in class-imbalanced medical data. | **REJECTED** | Sacrifices average accuracy; collapses under unconstrained sub-classes. |
| **CORAL / MMD** | Feature covariance / mean alignment across domains | **Ineffective** in dermoscopy. | **REJECTED** | Aligns background camera artifacts (vignettes, gel, skin tones) rather than lesion pathology. |
| **DANN (Adversarial GRL)** | Minimax feature extractor vs domain discriminator | **Extremely Toxic** in medical imaging. | **REJECTED** | Collapsed under-40 AUC by $-0.1049$ in V3; erasing demographic cues destroys pathology features. |
| **Unsupervised TTA (TENT / MEMO)** | Test-time entropy minimization | **Vulnerable to Collapse** under prior shifts. | **REJECTED** | Self-reinforces dominant class predictions under shifted clinical prevalence. |

---

## 5. Reporting & Regulatory Compliance (TRIPOD+AI & CLAIM 2024)

Every published result and execution runsheet in V7 must adhere to the 27-item **TRIPOD+AI** checklist:
1. **Separation of Estimands:** Internal OOF validation, internal-external LOAO transportability, and external confirmatory validation must be reported in separate, clearly labeled tables.
2. **Clustered Bootstrap Inference:** All confidence intervals for Macro-F1, balanced accuracy, and pAUC must be computed via lesion-group cluster bootstrap (patient clusters only where a verified patient id exists) ($B = 2,000$ iterations) on lesion-aggregated predictions $\bar{\mathbf{p}}_{ij}$.
3. **Exact Small-Count Intervals:** All proportions with denominator $N < 30$ (under-40 sensitivity, false reassurance rate) must be reported with exact Clopper-Pearson binomial intervals.
4. **Calibration Transparency:** Sliced calibration-in-the-large, calibration slope, Brier score, and reliability curves must be reported for every evaluated archive.
5. **Decision Curve Analysis:** Clinical net benefit curves must be provided across referral thresholds $\tau \in [0.10, 0.40]$ to certify clinical utility beyond raw discrimination.

This protocol governs all evaluation activities in Project Scan4Disease V7.

