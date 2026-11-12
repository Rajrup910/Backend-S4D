# V7 Forensic Repository Audit & Historical Lineage Reconciliation

**Project:** Scan4Disease (S4D) / Backend-S4D  
**Active Working Tree:** `C:\Users\RAJ\Downloads\Capstone review 1`  
**Audit Date:** 8 November 2026  
**Auditor:** Lead Research Integrator (synthesizing Agent 01 Forensics & Agent 12 Statistical Audit)  
**Status:** Certified Authoritative Forensic Audit

---

## 1. Git Repository Topology, Branching & Commit Forensic

### 1.1 Active Branch & HEAD State
- **Active Branch:** `v5-confirmation-review2`
- **Active Local HEAD SHA:** `4e6d32b542999d1d3823807791a4149a16f2af25`
- **HEAD Commit Message:** `V6-6 finished: HIBA external confirmation read, report, decisions, and documentation` (Committed 7 Nov 2026).
- **Working Tree State:** Clean with respect to tracked files (uncommitted edits restricted to `.relay/GEMINI_NOTES.md` and unstaged line in `CHANGELOG.md`; untracked files restricted to newly authored research plans in `docs/` and reports in `docs/v7_final_revision/`).

### 1.2 Remote Topology & Resolution of the Remote vs Local Discrepancy
- **Remote Tracking URL:** `origin` -> `https://github.com/Rajrup910/Backend-S4D.git`
- **Remote `origin/main` SHA:** `86519659944e04fc922a914d6b507465ce9164e8` (Committed 3 Nov 2026, *"README: separate the tool icons from the hero animation"*).
- **Remote `origin/v5-confirmation-review2` SHA:** `91ce96eb107a9cc041d153bf23a7d8122d8602c7` (Committed 5 Nov 2026).
- **Forensic Explanation of Discrepancy:**  
  The public GitHub `main` branch describes V6 as "frozen, no results" because public `main` was pinned at commit `8651965` prior to V6 execution. All V6 development, training, OOF ensembling, and HIBA external validation were executed locally on branch `v5-confirmation-review2` across **14 local commits** (`b50270b` through `4e6d32b`).
  Per the project's standing multi-agent rules (`.relay/HANDOFF.md` and `CLAUDE.md`):
  > *"Work only on the current branch. Never run git checkout, git switch, git merge, git rebase, git reset or git push, and never commit on main. Merging is Claude's job after review. Next action: Review and push branch v5-confirmation-review2."*
  Therefore, the local branch is intentionally ahead of remote `origin` pending human/Claude review.

### 1.3 Resolution of the "404 Not Found" Result Artifacts
Earlier remote audits reported 404 HTTP errors when querying GitHub URLs for key V6 result files. A comprehensive disk and hash audit verifies that **100% of these files exist locally on disk, are valid JSON, and are committed in local git history**:
1. `results/v6/e_primary.json`: Present on disk (Committed in `4d406ba`). Evaluates Folds 1–4 held-out images ($N=12,235$; 7,002 lesions).
2. `results/v6/claim3.json`: Present on disk (Committed in `5c0e749`). Evaluates the PanDerm under-40 representation hypothesis.
3. `results/v6/hiba/hiba_report.json`: Present on disk at `results/v6/hiba/hiba_report.json` (Committed in `4e6d32b`). Evaluates external validation on $N=1,199$ HIBA lesions.
4. `results/v6/hiba_receipt.json`: Present on disk (Committed in `4e6d32b`). Cryptographic verification receipt of the locked HIBA read.
5. `results/v6/refit/v6_vote/per_band_report.json`: Present on disk at `results/v6/refit/v6_vote/per_band_report.json` (Committed in `4d406ba`). Reports per-age-band metrics for the deployed V6 vote.

---

## 2. Locked V6 Incumbent Roster & Execution Baseline

The incumbent system against which all V7 innovations must be evaluated is the **V6 Deployed Vote (`v6_vote`)**, officially locked in `results/v6/e_primary.json`, `docs/V6_RUNSHEET.md`, and commit `4e6d32b`.

### 2.1 Incumbent Model Members (4 Heterogeneous Encoders)
1. **`anchor_h7` (ConvNeXt-Tiny @ 224 px):** Supervised ImageNet-22k baseline, 7-class head, cross-entropy with anti-dilution youth sampling, SWAD weight averaging.
2. **`anchor384_h7` (ConvNeXt-Tiny @ 384 px):** Supervised ImageNet-22k/1k baseline, native 384×384 px input, 7-class head, SWAD weight averaging.
3. **`panderm_b_h7` (PanDerm-Base ViT-B/16 @ 224 px):** Dermatology domain-specific foundation model (Yan et al., Nature Medicine 2025; 2.1M images), masked latent modeling + CLIP alignment, 7-class head, SWAD weight averaging.
4. **`dinov2_b_h7` (DINOv2 ViT-B/14 @ 224 px):** General vision self-supervised foundation model (Oquab et al., 2023; LVD-142M), self-distillation, 7-class head, SWAD weight averaging.

### 2.2 Roster Clarifications & Retractions
- **MaxViT-Tiny:** MaxViT-Tiny was screened in Phase V6-3 (`15_train_maxvit_tiny_retry.log`, test Macro-F1 0.7525) but was **not promoted** into the final V6-4/V6-5/V6-6 vote. It remains a candidate for V7 under Gate M-VAL.
- **SigLIP 2-Base:** Evaluated in V6-3 (`runs/siglip2_b_h7_f0_s42_v63.json`) and rejected due to marginal delta $\Delta = +0.0021$ failing the promotion threshold.
- **Weighting Policy:** Uniform arithmetic soft voting ($w_m = 0.25$) is the official deployed baseline. Nested simplex forward selection showed a marginal $+0.0088$ Macro-F1 gain, but its 95% bootstrap CI crossed zero (`[-0.00004, +0.0182]`); therefore, uniform soft voting remains the locked comparator.
- **Test-Time Augmentation:** 24-view dihedral/scale TTA with entropy-weighted pooling per member at its native training resolution.

---

## 3. Verified Quantitative Results: V6 Common OOF vs HIBA External Read

| Evaluation Horizon | Dataset & Partition | Sample Size ($N$) | Metric / Endpoint | Incumbent (`v6_vote`) | Comparator (V5 / Baseline) | Paired Delta ($\Delta$) | 95% Confidence Interval | Outcome / Gate Status |
|---|---|---|---|---|---|---|---|---|
| **In-Domain OOF** | Multi-Archive Dev (Folds 1–4) | 12,235 images (7,002 lesions, 6,989 lesion groups) | 7-Class Macro-F1 | **0.7433** | 0.7005 (V5) | **+0.0427** | $[+0.0245, +0.0618]$ | **E-Primary PASSED** |
| **In-Domain OOF** | Multi-Archive Dev (Folds 1–4) | 12,235 images (7,002 lesions, 6,989 lesion groups) | All-Age pAUC (FPR $\le 0.20$) | **0.8890** | 0.8553 (V5) | **+0.0337** | $[+0.0263, +0.0415]$ | **Co-Primary PASSED** |
| **In-Domain OOF** | Multi-Archive Dev (Folds 1–4) | 12,235 images (7,002 lesions, 6,989 lesion groups) | Escalation Sensitivity | **82.55%** | 81.40% (V5) | **+1.15%** | $[+0.05\%, +2.31\%]$ | **No-Harm PASSED** |
| **In-Domain OOF** | Under-40 Dev Cohort | 65 under-40 escalating lesion groups (48 melanoma, 15 BCC, 2 AKIEC) | Under-40 pAUC (FPR $\le 0.20$) | **0.8329** | 0.7949 (V5) | **+0.0381** | $[+0.0107, +0.0711]$ | **Gate A FAILED** (Missed +0.050 MCID) |
| **In-Domain OOF** | Excl. 380 Duplicate Lesions | 11,944 lesions | 7-Class Macro-F1 | **0.7441** | 0.7008 (V5) | **+0.0433** | $[+0.0242, +0.0626]$ | **Leak Sensitivity PASSED** |
| **In-Domain OOF** | Excl. 426 Extended Duplicates | 11,905 lesions | 7-Class Macro-F1 | **0.7442** | 0.7008 (V5) | **+0.0434** | $[+0.0242, +0.0626]$ | **Extended Leak PASSED** |
| **External Read** | HIBA Clinical Cohort | 1,199 lesions | Full-Coverage Macro-F1 | **0.7348** | 0.5715 (V5) | **+0.1633** | $[+0.1235, +0.2064]$ | **External Validation PASSED** |
| **External Read** | HIBA Clinical Cohort | 311 mel/nevus | Mechanism pAUC (FPR $\le 0.20$) | **0.8104** | 0.7530 (V5) | **+0.0574** | $[+0.0059, +0.1099]$ | **External Mechanism PASSED** |
| **External Read** | HIBA Clinical Cohort | 507 escalating | All-Age Escalation Sens (@ 20%) | **91.3%** (463/507) | 83.2% (422/507) | **+8.1%** | $[+4.8\%, +11.5\%]$ | **Workload Target PASSED** |
| **External Read** | HIBA Under-40 Cohort | 20 young escalating | Under-40 Sens (@ 20% budget) | **90.0%** (18/20) | 70.0% (14/20) | **+20.0%** | Exact CP: $[68.3\%, 98.8\%]$ | **S70 Target ($\ge 85.5\%$) MET** |
| **External Read** | HIBA Under-40 Cohort | 20 young escalating | False Reassurance Rate (AMCP) | **0.0%** (0/20) | 15.0% (3/20) | **-15.0%** | Exact CP: $[0.0\%, 16.8\%]$ | **Zero Observed Misses** |

---

## 4. Resolution of the In-Domain vs. Out-of-Domain Ambiguity

A critical epistemic correction in this audit is resolving the superficial claim that *"OOD performance on HIBA (0.7348 / 90% sens) exceeded in-domain OOF performance (0.7433 / Gate A fail)"*.

### 4.1 Endpoint Non-Equivalence
- **OOF Gate A Endpoint:** Continuous McClish-standardized pAUC over $\text{FPR} \in [0.00, 0.20]$. It evaluates discrimination across all low-false-positive decision thresholds on uncalibrated model outputs. V6 scored $0.8329$, improving by $+0.0381$, which fell short of the aggressive $+0.050$ MCID.
- **HIBA Under-40 Sensitivity Endpoint:** Discrete operational sensitivity at a fixed **20% selective referral budget** ($R = 0.20$) under S56 selective classification. The model caught 18/20 young escalating lesions because uncertain lesions were diverted to dermatologist referral. This confirms selective referral efficacy; it does **not** prove that raw uncalibrated argmax ranking is resolved.

### 4.2 Statistical Power & Denominator Disparity
- **HIBA Sample Power:** HIBA contains only **$N = 20$ young escalating lesions**. The exact 95% Clopper-Pearson interval for $18/20$ is **$[68.3\%, 98.8\%]$** (a width of $30.5\%$). A single misclassification shifts the point estimate by $5.0\%$. Under this small sample, statistical power to detect MCID $+0.050$ was only **$32.0\%$**.
- **OOF sample:** 65 under-40 escalating lesion groups (48 melanoma, 15 BCC, 2 AKIEC; 158 images). Power to exclude zero at a true +0.050 is 76.8%; the probability of meeting a +0.050 point-estimate target is at most 50%.
- **Conclusion:** HIBA is underpowered for gating; Gate A2 must remain anchored to the pooled OOF development cohort (65 under-40 escalating lesion groups, 48 melanoma).

### 4.3 Generalization vs. Collapse Across Generations
- In V5, Macro-F1 collapsed by **$-0.1290$** between OOF ($0.7005$) and HIBA ($0.5715$) due to brittle multi-task ranking heads (`m4`, `twostep`) that overfit development artifacts.
- V6 Macro-F1 is 0.7433 on development folds 1-4 and 0.7348 on HIBA. These are different cohorts with different class mix, so their ratio is not a retention figure, and no ablation ties the HIBA result to weight averaging.

---

## 5. Status of V1 Legacy CNN Baselines

A central question addressed in this audit is: **"Why were EfficientNet-B0, EfficientNet-B3, DenseNet-121, and ResNet-50 excluded from V6?"**

### 5.1 Were They Fairly Re-Tested?
**NO.** Their exclusion was an operational decision driven by compute budgets, not empirical rejection on the modern pooled dataset:
1. **Historical Baseline (V1):** In V1, the 6 CNNs trained on single-archive HAM10000 achieved test Macro-F1s: ConvNeXt-Tiny (0.7459), ConvNeXt-Small (0.7250), EfficientNet-B0 (0.7264), ResNet-50 (0.7058), DenseNet-121 (0.6969), EfficientNet-B3 (0.6894). The uniform soft vote scored $0.7718$ ($+0.0259$ over ConvNeXt-Tiny alone).
2. **Phase V6-3b Abandonment:** Phase V6-3b was formally specified in `docs/V6_RUNSHEET.md` to retrain legacy CNNs on the expanded pool. However, Section 0.9 officially superseded V6-3b to dedicate remaining GPU budget on the RTX 5050 to foundation models (PanDerm and DINOv2).
3. **Epistemic Classification:** The legacy CNNs are classified as **`NOT FAIRLY RE-TESTED`** on the V6 multi-archive pool.

### 5.2 Should They Be Re-Tested in V7?
- **Theoretical Basis (Gontijo-Lopes et al., ICLR 2022):** Error diversity is primarily driven by pretraining objective and pretraining data divergence, not architectural topology under the same supervised objective.
- **Compute Reality:** Retraining 5 legacy CNNs across 5 folds would require 25 runs ($\approx 20\text{--}24$ GPU hours) on the RTX 5050 (8.55 GB VRAM).
- **V7 Recommendation:** Legacy CNNs are relegated to an optional, low-priority screen. Priority compute is allocated to foundation model adaptation (PanDerm LP-FT), multi-resolution input streams (384 px), and registered self-distillation (DINOv2-Reg).

---

## 6. Audit of V7 Specification (Revision 8) & Current Implementation State

### 6.1 State of `docs/V7_MASTER_RESEARCH_PLAN.md`
- Confirmed locally at `docs/V7_MASTER_RESEARCH_PLAN.md` (2,824 lines, 273,489 bytes).
- Title: *Revision 8 — Final 4-Branch Metadata, Morphology & Patient-Relative Specification*.
- Epistemic Status: **Formulated / Confirmatory Protocol Frozen**.

### 6.2 Implementation Inventory (Code, Data, Results)
- **Implemented Code:** [`scripts/audit_manifest_v7.py`](file:///C:/Users/RAJ/Downloads/Capstone%20review%201/scripts/audit_manifest_v7.py) (metadata missingness auditor, verified clean).
- **Unimplemented Code:** Zero V7 training pipelines, multi-stream loaders, or fusion modules exist in `ml/` or `research/`.
- **Data Manifests:** `ml/data/v7/master_v7.csv` has not yet been built. `V7_DATA_PLAN.md` remains unadopted.
- **Empirical Execution:** **0.0% executed.** V7 remains strictly a design specification.

---

## 7. Master Discrepancy & Lineage Reconciliation Matrix

The complete quantitative evidence from V1 to V6 has been cross-referenced and recorded in:
- [`agent01_lineage_evidence.csv`](file:///c:/Users/RAJ/Downloads/Capstone%20review%201/docs/v7_final_revision/agent01_lineage_evidence.csv) (28 milestones across 6 versions)
- [`agent01_state_discrepancies.csv`](file:///c:/Users/RAJ/Downloads/Capstone%20review%201/docs/v7_final_revision/agent01_state_discrepancies.csv) (14 reconciled anomalies, bugs, and historical retractions)

This concludes the authoritative forensic audit.

