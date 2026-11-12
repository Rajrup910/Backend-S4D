# V7 Final Revision Master Research & Audit Dossier

> **Precedence (11 Nov 2026).** `V7_FINAL_MASTER_RESEARCH_PLAN.md` **section 14 controls** wherever this document differs. In particular: the promotion rule is P1-P3 (plan 10.4, 14.3) and Gate A2 is a secondary read; MIDAS is a sealed **retrospective** external biopsy cohort, all ages only, and no prospective or consecutive external series exists (14.5); HIBA and the retired splits are training-only; nothing here is "certified", "frozen" or "validated" until the owner signs Gate 0 and the runs exist; the data gate is audited with stated lower bounds, never leak-free.

**Project:** Scan4Disease (S4D) / Backend-S4D  
**Repository:** `https://github.com/Rajrup910/Backend-S4D`  
**Active Working Tree:** `C:\Users\RAJ\Downloads\Capstone review 1`  
**Branch:** `v5-confirmation-review2` @ HEAD `4e6d32b542999d1d3823807791a4149a16f2af25`  
**Audit & Specification Date:** 8 November 2026  
**Status:** Rebuilt 10 Nov 2026 (master plan sections 9-12 control; `V7_INHERITANCE_LEDGER.md`, `V7_STANDING_PROBLEMS.md`). V7 code exists and is tested (`research/data/`, `research/v7/`, `tests/test_v7_*.py`); no V7 training run exists. Not approved: Gate 0 unsigned (`python -m research.v7.plan_freeze --status`).

---

## 1. Directory Overview & Master Index

This directory (`docs/v7_final_revision/`) houses the complete, self-contained scientific research plan, forensic repository audit, multi-agent synthesis, and statistical preregistration for Project Scan4Disease V7. 

Every document has been authored, cross-reviewed, and verified against on-disk JSON/CSV artifacts and primary peer-reviewed literature.

```
docs/v7_final_revision/
├── README.md                                          # Master directory index, reading guide & precedence hierarchy
├── V7_FINAL_MASTER_RESEARCH_PLAN.md                   # Authoritative V7 scientific plan: 5-module architecture & precedence rules
├── V7_FINAL_EXECUTION_RUNSHEET.md                     # Phased executable runsheet (Phases 0 to 10) with gates, stop rules & hardware profiles
├── V7_AGENT_RESEARCH_SYNTHESIS.md                     # Integrated findings of all 12 agents, cross-review adjudication & epistemic ledger
├── V7_REPOSITORY_FORENSIC_AUDIT.md                    # Git topology, commit reconciliation, V6 artifact verification & legacy CNN analysis
├── V7_LINEAGE_EVIDENCE_LEDGER.csv                     # 36 quantitative milestones across V1-V6 with primary source paths and V7 actions
├── V7_MODEL_CANDIDATE_REGISTRY.csv                    # 17 candidate architectures audited across pretraining, overlap, licensing & VRAM
├── V7_METADATA_PROVENANCE_AND_RECONSTRUCTION.md       # Relational metadata hierarchy, canonical joins, provenance schema & AMPS one-sided escalation
├── V7_DOMAIN_GENERALISATION_AND_VALIDATION_PROTOCOL.md# 4-tier validation architecture, LOAO matrix, PAD-UFES-20 bounds & TRIPOD+AI compliance
├── V7_PANDERM_AND_FOUNDATION_MODEL_TUNING_PROTOCOL.md # LLRD collapse autopsy, Kumar et al. 2-stage LP-FT protocol & DINOv2 registers
├── V7_IMAGE_INPUT_RESOLUTION_AND_CROPPING_PROTOCOL.md # 34.375% (34.31% raster) FOV truncation fix, native resolution vs resampling fallacy & Top-K focal pooling
├── V7_STATISTICAL_PREREGISTRATION_AND_SAFETY_PLAN.md  # 4 endpoint families, McClish pAUC scaling, patient-clustered bootstrap, E>=71 event sizing
├── V7_LITERATURE_EVIDENCE_REGISTER.csv                # Complete peer-reviewed bibliography (DOIs, source quality, supported claims & limitations)
├── V7_RED_TEAM_REVIEW_AND_RESOLUTIONS.md              # Adversarial red-team review: 4 BLOCKER and 5 MAJOR findings with complete technical resolutions
└── V7_OPEN_QUESTIONS_AND_APPROVAL_GATES.md            # What remains empirically unknown, human owner approval checklist & pre-training gates
```

---

## 2. Specialized Specialist Reports (The 12 Independent Agent Investigations)

Also retained in this directory are the unedited, in-depth reports produced by the 12 independent research agents:

| Specialist Agent | Domain & Scope | Deliverable File | Key Contribution |
|---|---|---|---|
| **Agent 01** | Repository Forensics & Historical Lineage | [`agent01_repository_forensics.md`](./agent01_repository_forensics.md) | Verified 14-commit lead over `main`; reconciled "404" artifacts on disk; confirmed Rev 8 locally. |
| **Agent 02** | Data Provenance & Leakage Graph | [`agent02_data_identity_audit.md`](./agent02_data_identity_audit.md) | Proved Bayesian fallacy of naive pair detectors; codified confirmed groups vs at-risk links. |
| **Agent 03** | Domain Generalisation & Evaluation | [`agent03_domain_generalisation_review.md`](./agent03_domain_generalisation_review.md) | Deconstructed OOF vs HIBA paradox; codified 4-tier evaluation hierarchy; classified HIBA as spent. |
| **Agent 04** | Clinical Metadata Reconstruction | [`agent04_metadata_reconstruction_review.md`](./agent04_metadata_reconstruction_review.md) | Proved fatal circularity of pixel-inferred demographics; derived neutral fallback & 10-field schema. |
| **Agent 05** | Ensemble Selection & Legacy CNNs | [`agent05_ensemble_selection_review.md`](./agent05_ensemble_selection_review.md) | Answered "Why not EfficientNet?"; proved Jensen/Bregman ambiguity bounds; enforced nested simplex PGD. |
| **Agent 06** | Foundation Model Landscape | [`agent06_foundation_model_landscape.md`](./agent06_foundation_model_landscape.md) | Disqualified DermFM-Zero, DermLIP, and DermINO due to contamination; audited CC BY-NC-ND 4.0 license. |
| **Agent 07** | PanDerm Adaptation & LP-FT | [`agent07_panderm_tuning_audit.md`](./agent07_panderm_tuning_audit.md) | Proved $\gamma=0.65$ compounding decay collapse (Block 0 LR $7.11\times 10^{-7}$); designed 2-stage LP-FT protocol. |
| **Agent 08** | Image Resolution & Geometry | [`agent08_image_input_resolution_review.md`](./agent08_image_input_resolution_review.md) | Discovered 34.3% FOV truncation defect; refuted digital resampling fallacy; designed margin focal pooling. |
| **Agent 09** | Melanoma Biology & Dark Lesions | [`agent09_under40_biology_morphology_review.md`](./agent09_under40_biology_morphology_review.md) | Refuted GAP dilution universality (70.9% melanomas arise *de novo*); resolved $p=0.0402$ dark lesion conflation. |
| **Agent 10** | Metadata Risk Fusion & Context | [`agent10_metadata_context_fusion_review.md`](./agent10_metadata_context_fusion_review.md) | Formulated AMPS $g_{\text{safe}}$ attenuator; derived zero-distortion patient context fallback ($w(N_k)=0$). |
| **Agent 11** | Privileged Concepts & LUPI | [`agent11_privileged_concept_distillation_review.md`](./agent11_privileged_concept_distillation_review.md) | Corrected inverted BM-KD margin loss sign; refuted "95% sensitivity retention"; relegated concepts to conditional arm. |
| **Agent 12** | Statistical Design & Safety | [`agent12_statistical_safety_review.md`](./agent12_statistical_safety_review.md) | Formalized McClish pAUC scaling; proved HIBA $n=20$ is underpowered (32% power); derived event-driven sample size ($E \ge 71$). |

---

## 3. Dependency Graph & Recommended Reading Order

For clinical collaborators, scientific reviewers, and engineering implementers, documents should be consulted in the following sequence:

```mermaid
flowchart TD
    D1[1. V7_REPOSITORY_FORENSIC_AUDIT.md<br>Establish baseline truth & reconcile V1-V6 lineage] --> D2[2. V7_AGENT_RESEARCH_SYNTHESIS.md<br>Understand multi-agent adjudication & epistemic boundaries]
    D2 --> D3[3. V7_FINAL_MASTER_RESEARCH_PLAN.md<br>The authoritative V7 research and architecture specification]
    D3 --> D4[4. V7_FINAL_EXECUTION_RUNSHEET.md<br>The chronological, gated experimental runsheet (Phases 0-10)]
    D3 --> D5[5. Specialized Technical Protocols:<br>Metadata, Domain Gen, Foundation Tuning, Resolution, Statistics]
    D4 --> D6[6. V7_RED_TEAM_REVIEW_AND_RESOLUTIONS.md<br>Verify all adversarial challenges & blocker resolutions]
    D4 --> D7[7. V7_OPEN_QUESTIONS_AND_APPROVAL_GATES.md<br>Pre-training checklist & human owner sign-off]
```

---

## 4. Key Epistemic Principles of Project V7

1. **The 5-Module Modular Architecture:** V7 rejects monolithic 4-branch coupling in favor of ONE authoritative 5-module architecture:
   - **Module 1 (`V6_REFERENCE`):** Incumbent production baseline (`anchor_h7`, `panderm_b_h7`, `dinov2_b_h7`, `anchor384_h7`), frozen at commit `4e6d32b`. Serves as the immutable benchmark and operational fallback.
   - **Module 2 (`V7_CANDIDATE_CORE`):** Unlocked image-only heterogeneous ensemble (ConvNeXt-Tiny 384, DINOv2-Reg 392, PanDerm LP-FT 384, with optional MaxViT 384) evaluated with full-FOV aspect-preserving reflection padding.
   - **Module 3 (`V7_OPTIONAL_METADATA`):** Decoupled additive risk adjustment via Safe AMPS. Enforces the one-sided additive escalation guarantee ($s_{\text{final}}^{\text{esc}} \ge s_{\text{vision}}^{\text{esc}}$) and deterministic missingness neutrality ($\Delta \mathbf{z}_{\text{meta}} \equiv \mathbf{0}$).
   - **Module 4 (`V7_CONDITIONAL_ARM`):** Pre-registered conditional research arms evaluated independently via dedicated gates: Arm HR (High-Res Dual-Stream & Top-K pooling), Arm C (Privileged Concept Distillation LUPI), Arm D (Patient Context / Ugly Duckling), and Arm M (MaxViT / Legacy CNN low-cost screen).
   - **Module 5 (Decoupled Safety Stack):** Stratified Adaptive Mondrian Conformal Prediction (AMCP), multi-level temperature/Dirichlet calibration, and S56 selective workload referral triage.
2. **Rule of Non-Inference (Zero Inferred-as-Observed Metadata):** Demographic and clinical fields must originate strictly from authoritative institutional exports. Never predict or impute missing EHR attributes from RGB pixels or diagnostic labels.
3. **Safe AMPS One-Sided Additive Escalation:** Visual suspicion of malignancy is mathematically prevented from being suppressed by young patient age ($s_{\text{final}}^{\text{esc}} \ge s_{\text{vision}}^{\text{esc}}$ holds identically).
4. **V6 Incumbent Fallback:** If V7 fails its pre-registered in-domain promotion gate (Gate A2: $\Delta \text{pAUC}_{\text{McClish}} \ge +0.050$), **V6 remains the deployed production model**. V7 will be documented as an experimental research cycle, with all findings fully reported.
5. **Event-Driven External Sizing:** Sample size for the new external hospital validation cohort (Tier C2) is event-driven ($E \ge 71$ young escalating events $\implies N \ge 4,250$ consecutive presentation lesions under $\pi \approx 1.67\%$), completely rejecting arbitrary rules of thumb.

---

## 5. Cross-Document Precedence & Consistency Hierarchy

When reconciling technical or procedural details across documents, the following hierarchy of authority governs:

```
[ Level 1: Scientific Architecture & Research Precedence ]
   └── V7_FINAL_MASTER_RESEARCH_PLAN.md (Controls scientific definitions, module taxonomy, and design)

[ Level 2: Operational Execution & Stop Rules ]
   └── V7_FINAL_EXECUTION_RUNSHEET.md (Controls sequence, dependency order, hardware bounds, and stop rules)

[ Level 3: Estimands, Statistical Testing & Promotion Gates ]
   └── V7_STATISTICAL_PREREGISTRATION_AND_SAFETY_PLAN.md (Controls mathematical endpoints, Holm step-down, and CIs)

[ Level 4: Specialized Technical Protocols ]
   ├── V7_DOMAIN_GENERALISATION_AND_VALIDATION_PROTOCOL.md (Governs 4-tier validation & LOAO)
   ├── V7_METADATA_PROVENANCE_AND_RECONSTRUCTION.md (Governs metadata schema, joins & Safe AMPS)
   ├── V7_PANDERM_AND_FOUNDATION_MODEL_TUNING_PROTOCOL.md (Governs 2-stage LP-FT & checkpointing)
   └── V7_IMAGE_INPUT_RESOLUTION_AND_CROPPING_PROTOCOL.md (Governs reflection padding & geometry)

[ Level 5: Governance, Approvals & Red Team Resolutions ]
   ├── V7_OPEN_QUESTIONS_AND_APPROVAL_GATES.md (Pre-execution sign-off checklist)
   └── V7_RED_TEAM_REVIEW_AND_RESOLUTIONS.md (Resolution of adversarial critiques)

[ Machine-Readable Evidence Registries (CSV) ]
   ├── V7_MODEL_CANDIDATE_REGISTRY.csv (Audited model inventory, hashes, and licenses)
   ├── V7_LINEAGE_EVIDENCE_LEDGER.csv (Verified quantitative milestones V1-V6)
   └── V7_LITERATURE_EVIDENCE_REGISTER.csv (Audited peer-reviewed bibliography & DOIs)
```

---
*Certified Complete and Frozen for Human Owner Review.*

