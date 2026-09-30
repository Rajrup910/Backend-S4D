# V5 Final Forensic Correction Audit Report
**Programme:** Backend-S4D (V5 Research Freeze)  
**Audit Timestamp:** 2026-09-18T21:30:00+05:30  
**Status:** All Forensic Issues Resolved & Synchronized Across Master Plan, Runsheet, and Specifications  

---

## 1. Executive Summary

This forensic audit represents the final correction loop reconciling the V5 Master Research Plan (`docs/V5_MASTER_RESEARCH_PLAN_REVISED.md`), the V5 Execution Runsheet (`docs/V5_FINAL_RUNSHEET.md`), supporting specifications, and repository implementations.

- **Total Issues Found:** 10
- **Total Issues Corrected:** 10
- **Unresolved Issues Remaining:** 0
- **Did Scientific Intent Change?:** **No.** All corrections tighten statistical validity, eliminate mathematical and logical ambiguities, harmonize sample size terminology with actual repository manifests, and ensure direct, unblocked execution on the NVIDIA GeForce RTX 5050 Laptop GPU.

---

## 2. Itemized Forensic Findings & Corrections

### Finding 1: S01 Compute vs. Full 5-Fold OOF Contradiction
- **Severity:** Critical
- **Exact File / Section:** `docs/V5_MASTER_RESEARCH_PLAN_REVISED.md` §6, §17; `docs/V5_FINAL_RUNSHEET.md` Stage 1; `results/v5/v5_plan_preservation_audit.json`.
- **Pre-Correction State:** The plan described S01 as a paired 384px vs 224px "development/OOF comparison" while allocating only ~4.6 h of compute. A full 5-fold OOF cross-validation across 3 seeds would require 15 training runs (~23–27 h), completely violating the staged-budget philosophy.
- **Correction Made:** Explicitly specified S01 as a **multi-seed development screening comparison on Fold 0 of the frozen S71 partition** ($N_{\text{train}}=12,235$, $N_{\text{val}}=3,059$ held-out images). Evaluates 3 independent seeds (`42, 43, 44`) for a total compute of $3 \times 92.6\text{ min} \approx 4.6\text{ h}$, paired against the banked S72 Fold 0 224px control ($0.6508$). Clarified that full 5-fold cross-fitting ($N=15,294$ pooled OOF) is reserved for Tier 3 finalist confirmation (Stage 5 backbones and Stage 6 ensembling).
- **Reason for Correction:** Preserves the compute-budget boundary and staged anti-explosion philosophy while maintaining rigorous multi-seed comparative benchmarking.
- **Scientific Intent Changed:** No.
- **Verification Status:** VERIFIED.

---

### Finding 2: MCID Inconsistency (+0.010 vs +0.015)
- **Severity:** Major
- **Exact File / Section:** `docs/V5_MASTER_RESEARCH_PLAN_REVISED.md` §6, §18; `docs/V5_FINAL_RUNSHEET.md` Stage 1; `results/v5/v5_plan_preservation_audit.json`.
- **Pre-Correction State:** Conflicting textual references cited MCID as $+0.010$ in some audit notes and $+0.015$ in the main plan and runsheet.
- **Correction Made:** Established exactly ONE authoritative S01 resolution improvement MCID: **$\text{MCID} = +0.015$ Macro-F1**. Clarified that Gate B's degradation tolerance is $\Delta \text{Macro-F1} \ge -0.010$ (maximum acceptable loss relative to the 384px control for subsequent representation interventions). Propagated $+0.015$ universally.
- **Reason for Correction:** In S53r, the historical resolution gain was $+0.0297$. A meaningful replication of this effect must achieve at least $\approx 50\%$ of the historical gain ($\ge +0.015$). Small gains ($< +0.010$) fall within seed noise.
- **Scientific Intent Changed:** No.
- **Verification Status:** VERIFIED.

---

### Finding 3: S01 Two-Hurdle Gate Logic ("OR" Formulation Flaw)
- **Severity:** Critical
- **Exact File / Section:** `docs/V5_MASTER_RESEARCH_PLAN_REVISED.md` §6, §17; `docs/V5_FINAL_RUNSHEET.md` Stage 1.
- **Pre-Correction State:** The previous text required: *"95% bootstrap confidence interval of paired $\Delta \text{Macro-F1}$ excluding zero OR exceeding the declared MCID (+0.015)"*. This allowed a tiny gain of $+0.002$ whose confidence interval excluded zero to pass, completely bypassing the MCID requirement.
- **Correction Made:** Replaced the ambiguous "OR" with a rigorous **Two-Hurdle Acceptance Rule**:
  1. **Statistical Superiority:** 95% bootstrap confidence interval lower bound of paired $\Delta \text{Macro-F1}$ must strictly exceed $0.000$ (excludes zero);
  2. **Meaningful Magnitude:** Sample mean paired improvement must meet or exceed MCID ($\overline{\Delta \text{Macro-F1}} \ge +0.015$);
  3. **Direction & Consistency:** At least 2 of 3 seeds strictly positive ($\Delta > 0$), 3-seed mean $\ge +0.015$, and min seed $\ge -0.010$;
  4. **Per-Class Safety:** No material degradation in critical classes (MEL F1 $\Delta \ge -0.020$, BCC recall $\Delta \ge -0.020$).
- **Reason for Correction:** Standard biostatistical practice requires proving both non-zero effect and clinically meaningful effect magnitude.
- **Scientific Intent Changed:** No.
- **Verification Status:** VERIFIED.

---

### Finding 4: Gate D Arbitrary Seed Variance Cap ($\sigma \le 0.050$)
- **Severity:** Major
- **Exact File / Section:** `docs/V5_MASTER_RESEARCH_PLAN_REVISED.md` §18 (Gate D); `docs/V5_FINAL_RUNSHEET.md` §2.
- **Pre-Correction State:** Gate D contained an ungrounded hard threshold: *"inter-seed standard deviation $\sigma \le 0.050$"*.
- **Correction Made:** Replaced the arbitrary cap with empirical dispersion reporting and an explicit stability warning rule:
  - For Macro-F1, an instability warning is flagged if $s_{\text{seed}} > 0.015$ (which indicates that seed dispersion reaches the full magnitude of the resolution MCID, signalling erratic optimization);
  - For under-40 $pAUC$, dispersion is reported descriptively given small sample variance;
  - Re-asserted that multi-seed confirmation focuses on directional consistency (mean $> 0$, $\ge 2/3$ positive, min seed $\ge -0.010$).
- **Reason for Correction:** Historical S44 data demonstrated same-data Macro-F1 seed spread was $\approx 0.0027$; an arbitrary $0.050$ threshold was mathematically ungrounded. Linking stability directly to the MCID ($0.015$) provides a principled benchmark.
- **Scientific Intent Changed:** No.
- **Verification Status:** VERIFIED.

---

### Finding 5: OOF Sample-Size Terminology Confusion ($N=12,235$ vs $N=15,294$)
- **Severity:** Major
- **Exact File / Section:** `docs/V5_MASTER_RESEARCH_PLAN_REVISED.md` §18 (Gate B); `results/v5/v5_oof_matrix_spec.md`.
- **Pre-Correction State:** Gate B incorrectly wrote *"complete 5-fold OOF ($N=12,235$)"*.
- **Correction Made:** Corrected sample size terminology against `s71_plan.json` and `ml/data/manifest_v4.csv`:
  - Per-fold training subset: $N_{\text{train}} = 12,235$ (or 12,236 for Fold 4);
  - Single-fold development validation split: $N_{\text{val}} = 3,059$ (or 3,058 for Fold 4);
  - Complete pooled 5-fold OOF population: **$N_{\text{pooled\_OOF}} = 15,294$ images** ($8,748$ unique lesions across $8,734$ groups).
- **Reason for Correction:** Ground truth in `manifest_v4.csv` confirms pooled training split has 15,294 rows. 12,235 was the per-fold training size, not the pooled OOF size.
- **Scientific Intent Changed:** No.
- **Verification Status:** VERIFIED.

---

### Finding 6: S20 External Cohort Statistical Power Under-Specification
- **Severity:** Critical
- **Exact File / Section:** `docs/V5_MASTER_RESEARCH_PLAN_REVISED.md` §15, §18; `docs/V5_FINAL_RUNSHEET.md` Stage 8; `results/v5/v5_external_analysis_spec.md`.
- **Pre-Correction State:** Documents claimed $N \ge 35$ independent under-40 escalating lesions provided $84.2\%$ power for a one-sided exact binomial test ($H_0 \le 0.50$ vs $H_1 \ge 0.70$, $\alpha = 0.05$). Additionally, endpoints were termed "Co-Primary" despite being tested under a sequential hierarchy.
- **Correction Made:**
  - Evaluated exact binomial distributions from first principles: At $N=35$, exact power at $p=0.70$ is **77.3%** (the previous $84.2\%$ was derived from an uncorrected continuous normal approximation).
  - Established exact sample size floor: **$N_{\text{esc,<40}} \ge 40$ independent lesions** provides **80.7% exact power** (rejection threshold $k \ge 26/40$, exact $\alpha = 0.0403$).
  - Established target sample size: **$N_{\text{esc,<40}} \ge 50$ independent lesions** provides **85.9% exact power** (rejection threshold $k \ge 32/50$, exact $\alpha = 0.0325$, and $97.1\%$ at $p=0.75$).
  - Harmonized endpoint hierarchy: Formally designated as **Fixed-Sequence Hierarchical Testing**: Primary Confirmatory Endpoint (7-class Macro-F1 $\ge 0.7500$ at $\alpha = 0.05$) $\to$ Key Secondary Confirmatory Subgroup Endpoint (Under-40 escalating lesion sensitivity $\ge 0.700$ at $\ge 80\%$ non-escalating specificity, tested only if Primary Endpoint is significant).
- **Reason for Correction:** Eliminates statistical power inflation and aligns endpoint nomenclature with ICH E9 / FDA regulatory guidelines for hierarchical testing.
- **Scientific Intent Changed:** No.
- **Verification Status:** VERIFIED.

---

### Finding 7: Path & Lesion Metadata Discrepancy in `v5_oof_matrix_spec.md`
- **Severity:** Minor
- **Exact File / Section:** `results/v5/v5_oof_matrix_spec.md` §1, §3.
- **Pre-Correction State:** Referenced `results/v4/kfold/assignments.csv` (actual filename is `fold_assignments.csv`), cited `10,288 unique lesion clusters` (actual count is $8,748$ unique lesions), and cited reserved cohort as $N=1,040$ (actual reserved size in `manifest_v4.csv` is $N=4,733$).
- **Correction Made:** Updated file path to `results/v4/kfold/fold_assignments.csv`, updated lesion count to $8,748$ lesions ($8,734$ groups), and corrected split counts: HAM test ($N=1,502$), reserved cohort ($N=4,733$), validation ($N=2,270$).
- **Reason for Correction:** Exact alignment with repository filesystem and `manifest_v4.csv`.
- **Scientific Intent Changed:** No.
- **Verification Status:** VERIFIED.

---

### Finding 8: Global Seed Policy Audit & Discrepancies
- **Severity:** Minor
- **Exact File / Section:** Global documentation & relay notes.
- **Pre-Correction State:** Scratch notes occasionally referenced seeds `42, 1337, 2024` alongside the canonical `42, 43, 44`.
- **Correction Made:** Verified that all normative plan documents and runsheets strictly adhere to the historical canonical seed policy:
  - Tier 1 (Screening): Seed `42`;
  - Tier 2 (Local Ablation): Seed `42`;
  - Tier 3 (Multi-Seed / Finalists): Seeds `42, 43, 44`.
- **Reason for Correction:** Prevents researcher confusion and maintains parity with V4 recipe ladder benchmarks.
- **Scientific Intent Changed:** No.
- **Verification Status:** VERIFIED.

---

### Finding 9: SwinV2-Tiny-384 Pretrained Identifier Verification
- **Severity:** Informational
- **Exact File / Section:** `docs/V5_MASTER_RESEARCH_PLAN_REVISED.md` §11; `docs/V5_FINAL_RUNSHEET.md` Stage 5.
- **Pre-Correction State:** Master Plan previously contained slash options ("Tiny/Base").
- **Correction Made:** Frozen strictly to `swinv2_tiny_window12to16_192to384.ms_in22k_ft_in1k` (28.35M parameters, $16 \times 16$ window, 6.4 GB peak VRAM).
- **Reason for Correction:** Eliminates pre-registration ambiguity and guarantees safe execution within 8.55 GB VRAM on the RTX 5050 Laptop GPU.
- **Scientific Intent Changed:** No.
- **Verification Status:** VERIFIED.

---

### Finding 10: User Downloads Directory Mirroring
- **Severity:** Minor
- **Exact File / Section:** `C:\Users\RAJ\Downloads\V5_MASTER_RESEARCH_PLAN_REVISED.md` and `C:\Users\RAJ\Downloads\V5_FINAL_RUNSHEET.md`.
- **Pre-Correction State:** Local workspace `docs/` edits risked desynchronization from the external user downloads folder.
- **Correction Made:** Copied updated files to `C:\Users\RAJ\Downloads\` and verified cryptographic SHA-256 byte-for-byte identity.
- **Reason for Correction:** Adheres to Paired Programming protocol.
- **Scientific Intent Changed:** No.
- **Verification Status:** VERIFIED.

---

### Finding 11: S01 Scope Boundary & Reporting Invariant
- **Severity:** Informational
- **Exact File / Section:** `docs/V5_MASTER_RESEARCH_PLAN_REVISED.md` §6; `docs/V5_FINAL_RUNSHEET.md` §2.
- **Pre-Correction State:** Implicit understanding that S01 was a screening gate could lead future analysis/reporting scripts to mislabel the 3-seed Fold 0 result as a full cross-validated performance estimate.
- **Correction Made:** Explicitly added the methodological scope invariant: *"S01 answers whether 384px is reproducibly promising on the frozen Fold-0 screening split ($N_{\text{val}}=3,059$). The three-seed Fold-0 result is an initial reproducibility screen, NOT a full cross-validated estimate. Final performance claims and ensemble selection strictly require the complete $N=15,294$ cross-fitted OOF matrix (Stage 5/6)."*
- **Reason for Correction:** Prevents scope drift and guarantees honest, auditable scientific reporting in all generated artifacts and manuscripts.
- **Scientific Intent Changed:** No.
- **Verification Status:** VERIFIED.

---

## 3. Summary of Document Synchronization

| Artifact Path | SHA-256 Checksum | Synchronization Status |
|---|---|---|
| `docs/V5_MASTER_RESEARCH_PLAN_REVISED.md` | `F862390E916FFFAE273DDA93A7FDF317D27036E411929405AB80CF91A00655E7` | Verified |
| `C:\Users\RAJ\Downloads\V5_MASTER_RESEARCH_PLAN_REVISED.md` | `F862390E916FFFAE273DDA93A7FDF317D27036E411929405AB80CF91A00655E7` | Verified (Identical) |
| `docs/V5_FINAL_RUNSHEET.md` | `ED7D3AB38B7132454141940CBB814B29A2CBDB10F201039154CF4B72FDA3BE3A` | Verified |
| `C:\Users\RAJ\Downloads\V5_FINAL_RUNSHEET.md` | `ED7D3AB38B7132454141940CBB814B29A2CBDB10F201039154CF4B72FDA3BE3A` | Verified (Identical) |
| `results/v5/v5_oof_matrix_spec.md` | `C403330D66518271989EC0A0BC0E4BDEC36ADB56F514D018EDD405F40CC15380` | Verified |
| `results/v5/v5_external_analysis_spec.md` | `FE978BC73564DB7C6ACB0AB6E13F647E5A501DC30099F7D1E85CB503D6F99621` | Verified |
| `results/v5/v5_no_repeat_registry.json` | `ED188C8672D998056FFFF1F74770536283F095292884A7140C62D97EC64D71C9` | Verified |
| `results/v5/v5_plan_preservation_audit.json` | `A7D482A88DC88EB6D79A0676068110E37B9B1C489143399581EBCA083792BF95` | Verified |
| `results/v5/V5_FINAL_FREEZE_CHECKLIST.json` | `D368240249CB5B658E7240C951505901FB8731D8B9758A978CCA70C176E4A722` | Verified |
| `results/v5/v5_plan_freeze.json` | Verified & Frozen | Verified |

---

## 4. Final Verdict

All forensic consistency checks have passed. There are zero unresolved contradictions, zero ambiguous gates, and zero ungrounded statistical assumptions remaining in the V5 research programme.

