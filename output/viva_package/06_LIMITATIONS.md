# Comprehensive Limitations & Defense Strategy

**Source Basis:** Manuscript Section VI (Limitations), CLAIM 2024 Checklist, TRIPOD+AI Cross-Walk, and `CHANGELOG.md`.

This document ranks every methodological, empirical, and clinical limitation of the study, explains its root cause, and provides scientifically grounded defense scripts for oral examination.

---

## 1. Ranked Hierarchy of Study Limitations

### LEVEL 1: CRITICAL LIMITATIONS (Must Concede Immediately)

#### 1. Underpowered Darker Skin-Tone Representation (Fitzpatrick V & VI)
- **The Limitation:** The primary development cohort (HAM10000) ships zero Fitzpatrick skin phototype annotations. Image-derived Individual Typology Angle (ITA) is mathematically invalid on dermoscopy (vignetting, peripheral shadow, and inflammatory erythema corrupt the colorimetry). The external cohort (PAD-UFES-20) contains only 8 Type V images and 1 Type VI image ($n=9$ total), both suppressed below the pre-registered reporting gate ($N \ge 30$).
- **Impact:** The study has zero empirical validity for individuals with dark brown or black skin (Fitzpatrick V–VI)—the exact demographic where literature reports the highest rates of late-stage melanoma diagnosis and algorithm disparity.
- **Viva Defense Script:**
  > *"Examiner: Isn't the lack of skin-tone analysis a fatal flaw in a medical AI paper claiming to address fairness?"*
  > **Candidate Defense:** *"Yes, it is a major limitation, and we explicitly state it as such in Section VI. HAM10000 does not record Fitzpatrick types, and we verified that calculating ITA from dermoscopy images is scientifically invalid due to optical vignetting. On our external cohort, Types V and VI represented only nine images, so we strictly suppressed them under our pre-registered power gates rather than publishing unreliable numbers. We do not claim universal fairness; our fairness analysis is strictly bounded to age and sex in fair-to-medium skin. Evaluating this system on dark skin requires dedicated corpora like Fitzpatrick17k, which is our immediate next step."*

#### 2. The Under-40 Blind Spot is Mitigated, Not Closed
- **The Limitation:** Despite applying the out-of-fold fitted $\lambda$ escalation rule, test escalation sensitivity in patients under 40 rises only from 0.143 to 0.238 [95% CI: 0.082, 0.472]. More than 75% of malignancies in young adults remain misclassified as benign moles.
- **Impact:** The system cannot be safely deployed in clinical practice for young adult populations.
- **Viva Defense Script:**
  > *"Examiner: If your age-conditional rule only brings under-40 sensitivity to 24%, how can you claim your mitigation worked?"*
  > **Candidate Defense:** *"We explicitly do not claim the problem is solved. In Section IV-D and the Abstract, we state that the blind spot is mitigated, not closed. Because within-band escalation-mass AUC dropped to 0.810 on test, part of this failure is genuine information loss that no post-hoc threshold adjustment can recover. What our contribution demonstrates is how to measure this failure, price its trade-off in Number Needed to Biopsy, and prove that standard abstention and marginal conformal prediction completely fail to catch it. We explicitly warn against deploying this system in under-40 patients."*

#### 3. Forfeited Finite-Sample Conformal Guarantee Under Out-of-Fold Calibration
- **The Limitation:** Split conformal prediction guarantees finite-sample coverage $1 - \alpha$ under the assumption that calibration scores and test scores are exchangeable draws from the same distribution under a single, fixed score function. In our OOF protocol, calibration non-conformity scores are emitted by 5 distinct fold models trained on 80% data, while test scores are emitted by the full-train ensemble.
- **Impact:** The formal mathematical theorem does not transfer to the OOF-calibrated test scores; coverage is an empirical audit, not a certified mathematical bound.
- **Viva Defense Script:**
  > *"Examiner: Your paper title promises 'conformal guarantees', but doesn't OOF calibration break exchangeability?"*
  > **Candidate Defense:** *"You are entirely correct. We state this explicitly in Section III-F: OOF calibration buys sample size—quintupling rare-class support and eliminating degenerate cells—but forfeits the formal theorem because fold models and the full-train ensemble are not identical score functions. That is why we retain the validation-fitted configuration in Table III, which preserves the exact finite-sample theorem, while reporting the OOF results in Table IV strictly as an empirical audit. To restore a $(1 - 2\alpha)$ formal guarantee would require CV+ or cross-conformal prediction, which necessitates scoring test with all 30 fold models—a second test read that our pre-registered single-pass protocol prohibited."*

---

### LEVEL 2: HIGH LIMITATIONS (Substantial Methodological Trade-offs)

#### 4. The Dose-Response Skepticism & Non-Decisive Multi-Centre Transfer
- **The Limitation:** In Workstream E1, our pre-registered mechanistic hypothesis (Claim B)—that under-40 sensitivity would inversely track prior skew—failed backwards across HAM10000, MSKCC, and BCN-20000.
- **Impact:** We cannot prove causality. We cannot claim that demographic prior skew in the training set is the sole or primary cause of the young-patient failure.
- **Viva Defense Script:**
  > *"Examiner: Your dose-response experiment failed completely. Doesn't that destroy your explanation for the under-40 failure?"*
  > **Candidate Defense:** *"It falsified the hypothesis that prior skew is the sole explanatory variable across centres, and we reported that failure without post-hoc rationalization. In Section IV-E, we explain that the dose variable is heavily entangled with cohort transfer quality: BCN had the lowest prior skew, but our frozen model transferred to it worst (Macro-F1 0.402 vs 0.784 on HAM). What survived across all three centres was the operating point: the frozen $\lambda$ threshold improved sensitivity in all cohorts. Our conclusion is nuanced: the operating point transports, but the causal explanation does not."*

#### 5. Orthogonality Failure in the Target Subgroup
- **The Limitation:** Margin-based abstention and the $\lambda$ escalation rule were intended as complementary safety nets. In the under-40 band, of 18 missed malignancies, both mechanisms caught the exact same 2 cases (Jaccard index 1.00), leaving 16 misses completely unrescued.
- **Impact:** The safety nets are entirely redundant precisely where the clinical failure is most severe.
- **Viva Defense Script:**
  > *"Examiner: If your two safety mechanisms have a Jaccard overlap of 1.0 in young patients, isn't your system architecture redundant?"*
  > **Candidate Defense:** *"In that specific band, yes, and Table VII documents that exact failure. In patients aged 40–59, the mechanisms are highly complementary with a Jaccard overlap of only 0.18, where the rule catches 11 misses and abstention catches 2. But under 40, the model is confidently wrong: 16 of the 18 missed melanomas have wide margins and low escalation mass. Reporting this Jaccard overlap of 1.0 is one of our primary negative results: it proves that stacking post-hoc safety nets cannot compensate for missing feature representations."*

#### 6. Retrospective Curated Cohorts & Artificial Screening Prevalence
- **The Limitation:** HAM10000 is a curated tertiary-care dermatopathology dataset with an escalating lesion prevalence of 19.3%. Primary care screening typically exhibits an escalating prevalence between 1% and 5%.
- **Impact:** Raw positive predictive value and unadjusted Number Needed to Biopsy (NNB = 3.0) cannot be compared to clinical practice.
- **Viva Defense Script:**
  > *"Examiner: Dermatologists have an NNB between 8 and 15. Your paper claims an NNB of 3.0. Isn't that unrealistic?"*
  > **Candidate Defense:** *"An unadjusted NNB of 3.0 is entirely a byproduct of HAM10000's enriched 19.3% prevalence. That is why Section III-B and Equation 3 explicitly define prevalence-reweighted $\text{NNB}_\pi$. When reweighted to a realistic primary-care prevalence of $\pi = 0.03$, our base NNB rises to 3.0 and the age-conditional rule moves it to 6.2 (with a sensitivity range of 7.1 to 16.9 at $\pi = 0.01$). We explicitly instruct readers never to quote the raw 3.0 figure in a clinical context."*

---

### LEVEL 3: MODERATE LIMITATIONS (Technical Constraints)

#### 7. Small Sample Size in Critical Subgroups
- **The Limitation:** The under-40 escalating test subset contains only 21 positive cases ($n=21$); the intersectional under-40 male cell contains only 9 escalating cases and was suppressed.
- **Impact:** Point estimates carry wide uncertainty (under-40 test sensitivity 0.143 has a 95% Clopper-Pearson interval of [0.030, 0.363]).
- **Viva Defense Script:**
  > *"Examiner: Can you really draw major conclusions from only 21 young cancer cases?"*
  > **Candidate Defense:** *"The wide confidence interval is precisely why we report exact Clopper-Pearson intervals rather than asymptotic intervals, and why we compare it to out-of-fold data where we have 64 cases. On OOF, the sensitivity is 0.547, and the confirmatory test against 60+ remains statistically significant (Holm $p = 0.036$). The point estimate of 0.143 is imprecise, but the deficit relative to older bands is established across every split."*

#### 8. Stacking Mismatch in OOF Dirichlet Fitting
- **The Limitation:** The deployed Dirichlet calibrator is fitted on 5-fold OOF probabilities from models trained on 80% data, but applied to full-train checkpoints trained on 100% data.
- **Impact:** Full models are slightly sharper, making the OOF calibrator slightly over-sharp (A7-oof loses 0.018 Macro-F1 vs A7-val).
- **Viva Defense Script:**
  > *"Candidate Defense: This is the classic stacking mismatch. We chose to accept this measured 0.018 Macro-F1 penalty in exchange for quintupling rare-class calibration support from 11 to 45 samples and eliminating degenerate conformal cells. We report both A7-val and A7-oof side-by-side in Table IX and XI."*

#### 9. Absence of Single-Centre Same-Modality Dermoscopy Validation
- **The Limitation:** PAD-UFES-20 evaluated smartphone photography (modality shift). BCN-20000 and MSKCC were evaluated on frozen features, but BCN's baseline transfer was poor (Macro-F1 0.402).
- **Impact:** True clinical transportability within an external primary-care dermoscopy workflow remains unvalidated.

---

### LEVEL 4: MINOR LIMITATIONS (Reporting & Cosmetic Constraints)

#### 10. Planned Components Dropped During Development
- **The Limitation:** Shades-of-Grey color constancy was planned in early roadmaps but never implemented. The multi-centre comparison retraining arm (E0) was withdrawn.
- **Impact:** The ablation ladder evaluates 11 rungs rather than the originally scoped 13.
- **Viva Defense Script:**
  > *"Candidate Defense: Rather than leaving phantom rungs in our ablation ladder, we explicitly reported color constancy as absent in our CLAIM checklist (Item 12). For arm E0, because our published checkpoints were frozen, running it would have required new retraining, so it was withdrawn and entered into the Holm family at $p = 1.0$."*

#### 11. Single GPU Hardware Constraint
- **The Limitation:** All training was performed on a single laptop-class NVIDIA GPU with 8 GB VRAM.
- **Impact:** Constrained model architectures to moderate-capacity backbones ($224 \times 224$ input resolution) and prevented training larger vision transformer variants (e.g., Swin-Base, ViT-Large).
