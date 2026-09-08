# Formal Presentation Plan & Word-for-Word Speaker Script

**Target Duration:** 15–20 minutes (15 Slides)  
**Tone:** Authoritative, clinically grounded, methodologically rigorous, and scientifically candid.

---

## 1. Slide-by-Slide Presentation Structure

### SLIDE 1: Title & The Core Problem
- **Title:** Beyond Benchmark Discrimination: Subgroup Calibration, Abstention, and Conformal Guarantees in Dermoscopy
- **Bullets:**
  - Standard reporting conventions prioritize aggregate discrimination (Accuracy, AUC, Macro-F1).
  - High benchmark performance routinely conceals safety-critical clinical failures.
  - Deployability is a measurable property requiring subgroup-level auditing.
  - We reverse-engineer a 6-CNN ensemble on HAM10000 under strict leak-free pre-registration.
- **Visual:** Graphical abstract showing the pipeline from input image through ensembling to safety layers.
- **Time:** 1.0 minute.
- **What to Say:** Frame the research around the gap between leaderboard accuracy and clinical safety.
- **What NOT to Say:** Do not claim you built a "state-of-the-art diagnostic system ready for the clinic."
- **Likely Examiner Interruption:** *"Why HAM10000 again when newer datasets exist?"*

### SLIDE 2: The Benchmark Pathology: The 67% Nevus Imbalance
- **Title:** The Imbalance Trap: Why Accuracy is Clinically Deceptive
- **Bullets:**
  - HAM10000 test split: Melanocytic Nevus (`nv`) accounts for 66.8% of images.
  - An all-nevus dummy model achieves 66.8% accuracy with 0% cancer sensitivity.
  - Seven diagnostic classes mapped into an escalating triage set: $\mathcal{E} = \{\text{AKIEC}, \text{BCC}, \text{MEL}\}$.
  - Strict lesion-grouped partitioning prevents near-duplicate leakage (10,015 images on 7,470 lesions).
- **Visual:** Table I (Class distribution and actionability tiers) + diagram of near-duplicate lesion leakage.
- **Time:** 1.0 minute.
- **What to Say:** Explain why lesion-level splitting is Hard Rule 1 and why Macro-F1 replaces accuracy.
- **What NOT to Say:** Do not confuse lesion counts with image counts.

### SLIDE 3: Architectural Exhaustion: The Ablation Ladder
- **Title:** The Ablation Ladder & The Ceiling of Architectural Tuning
- **Bullets:**
  - 6 ImageNet CNNs trained under identical 2-stage transfer learning schedules.
  - Backbone spread (ResNet-50 0.7058 to ConvNeXt-Tiny 0.7459) is statistically unresolvable ($p = 0.42$).
  - SwinV2-Tiny (0.7273) and Gated Metadata Fusion (0.7411) do not beat the CNN baseline.
  - 6-CNN uniform soft-voting is the **only** rung clearing significance ($0.7718$, McNemar Holm $p = 2.0 \times 10^{-4}$).
- **Visual:** Table IX (Block A ladder rungs) with bootstrap confidence intervals and McNemar annotations.
- **Time:** 1.5 minutes.
- **What to Say:** Emphasize that uniform soft-voting was chosen over ridge stacking to avoid test snooping.
- **What NOT to Say:** Do not claim you invented a new ensembling technique; emphasize the statistical proof.

### SLIDE 4: Five Levers, Five Negative Results
- **Title:** Negative Exhaustion: What Does Not Work Beyond Macro-F1 0.8047
- **Bullets:**
  - Adding Vision Transformers to the ensemble: Wash (0.7981 vs 0.7986).
  - Caruana greedy forward selection: Overfits validation variance (0.7640/0.7839).
  - Global and per-class prior logit adjustments: Inert / noise due to effective-number training loss.
  - 30-member fold-bagged ensemble (Rung A8): 0.7810 ($-0.0237$ vs A7, Holm $p = 0.242$).
  - Macro-F1 0.8047 is the empirical ceiling of these checkpoints.
- **Visual:** Table X (Negative combination levers summary).
- **Time:** 1.5 minutes.
- **What to Say:** Frame negative results as a primary scientific contribution: the interesting axis has moved from accuracy to usability.
- **What NOT to Say:** Do not apologize for negative results.

### SLIDE 5: Calibration Antinomy: Proper Scoring vs Clinical Safety
- **Title:** The Calibration Paradox: Better Probabilities, More Missed Cancers
- **Bullets:**
  - The soft-vote ensemble is systematically **underconfident** (confidence 0.7048 vs accuracy 0.8609, signed gap $-0.156$).
  - Dirichlet calibration cuts ECE from 0.1575 to 0.0206 and raises Macro-F1 to 0.8047.
  - **The Trade-off:** Dirichlet calibration degrades escalation sensitivity from 0.7862 to 0.7310.
  - Missed serious malignancies increase from 62 to 78 (+16 missed cancers).
- **Visual:** Fig. 2 (Reliability diagrams before and after Dirichlet calibration) + Sensitivity trade-off callout.
- **Time:** 1.5 minutes.
- **What to Say:** Explain the mechanism: soft-voting hedges runner-up probabilities, and Dirichlet shifts mass to nevi.
- **What NOT to Say:** Do not claim the model was overconfident like single networks.

### SLIDE 6: Subgroup Calibration: The Illusion of Global ECE
- **Title:** Subgroup Calibration: How Global Maps Conceal Opposite Errors
- **Bullets:**
  - Global calibration error of 0.017 masks large, opposite-signed demographic residuals.
  - Uncalibrated ECE: Under-40 is 0.167; 40–59 is 0.198; 60+ is 0.113.
  - After global Dirichlet calibration:
    - Patients 40–59 remain **underconfident** (signed gap $-0.023$).
    - Patients 60+ are pushed into **overconfidence** (signed gap $+0.034$).
  - Aggregate gap ($+0.004$) cancels opposite errors: the empirical case for multicalibration.
- **Visual:** Table II (Per-age-band calibration metrics before and after Dirichlet).
- **Time:** 1.0 minute.
- **What to Say:** Show how a single global affine map cannot simultaneously sharpen one group and soften another.
- **What NOT to Say:** Do not claim under-40 is always the worst calibrated; explain that rank order is split-unstable while sign disagreement replicates.

### SLIDE 7: Conformal Prediction: The Marginal Coverage Fallacy
- **Title:** Conformal Prediction: Marginal Guarantees Fail Cancer Patients
- **Bullets:**
  - Marginal LAC at $\alpha = 0.10$ achieves 90.4% empirical coverage—satisfying the mathematical theorem.
  - But coverage on malignant lesions collapses to **74.8%**.
  - Emits 63 prediction sets for cancer patients containing *zero* escalating diagnoses.
  - Mondrian class-conditional calibration repairs malignant coverage to 91.4% (LAC) and 94.1% (RAPS).
  - False Reassurance Rate (FRR) formalizes the clinical endpoint.
- **Visual:** Table III + Fig. 3 (Per-class empirical coverage profiles).
- **Time:** 1.5 minutes.
- **What to Say:** Explain that marginal guarantees are financed by the majority nevus class at the expense of cancer patients.
- **What NOT to Say:** Do not say conformal prediction is flawed; explain that *marginal conditioning* is clinically misaligned.

### SLIDE 8: The Under-40 Melanoma Blind Spot
- **Title:** Hidden Stratification: Acute Young-Patient Triage Failure
- **Bullets:**
  - Aggregate escalation sensitivity: 0.731.
  - Patients aged 60+: 0.764 sensitivity (152/199 caught).
  - **Patients under 40:** Escalation sensitivity collapses to **0.143 [0.030, 0.363]** (only 3/21 caught!).
  - Confirmatory test: Difference of $-0.621$ [$-0.786, -0.386$], Holm $p < 0.001$.
  - 18 young malignancies misclassified as benign moles.
- **Visual:** Table V (Escalation sensitivity across age bands on Val, OOF, and Test).
- **Time:** 1.5 minutes.
- **What to Say:** Highlight that early melanoma detection carries the greatest life-year benefit in young adults.
- **What NOT to Say:** Do not quote 0.143 bare without the [0.030, 0.363] confidence interval.

### SLIDE 9: Root Cause & The Confidently Wrong Failure Mode
- **Title:** Shortcut Learning & The Failure of Selective Abstention
- **Bullets:**
  - Training prior skew: Only 4.9% of lesions under 40 are escalating vs 35.5% at 60+ (7.3$\times$ skew).
  - The model learned an empirical shortcut: "young patient = benign nevus."
  - **Abstention Fails:** At 10% budget, the gate defers only 6.2% of young cases (vs 17.3% at 60+).
  - Abstention rescues only 2 of 18 young misses (11.1%).
  - The model is **confidently wrong**, evading standard uncertainty tripwires.
- **Visual:** Fig. 6 (Selective risk-coverage curve vs subgroup failure).
- **Time:** 1.5 minutes.
- **What to Say:** Explain why margin-based abstention is blind to high-confidence prior-driven errors.
- **What NOT to Say:** Do not blame the uncertainty score; explain that the probability vector itself lacks uncertainty signals here.

### SLIDE 10: Dissecting the Mechanism: Decision Rule vs Lost Information
- **Title:** Dissecting the Deficit: Signal Loss vs Decision-Rule Discarding
- **Bullets:**
  - Threshold-free diagnostic: Within-band escalation-mass AUC ($S_{\text{esc}} = \sum_{c \in \mathcal{E}} p_c$).
  - Validation suggested pure decision-rule failure (AUC 0.927 vs 0.934).
  - **Held-out test split revealed both:** Under-40 AUC dropped to **0.810** vs 0.975 (40–59) and 0.933 (60+).
  - The probability vector still carries signal (0.810 vs 0.50 chance), but ranking is genuinely impaired.
  - Decision rules can recover part of the gap, but cannot close it.
- **Visual:** AUC comparison bar chart across validation, OOF, and test.
- **Time:** 1.5 minutes.
- **What to Say:** Be transparent about retracting the "pure decision rule" claim; explain that test ranking is genuinely weaker.
- **What NOT to Say:** Do not pretend that within-band AUC is identical across age groups.

### SLIDE 11: The Age-Conditional Escalation Rule & Clinical NNB
- **Title:** A One-Parameter Mitigation & Its Clinical Biopsy Price
- **Bullets:**
  - Decision rule: $\hat{y}(x) = \arg\max_c (p_c(x) + \lambda_{b(x)} \mathbf{1}[c \in \mathcal{E}])$.
  - Frozen values from OOF grid search under 85% specificity floor: $\lambda_{<40}=0.26, \lambda_{40-59}=0.74, \lambda_{60+}=0.33$.
  - **Overall Impact:** Sensitivity rises $0.731 \to 0.831$; missed serious cases fall $78 \to 49$.
  - **Subgroup Impact:** Under-40 sensitivity rises $0.143 \to 0.238$ (mitigated, not solved).
  - **Clinical Price:** Re-weighted NNB ($\pi = 0.03$) moves from 3.0 to 6.2 (6.2 biopsies per cancer found).
- **Visual:** Table VI (Test split impact of the $\lambda$ rule and NNB).
- **Time:** 1.5 minutes.
- **What to Say:** Explain that one scalar avoids overfitting small positive counts (64 OOF cases).
- **What NOT to Say:** Do not claim the age rule solved the problem; it reached only 0.238.

### SLIDE 12: Safety Net Redundancy: Orthogonality Breakdown
- **Title:** Redundant Safety Nets: When Abstention and Decision Rules Collide
- **Bullets:**
  - Are abstention and the age rule complementary?
  - Patients 40–59: Complementary (Jaccard overlap 0.18, rule catches 11 misses, abstention catches 2).
  - Patients 60+: Largely redundant (Jaccard overlap 0.83).
  - **Patients under 40:** **100% Redundant (Jaccard index 1.00)**.
  - Both catch the exact same 2 cases; 16 confidently wrong misses lie outside both nets.
- **Visual:** Table VII (Jaccard overlap matrix across age bands).
- **Time:** 1.0 minute.
- **What to Say:** Acknowledge this negative result inside your own headline contribution openly.
- **What NOT to Say:** Do not claim your multi-layer architecture provides independent defenses for young patients.

### SLIDE 13: Multi-Centre Dermoscopy Replication (BCN-20000 & MSKCC)
- **Title:** Multi-Centre Replication: The Operating Point Exports, The Mechanism Does Not
- **Bullets:**
  - Evaluated on 11,982 BCN-20000 and 2,903 MSKCC dermoscopy images with zero target tuning.
  - **Pre-registered Claim A (Invariant AUC):** FAILED (spread 0.105 across centres).
  - **Pre-registered Claim B (Sensitivity tracks skew):** FAILED BACKWARDS (HAM > MSKCC > BCN).
  - **The Operating Point Survived:** Frozen $\lambda$ lifted under-40 sensitivity in all three centres:
    - HAM: $0.547 \to 0.625$ | BCN: $0.279 \to 0.352$ | MSKCC: $0.333 \to 0.389$.
  - BCN confirmatory McNemar: $p = 1.49 \times 10^{-8}$ (structurally one-sided).
- **Visual:** Fig. 4 (Dose-response test outcome) + Table VIII Panel A.
- **Time:** 1.5 minutes.
- **What to Say:** Emphasize that the operating point transports, but causal attribution does not.
- **What NOT to Say:** Do not try to invent an ad-hoc rescue for Claim B.

### SLIDE 14: Smartphone Modality Shift & Shift Detection (PAD-UFES-20)
- **Title:** Out-of-Distribution Shift: Ensembles Fail, Density Detectors Protect
- **Bullets:**
  - Applied frozen system to 2,106 PAD-UFES-20 smartphone photographs.
  - Macro-F1 collapsed to 0.167 (worse than single ConvNeXt-Small at 0.188).
  - Prior shift decoupling proves optical distortion accounts for the majority of transfer failure.
  - Unsupervised Saerens EM prior correction failed ($p = 2.89 \times 10^{-23}$ in the wrong direction).
  - **The Tripwire:** Penultimate Mahalanobis distance detects phone shift with **AUROC 0.9128**.
- **Visual:** Table VIII Panel B + Table XII Panel A (Shift detection and conformal widening).
- **Time:** 1.0 minute.
- **What to Say:** Position PAD strictly as a modality shift benchmark, highlighting Mahalanobis distance as an automated tripwire.
- **What NOT to Say:** Do not claim this validates smartphone clinical deployment.

### SLIDE 15: Limitations, Non-Claims & Final Verdict
- **Title:** Methodological Limitations & The Researcher's Verdict
- **Bullets:**
  - **What we do NOT claim:** We do not claim clinical deployability; we do not claim the young failure is solved; we do not claim validity on dark skin (Types V–VI underpowered).
  - **Critical Gaps:** Exchangeability broken under OOF conformal; retrospective curated data.
  - **Core Contribution:**
    1. Proved architectural tuning on HAM10000 has hit an empirical exhaustion ceiling.
    2. Documented and priced an acute age-stratified triage failure hidden by aggregate metrics.
    3. Demonstrated that safety nets fail where errors concentrate, establishing a template for subgroup auditing.
- **Visual:** Summary box contrasting traditional benchmark claims vs audited deployability reality.
- **Time:** 1.5 minutes.
- **What to Say:** Conclude with scientific humility and call for representation-learning interventions.
- **What NOT to Say:** Do not end on a generic "AI will transform dermatology" platitude.

---

## 2. Word-for-Word Spoken Presentation Script

### [Slide 1: Title & Framing]
"Good morning, members of the committee. 

Today I am presenting our work titled: *An Ablation-Grounded Ensemble for Dermoscopic Skin Lesion Classification: Subgroup-Conditional Calibration, Abstention, and Conformal Guarantees*.

For nearly a decade, dermatological AI literature has converged on a common reporting convention: papers train a new convolutional backbone or vision transformer on benchmarks like HAM10000, report a one-to-two point improvement in accuracy or Macro-F1, and imply the system is approaching clinical readiness.

This paper is organized around the claim that this convention has outlived its usefulness. We argue that clinical deployability is a measurable property, that measuring it changes which engineering design choices look worthwhile, and that measuring it in aggregate is dangerously insufficient. 

Over the next fifteen minutes, I will reverse-engineer our research pipeline to show why architectural optimization has plateaued, how standard calibration and safety nets fail in unexpected ways, and how aggregate benchmark metrics can conceal a catastrophic blind spot in young melanoma patients."

### [Slide 2: The Benchmark Pathology]
"To understand why benchmark metrics mislead, we must first look at the dataset structure. 

We used the HAM10000 benchmark—10,015 dermoscopic images covering seven disease categories. Our first critical methodological finding was that these 10,015 images sit on only 7,470 unique physical lesions. In the literature, many studies apply standard random image-level splits. That is a fatal error: near-duplicate views of the same lesion leak between training and test sets, allowing neural networks to memorize patient-specific artifacts rather than learning pathology. We instituted Hard Rule 1: every split must be grouped strictly by `lesion_id`.

Furthermore, look at Table I. Melanocytic nevi—ordinary benign moles—account for 66.8% of the held-out test split. An unweighted model that simply guesses 'nevus' every time achieves nearly 67% accuracy while missing every single cancer. That is why Hard Rule 3 of our project was: *never use accuracy as a selection criterion*. Instead, we evaluated Macro-F1 across all seven classes, and collapsed the diagnoses into an escalating set—Actinic Keratosis, Basal Cell Carcinoma, and Melanoma—to evaluate clinical triage sensitivity."

### [Slide 3: Architectural Exhaustion]
"We began by building an 11-rung ablation ladder on 1,502 held-out test images under a pre-registered analysis plan.

The first major finding is that architecture barely matters. Across six diverse ImageNet-pretrained CNN backbones, test Macro-F1 ranged from 0.7058 for ResNet-50 to 0.7459 for ConvNeXt-Tiny. As our paired bootstrap intervals and McNemar tests show, this gap is statistically unresolvable at the size of a standard test split ($p = 0.42$). Standalone Vision Transformers like SwinV2-Tiny scored 0.7273, and adding multimodal patient metadata reached only 0.7411. 

The important point here is that ensembling was the *only* design choice that cleared formal statistical significance. A uniform arithmetic soft-vote over the six CNNs reached Macro-F1 0.7718, achieving a decisive McNemar $p$-value of $2.0 \times 10^{-4}$ after Holm correction. Adding 24-view test-time augmentation brought us to 0.7859, and Dirichlet calibration reached 0.8047. 

Crucially, we resisted test-set snooping: a non-negative ridge stacking ensemble achieved test Macro-F1 of 0.7815, but had the worst validation score. Reporting ridge stacking would have been unscientific selection on the test set; we rejected it and reported uniform soft-voting."

### [Slide 4: Negative Combination Levers]
"Having reached Macro-F1 0.8047, the obvious next question was: can anything get past it?

We tested five further candidate levers under strict cross-fitting: adding vision transformers to the ensemble, Caruana greedy forward selection, post-hoc prior logit adjustments, per-class threshold shifts, and a 30-member fold-bagged ensemble. 

All five came back completely negative. Adding transformers was a wash; Caruana selection overfit validation sampling variance; logit adjustment failed because our training already used effective-number weighting; and 30-member fold bagging scored 0.7810. 

Macro-F1 0.8047 represents the empirical ceiling of these checkpoints. We report this exhaustion as a primary scientific result: single-split architectural benchmarking on HAM10000 has exhausted its informational content. The informative axis has moved from aggregate accuracy to usability and safety."

### [Slide 5: Calibration Antinomy]
"This brings us to calibration. When we evaluated our soft-vote ensemble, we discovered an unexpected phenomenon: the ensemble was systematically *underconfident*.

Mean confidence was 0.7048 against an empirical accuracy of 0.8609—a signed gap of $-0.156$ that accounts for essentially all of our 0.1575 calibration error. This is the exact opposite of the single-network overconfidence reported by Guo et al. The mechanism is soft-voting itself: averaging probability vectors that disagree on runner-up classes dilutes the peak probability, leaving the ensemble hedged. Multi-class Dirichlet calibration resolved this globally, cutting test ECE to 0.0206.

However, what we discovered next was a fundamental clinical antinomy: calibration and screening sensitivity pull in opposite directions. 

While Dirichlet calibration raised Macro-F1 to 0.8047, it dropped escalation sensitivity from 0.7862 to 0.7310, increasing missed serious malignancies from 62 to 78. Calibration redistributes probability mass toward the 67% nevus class. What is optimal for proper scoring rules is directly harmful in a screening pathway. A clinic cannot adopt calibration without also adopting a decision rule that restores sensitivity."

### [Slide 6: Subgroup Calibration]
"Furthermore, aggregate calibration conceals dangerous demographic disparities. 

As Table II demonstrates, our global post-calibration ECE of 0.017 looks exceptional. But slicing by age band reveals that a single global affine map cannot express subgroup calibration. Uncalibrated ECE ranged from 0.113 in elderly patients to 0.198 in middle-aged patients.

More consequentially, after Dirichlet calibration, the 40–59 band remains underconfident at $-0.023$, while the 60+ band is pushed into overconfidence at $+0.034$. Our near-perfect aggregate signed gap of $+0.004$ is an illusion created by two errors of opposite signs cancelling each other out. This is a direct empirical argument for group-wise multicalibration."

### [Slide 7: Conformal Fallacy]
"Next, we examined split conformal prediction to evaluate whether mathematical coverage guarantees ensure clinical safety.

Table III contains what we consider our sharpest methodological result. At error rate $\alpha = 0.10$, standard marginal Least Ambiguous Classifier sets achieved 90.4% empirical coverage. The mathematical theorem holds exactly as advertised. 

Yet, when we restricted evaluation to malignant lesions, coverage collapsed to 74.8%. The system produced 63 prediction sets for cancer patients that contained *no escalating diagnosis whatsoever*. The 90% average was being financed entirely by the benign majority.

Mondrian class-conditional calibration repaired aggregate cancer coverage to 94.1%, cutting false reassurances to 6. But as we will see, conditioning on the label still left demographic subgroups completely unprotected."

### [Slide 8: The Under-40 Melanoma Blind Spot]
"This brings us to the core clinical vulnerability exposed in our research: the under-40 melanoma blind spot.

Every aggregate metric reported to this point indicated a competent, well-calibrated classifier. Stratifying by patient age proved it is not. As shown in Table V, while the model catches 76.4% of escalating cancers in patients aged 60+ and 81.4% in middle-aged patients, escalation sensitivity in patients under 40 collapsed to **0.143** [95% CI: 0.030, 0.363].

Out of 21 young cancer patients in the test split, the model caught only 3. Eighteen patients with invasive cancers—predominantly melanomas—were classified as ordinary benign moles. Our pre-specified confirmatory test between under-40 and 60+ gave a difference of $-0.621$, significant at Holm $p < 0.001$."

### [Slide 9: Root Cause & Abstention Failure]
"Why did this happen? 

The source is shortcut learning driven by the training prior. In the training split, only 4.9% of lesions in patients under 40 required escalation, compared to 35.5% in elderly patients. The network learned a statistical shortcut: 'young patient, therefore mole.' Clinically, early melanoma detection in young adults carries the largest survival benefit, meaning the model failed worst exactly where errors cost the most life-years.

One might assume that selective classification would catch these errors. It does not. Because of the overwhelming nevus prior, the model is not uncertain about these young melanomas; it is **confidently wrong**. At a 10% abstention budget, the gate deferred only 6.2% of under-40 cases, rescuing only 2 of the 18 missed malignancies. The safety net thins out exactly where it is needed most."

### [Slide 10: Dissecting the Mechanism]
"To determine whether this was a decision-rule failure or an information-loss failure, we evaluated within-band escalation-mass AUC—a threshold-free diagnostic measuring whether the probability vector ranks cancers above moles.

On validation data, under-40 AUC was 0.927, leading us to initially hypothesize a pure decision-rule failure. However, on held-out test data, under-40 AUC dropped to 0.810 vs 0.975 in older bands. 

The honest conclusion is that the failure is twofold: it is partly decision-rule discarding (which a threshold can recover) and partly genuine feature representation loss (which no threshold can recover). The signal is not absent, which licenses attempting a decision rule, but ranking is genuinely impaired."

### [Slide 11: The Age-Conditional Escalation Rule]
"To mitigate this failure, we engineered a one-parameter age-conditional escalation rule:
$$\hat{y}(x) = \arg\max_{c} \left( p_c(x) + \lambda_{b(x)} \mathbf{1}[c \in \mathcal{E}] \right)$$
We optimized one scalar $\lambda$ per band on out-of-fold cross-validation predictions under an 85% specificity floor, freezing $\lambda_{<40} = 0.26, \lambda_{40-59} = 0.74,$ and $\lambda_{60+} = 0.33$.

On test data, the rule lifted overall sensitivity from 0.731 to 0.831, cutting missed malignancies from 78 to 49. But look at the young cohort: under-40 sensitivity rose only from 0.143 to 0.238. The blind spot was mitigated, not closed.

We priced this intervention in clinical currency: overall Number Needed to Biopsy at 3% screening prevalence increased from 3.0 to 6.2. The rule purchases 29 fewer missed cancers at the cost of roughly three additional benign biopsies per cancer found."

### [Slide 12: Redundant Safety Nets]
"We then tested whether our two safety nets—abstention and the $\lambda$ rule—were complementary.

In middle-aged patients, they were highly complementary: abstention caught 2 misses, the rule caught 11, with a Jaccard overlap of only 0.18. 

However, in the under-40 band, Table VII reveals a negative result inside our own contribution: of 18 missed malignancies, abstention deferred 2 and the rule caught 2—**the exact same 2 cases** (Jaccard index 1.00). 

In the very subpopulation the rule was designed for, it was not an independent safety net. Sixteen confidently wrong young melanoma misses evaded both mechanisms entirely."

### [Slide 13: Multi-Centre Dermoscopy Replication]
"To test whether these findings generalize, we scored two independent same-modality dermoscopy centres from ISIC-2019: BCN-20000 (11,982 images) and MSKCC (2,903 images) with zero parameter re-tuning.

We pre-registered two mechanistic claims: Claim A predicted invariant escalation-mass AUC; Claim B predicted sensitivity would track prior skew. 

Both mechanistic claims failed. Claim A failed with a 0.105 AUC spread, and Claim B failed backwards—under-40 sensitivity was highest in HAM and lowest in BCN. Following our pre-registered contingency, we dropped the trend language and offered no post-hoc excuse.

Yet, remarkably, the operating point transported: the frozen $\lambda$ rule lifted under-40 sensitivity across all three centres: $0.547 \to 0.625$ in HAM, $0.279 \to 0.352$ in BCN, and $0.333 \to 0.389$ in MSKCC. The central scientific lesson is that operating points transport, but causal explanations do not."

### [Slide 14: Smartphone Modality Shift]
"Finally, we evaluated the system under radical domain shift using 2,106 smartphone clinical photographs from PAD-UFES-20.

The ensemble collapsed completely to Macro-F1 0.167, performing worse than its best single member because domain shift caused correlated errors across backbones. Decomposing the shift proved that optical distortion accounted for the vast majority of the failure, and unsupervised Saerens EM prior correction made matters worse ($p = 2.89 \times 10^{-23}$).

However, we found a vital positive safety result: penultimate-layer Mahalanobis distance separated dermoscopy from smartphone photos with **AUROC 0.9128**, exhibiting an 18-fold separation. This demonstrates that while feature-space distance cannot rank in-distribution ambiguity, it serves as an exceptional tripwire to refuse out-of-distribution inputs."

### [Slide 15: Limitations & Final Verdict]
"To conclude, let me state plainly what this paper does NOT claim.

We do not claim this system is deployable. Even after mitigation, under-40 sensitivity is only 0.238, and we would not deploy this model in young adults. We do not claim validity on dark skin; HAM10000 has no Fitzpatrick labels, and Types V and VI on PAD had only 9 images and were suppressed. Furthermore, our OOF conformal analysis serves as an empirical audit, having forfeited the exact finite-sample theorem under cross-validation.

What we have contributed is a rigorous template for safety-net auditing:
First, we proved that architectural ensembling has hit an informational ceiling on HAM10000.
Second, we demonstrated that standard calibration and marginal conformal guarantees provide dangerous illusions of safety in screening.
Third, we documented an acute demographic blind spot and showed that safety nets fail where errors concentrate.
And fourth, we proved that post-processing operating points can transport across clinical centres even when mechanistic explanations fail.

True progress in medical AI will not come from chasing fractional benchmark gains, but from understanding where systems fail, pricing their clinical costs, and enforcing subgroup accountability.

Thank you. I welcome your questions."
