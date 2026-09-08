# Empirical Results Deep Dive: Tables, Figures & Statistical Proofs

**Source Basis:** Manuscript Section IV, Appendix B, `results/ablation_table.csv`, `results/mcnemar_delong.json`, `results/session9/`, `results/external/`, and frozen JSON artifacts.

---

## 1. Master Ablation Ladder (Table IX)

Evaluated on the held-out test split of HAM10000 ($N = 1,502$ images across 1,121 lesions). All 95% confidence intervals are 1,000-sample lesion-grouped bootstraps:

| Block | Rung ID | Configuration Description | Coverage | Test $N$ | Macro-F1 [95% CI] | Balanced Accuracy | Escalation Sensitivity | Missed Serious (AKIEC/BCC/MEL) |
|---|---|---|---|---|---|---|---|---|
| **A** | **A1** | ResNet-50 (weakest CNN baseline) | 1.000 | 1,502 | 0.7058 [0.6470, 0.7501] | 0.7217 | 0.7379 [0.6797, 0.7977] | 76 |
| **A** | **A2** | ConvNeXt-Tiny (best single model on val) | 1.000 | 1,502 | 0.7459 [0.6928, 0.7806] | 0.7792 | 0.7759 [0.7147, 0.8368] | 65 |
| **A** | **A3** | SwinV2-Tiny (Hierarchical Vision Transformer) | 1.000 | 1,502 | 0.7273 [0.6678, 0.7767] | 0.7116 | 0.6621 [0.5956, 0.7309] | 98 |
| **A** | **A4** | Gated Bilinear Metadata Fusion (CNN + Age/Sex/Site) | 1.000 | 1,502 | 0.7411 [0.6844, 0.7870] | 0.7433 | 0.6862 [0.6180, 0.7500] | 91 |
| **A** | **A5** | Uniform Arithmetic Soft-Vote ($K=6$ CNNs) | 1.000 | 1,502 | **0.7718** [0.7201, 0.8118] | 0.7923 | 0.7828 [0.7234, 0.8464] | 63 |
| **A** | **A6** | + 24-View Test-Time Augmentation (TTA) | 1.000 | 1,502 | **0.7859** [0.7345, 0.8242] | 0.8102 | 0.7862 [0.7269, 0.8487] | 62 |
| **A** | **A7** | + Multi-Class Dirichlet Calibration (val-fitted) | 1.000 | 1,502 | **0.8047** [0.7572, 0.8412] | 0.7939 | 0.7310 [0.6679, 0.7978] | **78** |
| **B** | **B1** | + Top-two Margin Abstention @ 5% target | 0.954 | 1,433 | 0.8298 [0.7797, 0.8689] | 0.8143 | 0.7519 [0.6850, 0.8186] | 65 |
| **B** | **B2** | + Top-two Margin Abstention @ 10% target | 0.892 | 1,340 | 0.8577 [0.8137, 0.8908] | 0.8448 | 0.7621 [0.6913, 0.8334] | 54 |
| **B** | **B3** | + Top-two Margin Abstention @ 15% target | 0.836 | 1,255 | 0.8696 [0.8252, 0.9020] | 0.8636 | 0.7750 [0.6956, 0.8550] | 45 |
| **B** | **B4** | + Top-two Margin Abstention @ 20% target | 0.779 | 1,170 | 0.8957 [0.8475, 0.9291] | 0.8806 | 0.7919 [0.7086, 0.8691] | 36 |

### Companion Rungs from Session 9 Single Test Pass
- **Rung A7-oof (OOF-fitted Dirichlet):** Macro-F1 0.7871 [0.7315, 0.8295], Balanced Acc 0.7771, Sensitivity 0.7310, Missed 78. Difference vs A7-val: $-0.0176$ [$-0.0451, +0.0054$], raw $p = 0.150$, Holm $p = 0.242$ (Not significant).
- **Rung A8 (30-member Fold-Bagged Ensemble):** Macro-F1 0.7810 [0.7207, 0.8258], Balanced Acc 0.7604, Sensitivity 0.7345, Missed 77. Difference vs A7-val: $-0.0237$ [$-0.0625, +0.0043$], raw $p = 0.121$, Holm $p = 0.242$ (Not significant).

### Paired Statistical Hypotheses & Significance
- **A2 vs A1 (Backbone Architecture):** Macro-F1 difference $+0.0401$ [$-0.0054, +0.0858$], McNemar $\chi^2 = 0.637, p = 0.425$ (101 only A2 correct vs 89 only A1 correct). **Architecture effect is statistically unresolvable.**
- **A3 vs A2 (Transformer vs Best CNN):** Macro-F1 difference $-0.0186$ [$-0.0744, +0.0340$], McNemar $p = 0.378$ (99 only A3 vs 86 only A2). **Vision Transformer does not beat CNN.**
- **A4 vs A2 (Metadata Fusion vs CNN):** Macro-F1 difference $-0.0048$ [$-0.0536, +0.0417$], McNemar $p = 0.169$ (95 only A4 vs 76 only A2). **Tabular metadata adds no significant gain.**
- **A5 vs A2 (Ensembling):** Macro-F1 difference $+0.0259$ [$-0.0109, +0.0617$], McNemar $\chi^2 = 17.20, p = 3.36 \times 10^{-5}$, **Holm-adjusted $p = 2.0 \times 10^{-4}$** (85 cases correct only for ensemble vs 38 only for ConvNeXt-Tiny). **Decisively significant.**
- **DeLong Per-Class Tests (A5 vs A2):** Across 42 structural tests, exactly one survives Holm correction: BKL one-vs-rest AUC rises from 0.9195 to 0.9572 ($z = 3.610$, raw $p = 3.06 \times 10^{-4}$, **Holm $p = 0.0129$**).

---

## 2. Five Negative Combination Levers (Table X)

All five candidate combination levers evaluated to push beyond Macro-F1 0.8047 failed [M Table X]:

| Lever Evaluated | Experimental Protocol | Held-Out Result | Verdict | Why It Failed |
|---|---|---|---|---|
| **1. 8-Member Mixed Ensemble** | Add SwinV2-Tiny & MaxViT-Tiny to 6 CNNs (soft-vote) | Val 0.7981 vs 0.7986 (6-CNN) | **Wash** | Transformers make correlated errors with CNNs on boundary cases. |
| **2. Caruana Greedy Selection** | Forward selection with replacement over 8 models | Val 0.7640 / 0.7839 vs 0.7945 | **Worse** | Selecting weights on validation split overfits sampling variance (winner's curse). |
| **3. Global Prior Logit Adjustment** | Post-hoc adjustment by tempered label prior $p / \pi^\tau$ | Val $-0.0035 / -0.0061$ | **Dead** | Training already used effective-number weighting; prior adjustment double-corrects. |
| **4. Per-Class Macro-F1 Offsets** | Greedy per-class threshold shifts to optimize Macro-F1 | Val $+0.0009$, signs flip across folds | **Noise** | Tuning 7 threshold scalars on validation noise fails to generalize. |
| **5. 30-Member Fold-Bagged Ensemble** | Average all 30 fold models across 5 folds | Test 0.7810 ($-0.0237$ vs A7, Holm $p=0.242$) | **Negative** | Averaging 30 models hedged probabilities further without improving decision boundaries. |

---

## 3. Calibration & The Screening Sensitivity Antinomy (Table II & Fig. 2)

### The Ensemble Underconfidence Discovery
- Single neural networks are typically overconfident (Guo et al. 2017).
- In this study, the 6-CNN soft-vote ensemble is systematically **underconfident** [M IV-A]:
  - Test Accuracy $= 0.8609$ (or $0.8688$ on raw predictions).
  - Test Mean Confidence $= 0.7048$ (or $0.7105$).
  - **Signed Calibration Gap** $= \text{Mean Confidence} - \text{Accuracy} = -0.156$ (to $-0.158$).
  - Uncalibrated Expected Calibration Error (ECE) $= 0.1575$.
- **Mechanism:** When six models disagree on runner-up classes, arithmetic probability averaging pulls the maximum probability down toward $1/7 \approx 0.143$. The ensemble is more hedged than any single member.
- **Calibrator Ranking:** Dirichlet calibration wins on validation ECE (0.0201) and test ECE (0.0206), beating Matrix Scaling (0.0279) and Temperature Scaling (0.0322). Temperature scaling cannot alter the argmax and leaves Macro-F1 unchanged at 0.7718; Dirichlet improves Macro-F1 to 0.8047 under TTA.

### Per-Age-Band Calibration Slices (Table II)
Evaluated on the held-out test split before and after global Dirichlet recalibration:

| Age Band | Test $n$ | Uncalibrated Signed Gap | Uncalibrated ECE | Dirichlet Signed Gap | Dirichlet ECE | Calibration Residual Behavior |
|---|---|---|---|---|---|---|
| **$<40$** | 290 | $-0.163$ | 0.167 | **$+0.015$** | 0.046 | Slightly overconfident |
| **40–59** | 671 | $-0.197$ | 0.198 | **$-0.023$** | 0.034 | **Remains underconfident** |
| **60+** | 532 | $-0.105$ | 0.113 | **$+0.034$** | 0.063 | **Pushed into overconfidence** |
| **All Ages** | 1,502 | $-0.158$ | 0.158 | **$+0.004$** | 0.017 | Apparent perfection hides opposites |

> **Critical Finding:** A single global affine transformation cannot simultaneously soften an overconfident subgroup and sharpen an underconfident subgroup. The aggregate signed gap of $+0.004$ is an illusion formed by the cancellation of opposite-signed residuals ($-0.023$ in 40–59 vs $+0.034$ in 60+).

### The Calibration-Sensitivity Antinomy
While Dirichlet calibration improves proper scoring rules (cutting ECE from 0.1575 to 0.0206 and boosting Macro-F1 from 0.7859 to 0.8047):
- **Escalation Sensitivity drops:** from $0.7862$ to $0.7310$.
- **Missed Serious Lesions increase:** from 62 to **78** (+16 missed malignancies).
- **Explanation:** Recalibration shifts probability mass toward the dominant `nv` class (66.8%). What is optimal for likelihood scoring is harmful for clinical cancer detection.

---

## 4. Conformal Prediction: Marginal vs Class-Conditional vs Bipartite (Table III, IV & Fig. 3)

### Marginal Coverage Failure (Table III, $\alpha = 0.10$, Validation-Fitted)
- **Marginal LAC:** Empirical coverage $= 90.4\%$ (satisfies nominal 90% theorem).
  - Escalating (Serious) lesion coverage: collapses to **74.8%**!
  - **False Reassurances:** 63 malignant lesions receive a prediction set containing *zero* escalating diagnoses!
- **Mondrian Class-Conditional LAC:**
  - Serious lesion coverage repaired to **91.4%**; false reassurances cut to 17 (mean set size $1.08 \to 2.16$).
- **Mondrian Class-Conditional RAPS:**
  - Serious coverage reaches **94.1%**; false reassurances cut to **6** (mean set size 2.68).

### The Subgroup Conformal Gap: The Under-40 Crisis (Table IV, Test Split, OOF-Calibrated)
When evaluated specifically on malignant lesions in patients under 40 ($n = 21$):

| Conformal Method | Calibration Scheme | Nominal Error $\alpha$ | Marginal Coverage | Serious Lesion Coverage | **Under-40 Serious Coverage ($n=21$)** | False Reassurance Rate (FRR) [95% CI] | Mean Set Size |
|---|---|---|---|---|---|---|---|
| **LAC** | Marginal | 0.10 | 0.901 | 0.734 | **0.238** (5/21 caught!) | 0.228 [0.169, 0.288] (66 missed) | 1.08 |
| **LAC** | Class-Conditional | 0.10 | 0.875 | 0.886 | **0.571** (12/21 caught) | 0.079 [0.051, 0.117] (23 missed) | 1.28 |
| **LAC** | **Bipartite (Equalized)** | 0.10 | 0.884 | 0.907 | **0.952** (20/21 caught!) | 0.059 [0.035, 0.092] (17 missed) | 1.26 |
| **RAPS** | Marginal | 0.10 | 0.902 | 0.752 | **0.143** (3/21 caught!) | 0.214 [0.155, 0.274] (62 missed) | 1.12 |
| **RAPS** | Class-Conditional | 0.10 | 0.900 | 0.903 | **0.667** (14/21 caught) | 0.052 [0.029, 0.084] (15 missed) | 1.40 |
| **RAPS** | **Bipartite (Equalized)** | 0.10 | 0.893 | 0.890 | **0.857** (18/21 caught) | 0.062 [0.037, 0.096] (18 missed) | 1.30 |
| **RAPS** | Marginal | 0.05 | 0.953 | 0.876 | **0.476** (10/21 caught) | 0.090 [0.059, 0.129] (26 missed) | 1.36 |
| **RAPS** | **Bipartite (Equalized)** | 0.05 | 0.954 | **0.955** | **0.905** (19/21 caught) | **0.014 [0.004, 0.035]** (4 missed) | 1.87 |

> **Takeaway:** Class-conditional calibration fixes label imbalance, but leaves patient demographic imbalance unaddressed. Bipartite calibration (conditioning on age band $\times$ escalation requirement) achieves 95.2% under-40 serious coverage at mean set size 1.26 (LAC) and bounds FRR at 1.4% (RAPS $\alpha=0.05$).

---

## 5. Hidden Stratification: The Under-40 Melanoma Blind Spot (Table V)

Escalation sensitivity across patient age bands on held-out test data ($N=1,502$):

| Age Band | Total Images | Escalating Cases ($k/n$) | Escalation Sensitivity [95% CI] | Interval Method | Within-Band Escalation-Mass AUC |
|---|---|---|---|---|---|
| **$<40$** | 290 | 21 | **0.143 [0.030, 0.363]** | Exact Clopper-Pearson | **0.810** |
| **40–59** | 671 | 70 | **0.814 [0.694, 0.918]** | Grouped Bootstrap | **0.975** |
| **60+** | 532 | 199 | **0.764 [0.683, 0.836]** | Grouped Bootstrap | **0.933** |
| **All Ages** | 1,502 | 290 | **0.731 [0.665, 0.793]** | Grouped Bootstrap | **0.950** |

- **Confirmatory Comparison ($<40$ vs $60+$):** Difference $= -0.621$ [$-0.786, -0.386$], **Holm-adjusted $p < 0.001$**.
- **Root Cause (Training Prior Skew):**
  - Under 40: 64 escalating cases out of 1,319 training images $= 4.85\%$.
  - 60+: 893 escalating cases out of 2,512 training images $= 35.55\%$.
  - Skew Ratio $= 7.33\times$ (or $8.45\times$ on lesion level).
- **The Abstention Blind Spot:**
  - 10% target abstention defers 17.3% of 60+ cases, but only **6.2%** of under-40 cases.
  - Of 18 missed malignancies under 40, abstention defers only **2** (11.1% rescue rate vs 36.2% at 60+).
  - The model is **confidently wrong**, rendering uncertainty-based safety nets inoperative.

---

## 6. The Age-Conditional Escalation Rule & Clinical Cost (Table VI, VII, VIII-D & Fig. 5)

Decision rule: $\hat{y}(x) = \arg\max_c (p_c(x) + \lambda_{b(x)} \mathbf{1}[c \in \mathcal{E}])$.

### Test Split Impact (Table VI)
Frozen parameters from OOF: $\lambda_{<40} = 0.26, \lambda_{40-59} = 0.74, \lambda_{60+} = 0.33$.

| Band | $\lambda_b$ | Decision Rule | Escalation Sensitivity [95% CI] | Referral Rate | Observed NNB | NNB at $\pi = 0.03$ |
|---|---|---|---|---|---|---|
| **$<40$** | 0.26 | Argmax | 0.143 [0.030, 0.363] | 0.028 | 2.67 | 5.2 |
| | | **$+\lambda$** | **0.238 [0.082, 0.472]** | 0.066 | 3.80 | **8.1** |
| **40–59** | 0.74 | Argmax | 0.814 [0.694, 0.918] | 0.109 | 1.28 | 2.1 |
| | | **$+\lambda$** | **0.971 [0.926, 1.000]** | 0.234 | 2.31 | **5.9** |
| **60+** | 0.33 | Argmax | 0.764 [0.683, 0.836] | 0.350 | 1.22 | 5.3 |
| | | **$+\lambda$** | **0.844 [0.780, 0.903]** | 0.423 | 1.34 | **7.6** |
| **All Ages** | — | Argmax | 0.731 [0.665, 0.793] | 0.178 | 1.26 | 3.0 |
| | | **$+\lambda$** | **0.831 [0.776, 0.885]** | 0.268 | 1.67 | **6.2** |

- Overall missed serious cases cut from 78 to **49** (29 cancers rescued).
- Macro-F1 cost: $0.7871 \to 0.7492$ ($-0.038$).
- Under-40 sensitivity improves from 0.143 to 0.238: **mitigated, not solved**.

### Orthogonality Breakdown (Table VII)
Intersection of cases deferred by 10% abstention vs rescued by the $\lambda$ rule among argmax misses:
- **$<40$ ($n=18$ misses):** Abstention defers 2; $\lambda$ rescues 2; **Overlap $= 2$**; **Jaccard $= 1.00$**! (Complete redundancy: 16 misses outside both safety nets).
- **40–59 ($n=13$ misses):** Abstention defers 2; $\lambda$ rescues 11; Overlap $= 2$; **Jaccard $= 0.18$** (Highly complementary).
- **60+ ($n=47$ misses):** Abstention defers 17; $\lambda$ rescues 16; Overlap $= 15$; **Jaccard $= 0.83$** (Largely redundant).
- **All Ages ($n=78$ misses):** Abstention defers 21; $\lambda$ rescues 29; Overlap $= 19$; Jaccard $= 0.61$.

### Decision Curve Analysis Net Benefit (Table VIII Panel D & Fig. 5)
Net Benefit $\text{NB} = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \frac{p_t}{1 - p_t}$:
- **All Ages at $p_t = 0.10$:** $\Delta\text{NB} = \text{NB}(\lambda) - \text{NB}(\text{argmax}) = \mathbf{+0.0228}$ [95% CI: $+0.0178, +0.0282$] (Statistically significant net benefit).
- **Under-40 at $p_t = 0.10$:** $\Delta\text{NB} = +0.0015$ [95% CI: $-0.0015, +0.0053$] (**Null**, CI crosses zero).
- **Under-40 at $p_t = 0.20$:** $\Delta\text{NB} = -0.0013$ [95% CI: $-0.0053, +0.0029$] (**Turns negative**).
- *Clinical Interpretation:* The rule earns its referrals overall, but does not achieve net clinical benefit in the under-40 subpopulation because ranking is impaired and false positives carry high decision penalties.

---

## 7. External Multi-Centre Evaluation (Table VIII & Fig. 4)

Zero-target parameter transfer to BCN-20000 ($N=11,982$), MSKCC ($N=2,903$), and PAD-UFES-20 ($N=2,106$):

### Panel A: Three-Centre Dose-Response Replication (ISIC-2019 Dermoscopy)
| Centre / Cohort | Prior Skew (60+ / $<40$) | Escalation-Mass AUC $<40$ [95% CI] | Escalation-Mass AUC 60+ | Under-40 Argmax Sensitivity [95% CI] | Under-40 Frozen $\lambda$ Sensitivity | $\Delta$ Referral Rate |
|---|---|---|---|---|---|---|
| **BCN-20000** | 3.83$\times$ (Least skewed) | 0.791 [0.726, 0.843] | 0.795 | 0.279 [0.179, 0.382] | **0.352** | $+0.026$ |
| **MSKCC** | 6.54$\times$ | 0.790 [0.703, 0.867] | 0.694 | 0.333 [0.186, 0.510] | **0.389** | $+0.019$ |
| **HAM10000** (Source) | 8.45$\times$ (Most skewed) | 0.895 [0.837, 0.944] | 0.934 | 0.547 [0.391, 0.708] | **0.625** | $+0.024$ |

#### Outcome of Pre-Registered Claims:
- **Claim A (Cohort-Invariant AUC):** FAILED. Spread across centres is $0.105$ (0.895 vs 0.791 vs 0.790).
- **Claim B (Sensitivity Tracks Skew Inversely):** FAILED BACKWARDS. Predicted BCN > MSKCC > HAM. Observed: HAM 0.547 > MSKCC 0.333 > BCN 0.279.
- **Operating Point Transfer:** SUCCEEDED. The frozen $\lambda$ lifted under-40 sensitivity across all three centres without refitting ($+0.078$ on HAM, $+0.073$ on BCN, $+0.056$ on MSKCC).
- **McNemar Confirmatory Test (BCN under-40):** $p = 1.49 \times 10^{-8}$ (Holm bound $7.45 \times 10^{-8}$). Structurally one-sided: certifies that 27 cases were rescued, not that net benefit was achieved.

### Panel B: Smartphone Clinical Photography Transfer (PAD-UFES-20)
| Model Variant | Deployable? | Macro-F1 | Balanced Acc | Accuracy | Escalation Sensitivity | Melanoma Recall | Missed Serious Cases |
|---|---|---|---|---|---|---|---|
| **Raw Soft-Vote** | Yes | 0.1669 | 0.3097 | 0.2626 | 0.3565 | 0.1154 | 1,047 |
| **Dirichlet (Deployed)** | Yes | 0.1303 | 0.2931 | 0.2137 | 0.2207 | 0.1154 | 1,268 |
| **EM Prior (Saerens)** | Yes | 0.1018 | 0.1035 | 0.1638 | 0.4155 | 0.0000 | 951 |
| **Oracle Target Prior** | ORACLE | 0.2534 | 0.3327 | 0.4435 | 0.8949 | 0.0577 | 171 |
| **Dirichlet $\to$ Frozen $\lambda$** | Yes | 0.1723 | 0.3079 | 0.2203 | 0.5679 | 0.3269 | 703 |

- **Optical vs Prior Shift:** Oracle prior boosts Macro-F1 from 0.167 to 0.253 (52% relative gain), but 0.253 remains far below the in-distribution 0.805: optical distortion accounts for the vast majority of collapse.
- **Saerens EM Catastrophe:** Significant in the wrong direction ($\chi^2 = 98.73, p = 2.89 \times 10^{-23}$, 321 cases correct only without correction vs 113 with it).
- **OOD Tripwire:** Penultimate Mahalanobis distance separates HAM from PAD at **AUROC 0.9128** (18-fold separation, median score 485 vs 8,717).

---

## 8. Explainability: Grad-CAM Saliency & The Predicted-Class Artifact (Fig. 7)

Evaluated across all 1,502 test images against Tschandl's binary lesion segmentations [M App. B-D, 47]:
- **Lesion Interior Mass Fraction:** Mean $= 0.5227$ [95% CI: 0.5091, 0.5355].
- **Lesion Area Fraction:** Mean $= 0.2561$ of frame.
- **Concentration Ratio:** $\frac{0.5227}{0.2561} = \mathbf{3.454}$. Roughly half of attribution mass concentrates on a quarter of the image.
- **The Predicted-Class Confound / Artifact:**
  - Correct predictions: interior fraction $= 0.5130$ [0.4986, 0.5276].
  - **Incorrect predictions:** interior fraction $= \mathbf{0.5700}$ [0.5327, 0.6112].
  - Misclassified images score *higher* on lesion concentration than correct images!
  - *Why?* Grad-CAM is computed with respect to the predicted class logit. When the model misclassifies a melanoma as `nv`, it strongly activates on pigmented nevus-like textures on the lesion. Saliency maps show *where* the model looked, not that correct features were extracted. Grad-CAM fails as causal mechanistic evidence.
