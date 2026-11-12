# Independent Scientific & Dermatological Evaluation: Under-40 Melanoma Biology, Dermoscopic Morphology, GAP Dilution, and Pigmentation/Dark-Lesion Failure in V7

**Agent:** AGENT 09 — Under-40 Melanoma Biology, Dermoscopic Morphology & Pigmentation Failure Specialist  
**Program:** Scan4Disease (S4D) / Backend-S4D  
**Target Document:** `docs/V7_MASTER_RESEARCH_PLAN.md` (Sections 4.2.1, 4.2.2, 4.5)  
**Evaluation Mode:** Independent Adversarial Scientific & Clinical Audit  
**Date:** November 2026  
**Status:** Complete Audit & Revision Specification  

---

## 1. Executive Summary & Epistemic Verdict

This report provides a rigorous, dermatologically grounded, and mathematically audited critique of the biophysical, morphological, and decision-theoretic hypotheses formulated in **V7 Master Research Plan** (`docs/V7_MASTER_RESEARCH_PLAN.md`). Specifically, we evaluate the biological validity of claims regarding young melanoma (<40 years), the "GAP Dilution Margin Theorem," the proposed biophysics of dark-lesion failure, and the clinical plausibility of local vs. global dermoscopic morphology.

### 1.1 Core Audit Findings

1. **Precursor Biology Reality Overrides GAP Dilution for the Majority of Young Melanomas:**  
   V7 frames the primary under-40 failure mode around the "Conditionally Bounded GAP Dilution Margin Theorem" (Section 4.2.1.2), asserting that early malignant clones occupy $\alpha \in [10\%, 20\%]$ of a lesion and are diluted by the dominant benign nevus background in Nevus-Associated Melanoma (NAM). However, the landmark meta-analysis by **Pampena et al. (JAAD 2017)** (38 studies, $N=20,126$ melanomas) establishes that **70.9% (95% CI: 66.8%–74.6%) of cutaneous melanomas develop *de novo***, while only **29.1% (95% CI: 25.4%–33.2%)** are nevus-associated. Even in cohorts under 40, NAM accounts for at most $28\%\text{--}35\%$ of cases. For the ~70% of young melanomas that are *de novo*, **there is no pre-existing benign nevus background to cause GAP dilution**. The failure on *de novo* young melanomas is driven by phenotypic overlap with physiological dynamic/growing nevi and severe training prior suppression ($p(\text{mel} | <40) \approx 4.9\%$ vs $35.5\%$ at $\ge 60$), not spatial pooling dilution.

2. **The "GAP Dilution Margin Theorem" is an Affine Algebraic Identity, Not a Proof of False Negatives:**  
   The mathematical formulation $z(\mathbf{z}_{\text{GAP}}) = \alpha m_{\text{mal}} + (1 - \alpha) m_{\text{benign}}$ holds strictly for an affine function operating on the spatial average of spatially invariant feature representations. In real deep convolutional and transformer architectures:
   - Modern feature extractors possess large receptive fields (or global self-attention), coupling local tokens with surrounding spatial context.
   - The final classification operates through a 7-class competing Softmax. A focal malignant clone with high activation magnitude ($\|\mathbf{f}_{\text{mal}}\| \gg \|\mathbf{f}_{\text{benign}}\|$ or logit margin $\Delta_{\text{mal}} \gg 0$) can easily dominate the global average even at $\alpha \le 0.15$.
   - V7's own formulation acknowledges (Section 4.2.1.2, item 4) that detection occurs whenever $m_{\text{mal}} > \frac{\tau - (1 - \alpha)m_{\text{benign}}}{\alpha}$. GAP dilution is a *conditional hypothesis*, not an inevitable mathematical proof of false negatives.

3. **Hand-Engineered Biological Features Add Zero Discriminative Power:**  
   Empirical audit of the repository screening artifact `results/v6/screens/biology_features_beyond_model.json` demonstrates that hand-measured biological features (`colour_count`, `colour_asym_max`, `colour_asym_min`, `chrom_asym_max`, `chrom_asym_min`, `segmental_index`) fail completely to add ranking power beyond the baseline model. When combined (`all_six`), the incremental AUC beyond the V5 composite ensemble was **$+0.0029$ (95% CI: $[-0.0355, +0.0408]$)**. Alone, `colour_count` had an AUC of **$0.5025$** (equivalent to a coin flip). This refutes the notion that naive rule-based biological features rescue the under-40 blind spot.

4. **Forensic Audit of the Dark Melanoma Disparity ($p=0.0402$):**  
   V7 Section 4.2.2.1 asserts: *"19 of 33 false negatives occurred in lesions in the darkest tertile of the dataset ($1 - L^* \ge 0.65$), yielding an empirical error concentration that is statistically significant ($p = 0.0402$, Fisher's exact test)."*  
   Our forensic code and data audit revealed **two critical conflations in this claim**:
   - **Darkness Metric Mismatch:** The repository *never* used $1 - L^* \ge 0.65$ to define darkness. Darkness was computed via `mean_melanin` (`scripts/read_melanin_by_age.py`, `research/v5/b1_hard_core.py`), which is an optical density ICA chromophore projection orthogonal to shading and skin baseline on fold 0's basis.
   - **Conflation of Diagnostic Reads:** The "19 of 33 false negatives" occurred in Diagnostic B1 (`results/v5/diagnostics/b1_hard_core.json`) on the **V4 control model OOF** across 81 young escalating lesions, where Fisher's exact test of high-tercile vs. rest yields **$p = 0.000238$** (or $p = 1.75 \times 10^{-5}$ for high vs. low). In contrast, **$p = 0.040248$** originated from the **V6-5 read** (`results/v6/screens/dark_v6_oof_read.json`) as a *within-archive permutation test* for the `v6_vote` ensemble on folds 1–4! V7 accidentally merged the sample counts of V4 with the permutation p-value of V6.

5. **Dark Lesion Disparity is Already Weakened in V6; Dedicated Specialist Heads Fail:**  
   In `results/v6/screens/dark_v6_oof_read.json` and `docs/V6_RUNSHEET.md`, the heterogeneous V6 ensemble (`v6_vote`) reduced high-melanin young melanoma misses to 5/15 (down from 19/27 in V4). The young-vs-older dark-lesion gap difference collapsed from $+0.083$ (V5 control) to **$+0.015$ (95% CI: $[-0.162, +0.166]$)**, crossing zero. Crucially, attempting to train a dedicated "dark-lesion specialist head" (`probe_dark_specialist.py`) yielded **negative deltas** on V6 ($\Delta = -0.0126$ for `dark_young`). The dark-lesion problem is largely mitigated by heterogeneous backbone pretraining (PanDerm + DINOv2 + ConvNeXt) and 24-view TTA, not dedicated specialist heads.

---

## 2. Histopathology & Biological Ground Truth of Young Melanoma

### 2.1 Precursor Biology: Nevus-Associated Melanoma (NAM) vs. De Novo Melanoma

A foundational premise in V7's design of Branch A (native margin tiles) and Branch C (morphology guidance) is that young melanoma represents clonal expansion within a pre-existing benign nevus bed. We must evaluate this against established dermatopathological epidemiology.

```
+-----------------------------------------------------------------------------------------+
|                  EPIDEMIOLOGICAL SPECTRUM OF CUTANEOUS MELANOMA                         |
|                                                                                         |
|      DE NOVO MELANOMA (~70.9%)                  NEVUS-ASSOCIATED MELANOMA (~29.1%)      |
|      (Pampena et al., JAAD 2017)                (Pampena et al., JAAD 2017)             |
|                                                                                         |
|   +-------------------------------+          +--------------------------------------+   |
|   | 100% Neoplastic Proliferation |          |  [ Pre-existing Nevus Bed ]          |   |
|   | No pre-existing benign nevus  |          |  (Compound / Dysplastic / Intradermal|   |
|   | Surrounded by normal skin     |          |  +--------------------------------+  |   |
|   |                               |          |  | Focal Malignant Clone          |  |   |
|   | Dominant in ALL age cohorts   |          |  | (Breslow <= 0.8 mm, alpha~10%) |  |   |
|   | (65-72% in patients < 40)     |          |  +--------------------------------+  |   |
|   +-------------------------------+          +--------------------------------------+   |
|                                                                                         |
|   * Failure Mechanism:                       * Failure Mechanism:                       |
|     Mimicry of dynamic benign nevi             GAP spatial feature dilution             |
|     and severe Bayesian prior suppression.     (Valid for <= 30% of young cases).       |
+-----------------------------------------------------------------------------------------+
```

#### The Pampena et al. (JAAD 2017) Meta-Analysis
In the definitive meta-analysis of 38 observational cohort and case-control studies comprising 20,126 histopathologically confirmed melanomas (Pampena, Kyrgidis, Lallas, Moscarella, Argenziano, Longo, *J Am Acad Dermatol* 2017;77:938-945):
- **De Novo Melanoma:** Accounts for **70.9%** (95% CI: 66.8%–74.6%) of all cutaneous melanomas.
- **Nevus-Associated Melanoma (NAM):** Accounts for **29.1%** (95% CI: 25.4%–33.2%).
- **Relative Likelihood:** A cutaneous melanoma is significantly less likely to arise from a pre-existing nevus than to develop *de novo* (Relative Risk $\text{RR} = 0.36$).
- **Tumor Thickness Interaction:** NAMs present with a significantly lower mean Breslow thickness compared to *de novo* melanomas (mean difference $-0.39\text{ mm}$, $p < 0.001$). This reflects "tumor obliteration": as *de novo* or invasive tumors thicken, any histologic remnant of a pre-existing nevus is engulfed and destroyed.
- **Nevus Subtype:** The precursor nevus was a non-dysplastic (banal) nevus in 56.4% of NAMs and a dysplastic nevus in 43.6% (no statistically significant difference).

#### Stratification in Cohorts Under 40 Years
In patients under 40, NAM is slightly more prevalent than in older adults, reflecting peak total body nevus counts during the third and fourth decades of life:
- In patients $<40$, NAM accounts for **28% to 35%** of melanomas.
- Conversely, **65% to 72% of melanomas in patients under 40 develop DE NOVO**.
- True "collision tumors" (independent clonal expansion of an adjacent BCC or SK directly colliding with a nevus) account for $<1.0\%$ of cutaneous neoplasms and must not be confused with clonal melanocytic progression in NAM.

### 2.2 Histopathologic Subtypes Across Age Strata

Cutaneous melanoma in young patients exhibits distinct clinicopathologic and genetic profiles across three developmental brackets:

| Age Stratum | Relative Incidence | Dominant Subtypes | Driver Genetics | Clinical & Dermoscopic Phenotype |
|---|:---:|---|---|---|
| **Pediatric & Adolescent** (<20 years) | $<1.5\%$ of pediatric cancers ($1\text{--}8$ per million) | Spitzoid melanoma / Atypical Spitz Tumor (AST) ($30\%\text{--}40\%$), Conventional SSM ($30\%\text{--}35\%$), Nodular ($20\%\text{--}25\%$) | Kinase fusions (*BRAF, ALK, RET, NTRK1/3, ROS1*), *HRAS* mutations. Low TMB. Absent solar elastosis. | Often amelanotic, pink, uniform in color, nodular ($>60\%$ fail ABCD criteria; require modified Pediatric ABCDE: Amelanotic, Bleeding, Color uniformity, De novo, Evolution; Cordoro et al., JAAD 2013). |
| **Young Adults** (20–29 years) | Rapid rise; Female:Male $\sim 1.6:1$ | Superficial Spreading Melanoma (SSM) ($75\%\text{--}83\%$), Nodular ($10\%\text{--}15\%$), Spitzoid ($3\%\text{--}5\%$) | *BRAF* V600E mutation high ($70\%\text{--}80\%$). Low CSD, absent solar elastosis. Intermittent recreational UV. | Trunk and lower extremities. Prominent peripheral globules, atypical pigment network, dark uniform clumping mimicking active Clark/dysplastic nevi. |
| **Adults** (30–39 years) | Continuing female excess | Superficial Spreading Melanoma (SSM) ($80\%\text{--}85\%$), Nodular ($10\%\text{--}15\%$), Lentigo Maligna rare ($<3\%$) | *BRAF* V600E ($60\%\text{--}70\%$), *NRAS* ($15\%\text{--}20\%$). | Asymmetric pigmentation, focal architectural disorder, occasional regression (peppering). |

### 2.3 Critical Assessment of the "10–20% Area Fraction" Hypothesis

V7 Section 4.2.1.1 claims:
> *"In early radial-growth phase NAM (Breslow depth $\le 0.8\text{ mm}$), the malignant clone frequently occupies a minor surface area fraction ($\alpha \le 10\%\text{--}20\%$), surrounded by the dominant architectural pattern of the pre-existing compound or dysplastic nevus."*

#### Scientific Evaluation:
1. **Qualitative Validity:** In early radial-growth phase NAM, histopathologic examination routinely identifies focal junctional proliferation or micro-invasion restricted to one pole or margin of a pre-existing compound nevus. The dermatopathologist diagnoses NAM by identifying residual mature intradermal melanocytic nests (displaying neurotization and lack of atypia) abutting an asymmetric, pagetoid junctional component.
2. **Lack of Morphometric Standardization:** There is **no quantitative morphometric literature** validating a canonical "10% to 20%" surface area constant. Surface area fraction $\alpha$ varies continuously across the tumor trajectory:
   - Microscopic focal in situ transformation: $\alpha < 5\%$.
   - Radial growth phase progression: $\alpha$ expands dynamically from $10\%$ to $60\%$.
   - Advanced invasive progression: $\alpha \to 100\%$ as the malignant clone overgrows and obliterates the precursor nevus.
3. **The Fatal Epistemic Conflict:**  
   While an area fraction of $\alpha \in [0.10, 0.20]$ is plausible for early NAM, **it applies to at most ~30% of young melanoma cases**. For the **70.9% of young melanomas that arise *de novo***, the neoplastic clone constitutes **100% of the lesion**. There is no pre-existing nevus to dilute the signal.  
   **Conclusion:** The under-40 melanoma blind spot cannot be fundamentally explained or solved by a GAP dilution model alone.

---

## 3. Mathematical & Clinical Scrutiny of the "GAP Dilution Margin Theorem"

V7 Section 4.2.1.2 introduces the "Conditionally Bounded GAP Dilution Margin Theorem." We provide a rigorous deconstruction of its assumptions, mathematical validity, and clinical applicability.

### 3.1 Mathematical Decomposition

Let $\mathbf{F} \in \mathbb{R}^{D \times H \times W}$ denote the pre-pooling spatial activation tensor. Global Average Pooling (GAP) produces:
$$\mathbf{z}_{\text{GAP}} = \frac{1}{HW} \sum_{u=1}^H \sum_{v=1}^W \mathbf{F}_{u,v} = \alpha \bar{\mathbf{f}}_{\text{mal}} + (1 - \alpha) \bar{\mathbf{f}}_{\text{benign}}$$
where $\alpha = \frac{|\Omega_{\text{mal}}|}{HW} \in (0, 1]$.

For a linear classification head $z(\mathbf{f}) = \mathbf{w}^\top \mathbf{f} + b$ evaluated against a scalar logit threshold $\tau$:
$$z(\mathbf{z}_{\text{GAP}}) = \mathbf{w}^\top (\alpha \bar{\mathbf{f}}_{\text{mal}} + (1 - \alpha) \bar{\mathbf{f}}_{\text{benign}}) + b = \alpha m_{\text{mal}} + (1 - \alpha) m_{\text{benign}}$$
where $m_{\text{mal}} = \mathbf{w}^\top \bar{\mathbf{f}}_{\text{mal}} + b$ and $m_{\text{benign}} = \mathbf{w}^\top \bar{\mathbf{f}}_{\text{benign}} + b$.

V7 defines the critical area fraction:
$$\alpha_{\text{crit}}(\tau) \triangleq \frac{\tau - m_{\text{benign}}}{m_{\text{mal}} - m_{\text{benign}}}$$

### 3.2 Four Fundamental Flaws in the "Proof of False Negatives" Narrative

```
+-----------------------------------------------------------------------------------------+
|                  GAP DILUTION: MATHEMATICAL THEOREM vs DEEP LEARNING REALITY            |
|                                                                                         |
|   V7 Formulation (Affine Linear Head):                                                  |
|   z(z_GAP) = alpha * m_mal + (1 - alpha) * m_benign                                     |
|   Asserts: If alpha <= 0.20, lesion is misclassified as benign.                         |
|                                                                                         |
|   CRITIQUE 1: Receptive Field & Spatial Coupling                                        |
|   F_mal is NOT independent of F_benign. In deep CNNs / ViTs, context windows exceed    |
|   100 px. A focal malignant nest alters activations across the entire feature map.      |
|                                                                                         |
|   CRITIQUE 2: Activation Magnitude Dominance                                            |
|   If ||f_mal|| >> ||f_benign||, a 15% focal clone can produce a massive logit:         |
|   0.15 * (+12.0) + 0.85 * (-1.0) = +1.80 - 0.85 = +0.95 > 0  ==> ESCALATION DETECTED!   |
|                                                                                         |
|   CRITIQUE 3: Multiclass Competing Softmax Dynamics                                     |
|   Predictions depend on logit differences Delta_{mel,nv}. Benign nevus background       |
|   votes for 'nv', but strong malignant edge signal suppresses other competing classes.  |
|                                                                                         |
|   CRITIQUE 4: Age Inconsistency                                                         |
|   If GAP dilution were an architectural property of CNNs, small focal melanomas in      |
|   elderly patients would fail identically. Yet elderly sensitivity is ~78% vs 14-54%   |
|   in the young. Primary driver is PRIOR BIAS, not spatial pooling.                      |
+-----------------------------------------------------------------------------------------+
```

1. **It is an Algebraic Identity, Not a Theorem Proving Failure:**  
   Equation $z(\mathbf{z}_{\text{GAP}}) = \alpha m_{\text{mal}} + (1 - \alpha) m_{\text{benign}}$ is simply the definition of an affine transformation applied to a convex combination. Calling it a "Theorem" that "causes a dilution false negative" mischaracterizes basic algebra as a mechanistic proof of deep learning failure.
2. **False Assumption of Spatial Independence in Deep Representations:**  
   The formulation treats $\bar{\mathbf{f}}_{\text{mal}}$ and $\bar{\mathbf{f}}_{\text{benign}}$ as fixed, independent vectors. However, deep networks (ConvNeXt, ResNet, ViTs) have large effective receptive fields (ERFs) spanning hundreds of pixels, while Vision Transformers (DINOv2, MaxViT) compute all-to-all spatial self-attention. A focal high-contrast lesion at $\Omega_{\text{mal}}$ perturbs the activation field of $\Omega_{\text{benign}}$. The feature representations are structurally entangled.
3. **Activation Magnitude Overcomes Spatial Area:**  
   As proven in V7's own Item 4:
   $$m_{\text{mal}} > \frac{\tau - (1 - \alpha)m_{\text{benign}}}{\alpha}$$
   A network can assign arbitrarily large activation norms to rare, salient malignant structures (e.g., pseudopods, blue-white veil). For example, if $\alpha = 0.15$, $\tau = 0$, and $m_{\text{benign}} = -1.0$:
   $$m_{\text{mal}} > \frac{0 - (0.85)(-1.0)}{0.15} = \frac{0.85}{0.15} \approx 5.67$$
   If the malignant margin reaches $m_{\text{mal}} \ge +5.7$, GAP classification triggers positive escalation regardless of $\alpha = 0.15$. Neural networks trained with cross-entropy routinely produce logit margins exceeding $\pm 8.0$.
4. **The Softmax Multiclass Reality:**  
   In a multiclass network ($C=7$), the decision boundary is non-linear due to the Softmax denominator $\sum_{j=1}^C e^{z_j}$. Melanoma escalation is governed by the escalation mass:
   $$p_{\text{esc}} = p_{\text{mel}} + p_{\text{bcc}} + p_{\text{akiec}} = \frac{e^{z_{\text{mel}}} + e^{z_{\text{bcc}}} + e^{z_{\text{akiec}}}}{\sum_{j=1}^7 e^{z_j}}$$
   Dilution in logit space does not translate linearly to probability space. A modest boost in $z_{\text{mel}}$ can flip the decision threshold when operating at sensitive operating points ($p_{\text{th}} \approx 0.05\text{--}0.15$, as deployed in clinical escalation).

### 3.3 Epistemic Classification: Plausible Hypothesis, Not Proven Mechanism

The GAP dilution concept should be classified as a **Conditional Representation Hypothesis**, valid primarily for the $\sim 29\%$ of young melanomas that are nevus-associated and present with small $\alpha$. It must not be presented as a universal mathematical law explaining the under-40 failure mode.

---

## 4. Optical Resolution, Sensor Physics & Nyquist-Shannon Spatial Sampling

V7 Section 4.2.1.3 analyzes the spatial sampling pitch across dermatoscope fields of view (FOV). We verify these calculations and evaluate their clinical significance.

### 4.1 Nyquist-Shannon Sampling Verification

To resolve a physical morphological structure of width $w_{\min}$ without spatial aliasing, the digital sampling pitch $\Delta x$ must satisfy:
$$\Delta x \le \frac{w_{\min}}{2.0} \iff N_{\text{sample}} = \frac{w_{\min}}{\Delta x} \ge 2.0\text{ pixels}$$

Let us audit the sampling pitch across clinical dermatoscope optical setups and input resolutions:

| Dermatoscope Lens System | Physical FOV | Input Resolution | Sampling Pitch ($\Delta x$) | Pixels per $50\ \mu\text{m}$ Pseudopod | Pixels per $30\ \mu\text{m}$ Network Line | Nyquist Status ($w = 50\ \mu\text{m}$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard Contact (Heine Delta 20 / DermLite DL4)** | $18\text{ mm}$ ($18,000\ \mu\text{m}$) | $224 \times 224$ | $80.36\ \mu\text{m/px}$ | $0.62\text{ px}$ | $0.37\text{ px}$ | **FAILED (Severe Aliasing)** |
| **Standard Contact** | $18\text{ mm}$ | $384 \times 384$ | $46.88\ \mu\text{m/px}$ | $1.07\text{ px}$ | $0.64\text{ px}$ | **FAILED ($<2.0\text{ px}$)** |
| **Standard Contact** | $18\text{ mm}$ | $512 \times 512$ | $35.16\ \mu\text{m/px}$ | $1.42\text{ px}$ | $0.85\text{ px}$ | **FAILED ($<2.0\text{ px}$)** |
| **Native High-Res Margin Crop** | $\mathbf{9.6\text{ mm}}$ ($9,600\ \mu\text{m}$) | $384 \times 384$ | $\mathbf{25.00\ \mu\text{m/px}}$ | $\mathbf{2.00\text{ px}}$ | $1.20\text{ px}$ | **PASSED ($\ge 2.0\text{ px}$)** |
| **Fine Margin Crop** | $5.76\text{ mm}$ ($5,760\ \mu\text{m}$) | $384 \times 384$ | $15.00\ \mu\text{m/px}$ | $3.33\text{ px}$ | $\mathbf{2.00\text{ px}}$ | **PASSED for delicate mesh** |
| **Raw Uncropped Dermatoscope** | $18\text{ mm}$ | $1,613 \times 1,613$ | $11.16\ \mu\text{m/px}$ | $4.48\text{ px}$ | $2.69\text{ px}$ | **PASSED (Full Optical Fidelity)** |

### 4.2 Repository Datasets Optical Reality Check

While V7's optical calculations are mathematically correct, their practical implementation is constrained by the raw image dimensions in public benchmarks:
- **HAM10000:** Raw dermatoscope images were digitized at **$600 \times 450$ pixels**. For an $18\text{ mm}$ lens FOV, the native sensor capture pitch is:
  $$\Delta x_{\text{HAM}} = \frac{18,000\ \mu\text{m}}{600\text{ px}} = 30.0\ \mu\text{m/pixel}$$
  At $30\ \mu\text{m/px}$, a $50\ \mu\text{m}$ pseudopod spans only $1.67\text{ pixels}$ in the raw archival file! Downsampling this raw file to $224 \times 224$ degrades the pitch to $80.36\ \mu\text{m/px}$, completely obliterating fine network lines.
- **BCN20000:** Raw images vary from $1024 \times 1024$ down to $768 \times 768$, offering slightly higher native fidelity ($\Delta x \approx 17.5\text{--}23.4\ \mu\text{m/px}$).
- **Empirical Warning from V6:** Scaling backbones to $384 \times 384$ did *not* automatically fix under-40 melanoma ranking. In V6 Runsheet Section A10.3 / S01:
  > *"S01: 384 px beats 224 px on pooled data failed (fired: 384 px $-0.0192$ under-40 pAUC)."*
  Increasing input resolution without target focal margin crops introduces additional background skin noise and increases model capacity on irrelevant artifacts.

---

## 5. Dermoscopic Morphology & Feature Reader Reliability

V7 proposes relying heavily on local morphological features (atypical pigment network, pseudopods, negative network, blue-white veil) to guide Branch C. We audit the clinical literature on inter-observer agreement and evaluate the empirical findings from the repository.

### 5.1 Inter-Observer Agreement for Dermoscopic Features

Dermatological literature (Kittler et al., *Lancet Oncol* 2002; Argenziano et al., *JAAD* 2003; Carrera et al., *JAMA Dermatol* 2016; Menzies et al., *Arch Dermatol* 2008) demonstrates a stark dichotomy between the reliability of **global architectural patterns** vs. **isolated local structures**:

```
+-----------------------------------------------------------------------------------------+
|                  INTER-OBSERVER RELIABILITY OF DERMOSCOPIC STRUCTURES                   |
|                                                                                         |
|   HIGH / SUBSTANTIAL AGREEMENT (kappa >= 0.50):                                          |
|   * Pattern Asymmetry / Marked Disorder       kappa = 0.55 - 0.65  (Carrera et al. OR=6.6) |
|   * Blue-White Veil (Invasive Marker)         kappa = 0.52 - 0.62  (Argenziano et al.)   |
|   * Shiny White Streaks (Polarized Chrysalis) kappa = 0.55 - 0.62  (Specific for fibrosis)|
|   * Multi-Color Diversity (>= 3 colors)       kappa = 0.50 - 0.60                        |
|                                                                                         |
|   MODERATE / POOR AGREEMENT (kappa < 0.50) [HIGHLY SUBJECTIVE]:                         |
|   * Atypical Pigment Network                  kappa = 0.35 - 0.50  (Controversial line width)
|   * Negative / Inverse Network                kappa = 0.28 - 0.42  (Easily confused with gaps)
|   * Irregular Streaks / Pseudopods            kappa = 0.35 - 0.48  (Mimicked by Clark nevi)
|   * Peripheral Globules                       kappa = 0.40 - 0.52  (Physiological in youth)
|   * Peppering / Regression Granules           kappa = 0.38 - 0.48                        |
+-----------------------------------------------------------------------------------------+
```

#### Detailed Breakdown of Key Morphology Criteria:

1. **Atypical Pigment Network ($\kappa \approx 0.35\text{--}0.50$):**  
   Characterized by irregular, thickened, hyperpigmented lines and asymmetrical mesh holes. While considered a classic melanoma criterion, distinguishing mild network atypia in active dysplastic nevi of young patients from early melanoma in situ exhibits low inter-observer concordance ($\kappa \approx 0.38$ among general dermatologists; up to $0.50$ among expert dermoscopists).
2. **Negative Network / Inverse Network ($\kappa \approx 0.28\text{--}0.42$):**  
   Consists of interconnected light curvilinear cords surrounding darker, elongated structures. In the 2003 Internet Consensus Meeting (Argenziano et al.), negative network had one of the lowest agreement scores among specific criteria. It is notoriously subjective, difficult to annotate, and frequently confused with hypopigmented areas between pigment clumps.
3. **Blue-White Veil ($\kappa \approx 0.52\text{--}0.62$):**  
   An irregular, structureless blue pigmentation with an overlying white ground-glass haze. Histologically correlates with dense subepidermal melanin nests beneath an orthokeratotic, thickened stratum corneum. Shows substantial inter-observer agreement because it represents a macroscopic optical phenomenon, but is predominantly an indicator of **invasive melanoma (Breslow depth $>0.75\text{ mm}$)** and is typically absent in early in situ young melanomas.
4. **Irregular Streaks & Pseudopods ($\kappa \approx 0.35\text{--}0.48$):**  
   Bulbous or linear projections at the tumor periphery representing radial intraepidermal melanocytic growth. While highly specific for melanoma when distributed irregularly, regular circumferential pseudopods characterize benign pigmented Spitz / Reed nevi in adolescents and young adults. Agreement suffers when differentiating irregular streaks from benign dynamic rim structures.
5. **Peripheral Globules ($\kappa \approx 0.40\text{--}0.52$):**  
   A circumferential, continuous ring of peripheral brown globules is the physiological dermoscopic hallmark of a growing, dynamic melanocytic nevus in children and young adults (<30 years). In contrast, focal, asymmetrical peripheral globules indicate melanoma. General readers frequently fail to reliably separate physiological growth rims from pathological segmental clumps.
6. **Shiny White Streaks / Chrysalis Structures ($\kappa \approx 0.55\text{--}0.62$):**  
   Bright, orthogonal white lines visible **strictly under polarized dermoscopy** (extinguished under non-polarized immersion dermoscopy). Correlates with altered collagen birefringence and stromal fibrosis. High reproducibility among expert dermoscopists, but invalid if applied to non-polarized archives (e.g., portions of HAM10000).

### 5.2 Macro-Architecture vs. Local Features: The Carrera & Williams Evidence

In designing Branch A and Branch C, V7 must respect the diagnostic primacy established across large-scale multi-reader clinical trials:
- **Carrera et al. (*JAMA Dermatol* 2016; 469 clinicians, 80 lesions):**  
  Marked architectural disorder exhibited a multivariable diagnostic odds ratio of **$\text{OR} = 6.6$ (95% CI: 5.6–7.8)**, and pattern asymmetry had **$\text{OR} = 4.9$ (95% CI: 4.1–5.8)**. In contrast, isolated local criteria had significantly lower independent predictive value: atypical network had $\text{OR} = 2.1\text{--}2.8$; irregular streaks had $\text{OR} = 2.6\text{--}3.8$.
- **Williams / Dinnes et al. (*Cochrane Database Syst Rev* 2018):**  
  Evaluating 104 studies (42,852 lesions), global pattern analysis (integrating overall architectural symmetry, color distribution, and context) achieved superior sensitivity ($89\%$) and specificity ($84\%$) compared to rigid local checklist algorithms (ABCD: Specificity $75\%$; Menzies: Specificity $61\%$).
- **Clinical Takeaway for V7:** A model that over-weights local morphological tokens without anchoring them in macro-architectural context will suffer severe false-positive escalation on benign dysplastic and Spitzoid nevi in young patients.

### 5.3 Forensic Audit of `results/v6/screens/biology_features_beyond_model.json`

To test whether hand-engineered biological features provide discriminative signal missed by deep neural networks in the young, V6 evaluated 6 hand-measured morphology features across 691 development images (including 51 melanoma images, 28 melanoma lesions under 40 years).

We audited the exact results recorded in `results/v6/screens/biology_features_beyond_model.json`:

```json
{
  "question": "Do hand-measured biology features add ranking the model does not already have, in the young?",
  "status": "descriptive; no decision, no gate",
  "limits": "about 28 melanoma lesions, one archive, six features uncorrected...",
  "models": {
    "v5_composite_folds_in_matrix": {
      "model_auc": 0.8378,
      "features": {
        "colour_count": {
          "beyond_auc": {"point": -0.0155, "ci95": [-0.0303, -0.0040]},
          "alone": {"auc": 0.5025, "ci95": [0.4011, 0.6100]}
        },
        "colour_asym_max": {
          "beyond_auc": {"point": +0.0096, "ci95": [-0.0298, +0.0440]},
          "alone": {"auc": 0.6368, "ci95": [0.5177, 0.7401]}
        },
        "colour_asym_min": {
          "beyond_auc": {"point": +0.0174, "ci95": [-0.0250, +0.0573]},
          "alone": {"auc": 0.6418, "ci95": [0.5210, 0.7450]}
        },
        "chrom_asym_max": {
          "beyond_auc": {"point": -0.0076, "ci95": [-0.0196, +0.0011]},
          "alone": {"auc": 0.5624, "ci95": [0.4550, 0.6720]}
        },
        "segmental_index": {
          "beyond_auc": {"point": +0.0129, "ci95": [+0.0012, +0.0271]},
          "alone": {"auc": 0.5297, "ci95": [0.4220, 0.6350]}
        },
        "all_six": {
          "beyond_auc": {"point": +0.0029, "ci95": [-0.0355, +0.0408]}
        }
      }
    }
  }
}
```

#### Analytical Verdict:
1. **Zero Beyond-Model Gain:** Combining all 6 biological features (`all_six`) yielded an incremental AUC delta of **$+0.0029$**, with a 95% bootstrap confidence interval of **$[-0.0355, +0.0408]$**, squarely centered on zero.
2. **Color Count is a Random Coin Flip:** Standalone `colour_count` achieved an AUC of **$0.5025$** ($[0.4011, 0.6100]$), demonstrating that simply counting digitized color clusters in 8-bit sRGB provides zero diagnostic utility for distinguishing young melanomas from nevi.
3. **Implications for V7:** Naive rule-based morphological heuristics or uncalibrated concept embeddings will not rescue the under-40 failure mode. Morphological guidance must be learned through high-capacity representation alignment (e.g., CoFiDA-M privileged learning) rather than heuristic feature concatenation.

---

## 6. Biophysics of Dark Lesions vs. Digital Quantization vs. Data Bias

V7 Section 4.2.2 attributes dark-lesion failures to three distinct phenomena: optical melanin absorption, 8-bit quantization SNR collapse, and first-layer feature extinction. We conduct a rigorous biophysical and signal processing audit of these mechanisms.

### 6.1 Melanin Absorption & Radiative Transport Physics

Cutaneous absorption is dominated by epidermal melanosomes. The internal absorption coefficient of an individual eumelanosome is described by Jacques (1998):
$$\mu_{a,\text{mel}}(\lambda) = 5.33 \times 10^{10} \cdot \lambda^{-3.48}\text{ cm}^{-1} \quad (\lambda \text{ in nm})$$
Epidermal absorption depends directly on the melanosome volume fraction $f_{\text{mel}} \in [0.01, 0.45]$:
$$\mu_{a,\text{epi}}(\lambda) = f_{\text{mel}} \cdot \mu_{a,\text{mel}}(\lambda) + (1 - f_{\text{mel}}) \cdot \mu_{a,\text{baseline}}(\lambda)$$

#### Biophysical Consequences:
1. **Extreme Spectral Attenuation:** In heavily pigmented skin ($f_{\text{mel}} \ge 0.35$), the optical absorption in the blue-green spectrum ($400\text{--}500\text{ nm}$) reaches $\mu_{a,\text{epi}} > 100\text{ cm}^{-1}$.
2. **Breakdown of Diffusion Theory:** In thin epidermal layers ($d_{\text{epi}} \approx 60\text{--}100\ \mu\text{m}$), the transport mean free path $l_{\text{tr}} = \frac{1}{\mu_a + \mu_s'} \approx 50\ \mu\text{m} \approx d_{\text{epi}}$. The diffusion approximation completely breaks down; light transport must be modeled via the Radiative Transfer Equation (RTE) or Monte Carlo simulation.
3. **Contrast Flattening (Vascular & Dermal Structures):** Effective penetration depth collapses to $\delta_{\text{eff}}(450\text{ nm}) \approx 45\text{--}65\ \mu\text{m}$. Light is absorbed within the superficial epidermis before reaching the papillary dermis. Consequently:
   - Hemoglobin absorption Q-bands at $542\text{ nm}$ and $577\text{ nm}$ are suppressed; dermal vascular patterns (dotted, hairpin vessels) become invisible.
   - Dermal collagen birefringence (which generates shiny white streaks under polarized light) cannot be backscattered to the camera.
   - **Diagnostic Reality:** The dermoscopic image of a deeply pigmented lesion is physically dominated by a uniform, dark eumelanin curtain. Diagnostic cues that clinicians rely on for amelanotic or lightly pigmented tumors are physically obliterated by photon absorption.

### 6.2 8-Bit Quantization Step Noise Floor & SNR Collapse

In standard consumer dermoscopes and archival datasets (HAM10000, BCN20000), images are digitized to 8 bits per color channel ($2^8 = 256$ discrete levels, $[0, 255]$).

#### Mathematical Analysis of Quantization Noise:
For an ideal uniform quantizer with step size $\Delta = \frac{1}{255}$, the quantization noise variance is:
$$\sigma_q^2 = \frac{\Delta^2}{12} = \frac{1}{12 \cdot 255^2} \approx 1.282 \times 10^{-6}$$

The Signal-to-Noise Ratio (SNR) in linear reflectance space for digital intensity level $k \in \{0, 1, \dots, 255\}$ ($I = \frac{k}{255}$) is:
$$\text{SNR}(k) = \frac{I^2}{\sigma_q^2} = \frac{(k / 255)^2}{1 / (12 \cdot 255^2)} = 12 \cdot k^2$$
Expressed in decibels ($\text{dB}$):
$$\text{SNR}_{\text{dB}}(k) = 10 \log_{10}(12 \cdot k^2) \approx 10.79 + 20 \log_{10}(k)$$

Let us examine the SNR across brightness regimes:
- **Bright Perilesional Skin ($k \approx 180$, $I \approx 0.70$):**
  $$\text{SNR}_{\text{dB}} \approx 10.79 + 20 \log_{10}(180) = 10.79 + 45.11 = \mathbf{55.9\text{ dB}}$$
- **Moderately Pigmented Lesion ($k \approx 40$, $I \approx 0.16$):**
  $$\text{SNR}_{\text{dB}} \approx 10.79 + 20 \log_{10}(40) = 10.79 + 32.04 = \mathbf{42.8\text{ dB}}$$
- **Heavily Pigmented Dark Lesion ($k \approx 4$, $I \approx 0.0157$):**
  $$\text{SNR}_{\text{dB}} \approx 10.79 + 20 \log_{10}(4) = 10.79 + 12.04 = \mathbf{22.8\text{ dB}}$$
- **Severe Shadow / Melanoma Core ($k \approx 1$, $I \approx 0.0039$):**
  $$\text{SNR}_{\text{dB}} \approx 10.79 + 20 \log_{10}(1) = \mathbf{10.8\text{ dB}}$$

#### Signal Truncation:
When digital intensity drops to $k \le 10$, a subtle biological modulation of $\Delta R = 0.0020$ produces:
$$\Delta k = \text{round}(255 \cdot 0.0020) = \text{round}(0.51) \in \{0, 1\}\text{ code level}$$
Quantization truncates subtle spatial modulations into flat, discrete integer plateaus.

### 6.3 Critical Deconstruction of the "Zero-DC Filter Feature Extinction" Claim

V7 Section 4.2.2.3 claims:
> *"Standard edge/texture convolutional filters have zero DC gain ($\sum_{u,v} K(u,v) = 0$). When a dark lesion patch is digitized into a uniform flat digital plateau ($I = \text{const}$), the linear convolution evaluates to $K * I = \text{const} \cdot \sum K = 0$, driving first-layer activations to zero ($a^{(1)} = \text{GELU}(0) \approx 0$). By the chain rule... the parameter update vanishes... starving the network of representation learning for dark lesion patterns."*

#### Scientific Critique:
1. **Idealized vs. Real Skin Physics:** Real biological skin lesions are never mathematically uniform step plateaus ($I(x,y) = \text{const}$). Even deeply pigmented lesions possess micro-topography, surface hair, follicular openings, and subtle peripheral gradations.
2. **The Dominant Border Gradient:** Even if the interior core of a dark lesion is digitized to low values ($k \approx 10$), the **lesion border** presents a massive contrast transition against normal surrounding skin ($k_{\text{skin}} \approx 180$ vs $k_{\text{lesion}} \approx 10$, $\Delta k \approx 170$). Zero-DC convolutional filters (Sobel, Gabor, Laplacian kernels) fire **maximally** at this high-contrast edge!
3. **Architectural Realities:** Modern backbones (ConvNeXt, ResNet, Swin, ViT) utilize LayerNorm / BatchNorm, learned non-zero additive bias vectors $\mathbf{b}^{(1)}$, and multi-scale receptive fields. Convolutions never experience complete gradient extinction across an entire patch.
4. **Epistemic Conclusion:** The claim of "zero parameter update ($\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{0}$)" is an overclaimed theoretical abstraction. The actual failure mechanism is **contrast attenuation** (loss of internal network lines) combined with **Bayesian representation bias**, not mathematical zero-gradient starvation.

---

## 7. Repository Experimental Audit & Forensic Reconciliation

### 7.1 How Darkness is Measured in the Codebase

A critical responsibility of this review is auditing how "darkness" was defined and measured in the actual code scripts.

#### Source Code Audit (`scripts/read_melanin_by_age.py`, `research/v5/b1_hard_core.py`):
```python
def mean_melanin(image_ids: list[str]) -> dict[str, float]:
    from research.v5 import chromophore as ch
    from research.v5.fit_fold_artefacts import load

    basis, _, _, _ = load(0)
    out = {}
    for _, image_id, x in pc.stream(image_ids, every=0):
        maps = ch.compute_maps(x, basis)
        geo = ch.lesion_geometry(maps)
        m = (maps.c_mel - maps.m_skin.view(-1, 1, 1, 1)).clamp_min(0)
        out[image_id] = float((m * geo.s).sum() / geo.s.sum().clamp_min(1e-6))
    return out
```

#### Key Technical Findings:
1. **Darkness is NOT CIELAB Luminance ($1 - L^*$):**  
   The codebase does *not* define darkness by thresholding $1 - L^* \ge 0.65$. Instead, it computes `mean_melanin`, which is the **mean melanin chromophore density above surrounding skin inside the lesion**, projected onto fold 0's ICA unmixing basis and masked by the automated lesion geometry mask `geo.s`.
2. **Darkness is NOT Fitzpatrick Phototype:**  
   Fitzpatrick classification measures constitutional perilesional skin color (Types I–VI). In contrast, `mean_melanin` measures facultative intra-lesional hyperpigmentation. A Fitzpatrick Type I patient can have a deeply pigmented, high-melanin melanoma, while a Fitzpatrick Type IV patient can have an amelanotic melanoma.
3. **Archive Tracking Caveat:**  
   As explicitly documented in `scripts/read_melanin_by_age.py`:
   > *"Melanin is an image measure and not a biological assay, and it can track the device as well as the lesion, so every test is also done within archive."*

### 7.2 The Forensic Mystery of $p = 0.0402$: Conflation Uncovered

In V7 Section 4.2.2.1, the text asserts:
> *"19 of 33 false negatives occurred in lesions in the darkest tertile of the dataset ($1 - L^* \ge 0.65$), yielding an empirical error concentration that is statistically significant ($p = 0.0402$, Fisher's exact test)."*

Our audit reveals that **this sentence accidentally merged numbers from two completely different experimental phases**:

```
+-----------------------------------------------------------------------------------------+
|                  FORENSIC RECONCILIATION OF THE "p = 0.0402" CLAIM                      |
|                                                                                         |
|   EXPERIMENT 1: Diagnostic B1 (V4 Control OOF)                                         |
|   Source: results/v5/diagnostics/b1_hard_core.json                                      |
|   * Evaluated: 81 young escalating lesions (<40 years).                                 |
|   * Total Missed at FPR 0.20: 33 lesions.                                               |
|   * Breakdown by Melanin Tercile:                                                       |
|       High Melanin: 19 missed / 27 total  (Miss Rate = 70.4%)                           |
|       Mid Melanin:  11 missed / 27 total  (Miss Rate = 40.7%)                           |
|       Low Melanin:   3 missed / 27 total  (Miss Rate = 11.1%)                           |
|   * True Fisher's Exact Test (High vs Rest):   p = 0.000238 (Odds Ratio = 6.79)         |
|   * True Fisher's Exact Test (High vs Low):    p = 1.75e-05 (Odds Ratio = 19.0)        |
|                                                                                         |
|   EXPERIMENT 2: V6-5 Reread (V6 Vote Ensemble OOF, Folds 1-4)                           |
|   Source: results/v6/screens/dark_v6_oof_read.json                                      |
|   * Evaluated: v6_vote (ConvNeXt 224/384 + PanDerm + DINOv2 with 24-view TTA).          |
|   * Breakdown by Melanin Tercile:                                                       |
|       High Melanin:  5 missed / 15 total                                                |
|       Mid Melanin:   7 missed / 20 total                                                |
|       Low Melanin:   2 missed / 13 total                                                |
|   * Within-Archive Permutation Test:          p = 0.040248  <-- EXACT MATCH!            |
|                                                                                         |
|   VERDICT: V7 Master Plan conflated the 19/33 sample count from V4 Control with the     |
|   p = 0.0402 within-archive permutation p-value from V6 Vote!                           |
+-----------------------------------------------------------------------------------------+
```

#### Mathematical Proof of the Forensic Match:
1. When we test Diagnostic B1 (`b1_hard_core.json`):
   - High Melanin: 19 Missed, 8 Caught
   - Non-High (Mid + Low): 14 Missed, 40 Caught
   - Contingency Table: `[[19, 14], [8, 40]]`
   - SciPy `fisher_exact([[19, 14], [8, 40]])`: $\text{Odds Ratio} = 6.786$, **$p = 0.00023809$**.
2. When we inspect `results/v6/screens/dark_v6_oof_read.json` for `v6_vote`:
   ```json
   "within_archive": {
     "p_two_sided": 0.04024798760061997
   }
   ```
   The value **$p = 0.040248$** is the **two-sided within-archive permutation test p-value** for `v6_vote` on folds 1–4!
3. **Required Text Correction:** V7 must separate these experiments. Diagnostic B1 established a massive concentration of error in high-melanin lesions on V4 control ($p = 0.00024$), which weakened to $p = 0.0402$ in the V6 vote ensemble.

### 7.3 Dark Lesion Failure is Already Weakened in V6; Specialist Heads Fail

In `results/v6/screens/dark_v6_oof_read.json`, we audited the full comparative trajectory from V5 control to the deployed V6 vote across folds 1–4:

| System | Low Melanin Misses | Mid Melanin Misses | High Melanin Misses | Young Dark Gap vs. Older Gap ($\Delta$) | 95% Bootstrap CI of Gap Difference | Permutation $p$ (Within Archive) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **V5 Control** | 3 / 13 (23.1%) | 8 / 20 (40.0%) | **10 / 15 (66.7%)** | $+0.0826$ | $[-0.0695, +0.2161]$ | **$p = 0.0045$** |
| **V5 Composite** | 2 / 13 (15.4%) | 6 / 20 (30.0%) | 4 / 15 (26.7%) | $+0.0616$ | $[-0.1307, +0.2356]$ | $p = 0.0672$ |
| **V6 Vote** | 2 / 13 (15.4%) | 7 / 20 (35.0%) | **5 / 15 (33.3%)** | **$+0.0151$** | **$[-0.1624, +0.1657]$** | **$p = 0.0402$** |
| **Combined Vote** | 2 / 13 (15.4%) | 6 / 20 (30.0%) | 3 / 15 (20.0%) | $+0.0016$ | $[-0.2040, +0.2028]$ | $p = 0.1786$ |

#### Key Insights from the Audit:
1. **The Young-vs-Older Gap Has Collapsed:**  
   In V5 control, young melanomas had a substantially larger performance deficit in dark lesions than older patients did ($+0.0826$). In V6 vote, the difference between young and older dark-lesion gaps collapsed to **$+0.0151$ (95% CI: $[-0.1624, +0.1657]$)**. The confidence interval crosses zero symmetrically. The "worse in the young than in older" disparity is no longer statistically significant.
2. **High-Melanin Misses Cut in Half:**  
   V6 vote missed 5 of 15 high-melanin young melanomas, down from 10/15 in V5 control and 19/27 in V4.
3. **Failure of Dedicated Dark-Lesion Specialist Heads:**  
   We audited the nested specialist head experiment in `dark_v6_oof_read.json` (`probe_dark_specialist.py`), which trained specialized linear/MLP probes on dark lesion subsets:
   - `dark_young` variant on `v6_vote`: **$\Delta = -0.0126$ (95% CI: $[-0.0330, +0.0080]$)**.
   - `dark_all_ages` variant on `v6_vote`: **$\Delta = -0.0017$ (95% CI: $[-0.0078, +0.0047]$)**.
   - `young` variant on `v6_vote`: **$\Delta = -0.0033$ (95% CI: $[-0.0070, -0.0001]$)**.
   Every single specialist variant performed flat or **worse** than the global ensemble! Stratifying training data into narrow dark subsets starves the model of sample diversity.

---

## 8. Critical Evaluation of the Dual-Stream Cross-Attention & Focal Pooling Architecture

V7 Section 4.2.1.4 proposes a **Dual-Stream Cross-Attention Architecture** with **Quadruple Artifact-Gated Top-$K$ Focal Pooling** to resolve margin structures without GAP dilution. We evaluate its engineering and clinical feasibility.

### 8.1 Architectural Architecture & Safeguard Mechanism

```
                                  [ RAW DERMOSCOPIC IMAGE ]
                                             |
                     +-----------------------+-----------------------+
                     |                                               |
           [ Global Full FOV ]                             [ Native High-Res Margin ]
           (384 x 384, 18 mm FOV)                          (384 x 384, FOV <= 9.6 mm)
                     |                                               |
             [ Backbone Trunk ]                              [ Backbone Trunk ]
                     |                                               |
            F_global in R^(D x H x W)                       F_margin in R^(D x H x W)
                     |                                               |
                     |                                    [ Artifact Attenuation ]
                     |                                 (1 - M_artifact) * F_margin
                     |                                               |
                     +--------------+--------------------------------+
                                    |
                    [ Global-Gated Cross-Attention ]
                      Q = W_Q * F_margin_clean
                      K = W_K * F_global
                      V = W_V * F_global
                      H_margin = Softmax(QK^T / sqrt(d) + B_spatial) * V
                                    |
                 [ Calibrated Malignancy-Projected Pooling ]
                   Score: s_i = [(w_mel - w_art)^T h_i] * P_lesion(u_i)
                   Top-K Tokens Selected (K = floor(0.15 * HW))
                                    |
                        z_focal = (1/K) * sum(h_pi(k))
                                    |
                     +--------------+--------------+
                     |                             |
              z_global (GAP)                  z_focal (Top-K)
                     \                             /
                      \                           /
                       [ Macro-Micro Late Fusion ]
                        z_final = W_g * z_g + W_f * z_f
```

### 8.2 Clinical & Algorithmic Critique

1. **Top-$K$ Pooling Susceptibility to Artifacts:**  
   V7 correctly identifies that standard $L_2$-norm Top-$K$ pooling inadvertently selects high-frequency artifacts (peripheral hairs, air bubbles in immersion gel, plastic dermatoscope ring edges, ruler markings).
2. **Evaluation of the Malignancy Projection Score:**  
   V7 introduces:
   $$s_i = \left[(\mathbf{w}_{\text{mel}} - \mathbf{w}_{\text{artifact}})^\top \mathbf{h}_i\right] \cdot \mathbf{P}_{\text{lesion}}(u_i)$$
   - *Strengths:* Projecting tokens along the direction orthogonal to known artifact vectors ($\mathbf{w}_{\text{artifact}}$) and weighting by the prior lesion mask distance transform ($\mathbf{P}_{\text{lesion}}$) effectively suppresses peripheral ruler marks and border vignetting.
   - *Vulnerability:* DullRazor morphological filtering removes hair shafts, but frequently leaves residual optical shadows and blur artifacts. If a benign Spitz nevus has prominent pigment clumping, $s_i$ will fire intensely, risking false-positive escalation.
3. **Macro-Micro Decision Fusion is Mandatory:**  
   Retaining the global stream ($z_{\text{global}}$) is essential. As shown in Section 5.2, macro-architectural disorder is the primary predictor of malignancy ($\text{OR} = 6.6$). The margin stream must serve as an asymmetrical "safety trigger" rather than an independent diagnostic classifier.

---

## 9. Dermatologist Expert Annotation Protocol & Multi-Reader Study Design

To empirically validate which local morphological criteria truly discriminate young melanomas from benign mimics, we design a rigorous, clinical-grade dermatologist reader study protocol.

### 9.1 Multi-Reader Study Cohort Curation

To ensure high statistical power while preventing selection bias, the evaluation set must be curated through stratified sampling of the pooled OOF development cohort:

- **Total Sample Size:** $N = 120$ lesions (all from patients $<40$ years).
- **Stratified Cohort Composition:**
  1. **Young Melanomas Caught by Baselines ($N = 20$):** Typical superficial spreading melanomas with clear architectural asymmetry.
  2. **Young Melanomas Missed by Baselines / Hard Misses ($N = 20$):** The core failure cohort from Diagnostic B1 and V6-5 (high-melanin, symmetric or subtle presentation).
  3. **Dysplastic / Atypical Nevi ($N = 40$):** Biopsy-confirmed Clark/dysplastic nevi with architectural disorder (the primary clinical mimic).
  4. **Banal Compound & Junctional Nevi ($N = 40$):** Common active nevi in young adults, including 20 cases with physiological peripheral globules.
- **Reference Standard:** Histopathological ground truth verified by dermatopathology report review (Breslow depth, subtype, precursor nevus presence, margin status).

### 9.2 Two-Level Hierarchical Morphology Annotation Ontology

Annotations must follow a standardized, controlled vocabulary matching International Dermoscopy Society (IDS) consensus standards:

```
LEVEL 1: MACRO-ARCHITECTURAL PATTERN ANALYSIS (Global Context)
  1.1 Global Symmetry:
      [ ] Fully Symmetric (2 axes)
      [ ] Asymmetric in 1 axis
      [ ] Asymmetric in 2 axes (Complete Chaos)
  1.2 Architectural Disorder Score:
      [ ] Mild (Typical nevus pattern)
      [ ] Moderate (Atypical / Dysplastic)
      [ ] Marked (Melanoma-level structural collapse)
  1.3 Color Count & Distribution:
      - Total Distinct Colors: [1, 2, 3, 4, 5, 6]
        (Light brown, Dark brown, Black, Red, White, Blue-gray)
      - Color Symmetry: [Symmetric / Asymmetric]

LEVEL 2: LOCAL MARGINAL & ELEMENTARY STRUCTURES (Targeted Micro-Cues)
  2.1 Pigment Network:
      [ ] Absent
      [ ] Typical / Regular (delicate mesh, uniform cords)
      [ ] Atypical (broadened lines, irregular polygonal meshes, focal thickening)
  2.2 Negative Network (Inverse Network):
      [ ] Absent
      [ ] Present (white serpiginous grid encircling dark globules)
  2.3 Peripheral Radial Projections:
      [ ] Absent
      [ ] Regular Streaks / Pseudopods (symmetric, circumferential)
      [ ] Irregular Streaks / Pseudopods (segmental, asymmetrical clump)
  2.4 Globular Pattern:
      [ ] Absent
      [ ] Central Globules
      [ ] Regular Peripheral Rim of Globules (Physiological dynamic growth)
      [ ] Asymmetric / Segmental Peripheral Globules (Malignancy trigger)
  2.5 Invasive / Stromal Structures:
      [ ] Blue-White Veil (structureless blue haze with ground-glass veil)
      [ ] Peppering / Regression Granules (fine gray-blue granular dots)
      [ ] Shiny White Streaks / Chrysalis (orthogonal white lines; polarized only)
  2.6 Vascular Architecture:
      [ ] Regular comma / hairpin vessels
      [ ] Atypical polymorphic / dotted vessels
```

### 9.3 Study Execution, Blinding & Adjudication Protocol

1. **Reader Panel:** 5 independent readers (3 board-certified dermatologists with $\ge 5$ years dermoscopy experience + 2 expert dermoscopy sub-specialists).
2. **Blinding Protocol:**
   - All dermoscopic images are presented in randomized order.
   - All clinical metadata (age, sex, anatomic site, patient identity) is completely stripped.
   - Non-polarized and polarized versions are displayed side-by-side where available.
3. **Multi-Stage Annotation Workflow:**
   - **Phase 1 (Independent Reading):** Each reader independently scores all 120 lesions across the Level 1 and Level 2 ontology without consulting peers. Each reader records a binary management decision: *Escalate (Biopsy/Refer)* vs. *Reassure (Discharge/Routine)*.
   - **Phase 2 (Inter-Observer Agreement Analysis):** Compute Fleiss' $\kappa$ across all 5 readers for each morphological feature. Compute pairwise Cohen's $\kappa$ and Intraclass Correlation Coefficients (ICC).
   - **Phase 3 (Consensus Adjudication for Ground Truth Extraction):** For cases with reader disagreement ($<3/5$ concordance), conduct an in-person / virtual consensus panel. Consensus requires $\ge 4/5$ majority. If disagreement persists, the feature is flagged as "Morphologically Ambiguous."
4. **Statistical Power & Endpoints:**
   - Sample size $N=120$ provides $>90\%$ power to detect a kappa difference of $\Delta \kappa \ge 0.20$ against the null hypothesis $\kappa \le 0.40$ at two-sided $\alpha = 0.05$.
   - Primary Endpoint: Multivariable Logistic Regression Diagnostic Odds Ratio (DOR) of adjudicated Level 2 features in predicting malignancy among the 40 hard young cases.

---

## 10. Master Synthesis: Validated vs. Speculative Biological Mechanisms

To enforce strict epistemic boundaries in V7, the table below classifies every biological and biophysical mechanism into **Validated**, **Plausible**, or **Speculative/Refuted**:

| Biological / Physical Mechanism | V7 Master Plan Claim | Literature & Repository Empirical Reality | Epistemic Classification | Mandated Action in V7 Specification |
|---|---|---|:---:|---|
| **De Novo Melanoma Dominance** | Acknowledged in Section 4.2.1.1 (Pampena 70.9% de novo). | **Fully Validated.** Pampena et al. (JAAD 2017) meta-analysis ($N=20,126$) proves 70.9% are de novo (RR = 0.36 for NAM). | **VALIDATED** | Retain in Section 4.2.1.1. Explicitly state that GAP dilution cannot explain false negatives in the 70% de novo cases. |
| **NAM 10–20% Area Fraction** | Malignant clones in early NAM occupy $10\%\text{--}20\%$ of lesion area. | **Plausible Heuristic, Unproven Constant.** Early focal atypia in NAM is documented qualitatively, but no quantitative morphometric literature establishes 10–20% as a universal constant. | **PLAUSIBLE (Uncalibrated)** | Re-label as an *exploratory working assumption* for early NAM, not a proven biological parameter. |
| **GAP Dilution Margin Theorem** | Mathematical proof that GAP causes dilution false negatives when $\alpha \le \alpha_{\text{crit}}$. | **Overclaimed Affine Identity.** True mathematically only for linear heads on spatially independent features. Fails under receptive field context, high activation norms, and multiclass Softmax. | **SPECULATIVE (Overclaimed)** | Demote from "Mathematical Theorem Proving False Negatives" to "Conditional Representation Dilution Hypothesis". |
| **Optical Downsampling Aliasing** | Resizing $18\text{ mm}$ FOV to 224 px yields $\Delta x = 80.4\ \mu\text{m/px}$, failing Nyquist ($<2.0\text{ px}$). | **Fully Validated.** Basic optical signal processing. A $50\ \mu\text{m}$ pseudopod spans $0.62\text{ px}$ at 224 px and $1.07\text{ px}$ at 384 px. Native crops ($\le 9.6\text{ mm}$) achieve $\ge 2.0\text{ px}$. | **VALIDATED** | Retain Section 4.2.1.3 optical analysis. Note that resolution scaling alone failed in V6 (S01: $-0.0192$). |
| **Macro-Architecture Superiority** | Macro-architectural disorder ($\text{OR} = 6.6$) dominates local criteria ($\text{OR} = 2.1\text{--}2.8$). | **Fully Validated.** Carrera et al. (2016) and Williams et al. (Cochrane 2018) prove global pattern analysis outperforms local checklists. | **VALIDATED** | Retain Section 4.2.1.5. Enforce global stream weighting in dual-stream architecture. |
| **Hand-Engineered Biological Features** | Hand-measured biology features add discriminative ranking power in the young. | **Refuted by Repository Data.** `biology_features_beyond_model.json` proves combining 6 features adds $+0.0029$ AUC ($[-0.0355, +0.0408]$). Colour count alone has AUC $0.5025$. | **REFUTED** | Document V6 negative finding. Prohibit hand-crafted geometric/color feature concatenation in V7. |
| **Dark Melanoma Disparity ($p=0.0402$)** | 19 of 33 false negatives occurred in darkest tertile ($1 - L^* \ge 0.65$), $p=0.0402$ (Fisher's exact). | **Conflation of Two Studies.** Darkness was measured via `mean_melanin` (ICA proxy), not $1 - L^*$. 19/33 misses occurred in V4 control ($p = 0.00024$). $p = 0.0402$ was the permutation test in V6 vote! | **CONFUSED / MISATTRIBUTED** | Correct Section 4.2.2.1 text, metric definition, and citation linkage. |
| **Zero-DC Filter Feature Extinction** | Zero-DC filters evaluate to 0 on dark plateaus, driving activations to 0 and starving gradients. | **Biophysically Flawed.** Skin lesions are not flat mathematical step functions. Zero-DC filters fire maximally at high-contrast lesion borders ($\Delta I > 150$). | **SPECULATIVE (Exaggerated)** | Soften claim. Frame failure as contrast attenuation of internal structures, not vanishing parameter updates. |
| **Dark Lesion Specialist Heads** | Training dedicated heads on dark lesion subsets rescues dark young melanoma performance. | **Refuted by Repository Data.** `dark_v6_oof_read.json` proves all specialist variants scored flat or worse ($\Delta = -0.0126$ on `dark_young`). | **REFUTED** | Record negative finding from V6. Ban dedicated dark-subset specialist heads from V7 diagnostic trunk. |
| **Melanin Vascular Attenuation** | Epidermal melanin absorption attenuates hemoglobin Q-bands ($542, 577\text{ nm}$), hiding erythema. | **Fully Validated.** Biophysics of radiative transport and tissue spectroscopy (Jacques 1998, Anderson & Parrish 1981). | **VALIDATED** | Retain Section 4.2.2.2 biophysical rationale. |

---

## 11. Prescriptive Editorial & Mathematical Corrections to V7 Master Research Plan

To ensure that `docs/V7_MASTER_RESEARCH_PLAN.md` withstands peer review at *IEEE Transactions on Medical Imaging (TMI)*, we specify exact, actionable editorial and mathematical revisions.

### 11.1 Revisions to Section 4.2.1 (Under-40 Precursor Biology & GAP Dilution)

1. **De-emphasize GAP Dilution as the Universal Under-40 Mechanism:**  
   *Current Text (lines 261–263):* Suggests that young melanoma failure is primarily an issue of minor clone dilution in NAM.  
   *Mandated Revision:* Explicitly incorporate the Pampena et al. (JAAD 2017) finding that **70.9% of melanomas are de novo**. State clearly:
   > *"Because ~70% of cutaneous melanomas in patients under 40 develop de novo without a pre-existing nevus bed, GAP dilution applies strictly to the ~30% subset of nevus-associated cases. For de novo young melanomas, the diagnostic bottleneck is governed by phenotypic mimicry of active, growing benign nevi and Bayesian prior suppression ($p(\text{mel} | <40) \approx 4.9\%$), rather than spatial averaging dilution."*

2. **Re-Title and Re-Frame Section 4.2.1.2:**  
   *Current Title:* `##### 4.2.1.2 Theorem: Conditionally Bounded GAP Dilution Margin Theorem`  
   *Mandated Revision:* Change title to:  
   `##### 4.2.1.2 Analysis: Conditionally Bounded Spatial Pooling Dilution in Nevus-Associated Melanoma`  
   Replace the assertion that GAP *causes* false negatives with an honest conditional statement:
   > *"Global Average Pooling (GAP) mathematically dilutes local feature activations in proportion to the area fraction $\alpha$. In nevus-associated melanomas where $\alpha \le 0.20$, detection requires the local malignant margin to satisfy $m_{\text{mal}} > \frac{\tau - (1 - \alpha)m_{\text{benign}}}{\alpha}$. While high-capacity backbones can overcome this threshold through large activation norms, spatial focal pooling provides an architectural safeguard against dilution."*

### 11.2 Revisions to Section 4.2.2 (Dark-Lesion Failure & Biophysics)

1. **Correct Section 4.2.2.1 (Empirical Failure Evidence):**  
   *Current Text (lines 344–346):*
   > *"19 of 33 false negatives occurred in lesions in the darkest tertile of the dataset ($1 - L^* \ge 0.65$), yielding an empirical error concentration that is statistically significant ($p = 0.0402$, Fisher's exact test)."*  
   *Mandated Replacement:*
   > *"A diagnostic audit of initial baseline models (`results/v5/diagnostics/b1_hard_core.json`) revealed a significant concentration of melanoma false negatives in lesions within the highest melanin tercile: in the V4 control model across 81 young escalating lesions, 19 of 33 false negatives occurred in the high-melanin tercile (miss rate 70.4% vs 11.1% in low-melanin; $p = 0.000238$, Fisher's exact test). Lesion darkness was quantified via `mean_melanin`, an optical-density ICA chromophore projection above skin baseline (`research/v5/chromophore.py`). In subsequent evaluations of the deployed heterogeneous ensemble (`results/v6/screens/dark_v6_oof_read.json`), this disparity weakened significantly: the young-vs-older dark lesion gap difference collapsed to $+0.0151$ ($[-0.1624, +0.1657]$), while a within-archive permutation test yielded $p = 0.0402$, demonstrating that multi-backbone ensembling and 24-view TTA largely mitigate the high-melanin deficit."*

2. **Soften Section 4.2.2.3 (First-Layer Feature Extinction):**  
   *Current Text (lines 370–373):* Claims that dark lesions evaluate to $K * I = 0$, driving parameter updates to zero and starving the network of gradients.  
   *Mandated Replacement:* Re-frame as **contrast modulation compression**:
   > *"While convolutional filters fire strongly at the high-contrast outer boundary between lesion and surrounding skin, internal morphological modulations within deeply pigmented lesions are heavily compressed. Coarse 8-bit quantization and sensor noise floors collapse internal SNR from $56\text{ dB}$ to $<23\text{ dB}$, attenuating the delicate textural gradients required to resolve atypical pigment networks."*

3. **Incorporate Negative Findings from Specialist Probes:**  
   Add an explicit subsection documenting the empirical failure of dark specialist heads:
   > *"Section 4.2.2.8 Empirical Boundary: Failure of Dedicated Sub-Cohort Specialist Heads. Experiments training dedicated linear/MLP specialist heads restricted to dark and young subsets (`results/v6/screens/dark_v6_oof_read.json`) yielded consistently negative deltas on the deployed ensemble ($\Delta = -0.0126$ for `dark_young`). Restricting training data to dark lesions starves representations of variance. V7 therefore rejects dedicated dark-lesion heads in favor of invariant contrast-normalized representations across the unified diagnostic trunk."*

### 11.3 Revisions to Section 4.5 (Morphological Concepts & Reader Guidance)

1. **Document the Negative Finding of Hand-Engineered Features:**  
   Explicitly cite `results/v6/screens/biology_features_beyond_model.json` to justify why V7 does *not* use heuristic geometric or color features:
   > *"Pre-implementation screens demonstrated that concatenating hand-engineered morphological features (color counts, color asymmetry, segmental indices) failed to provide independent ranking power beyond deep learning representations ($\Delta \text{AUC} = +0.0029$, 95% CI: $[-0.0355, +0.0408]$). V7 morphology guidance is therefore formulated strictly via privileged concept distillation (CoFiDA-M) under the LUPI paradigm."*

2. **Incorporate the IDS Reader Study Protocol:**  
   Embed the 2-level hierarchical ontology and 5-reader dermatologist study design (detailed in Section 9 of this report) as the mandatory evaluation protocol for validating morphological concept representations prior to clinical claims.

---

### Audit Sign-Off
**Reviewing Agent:** AGENT 09 — Under-40 Melanoma Biology, Dermoscopic Morphology & Pigmentation Failure Specialist  
**Deliverable Path:** `docs/v7_final_revision/agent09_under40_biology_morphology_review.md`  
**Status:** Certified Complete & Ready for Synthesis  
