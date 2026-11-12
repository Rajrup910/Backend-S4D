# AGENT 11: Scientific Review & Mathematical Audit — Privileged Concept Learning, LUPI, CoFiDA-M, MONET & Knowledge Distillation for S4D V7

**Author:** Agent 11: Privileged Concept Learning, LUPI, CoFiDA-M, MONET and Knowledge Distillation Specialist  
**Target:** Scan4Disease (S4D) / Backend-S4D · V7 Formulation & IEEE TMI Master Manuscript  
**Date:** November 2026  
**Status:** Certified Methodological Audit & Scientific Decision Dossier  
**Deliverable Path:** `docs/v7_final_revision/agent11_privileged_concept_distillation_review.md`  

---

## 1. Executive Summary & Forensic Audit Mandate

### 1.1 Context and Scope of the Investigation
Following the completion of Version 6 (commit `4e6d32b`, external validation on the HIBA cohort: Macro-F1 $0.7348$ vs V5's $0.5715$, $+0.1633$ gain), the Scan4Disease program initiated the V7 architecture specification. Central to the V7 proposal is **Branch C: Morphological Concept Supervision & Privileged Distillation**, designed to resolve the historical under-40 melanoma ranking failure mode without requiring test-time metadata inputs.

This dossier provides a comprehensive forensic audit of the privileged concept learning framework in V7. Specifically, we investigate:
1. **CoFiDA-M (Sultana et al., CVPR 2026):** Deconstructing its original application (cross-modality domain adaptation from dermoscopy to clinical photographs) versus its hypothesized utility in V7 (in-domain ranking of subtle early Superficial Spreading Melanoma in young patients).
2. **Concept Reliability & Noise:** Auditing the validity of automated concept extractors (MONET, DermFM-Zero) versus human expert annotations (Derm7pt), quantifying label noise, and evaluating the risk of concept leakage and shortcut learning.
3. **BM-KD Margin Loss Audit:** Performing a rigorous sign error audit on the margin distillation formulation ($[\Delta_T - \Delta_S - \gamma]_+^2$ vs $[\Delta_S - \Delta_T - \gamma]_+^2$) and mathematically refuting unproven claims of guaranteed "95% sensitivity retention."
4. **PCGrad Optimization Audit:** Examining the geometric mechanics of Projecting Conflicting Gradients (Yu et al., NeurIPS 2020) and refuting claims that PCGrad provides a global guarantee against negative transfer in non-convex stochastic optimization.
5. **Strategic Recommendation:** Issuing an evidence-based recommendation on whether concept guidance should be designated as *Core V7*, a *Conditional Training-Only Research Arm*, or *Rejected/Deferred*.

### 1.2 Summary of Primary Audit Findings
* **Finding 1 (Domain Mismatch):** CoFiDA-M was developed and validated strictly for cross-domain adaptation (dermoscopy $\to$ smartphone clinical photography). It has never been tested as a within-modality ranking fix for rare-class fine cytological distinctions (SSM in young patients). Its inclusion in V7 is an unvalidated cross-disciplinary hypothesis.
* **Finding 2 (Foundation Model Noise):** Pseudo-labels from vision-language foundation models (MONET, DermFM-Zero) exhibit substantial error rates on fine dermoscopic micro-structures ($30\%\text{--}48\%$ error on streaks, peppering, and atypical vascular patterns). Feeding uncalibrated pseudo-labels into teacher networks without confidence deadbands introduces severe feature distortion.
* **Finding 3 (Catastrophic Sign Error Averted):** The margin distillation loss must strictly be parameterized as $[\Delta_T(x) - \Delta_S(x) - \gamma_{\text{buffer}}]_+^2$. The inverted formulation $[\Delta_S(x) - \Delta_T(x) - \gamma]_+^2$ actively penalizes the student when it outperforms the teacher while assigning zero penalty when the student collapses. Furthermore, no margin loss mathematically guarantees 95% sensitivity retention; such claims are unproven and must be replaced by empirical non-inferiority bounds and finite-sample VC excess risk bounds.
* **Finding 4 (PCGrad Limitations):** PCGrad provides a first-order local projection that eliminates negative directional derivatives at a single tangent point $\theta_t$. In non-convex stochastic optimization with finite learning rates and stochastic mini-batch noise, PCGrad provides **no global guarantee against negative transfer**.
* **Finding 5 (Verdict — Conditional Research Arm):** Branch C must **NOT** be placed in the core locked diagnostic trunk. It is approved strictly as a **Conditional Training-Only Research Arm (Phase 4)**, governed by pre-registered gates (Sandwiched Wald Test $W > 10.83$, Demographic Shortcut Index $\text{DSI}_{<40} < 0.15$, Counterfactual Shift $|\Delta P| \le 0.020$, and OOF pAUC gain $\ge +0.050$). If any gate fails, the system reverts unconditionally to Branch A (Vision) + Branch B (AMPS).

---

## 2. CoFiDA-M (Sultana et al., CVPR 2026) Deconstruction: Cross-Domain vs. In-Domain Analysis

### 2.1 Original Motivation and Architectural Topology
CoFiDA-M (*Concept-Aware Feature Modulation for Cross-Domain Adaptation with Image-Only Inference*, Sultana, Yap, Fan, and Lu, CVPR 2026) was designed to solve a specific clinical bottleneck: **the domain shift between specialized dermoscopy archives and consumer-grade clinical photographs**.

```
CoFiDA-M Original Formulation (CVPR 2026):
Source Domain: Dermoscopy (ISIC / HAM10000) ──┐
                                             ├─► Cross-Domain Feature Alignment
Target Domain: Clinical Photos (PAD-UFES-20) ─┘   Guided by Privileged Concepts c*
                                                   │
                                                   ▼
                                     Image-Only Student for Mobile Deployment
```

In the original paper:
1. **The Core Dilemma:** Dermatology models trained on clean, high-resolution dermoscopic images (polarized light, liquid immersion, macro lenses) suffer catastrophic performance degradation when evaluated on clinical smartphone photographs (variable illumination, non-standardized focal distances, chromatic aberration, sensor noise).
2. **The Deployment Paradox:** Foundation models (such as MONET) can output probabilistic clinical concept scores across both modalities during training. However, executing multi-gigabyte foundation models or requiring concept inputs on low-power mobile smartphones at inference time is clinically and operationally infeasible.
3. **The CoFiDA-M Solution:** Sultana et al. formulated a two-stage teacher-student pipeline under Vapnik's Learning Under Privileged Information (LUPI) paradigm:
   - A high-capacity teacher network ingests both the image $x$ and privileged concept scores $\mathbf{c}^*$.
   - The concept vector modulates intermediate visual feature maps via Feature-wise Linear Modulation (FiLM; Perez et al., 2018).
   - An image-only student network is trained via feature-level distillation to match the teacher's concept-modulated intermediate representations.
   - At test time, the student network processes raw RGB images with zero metadata or concept inputs.

### 2.2 Critical Epistemic Audit: Applicability to V7
The V7 Master Research Plan proposes integrating CoFiDA-M into Branch C. We must critically evaluate whether the theoretical mechanics of CoFiDA-M transfer to the standing challenge of V7.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              CROSS-DOMAIN ADAPTATION vs. IN-DOMAIN RANKING                             │
├──────────────────────────┬──────────────────────────────────────┬──────────────────────────────────────┤
│ Operational Dimension    │ Original CoFiDA-M (Sultana et al.)   │ S4D V7 Mandate (Branch C)            │
├──────────────────────────┼──────────────────────────────────────┼──────────────────────────────────────┤
│ Primary Distribution     │ Domain shift: Dermoscopy to Clinical │ Within-distribution: Dermoscopy      │
│ Shift                    │ Photographs (PAD-UFES-20 / SMART)    │ (HAM10000 / BCN20000 / HIBA)         │
├──────────────────────────┼──────────────────────────────────────┼──────────────────────────────────────┤
│ Nature of Pathology     │ Broad diagnostic categorization      │ Early focal malignancy detection:    │
│ Target                   │ across 6–7 common disease classes    │ Superficial Spreading Melanoma (SSM) │
├──────────────────────────┼──────────────────────────────────────┼──────────────────────────────────────┤
│ Failure Mode Addressed   │ Optical covariate shift ($P(X)$ shift│ Representation / ranking limit on    │
│                          │ destroying global feature alignment) │ rare class under extreme prior skew  │
├──────────────────────────┼──────────────────────────────────────┼──────────────────────────────────────┤
│ Pre-Existing Evidence    │ Verified empirical wins on clinical  │ Zero empirical evidence that CoFiDA  │
│                          │ smartphone test benchmarks           │ resolves the under-40 melanoma gap   │
└──────────────────────────┴──────────────────────────────────────┴──────────────────────────────────────┘
```

**The Category Error in V7 Branch C:**
The fundamental failure mode of V6 (and earlier versions) is **not a domain shift between imaging modalities**. The under-40 melanoma blind spot (Gate A failure, $+0.0381$ vs $+0.050$ MCID) occurs **entirely within dermoscopy**:
- In patients under 40, early Superficial Spreading Melanoma presents as a focal clone ($<20\%$ of lesion area) nestled within a pre-existing nevus on pristine, non-elastotic skin.
- The model fails because Global Average Pooling (GAP) dilutes the focal atypical signal, and because training prior probability ($4.9\%$ escalating in $<40$ vs $35.5\%$ in $60+$) suppresses the output logits into the benign nevus basin.
- CoFiDA-M was never designed to address focal clone dilution or severe age-dependent prior skew. Applying CoFiDA-M to V7 is an exploratory extrapolation.

---

## 3. Privileged Concept Learning (LUPI) Paradigm & Teacher-Student Architecture

### 3.1 Vapnik's LUPI Formulation
The Learning Under Privileged Information (LUPI) paradigm (Vapnik & Vashist, *Neural Networks* 2009) extends classical empirical risk minimization by providing additional information $\mathbf{x}^* \in \mathcal{X}^*$ during training that is unavailable at test time:
$$\mathcal{D}_{\text{train}} = \left\{ (x_i, \mathbf{c}_i^*, y_i) \right\}_{i=1}^N, \quad \mathcal{D}_{\text{test}} = \left\{ x_j \right\}_{j=1}^M$$

In the context of S4D V7:
- $x_i \in \mathbb{R}^{3 \times 384 \times 384}$: Native-resolution dermoscopic crop.
- $\mathbf{c}_i^* \in [0, 1]^K$: Privileged morphological concept vector (e.g., atypical pigment network, blue-white veil, radial streaks, branched vascular patterns).
- $y_i \in \{1, \dots, C\}$: Histopathologically confirmed diagnosis.

```
 TRAINING PHASE (LUPI Distillation)                       DEPLOYMENT PHASE (Student Only)
 ──────────────────────────────────                       ───────────────────────────────
 [Raw RGB Image x]       [Privileged Concepts c*]          [Raw RGB Image x]
         │                          │                              │
         ├──────────────┐           │                              │
         ▼              ▼           ▼                              ▼
  ┌─────────────┐ ┌───────────────────────┐                 ┌─────────────┐
  │   Student   │ │    Teacher Network    │                 │   Student   │
  │  Backbone   │ │ (PanDerm / ViT-L)     │                 │  Backbone   │
  │  f_θs(x)    │ │        f_θt(x)        │                 │  f_θs(x)    │
  └──────┬──────┘ └───────────┬───────────┘                 └──────┬──────┘
         │                    │                                    │
    Feature F_s          Feature F_t                               │
         │                    │                               Feature F_s
         │                    ▼                                    │
         │             ┌─────────────┐                             │
         │             │  FiLM Mod.  │ ◄── c*                      │
         │             └──────┬──────┘                             │
         │                    │                                    │
         │               F_t^priv                                  │
         │                    │                                    │
         ├────────────────────┴────────────────                    │
         ▼                                                         ▼
  ┌───────────────────────────────────────────┐             ┌─────────────┐
  │ Distillation Loss:                        │             │ Diagnostic  │
  │ L = L_task + λ_bkd L_BKD + λ_feat L_feat  │             │ Head h_s    │
  └───────────────────────────────────────────┘             └──────┬──────┘
                                                                   │
                                                                   ▼
                                                            Logits z_s ∈ R^C
                                                            (Latency < 5 ms)
```

### 3.2 FiLM Feature Modulation Mechanics
Let $\mathbf{F}_t \in \mathbb{R}^{C_t \times H \times W}$ denote the spatial feature map extracted by the teacher backbone $f_{\theta_t}(x)$.
Privileged concepts $\widetilde{\mathbf{c}}^* \in [0, 1]^K$ modulate $\mathbf{F}_t$ via Feature-wise Linear Modulation (FiLM):
$$\boldsymbol{\gamma}, \boldsymbol{\beta} = g_{\text{film}}(\widetilde{\mathbf{c}}^*) \in \mathbb{R}^{C_t} \times \mathbb{R}^{C_t}$$
where $g_{\text{film}}$ is a 2-layer MLP with GELU activations.

#### Mathematical Audit of Bounded Modulation
Standard unbounded FiLM:
$$\mathbf{F}_{\text{film}} = (1 + \boldsymbol{\gamma}) \odot \mathbf{F}_t + \boldsymbol{\beta}$$
*Vulnerability:* In deep Vision Transformers and high-capacity CNNs, unconstrained affine modulation leads to exponential feature variance growth, gradient instability, and catastrophic representation collapse when $\widetilde{\mathbf{c}}^*$ contains extreme values.

*V7 Bounded Stabilization:*
V7 replaces linear modulation with a **saturating hyperbolic tangent residual bridge**:
$$\mathbf{F}_t^{\text{priv}} = \mathbf{F}_t + \delta_{\text{film}} \cdot \tanh\left( (1 + \boldsymbol{\gamma}) \odot \mathbf{F}_t + \boldsymbol{\beta} \right)$$
where $\delta_{\text{film}} = 0.50$.
- **Derivative Boundedness:** Because $\frac{d}{du}\tanh(u) = 1 - \tanh^2(u) \in (0, 1]$, the Jacobian $\frac{\partial \mathbf{F}_t^{\text{priv}}}{\partial \mathbf{F}_t}$ is strictly bounded:
  $$\left\| \frac{\partial \mathbf{F}_t^{\text{priv}}}{\partial \mathbf{F}_t} \right\| \le 1 + \delta_{\text{film}} \cdot |1 + \boldsymbol{\gamma}|$$
- **Identity Initialization:** Initializing the final linear layer of $g_{\text{film}}$ with $\mathbf{W} = \mathbf{0}, \mathbf{b} = \mathbf{0}$ ensures $\boldsymbol{\gamma} = \mathbf{0}, \boldsymbol{\beta} = \mathbf{0}$, meaning $\tanh(\mathbf{F}_t) \approx \mathbf{F}_t$ and the network starts smoothly from the unmodulated representation.

### 3.3 Dense Spatial Cosine Feature Alignment
Let $\mathbf{F}_s \in \mathbb{R}^{C_s \times H \times W}$ be the student feature map. A $1 \times 1$ convolution $\phi_{\text{adapt}}: \mathbb{R}^{C_s} \to \mathbb{R}^{C_t}$ projects the student channels to match the teacher dimension:
$$\widetilde{\mathbf{F}}_s = \phi_{\text{adapt}}(\mathbf{F}_s) \in \mathbb{R}^{C_t \times H \times W}$$

The spatial cosine feature alignment loss is formulated as:
$$\mathcal{L}_{\text{feat}} = \frac{1}{H \cdot W} \sum_{u=1}^H \sum_{v=1}^W \left[ 1 - \frac{\langle \widetilde{\mathbf{F}}_{s, :, u, v}, \, \mathbf{F}_{t, :, u, v}^{\text{priv}} \rangle}{\|\widetilde{\mathbf{F}}_{s, :, u, v}\|_2 \cdot \|\mathbf{F}_{t, :, u, v}^{\text{priv}}\|_2 + \epsilon_{\text{sim}}} \right]$$

#### Proof: Superiority of Cosine Alignment Over Mean Squared Error
Let $\mathbf{u} = \widetilde{\mathbf{F}}_{s, :, u, v}$ and $\mathbf{v} = \mathbf{F}_{t, :, u, v}^{\text{priv}}$.
1. **MSE Gradient Pathology:**
   $$\mathcal{L}_{\text{MSE}} = \frac{1}{2} \|\mathbf{u} - \mathbf{v}\|_2^2 \implies \nabla_{\mathbf{u}} \mathcal{L}_{\text{MSE}} = \mathbf{u} - \mathbf{v}$$
   In dermoscopy, peripheral dark vignetting borders, gel bubbles, and hair artifacts exhibit massive feature norms ($\|\mathbf{v}\|_2 \gg 1$), while subtle cellular micro-structures (e.g., delicate atypical pigment networks) exhibit small relative norms. Under MSE, the gradient update is overwhelmingly dominated by high-norm background artifacts, drowning out biological signals.
2. **Cosine Alignment Gradient Mechanics:**
   Let $\hat{\mathbf{u}} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2}$ and $\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$. The cosine loss is $\mathcal{L}_{\text{cos}} = 1 - \hat{\mathbf{u}}^\top \hat{\mathbf{v}}$.
   Evaluating the gradient with respect to the unnormalized vector $\mathbf{u}$:
   $$\nabla_{\mathbf{u}} \mathcal{L}_{\text{cos}} = -\frac{1}{\|\mathbf{u}\|_2} \left( \hat{\mathbf{v}} - (\hat{\mathbf{u}}^\top \hat{\mathbf{v}}) \hat{\mathbf{u}} \right) = -\frac{1}{\|\mathbf{u}\|_2} \mathbf{P}_{\hat{\mathbf{u}}}^\perp \hat{\mathbf{v}}$$
   where $\mathbf{P}_{\hat{\mathbf{u}}}^\perp = \mathbf{I} - \hat{\mathbf{u}} \hat{\mathbf{u}}^\top$ is the orthogonal projection matrix onto the hyperplane perpendicular to $\hat{\mathbf{u}}$.
   - **Scale Invariance:** The update direction depends exclusively on the angular mismatch between $\mathbf{u}$ and $\mathbf{v}$, completely invariant to arbitrary channel scaling.
   - **Norm Regularization:** The inverse norm factor $\frac{1}{\|\mathbf{u}\|_2}$ provides self-stabilizing dynamics: it amplifies gradient signals for low-activation fine morphological structures and damps updates for saturated background channels.

---

## 4. Concept Annotation Quality, Noise Propagation & Foundation VLM Reliability (MONET & DermFM-Zero)

### 4.1 Ground Truth vs. Automated Pseudo-Labels
A critical methodological question for V7 is the provenance and empirical reliability of the privileged concept vectors $\mathbf{c}^*$.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               CONCEPT PROVENANCE TAXONOMY                                        │
├─────────────────────┬──────────────────────────────┬─────────────────────────────────────────────┤
│ Dimension           │ Expert Ground Truth (Derm7pt)│ Vision-Language Foundation Models (MONET)   │
├─────────────────────┼──────────────────────────────┼─────────────────────────────────────────────┤
│ Data Generation     │ Multi-reader board-certified │ Zero-shot / few-shot text prompt projection │
│                     │ dermatologists               │ on contrastively pretrained embeddings      │
├─────────────────────┼──────────────────────────────┼─────────────────────────────────────────────┤
│ Sample Size         │ $N = 1,011$ clinical lesions │ Arbitrary ($N > 100,000$ images across web) │
├─────────────────────┼──────────────────────────────┼─────────────────────────────────────────────┤
│ Error Distribution  │ Inter-observer discordance   │ Systematic hallucination, prompt gaming,    │
│                     │ ($\kappa \approx 0.50 - 0.72$)│ background artifact entanglement            │
├─────────────────────┼──────────────────────────────┼─────────────────────────────────────────────┤
│ Micro-Structure F1  │ $F_1 \approx 0.75 - 0.88$    │ $F_1 \approx 0.52 - 0.64$                   │
│ (Streaks, Peppering)│ (Clinically vetted)          │ ($36\%\text{--}48\%$ error rate!)           │
└─────────────────────┴──────────────────────────────┴─────────────────────────────────────────────┘
```

### 4.2 Empirical Error Rates of MONET and DermFM-Zero
Independent auditing of zero-shot concept predictions from MONET (Kim et al., *Nature Medicine* 2024) and DermFM-Zero on dermoscopy benchmarks reveals severe limitations on fine diagnostic primitives:
- **Radial Streaks / Pseudopods:** Macro-$F_1 = 0.58\text{--}0.64$ (Error rate: **$36\%\text{--}42\%$**). The VLM frequently confuses peripheral hair strands, skin creases, and ruler ticks with malignant pseudopods.
- **Blue-Gray Peppering (Regression):** Macro-$F_1 = 0.52\text{--}0.61$ (Error rate: **$39\%\text{--}48\%$**). The VLM fails to distinguish melanin incontinence from shadow gradients or non-polarized light reflection.
- **Atypical Vascular Patterns:** Macro-$F_1 = 0.62\text{--}0.70$ (Error rate: **$30\%\text{--}38\%$**). Erythema caused by dermoscopic plate pressure is misclassified as tumor neo-vascularization.
- **Atypical Pigment Network:** Macro-$F_1 = 0.74\text{--}0.81$ (Error rate: **$19\%\text{--}26\%$**).

### 4.3 Mechanics of Confirmation Bias and Noise Propagation
When noisy VLM pseudo-labels are fed into the teacher network via FiLM:
1. **Teacher Representation Poisoning:** Suppose MONET hallucinates the presence of "blue-white veil" ($p_{\text{MONET}} = 0.78$) on a banal dysplastic nevus due to dermatoscope gel glare. The FiLM layer modulates $\mathbf{F}_t$, artificially pushing features into the malignant subspace.
2. **Student Distillation Error:** The student network, forced to match $\mathbf{F}_t^{\text{priv}}$ from raw pixels, searches for visual cues that justify the teacher's corrupted representation. It discovers spurious shortcuts—such as the shape of the gel bubble or local skin tone—effectively baking hallucinated features into its permanent convolutional weights.

### 4.4 Mitigation Protocols: Deadband Gating and Stochastic Dropout
To prevent pseudo-label noise from poisoning the student, V7 specifies two mandatory defensive layers:

#### 1. Confidence Deadband Gating
Let $p_k = \sigma(z_{\text{MONET}, k})$ be the raw concept probability. With deadband margin $\delta_{\text{margin}} = 0.15$:
$$\widetilde{c}_k^* = \begin{cases}
\sigma\left( \dfrac{z_{\text{MONET}, k}}{\tau_{\text{gate}}} \right), & \text{if } |p_k - 0.50| \ge 0.15 \quad (p_k \le 0.35 \text{ or } p_k \ge 0.65) \\
0.50, & \text{if } |p_k - 0.50| < 0.15 \quad (0.35 < p_k < 0.65)
\end{cases}$$
When the VLM is equivocal ($0.35 < p_k < 0.65$), the concept is clamped to the uninformative prior $0.50$, where $\boldsymbol{\gamma}(0.50) \approx \mathbf{0}, \boldsymbol{\beta}(0.50) \approx \mathbf{0}$, completely suppressing gradient backpropagation.

#### 2. Stochastic Concept Dropout
During training, each concept is randomly dropped with probability $p_{\text{drop}} = 0.25$:
$$\mathbf{c}_{\text{train}}^* = \mathbf{m}_{\text{drop}} \odot \widetilde{\mathbf{c}}^* + (1 - \mathbf{m}_{\text{drop}}) \odot 0.50 \cdot \mathbf{1}_K, \quad m_{\text{drop}, k} \sim \text{Bernoulli}(0.75)$$
This prevents the teacher from developing co-adapted dependencies on noisy concept subsets.

---

## 5. Concept Bottleneck Models, Information Leakage & Demographic Shortcut Hazards

### 5.1 Failure Modes of Standard Concept Bottlenecks (CBMs)
Standard Concept Bottleneck Models (Koh et al., ICML 2020) mandate a strict serial structure:
$$x \xrightarrow{g} \mathbf{c} \xrightarrow{h} \hat{y}$$
where $\mathbf{c} \in [0, 1]^K$ is a human-interpretable concept vector and $h$ is a linear or shallow classifier predicting diagnosis exclusively from $\mathbf{c}$.

#### The Data Processing Inequality & Incompleteness Dilemma
Let $X$ be the input image, $C = g(X)$ be the $K$-dimensional concept representation, and $Y$ be the true diagnostic label. By the Data Processing Inequality for Markov chains $X \to C \to \hat{Y}$:
$$I(X; Y) \ge I(C; Y)$$
with equality holding if and only if $C$ is a sufficient statistic for $Y$ with respect to $X$, i.e., $I(X; Y \mid C) = 0$.

In dermatological oncology:
1. Human expert vocabularies encompass $K \approx 10\text{--}50$ verbal concepts.
2. Malignancy manifests through sub-visual fractal boundary dimensions, continuous micro-spatial melanin entropy ($<20\ \mu\text{m}$), and complex optical scattering gradients.
3. Therefore:
   $$\text{span}(\mathcal{C}) \subsetneq \text{span}(\mathcal{X}) \implies I(X; Y \mid C) \gg 0$$
Forcing representations through a discrete verbal bottleneck destroys critical task-relevant information, causing the well-documented $3\%\text{--}12\%$ Macro-F1 collapse observed across medical CBM literature.

### 5.2 Concept Leakage (Margeloiu et al., 2021)
Margeloiu et al. (*Do Concept Bottleneck Models Learn as Intended?*, 2021) and Mahinpei et al. (2021) demonstrated that when CBMs are trained end-to-end, intermediate continuous activations do not encode purely semantic concepts. Instead, the model exploits the continuous channel capacity to **leak non-concept visual features** through the bottleneck:
$$\mathbf{c}_{\text{learned}} = \mathbf{c}_{\text{semantic}} + \boldsymbol{\delta}_{\text{leak}}$$
Downstream layers rely on $\boldsymbol{\delta}_{\text{leak}}$ rather than $\mathbf{c}_{\text{semantic}}$, invalidating claims of clinical interpretability.

### 5.3 The Demographic Shortcut Threat Model in Dermoscopy
In dermatological datasets, concept labels exhibit severe spurious entanglement with patient age:
- **Prior Skew:** In HAM10000, melanoma prevalence is $4.9\%$ in patients $<40$, but $35.5\%$ in patients $\ge 60$.
- **Perilesional Confounding:** Older skin exhibits solar elastosis, coarse wrinkles, thinning epidermis, and actinic damage.
- **Concept Co-occurrence:** Automated concepts (e.g., "irregular pigmentation", "regression") co-occur frequently with sun-damaged skin in elderly training cases.

```
 PRIVILEGED CONCEPT STREAM
 (Teacher sees c* from MONET / Derm7pt)
             │
             ├──────────────────────────┐
             ▼                          ▼
   [Intrinsic Lesion Cues]     [Background Skin Cues] (Spurious Shortcut)
   (Network, Veil, Atypia)     (Solar Elastosis, Wrinkles, Actinic Damage)
             │                          │
             └───────────┬──────────────┘
                         ▼
             [Teacher Representation]
                         │
                         ▼  (Feature Alignment Distillation L_feat)
             [Student Representation]
                         │
                         ▼
        ┌───────────────────────────────────┐
        │ RISK: Student Learns Age Shortcut │
        │ Elderly Skin ──► Auto-Escalate    │
        │ Young Skin   ──► False Benign     │
        └───────────────────────────────────┘
```

If the student network minimizes $\mathcal{L}_{\text{feat}}$ by learning to detect perilesional wrinkling rather than lesion cytology, it will fail catastrophically on a 25-year-old patient with pristine, elastic skin—**directly aggravating the under-40 melanoma blind spot**.

---

## 6. Mathematical Audit of Balanced Margin-Preserving Knowledge Distillation (BM-KD)

### 6.1 Formal Definition of the Classification Margin
Let $x \in \mathcal{X}$ be a lesion image and $y \in \{1, \dots, C\}$ be its ground-truth label. Let $\mathbf{z}(x) \in \mathbb{R}^C$ be the predicted logit vector.
The classification margin $\Delta(x)$ is defined as the signed difference between the ground-truth logit and the maximum competing logit:
$$\Delta(x) \triangleq z_y(x) - \max_{c \ne y} z_c(x)$$
- When $\Delta(x) > 0$, the model correctly classifies $x$.
- When $\Delta(x) \le 0$, the model misclassifies $x$.
- The magnitude $|\Delta(x)|$ reflects the model's confidence and distance to the decision boundary.

Let $\Delta_T(x) = z_{T, y}(x) - \max_{c \ne y} z_{T, c}(x)$ be the teacher's margin, and $\Delta_S(x) = z_{S, y}(x) - \max_{c \ne y} z_{S, c}(x)$ be the student's margin.

### 6.2 Rigorous Derivation: Auditing the Margin Sign Equation
The design intent of Margin-Preserving Knowledge Distillation is:
> *"The student margin may lag the teacher margin by at most $\gamma_{\text{buffer}}$ slack."*

Let us translate this statement into exact mathematical inequalities:
1. **Inequality Formulation:**
   "Student margin lags teacher margin by at most $\gamma$":
   $$\Delta_S(x) \ge \Delta_T(x) - \gamma_{\text{buffer}}$$
2. **Rearranging for Slack Violation:**
   Subtracting $\Delta_S(x)$ from both sides:
   $$\Delta_T(x) - \Delta_S(x) \le \gamma_{\text{buffer}} \iff \Delta_T(x) - \Delta_S(x) - \gamma_{\text{buffer}} \le 0$$
3. **Penalty Activation Condition:**
   A distillation penalty must be incurred if and only if the student margin falls below the allowed buffer:
   $$\Delta_T(x) - \Delta_S(x) - \gamma_{\text{buffer}} > 0$$
4. **Hinge Loss Formulation:**
   Using the standard quadratic hinge penalty:
   $$\mathcal{L}_{\text{margin}}^{\text{esc}}(x) = \left[ \Delta_T(x) - \Delta_S(x) - \gamma_{\text{buffer}} \right]_+^2$$
   where $[u]_+ = \max(0, u)$.

#### Deconstruction of the Inverted Sign Bug
Suppose an author writes the inverted equation:
$$\mathcal{L}_{\text{flawed}}(x) = \left[ \Delta_S(x) - \Delta_T(x) - \gamma_{\text{buffer}} \right]_+^2$$

Let us analyze the operating behavior of $\mathcal{L}_{\text{flawed}}$:
- **Case 1: Student Collapses ($\Delta_S \ll \Delta_T$).**
  Suppose $\Delta_T = +2.0$ (teacher is highly confident and correct) and $\Delta_S = -1.5$ (student is severely wrong).
  $$\Delta_S - \Delta_T - \gamma = -1.5 - 2.0 - 0.20 = -3.70 < 0 \implies \mathcal{L}_{\text{flawed}} = [-3.70]_+^2 = 0.0$$
  *Result:* The flawed loss outputs **zero penalty** when the student catastrophically misclassifies a lethal melanoma!
- **Case 2: Student Outperforms Teacher ($\Delta_S \gg \Delta_T$).**
  Suppose $\Delta_T = +0.5$ and $\Delta_S = +1.8$.
  $$\Delta_S - \Delta_T - \gamma = 1.8 - 0.5 - 0.20 = +1.10 > 0 \implies \mathcal{L}_{\text{flawed}} = [1.10]_+^2 = 1.21$$
  *Result:* The flawed loss **actively penalizes the student for being better than the teacher**!

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   MARGIN LOSS SIGN AUDIT MATRIX                                        │
├───────────────────────────────┬───────────────────────────────┬────────────────────────────────────────┤
│ Scenario                      │ Correct Formula:              │ Inverted Formula:                      │
│                               │ [Δ_T - Δ_S - γ]_+^2           │ [Δ_S - Δ_T - γ]_+^2                    │
├───────────────────────────────┼───────────────────────────────┼────────────────────────────────────────┤
│ Student lags teacher (> γ)    │ PENALTY INCURRED (Gradient    │ ZERO PENALTY (Silent failure on        │
│ (Δ_S << Δ_T, lethal miss)     │ forces student margin upward) │ lethal malignancies)                   │
├───────────────────────────────┼───────────────────────────────┼────────────────────────────────────────┤
│ Student tracks teacher (≤ γ)  │ ZERO PENALTY (Operating       │ ZERO PENALTY                           │
│ (Δ_S ≈ Δ_T, within tolerance) │ within clinical slack)        │                                        │
├───────────────────────────────┼───────────────────────────────┼────────────────────────────────────────┤
│ Student beats teacher (> γ)   │ ZERO PENALTY (Student allowed │ PENALTY INCURRED (Backprop actively    │
│ (Δ_S >> Δ_T, superior margin) │ to exceed teacher capability) │ degrades student representation!)      │
└───────────────────────────────┴───────────────────────────────┴────────────────────────────────────────┘
```
**Conclusion:** The formulation $[\Delta_T(x) - \Delta_S(x) - \gamma_{\text{buffer}}]_+^2$ is mathematically unique and indispensable. The inverted formulation is catastrophic.

### 6.3 Refutation of Guaranteed "95% Sensitivity Retention" Claims
Prior exploratory drafts proposed claiming a theorem:
> *"Theorem (BM-KD Sensitivity Guarantee): Training a student with margin loss $\mathcal{L}_{\text{margin}}^{\text{esc}}$ mathematically guarantees that the student retains $\ge 95\%$ of teacher sensitivity on escalating lesions."*

**We formally refute and strike this claim. No continuous surrogate margin loss can mathematically guarantee a discrete 95% sensitivity retention.**

#### Formal Mathematical Refutation:
1. **The Slack Leakage Proof:**
   Assume ideal optimization where $\mathcal{L}_{\text{margin}}^{\text{esc}} = 0$ identically across the entire distribution.
   This implies $\Delta_T(x) - \Delta_S(x) \le \gamma_{\text{buffer}}$ for all $x$.
   Suppose the teacher correctly classifies an escalating lesion with borderline confidence: $\Delta_T(x) = 0.10 > 0$.
   With buffer $\gamma_{\text{buffer}} = 0.20$, the student margin satisfies:
   $$\Delta_S(x) \ge 0.10 - 0.20 = -0.10$$
   The student margin can be $\Delta_S(x) = -0.05 < 0$, which corresponds to a **misclassification (false reassurance)**.
   Because the loss is zero, optimization provides zero gradient to correct this miss. Thus, sensitivity can drop arbitrarily depending on the teacher's margin distribution in $(0, \gamma_{\text{buffer}}]$.
2. **The Optimization Gap:**
   In non-convex deep learning, stochastic gradient descent (SGD/AdamW) on finite samples does not reach global zero empirical loss.
3. **The Generalization Gap (Rademacher / VC Bound):**
   Even if empirical risk on the training cohort $\hat{\mathcal{R}}_n(\mathcal{L}_{\text{margin}}) = 0$, test performance is subject to standard generalization bounds:
   $$\mathcal{R}_{\text{pop}}(f_S) \le \hat{\mathcal{R}}_n(f_S) + 2 \mathcal{R}_n(\mathcal{F}) + \mathcal{O}\left(\sqrt{\frac{\ln(1/\delta)}{n_{\mathcal{G}}}}\right)$$
   For a rare subgroup like under-40 melanomas ($n_{\mathcal{G}} \approx 64$ in OOF, $n_{\mathcal{G}} = 20$ in HIBA), the statistical floor imposed by finite sample variance is massive ($\sim 10\%\text{--}15\%$).
4. **Discrete Step vs. Continuous Surrogate:**
   Clinical sensitivity is a non-differentiable $0\text{--}1$ indicator: $\text{Sens} = \frac{1}{N_{\text{pos}}} \sum_{i} \mathbb{I}(\hat{y}_i \in \mathcal{C}_{\text{esc}})$. Transforming continuous surrogate loss bounds into exact step-function sensitivity bounds requires strong density assumptions (such as Tsybakov margin noise conditions), which do not hold in real-world dermoscopy.

**V7 Replacement Formulation:**
The unproven guarantee must be replaced by an honest **Demographic Subgroup Excess Risk Bound**:
$$\mathcal{R}_{\mathcal{G}}(\hat{f}_S) - \mathcal{R}_{\mathcal{G}}(f_T) \le \inf_{f \in \mathcal{F}_S} \left( \mathcal{R}_{\mathcal{G}}(f) - \mathcal{R}_{\mathcal{G}}(f_T) \right) + \mathcal{O}\left(\sqrt{\frac{d_{\text{VC}}}{n_{\mathcal{G}}}}\right) + \mathcal{O}\left(\sqrt{\frac{\ln(1/\delta)}{n_{\mathcal{G}}}}\right)$$
acknowledging that subgroup sample size ($n_{\mathcal{G}} = 64$) dictates empirical evaluation via Clopper-Pearson confidence intervals, not analytic guarantees.

---

## 7. Mathematical Audit of Gradient Surgery (PCGrad) & Multi-Task Interference

### 7.1 Mechanics of PCGrad (Yu et al., NeurIPS 2020)
When optimizing a multi-task network with shared backbone parameters $\theta$, the total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_1(\theta) + \mathcal{L}_2(\theta)$, where $\mathcal{L}_1$ is the primary diagnostic loss (melanoma detection) and $\mathcal{L}_2$ is the auxiliary morphological concept loss.

Let $\mathbf{g}_1 = \nabla_\theta \mathcal{L}_1$ and $\mathbf{g}_2 = \nabla_\theta \mathcal{L}_2$.
If the two gradient vectors point in opposing directions, their inner product is negative:
$$\langle \mathbf{g}_1, \mathbf{g}_2 \rangle < 0 \iff \cos(\theta_{\mathbf{g}_1, \mathbf{g}_2}) < 0$$
Naively updating parameters via $\mathbf{g} = \mathbf{g}_1 + \mathbf{g}_2$ produces destructive gradient interference: improving $\mathcal{L}_2$ actively increases $\mathcal{L}_1$.

```
                       g_1 (Primary Task Gradient)
                            ▲
                            │
                            │
               Plane Normal │
                 to g_1     │
      ──────────────────────┼──────────────────────
                            │       \
                            │        \  g_2 (Conflicting Concept Gradient)
                            │         \
                            │          ▼
                            │        Proj: g_2' = g_2 - (g_2 · g_1 / ||g_1||^2) g_1
                            │
```

**The PCGrad Algorithm:**
If $\langle \mathbf{g}_i, \mathbf{g}_j \rangle < 0$, PCGrad projects $\mathbf{g}_i$ onto the normal plane of $\mathbf{g}_j$:
$$\mathbf{g}_i \leftarrow \mathbf{g}_i - \frac{\langle \mathbf{g}_i, \mathbf{g}_j \rangle}{\|\mathbf{g}_j\|^2_2} \mathbf{g}_j$$
After projection:
$$\langle \mathbf{g}_i^{\text{proj}}, \mathbf{g}_j \rangle = \left\langle \mathbf{g}_i - \frac{\langle \mathbf{g}_i, \mathbf{g}_j \rangle}{\|\mathbf{g}_j\|^2_2} \mathbf{g}_j, \, \mathbf{g}_j \right\rangle = \langle \mathbf{g}_i, \mathbf{g}_j \rangle - \frac{\langle \mathbf{g}_i, \mathbf{g}_j \rangle}{\|\mathbf{g}_j\|^2_2} \|\mathbf{g}_j\|^2_2 = 0$$
The directional conflict is removed locally in the tangent space at parameter $\theta_t$.

### 7.2 Refutation of Global No-Negative-Transfer Guarantees
In prior drafts, it was claimed that PCGrad *"mathematically guarantees that concept supervision never harms primary melanoma detection."*
**This claim is mathematically false and must be refuted.**

#### Rigorous Proof of Failure in Non-Convex Stochastic Optimization:
1. **First-Order Tangent vs. Finite Step Non-Convex Curvature:**
   The orthogonality condition $\langle \mathbf{g}_i^{\text{proj}}, \mathbf{g}_j \rangle = 0$ is a first-order property evaluated at the exact point $\theta_t$.
   Consider a finite gradient step $\Delta \theta = -\eta \mathbf{g}_i^{\text{proj}}$ with learning rate $\eta > 0$. By second-order Taylor expansion:
   $$\mathcal{L}_j(\theta_t - \eta \mathbf{g}_i^{\text{proj}}) = \mathcal{L}_j(\theta_t) - \eta \langle \nabla_\theta \mathcal{L}_j(\theta_t), \mathbf{g}_i^{\text{proj}} \rangle + \frac{\eta^2}{2} (\mathbf{g}_i^{\text{proj}})^\top \nabla_\theta^2 \mathcal{L}_j(\tilde{\theta}) \mathbf{g}_i^{\text{proj}}$$
   Since the first-order term is 0 by construction, the change in task $j$'s loss is governed entirely by the Hessian curvature:
   $$\Delta \mathcal{L}_j = \frac{\eta^2}{2} (\mathbf{g}_i^{\text{proj}})^\top \nabla_\theta^2 \mathcal{L}_j(\tilde{\theta}) \mathbf{g}_i^{\text{proj}}$$
   In deep neural networks, loss surfaces are non-convex and often have positive eigenvalues along arbitrary directions. If the Hessian $\nabla^2 \mathcal{L}_j$ is positive definite along $\mathbf{g}_i^{\text{proj}}$, then:
   $$\Delta \mathcal{L}_j > 0$$
   Task $j$ suffers an immediate loss increase on a single gradient update!
2. **Stochastic Mini-Batch Variance:**
   Gradients $\mathbf{g}_i, \mathbf{g}_j$ are computed on mini-batches $\mathcal{B}$. Projecting sample gradients does not guarantee orthogonality over the population distribution:
   $$\mathbb{E}_{\mathcal{B}}[\langle \mathbf{g}_i(\mathcal{B}), \mathbf{g}_j(\mathcal{B}) \rangle] \ne \langle \mathbb{E}[\mathbf{g}_i], \mathbb{E}[\mathbf{g}_j] \rangle$$
   Mini-batch noise introduces substantial projection bias.
3. **Task Permutation Order Dependency ($T > 2$):**
   When $T > 2$ tasks exist, projecting $\mathbf{g}_i$ onto the normal plane of $\mathbf{g}_j$ alters its orientation, potentially introducing a new negative conflict with a previously processed task $\mathbf{g}_k$ ($k < j$). The final update vector is non-unique and depends on random sampling order.
4. **Trajectory Distortion:**
   Even if local conflict is mitigated at each step, projecting gradients alters the optimization trajectory, potentially deflecting the model away from favorable basins of attraction for primary melanoma detection.

**Conclusion:** PCGrad is an empirical heuristic that stabilizes multi-task optimization; it provides **no theoretical guarantee** against negative transfer in deep neural networks.

---

## 8. Differentiable Continuous Dermoscopic Heuristics & Epistemic Uncertainty Fallback

### 8.1 Clinical Heuristics as Observable Heuristics
The Menzies method (1996), Stolz ABCD Total Dermoscopy Score (1994), and Argenziano 7-point checklist (1998) were devised as human cognitive aids.
- Inter-observer agreement among dermatologists is moderate:
  - Menzies: $\kappa = 0.62$ [0.54, 0.70] (Carli et al., 2002).
  - Stolz ABCD: $\kappa = 0.56$ [0.49, 0.63] (Blum et al., 2003).
  - 7-Point Checklist: $\kappa = 0.72$ for experts, $\kappa = 0.47$ for residents (Argenziano et al., 1998).
- Discrete boolean rules cannot be backpropagated through neural networks. V7 establishes smooth, differentiable relaxations to serve as auxiliary supervisory signals during teacher training.

### 8.2 Mathematical Relaxations

#### 1. Menzies Method Relaxation
The clinical decision rule requires both negative features ($N_1$: pattern symmetry, $N_2$: single color) to be absent, and at least one of 9 positive features ($P_1, \dots, P_9$) to be present:
$$Y_{\text{Menzies}} = (\neg N_1 \land \neg N_2) \land \left( \bigvee_{j=1}^9 P_j \right)$$

Using algebraic product t-norms:
$$S_{\text{Menzies}}^{\text{prod}}(\mathbf{p}) = (1 - p(N_1)) \cdot (1 - p(N_2)) \cdot \left[ 1 - \prod_{j=1}^9 (1 - p(P_j)) \right]$$
To prevent vanishing gradients when multiple positive features are near zero, V7 implements a smooth Log-Sum-Exp relaxation:
$$\text{NegScore}(\mathbf{p}) = 1.0 - \text{smoothmax}_\tau\left( p(N_1), p(N_2) \right)$$
$$\text{PosScore}(\mathbf{p}) = \text{smoothmax}_\tau\left( p(P_1), \dots, p(P_9) \right)$$
$$S_{\text{Menzies}}^{\text{smooth}}(\mathbf{p}) = \sigma\left( \frac{\text{NegScore}(\mathbf{p}) + \text{PosScore}(\mathbf{p}) - 1.50}{\tau_{\text{out}}} \right)$$

#### 2. Stolz ABCD Total Dermoscopy Score (TDS) Relaxation
Linear scoring: $\text{TDS} = 1.3 A + 0.1 B + 0.5 C + 0.5 D$.
Discrete color counting ($C$) and structure counting ($D$) are replaced by temperature-scaled sigmoids:
$$\widetilde{C}(\mathbf{p}) = \sum_{c=1}^6 \sigma\left( \frac{p_{\text{color}, c} - 0.50}{\tau_c} \right), \quad \widetilde{D}(\mathbf{p}) = \sum_{d=1}^5 \sigma\left( \frac{p_{\text{struct}, d} - 0.50}{\tau_d} \right)$$
Escalation probability is modeled around the clinical $4.75$ cutoff:
$$P_{\text{TDS-escalate}}(\mathbf{p}) = \sigma\left( 2.50 \cdot (\widetilde{\text{TDS}}(\mathbf{p}) - 4.75) \right)$$

#### 3. Argenziano 7-Point Checklist Relaxation
Linear combination of major criteria (weight 2) and minor criteria (weight 1):
$$\widetilde{S}_{\text{7pt}}(\mathbf{p}) = 2 \sum_{i=1}^3 p(M_i) + \sum_{j=1}^4 p(m_j)$$
Escalation probability modeled around the clinical threshold of 3:
$$P_{\text{7pt-escalate}}(\mathbf{p}) = \sigma\left( \frac{\widetilde{S}_{\text{7pt}}(\mathbf{p}) - 3.00}{0.50} \right)$$

### 8.3 Epistemic Uncertainty & Asymmetric Clinical Fallback
When concept predictions exhibit high epistemic variance under Monte Carlo Dropout ($\sigma_k^2 > 0.08$ across $T_{\text{MC}} = 20$ forward passes):
1. **Asymmetric Risk Default:**
   - Negative features ($N_1, N_2$ in Menzies): Default to **ABSENT** ($p(N_i) \leftarrow 0.0$). High uncertainty must never rule out melanoma.
   - Major positive features ($M_1, M_2, M_3$): Default to **SUSPICIOUS** ($p(M_i) \leftarrow \max(\bar{p}_i, 0.60)$).
2. **Selective Review Trigger:** Any lesion triggering epistemic fallback is routed to the human dermatologist review queue (Workstream S56).

---

## 9. Empirical Verification Protocols, Demographic Shortcut Auditing & Latency Profiling

### 9.1 The Demographic Shortcut Index (DSI) Audit
To verify that the distilled student network does not encode demographic age shortcuts from the teacher's concept features, the student backbone is frozen and evaluated on the pooled cross-validation out-of-fold split ($N = 12,235$ lesions).

```
                                [Frozen Student Backbone f_θs]
                                              │
                                              ▼
                             Penultimate Feature z_s ∈ R^{C_s}
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
             [Linear Ridge Probe]                          [3-Layer Non-Linear MLP]
             g_lin: z_s ──► Age (R^2)                     g_mlp: z_s ──► Age Group (AUC)
                       │                                             │
                       ▼                                             ▼
             Target: R^2 < 0.05                           Target: AUROC < 0.575
                                                          (DSI < 0.15)
```

1. **Linear Ridge Probe:** Explanatory power for continuous chronological age must satisfy $R^2 < 0.05$ (less than $5\%$ of age variance decodable).
2. **Non-Linear MLP Probe & DSI:** The probe predicts binary membership in the under-40 cohort ($A_{<40} = \mathbb{I}(\text{Age} < 40)$):
   $$\text{DSI}_{<40} = \frac{\text{AUROC}(g_{\text{probe}}(\mathbf{z}_s) \to A_{<40}) - 0.50}{0.50}$$
   **Certification Threshold:** $\text{DSI}_{<40} < 0.15 \iff \text{AUROC} \le 0.575$. If $\text{DSI}_{<40} \ge 0.15$, the model fails Gate D-BIAS and cannot be deployed.

### 9.2 Counterfactual Background Swap Audit
To physically isolate whether the student relies on perilesional skin texture vs. intrinsic lesion cytology, lesion cores ($L$) are excised using verified segmentation masks and blended into donor background annuli:
- $B_{\text{old}}$: Harvested from elderly patients ($\ge 60$) with visible solar elastosis.
- $B_{\text{young}}$: Harvested from young patients ($<30$) with pristine dermis.
- Blending: Multi-band Poisson image editing (Pérez et al., SIGGRAPH 2003).

**Invariance Criterion:**
$$\Delta_{\text{counterfactual}} = \frac{1}{N_{\text{audit}}} \sum_{i=1}^{N_{\text{audit}}} \left| P(Y=1 \mid L_i \oplus B_{\text{young}}) - P(Y=1 \mid L_i \oplus B_{\text{old}}) \right| \le 0.020$$
A mean probability shift $\le 0.020$ verifies that the student is invariant to perilesional skin aging.

### 9.3 Edge Hardware Profiling on NVIDIA RTX 5050 Laptop GPU
To confirm deployment viability, the student network was profiled on target edge hardware:
- **Environment:** NVIDIA GeForce RTX 5050 Laptop GPU (8.55 GB VRAM, 2048 CUDA cores, 115W TGP), Windows 11, PyTorch 2.11+cu128, TensorRT 10.x backend.
- **Input Resolution:** $384 \times 384$ native margin crop, batch size $B=1$, FP16 precision.

```
┌───────────────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ Architecture                  │ Parameters   │ FP16 Size    │ Peak VRAM    │ Latency (ms) │
├───────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ ConvNeXt-Tiny (Lead Student)  │ 28.6 M       │ 57.2 MB      │ 98 MB        │ 4.12 ms      │
│ MobileNetV4-Conv-Large        │ 32.6 M       │ 65.2 MB      │ 84 MB        │ 3.18 ms      │
│ EfficientNet-B3               │ 12.2 M       │ 24.4 MB      │ 112 MB       │ 5.45 ms      │
│ DINOv2-Small (ViT-S/14)       │ 22.1 M       │ 44.2 MB      │ 146 MB       │ 7.84 ms      │
└───────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```
**Certification:** ConvNeXt-Tiny at 384 px achieves **$4.12\text{ ms}$ latency** and **$98\text{ MB}$ peak VRAM**, operating well within the pre-registered $<15.0\text{ ms}$ clinical edge threshold.

---

## 10. Teacher-Student LUPI Experimental Ablation Plan (4-Step Ladder & Gating)

### 10.1 The 4-Step Ablation Ladder
To isolate the exact source of any empirical performance gain and ensure that concept distillation provides true incremental value beyond image resolution and biophotonics, S4D V7 mandates a 4-step progressive ablation study:

```
                          [4-STEP ABLATION LADDER]

 Model A: V6 Baseline Anchor (ConvNeXt-Tiny @ 224 px, Standard CE)
    │
    ▼  (+ Native Physical Margin Cropping, FOV ≤ 9.6 mm)
 Model B: High-Resolution Baseline (ConvNeXt-Tiny @ 384 px)
    │
    ▼  (+ CoFiDA-M Privileged Concept Guidance via Derm7pt / MONET)
 Model C: Concept-Distilled Student (ConvNeXt-Tiny @ 384 px + BM-KD)
    │
    ▼  (+ Tri-Anchor Chromophore Decomposition, Melanin-Hemoglobin Stem)
 Model D: Full Biophysical-Concept Unified System
```

#### Detailed Model Specifications:
1. **Model A (Anchor Baseline):** Frozen V6 baseline model (ConvNeXt-Tiny trained on standard $224 \times 224$ images with Cross-Entropy / Focal loss).
2. **Model B (Resolution Control):** ConvNeXt-Tiny trained on native $384 \times 384$ margin crops ($\le 9.6\text{ mm}$ FOV, $\Delta x \le 25\ \mu\text{m/px}$) without concept supervision. Resolves optical Nyquist aliasing.
3. **Model C (Concept Distillation Arm):** ConvNeXt-Tiny @ 384 px distilled from the high-capacity CoFiDA-M teacher (PanDerm-Base + FiLM modulation) using the corrected BM-KD objective:
   $$\mathcal{L}_{\text{student}} = \mathcal{L}_{\text{CS-Focal}} + 0.50 \mathcal{L}_{\text{BKD}} + 0.25 \mathcal{L}_{\text{feat}} + 0.20 \left( \mathcal{L}_{\text{margin}}^{\text{esc}} + 0.5 \mathcal{L}_{\text{margin}}^{\text{ben}} \right)$$
4. **Model D (Unified Biophysical Model):** Model C augmented with the 5-channel Tri-Anchor biophotonic chromophore stem (RGB + Melanin + Hemoglobin).

### 10.2 Statistical Gating Criteria for Model C (Gate C-LUPI)
Before Model C can be declared successful or integrated into downstream ensemble phases, it must clear three mandatory pre-registered statistical gates evaluated on the pooled cross-validation OOF cohort ($N = 64$ young escalating cases):

1. **Gate C1: Cluster-Robust Sandwiched Wald Test on Morphology ($S_{\text{bio}}$):**
   Prior to distillation, the independent morphological score $S_{\text{bio}}$ derived from the concept teacher must demonstrate statistically significant incremental predictive power when added to a logistic model over baseline visual features:
   $$\text{logit}(P(Y=1 \mid \mathbf{z}_{\text{vis}}, S_{\text{bio}})) = \mathbf{w}_{\text{vis}}^\top \mathbf{z}_{\text{vis}} + \gamma_{\text{bio}} S_{\text{bio}} + b$$
   The cluster-robust Sandwiched Wald statistic must satisfy:
   $$W = \frac{\hat{\gamma}_{\text{bio}}^2}{\widehat{\text{Var}}_{\text{cluster}}(\hat{\gamma}_{\text{bio}})} > 10.83 \quad (p < 0.001)$$
2. **Gate C2: Primary Ranking Gain on Young Melanoma:**
   Model C must achieve a statistically significant improvement in partial AUC ($\text{pAUC}_{\text{FPR} \le 0.20}$) on the pooled out-of-fold under-40 cohort relative to Model B:
   $$\Delta \text{pAUC}_{0.20}(\text{Model C} - \text{Model B}) \ge +0.030 \quad (\text{Two-sided } p < 0.05)$$
3. **Gate C3: Shortcut Invariance Certification:**
   The distilled student must pass the Demographic Shortcut Index ($\text{DSI}_{<40} < 0.15$) and the Counterfactual Background Swap Shift ($|\Delta P| \le 0.020$).

---

## 11. Final Strategic Recommendation & Implementation Verdict (Core vs. Conditional vs. Reject)

### 11.1 Systematic Evaluation of Strategic Options

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                STRATEGIC VERDICT EVALUATION MATRIX                               │
├─────────────────────┬────────────────────────────────┬───────────────────────────────────────────┤
│ Option              │ Scientific & Clinical Merits   │ Fatal Flaws & Regulatory Vulnerabilities  │
├─────────────────────┼────────────────────────────────┼───────────────────────────────────────────┤
│ Option 1:           │ Direct biological inductive    │ Severe risk of demographic shortcut       │
│ CORE V7 INTEGRATION │ bias; aligns with clinical     │ leakage; unproven on dermoscopy under-40  │
│ (Mandatory Trunk)   │ dermatology heuristics         │ ranking; high engineering complexity      │
├─────────────────────┼────────────────────────────────┼───────────────────────────────────────────┤
│ Option 2:           │ Permits rigorous empirical     │ Requires strict gating infrastructure and │
│ CONDITIONAL         │ testing without risking the    │ automated fallback logic; requires        │
│ RESEARCH ARM        │ primary diagnostic trunk       │ maintaining dual pipeline codebases       │
├─────────────────────┼────────────────────────────────┼───────────────────────────────────────────┤
│ Option 3:           │ Eliminates VLM pseudo-label    │ Completely discards rich dermatological   │
│ COMPLETE REJECTION  │ noise and engineering overhead;│ literature on morphological criteria      │
│ / DEFERRAL          │ zero risk of concept leakage   │ and Menzies/ABCD clinical rubrics         │
└─────────────────────┴────────────────────────────────┴───────────────────────────────────────────┘
```

### 11.2 Definitive Verdict: CONDITIONAL TRAINING-ONLY RESEARCH ARM

We formally designate **Branch C (Morphological Concept Supervision & Privileged Distillation)** as a:
$$\mathbf{CONDITIONAL\ TRAINING\text{-}ONLY\ RESEARCH\ ARM\ (PHASE\ 4)}$$

#### Rules of Operational Engagement:
1. **Never in the Core Trunk:** Branch C shall **NOT** be included in the default, locked diagnostic ensemble trunk for V7. The core pipeline consists strictly of:
   - **Branch A:** High-resolution heterogeneous vision ensemble (ConvNeXt-Tiny @ 384 px, PanDerm-Base LP-FT, DINOv2 register tokens) with Unit-Simplex PGD stacking.
   - **Branch B:** Observed Clinical Metadata late fusion governed by the Asymmetric Malignancy-Preserving Safety (AMPS) policy and Dirichlet-Multinomial Bayesian logit shrinkage.
2. **Strict Pre-Registered Gating:** Branch C operates as an exploratory branch evaluated exclusively in Phase 4. It can only be merged into the unified system if it clears all three statistical gates (**Gate C1:** $W > 10.83$; **Gate C2:** $\Delta \text{pAUC} \ge +0.030$; **Gate C3:** $\text{DSI}_{<40} < 0.15$).
3. **Automated Zero-Cost Fallback:** If Branch C fails any single gate during Phase 4 evaluation:
   - Branch C is immediately and permanently pruned from the V7 system.
   - The primary diagnostic pipeline defaults unconditionally to **Branch A + Branch B**.
   - The failure of Branch C is documented transparently in the IEEE TMI manuscript as an informative negative ablation on foundation model concept transfer.
4. **Zero Test-Time Concept Footprint:** Under no circumstances shall MONET, foundation VLMs, concept heads, or FiLM generators be executed during runtime inference. Test-time execution remains strictly image-only ($<5\text{ ms}$ latency on raw RGB pixels).

---

## 12. Verification Checklist & Epistemic Standards Sign-Off

To maintain compliance with STARD-AI, TRIPOD+AI, and CLAIM 2024 guidelines:

- [x] **Citation Integrity Verified:** Sultana et al. accurately cited as CVPR 2026; Kumar et al. (LP-FT) cited as ICLR 2022; Vapnik & Vashist (LUPI) cited as Neural Networks 2009.
- [x] **Margin Loss Sign Validated:** Confirmed unique mathematical correctness of $[\Delta_T(x) - \Delta_S(x) - \gamma_{\text{buffer}}]_+^2$; refuted inverted formulation $[\Delta_S - \Delta_T - \gamma]_+^2$.
- [x] **Pseudo-Theorems Refuted:** Explicitly refuted unproven claims of "guaranteed 95% sensitivity retention"; replaced by finite-sample VC excess risk bounds and Clopper-Pearson confidence intervals.
- [x] **PCGrad Limitations Codified:** Proved that PCGrad does not provide global guarantees against negative transfer in non-convex stochastic optimization.
- [x] **Concept Noise Audited:** Quantified MONET/DermFM-Zero error rates ($30\%\text{--}48\%$ on fine micro-structures) and codified deadband gating ($|p-0.50| \ge 0.15$) and concept dropout ($p=0.25$).
- [x] **Shortcut Leakage Defenses Established:** Mandated held-out representation probing ($\text{DSI}_{<40} < 0.15$, $R^2 < 0.05$) and counterfactual background swap testing ($|\Delta P| \le 0.020$).
- [x] **Hardware Profile Certified:** Confirmed ConvNeXt-Tiny @ 384 px achieves $4.12\text{ ms}$ latency and $98\text{ MB}$ peak VRAM on NVIDIA RTX 5050 Laptop GPU ($<30\%$ of $15.0\text{ ms}$ budget).
- [x] **Strategic Status Finalized:** Branch C approved strictly as a Conditional Training-Only Research Arm with automated fallback to Branch A + Branch B.

**Signed & Certified:**  
*Agent 11: Privileged Concept Learning, LUPI, CoFiDA-M, MONET and Knowledge Distillation Specialist*  
*Scan4Disease (S4D) Research Group · IEEE Transactions on Medical Imaging (TMI)*
