# Methodological Deep Dive: Architectural Reverse-Engineering

**Source Basis:** Manuscript Section III, Appendix A, `ml/` source code, `research/` implementations, and frozen parameter JSONs.

---

## 1. Dataset Architecture & Leak-Free Partitioning

### HAM10000 Cohort Specifications
The primary development cohort is the public HAM10000 collection [M III-A, 2]:
- **Images:** 10,015 dermatoscopic image files ($600 \times 450$ resolution, 24-bit RGB).
- **Physical Lesions:** 7,470 unique lesion identifiers (`lesion_id`).
- **Imaging Modality:** Polarized and non-polarized contact and non-contact dermoscopy.
- **Ground Truth / Reference Standard:**
  - Histopathology (`histo`): 53.3% of the collection (excisional biopsy / dermatopathology).
  - Follow-up examination (`follow_up`): Digital dermoscopy monitoring over time showing stability.
  - Expert consensus (`consensus`): Agreement of 3+ expert dermatologists.
  - In-vivo Confocal Microscopy (`confocal`).

### The Lesion-Level vs Image-Level Split Dilemma
In HAM10000, 2,545 images represent repeat captures of existing lesions (taken from multiple angles, before/after oil application, or across magnifications).
- **Image-Level Splitting (Common Field Error):** A random 70/15/15 image-level split distributes near-duplicate views of the same physical lesion across training and test partitions. Neural networks memorize unique visual artifacts (skin wrinkles, hair follicles, lighting vignetting, anatomical quirks), inflating test Macro-F1 by 3–5 percentage points without learning diagnostic morphology.
- **Lesion-Level Splitting (Strict Protocol):** All partitions in this project are grouped strictly by `lesion_id` using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` [M III-A]:
  - **Train (70%):** 6,981 images across 5,229 lesions.
  - **Validation (15%):** 1,532 images across 1,120 lesions.
  - **Held-Out Test (15%):** 1,502 images across 1,121 lesions.
- **Leakage Assertion:** Every script re-runs `assert_no_leakage()`, verifying:
  $$\text{Lesions}(\text{Train}) \cap \text{Lesions}(\text{Val}) = \emptyset, \quad \text{Lesions}(\text{Train}) \cap \text{Lesions}(\text{Test}) = \emptyset, \quad \text{Lesions}(\text{Val}) \cap \text{Lesions}(\text{Test}) = \emptyset$$

### Dataset Partition Flow
```
10,015 HAM10000 Images (7,470 Physical Lesions)
  │
  ├─► Stratified Grouped Split by lesion_id (70/15/15, Seed 42)
  │     │
  │     ├─► TRAINING SET: 6,981 images (5,229 lesions)
  │     │     │
  │     │     ├─► 5-Fold StratifiedGroupKFold Cross-Validation
  │     │     │     └─► 30 Fold Models (6 archs × 5 folds)
  │     │     │           └─► 6,981 OOF Prediction Rows (Train Set Support)
  │     │     │                 ├─► Rare-Class Conformal Support: DF=71, VASC=99
  │     │     │                 ├─► Under-40 Escalating Positives: 64 cases
  │     │     │                 ├─► Fits Deployed Dirichlet Calibrator
  │     │     │                 └─► Fits Frozen λ Age-Conditional Rule
  │     │     │
  │     │     └─► Full-Train Checkpoints (6 CNNs trained on all 6,981 images)
  │     │
  │     ├─► VALIDATION SET: 1,532 images (1,120 lesions)
  │     │     ├─► Early Stopping / Checkpoint Selection (Macro-F1, patience 8)
  │     │     ├─► Ensembling Method Selection (Uniform soft-vote chosen)
  │     │     ├─► Calibrator Selection (Dirichlet chosen via Val ECE)
  │     │     ├─► Uncertainty Score Selection (Margin chosen via Val AURC)
  │     │     └─► Conformal RAPS Hyperparameters (k_reg=1, lambda=0.20/0.05)
  │     │
  │     └─► HELD-OUT TEST SET: 1,502 images (1,121 lesions)
  │           └─► Pre-registered single pass (results/analysis_plan.json)
  │                 └─► Emits 19 quantities (results/test_pass_receipt.json)
  │
  └─► EXTERNAL ZERO-TARGET EVALUATION COHORTS (Nothing Re-fitted)
        ├─► PAD-UFES-20: 2,106 Smartphone Clinical Photos (Modality Shift Benchmark)
        ├─► BCN-20000: 11,982 ISIC-2019 Dermoscopy Images (Multi-centre Same-modality)
        └─► MSKCC: 2,903 ISIC-2019 Dermoscopy Images (Multi-centre Same-modality)
```

---

## 2. Seven-Class Diagnostic Taxonomy & Actionability Tiers

The dataset spans seven disease entities defined in dermatopathology [M Table I]:

| Code | Disease Entity | Clinical Meaning | Malignancy Tier | Test $n$ | Train $n$ | Clinical Action Required |
|---|---|---|---|---|---|---|
| **AKIEC** | Actinic Keratosis / Intraepithelial Carcinoma | Pre-cancerous squamous dysplastic proliferation | Premalignant* (Tier 2 / Escalating) | 52 (3.5%) | 229 | Dermatologic consultation / cryotherapy / topical 5-FU |
| **BCC** | Basal Cell Carcinoma | Locally invasive, slow-growing non-melanoma skin cancer | Malignant* (Tier 1 / Escalating) | 71 (4.7%) | 358 | Surgical excision / Mohs micrographic surgery |
| **BKL** | Benign Keratosis (Seborrheic keratosis, solar lentigo) | Common benign senile proliferation of keratinocytes | Benign (Tier 3 / Non-escalating) | 167 (11.1%) | 769 | Reassurance / no surgical intervention |
| **DF** | Dermatofibroma | Benign fibrous histiocytoma | Benign (Tier 3 / Non-escalating) | 20 (1.3%) | 81 | Discharge / clinical reassurance |
| **MEL** | Malignant Melanoma | Highly lethal malignancy arising from melanocytes | Malignant* (Tier 1 / Escalating) | 167 (11.1%) | 779 | Urgent wide local excision / sentinel node biopsy |
| **NV** | Melanocytic Nevus | Ordinary mole / benign melanocyte proliferation | Benign (Tier 3 / Non-escalating) | 1,004 (66.8%) | 4,675 | Discharge / routine monitoring |
| **VASC** | Vascular Lesion (Angioma, pyogenic granuloma) | Benign vascular malformation | Benign (Tier 3 / Non-escalating) | 21 (1.4%) | 90 | Discharge / reassure |

### The Escalating Class Set
For binary screening and clinical triage, the seven-class problem is collapsed into the **Escalating Set** $\mathcal{E}$ [M III-B]:
$$\mathcal{E} = \{\text{AKIEC}, \text{BCC}, \text{MEL}\}$$
- Total escalating lesions in test split: $52 + 71 + 167 = 290$ images (19.31% prevalence).
- Total benign lesions in test split: $167 + 20 + 1,004 + 21 = 1,212$ images (80.69% prevalence).

### Why Accuracy is a Clinically Toxic Metric
In HAM10000, `nv` alone accounts for 66.8% of the test split. A pathological dummy classifier that predicts `nv` for every single lesion achieves:
$$\text{Accuracy} = 66.84\%, \quad \text{Macro-F1} = 0.114, \quad \text{Escalation Sensitivity} = 0.00\%$$
It misses 100% of melanomas and non-melanoma skin cancers. Accuracy rewards matching the dominant benign prior while ignoring fatal malignancies.

---

## 3. End-to-End System Architecture Pipeline

The complete inference pipeline reverse-engineered from `manuscript.tex`, `ml/`, and `research/`:

```
                    INPUT DERMOSCOPIC IMAGE (224 × 224 RGB)
                                       │
                                       ▼
                   TEST-TIME AUGMENTATION (TTA) ENGINE
         [24 Deterministic Views: 8 Dihedral Transforms × 3 Scales]
         [Scales: 1.0 (224×224), 0.9 (248 cropped), 1.1 (204 padded)]
                                       │
                                       ▼
                   SIX FROZEN CNN BACKBONES IN PARALLEL
       ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
       ▼          ▼          ▼          ▼          ▼          ▼          ▼
    ResNet-50  DenseNet-  Efficient- Efficient-  ConvNeXt-  ConvNeXt- (Features:
                 121       Net-B0     Net-B3       Tiny      Small    Penultimate
                                                                       d=768)
       └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
                                       │
                                       ▼
                   INDIVIDUAL BACKBONE PROBABILITIES
                 6 Probability Vectors p_m ∈ Δ^6 per view
                                       │
                                       ▼
                   ENTROPY-WEIGHTED VIEW POOLING
                     w_v ∝ exp(-H(p_v) / τ_TTA)
                                       │
                                       ▼
                   UNIFORM SOFT-VOTING ENSEMBLE
                        p_ens = (1/6) ∑_{m=1}^6 p_m
                                       │
                                       ▼
                   POST-HOC DIRICHLET CALIBRATION
                     z_cal = W · ln(p_ens) + b
                     p_cal = softmax(z_cal)
                     (W ∈ R^{7×7}, b ∈ R^7, fitted with L2 on OOF)
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
   DECISION LAYER 1              DECISION LAYER 2              DECISION LAYER 3
  Selective Abstention          Conformal Prediction         Age-Conditional Rule
[Top-two Margin Gate]         [RAPS Bipartite Scheme]         [Post-Processing]
 s(x) = 1 - (p_(1) - p_(2))    Non-conformity score S_i       y_hat = argmax_c [
 If s(x) > τ_10% (0.3855)      Quantiles q_{b,E} per cell      p_cal,c + λ_b 1[c∈E] ]
   ──► DEFER / REFER            ──► EMIT PREDICTION SET        λ_<40=0.26, λ_40-59=0.74,
 Else                           C(x) ⊆ {1,...,7}               λ_60+=0.33
   ──► RETAIN FOR DIAGNOSIS     P(y ∈ C(x)) ≥ 1 - α           ──► TRIAGE / BIOPSY
```

---

## 4. Detailed Component Specifications

### A. Backbone Architectures & Training Protocol
Six CNN backbones initialized from ImageNet-1k pretrained weights via `timm` [M App. A-C]:
1. **ResNet-50:** Residual bottleneck blocks, 25.6M parameters.
2. **DenseNet-121:** Densely connected feature concatenation, 8.0M parameters.
3. **EfficientNet-B0:** Compound scaling baseline, MBConv blocks, 5.3M parameters.
4. **EfficientNet-B3:** Higher-capacity compound scaled network, 12.2M parameters.
5. **ConvNeXt-Tiny:** Modernized pure-CNN with 7×7 depthwise convolutions, inverted bottlenecks, 28.6M parameters.
6. **ConvNeXt-Small:** Scaled modern CNN, 50.2M parameters.

#### Training Recipe (Pinned in `ml/configs/training_config.yaml`):
- **Input Resolution:** $224 \times 224$ pixels.
- **Two-Stage Schedule:**
  - *Stage 1 (Head Warmup):* 3 epochs with CNN backbone frozen; classifier head learning rate $= 10^{-3}$.
  - *Stage 2 (Fine-Tuning):* Full model fine-tuning at backbone learning rate $= 10^{-4}$.
- **Optimizer:** AdamW (weight decay $= 10^{-4}$, $\beta_1 = 0.9, \beta_2 = 0.999$).
- **Learning Rate Schedule:** Cosine annealing decaying to $10^{-6}$.
- **Batch Size:** 32.
- **Regularization:** Label smoothing $\epsilon = 0.05$, gradient clipping at norm $5.0$, mixed precision (AMP fp16).
- **Early Stopping:** Monitored on Validation Macro-F1 with patience of 8 epochs (maximum 30 epochs).
- **Class Imbalance Loss:** Effective-number weighting (Cui et al., CVPR 2019) with hyperparameter $\beta = 0.999$:
  $$w_c = \frac{1 - \beta}{1 - \beta^{N_c}}, \quad \mathcal{L}_{\text{CE}} = - \sum_{c=1}^7 w_c \cdot y_c \ln(p_c)$$
  Where $N_c$ is the training sample count of class $c$.
- **Augmentation Policy:** Deliberately conservative: Random resized crops (scale [0.8, 1.0]), horizontal and vertical flips, $\pm 20^\circ$ random rotations, and mild color jitter (brightness 0.15, contrast 0.15, saturation 0.10, hue 0.02). Heavy color distortion was excluded because pigmentation hue is diagnostic.

### B. Ensembling & Method Selection
The ensemble combines the 6 CNN probability vectors via uniform arithmetic soft-voting [M III-D]:
$$p_{\text{ens}}(x) = \frac{1}{6} \sum_{m=1}^6 p_m(x)$$
- **Why Uniform Soft-Voting?** Evaluated against Nelder-Mead simplex weights, rank-averaging, geometric voting, Caruana greedy forward selection with replacement, and non-negative ridge stacking on out-of-fold validation logits.
- **Nelder-Mead Convergence:** Nelder-Mead optimization initialized from random points converged to $[0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]$, corroborating uniform weighting.
- **Rejection of Ridge Stacking:** Non-negative ridge stacking achieved test Macro-F1 0.7815 (the highest test number observed), but had an out-of-fold validation score of 0.7583 (vs uniform soft-vote validation 0.7911). Reporting ridge stacking would violate leak-free discipline by selecting on test performance.

### C. 24-View Test-Time Augmentation (TTA)
- **Transformations:** Evaluates each image under the 8 discrete dihedral transformations of the square ($D_4$: 4 rotations $0^\circ, 90^\circ, 180^\circ, 270^\circ \times 2$ horizontal reflections) across 3 scale crops ($1.0\times, 0.9\times, 1.1\times$) $= 24$ deterministic views [M App. A-E].
- **Deterministic Transforms:** Uses only evaluation-time operations (resize, center crop, normalize). Zero training-time stochastic noise enters inference.
- **Entropy-Weighted Pooling:** Views with lower Shannon entropy $H(p_v) = -\sum_c p_{vc} \ln p_{vc}$ (sharper diagnostic commitment) are given higher pooling weight:
  $$w_v = \frac{\exp(-H(p_v)/\tau_{\text{TTA}})}{\sum_{u=1}^{24} \exp(-H(p_u)/\tau_{\text{TTA}})}, \quad p_{\text{TTA}}(x) = \sum_{v=1}^{24} w_v p_v(x)$$

### D. Multi-Class Dirichlet Calibration
- **Mathematical Formulation:** Given uncalibrated probability vector $p \in \Delta^6$, Dirichlet calibration fits an affine transformation on log-probabilities [M IV-A, 15]:
  $$\ln \tilde{p} = W \ln p + b, \quad p_{\text{cal}} = \text{softmax}(W \ln p + b)$$
  Where $W \in \mathbb{R}^{7 \times 7}$ and $b \in \mathbb{R}^7$, regularized by $L_2$ penalty on off-diagonal entries:
  $$\Omega(W) = \lambda_{\text{cal}} \sum_{i \neq j} W_{ij}^2$$
- **Why Temperature Scaling Failed:** Single-network literature assumes models are *overconfident*, where temperature $T > 1$ uniformly softens probabilities. Soft-voting six models produces *underconfidence* (mean confidence 0.7048 vs accuracy 0.8609). Because underconfidence varies across classes (worse on minority classes), single-scalar temperature scaling cannot correct the distortion. Dirichlet calibration allows full class-conditional re-alignment.

### E. Selective Classification (Abstention Gate)
- **Policy Score:** Top-two Probability Margin [M App. A-F]:
  $$s(x) = 1 - \left( p_{(1)}(x) - p_{(2)}(x) \right)$$
  Where $p_{(1)}$ is the maximum predicted probability and $p_{(2)}$ is the runner-up probability. $s(x) \in [0, 1]$ acts as an uncertainty score (larger $=$ less certain).
- **Abstention Rule:** A case is deferred to specialist review if $s(x) > \tau_q$, where $\tau_q$ is the empirical $q$-th quantile computed on fitting data.
- **Why Mahalanobis Distance Failed In-Distribution:** Penultimate-layer Mahalanobis distance measures feature-space distance from training cluster centroids. Test images from HAM10000 are in-distribution, lying within the feature manifold. The errors are subtle boundary ambiguities, which probability margins capture, but feature density does not.

### F. Split Conformal Prediction
Conformal prediction constructs a prediction set $C(x) \subseteq \{1, \dots, 7\}$ guaranteeing finite-sample coverage at error level $\alpha \in (0, 1)$ [M III-F, 26]:
$$P(Y_{n+1} \in C(X_{n+1})) \ge 1 - \alpha$$
- **Non-Conformity Scores Evaluated:**
  1. *LAC (Least Ambiguous Classifier):* $S_i = 1 - p(y_i \mid x_i)$.
  2. *APS (Adaptive Prediction Sets):* $S_i = \sum_{c: p_c \ge p_{y_i}} p_c(x_i)$.
  3. *RAPS (Regularized APS):* $S_i = \sum_{c: p_c \ge p_{y_i}} p_c(x_i) + \lambda_{\text{reg}} \max(0, \text{rank}(y_i) - k_{\text{reg}})$. Penalizes large sets.
- **Partitioning Schemes:**
  - *Marginal:* Single global quantile $\hat{q}$ across all calibration data.
  - *Mondrian Class-Conditional:* Separate quantile $\hat{q}_c$ for each of the 7 diagnostic classes. Requires $\lceil (n_c + 1)(1 - \alpha) \rceil \le n_c$.
  - *Equalized Bipartite Scheme:* Partitions calibration data into 6 cells by the Cartesian product of age band and escalation requirement:
    $$\text{Cell}(x_i) = \left( b(x_i), \mathbf{1}[y_i \in \mathcal{E}] \right) \in \{<40, 40\text{--}59, 60+\} \times \{0, 1\}$$
    Conditions coverage jointly on patient age and disease seriousness.

### G. Out-of-Fold (OOF) Training Protocol
To prevent overloading the 1,532-image validation split, 30 fold models were trained under 5-fold cross-validation over the 6,981 training images [M App. A-A]:
- **Yield:** 6,981 out-of-fold predictions.
- Rare-class sample size increased: DF from 24 to 71; VASC from 22 to 99.
- Under-40 escalating cases increased: from 22 to 64.
- **Stacking Mismatch:** OOF fold models were trained on 80% data ($\approx 5,585$ images), making them slightly less confident than full-train checkpoints. Calibrators fitted on OOF are slightly over-sharp when applied to full-train checkpoints (Rung A7-oof Macro-F1 0.7871 vs A7-val 0.8047).

### H. The Age-Conditional Escalation Rule
Post-processing decision rule modifying the argmax classification [M III-C, Eq. (1)]:
$$\hat{y}(x) = \arg\max_{c \in \{1,\dots,7\}} \left( p_c(x) + \lambda_{b(x)} \mathbf{1}[c \in \mathcal{E}] \right)$$
- $\mathcal{E} = \{\text{AKIEC}, \text{BCC}, \text{MEL}\}$.
- $b(x) \in \{<40, 40\text{--}59, 60+, \text{unknown}\}$.
- Equivalent to applying per-class thresholds $\theta_c = -\lambda_{b(x)}$ for $c \in \mathcal{E}$ and $\theta_c = 0$ for benign classes.
- **Fitting Optimization:** Exhaustive search over a 61-point grid $\lambda \in [0.0, 1.2]$ to minimize expected clinical cost under a per-band specificity floor of 0.85:
  $$\min_\lambda \mathbb{E}[\text{Cost}(\lambda)] \quad \text{s.t.} \quad \text{Specificity}_{\text{escalate}}(\lambda) \ge 0.85$$
- **Positives Gate:** A band must contain $\ge 30$ true escalating cases to receive its own $\lambda$ (under-40 has 64 in OOF; falls back to pooled $\lambda = 0.65$ if below gate).
- **Frozen Values:** $\lambda_{<40} = 0.26$ (95% CI: [0.00, 0.61]), $\lambda_{40-59} = 0.74$, $\lambda_{60+} = 0.33$, $\lambda_{\text{pooled}} = 0.65$.

### I. Number Needed to Biopsy (NNB) & Clinical Cost Formulation
NNB is the reciprocal of positive predictive value:
$$\text{NNB} = \frac{\text{TP} + \text{FP}}{\text{TP}} = 1 + \frac{\text{FP}}{\text{TP}}$$
- **Prevalence-Reweighted NNB:** HAM10000 has 19.3% escalating prevalence on test, whereas primary-care screening prevalence $\pi$ is 1% to 5%. Reporting unadjusted NNB gives false optimism. Importance-weighting benign false positives to reference prevalence $\pi = 0.03$:
  $$w = \frac{1 - \pi}{\pi} \cdot \frac{p}{1 - p} = \frac{0.97}{0.03} \cdot \frac{0.1931}{0.8069} \approx 7.74$$
  $$\text{NNB}_\pi = 1 + w \frac{\text{FP}}{\text{TP}}$$
