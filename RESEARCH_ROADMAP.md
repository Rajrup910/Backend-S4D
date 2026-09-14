# Research Roadmap: Multimodal & Cross-Paradigmatic Ensembling for Safe Skin-Lesion Classification

## Executive Objective

Maximize held-out **Macro-F1, Balanced Accuracy, and Escalation Sensitivity** on the 7-class HAM10000 and 6-class PAD-UFES-20 dermatology tasks, systematically advancing beyond individual CNN baselines to an **award-winning, publication-ready research paper** targeting top-tier venues (*IEEE TMI / Medical Image Analysis (MedIA) / MICCAI / Lancet Digital Health*).

### Methodological Hygiene (Non-Negotiable Core Rules)
1. **Lesion-Grouped Splitting**: All splits are partitioned strictly by `lesion_id` (`assert_no_leakage`) so multiple images of the same lesion never cross train/val/test boundaries.
2. **Leak-Free Parameter Fitting**: Every ensemble weight, scaling temperature, decision threshold, and stacking meta-learner is fit strictly on **Out-of-Fold (OOF)** cross-validation or the **validation split**. The **held-out test split is evaluated exactly once per experiment**.
3. **Clinical Priority Over Raw Accuracy**: Accuracy is reported for completeness but never used as a selection criterion due to extreme class imbalance (~67% `nv` benign nevi).

---

## 1. Unified Metric Suite

All single models, test-time transformations, and ensemble configurations are evaluated through the identical entry point (`ml/evaluation/metrics.py::compute_metrics`):

| Category | Metrics Reported | Clinical / Methodological Rationale |
|---|---|---|
| **Primary Headline** | **Macro-F1**, **Balanced Accuracy** | Equal-weight performance across all 7 classes, penalizing majority-class bias. |
| **Clinical Safety** | **Escalation Sensitivity**, **Missed Serious Cases**, **Malignant Recall (`mel`, `bcc`)** | Screening safety: missing a melanoma or basal cell carcinoma carries severe clinical harm. |
| **Discrimination** | **Macro ROC-AUC (OvR)**, **Macro PR-AUC / Average Precision** | Threshold-independent ranking capability on rare versus common classes. |
| **Calibration** | **Expected Calibration Error (ECE)**, **Max Calibration Error (MCE)**, **Reliability Bins** | Alignment between predicted confidence scores and empirical accuracy. |
| **Statistical Rigor** | **Bootstrap 95% Confidence Intervals (N=1,000)**, **McNemar's Paired Test**, **DeLong's Test** | Proving statistical significance ($p < 0.05, p < 0.01$) against the deployed ResNet-50 baseline. |
| **Ensemble Diversity** | **Yule's Q-Statistic**, **Pairwise Disagreement Rate ($D$)**, **Double-Fault Ratio ($DF$)** | Quantifying statistical orthogonality and complementary error distributions across architectures. |
| **Clinical Utility** | **Net Benefit (Decision Curve Analysis - DCA)**, **Cost-Penalized Loss** | Assessing real-world utility across intervention probability thresholds ($p_t \in [0.01, 0.50]$). |
| **Selective Prediction** | **AURC**, **Excess-AURC (E-AURC)**, **Selective Risk @ Coverage**, **Selective-Accuracy-Constraint (SAC) coverage** | Quality of the abstention ranking — how safely the system defers its least-certain cases to a human. |
| **Conformal Guarantees** | **Marginal & class-conditional (Mondrian) coverage**, **Average prediction-set size** | Distribution-free, finite-sample guarantee that the true class lies in the emitted set at $1-\alpha$. |
| **Fairness / Equity** | **Per-subgroup sensitivity gap**, **Equalized-odds gap**, **Subgroup ECE** (bootstrapped CIs, Holm–Bonferroni corrected) | Equitable safety across Fitzpatrick skin types and acquisition sites; corrected for multiple comparisons. |

---

## 2. Research Execution Campaign (Phases 0 — 5)

### Phase 0 — Baseline Calibration & Dermatological Preprocessing Standardization
- **Environment & Checkpoint Validation**:
  - Verify CUDA sm_120 kernel execution and dependency graph (`scripts/verify_env.py`).
  - Run evaluation across all 6 baseline checkpoints on both `val` and `test` splits:
    - `convnext_tiny_best.HAM-only.pt`
    - `efficientnet_b0_best.HAM-only.pt`
    - `convnext_small_best.HAM-only.pt`
    - `resnet50_best.HAM-only.pt`
    - `densenet121_best.HAM-only.pt`
    - `efficientnet_b3_best.HAM-only.pt`
  - Output standardized prediction matrices (`research/predictions/<arch>_<split>.csv`) containing image IDs, true labels, predicted classes, raw logits, and calibrated probabilities.
- **Dermatology-Specific Preprocessing Pipeline**:
  - Implement **Shades of Grey (SoG) / Grey World Color Constancy** to normalize illumination variations across clinical collection centers.
  - Implement **DullRazor** morphological filtering to eliminate dark hair occlusions and skin marker artifacts without blurring pigment lesions.
  - Formulate an ablation study documenting metric shifts from raw RGB versus preprocessed imagery.

### Phase 1 — Out-of-Fold (OOF) Ensembling & Diversity Quantification
*Inputs: Aligned probability matrices across the 6 frozen backbones.*
- **Preventing Meta-Overfitting**:
  - Implement 5-Fold Stratified Lesion-Grouped Cross-Validation to generate unbiased Out-of-Fold (OOF) predictions for meta-learners.
- **Ensemble Algorithms**:
  1. **Uniform Soft-Voting**: Logit-space and probability-space geometric/arithmetic averaging.
  2. **Rank Averaging**: Class-wise rank transformation to mitigate model-specific confidence over-calibration.
  3. **Nelder-Mead Simplex Optimization**: Constrained optimization of architecture weight vectors $\mathbf{w} \in \Delta^{K-1}$ directly maximizing validation Macro-F1:
     $$\mathbf{w}^* = \arg\max_{\mathbf{w}} \text{Macro-F1}\left(\sum_{k=1}^K w_k \hat{\mathbf{P}}_k, \mathbf{y}_{\text{val}}\right)$$
  4. **Non-Negative Stacking Meta-Learner**: ElasticNet / LightGBM / Ridge classifier trained on OOF logit vectors with non-negativity constraints ($w_k \ge 0$).
  5. **Caruana Greedy Selection**: Iterative ensemble selection with replacement, tracking performance curves as ensemble size grows from $k=1$ to $k=20$.
- **Diversity Audit**:
  - Compute pairwise error correlation matrices, Yule's Q-statistic, and disagreement rates to prove complementary decision boundaries between ConvNeXt (large kernel), DenseNet (feature reuse), and EfficientNet (compound scaled).
- **Deliverables**: Log all validation and held-out test runs in `research/experiments.csv`; compute bootstrap 95% CIs and McNemar's $p$-values vs. single best ConvNeXt-Tiny.

### Phase 2 — Multi-Scale TTA, Dirichlet Calibration & Cost-Sensitive Risk Thresholds
- **Multi-Scale Test-Time Augmentation (TTA)**:
  - 8-fold dihedral transformations (horizontal flip, vertical flip, $90^\circ, 180^\circ, 270^\circ$ rotations) combined with multi-scale crops ($s \in \{0.9, 1.0, 1.1\}$).
  - Implement uncertainty-weighted pooling: down-weighting augmented views with high predictive entropy.
- **Multi-Class Probability Calibration**:
  - Fit **Dirichlet Calibration** with L2 regularization and **Matrix Temperature Scaling** on validation OOF predictions to eliminate class-conditional confidence distortions.
  - Generate 15-bin Reliability Diagrams and ECE tracking across all methods.
- **Cost-Sensitive Asymmetric Risk Thresholding**:
  - Clinically, missing a malignant lesion (`mel`, `bcc`) has vastly higher cost than false alarms on benign nevi (`nv`).
  - Formulate a clinical loss matrix $C_{ij}$ representing the penalty of predicting class $j$ when the true diagnosis is class $i$:
    $$C = \begin{pmatrix} 0 & 1 & 1 & 1 & 1 & 1 & 1 \\ 5 & 0 & 1 & 1 & 5 & 1 & 1 \\ \dots & \dots & \dots & \dots & \dots & \dots & \dots \\ 10 & 5 & 1 & 1 & 0 & 1 & 1 \end{pmatrix}$$
  - Optimize per-class decision thresholds $\boldsymbol{\theta} \in \mathbb{R}^C$ on validation data to minimize clinical cost while maintaining specificity $\ge 85\%$.
- **Decision Curve Analysis (DCA)**:
  - Evaluate Net Clinical Benefit across decision thresholds $p_t \in [0.01, 0.50]$:
    $$\text{Net Benefit}(p_t) = \frac{\text{True Positives}}{N} - \frac{\text{False Positives}}{N} \left(\frac{p_t}{1 - p_t}\right)$$
  - Plot Net Benefit curves comparing our optimized ensemble against "Treat All", "Treat None", and single models.

### Phase 3 — Deep Retraining: Vision Transformers, Advanced Margin Losses & Multimodal Fusion
- **Introducing Transformer Paradigms via `timm`**:
  - Fine-tune modern Vision Transformers: **Swin Transformer V2** (`swinv2_tiny_window16_256`) and **MaxViT / EVA-02** to introduce self-attention representations that complement convolutional inductive bias.
- **Class-Imbalanced Margin Loss Functions**:
  - Replace standard cross-entropy with **LDAM-DRW** (Label-Distribution-Aware Margin with Deferred Re-Weighting; Cao et al., NeurIPS 2019):
    $$\mathcal{L}_{\text{LDAM}}(x, y) = -\log \frac{e^{z_y - \Delta_y}}{e^{z_y - \Delta_y} + \sum_{j \ne y} e^{z_j}}, \quad \Delta_j = \frac{C}{n_j^{1/4}}$$
  - Implement **Asymmetric Loss (ASL)** (Ridnik et al., ICCV 2021) to dynamically suppress easy negative mole gradients without discarding rare lesion signals.
- **Multimodal Tabular Clinical Metadata Fusion**:
  - Extract patient metadata present in `ml/data/manifest.csv`: `age` (normalized scalar), `sex` (one-hot), and `localization` (15 anatomical sites, one-hot).
  - Implement a hybrid network:
    - Vision branch: Image feature vector from backbone $\mathbf{f}_v \in \mathbb{R}^{D_v}$.
    - Tabular branch: Metadata projection MLP $\mathbf{f}_m \in \mathbb{R}^{D_m}$.
    - Cross-attention or gated bilinear fusion layer: $\mathbf{f}_{\text{fused}} = \text{LayerNorm}(\mathbf{f}_v \oplus \mathbf{f}_m + \text{CrossAttn}(\mathbf{f}_v, \mathbf{f}_m))$.
  - Quantify performance delta: Image-only vs. Multimodal Image+Metadata.
- **High-Resolution Fine-Tuning**:
  - Train top backbones at $384 \times 384$ input resolution to preserve micro-architectural pigment networks and atypical vascular patterns.

### Phase 4 — Cross-Domain Adaptation (HAM $\leftrightarrow$ PAD) & Safe Selective Classification
- **Domain Adaptation Study (Dermoscopy vs. Clinical Smartphone)**:
  - Benchmark domain transfer between HAM10000 (dermoscopy) and PAD-UFES-20 (clinical smartphone).
  - Quantify the **Warm-Start Representation Transfer**:
    - ImageNet $\to$ PAD (fresh) vs. ImageNet $\to$ HAM $\to$ PAD (warm).
    - Analyze layer-wise feature alignment and Centered Kernel Alignment (CKA) between representations.
- **Selective Classification (Uncertainty-Guided Rejection / Abstention)**:
  - Clinical safety requirement: an automated screening system must abstain when uncertain.
  - Implement dual uncertainty estimation:
    1. *Predictive Entropy*: $\mathcal{H}(p) = -\sum_c p_c \log p_c$.
    2. *Mahalanobis Feature Distance*: Distance of latent embedding from class centroid representations.
  - Generate **Risk-Coverage Curves**: evaluate Macro-F1, accuracy, and missed serious cases as the model is permitted to abstain on the top 5%, 10%, 15%, and 20% most uncertain cases and refer them for in-person biopsy.
- **Demographic & Skin-Tone Fairness Evaluation**:
  - Partition PAD-UFES-20 test predictions by Fitzpatrick skin type (Types I through VI).
  - Assess equalized odds, demographic parity, and per-skin-type sensitivity to ensure equitable algorithmic safety across patient demographics.

### Phase 5 — Full Ablation Synthesis, Statistical Inference & Publication Package
- **Comprehensive Ablation Study**:
  - Quantify incremental gains across every component:
    1. Baseline ResNet-50
    2. Best Single Model (ConvNeXt-Tiny)
    3. + Color Constancy Preprocessing (Shades of Grey)
    4. + Vision Transformer Inclusion (Swin-V2)
    5. + Multimodal Metadata Fusion (Age/Sex/Site)
    6. + Out-of-Fold Stacking Ensemble
    7. + Test-Time Augmentation (TTA)
    8. + Dirichlet Calibration & Cost-Sensitive Thresholds
    9. + Selective Abstention (Top 10% uncertain deferred)
- **Statistical Significance & Inferential Proof**:
  - Compute 1,000-sample stratified bootstrap 95% confidence intervals for all final figures.
  - Run paired McNemar tests for classification disagreements ($p < 0.001$).
  - Execute DeLong tests comparing multi-class ROC-AUC distributions.
- **Publication Manuscript Preparation (`paper/`)**:
  - `manuscript.tex`: Full IEEE/Springer template manuscript with complete theoretical formulations, related work review, and clinical discussion.
  - Camera-ready vector figures:
    - Figure 1: Architectural diagram of the multimodal cross-paradigmatic pipeline.
    - Figure 2: Reliability calibration diagrams before and after Dirichlet scaling.
    - Figure 3: Decision Curve Analysis (Net Clinical Benefit vs. Threshold Probability).
    - Figure 4: Risk-Coverage Selective Prediction curves.
    - Figure 5: Grad-CAM++ vs. Dermatological ABCDE feature attribution maps.
  - Automated generation scripts ensuring 100% of paper tables and plots compile directly from `results/` artifacts.

---

## 2A. Advanced Method Extensions (Directly Targeting the Six Accuracy Bottlenecks)

Each subsection names the limitation, the *additional* techniques layered on top of the core phases above, and where they slot in.

### E1 — Hardening the Ensemble Against Validation Overfitting *(extends Phase 1)*
- **Nested cross-validation**: an inner CV loop selects meta-learner hyperparameters (regularization strength, tree depth) so the reported OOF score is not optimistically biased by tuning on the same folds used to score it.
- **Repeated multi-seed grouped OOF** (e.g. 3 repeats × 5 folds): average OOF probabilities across seeds to damp fold-assignment variance; report the stacker as mean ± std, not a single point.
- **Complexity control**: prefer strongly regularized, low-capacity meta-learners (non-negative Ridge / ElasticNet / shallow LightGBM with early stopping). On only six inputs, a convex-combination weight vector is often the safest, most reproducible stacker.
- **Generalization-gap diagnostic**: log `OOF Macro-F1 − test Macro-F1` for every stacker; a gap beyond ~0.02 flags meta-overfitting and vetoes that configuration *before* the single test-set read.

### E2 — Dermatological Image Standardization Beyond Color/Hair *(extends Phase 0)*
- **Lesion ROI localization & vignette removal**: detect and crop the dark circular dermoscope frame and black borders (Otsu + largest connected component / Hough circle), then center on the lesion so the network is never fed corner artifacts.
- **CLAHE** (Contrast-Limited Adaptive Histogram Equalization) on the LAB-L channel to normalize local contrast without shifting hue (color is diagnostic).
- **Artifact inpainting**: remove rulers, ink markings, and gel bubbles via mask + Telea / Navier–Stokes inpainting, complementing DullRazor hair removal.
- **Color-constancy family & selection**: compare Grey-World, max-RGB, and Shades-of-Grey (Minkowski norm $p \in \{1,6,\infty\}$; Finlayson & Trezzi 2004) and pick the per-metric best on validation.
- **Preprocessing ablation matrix**: raw → +color-constancy → +hair/artifact removal → +ROI crop, each scored, so gain is attributed to each step rather than claimed wholesale.

### E3 — Maximizing Inductive-Bias Diversity *(extends Phase 3)*
- **Transformer & hybrid backbones**: Swin-V2, MaxViT, EVA-02 (already planned) plus **CoAtNet** (conv+attention hybrid; Dai et al. 2021) and **DeiT-III** (Touvron et al. 2022) for a pure-ViT contrast.
- **Self-supervised / foundation features**: a frozen **DINOv2** (Oquab et al. 2023) encoder with a lightweight linear / kNN head as an architecturally-orthogonal ensemble member — high diversity at near-zero training cost.
- **Diversity-driven selection**: feed the Yule's-Q / double-fault scores from Phase 1 back into backbone choice, explicitly picking members that are accurate *and* de-correlated, not merely the top-k by accuracy.

### E4 — Richer Multimodal Metadata Fusion *(extends Phase 3)*
- **FiLM conditioning** (Perez et al. 2018): metadata generates per-channel affine (γ, β) modulations of visual feature maps — a stronger interaction than late concatenation.
- **Tabular transformer branch**: **FT-Transformer / TabTransformer** (Gorishniy et al. 2021) over `age`, `sex`, `localization` in place of a plain MLP.
- **Metadata-only baseline**: quantify the standalone signal (age × anatomical-site priors on malignancy) so the paper isolates true multimodal lift rather than a re-learned prior.
- **Missing-metadata robustness**: learned "missing" embeddings + an inference-time sensitivity test (drop each field), since real records are incomplete.

### E5 — Loss Landscape for Rare Lethal Classes *(extends Phase 3)*
Beyond LDAM-DRW and ASL:
- **Logit Adjustment** (Menon et al. 2021) / **Balanced Softmax** (Ren et al. 2020): Bayes-consistent prior correction for long-tailed labels.
- **Class-Balanced Focal** (Cui et al. 2019 — effective number extended to focal) and **Seesaw Loss** (Wang et al. 2021).
- **Decoupled representation/classifier learning** — **cRT / LWS / τ-normalization** (Kang et al. 2020): learn features with instance sampling, then re-balance only the classifier; consistently strong on long-tailed benchmarks.
- **Regularizing augmentations**: mixup (Zhang 2018), CutMix (Yun 2019), RandAugment (Cubuk 2020), tuned to preserve diagnostic color.
- **Selection rule**: each loss is a training-time swap judged on validation Macro-F1 and minority-class (`df`, `vasc`, `akiec`) recall; the winner feeds the ensemble.

### E6 — Clinical Safety & Trust Layer *(extends Phases 2 & 4)*
- **Conformal prediction** (split / APS / RAPS; Angelopoulos & Bates 2021) with **Mondrian (class-conditional) calibration**: a distribution-free, finite-sample guarantee that the true label is in the prediction set at $1-\alpha$. Report marginal *and* per-class coverage plus average set size; the abstention rule becomes "defer when the set is not a singleton."
- **Principled selective metrics**: **AURC / Excess-AURC** (Geifman & El-Yaniv 2017) and **SAC coverage**, alongside the risk-coverage curves already planned.
- **Epistemic uncertainty from the ensemble**: inter-model probability variance (a deep-ensemble signal) as a second abstention axis beyond entropy and Mahalanobis distance.
- **Subgroup calibration & corrected fairness testing**: per-Fitzpatrick / per-site ECE and sensitivity gaps with bootstrapped CIs and **Holm–Bonferroni** correction across the many subgroup comparisons.

---

## 2B. Reproducibility, Determinism & Compute Budget

- **Determinism**: fixed seeds, `torch.use_deterministic_algorithms(True)` where feasible, cuDNN deterministic flags; every experiment records its effective config.
- **Environment & data locks**: a `pip freeze` snapshot per phase; SHA-256 checksums of the manifest and split CSVs verified before each run so "same split" is provable, not assumed.
- **Single-read discipline**: a guarded ledger records each test-split evaluation; anything fit on data is fit on OOF/validation only.
- **Efficiency reporting**: parameters, single-image latency (already emitted by `evaluate.py`), and training GPU-hours per model — accuracy *per unit compute* strengthens the EfficientNet-B0 (4 M-param) narrative.
- **Result provenance**: every number in `README.md` and the manuscript is regenerated by a script from `results/`; the current tables are treated as targets to reproduce, not as final reported values.

---

## 2C. Novelty Positioning & Focus Discipline

- **Headline contribution (what is genuinely new)**: the *integration* of cross-paradigmatic OOF-stacked ensembling **with** conformal-guaranteed selective abstention and cost-sensitive thresholding, validated across the dermoscopy→smartphone domain gap and audited for skin-tone equity. Each component exists in isolation in the literature; unifying them under one leak-free protocol on this task is the paper's claim.
- **Core vs. appendix (to keep the paper focused)**: a top-venue paper shows a tight main result plus disciplined ablations — not every method at once. Recommended *main-body* pipeline = best preprocessing + cross-paradigm ensemble + calibration + cost-sensitive thresholds + conformal abstention; the remaining losses, backbones, and fusion variants live in an ablation appendix. This avoids the "we tried twenty things, each +0.3%" dilution that reviewers penalize.

---

## 2D. Phase 6 — Post-Manuscript Falsification Programme (V2 · V3, sessions S28–S47)

Phases 0–5 delivered the manuscript. Phase 6 asks the question the manuscript could not answer:
**is the under-40 blind spot fixable at all, or is it intrinsic?** It is deliberately structured as
a falsification programme rather than a search for improvement — each campaign declares hypotheses,
gates and an MCID *before* the runs that test them, and a null is a publishable result.

### Standing constraints (non-negotiable)

- HAM val/test inherited byte-identically from `ml/configs/splits/split_v1.csv`
- `results/test_pass_receipt.json` stays at `n_executions: 2` — **no test read in either campaign**
- The six `*_best.HAM-only.pt` checkpoints stay byte-identical (`research.v2.frozen_checkpoints --check`)
- Every number traceable to `results/`; an interval containing the null is `NOT_CERTIFIED`, never "zero"

### V3 hypothesis register (S40–S47)

| ID | Phase | Hypothesis | Instrument | Verdict |
|:--|:--|:--|:--|:--|
| H1 | A | The prior escalation-head gain survives out-of-sample extraction | `research/v3/oos_probe.py` | ❌ Falsified |
| H2 | B | A better head on the frozen representation recovers the gap | `research/v3/ceiling.py` ($\rho_g$, $\Delta_{\text{head}}$) | ❌ Falsified |
| H3 | B | The representation is age-entangled | `research/v3/probes.py` | ✅ **Certified** |
| H4 | C | Pooling HAM + BCN + MSKCC clears a $+0.03$ Macro-F1 gate | `research/v3/eval_conditions.py` | ❌ Falsified |
| H5 | C | Archive breadth buys zero-shot cross-archive robustness | `research/v3/external_by_cohort.py` | ❌ Falsified |
| H6 | D | Removing the entanglement improves under-40 ranking | `research/v3/eval_d2.py` | ❌ Falsified |

### The four multi-archive conditions (Phase C)

| Condition | Train images | Purpose |
|:--|--:|:--|
| `ham_only` | 6,981 | control — must reproduce the published val figure |
| `ham_mskcc` | 9,010 | sample size, almost no domain breadth |
| `ham_bcn` | 15,396 | rare-class injection, second dermoscopy site |
| `all_three` | 17,425 | the deployable object |

All four validate on the **same** fixed 1,532-image HAM val set, so per-epoch figures are
comparable and no condition selects against a different target.

### Methodological products worth reusing

1. **In-domain vs zero-shot disaggregation** — a pooled external endpoint whose holdout is 80% one
   archive cannot measure robustness. Cells are labelled from the training composition, not by hand.
2. **Noise-floor calibration of an endpoint** — before attributing a subgroup difference to a
   condition, measure how far two models trained on *identical* data drift on that endpoint. Here
   the same-data spread (0.2273) exceeded the between-condition spread (0.1818), retiring the
   endpoint.
3. **Mechanism checks gate endpoint claims** — an intervention that did not move the representation
   cannot be credited with moving an outcome. `eval_d2.py` encodes this as a 2×2 verdict table with
   an explicit `CONFOUNDED` cell.
4. **Reliance $\ne$ invariance** — adversarial removal can stop a head *using* an attribute without
   deleting it. Both halves are asserted so neither can silently drift.

### Outcome

The 0.80 Macro-F1 ceiling was **not** broken (best condition 0.7869, not certified above control).
The contribution type, derived from the verdict pattern rather than chosen in advance, is
**diagnostic and falsificatory**:

> The under-40 escalation gap is **not caused by a removable age shortcut**. The representation is
> certifiably age-entangled, but stripping that entanglement degrades the very subgroup it was
> meant to rescue — the age signal is load-bearing diagnostic signal. No head-level fix exists,
> archive breadth does not help, and the result that motivated three earlier arms was an artifact.

---

## 3. Experiment Registry Schema (`research/experiments.csv`)

Every experimental run appends a structured record:

```csv
timestamp,session,method,split,macro_f1,accuracy,balanced_accuracy,weighted_f1,macro_roc_auc,ece,escalation_sens,missed_serious,p_value_vs_baseline,notes
```

---

## 4. Programme Status

| Phase | Scope | Status |
|:--|:--|:--|
| 0 | Baseline calibration & preprocessing | ✅ Complete — 6 backbones trained and evaluated |
| 1 | OOF ensembling & diversity | ✅ Complete — soft-vote is the only certifiable rung |
| 2 | TTA, Dirichlet calibration, cost-sensitive thresholds, DCA | ✅ Complete |
| 3 | Vision transformers, margin losses, multimodal fusion | ✅ Complete — no arm beats the CNN soft-vote |
| 4 | Cross-domain adaptation, selective classification, fairness, conformal | ✅ Complete |
| 5 | Ablation synthesis, statistical inference, manuscript | ✅ Complete — manuscript + supplementary frozen |
| 6 | Post-manuscript falsification programme (V2 · V3) | ✅ Complete — 5 of 6 hypotheses falsified |

### Open items

1. **The remaining test read is unspent.** V3's gate (winning system beats the control by
   $\ge 0.03$ HAM-val Macro-F1 with a CI excluding zero) was **not met** by any candidate, so S47
   did not spend it. Whether the V3 result justifies spending it is a manuscript question, not a
   pipeline one.
2. **Page budget.** `research/ablation/estimate_pages.py` puts the full manuscript at ~24 pages
   against IEEE TMI's 10; prose is 16.2 of those, so deleting every float still leaves ~18. The
   honest options remain splitting the paper or targeting a venue without a ten-page limit.
3. **Optional — a converged invariance frontier.** Two points exist: $\lambda = 1$ (inert; no cost,
   no benefit) and $\lambda = 3$ (engaged; $-0.1422$ Macro-F1, $-0.1049$ under-40 AUC, but only 6
   epochs and not re-converged). Three full runs (~6 h) would turn "no cheap setting was found"
   into a measured cost curve. Only worth it if the invariance claim becomes load-bearing.
4. **D1 (dual-view input) remains documented and unrefuted**, though Phase C weakened its premise —
   pooling archives produced 0 of 2 zero-shot transfer gains.
