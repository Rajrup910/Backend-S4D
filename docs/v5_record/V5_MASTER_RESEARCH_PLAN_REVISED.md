# V5 Master Research Plan — Biology-Grounded Under-40 Rescue and Macro-F1 Improvement

**Repository:** https://github.com/Rajrup910/Backend-S4D  
**Revision date:** September 18, 2026  
**Status:** Pre-registration and execution blueprint for V5  
**Target venues:** IEEE Transactions on Medical Imaging (TMI) / Medical Image Analysis (MedIA)

---

## 0. Purpose of This Revision

This document replaces the earlier V5 draft as the execution blueprint.

The revision makes three changes that are essential for scientific correctness:

1. It separates **what V1–V4 actually implemented and measured** from what V5 proposes.
2. It removes or demotes ideas that the repository has already shown to be null, harmful, incomplete, or invalid.
3. It adds biology-to-computer-vision experiments that attack the remaining bottleneck at the **representation and morphology level**, rather than repeatedly manipulating the existing posterior.

The project rule remains:

> **Do not spend GPU time on a method whose mechanism has already been falsified by an equivalent experiment unless the new experiment changes the mechanism in a scientifically meaningful way.**

---

# 1. Current Research State

## 1.1 What the completed programme has established

The project has investigated:

- lesion-grouped classification;
- six heterogeneous CNN backbones;
- uniform soft-voting;
- 24-view TTA;
- probability calibration;
- subgroup calibration;
- conformal prediction;
- age-conditional decision rules;
- selective abstention;
- multi-archive/domain probes;
- metadata fusion;
- specialist-head probes;
- foundation-model probes;
- recipe-level training ablations;
- V4 multi-centre analysis;
- K-fold cross-fitting;
- target-side referral recalibration;
- smartphone modality admissibility.

The core in-domain V1/V2 system established a strong aggregate classification baseline:

- ConvNeXt-Tiny single-model Macro-F1: **0.7459**
- six-CNN uniform soft-vote: **0.7718**
- +24-view TTA: **0.7859**
- + Dirichlet calibration: **0.8047**
- Dirichlet test ECE: **0.0206**

These figures are historical benchmark results and are not V5 selection targets by themselves.

The six CNNs are:

- ConvNeXt-Tiny
- ConvNeXt-Small
- EfficientNet-B0
- EfficientNet-B3
- ResNet-50
- DenseNet-121

The six-model soft-vote is retained as the aggregation control because its validation/OOF selection was stronger than learned stacking; the higher test score of ridge stacking cannot be used as a model-selection argument without violating the project's selection protocol.

---

# 2. What V4 Actually Learned About the Under-40 Problem

## 2.1 The observed failure

On the earlier held-out HAM evaluation:

- under-40 escalating sensitivity was **0.143**
- 18 of 21 escalating test images were missed
- the corresponding 60+ sensitivity was **0.764** (historical baseline; closest validation value is 0.774, OOF value 0.733)

> [!IMPORTANT]
> **Unit-of-Analysis Limitation (S48 Audit):** The original held-out estimate contains 21 images but only 10 independent lesions (image count = 21, independent lesion count = 10); therefore the point estimate is descriptive and carries substantial uncertainty. Confirmatory under-40 claims require a larger independent cohort.

The disparity is associated with a strong age-band prevalence difference in the training distribution.

The important conclusion is not:

> “Age is the cause of the biological failure.”

The defensible conclusion is:

> **The learned representation and decision system encode age-correlated information that materially changes malignant ranking across age groups. The repository has not established that this information is purely biological, purely acquisition-related, or purely confounding.**

That distinction must remain explicit in V5.

## 2.2 What happened when age was removed

V3 adversarial age-invariance reduced the measured age entanglement but also reduced under-40 ranking performance.

Therefore V5 must NOT attempt to erase age information wholesale.

The correct question becomes:

> Can the model learn more lesion-specific malignant morphology while keeping useful age-correlated diagnostic information and reducing dependence on non-diagnostic context?

## 2.3 Decision-layer ceiling

The repository has already tested:

- temperature scaling;
- Dirichlet calibration;
- band-conditional calibration;
- fixed age-conditioned lambda;
- continuous lambda(age);
- selective abstention;
- combined policy layers.

These can improve operating points or calibration, but the V4 ceiling analysis shows that they cannot create substantial new ranking information at matched referral.

**V5 therefore moves the main research effort below the logit/decision layer.**

---

# 3. V4 / Phase-Y Work That Is Already Implemented

The earlier V5 draft incorrectly treated several Phase-Y items as future work. They are now completed and must not be repeated as V5 experiments.

## S71 — K-fold partition / OOF matrix

Implemented:

- lesion-grouped 5-fold partition;
- cross-fitted OOF matrix;
- fold-specific validation predictions;
- leakage verification.

Banked artifact:

`results/v4/kfold/oof_predictions.csv`

The matrix contains **15,294** pooled-training rows.

## S72 — V4 K-fold control training

Implemented:

- 5 folds;
- pooled V4 training;
- 224px;
- 30 epochs;
- no patience;
- batch 32;
- 2 workers;
- RTX 5050 Laptop GPU.

All five folds completed and were banked.

Best held-out-fold validation Macro-F1:

| Fold | Macro-F1 | Best epoch |
|---:|---:|---:|
| 0 | 0.6508 | 20/30 |
| 1 | 0.6974 | 28/30 |
| 2 | 0.6738 | 22/30 |
| 3 | 0.6696 | 27/30 |
| 4 | 0.6600 | 26/30 |
| Mean | **0.6703** | — |

This is a K-fold control/reference artifact, not a replacement for the original V1 benchmark and not a new V5 result.

## S68 — Target-side referral recalibration

Implemented and evaluated without touching reserved/test.

Fit on BCN/MSKCC development rows and evaluated on the V4 validation target panel.

Mean reduction in referral-budget error was **+0.0426**, below the declared 0.05 MCID.

Verdict:

**NULL**

Therefore V5 must not repeat S68 as an accuracy intervention.

## S69 — Smartphone modality admissibility

Implemented.

The adopted modality classifier achieved:

- 100% PAD rejection;
- AUROC **0.9994**;
- escalating-minus-benign rejection gap approximately **0.0000** under the recorded safety test.

The Mahalanobis alternative failed because it preferentially rejected escalating cases.

Therefore:

> **Retain the modality classifier as a deployment/admissibility gate; do not make it a V5 accuracy experiment.**

## S70 — V4 decision checkpoint

Implemented.

The current decision is:

- source a genuinely fresh external cohort;
- keep the previously defined 0.855 sensitivity floors and referral constraint as the clinical-contract reference;
- retain the relative improvement term against the V1 stack;
- use the cross-fitted K-fold infrastructure as the correct basis for future trainable experiments.

---

# 4. Historical Falsification Ledger

| Proposal | Historical status | V5 treatment |
|---|---|---|
| 224 -> 384px | **SUPPORTED LEVER** | KEEP; first V5 foundation |
| Six-CNN uniform soft-vote | **SUPPORTED LEVER** | KEEP as aggregation control |
| 24-view TTA | **SUPPORTED LEVER** | KEEP |
| Dirichlet calibration | **SUPPORTED LEVER** | KEEP |
| Groupwise Dirichlet | **SUPPORTED SAFETY/QUALITY LEVER** | KEEP |
| Equalized bipartite conformal | **SUPPORTED SAFETY LEVER** | KEEP |
| Modality classifier | **ADOPTED DEPLOYMENT GATE** | KEEP |
| Ridge stacking | **OOF overfit / selection trap** | DO NOT USE as primary ensemble |
| Old MaxViT-Tiny result | **Test-vs-val selection trap** | Do not promote from old result |
| Metadata fusion | **NULL / FAIL** | Do not repeat as primary intervention |
| Frozen specialist head | **FALSIFIED** | Do not repeat; new V5 branch must be end-to-end trainable |
| PanDerm/DINOv2 frozen probes | **FALSIFIED for the tested configurations** | Do not repeat unchanged |
| Fixed lambda / lambda(age) | **Operating-point ceiling / cost reject** | Do not repeat as ranking solution |
| Global adversarial age-invariance | **FALSIFIED** | Do not repeat |
| Corrected Shades-of-Grey | **NULL** | No new GPU campaign solely for colour constancy |
| Original colour-constancy implementation | **INVALID due to bug** | Historical result must not be cited as valid |
| EMA/cosine historical ladder | **NULL / truncated depending on rung** | Do not rerun blindly; only use a full schedule as part of a new training recipe if justified |
| Class reweighting | **NULL in recipe ladder** | Do not make it a headline V5 lever |
| Target-side referral recalibration S68 | **NULL** | Do not repeat as accuracy work |
| Mahalanobis smartphone gate | **FAIL** | Do not use for the admissibility decision |
| Class/age decision-layer offsets | **Cannot create ranking** | Not a V5 representation intervention |

---

# 5. Important Correction to an Earlier V5 Claim

A previous internal draft stated that a direct classifier on cached ConvNeXt features achieved approximately 0.993 under-40 pAUC.

That result was subsequently proven to be **in-sample feature extraction leakage** during Session S40 (Stage 0 benchmark in `results/v3/oos_probe_report.json`, which reproduced the flawed S35 result with `np_pauc = 0.99298`): the feature extractor was evaluated on the exact images used to fit the checkpoint.

The authoritative, cross-fitted out-of-sample feature probe on `ham_oof_CROSSFIT` (`results/v3/oos_probe_report.json`, Session S40 Phase A1 Stage 1, under-40 band, n=1319, n_escalating=64) demonstrated that:
- the out-of-sample frozen feature probe achieved an under-40 partial AUC ($pAUC_{FPR \le 0.20}$) of **0.7182** (exact: `0.718193`, 95% CI `[0.6505, 0.7838]`);
- this frozen probe was actually **inferior** to the simple posterior baseline $s_{uniform}$ ($pAUC = 0.8096$, $\Delta pAUC = -0.0914$ `[-0.1756, -0.0090]`, verdict `N2_FALSIFIED`).

Therefore:

> **Do not use the 0.993 number as evidence that the frozen representation contains nearly perfect under-40 information. The 0.993 result is permanently invalidated as an artifact of in-sample feature extraction leakage and must never re-enter the V5 evidence chain.**

The valid V5 premise is narrower and grounded:

- a representation-level bottleneck remains plausible because logit-level and post-hoc thresholding methods hit a demonstrable ranking ceiling;
- trainable high-resolution representations (where the trunk is updated end-to-end) remain untested;
- current architecture-specific, high-resolution (384px), and multi-scale representations should be directly evaluated end-to-end.

This correction is mandatory in the paper and in Antigravity's generated summaries.

---

# 6. Biological Grounding

V5 should treat each diagnostic class as a different visual problem.

## 6.1 Melanoma

Relevant dermoscopic structures include:

- atypical pigment network;
- irregular peripheral dots/globules;
- pseudopods;
- radial streaming;
- blue-white veil;
- scar-like depigmentation/regression;
- negative network;
- irregular vascularity;
- asymmetric distribution of structure and colour.

Early melanoma can show subtle structures rather than the large, obvious abnormalities of advanced disease.

A 2025 age-stratified melanoma study of 285 histopathologically confirmed melanomas reported higher prevalence of growth-associated features such as pseudopods and asymmetric globules among patients younger than 40, while older patients more often showed regression-associated findings and solar elastosis.

**V5 implication:** test whether high-resolution local representation preserves growth-related microstructures that are diluted in a pooled classifier.

Source:

https://pubmed.ncbi.nlm.nih.gov/40805292/

DermNet:

https://dermnetnz.org/cme/dermoscopy-course/dermoscopy-of-melanoma

## 6.2 Basal Cell Carcinoma

Important features include:

- arborising vessels;
- blue-grey ovoid nests;
- blue-grey globules;
- leaf-like structures;
- spoke-wheel areas;
- ulceration;
- shiny white structures;
- fine telangiectasia.

These are structurally different from melanocytic pigment-network cues.

Source:

https://dermnetnz.org/topics/basal-cell-carcinoma-dermoscopy

## 6.3 Actinic Keratosis

Relevant features include:

- strawberry pattern;
- follicular openings with white halos;
- scale/keratin;
- erythematous or pink background;
- rosettes under polarised light;
- pigmented follicular structures in pigmented lesions.

Several of these are small, distributed structures.

Source:

https://dermnetnz.org/topics/actinic-keratosis-dermoscopy

## 6.4 Benign Keratosis

Relevant structures include:

- milia-like cysts;
- irregular crypts;
- fissures and ridges;
- fingerprint-like structures;
- cerebriform/fat-finger morphology;
- variable colours.

BKL is particularly important as a melanoma mimic.

Source:

https://dermnetnz.org/cme/dermoscopy-course/dermoscopy-of-seborrhoeic-keratosis

## 6.5 Dermatofibroma

The classic pattern includes a central white scar-like area with peripheral pigment network, but prospective studies show substantial morphological variation.

Source:

https://pubmed.ncbi.nlm.nih.gov/18209171/

## 6.6 Vascular Lesions

These require strong colour and vascular morphology rather than a pigment-network-centric representation.

## 6.7 Nevus

Benign nevi often provide the dominant training prior and include relatively regular, symmetric pigment structures.

V5 should focus especially on the **melanoma-vs-nevus boundary**, because that is the most direct biological differential for the observed under-40 melanoma problem.

---

# 7. V5 Core Hypothesis

> **A high-resolution, lesion-aware, multi-scale representation with an explicit malignant-escalation objective can improve the model's ability to separate subtle malignant morphology from benign mimics, particularly in younger patients, without removing age-correlated information wholesale.**

This is a hypothesis, not a result.

---

# 8. V5 Core Chassis

These five items remain mandatory.

## V5-A1 — 384px global control

Use 384 × 384 as the primary high-resolution control because the corrected V4 recipe ladder (S53r) identified 224px $\to$ 384px as the only consistently positive ranking rung.

### Matched Evaluation & Primary Reproducibility Gate
The primary scientific question is:
> **Does 384px resolution reproduce the previously observed performance improvement over the correctly matched 224px control under the frozen V5 protocol?**

Therefore, S01 evaluates a **paired 384px vs 224px development screening comparison on Fold 0 of the frozen S71 partition** ($N_{\text{train}} = 12,235$, $N_{\text{val}} = 3,059$):
- **Baseline:** Banked S72 Fold 0 224px ConvNeXt-Tiny control (validation Macro-F1 = 0.6508 on Fold 0; multi-seed 224px Fold 0 control for seeds 42, 43, 44; S72 pooled 5-fold mean 0.6703 as cross-fold benchmark).
- **Intervention:** 384px ConvNeXt-Tiny trained on Fold 0 with identical optimizer (AdamW, lr $3\times 10^{-5}$, effective batch 32, 30 epochs) across 3 independent seeds (seeds 42, 43, 44).
- **Runtime Budget:** 3 seeds $\times 92.6\text{ min} \approx 4.6\text{ h}$ (extrapolated from S72 Fold 0 measured 42.0 min $\times 2.26$).
- **Evaluation Endpoints:** Paired $\Delta \text{Macro-F1}$, under-40 $pAUC_{FPR \le 0.20}$, per-class F1 (MEL, BCC, AKIEC), and inter-seed variability.
- **Two-Hurdle Acceptance Rule:**
  1. **Statistical Superiority:** 95% bootstrap confidence interval of paired $\Delta \text{Macro-F1}$ must strictly exclude zero (lower bound $> 0.000$);
  2. **Meaningful Magnitude:** Sample mean paired improvement must meet or exceed the predefined MCID ($\overline{\Delta \text{Macro-F1}} \ge +0.015$);
  3. **Direction & Consistency:** At least 2 of 3 seeds strictly positive ($\Delta > 0$), 3-seed mean $\ge +0.015$, and no individual seed exhibiting severe negative reversal ($\Delta \text{Macro-F1} \ge -0.010$);
  4. **Per-Class Safety:** No material degradation of clinically critical classes (MEL F1 $\Delta \ge -0.020$, BCC recall $\Delta \ge -0.020$);
  5. **Finalist Staging:** Full 5-fold cross-fitting ($N=15,294$ pooled OOF) is reserved for Tier 3 finalist confirmation (Stage 5 backbones and Stage 6 ensembling).

*Contextual Note:* The historical $+0.0297$ Macro-F1 improvement (S53r) and the $0.7700$ absolute score are supporting prior evidence and secondary reference benchmarks, not rigid unconditional thresholds that every new seed must achieve.

*Execution Interpretation & Scope Boundary:* S01 answers whether 384px is reproducibly promising on the frozen Fold-0 screening split ($N_{\text{val}}=3,059$). The three-seed Fold-0 result is an initial reproducibility screen, NOT a full cross-validated estimate. Final performance claims and ensemble selection strictly require the complete $N=15,294$ cross-fitted OOF matrix.


## V5-A2 — Dual-stream global + lesion-aware local

Architecture:

```text
full image 384px
        +
lesion-centred crop 384px
        +
optional controlled context ring
```

Do not implement this as a blind repeat of the old R3 mask-guided crop.

The old R3 was dropped because its predefined gate did not establish a sufficient useful effect and the geometry had practical failure modes.

V5 instead tests:

- lesion-centred optical zoom;
- soft context retention;
- crop expansion;
- late fusion;
- cross-attention.

### Capacity Control Requirement for Dual-Stream
The scientific question is:
> **Does lesion-aware multi-stream information improve performance?**  
> NOT: *Does doubling network capacity improve performance?*

For every dual-stream experiment, the study must define:
- **CONTROL:** Single-stream model with comparable parameter count and compute budget (e.g. wider/deeper single-stream trunk or matched capacity baseline).
- **EXPERIMENT:** Dual-stream global + lesion/context representation.

Where practical, explicitly match:
1. Trainable parameter count ($\Delta \text{params} \approx 0$);
2. Number of optimization steps;
3. Approximate FLOPs / compute budget;
4. Training schedule and optimizer hyperparameters.

If exact parameter matching is technically infeasible, the study must explicitly document:
- Parameter delta ($\Delta \text{params}$);
- Compute delta ($\Delta \text{FLOPs}$ and wall-clock memory);
- Methodological rationale explaining why exact matching is infeasible.

This capacity-control requirement applies across all multi-branch variants (DWT branch, multi-resolution patch branch, hierarchical model, morphology-gated ensemble).

## V5-A3 — Group-DRO and subgroup-aware representation learning

Retain Group-DRO.

But do not define groups only as age.

Candidate groups:

```text
age_band × escalation
age_band × diagnosis
```

Group counts must be inspected first.

Do not create tiny unstable groups.

## V5-A4 — modern high-resolution heterogeneous backbones

Candidate pool (exactly one frozen pre-registered configuration per family):

1. **ConvNeXt-V2-Tiny (`convnextv2_tiny.fcmae_ft_in22k_in1k_384`):**
   - Pretrained: Fully Convolutional Masked Autoencoder (FCMAE) pretrained on ImageNet-22k, fine-tuned on ImageNet-1k at 384px.
   - Core mechanism: Global Response Normalization (GRN) prevents dominant nevus channels from suppressing rare malignant features.
   - Parameters: 28.6M | Input: 384×384 | Batch: 16 (accum 2) | Optimizer: AdamW, lr $3\times 10^{-5}$, wd $1\times 10^{-2}$.
2. **EfficientNetV2-M (`tf_efficientnetv2_m.in21k_ft_in1k`):**
   - Pretrained: ImageNet-21k pretrained, fine-tuned on ImageNet-1k at 384px.
   - Core mechanism: Fused-MBConv stages optimizing parameter efficiency and gradient flow across scales.
   - Parameters: 54.1M | Input: 384×384 | Batch: 16 (accum 2) | Optimizer: AdamW, lr $3\times 10^{-5}$, wd $1\times 10^{-2}$.
3. **SwinV2-Tiny-384 (`swinv2_tiny_window12to16_192to384.ms_in22k_ft_in1k`):**
   - Pretrained: ImageNet-22k pretrained, fine-tuned on ImageNet-1k at 384px.
   - Core mechanism: Hierarchical shifted-window self-attention with post-normalization (res-post-norm) and continuous log-spaced relative position bias (adapted from 12×12 to 16×16 window).
   - Parameters: 28.35M | Input: 384×384 | Window size: 16×16 | Batch: 16 (accum 2) | Optimizer: AdamW, lr $3\times 10^{-5}$, wd $1\times 10^{-2}$.
   - *Decision Note:* SwinV2-Tiny is chosen over SwinV2-Base to match the ~28M parameter budget of ConvNeXt-Tiny and prevent GPU VRAM exhaustion on the 8.55 GB hardware.

An architecture is promoted for ensemble consideration only after:

1. adequate individual validation performance;
2. under-40 ranking analysis;
3. error-diversity analysis;
4. multi-seed stability.

The old SwinV2-Tiny-224 result is not evidence against this 384px configuration; it only invalidated the un-warmed 224px implementation.

## V5-A5 — fresh external cohort

S75 is a **data-sourcing workstream**, not the name of the final evaluation itself.

The final confirmatory external read should occur only after the V5 development plan is frozen and after a pristine, deduplicated cohort is established.

The repository's current state is that the old reserved cohort has been read eight times and must no longer be treated as a pristine confirmatory holdout.

---

# 9. V5 Biology-to-CV Experiments

These are the main additions to the original five.

## V5-B1 — Age-stratified morphology audit

Measure only where metadata support exists.

Candidate image-level descriptors:

- lesion area fraction;
- asymmetry;
- perimeter irregularity;
- radial pigment distribution;
- local colour entropy;
- number/distribution of high-contrast structures;
- high-frequency energy;
- boundary complexity;
- patch-level texture statistics.

The purpose is **descriptive hypothesis generation**, not a replacement for learned representations.

Do not write “all 25,331 images have age labels” unless the manifest verifies it.

## V5-B2 — Trainable escalation head

Build:

```text
shared visual encoder
 ├── 7-class diagnosis head
 └── binary escalation head
```

Train jointly:

\[
L = L_{7class} + \lambda_{esc}L_{esc}
\]

Search only:

- λ = 0.25
- λ = 0.50
- λ = 1.00

This is NOT the failed frozen-specialist experiment.

The trunk must be trainable.

## V5-B3 — Hierarchical diagnosis

Compare:

1. flat 7-class;
2. 7-class + escalation head;
3. hierarchical classification:

```text
lesion
├── escalating
│   ├── melanoma
│   ├── BCC
│   └── AK
└── non-escalating
    ├── NV
    ├── BKL
    ├── DF
    └── VASC
```

The hierarchy is an architectural hypothesis, not a clinical truth claim.

## V5-B4 — Multi-resolution local patch attention

Architecture and resolution policy:

```text
global image (resized to 384×384)
       +
lesion-centred crop (resized to 384×384)
       +
4–8 local micro-patches (96×96, 128×128, or 160×160)
```

> [!IMPORTANT]
> **Resolution & Extraction Flow Correction:** Do NOT require 4–8 full 384×384 forward passes (which would create an unacceptable compute and memory explosion). Instead, use lower-resolution micro-patches (e.g. 96×96, 128×128, or 160×160).
> 
> Critically, the patches must be extracted from the **original/high-resolution image BEFORE aggressive resizing**.
> 
> **Preferred extraction pipeline:**
> ```text
> raw high-resolution image
>       ↓
> lesion localization / segmentation bounding box
>       ↓
> native-resolution local crop
>       ↓
> small micro-patch resize (96×96, 128×128, or 160×160)
>       ↓
> lightweight local patch encoder / cross-attention head
> ```
> Do not first downsample the entire image to 384×384 and then crop the patch. The biological hypothesis is that fine native microstructure (atypical pigment network, dots/globules, delicate streaks) is preserved prior to full-frame downsampling.

### Implementation & Data-Path QC Gate
Before model training begins, a unit test must verify the data pipeline integrity:
- Assert that bounding box crops originate directly from raw resolution dimensions ($1024\times 768$ or native ISIC dimensions) rather than pre-resized $384\times 384$ tensors.
- Do NOT claim "sub-millimeter preservation" as an empirical performance outcome unless physical pixel-to-millimeter magnification metadata is verified.

### Scientific Performance Gate
Evaluated strictly on development/OOF metrics:
- Under-40 partial AUC: $\Delta pAUC_{FPR \le 0.20} \ge +0.050$ over the 384px control;
- Macro-F1 retention: $\Delta \text{Macro-F1} \ge -0.010$;
- Capacity control: Must show information gain against a capacity-matched single-stream trunk.

Candidate patch sources:

- lesion centroid;
- lesion boundary / peripheral margin;
- high-gradient / high-entropy visual regions;
- random lesion-contained patches.

The final patch policy must be fixed on development data.

## V5-B5 — Soft interior / boundary / context decomposition

Generate:

- lesion interior;
- boundary ring;
- peri-lesional context.

Instead of deleting all context, compare:

\[
F' = M F + \alpha(1-M)F
\]

with:

- α = 0
- 0.10
- 0.25
- 0.50
- 1.00

This directly tests whether context is useful, harmful, or merely correlated with demographic/acquisition information.

### Segmentation Policy & Quality Gate (B4/B5 Dependency)
- Use actual, available ground-truth HAM lesion segmentation masks where validated.
- Do NOT assume the masks are perfect or complete.
- **Stage 0 Segmentation Audit must report:**
  1. Missing masks count;
  2. Corrupted/invalid mask formats;
  3. Tiny masks (lesion area $< 1\%$ of image frame);
  4. Implausible masks (lesion area $> 95\%$ of image frame or touching all 4 boundaries).
- **Segmentation-Quality Gate:** Masks must cover $\ge 90\%$ of the development split with valid boundaries before B4/B5 execution.
- If the segmentation pipeline fails or mask quality is inadequate, **keep the experiment documented in the plan**, but do **NOT silently substitute an unreliable heuristic or thresholded mask**.

## V5-B6 — Melanoma-versus-nevus end-to-end branch

This is specifically allowed because it differs from the already-failed frozen specialist.

The branch is jointly trained with the shared encoder.

Use hard negatives:

- young melanoma vs nevus;
- melanoma vs BKL;
- melanoma vs pigmented BCC.

Do not use test-derived hard negatives.

## V5-B7 — Class-aware supervised contrastive learning

Use positives within diagnosis and augmentations, while emphasizing biologically close negatives.

Candidate losses:

```text
CE
CE + SupCon
CE + escalation + SupCon
```

Search only the small pre-registered range.

## V5-B8 — High-frequency / wavelet branch

Use Haar DWT sub-bands:

- LH
- HL
- HH

to expose high-frequency local structures.

This is a targeted hypothesis for:

- fine pigment networks;
- streaks;
- small vascular patterns;
- small scale/keratin structures.

It must be compared against an equivalent parameter-budget control to avoid crediting capacity alone.

## V5-B9 — Boundary/radial representation

For segmented lesions, derive:

- radial distance from lesion centroid;
- angular distribution;
- boundary irregularity.

A polar-coordinate representation may help represent peripheral growth patterns such as pseudopods.

This is exploratory and should remain secondary to the simpler lesion-aware stream.

## V5-B10 — Morphology-gated ensemble

Eligible once any validated morphology signal exists (e.g., from B1 descriptive morphology audit, B5 boundary/context decomposition, B6 morphology-specific representation, or expert attribute supervision V5-MORPH-ATTR). Does not strictly require Derm7pt attribute supervision to succeed.

Possible gate inputs:

- visual morphology predictions;
- uncertainty;
- representation similarity.

Do not directly feed age into the gate.

Do not allow a high-capacity gate to overfit OOF folds.

A simple convex or shallow gate should be the default.

## V5-B11 — Adaptive Feature Pooling (GeM / Attention Pooling)

Purpose:
Test whether standard Global Average Pooling (GAP) discards sparse, highly localized diagnostic features (such as solitary atypical pigment networks, focal pseudopods, or isolated shiny white lines).

Compare:
- Baseline: Standard Global Average Pooling (GAP);
- Intervention: Generalized Mean (GeM) pooling:
  \[
  f = \left( \frac{1}{|R|} \sum_{x \in R} x^p \right)^{1/p}
  \]
  with parameter initialization $p = 3.0$ (optionally trainable/learnable $p$).

Protocol & Constraints:
- Keep this ablation CHEAP and lightweight.
- Do NOT turn this into a new large architecture family.
- Evaluate on the exact same trunk and training recipe.
- Status: Secondary / low-cost ablation. Do NOT promote it above primary multi-scale representation interventions.

## V5-B12 — Pairwise MEL-vs-NV Ranking Objective

Scientific Motivation:
The most clinically severe and frequent failure mode in the under-40 cohort is melanoma-versus-benign confusion, specifically melanoma misclassified as benign nevus (MEL vs NV). Standard multi-class cross-entropy optimizes posterior calibration over all seven classes, but does not explicitly enforce a strict score separation between malignant melanoma and benign melanocytic mimics.

Objective Formulation:
Test an auxiliary pairwise ranking objective that directly penalizes inversion of the relative score ordering:
\[
L_{rank} = \max\left(0, m - s_{MEL}(x_{MEL}) + s_{MEL}(x_{NV})\right)
\]
where $s_{MEL}(x)$ is the scalar melanoma escalation score (or logit), and $m > 0$ is a pre-specified margin.

Multi-Task Loss Combination:
Combine with the multi-task loss only after establishing the baseline:
\[
L = L_{7class} + \lambda_{esc} L_{escalation} + \lambda_{rank} L_{rank}
\]

Pre-registered Candidate Grid:
- Margin $m \in \{0.10, 0.20, 0.50\}$
- Coefficient $\lambda_{rank} \in \{0.05, 0.10, 0.25\}$
- Do NOT create a combinatorial hyperparameter grid.

Strict Integrity Constraints:
- Construct pairs **strictly from development/training information** within each mini-batch;
- **NEVER** use test-derived or reserved-derived pairs;
- Strictly preserve **lesion disjointness**: never pair images belonging to the same lesion;
- Do NOT use the same lesion as both positive and negative;
- This is an **end-to-end representation/ranking experiment**, NOT a post-hoc decision thresholding mechanism.

---

# 10. Auxiliary Morphology Supervision & Attribute Policy

This remains a promising direction for guiding visual representations to clinically meaningful dermoscopic structures, but it enforces a strict data and ground-truth policy.

### Biological Attribute Policy
Only use verified, **expert-labelled morphology attributes** from certified dermatological datasets.
Never manufacture attribute ground truth from:
- Grad-CAM heatmaps;
- Model predicted probabilities or logits;
- Attention rollouts or transformer attention maps;
- Pseudo-labels generated without formal dermatopathologist validation.

Candidate auxiliary attributes:
- atypical pigment network (melanoma vs nevus differential);
- pseudopods / radial streaks (active radial growth phase);
- blue-white veil / structures (deep dermal melanin/acanthosis);
- regression structures (scar-like depigmentation);
- polymorphic vascular structures (arborizing, dotted, hairpin, linear);
- follicular structures / comedo-like openings / milia-like cysts (keratinocytic markers);
- surface scale / hyperkeratosis.

Provenance and audit requirement:
Use an external expert-annotated dermoscopy resource where suitable, such as Derm7pt or another dataset whose annotations actually correspond to the desired attributes.
The transfer protocol must formally audit:
- class and attribute definition compatibility;
- image provenance and acquisition device metadata;
- patient and lesion overlap with HAM splits (strict disjointness assertion);
- inter-observer annotation agreement;
- train/validation/test partition integrity.

---

## 11. Hard-Negative Mining & Policy

### Hard-Negative Policy
Hard negatives must be strictly OOF/development-derived. Never use locked test or reserved images to build the training hard-negative set.

Priority failure buckets:
```text
<40 MEL -> NV              (young melanoma misclassified as benign nevus)
<40 MEL -> BKL             (young melanoma misclassified as seborrheic keratosis/lichenoid)
<40 BCC -> NV              (young basal cell carcinoma confused with nevus)
<40 AKIEC -> NV/BKL        (actinic keratosis confused with benign lesion)
MEL -> BKL                 (melanoma vs benign keratosis across all ages)
MEL -> NV                  (melanoma vs melanocytic nevus across all ages)
BCC -> melanocytic mimics  (pigmented BCC vs nevus/melanoma)
```

Two-stage procedure:
```text
Stage 1: Ordinary base training on development folds
Stage 2: Hard-negative fine-tuning or JTT-style (Just Train Twice) sample reweighting
```

Integrity assertion:
- Hard-negative pairs and error strata are extracted **exclusively from cross-validated OOF predictions** of the Stage 1 control.
- Never snoop on held-out evaluation splits to define mining priorities.

---

# 12. Losses That Are Still Worth Testing

Do not repeat the old LDAM/ASL experiment unchanged.

Candidate V5 losses:

- Cross-Entropy control (Stage 1);
- CE + escalation loss (Stage 2);
- CE + SupCon (Stage 3);
- CE + escalation + SupCon (Stage 3);
- Group-DRO (Stage 4);
- Group-DRO + SupCon (Stage 4);
- Balanced Softmax (**DEFERRED — NOT CORE V5**; secondary sensitivity probe);
- JTT-style hard-example retraining (Stage 3).

Losses are selected on development metrics only.

---

# 13. Foundation-Model Use After the Failed Frozen Probe (DEFERRED — NOT CORE V5)

PanDerm and DINOv2 frozen feature probes were already tested and did not beat the tested ConvNeXt configuration.

Therefore V5 must NOT repeat:

```text
frozen foundation embedding -> shallow classifier
```

Instead, a foundation model may be tested as a **teacher or distillation source**:

```text
dermatology foundation encoder
          ↓
teacher features
          ↓
high-resolution V5 student
```

Potential distillation targets:

- feature embeddings;
- attention maps;
- logits;
- intermediate representations.

The goal is knowledge transfer, not replacing the current strong task-specific trunk with an already-tested frozen representation.

---

# 14. Colour and Image Standardization

The corrected colour-constancy experiment was NULL.

Therefore no V5 compute campaign should be spent merely trying another colour-constancy parameter.

The standard V5 pipeline should preserve diagnostically relevant colour.

Safe augmentation family:

- geometric transforms;
- moderate brightness;
- moderate contrast;
- small scale variation;
- mild blur/noise.

Avoid broad hue/saturation perturbations that can destroy medically meaningful colour relationships.

---

# 14b. Exploratory and Deferred Secondary Experiments

The following three interventions are formally pre-registered with bounded scope and explicit demotion to prevent GPU waste:

## V5-X1 — 384px Mixup / CutMix Sensitivity Analysis
- **Status:** EXPLORATORY / LOW PRIORITY
- **Scientific Motivation:** The repository tested Mixup/CutMix at an earlier, non-optimal recipe configuration (S50/S53), but not within the final 384px V5 representation recipe. Therefore, the historical result must not be cited as a complete, universal falsification of all 384px Mixup/CutMix variants.
- **Execution Protocol:** Run only after the primary representation gates (Stages 1–3) have established surviving representations. Do NOT let it expand or delay the main campaign.
- **Augmentation Constraint:** Must use medically and colour-safe blending parameters ($\alpha \in [0.1, 0.2]$) to prevent creating unphysical, artifactual dermoscopic structures.

## V5-X2 — Optional SWA (Stochastic Weight Averaging) Schedule Ablation
- **Status:** DEFERRED / EXPLORATORY
- **Scientific Motivation:** The historical EMA/cosine experiment in V4 was not a clean general test of weight averaging because the run was prematurely stopped due to early-stopping interactions. Therefore, weight averaging is not formally falsified across all regimes.
- **Execution Protocol:** Do NOT add a full SWA campaign to the core V5 pipeline. Run only if the final surviving training recipe demonstrates terminal checkpoint instability or high validation variance across final epochs.

## V5-X3 — Exploratory Imbalance-Loss Sensitivity (Focal Loss)
- **Status:** DEFERRED / LOW PRIORITY
- **Scientific Motivation:** The repository has already extensively tested multiple imbalance-oriented loss families (LDAM, ASL, class-balanced focal variants, cost-sensitive matrix reweighting), while the central V5 hypothesis is that under-40 failure is a visual representation bottleneck, not a loss-gradient artifact.
- **Execution Protocol:** Do NOT make Focal Loss a core V5 branch. Do NOT burn a 5-fold campaign on it unless an isolated, low-cost screening probe shows compelling development-stage improvement.

## V5-X4 — Balanced Softmax Subgroup Loss
- **Status:** DEFERRED — NOT CORE V5
- **Scientific Motivation:** Probes whether label-distribution-aware logit adjustment can stabilize minority classes without harming calibrated posteriors.
- **Execution Protocol:** Secondary sensitivity probe. Non-blocking; do not promote to core V5 without isolated development justification.

## V5-X5 — Derm Foundation Distillation
- **Status:** DEFERRED — NOT CORE V5
- **Scientific Motivation:** Explores knowledge distillation from dermatology foundation models (e.g., PanDerm, DINOv2) to high-resolution student trunks without repeating the failed frozen feature probes.
- **Execution Protocol:** Deferred pending model access, weight verification, and core V5 completion.

---

# 15. Ensemble Strategy

Start with the established uniform soft-vote control.

### Mandatory Prerequisite: V5-OOF-COMMON-MATRIX
Before Stage 6 (S17) ensembling or model selection begins, every candidate model MUST have cross-validated predictions generated on the **exact same Out-of-Fold (OOF) rows** defined by the frozen S71 5-fold partition ($N=15,294$ images).

**Strict Prohibitions:**
- Do NOT combine models evaluated on different fold definitions or seed partitions;
- Do NOT mix validation split predictions ($N=2,270$) with OOF predictions ($N=15,294$);
- Do NOT evaluate ensemble weights using mismatched calibration populations.

**Prerequisite Schema (`V5-OOF-COMMON-MATRIX`):**
An integrity-checked aligned tabular matrix containing:
1. `image_id` (Unique image identifier);
2. `lesion_id` (Lesion grouping identifier; verified disjoint across folds);
3. `fold` (Partition fold index $k \in \{0, 1, 2, 3, 4\}$);
4. `true_label` (Ground-truth 7-class integer $\{0..6\}$);
5. `age_band` ($<40$, $40\text{--}59$, $60+$);
6. `escalation_status` (Binary clinical escalation truth $\{0, 1\}$);
7. `prob_{model_id}_{class}` (7-class calibrated probability vectors for each candidate model).

Stage 6 (`V5-S17-ENSEMBLE`) cannot execute until `V5-OOF-COMMON-MATRIX` passes automated row-alignment and zero-missing-value verification.

Candidate members:

- V5 384 ConvNeXt control;
- ConvNeXt-V2;
- EfficientNetV2-M;
- SwinV2-384;
- best lesion-aware multi-scale architecture;
- optional hierarchical architecture if it contributes complementary errors.

Measure:

- Macro-F1;
- melanoma F1;
- AKIEC F1;
- BCC sensitivity;
- under-40 pAUC;
- under-40 sensitivity;
- error disagreement;
- Yule's Q;
- double-fault;
- calibration.

Do not add a model solely because its individual score is high.

Do not replace uniform soft-voting with a learned stack unless the OOF protocol shows a repeatable advantage.

---

# 16. External Cohort Strategy

Potential external data should be separated into three roles.

## Role A — Confirmatory Clinical Evaluation (S20 Protocol)

A pristine, unread external cohort (S75, e.g. BCN20000/MSKCC/ISIC-2020 unread partition) with verified ground-truth histopathology.

### Pre-Registered Statistical Analysis Specification
The final confirmatory read (`V5-S20-EXTERNAL`) is governed by the following strict statistical protocol:

1. **Hierarchical Confirmatory Endpoints:**
   - **Primary Confirmatory Endpoint (Global Discrimination):** 7-class Macro-F1 across all external cases ($\ge 0.7500$ vs null $\le 0.7000$ at $\alpha = 0.05$).
   - **Key Secondary Confirmatory Subgroup Endpoint (Young Patient Escalation Rescue):** Malignant escalation sensitivity in patients aged $<40$ years at the pre-registered clinical referral threshold $\tau_{80}$ (fixed on development OOF to achieve $\ge 80\%$ specificity on non-escalating lesions). Evaluated conditionally only if the Primary Endpoint achieves statistical significance.
2. **Unit of Analysis:** Independent lesion cluster (never individual images; grouped by lesion/patient ID to prevent pseudo-replication).
3. **Target Population & Minimum Sample Size:**
   - Total external cases: $N_{\text{external}} \ge 1,000$ dermoscopic images.
   - **Sample Size Floor:** Minimum **$N_{\text{esc,<40}} \ge 40$ independent lesions** provides **80.7% exact statistical power** (exact binomial rejection critical value $k \ge 26/40$, exact $\alpha = 0.0403$) to reject $H_0: \text{Sensitivity} \le 0.500$ in favor of $H_1: \text{Sensitivity} \ge 0.700$ at $\alpha = 0.05$.
   - **Sample Size Target:** $N_{\text{esc,<40}} \ge 50$ independent lesions provides **85.9% exact statistical power** (exact binomial rejection critical value $k \ge 32/50$, exact $\alpha = 0.0325$, and $97.1\%$ power at $p=0.75$).
   - *(Note: At $N = 35$, exact power at $p=0.70$ is $77.3\%$; $N \ge 40$ is required for $\ge 80\%$ power)*.
4. **Endpoint Formulation:**
   - **Numerator:** Count of independent under-40 escalating lesions whose calibrated ensemble referral probability exceeds the operating threshold ($p_{\text{escalate}} \ge \tau_{80}$).
   - **Denominator:** Total count of independent under-40 escalating lesions in the external cohort.
5. **Statistical Hypothesis & Inference:**
   - Null Hypothesis $H_0$: Under-40 escalating lesion sensitivity $\le 0.500$ (representing failure of the representation intervention to achieve clinical utility).
   - Alternative Hypothesis $H_1$: Under-40 escalating lesion sensitivity $\ge 0.700$ (one-sided exact binomial test, significance level $\alpha = 0.05$).
   - Confidence intervals: 95% two-sided Wilson score interval with continuity correction and lesion-grouped percentile bootstrap with 2,000 resamples.
6. **Multiple-Testing Control (Fixed-Sequence Hierarchy):**
   - Step 1: Test Primary Endpoint (Macro-F1 $\ge 0.7500$ vs null $\le 0.7000$ at $\alpha=0.05$). If rejected, proceed to Step 2. If not rejected, stop; secondary confirmatory endpoint cannot be claimed.
   - Step 2: Test Secondary Confirmatory Endpoint (Under-40 sensitivity $\ge 0.700$ vs $H_0 \le 0.500$ at $\alpha=0.05$). Preserves overall family-wise error rate at $\alpha = 0.05$.
7. **Execution Invariants:**
   - **Locked Status:** Weights, decision thresholds, calibration maps, and preprocessing pipelines are frozen prior to external data unblinding.
   - **Single Read:** The external evaluation is executed exactly once.
   - **Zero Feedback:** External results are strictly confirmatory and cannot trigger retrospective model selection or hyperparameter tuning. If the gate is not met, the Pre-Registered Null Pathway (§26) is executed immediately.

## Role B — representation/attribute pretraining

Expert-annotated dermatology datasets with compatible labels.

## Role C — modality/domain-shift robustness

Clinical or total-body-photography datasets such as SLICE-3D may be used as domain-shift probes, but they should not be silently treated as equivalent to dermoscopy.

## ISIC-2020

ISIC-2020 is a candidate external source because it contains a large dermoscopy challenge corpus with patient/lesion curation.

Before use:

- duplicate screen;
- source-provenance audit;
- patient-level separation;
- lesion-level separation;
- image-hash/perceptual-hash checks where possible;
- frozen evaluation manifest.

External inference must not feed back into development selection.

---

# 17. V5 Experimental Order

The earlier 19-arm matrix was too broad for the available compute and would create unnecessary selection degrees of freedom.

Use this staged, compute-prioritized campaign.

## Stage 0 — Integrity and Dataset Audit
CPU only.
- Inspect git status and verify clean working tree;
- Verify repository hashes, manifest integrity, and split manifests;
- **Leakage Assertion:** Verify **0 detected patient/lesion overlap using all available identifiers, with strict lesion-group disjointness** enforced across train, validation, and test splits;
- Verify test split and reserved cohort read locks (assert test receipt $\le 2$, reserved receipt $\le 8$);
- Audit diagnostic metadata and label availability across age strata;
- **Segmentation Quality Audit:** Inspect HAM lesion mask segmentation coverage, format, and boundary quality ($\ge 90\%$ valid mask requirement);
- **External Cohort Pre-check:** Verify S75 external cohort status and provenance;
- Generate `results/v5/v5_no_repeat_registry.json` and compute plan SHA-256 hash.
- **STOP if any integrity assertion fails.**

## Stage 1 — 384 Control (Multi-Seed Paired Confirmation)
GPU. Train the authoritative 384px baseline.
- Single architecture first (ConvNeXt-Tiny matching S72 control);
- Fixed training recipe, documented seeds (42, 43, 44), Fold 0 of frozen S71 data partition ($N_{\text{train}} = 12,235$, $N_{\text{val}} = 3,059$), zero test read;
- Runtime budget: 3 seeds $\times 92.6\text{ min} \approx 4.6\text{ h}$ (extrapolated from S72 Fold 0 measured 42.0 min $\times 2.26$);
- **Primary Reproducibility Gate (Two-Hurdle Rule):** Paired 384px vs 224px development screening comparison on Fold 0. Requires:
  1. Statistical evidence: 95% bootstrap confidence interval of paired $\Delta \text{Macro-F1}$ lower bound $> 0.000$ (excludes zero);
  2. Meaningful magnitude: Mean paired improvement $\overline{\Delta \text{Macro-F1}} \ge +0.015$ (predefined MCID);
  3. Seed consistency: At least 2 of 3 seeds strictly positive ($\Delta > 0$), 3-seed mean $\ge +0.015$, and min seed $\ge -0.010$;
  4. Per-class safety: No material degradation in clinically critical classes (MEL F1 $\Delta \ge -0.020$, BCC recall $\Delta \ge -0.020$);
- Full 5-fold OOF cross-fitting ($N=15,294$) is reserved for promoted finalists (Stage 5 backbones and Stage 6 ensembling);
- The historical $+0.0297$ Macro-F1 delta (S53r) and $0.7700$ score serve as secondary reference context, not unconditional rigid thresholds.


## Stage 2 — High-Value Representation Tests
Screen candidate representation mechanisms in order of highest theoretical yield:
1. Dual-stream (global 384 + lesion-centred local 384);
2. Trainable multi-task escalation head;
3. Hierarchical diagnosis head;
4. Multi-resolution local patch attention (micro-patches 96–160px).

Each candidate must satisfy:
- Implementation sanity and zero-leakage check;
- Parameter-matched capacity control;
- Development endpoint gate (one seed for initial screening, 3 seeds for promoted candidates).
- Do NOT immediately test every loss on every backbone.

## Stage 3 — Biological Representation Refinements
Apply only to the best surviving Stage-2 representation(s):
1. Soft boundary / interior / context decomposition;
2. Development-derived hard-negative mining (MEL vs NV, MEL vs BKL);
3. Supervised contrastive learning (SupCon);
4. Pairwise MEL-vs-NV ranking objective ($L_{rank}$);
5. Wavelet / Haar DWT high-frequency branch;
6. Auxiliary morphology supervision (where expert annotations exist);
7. Adaptive feature pooling (GeM pooling ablation, $p=3$).

Preserve clear distinction between core representations and secondary/low-cost ablations.

## Stage 4 — Subgroup-Robust Training
Apply Group-DRO only to the strongest surviving representation.
- Do NOT train Group-DRO on every backbone;
- Compare: ERM control vs Group-DRO vs Group-DRO + SupCon;
- Use carefully justified group definitions ($G = \text{age\_band} \times \text{escalation}$);
- Verify group sample sizes; do not create tiny, unstable groups.

## Stage 5 — Modern Heterogeneous Backbones
Apply the surviving representation and training recipe to:
- ConvNeXt-V2 (with Global Response Normalization);
- EfficientNetV2-M (compound scaling);
- SwinV2-384 (hierarchical shifted-window self-attention).

Standardize preprocessing and training controls across all backbones. Select candidate ensemble members using validation performance, under-40 development ranking, error diversity, and multi-seed stability. Never use test performance for backbone selection.

## Stage 6 — Compact Ensemble
Combine top heterogeneous candidates using uniform soft voting as the primary aggregation control.
- Candidate pool: V5 384 control, ConvNeXt-V2, EfficientNetV2-M, SwinV2-384, best lesion-aware multi-scale model, best hierarchical/morphology model (if complementary);
- Ensembling metrics: Macro-F1, MEL F1, AKIEC F1, BCC sensitivity, under-40 pAUC, error disagreement, Yule's Q, double-fault index;
- Do NOT add redundant models that merely duplicate predictions.

## Stage 7 — Existing Safety Stack & 384px TTA Policy
Apply the validated downstream safety machinery:
- **24-view dihedral TTA:** Retain the proven 24-view TTA. DO NOT remove it.
  > [!TIP]
  > **TTA Cost-Aware Policy:** Add a practical engineering checkpoint comparing 24-view TTA against an optional reduced-view development TTA (e.g. 4-view or 8-view) *only* if 24-view extraction becomes the dominant compute bottleneck on the laptop GPU. This is an engineering optimization, NOT a headline scientific claim. Do not select a reduced-view policy based on test results.
- **S55 groupwise multi-calibration:** Group-conditional Dirichlet calibration to eliminate directional confidence gaps;
- **Conformal prediction safety net:** Equalized bipartite conformal prediction to enforce coverage guarantees;
- **S69 modality admissibility gate:** Logistic tripwire to reject non-dermoscopic/smartphone images before classification.
These are downstream safety mechanisms; do not reinterpret them as V5 representation breakthroughs.

## Stage 8 — Fresh External Cohort Confirmation
Perform the final confirmatory evaluation:
- Strictly freeze code, architecture weights, ensemble aggregation weights, preprocessing pipelines, calibration parameters, conformal quantile thresholds, external manifest, and pre-registered analysis plan;
- Perform a **single, locked confirmatory read** on the fresh external cohort (S75);
- Never feed external cohort results backward into model or parameter selection.

---

# 18. Acceptance Gates & Statistical Framework

Do not set arbitrary numerical gates merely to pass an experiment. Each gate documents its clinical endpoint, unit of analysis, Minimum Clinically Important Difference (MCID), confidence interval method, minimum useful sample size, and failure consequence.

> [!WARNING]
> **Subgroup Statistical Power Limitation:** Under-40 OOF performance is a development-stage directional signal rather than a precise confirmatory estimate. The current development OOF cohort contains approximately 34 independent escalating lesions (across 64 images), yielding wide uncertainty. Confirmatory under-40 claims strictly require the fresh external cohort (Stage 8).

### Gate A — Representation (Under-40 Malignant Ranking)
- **Endpoint:** Partial AUC ($pAUC_{FPR \le 0.20}$) for escalating lesions in the under-40 age band.
- **Unit of Analysis:** Lesion cluster (enforcing patient/lesion grouping to avoid pseudo-replication).
- **MCID:** $\Delta pAUC \ge +0.050$ over the 384px ConvNeXt control.
- **Confidence Interval Method:** Lesion-grouped percentile bootstrap with 2,000 resamples (evaluating paired $\Delta pAUC$).
- **Minimum Useful Sample Size:** Development OOF subset ($n=64$ images, $\approx 34$ independent lesions; directional signal).
- **Failure Consequence:** Candidate fails representation gate; do NOT promote to multi-seed or Group-DRO training.

### Gate B — Global Quality (Macro-F1 Retention)
- **Endpoint:** 7-class Macro-F1 across all diagnostic categories.
- **Unit of Analysis:** Held-out image / lesion group.
- **MCID:** $\Delta \text{Macro-F1} \ge -0.010$ (maximum allowable degradation tolerance vs 384 control).
- **Confidence Interval Method:** Stratified bootstrap (2,000 resamples).
- **Minimum Useful Sample Size:** Single-fold development validation split ($N=3,059$) or complete pooled 5-fold OOF ($N=15,294$). (Note: $N=12,235$ represents the per-fold training subset).
- **Failure Consequence:** Disqualification; no candidate that impairs general diagnostic accuracy may be promoted.

### Gate C — Subgroup Safety (Cross-Strata Parity)
- **Endpoint:** Malignant escalation sensitivity in 40–59 and 60+ age bands at fixed referral specificity.
- **Unit of Analysis:** Lesion cluster within age band.
- **MCID:** No severe degradation in older age strata ($\Delta \text{sensitivity} \ge -0.030$).
- **Confidence Interval Method:** Lesion-grouped percentile bootstrap within stratum.
- **Minimum Useful Sample Size:** 40–59 band ($n=397$ escalating OOF), 60+ band ($n=893$ escalating OOF).
- **Failure Consequence:** Rejection; under-40 gains achieved by harming older patients are clinically unacceptable.

### Gate D — Stability (Multi-Seed Reproducibility)
- **Endpoint:** Mean, standard deviation, and direction of effect for paired $\Delta \text{Macro-F1}$ and under-40 $\Delta pAUC$ across 3 independent training seeds (seeds 42, 43, 44).
- **Unit of Analysis:** 3 independent training runs with identical recipe on frozen splits.
- **Reproducibility Criterion:**
  1. Mean paired improvement must be positive ($\overline{\Delta \text{Macro-F1}} > 0$, $\overline{\Delta pAUC} > 0$);
  2. Direction of effect must be consistent across seeds (at least 2 of 3 seeds strictly positive, mean positive, min seed $\Delta \ge -0.010$);
  3. Seed Stability Evaluation: Sample standard deviation across seeds $s_{\text{seed}}$ is reported as an empirical dispersion metric. For Macro-F1, an instability warning is flagged if $s_{\text{seed}} > 0.015$ (indicating that seed dispersion reaches the full magnitude of the resolution MCID). For under-40 $pAUC$, dispersion is reported descriptively given small sample variance.
- **Methodological Rule:** Historical seed variability ($\sigma_{seed} \ge 0.072$ on under-40 sensitivity documented in S44/S48) motivates multi-seed confirmation; it is not itself the acceptance threshold.
- **Finalist Requirement:** Promoted candidates advance to full 5-fold cross-fitting on the frozen S71 partition. Single-seed lucky outliers that fail multi-seed consistency are demoted immediately.

### Gate E — Complementarity (Ensemble Diversity)
- **Endpoint:** Prediction error disagreement, pairwise Yule’s $Q$ statistic, and double-fault rate.
- **Unit of Analysis:** Paired OOF error matrices across all common OOF rows ($N=15,294$).
- **MCID:** $Q < 0.70$ against existing ensemble members; double-fault rate strictly lower than individual error rates.
- **Confidence Interval Method:** Non-parametric paired bootstrap.
- **Minimum Useful Sample Size:** Full development OOF corpus.
- **Failure Consequence:** Exclude from ensemble; retain uniform soft-voting with existing core backbones.

### Gate F — Confirmatory External Evaluation (S20)
- **Evaluation Structure:** Fixed-Sequence Hierarchical Testing:
  - **Primary Endpoint:** 7-class Macro-F1 $\ge 0.7500$ (evaluated at $\alpha = 0.05$);
  - **Key Secondary Confirmatory Subgroup Endpoint:** Under-40 escalating lesion sensitivity $\ge 0.700$ at $\ge 80\%$ non-escalating specificity (evaluated at $\alpha = 0.05$ conditionally only if Primary Endpoint is significant).
- **Unit of Analysis:** Independent lesion cluster (Wilson score interval / 2,000-resample grouped bootstrap).
- **Hypothesis Testing:** $H_0: \text{Sensitivity}_{<40} \le 0.500$ vs $H_1: \text{Sensitivity}_{<40} \ge 0.700$ (one-sided exact binomial test, $\alpha = 0.05$).
- **Sample Size & Statistical Power:**
  - **Floor:** Minimum $N_{\text{esc,<40}} \ge 40$ independent escalating lesions provides **80.7% exact power** (exact binomial rejection critical value $k \ge 26/40$, exact $\alpha = 0.0403$);
  - **Target:** $N_{\text{esc,<40}} \ge 50$ independent escalating lesions provides **85.9% exact power** (exact binomial rejection critical value $k \ge 32/50$, exact $\alpha = 0.0325$, and $97.1\%$ power at $p=0.75$);
  - *(Note: At $N = 35$, exact power at $p=0.70$ is $77.3\%$; $N \ge 40$ is required for $\ge 80\%$ power)*.
- **Multiple Testing Policy:** Fixed-sequence hierarchical testing (Step 1: Macro-F1 $\to$ Step 2: Under-40 sensitivity) preserves overall family-wise error rate at $\alpha = 0.05$ without multiplicity penalty.
- **Failure Consequence:** Execute Pre-Registered Null Outcome Pathway (§26). External evaluation is locked and single-read; no retrospective model selection permitted.


---

# 19. Staged Hyperparameter Budget (Anti-Explosion Framework)

To prevent combinatorial grid explosion and GPU waste, V5 enforces a **strictly staged 3-tier execution framework**. Grid searches are never Cartesian products.

```text
Tier 1: Single Default Screening (1 seed, seed 42)
             ↓  (passes Gate A & Gate B)
Tier 2: Limited Local 1-D Ablation (2 adjacent values on 1 seed)
             ↓  (identifies optimum)
Tier 3: Finalist Multi-Seed Confirmation (3 seeds / 5-fold cross-fitting)
```

## Tier 1 — Pre-Registered Default Screening Configurations (Seed 42)
Every candidate representation evaluates exactly ONE pre-registered default configuration during initial screening:

- **Base Optimizer:** AdamW, cosine annealing, 30 epochs, patience=0.
- **Learning Rate:** Trunk $3\times 10^{-5}$ | Auxiliary Heads $1\times 10^{-4}$.
- **Weight Decay:** $1\times 10^{-2}$ (established V4 recipe default).
- **Escalation Loss Weight (V5-B2):** Default $\lambda_{esc} = 0.50$.
- **Context Retention (V5-B5):** Default $\alpha = 0.25$ ($1.25\times$ crop expansion).
- **Pairwise MEL-vs-NV Ranking (V5-B12):** Default margin $m = 0.20$, weight $\lambda_{rank} = 0.10$.
- **SupCon (V5-B7):** Default temperature $\tau = 0.07$, weight $\lambda_{supcon} = 0.10$.
- **Group-DRO (V5-A3):** Default step size $\eta_q = 0.01$, groups $G = \text{age\_band} \times \text{escalation}$.
- **GeM Pooling (V5-B11):** Default initialization $p = 3.0$ (learnable).
- **Local Micro-Patches (V5-B4):** Default size $128\times 128$ pixels (4 native-resolution crops).

## Tier 2 — Limited Local 1-D Ablations (Conditional)
Only triggered if Tier 1 screening successfully passes Gate A ($\Delta pAUC \ge +0.050$) and Gate B ($\Delta \text{Macro-F1} \ge -0.010$):
- **Escalation weight:** Test $\lambda_{esc} \in \{0.25, 1.00\}$ (2 runs, seed 42).
- **Context alpha:** Test $\alpha \in \{0.10, 0.50\}$ (2 runs, seed 42).
- **Ranking margin:** Test $m \in \{0.10, 0.50\}$ (2 runs, seed 42).
- **Patch resolution:** Test $\{96\times 96, 160\times 160\}$ (2 runs, seed 42).

## Tier 3 — Finalist Multi-Seed Confirmation
Only the single best surviving configuration per family advances to Tier 3 (3 seeds: 42, 43, 44; or full 5-fold cross-fitting).

No unconstrained Optuna searches or unbudgeted grid sweeps are permitted.

---

# 20. Required Metrics

Every V5 development run must report:

## Primary

- Macro-F1 (7-class unweighted harmonic mean);
- under-40 malignant pAUC ($pAUC_{FPR \le 0.20}$);
- under-40 sensitivity at declared operating point.

> [!IMPORTANT]
> **Subgroup Sample Size & Power Limitation:** In the development OOF cohort, under-40 escalating lesions correspond to only approximately 34 independent lesions (64 images). Consequently, development-stage statistical uncertainty is inherently large.
> 
> The plan explicitly pre-registers:
> > *“Under-40 OOF performance is a development-stage directional signal rather than a precise confirmatory estimate. The final confirmatory endpoint requires the fresh external cohort.”*
> 
> Under-40 development estimates must never be treated as possessing the evidential strength of an adequately powered external cohort. Confirmatory claims are strictly reserved for the fresh external validation (Stage 8).

## Class-specific

- MEL F1;
- BCC sensitivity/recall;
- AKIEC F1;
- BKL F1;
- NV F1;
- DF F1;
- VASC F1.

## Global

- balanced accuracy;
- macro ROC-AUC;
- macro PR-AUC.

## Calibration

- ECE;
- MCE;
- subgroup ECE.

## Safety

- missed malignant cases;
- sensitivity at fixed specificity;
- sensitivity at matched referral;
- false reassurance rate where applicable.

## Ensemble

- disagreement;
- Yule's Q;
- double-fault.

---

# 21. No-Repeat Registry

Create before the first V5 GPU run:

`results/v5/v5_no_repeat_registry.json`

Each entry must contain:

```text
proposal
historical_equivalent
historical_status
historical_artifact
new_mechanistic_difference
new_endpoint
why_not_duplicate
decision
```

A proposal may only enter the active queue if:

- it is genuinely new;
- it fixes a known defect;
- it tests a different mechanism;
- or it is an external validation of an existing result.

### Methods Permanently Barred from Repetition
Do NOT add or execute any new experiment that is materially equivalent to any of the following 12 falsified, invalid, or exhausted mechanisms:

1. **Frozen specialist head:** (S40/S42/S67 falsified; frozen features cannot be repaired by shallow classifiers).
2. **Tabular metadata fusion:** (S33/S53r R7/S67 null; clinical metadata does not improve latent visual ranking).
3. **Adversarial age invariance:** (V3 Phase D/S45 falsified; destroys essential diagnostic age-correlated signal).
4. **Frozen PanDerm probe:** (S51 falsified; linear probe on frozen dermatology ViT underperformed ConvNeXt by $\Delta pAUC = -0.0363$).
5. **Frozen DINOv2 probe:** (S51 falsified; $\Delta pAUC = -0.0725$).
6. **Global prior correction:** (S58 null; global prevalence adjustments do not fix subgroup rank inversion).
7. **Post-hoc age penalty $\lambda(\text{age})$:** (S57b/S65/S66 ceiling; shifts referrals without generating ranking discrimination).
8. **Per-band threshold offsets:** (S56/S65 ceiling; decision offsets hit mathematical ROC limit).
9. **Mahalanobis smartphone admissibility:** (S69 failed; 79% out-of-scope rejection with +0.3514 risk gap, superseded by Modality Classifier).
10. **Redundant ridge stacking:** (S9/S59 non-generalizing; test advantage not replicated in validated OOF selection).
11. **Buggy colour constancy:** (S50/S53 invalid; corrected Grey World was null in S53r).
12. **Repeated reserved-cohort evaluation:** (S70 audit; reserved cohort read 8 times across V4 and is completely exhausted for confirmatory claims).

If any method matching these mechanisms is proposed, it must be classified as a duplicate and rejected unless the mechanism is proven to be fundamentally distinct.

---

# 22. Provenance Requirements

Every V5 run records:

- git commit;
- data manifest hash;
- split hash;
- plan hash;
- random seed;
- architecture;
- input resolution;
- preprocessing;
- loss;
- hyperparameters;
- checkpoint hash;
- train time;
- peak VRAM;
- evaluation split;
- test-read status;
- reserved-read status;
- external-read status.

Results are append-only.

Never overwrite an old experiment to make a new result look like a continuation.

---

# 23. Hardware Strategy & Seed Policy

Current repository evidence indicates the RTX 5050 Laptop GPU can support 384px runs, but 384px is materially more expensive than 224px (approximately $2.26\times$ runtime and memory scaling).

Hardware runtime execution ladder:
1. Fast smoke test (1 epoch on fold 0);
2. One-seed development screening (seed 42);
3. Shortlist filtering via pre-registered acceptance gates;
4. Multi-seed confirmation (3 seeds);
5. Full 5-fold cross-validation only for shortlisted finalists.

Use mixed precision (AMP on). Keep the existing conservative data-loader settings (`--num-workers 2`, batch 16 $\times$ grad accum 2 fallback) that were shown to avoid Windows OS error 1455 paging crashes. Do not launch multiple memory-heavy 384px training jobs concurrently.

### Seed Policy
- **Stage 1 (384 Control):** Minimum **3 seeds** (seeds 42, 43, 44) required to establish the baseline and rule out seed-variance confounding ($\sigma_{seed} \ge 0.072$).
- **Stage 2 (Screening):** **1 seed** (seed 42) for initial screening of new representation hypotheses.
- **Stage 2 Survivors:** **3 seeds** for candidates meeting Gate A ($\Delta pAUC \ge +0.05$) and Gate B ($\Delta \text{Macro-F1} \ge -0.01$).
- **Finalists (Stages 4–6):** Full **5-fold cross-validation** across all partitioned training data.
- **Stopping Rule:** Do NOT spend 3 seeds on ideas that fail the initial 1-seed screening gate. Strictly record why each candidate was promoted or stopped in the experiment ledger.

---

# 23b. Capacity-Control Principle

Whenever V5 modifies network architecture by introducing additional capacity:
- multi-stream branches (dual-stream global + local);
- multi-resolution micro-patch encoders;
- discrete wavelet transform (DWT) branches;
- auxiliary escalation or morphology classification heads;
- multi-expert networks or morphology-gated ensembles;

the experimental plan must explicitly separate:
1. **Information gain** (deriving from multi-scale or biological visual features);
2. **Model capacity gain** (deriving simply from increased parameter count, wider receptive field, or extra training compute).

### Requirements:
- For every architectural addition, evaluate against a **capacity-matched single-stream control** (e.g. wider/deeper backbone with equal parameter budget and FLOPs) where practical.
- Every architectural experiment report must record:
  1. Trainable parameter count ($N_{params}$ and $\Delta N_{params}$ vs control);
  2. Approximate FLOPs / compute budget ($GFLOPs$);
  3. Peak GPU training memory (VRAM in GB);
  4. Wall-clock training duration per epoch.

---

# 24. Exact V5 Priority Queue & Run Architecture

The execution runsheet enforces a strictly gated sequence designed to maximize information gain while avoiding redundant GPU allocation:

### Core Progression (Gated Stages):
1. **Priority 1 (Stage 0):** `V5-S00-INTEGRITY` — Repository, split, manifest, and segmentation mask integrity audit (CPU).
2. **Priority 2 (Stage 1):** `V5-S01-384-CONTROL` — 384px ConvNeXt-Tiny control confirmation (3 seeds, GPU).
3. **Priority 3 (Stage 2):** High-Value Representation Tests:
   - `V5-S02-DUALSTREAM` — Dual-stream global 384 + lesion crop 384 (with capacity-matched control).
   - `V5-S03-ESCALATION` — Trainable multi-task escalation head ($\lambda_{esc} \in \{0.25, 0.5, 1.0\}$).
   - `V5-S04-HIERARCHICAL` — Hierarchical coarse-to-fine diagnosis tree.
   - `V5-S05-LOCALPATCH` — Multi-resolution native-resolution micro-patches (96–160px).
4. **Priority 4 (Stage 3):** Biological Representation Refinements (applied strictly to Stage 2 survivors):
   - `V5-S06-CONTEXT` — Soft boundary / interior / context decomposition ($\alpha \in [0, 1]$).
   - `V5-S07-HARDNEG` — Development OOF hard-negative mining / JTT reweighting.
   - `V5-S08-SUPCON` — Supervised contrastive representation learning.
   - `V5-S09-MELNV-RANK` — Pairwise melanoma-vs-nevus margin ranking loss ($L_{rank}$).
   - `V5-S10-DWT` — High-frequency Haar wavelet sub-band branch (parameter-matched).
   - `V5-S11-MORPH-ATTR` — Auxiliary supervision with expert morphology attributes.
   - `V5-S12-GEM` — Adaptive Generalized Mean (GeM) pooling ablation ($p=3$).
5. **Priority 5 (Stage 4):** `V5-S13-GROUPDRO` — Subgroup-robust Group-DRO on the single best surviving representation.
6. **Priority 6 (Stage 5):** Modern Heterogeneous Backbones:
   - `V5-S14-CONVNEXTV2` — ConvNeXt-V2 with Global Response Normalization.
   - `V5-S15-EFFNETV2` — EfficientNetV2-M.
   - `V5-S16-SWINV2` — SwinV2-384 hierarchical self-attention.
7. **Priority 7 (Stage 6):** `V5-S17-ENSEMBLE` — Error-diverse compact soft-voting ensemble.
8. **Priority 8 (Stage 7):** Existing Safety & Calibration Stack:
   - `V5-S18-TTA-CAL` — 24-view TTA (with cost-aware dev checkpoint) + S55 groupwise multi-calibration.
   - `V5-S19-SAFETY` — Conformal prediction safety net + S69 modality admissibility gate.
9. **Priority 9 (Stage 8):** `V5-S20-EXTERNAL` — Final locked confirmatory read on fresh external cohort (S75).

### Secondary & Deferred Queue (Executed only on development justification):
- `V5-X1` — 384px Mixup/CutMix sensitivity analysis (Exploratory / Low Priority).
- `V5-X2` — Optional SWA schedule ablation (Deferred / Exploratory).
- `V5-X3` — Exploratory imbalance-loss sensitivity / Focal Loss (Deferred / Low Priority).
- `V5-X4` — Balanced Softmax subgroup loss (Deferred — Not Core V5).
- `V5-X5` — Derm foundation distillation (Deferred — Not Core V5).

---

# 25. What Must Not Be Called a V5 Discovery

The following are historical facts only:

- 0.8047 Macro-F1 from the old calibrated V1 test result;
- 0.993 in-sample feature-probe result;
- old MaxViT test score;
- repeated reserved-cohort results as if they were pristine confirmation;
- any result derived from the old buggy colour-constancy implementation;
- any result obtained by selecting after looking at the locked test set.

These must never be recycled as evidence for a new V5 claim.

---

# 26. Expected Scientific Contribution

The paper should only claim a stronger contribution if V5 actually demonstrates it.

The intended hypothesis is:

> A high-resolution lesion-aware representation, augmented with an explicit escalation objective and biologically grounded local morphology learning, improves malignant ranking in the age-stratified dermoscopic setting without relying on post-hoc demographic thresholding.

Potential secondary contribution:

> A reproducible diagnostic pipeline can distinguish representation limitations from calibration, decision-policy and external-transport limitations through a sequence of pre-registered falsification experiments.

Do not state either claim as established before the results exist.

### Pre-Registered Null Outcome Pathway
A null result is a legitimate and rigorous scientific outcome; negative findings must not be treated as wasted work or masked by post-hoc threshold tweaks.

The study formally pre-registers the following null conclusion:
> **“If V5 representation interventions fail to produce reproducible improvement beyond the pre-specified MCID, the study will conclude that no evidence was obtained that the tested representation-learning mechanisms overcome the under-40 ranking limitation on the available development data. Alternative explanations, including limited subgroup sample size, case-mix differences, intrinsic visual ambiguity, acquisition effects, and unmeasured clinical factors, remain possible.”**

The study explicitly rejects the overly broad conclusion that *“the problem is caused entirely by the data distribution.”* Instead, a null outcome establishes that the tested architectural, multiscale, and representation interventions did not extract a discriminative signal sufficient to overcome the ranking ceiling on this cohort.

---

# 27. Required V5 Artifacts

Antigravity must create:

1. `results/v5/v5_no_repeat_registry.json`
2. `results/v5/v5_freeze_manifest.json`
3. `results/v5/v5_morphology_availability.csv`
4. `results/v5/v5_experiment_ledger.csv`
5. `docs/V5_BIOLOGY_TO_CV.md`
6. `docs/V5_RESULTS_SUMMARY.md`
7. `docs/V5_EXECUTION_RUNBOOK.md`

At the end of V5, also generate:

- `docs/V5_FINAL_VERDICT.md`
- `docs/V5_FAILED_EXPERIMENTS.md`
- `docs/V5_REPRODUCTION.md`

---

# 28. Implementation Summary to Carry Forward Into V5

This is the concise “what has already been implemented” record.

## Data and integrity

Implemented:

- lesion-grouped splitting;
- leakage assertions;
- test-read receipts;
- frozen analysis plans;
- OOF infrastructure;
- reproducibility/ledger mechanisms.

## Core vision system

Implemented:

- six CNN backbones;
- uniform soft voting;
- 24-view TTA;
- Grad-CAM/explainability tooling;
- evaluation and bootstrap tooling.

## Probability/safety

Implemented:

- temperature scaling;
- matrix scaling;
- Dirichlet calibration;
- groupwise calibration;
- conformal prediction;
- bipartite/equalized safeguards;
- selective prediction;
- age-conditional decision rules.

## V3/V4 diagnostic research

Implemented and evaluated:

- age-entanglement probes;
- archive/site probes;
- spatial shortcut analysis;
- specialist-head probes;
- metadata fusion experiments;
- foundation-model probes;
- representation/recipe ladder;
- 384px experiment;
- K-fold OOF infrastructure;
- under-40 ceiling analysis;
- target-side threshold recalibration;
- smartphone modality admissibility.

## Phase-Y completion

Implemented and banked:

- S68 target-side referral analysis;
- S69 modality classifier;
- S71 K-fold partition/OOF assembly;
- S72 five-fold 224px control training;
- S70 fresh-cohort decision checkpoint.

## Current deployment/engineering state

Implemented:

- six-model torch inference path;
- service self-tests;
- drift-hook computation;
- modality admissibility gate.

Known limitations remain:

- service model packaging is not a complete ONNX production export;
- several drift hooks remain analysis hooks rather than continuous production monitoring;
- fresh external confirmation remains pending.

---

# 29. Final V5 Execution Principle

V5 is not a larger version of the old experiment list.

It is a controlled attempt to answer one unresolved question:

> **Can the model be made better at seeing the fine morphology of malignant lesions, especially the subtle morphology that matters in younger patients, without destroying the diagnostic signal that the earlier safety programme showed to be load-bearing?**

Everything in V5 must serve that question.

The optimal workflow is:

```text
384 control
      ↓
lesion-aware representation
      ↓
explicit escalation representation
      ↓
multi-scale local morphology
      ↓
subgroup-robust training
      ↓
modern heterogeneous backbones
      ↓
error-diverse ensemble
      ↓
TTA + groupwise calibration + conformal safety
      ↓
fresh external confirmation
```

No shortcut, threshold, calibration map, or previously falsified mechanism may be presented as a substitute for improving the visual representation.

---

# References for biological grounding

1. DermNet — Melanoma dermoscopy:
   https://dermnetnz.org/cme/dermoscopy-course/dermoscopy-of-melanoma

2. DermNet — Basal cell carcinoma dermoscopy:
   https://dermnetnz.org/topics/basal-cell-carcinoma-dermoscopy

3. DermNet — Actinic keratosis dermoscopy:
   https://dermnetnz.org/topics/actinic-keratosis-dermoscopy

4. DermNet — Seborrhoeic keratosis dermoscopy:
   https://dermnetnz.org/cme/dermoscopy-course/dermoscopy-of-seborrhoeic-keratosis

5. Dermoscopy of Melanoma According to Age Groups, retrospective monocentric study of 285 histopathologically confirmed melanomas:
   https://pubmed.ncbi.nlm.nih.gov/40805292/

6. Dermoscopy of dermatofibromas: prospective morphological study of 412 cases:
   https://pubmed.ncbi.nlm.nih.gov/18209171/
