# When Safety Nets Fail: Subgroup-Conditional Calibration, Conformal Guarantees, and Age-Stratified Blind Spots in Dermoscopy Ensembles

<p align="center">
  <a href="#executive-summary"><img src="https://img.shields.io/badge/Status-IEEE_TMI_Target-blue.svg?style=for-the-badge&logo=ieee" alt="Status: IEEE TMI Target"></a>
  <a href="#strict-research-integrity--audit-gates"><img src="https://img.shields.io/badge/Audit-357%2F357_Passed-success.svg?style=for-the-badge&logo=checkmarx" alt="Audit: 357/357 Passed"></a>
  <a href="#multi-centre-replication--transportability"><img src="https://img.shields.io/badge/Multi--Centre-14%2C885_Lesions-purple.svg?style=for-the-badge&logo=databricks" alt="Multi-Centre: 14,885 Lesions"></a>
  <a href="#zero-leakage-protocol"><img src="https://img.shields.io/badge/Leakage_Guard-0_Cross--Split_Leaks-brightgreen.svg?style=for-the-badge&logo=shield" alt="Leakage Guard: 0 Leaks"></a>
  <a href="#how-to-reproduce"><img src="https://img.shields.io/badge/Python-3.12_%7C_PyTorch_2.11-yellow.svg?style=for-the-badge&logo=python" alt="Python 3.12 | PyTorch 2.11"></a>
</p>
<p align="center">
  <a href="#post-manuscript-falsification-programme-v2--v3"><img src="https://img.shields.io/badge/Falsification-5_of_6_Hypotheses-critical.svg?style=flat-square&logo=target" alt="Falsification: 5 of 6 hypotheses"></a>
  <a href="#strict-research-integrity--audit-gates"><img src="https://img.shields.io/badge/Pre--Registered-SHA256_Frozen_Plans-informational.svg?style=flat-square&logo=gitbook" alt="Pre-registered SHA256 frozen plans"></a>
  <a href="#strict-research-integrity--audit-gates"><img src="https://img.shields.io/badge/Test_Reads-2_(Locked)-important.svg?style=flat-square&logo=lock" alt="Test reads: 2, locked"></a>
  <a href="#an-endpoint-retired-on-evidence"><img src="https://img.shields.io/badge/Negative_Results-Retained-blueviolet.svg?style=flat-square&logo=bookstack" alt="Negative results retained"></a>
</p>

---

## Executive Summary & Central Thesis

Automated dermoscopy classification benchmarks have saturated on aggregate metrics like accuracy and Macro-F1. However, standard accuracy on benchmarks like **HAM10000** is deceptive: melanocytic nevi (`nv`) constitute **66.8%** of cases, allowing a naive predictor guessing "benign mole" every time to achieve ~67% accuracy while missing **100%** of lethal melanomas.

This repository hosts a publication-grade research framework and safety audit pipeline investigating classification performance, ensemble diversity, uncertainty calibration, conformal coverage guarantees, and multi-centre generalization across **14,885 lesions** from four medical institutions (**HAM10000**, **BCN20000**, **MSKCC**, and **PAD-UFES-20**).

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                        CENTRAL THESIS                                             ║
║                                                                                                   ║
║  Aggregate discrimination benchmarks conceal severe, confidently-wrong subgroup triage failures   ║
║  driven by demographic training priors. Post-hoc calibration, abstention, and marginal conformal  ║
║  guarantees fail to protect vulnerable sub-cohorts, but risk can be partially mitigated by        ║
║  group-conditional decision rules explicitly priced in clinical biopsy burden.                    ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝
```

### Key Discoveries & Headline Numbers

1. **The Architectural Ceiling & Ensembling Power**: Across 6 heterogeneous CNN backbones (ConvNeXt-Tiny/Small, DenseNet-121, EfficientNet-B0/B3, ResNet-50), single-model discrimination reaches Macro-F1 **0.7459** (ConvNeXt-Tiny). An arithmetic soft-vote ensemble achieves Macro-F1 **0.7718** (McNemar Holm $p = 2.0 \times 10^{-4}$)—the *only* statistically certifiable rung on the ablation ladder. 24-view Test-Time Augmentation (TTA) lifts Macro-F1 to **0.7859**, and Dirichlet recalibration reaches **0.8047**. Standalone vision transformers, multimodal tabular metadata fusion, and five post-hoc combination rungs (including 30-member fold-bagging) all show negative or unresolvable gains.
2. **The Calibration-Sensitivity Antinomy**: The soft-vote ensemble is systematically **under-confident** (mean confidence 0.7048 vs accuracy 0.8609; signed gap $-0.156$), which explains why multi-class Dirichlet calibration dramatically outperforms single-parameter temperature scaling (cutting Expected Calibration Error from **0.1575 $\to$ 0.0206**). However, calibration introduces a severe clinical trade-off: redistributing probability mass to the majority nevus class drops escalation sensitivity from **0.7862 $\to$ 0.7310**, generating **16 additional missed malignancies**.
3. **The Under-40 Melanoma Blind Spot**: The ensemble exhibits a catastrophic age-stratified blind spot: escalation sensitivity collapses to **0.143 [0.030, 0.363]** in patients aged $<40$ (missing 18 of 21 malignancies) compared to **0.764 [0.683, 0.836]** in patients $\ge 60$ (difference $-0.621$, Holm $p < 0.001$). This is driven by extreme training prior skew (4.9% escalating under 40 vs. 35.5% at 60+), leading the network to use age as a diagnostic shortcut.
4. **Failure of Standard Safety Nets**:
   - **Selective Abstention**: Because errors in young patients are *confidently wrong* (not uncertain), margin-based abstention defers only **11.1%** (2 of 18) of missed young malignancies at a 10% referral budget.
   - **Conformal Guarantees**: A marginal split-conformal guarantee at $\alpha = 0.10$ achieves 90.1% overall coverage, but covers only **23.8%** of malignant lesions in patients under 40. Only **equalized bipartite conformal prediction** (conditioning calibration quantiles jointly on age band $\times$ escalation requirement) restores young malignant coverage to **95.2%**.
5. **Mitigation Priced in Biopsy Burden**: An out-of-fold cross-fitted age-conditional decision rule ($\hat{y} = \arg\max_c [p_c + \lambda_{b} \mathbf{1}_{c \in \mathcal{E}}]$) raises all-ages sensitivity from **0.731 $\to$ 0.831**, cutting missed malignancies from 78 to 49. In clinical currency, this increases the Number Needed to Biopsy (NNB) from **$3.0 \to 6.2$** at 3% reference prevalence. Under-40 sensitivity improves to **0.238 [0.082, 0.472]**—mitigating, but not closing, the blind spot.
6. **Multi-Centre External Generalization**: Frozen evaluation across **11,982 BCN20000** and **2,903 MSKCC** external dermoscopy lesions reveals a transportability paradox: the $\lambda$-rule operating point transports (consistently lifting under-40 sensitivity across all centres), but pre-registered causal prior-shift hypotheses fail: **the operating policy transports, but the explanation does not**.
7. **The Age Shortcut Is Load-Bearing, Not Removable**: A dedicated falsification programme (V3) established that the representation is *certifiably* age-entangled — it decodes age band at AUC **0.6922 [0.6638, 0.7187]**, and the escalation score rides on age even with the true class held fixed (`age_residual` **+0.1226 [+0.0924, +0.1513]**). Yet **removing that entanglement makes the blind spot worse**: adversarial age-invariance training drove `age_residual` to **$-0.0452$** (sign flipped) and simultaneously cut under-40 escalation ranking AUC from **0.8249 $\to$ 0.7201**. The age signal is **load-bearing diagnostic signal, not a separable nuisance** — which is why every logit-level remedy has failed.


---

## End-to-End System Architecture


```
                                              ┌──────────────────────────────────────────────┐
                                              │          Input Dermoscopy Image (x)          │
                                              └──────────────────────┬───────────────────────┘
                                                                     │
                                    ┌────────────────────────────────┴───────────────────────────────┐
                                    ▼                                                                ▼
                    ┌───────────────────────────────┐                                ┌───────────────────────────────┐
                    │     Model 1: ConvNeXt-Tiny    │                                │     Model 4: ResNet-50        │
                    │     Model 2: EfficientNet-B0  │  ────── 6 Checkpoints ───────► │     Model 5: DenseNet-121     │
                    │     Model 3: ConvNeXt-Small   │   (Two-Stage Fine-Tuned)       │     Model 6: EfficientNet-B3  │
                    └───────────────┬───────────────┘                                └───────────────┬───────────────┘
                                    └────────────────────────────────┬───────────────────────────────┘
                                                                     │  6 Softmax Probability Vectors
                                                                     ▼
                                              ┌──────────────────────────────────────────────┐
                                              │      Arithmetic Soft-Voting Ensemble         │
                                              │     p_ens = 1/6 ∑ p_m  (Macro-F1 0.7718)      │
                                              └──────────────────────┬───────────────────────┘
                                                                     │
                                                                     ▼
                                              ┌──────────────────────────────────────────────┐
                                              │       24-View Test-Time Augmentation         │
                                              │   (8 Dihedral Flips × 3 Scales: 0.9, 1.0, 1.1)│
                                              └──────────────────────┬───────────────────────┘
                                                                     │  p_tta (Macro-F1 0.7859)
                                                                     ▼
                                              ┌──────────────────────────────────────────────┐
                                              │         Multi-Class Dirichlet Calibrator     │
                                              │  p_cal = σ(W · ln(p) + b), ECE: 0.158 → 0.021│
                                              └──────────────────────┬───────────────────────┘
                                                                     │
                                    ┌────────────────────────────────┴───────────────────────────────┐
                                    │                                                                │
                                    ▼                                                                ▼
    ┌──────────────────────────────────────────────────────────────┐ ┌──────────────────────────────────────────────────────────────┐
    │              Equalized Bipartite Conformal Sets              │ │              Age-Conditional Decision Rule (λ)             │
    │  Condition quantiles jointly on: age_band × {benign, serious}│ │     ŷ = argmax_c [ p_c + λ_band · 1(c ∈ {MEL,BCC,AK}) ]      │
    │  • Guarantees 95.2% malignant coverage in patients < 40      │ │     • Tuned OOF with 85% specificity floor                 │
    │  • False Reassurance Rate (FRR): 0.014 [0.004, 0.035]        │ │     • Overall Sensitivity: 0.731 → 0.831 (NNB: 3.0 → 6.2)  │
    └───────────────────────────────┬──────────────────────────────┘ └───────────────────────────────┬──────────────────────────────┘
                                    │                                                                │
                                    └────────────────────────────────┬───────────────────────────────┘
                                                                     ▼
                                              ┌──────────────────────────────────────────────┐
                                              │          Clinical Action & Triage            │
                                              │  • Direct Specialist Referral / Biopsy       │
                                              │  • Selective Abstention / Dermatology Review │
                                              │  • Low-Risk Primary Care Discharge           │
                                              └──────────────────────────────────────────────┘
```

> **This pipeline is unchanged by Phase 6.** The falsification programme tested four candidate
> modifications — multi-archive training, head refitting, fold-bagging and adversarial
> age-invariance — and **none was adopted**, because none cleared its pre-registered gate. The
> deployed system remains the 6-CNN soft-vote with TTA, Dirichlet calibration, bipartite conformal
> sets and the frozen $\lambda$ rule.

---

## Empirical Benchmark Results

### 1. In-Domain Single Backbones vs. Ensembling Ladder (HAM10000 Test, N=1,502)

All baseline models were trained with identical random seeds (42), effective-number class weighting, two-stage transfer learning, and evaluated under leak-free lesion-grouped partitioning:

| Ladder Rung | Architecture / Technique | Macro-F1 [95% CI] | Balanced Acc | Escalation Sens. | ECE | Missed Serious (of 290) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline 1** | ResNet-50 | 0.7058 [0.655, 0.754] | 0.7222 | 0.7379 | 0.1012 | 76 |
| **Baseline 2** | DenseNet-121 | 0.6974 [0.648, 0.744] | 0.7342 | 0.7379 | 0.0489 | 76 |
| **Baseline 3** | EfficientNet-B0 | 0.7257 [0.677, 0.771] | 0.7394 | 0.7862 | 0.0322 | 62 |
| **Baseline 4** | EfficientNet-B3 | 0.6892 [0.638, 0.739] | 0.7132 | 0.7276 | 0.0478 | 79 |
| **Baseline 5** | ConvNeXt-Small | 0.7252 [0.677, 0.772] | 0.7281 | 0.7138 | 0.0611 | 83 |
| **Baseline 6** | **ConvNeXt-Tiny (Best Single)** | **0.7459** [0.700, 0.791] | **0.7792** | 0.7759 | 0.0967 | 65 |
| **Ladder A7** | **6-CNN Uniform Soft-Vote** | **0.7718** [0.727, 0.814] | 0.7942 | 0.7621 | 0.1575 | 69 |
| **Ladder A7+TTA**| + 24-View Test-Time Augmentation | **0.7859** [0.743, 0.826] | 0.8037 | **0.7862** | 0.1345 | **62** |
| **Ladder A8** | **+ Dirichlet Calibration (Final)**| **0.8047** [0.764, 0.842] | **0.8073** | 0.7310 | **0.0206** | 78 |

> **Key Statistical Finding**: Soft-voting achieves a statistically significant improvement over ConvNeXt-Tiny (McNemar Holm-adjusted $p = 2.0 \times 10^{-4}$). In contrast, SwinV2-Tiny (Macro-F1 0.7273, $p = 0.38$) and multimodal patient metadata fusion (Macro-F1 0.7411, $p = 0.79$) failed to surpass the best CNN.

---

### 2. The Age-Stratified Blind Spot & Safety Net Breakdown

Evaluation stratified across patient age brackets on held-out test data reveals the extreme demographic disparity:

| Metric | Patient Age < 40 | Patient Age 40–59 | Patient Age $\ge 60$ | Demographic Disparity ($\Delta$) |
|---|:---:|:---:|:---:|:---:|
| **Malignant Prevalence in Training** | 4.9% (68 / 1,399) | 16.7% (385 / 2,306) | 35.5% (1,069 / 3,010) | **7.2× prior skew** |
| **Test Serious Lesions ($N$)** | 21 | 100 | 169 | — |
| **Argmax Escalation Sensitivity** | **0.143** [0.030, 0.363] | **0.760** [0.670, 0.840] | **0.764** [0.683, 0.836] | **−0.621 (collapses under 40)** |
| **Missed Malignancies** | **18 of 21 (85.7% missed)**| 24 of 100 | 40 of 169 | — |
| **Abstention Referral Rate on Misses** | **11.1% (2 of 18)** | 33.3% (8 of 24) | 37.5% (15 of 40) | **Confidently wrong** |
| **Marginal Conformal Coverage ($\alpha=0.10$)**| **23.8% (5 of 21)** | 70.0% (70 of 100) | 81.7% (138 of 169) | **−57.9% deficit** |
| **Bipartite Conformal Coverage ($\alpha=0.10$)**| **95.2% (20 of 21)** | 91.0% (91 of 100) | 90.5% (153 of 169) | **Restored to safety floor** |
| **Post-λ Rule Sensitivity** | **0.238** [0.082, 0.472] | **0.860** [0.785, 0.923] | **0.888** [0.833, 0.933] | **+0.095 gain (partially mitigated)**|

---

### 3. Conformal Coverage & False Reassurance Across Methods

Finite-sample prediction set guarantees evaluated on HAM10000 held-out test split:

| Method | Target $\alpha$ | Calibration Level | Marginal Coverage | Malignant Coverage | Under-40 Malignant Coverage | Mean Set Size | False Reassurance Rate (FRR) |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **LAC** | 0.10 | Marginal | 90.4% | 74.8% | 23.8% | 1.34 | 0.057 |
| **LAC** | 0.10 | Mondrian (Class-Cond.) | 91.1% | 91.4% | 57.1% | 1.62 | 0.022 |
| **RAPS**| 0.10 | Mondrian (Class-Cond.) | 90.9% | 94.1% | 66.7% | 1.58 | 0.015 |
| **LAC** | 0.10 | **Equalized Bipartite** | 90.8% | **94.8%** | **95.2%** | 1.76 | 0.013 |
| **RAPS**| 0.05 | **Equalized Bipartite** | **95.4%** | **98.3%** | **95.2%** | **1.87** | **0.014 [0.004, 0.035]** |

---

### 4. Multi-Centre External Transportability (N=14,885 Lesions)

Evaluating the frozen HAM-trained ensemble and age-conditional decision rule across independent clinical domains:

| Cohort | Clinical Modality | Number of Images | Baseline Argmax Sens. (<40) | Post-λ Rule Sens. (<40) | Absolute Δ Sensitivity | Transportability Result |
|---|---|:---:|:---:|:---:|:---:|---|
| **HAM10000** | Primary Dermoscopy (In-Domain Test) | 1,502 | 0.143 | **0.238** | **+0.095** | In-domain mitigation |
| **BCN20000** | External Dermoscopy (Hospital Clínic Barcelona) | 11,982 | 0.279 | **0.352** | **+0.073** | Policy transports successfully |
| **MSKCC** | External Dermoscopy (Memorial Sloan Kettering) | 2,903 | 0.333 | **0.389** | **+0.056** | Policy transports successfully |
| **PAD-UFES-20**| Clinical Smartphone Photography | 2,298 | 0.000 (Mel. Recall) | 0.625 (Warm-Start) | +0.625 | Modality collapse; OOD detected |

> **Takeaway**: Class-conditional Mahalanobis distance separates dermoscopy from smartphone clinical photography with **AUROC 0.913**, providing a reliable tripwire against unintended modality shift.

> **Note on scope.** In Sections 1–4 above, BCN20000 and MSKCC serve as **frozen external
> evaluation** cohorts only. Section 5 promotes them to **training archives** — a distinct
> capability introduced by the multi-archive corpus below.

---

### 5. Multi-Archive Training Corpus (`ml/data/manifest_v3.csv`)

A unified **24,900-image / 13,809-lesion** corpus spanning three dermoscopy archives, built for the
Phase 6 experiments. Image IDs and lesion IDs were checked for collisions across archives rather
than assumed disjoint (HAM images also live in the ISIC archive), and MSKCC's 2,084 null lesion IDs
become **singleton** clusters rather than one shared "unknown" group.

| Archive | Images | Role before Phase 6 | Role in Phase 6 |
|---|--:|---|---|
| **HAM10000** | 10,015 | train / val / test | train / val / test (inherited byte-identically) |
| **BCN20000** | 11,982 | external evaluation | **training archive** + held-out |
| **MSKCC** | 2,903 | external evaluation | **training archive** + held-out |
| **Total** | **24,900** | — | 13,809 lesion clusters |

Four training conditions isolate *sample size* from *domain breadth*. **Every condition validates
on the same fixed 1,532-image HAM val set**, so no condition selects against a different target and
the differences are attributable to training composition alone. No external image ever enters val
or test.

| Condition | Train images | Added | HAM-val Macro-F1 [95% CI] | ECE raw → Dirichlet | External Macro-F1 |
|---|--:|---|:---:|:---:|:---:|
| `ham_only` *(control)* | 6,981 | — | 0.7509 [0.699, 0.791] | 0.1178 → 0.0626 | 0.3564 |
| `ham_mskcc` | 9,010 | +2,029 MSKCC | **0.7869** [0.736, 0.826] | 0.1072 → **0.0392** | 0.3958 |
| `ham_bcn` | 15,396 | +8,415 BCN | 0.7614 [0.708, 0.804] | **0.0647** → 0.0420 | 0.5517 |
| `all_three` | 17,425 | +both | 0.7421 [0.685, 0.784] | 0.0796 → 0.0545 | **0.5754** |

> **Reading this table correctly.** `ham_mskcc` is the *nominal* winner, but it does **not** survive
> Holm correction over the declared family of three ($0.0220 \times 3 = 0.0660$), and it was not the
> pre-registered arm. `all_three`'s strong external column is **in-domain fit**, not robustness —
> the external holdout is 80% BCN. **No condition is certified better than the control**, which is
> why the safety stack was refit for all four rather than for a chosen winner.

```
                    PHASE 6 EXPERIMENTAL DESIGN — what each arm tests
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │  HAM10000 (10,015)      BCN20000 (11,982)         MSKCC (2,903)                   │
  └───────────┬──────────────────────┬──────────────────────┬────────────────────────┘
              │                      │                      │
              ▼                      ▼                      ▼
      ┌───────────────────────────────────────────────────────────┐
      │   manifest_v3.csv — 24,900 images / 13,809 lesion clusters │
      │   collision-checked · lesion-grouped · val+test frozen     │
      └───────────────────────────┬───────────────────────────────┘
                                  │
        ┌─────────────┬───────────┴───────────┬─────────────┐
        ▼             ▼                       ▼             ▼
   ┌─────────┐  ┌───────────┐          ┌───────────┐  ┌───────────┐
   │ham_only │  │ham_mskcc  │          │ ham_bcn   │  │all_three  │   PHASE C
   │ control │  │+sample sz │          │+rare class│  │ deployable│   (H4, H5)
   └────┬────┘  └─────┬─────┘          └─────┬─────┘  └─────┬─────┘
        └─────────────┴──────────┬───────────┴──────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │  SAME fixed HAM val (1,532)   │  ← gate: +0.03 Macro-F1, CI excl. 0
                 │  + external holdout (2,232)   │  ← split by cohort: in-domain vs zero-shot
                 └───────────────┬───────────────┘
                                 │  gate NOT met
                                 ▼
                 ┌───────────────────────────────┐
                 │  PHASE D — age-invariance     │   H6: remove the certified
                 │  gradient reversal on z(768)  │       entanglement and re-measure
                 └───────────────┬───────────────┘
                                 ▼
                    under-40 ranking AUC 0.8249 → 0.7201   ✗ worse
```

---

## Post-Manuscript Falsification Programme (V2 · V3)

After the manuscript was frozen, two further campaigns ran **not** to improve the headline number
but to test whether the under-40 blind spot could be *closed at all*. Both were pre-registered, and
**neither read the test split** — `results/test_pass_receipt.json` remains at `n_executions: 2`
throughout, and the six frozen `*_best.HAM-only.pt` checkpoints are byte-identical.

V3 (sessions S40–S47) posed six falsifiable hypotheses. **Five were falsified; one was certified
and then shown not to be a fixable cause.**

| | Hypothesis | Verdict | Decisive evidence |
|:--|:--|:--|:--|
| **H1** | The prior escalation-head result survives out-of-sample feature extraction | ❌ **Falsified** | $\Delta$pAUC $-0.0670$, CI $[-0.245, +0.072]$ — an in-sample extraction artifact |
| **H2** | A better head on the frozen representation recovers the gap | ❌ **Falsified** | Every matched $\Delta_{\text{head}}$ negative; none certified positive across linear / MLP / GBM probes |
| **H3** | The representation is age-entangled | ✅ **Certified** | `age_band` AUC **0.6922** [0.6638, 0.7187]; `age_residual` **+0.1226** [+0.0924, +0.1513] |
| **H4** | Pooling HAM + BCN + MSKCC beats the control by $\ge 0.03$ Macro-F1 | ❌ **Falsified** | $-0.0088$ $[-0.0554, +0.0354]$; no comparison survives Holm |
| **H5** | Archive breadth buys cross-archive robustness | ❌ **Falsified** | **0 of 2** genuine zero-shot transfer contrasts exclude zero |
| **H6** | Removing the entanglement improves under-40 ranking | ❌ **Falsified** | Entanglement removed, ranking AUC **fell** 0.8249 $\to$ 0.7201 |

### Why H4 and H5 are separate

A naive read of the pooled external endpoint suggests breadth buys robustness. It does not. The
external holdout is **80% BCN** (1,794 BCN / 438 MSKCC), so any BCN-trained condition is scored
largely on an archive it *trained on*. Disaggregating by cohort and labelling each cell from the
training composition isolates the only two honest cross-archive contrasts — a model trained on one
external archive, scored on the other, which it never saw:

| Contrast | $\Delta$ vs control | 95% CI | Verdict |
|:--|:--|:--|:--|
| `ham_mskcc` on BCN (never saw BCN) | $+0.0159$ | $[-0.0271, +0.0534]$ | null |
| `ham_bcn` on MSKCC (never saw MSKCC) | $+0.0260$ | $[-0.0011, +0.0543]$ | null |

while every *in-domain* cell is large and certain (e.g. `ham_bcn` on BCN $+0.2061$ $[+0.1318, +0.2566]$).
**Adding a second archive does not make the model robust to a third, unseen one.**

### An endpoint retired on evidence

Thresholded under-40 escalation sensitivity was **withdrawn as a discriminating endpoint**. Two
ConvNeXt-Tiny models trained on *byte-identical* data differ by **5 of 22** cases on it
(0.636 vs 0.409) — a same-data spread of **0.2273**, larger than the entire between-condition
spread of **0.1818**. It was replaced by under-40 escalation-mass **AUC pooled across HAM val and
the external holdout**, raising the positive count from 22 to **76**.

> **Net result.** V3 set out to break a 0.80 Macro-F1 ceiling. The best condition reaches
> **0.7869** and is not certified better than the control. The contribution is therefore
> *diagnostic and falsificatory*: the under-40 gap is not caused by a removable age shortcut, no
> head-level fix exists, archive breadth does not help, and the result that motivated three earlier
> arms was an artifact. The frozen plan is at `results/v3/analysis_plan_v3.json`; per-hypothesis
> verdicts are computed, not asserted, in `results/v3/final_verdict.json`.

---

## Strict Research Integrity & Audit Gates

This codebase enforces strict automated verification protocols to prevent data leakage, metric inflation, or hand-entered results:

1. **Lesion-Grouped Splitting (`assert_no_leakage`)**: Splitting is strictly performed on `lesion_id`, ensuring no patient has images shared between train, validation, and test splits.
2. **Single-Test-Read Discipline**: Evaluated through `research/testguard.py` using process-wide file locks and receipts stored in `results/test_pass_receipt.json`. Test sets are read only for final pre-registered confirmations.
3. **100% Scripted Table Reconstruction**: All manuscript numbers, p-values, confidence intervals, and tables are generated by reproducible code.
   - `python research/ablation/audit_manuscript.py`: Asserts 357 numerical checks across the published paper.
   - `python research/ablation/audit_manuscript.py --target paper/manuscript_edited.tex`: Validates the condensed 11-page manuscript (269 passed, 91 intentional skips, 0 failed).
   - `python research/ablation/verify_edited_tables.py`: Reconstructs 4 edited LaTeX tables (137 lines) byte-for-byte from underlying JSON/CSV artifacts.
   - `python research/ablation/validate_structure.py`: Validates citation, label, figure, and nested input resolution.
4. **Pre-Registration Before Data**: Every post-manuscript campaign freezes its hypotheses, gates
   and minimum clinically important difference *before* the runs that test them, with a
   self-verifying SHA256 (`results/v3/analysis_plan_v3.json`, `python -m research.v3.plan --check`).
   Deviations are numbered and logged (D1–D11) rather than silently applied.
5. **Certification Asymmetry**: An interval containing the null is reported `NOT_CERTIFIED`, never
   as evidence of absence. Negative results are retained and published, never discarded — six
   levers have now been killed on the record.
6. **Multiplicity Discipline**: Declared comparison families are Holm-corrected
   (`research/stats/families.py`). A nominal winner that does not survive correction is reported as
   a nominal winner, not promoted.

---

## Repository Map

```
Backend-S4D/
├── ml/                                 # Core Deep Learning Engine
│   ├── checkpoints/                   # Checkpoints & weights (HAM-only, PAD-only, PAD-warm)
│   ├── configs/                       # Class configurations & 7-class taxonomy definitions
│   ├── data/                          # Dataset indexing manifests (HAM10000, PAD-UFES-20)
│   ├── evaluation/                    # Metrics, confusion matrices, DeLong & McNemar tests
│   ├── preprocessing/                 # Lesion-grouped splitter, transforms, color constancy
│   └── training/                      # Two-stage fine-tuning engine, loss definitions
├── research/                           # Research Pipelines & Scientific Audit Suite
│   ├── ablation/                      # Verification scripts (audit_manuscript.py, validate_structure.py)
│   ├── agerule/                       # Subgroup-conditional lambda decision rule optimization
│   ├── calibration/                   # Temperature scaling, matrix scaling, Dirichlet calibrators
│   ├── conformal/                     # Split, Mondrian, and equalized bipartite conformal prediction
│   ├── dca/                           # Decision Curve Analysis & Net Benefit calculations
│   ├── ensembling/                    # Arithmetic soft-vote, stacking, Caruana greedy search
│   ├── external/                      # Multi-centre replication engine (BCN20000, MSKCC, PAD-UFES)
│   ├── selective/                     # Selective classification, margin & MSP risk-coverage curves
│   ├── tta/                           # 24-view dihedral test-time augmentation pipelines
│   ├── v2/                            # Post-manuscript campaign: frontier efficiency, escalation
│   │                                  #   heads, rescue conformal, transport decomposition
│   ├── v3/                            # Falsification programme (S40–S47)
│   │   ├── ceiling.py                 #   Representation ceiling instrument (rho_g, Delta_head)
│   │   ├── probes.py                  #   Bottleneck battery: age_band, age_residual, archive
│   │   ├── build_multiarchive.py      #   4 multi-archive training conditions + split generation
│   │   ├── eval_conditions.py         #   Condition evaluation with the pre-registered gate
│   │   ├── external_by_cohort.py      #   Separates in-domain fit from zero-shot transfer
│   │   ├── age_invariant.py           #   Gradient-reversal age-invariance primitives
│   │   ├── train_age_invariant.py     #   Adversarial trainer (alternating k_inner schedule)
│   │   ├── eval_d2.py                 #   Paired mechanism + endpoint evaluation
│   │   ├── safety_refit.py            #   Dirichlet / abstention / conformal refit
│   │   ├── plan.py                    #   Frozen analysis plan + artifact registration
│   │   └── final_verdict.py           #   Per-hypothesis verdicts computed from artifacts
│   └── experiments.csv                # Central append-only ledger of all experimental runs
├── paper/                              # Publication Manuscripts & Camera-Ready Artifacts
│   ├── manuscript.tex                 # Full comprehensive paper draft (~24 pages)
│   ├── manuscript_edited.tex          # Condensed 11-page draft targeting IEEE TMI
│   ├── supplementary.tex              # Comprehensive supplementary material & CLAIM/TRIPOD tables
│   ├── tables/                        # Generated LaTeX tables for full manuscript
│   ├── tables_edited/                 # Formatted LaTeX tables for 11-page edited paper
│   └── figures/                       # Vector & high-res PNG camera-ready figures
├── results/                            # Frozen JSON, CSV, and ledger experimental artifacts
│   ├── frozen_artifacts.json          # SHA256 registry: 34 prediction matrices + analysis plans
│   ├── test_pass_receipt.json         # Append-only test-read receipt (n_executions: 2)
│   ├── v2/                            # V2 campaign outputs, panels & frozen plan
│   └── v3/                            # V3 outputs: conditions, probes, verdicts, frozen plan
├── scripts/                            # Utility Scripts
│   ├── verify_env.py                  # CUDA kernel launch & package environment verifier
│   └── sample_predict.py              # CLI sample inference and triage verification
└── README.md                           # This document
```

---

## How to Reproduce

### 1. Environment Setup

The research suite requires Python 3.12 with PyTorch and CUDA support:

```bash
# Clone the repository
git clone https://github.com/Rajrup910/Backend-S4D-.git
cd Backend-S4D-

# Activate the virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Verify CUDA kernel integrity and dependencies
python scripts/verify_env.py
```

### 2. Verify Manuscript Numerical Integrity & Table Reconstruction

Verify that every number and table reconstructs byte-for-byte from raw experiment artifacts:

```bash
# 1. Audit published manuscript claims (357 checks)
python research/ablation/audit_manuscript.py

# 2. Audit edited 11-page IEEE TMI manuscript (269 checks)
python research/ablation/audit_manuscript.py --target paper/manuscript_edited.tex

# 3. Verify edited LaTeX tables against raw artifacts
python research/ablation/verify_edited_tables.py

# 4. Check structural integrity of LaTeX references and figures
python research/ablation/validate_structure.py --manuscript manuscript_edited.tex

# 5. Measure IEEE TMI two-column page budget
python research/ablation/estimate_pages.py --manuscript paper/manuscript_edited.tex
```

### 3. Run Inference & Clinical Triage Simulation

```bash
# Evaluate baseline model on held-out test split
python -m ml.evaluation.evaluate --checkpoint ml/checkpoints/convnext_tiny_best.HAM-only.pt --split test

# Execute clinical workflow triage simulation with iso-referral baselines
python research/external/simulate_clinical_workflow.py

# Build camera-ready Overleaf bundle for submission
python research/ablation/build_overleaf_bundle.py --manuscript paper/manuscript_edited.tex
```

---

## Publication Details & Manuscripts

This project prepares two companion manuscripts for academic dissemination:

1. **Condensed 11-Page Manuscript** ([`paper/manuscript_edited.tex`](paper/manuscript_edited.tex)):
   - **Target**: *IEEE Transactions on Medical Imaging (TMI)*
   - **Focus**: Age-stratified blind spots, conformal coverage guarantees, and subgroup decision rules.
   - **Budget**: 10.6–11.0 pages, 4 tables, 3 figures, 39 references.
2. **Comprehensive Full Monograph** ([`paper/manuscript.tex`](paper/manuscript.tex)):
   - **Target**: Extended monograph / *Medical Image Analysis (MedIA)*
   - **Focus**: Full 11-rung ablation ladder, cross-domain smartphone shift (PAD-UFES-20), Grad-CAM explainability, and TRIPOD+AI / CLAIM checklists.
   - **Budget**: ~24 pages with full supplementary appendices.

---

## Citation & Authors

If you utilize this codebase, benchmark protocols, or safety audit pipelines in your research, please cite:

```bibtex
@article{chhabra2026safetynets,
  title={When Safety Nets Fail: Subgroup-Conditional Calibration, Conformal Guarantees, and Age-Stratified Blind Spots in Dermoscopy Ensembles},
  author={Chhabra, Prateek and Roy Chowdhury, Rajrup and Srivastava, Aditya and Sonare, Kanak Pravin and Gupta, Manishka and Gupta, Uddhav},
  journal={IEEE Transactions on Medical Imaging (Under Review)},
  year={2026},
  publisher={IEEE}
}
```

**Project Authors (Team 193):**
- **Prateek Chhabra** (23BAI10169) — Ensembling & TTA
- **Rajrup Roy Chowdhury** (23BAI10213, *Corresponding Author*) — Backbones & Calibration
- **Aditya Srivastava** (23BAI10303) — Vision Transformers
- **Kanak Pravin Sonare** (23BAI11369) — Data & Zero-Leakage Shield
- **Manishka Gupta** (23BAI11303) — Multimodal Metadata Fusion
- **Uddhav Gupta** (23BAI10146) — Statistical Audits & Bootstrapping

*Affiliation: School of Computing Science and Engineering, VIT Bhopal University*  
*Repository: [Rajrup910/Backend-S4D-](https://github.com/Rajrup910/Backend-S4D-)*
