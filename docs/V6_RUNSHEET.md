# V6 Runsheet — the single V6 plan and execution document

**Date:** 2026-09-29
**Status:** **DRAFT for Review 2 (9 Oct 2026).** V6 runs **after** Review 2. Before any V6 GPU
run:
- §A5 (the decision table) is filled in with V5's results;
- this file is frozen by hash in its own freeze file, `results/v6/v6_plan_freeze.json`, following
  the V5 procedure.

Nothing here is pre-registered yet.
**Revision 30 Sep (owner request, pre-freeze, before any V6 result):** the six V1 architectures
enter V6 as a portability test of the V5 features and as heterogeneous-ensemble members (§A3b,
§A4 G3, §A5, §A6, phases V6-2b / V6-3b / V6-10, §B3, §B4). This is a revision of an unfrozen draft,
not an amendment; it is hashed with the rest of the plan at V6-0.
**This is the only V6 file.** Part A is the plan (goals, data, models, techniques, decision table,
audit). Part B is the execution (ordered, gated phases, costs, stop rules).
**Inherits from V5:**
- the standing rules: test lock; reserved cohort exhausted; lesion-grouped statistics; `_last`
  primary; 2 workers; ≥ 5 GB free on C:;
- the V5 dermatologist-reasoning heads that survive V5 (`docs/v5_design/V5_DERM_REASONING_ENGINE.md`);
- V5's measured noise floor.

**Contents:**
- **Part A — Plan:** A1 goals · A2 data and allocation · A3 models (A3b V1 architectures) · A4 CV techniques · A5 decision
  table · A6 registry and leakage audit · A7 Review-2 slide · A8 sources.
- **Part B — Execution:** B0 fixed rules · B1 core path · B2 conditional phases · B3 benchmarks ·
  B4 calendar.

---

# Part A — Plan
## A1. What V6 is for

| Goal | Where V5 leaves it | What V6 adds |
|---|---|---|
| **G1 · Under-40 escalation ranking** | V4 proved a representation ranking ceiling. V5 tests biology-derived heads on the same data and ConvNeXt-T trunk | Better **pretraining** (DINOv3, FCMAE), **more young histology-confirmed data**, subtype-aware objectives, a real second view |
| **G2 · Cross-domain Macro-F1** | V5 measures it on LOAO and on MILK10k (S84) | Domain-adaptive pretraining, conditional invariance, SWAD, test-time normalisation, smartphone routing |
| **G3 · One deployable, explainable model** | V5 composite + S56 + S69 | Heterogeneous ensemble → **distilled single model**; `explain` endpoint; updated TRIPOD+AI model card |

---

## A2. Data plan and allocation (declared before any download)

| Source | Size / content | Role in V6 | Checks before use |
|---|---|---|---|
| ISIC-2019 pooled development split (S71) | 15,294 images | Training (5-fold, as V5) | — |
| **MILK10k** | 5,240 lesions; paired clinical + dermoscopic; 95.7% histopathology; age in 5-year bins; MONET concept probabilities in the metadata; CC-BY-NC | **Training** (G1 young data, G2 sites, N12 clinical pairs), **only after V5's S84 read** | Deduplicate against ISIC-2019 (ISIC ID + perceptual hash); lesion-grouped fold assignment |
| **ISIC-2020** | 33,126 images; 584 malignant; **patient IDs**; age; shared institutions | Training: benign diversity, and N13 ugly-duckling context (patient IDs) | Patient-grouped; deduplicate against everything; its patient-held-out subset is reserved as the confirmation fallback. **Any patient whose images entered V5 `youngdata` (`results/v5/young_data/extra_train.csv`, `patient_id`) is excluded from that fallback** |
| **DERM12345** | Multisource dermoscopy with 38 subclasses (Turkey) | Training: subtype supervision for DRE-6 (*licence and subclass list to verify*) | Licence; subclass ↔ class mapping |
| **HIBA** | 1,270 contact-polarised dermoscopy images, 623 patients (Argentina) | **V6 confirmation cohort** (a new population and device): single read | Licence; deduplication; eligible under-40 escalating count and power statement before the lock |
| PAD-UFES-20 | Smartphone clinical images | N12 routing expert (patient-grouped CV; the old test split is already read) | Patient grouping |
| ISIC-2018 Task 2 | 2,594 images, 5 expert attribute masks | Concept supervision and MONET validation | Overlap with train rows only |
| Derm7pt | 7-point-checklist annotated cases | Concept supervision / validation | Definition audit (master §10) |

**Allocation rule (A01 B8, carried forward):**
- A source is used for training **or** confirmation, never both, within a version.
- MILK10k switches from V5 confirmation to V6 training only **after** its single V5 read.
- The V6 confirmation cohort (HIBA, or the pre-declared ISIC-2020 patient-held-out fallback) is
  never trained on, calibrated on or selected on.

---

## A3. Models to add (each carries the surviving V5 heads)

| Priority | Model (timm) | Why it is here | Cost / risk |
|---|---|---|---|
| **1** | **ConvNeXt-T pretraining: IN-22k (`convnext_tiny.fb_in22k_ft_in1k[_384]`, primary) and DINOv3 (`convnext_tiny.dinov3_lvd1689m`, secondary)** (27.8M) — **both screened in V5** (AU17, AU36); V6 uses V5's trunk result. Prior is negative: frozen DINOv2 lost under 40 in S51 (−0.0725 [−0.118, −0.029]) | **Same architecture** as the V4/V5 trunk (stage 3 = 384 channels at stride 16), so every DRE head ports unchanged. Self-supervised on LVD-1689M and distilled from ViT-7B. It isolates *pretraining quality* with fine-tuning; S51 barred only frozen probes | Compute = ConvNeXt-T (measured anchors apply). Pretrain-only weights, so a fixed layer-wise LR decay (0.8) and warm-up are declared in advance. DINOv3 licence |
| 2 | **ConvNeXt-V2-T** `convnextv2_tiny.fcmae_ft_in22k_in1k_384` | Masked-autoencoder pretraining and GRN; drop-in stages; the base for N10 domain-adaptive FCMAE | ≈ ConvNeXt-T; benchmark |
| 3 | **MaxViT-T** `maxvit_tiny_tf_384.in1k` | Hybrid local + global attention gives ensemble diversity. **Not** "the best model in this repo": it led on HAM *test* (0.7525 vs 0.7459) but trailed on *val* (0.7176 vs ConvNeXt-T 0.7482), and Phase 5 selected on val precisely to avoid test-set selection | VRAM at 384 **unmeasured**: benchmark in the unfrozen stage, batch 16 × 2 |
| 4 | **EfficientNetV2-S** `tf_efficientnetv2_s.in21k_ft_in1k` | A different inductive bias; memory-efficient; fits 8.55 GB | Benchmark |
| Conditional | DINOv3 ConvNeXt-S `convnext_small.dinov3_lvd1689m` (≈ 50M) | Only if DINOv3-T beats ImageNet ConvNeXt-T | ≈ 1.7× (extrapolated) |
| Conditional | PanDerm ViT-L, MedSigLIP-448 (LoRA + gradient checkpointing) | Only if V6-2 shows pretraining is the lever | **Leakage audit** of the pretraining corpora against HIBA and MILK10k first. Heavy |
| — | *All five timm tags above were verified to resolve in timm 1.0.28 on 2026-09-29.* | | |
| Not planned | SwinV2-T (repo 0.7273), EfficientNetV2-M (VRAM), Mamba variants (thin evidence), full ViT-L fine-tune | — | — |

## A3b. The six V1 architectures: V5-feature portability and ensemble members (revision 30 Sep)

**Why.** V5 screens and confirms every mechanism on one architecture (ConvNeXt-T). A V5 gain is
therefore shown for ConvNeXt-T only; whether it transfers, and whether it adds on top of an
architecture-diverse ensemble, is untested. The V1 six are the repo's own diverse set, and Phase 1
found ensembling the only intervention that cleared significance (McNemar p = 3.4e-05, Holm-surviving
DeLong on bkl, `results/mcnemar_delong.json`), on HAM. But the V1 checkpoints are **HAM-only** and transfer poorly (V1 stack on
BCN/MSKCC reserved Macro-F1 0.411, `results/v4/s54/s54_marginals.csv`), so they are **retrained in
domain**, not reused.

**"V5 features" carried to every architecture** (fixed at V6-0 from V5's results, nothing re-tuned):
- the V5 recipe: pooled S71 lesion-grouped folds, `_last` primary, 30 epochs, two-stage head /
  fine-tune, class-weighted CE with label smoothing 0.05, `--num-workers 2`;
- the S01 resolution (**224 px**: S01 failed, `results/v5/s01_decision.json`);
- the V5 heads in the **composite lock** (`results/v5/composite_lock.json`), each with its V5
  parameter card, and DRE-7 only if it passed its sign check;
- the V5 system's post-hoc layers: Dirichlet on cross-fitted OOF, S56 refit, per-band calibration.

**Pretrained weights:** the V1 weights family (torchvision ImageNet-1k, `ml.training.common.build_model`)
so the contrast isolates *in-domain training + V5 heads*. The ConvNeXt-T member uses V5's trunk
result (IN-22k, `results/v5/screens/gate_control_in22k_vs_control.json`).

**Porting the F3-reading heads** (clues, zoom, geometry, m5, and the DRE-7 concepts read the stride-16
map). Each architecture's stride-16 stage feeds a declared **adapter: 1×1 conv (C → 384) + LayerNorm2d**,
so every head sees a 384-channel, scale-normalised map. The normalisation is declared here because
V5 measured that a new trunk can shift F3 scale 57× (IN-22k, CHANGELOG 30 Sep). Heads that read the
pooled feature (`twostep`, `m4`, `memory`) take the architecture's own pooled, normalised feature.
A widened stem (`look`) zero-initialises the extra input channels of the first conv, as in V5.

| Architecture (V1) | Stride-16 stage (to verify in V6-0) | C | Adapter | Pooled dim |
|---|---|---:|---|---:|
| ConvNeXt-T | `features[:6]` | 384 | identity (native) | 768 |
| ConvNeXt-S | `features[:6]` | 384 | identity (native) | 768 |
| EfficientNet-B0 | `features[:6]` | 112 | 1×1 → 384 + LN | 1280 |
| EfficientNet-B3 | `features[:6]` | 136 | 1×1 → 384 + LN | 1536 |
| ResNet-50 | through `layer3` | 1024 | 1×1 → 384 + LN | 2048 |
| DenseNet-121 | through `denseblock3` (+`transition3` excluded) | 1024 | 1×1 → 384 + LN | 1024 |

A unit test in V6-0 must assert each stage's stride (16) and channel count on a 224 px input
before any run; the table is the plan, the test is the truth.

---

## A4. CV techniques by goal

| Goal | Technique | Biological / mechanistic reason | Gate (V5 result) | Falsifier |
|---|---|---|---|---|
| G1 | **Learned segmenter** (A11a) | Gives exact lesion geometry for the periphery, asymmetry and context statistics on every archive | Needed unless DRE-2 passed Q1 with median Dice ≥ 0.85 | Dice ≥ 0.85 on held-out HAM, ≤ 5% failures on the 50 + 50 BCN/MSKCC human QC |
| G1 | **Separate-trunk dual-stream** (global + lesion crop; late fusion or cross-attention) vs a **capacity-matched ConvNeXt-S** | Optical zoom on small lesions; context kept separately | DRE-8 / 8b showed "where to look" matters | Gain must beat the capacity-matched single stream and concentrate in the smallest lesion-size tercile (A01 B10) |
| G1 | **Wavelet space-to-depth** (768 px → Haar LL/LH/HL/HH × RGB = 12 channels at 384²) | Recovers fine structure lost when downsampling (BCN 2.67×, HAM 1.17×) at 384 px compute | DRE-8 gain larger on BCN/MSKCC | Gain on BCN/MSKCC, not on HAM |
| G1 | **Multi-proxy contrastive** loss (ECL-style, on DRE-6 prototypes) | Keeps subtypes separate while pulling rare classes together; avoids SupCon's class collapse | DRE-6 passed | Young mel prototype separation (χ²) improves |
| G1 | **Subclass Group-DRO** (N6) | Worst-subtype risk instead of tiny age groups (V5 audit AU6) | D4 found a low-pAUC, <40-enriched subclass | Worst-subclass pAUC rises; others retained |
| G1 | **Subtype supervision** (DERM12345 subclasses) | Explicit melanoma/nevus subtypes (e.g. superficial spreading, Spitz/Reed, congenital) | Licence OK; DRE-6 passed | Subtype-labelled young mel ranked better |
| G1 | **pAUC-DRO** loss (LibAUC) | Optimises low-FPR partial AUC directly, focusing on the hardest benign mimics | M4 passed | pAUC_histo improves over M4 |
| G1 | **N5 young-data expansion** with a learning curve | Only 81 under-40 escalating lesions in V5 | Always (a core lever) | OOF under-40 pAUC slope over 25/50/100% of added young data > 0 |
| G2 | **N10 FCMAE domain-adaptive pretraining** (ConvNeXt-V2 on unlabelled multi-device dermoscopy; no confirmation images) | The trunk learns every device's image statistics without labels | M7/LOAO showed an acquisition gap | LOAO Macro-F1 over the same fine-tune from ImageNet/IN-22k |
| G2 | **N9 class-conditional acquisition adversary** | Archive is not diagnostic (unlike age), so conditional invariance is safe | M7 lowered decodability but did not close the gap | Within-class decodability falls **and** LOAO rises; in-distribution retention ≥ −0.01 |
| G2 | **SWAD** (dense weight averaging) | Flat minima reduce the domain gap (+1.6% OOD on DG benchmarks) | V5 SWAD secondary positive on LOAO | LOAO Macro-F1 |
| G2 | **Test-time normalisation adaptation** | Adapts normalisation statistics to a new site from unlabelled images | — | LOAO Macro-F1; no label use |
| G2 | **N11 Saerens EM prior** (CPU) | New sites have different class priors; changes the argmax only (Macro-F1), never claimed as an under-40 fix. **Near-repeat warning:** registry item "Global prevalence prior correction (S58 null)" — run only as a LOAO Macro-F1 analysis, lowest priority | — | LOAO Macro-F1 with vs without |
| G2 | **N12 smartphone routing** | S69 gate AUROC 0.9994 → route to a clinical-photo expert instead of refusing (HAM→PAD warm start reached 0.760) | — | Patient-grouped PAD CV and MILK10k clinical pairs |
| G1 | **N13 ugly duckling** (set encoder over one patient's lesions) | In young patients with many nevi, the outlier lesion is the strongest clinical sign | ISIC-2020 patient IDs available | Young-mel rank among a patient's own lesions improves |
| G1/G3 | **MONET concept labels**, validated against Task-2 expert masks before use | Scales concept supervision (DRE-7) to every image; the policy forbids unvalidated pseudo-labels | DRE-7 sign check passed | Concept AUROC vs Task-2 masks ≥ a pre-declared bar; then concept-supervised DRE-7 |
| G3 | **Heterogeneous ensemble → morphology-gated ensemble (B10) → distillation** into one model | Deployable single model with ensemble accuracy | ≥ 2 finalists | Distilled model within 0.01 Macro-F1 / pAUC of the ensemble |
| G1/G3 | **V5-feature portability** across the six V1 architectures (§A3b, V6-2b) | A mechanism that only works on one trunk is a trunk artefact, not biology | V5 composite lock exists | Per architecture: (arch + V5 heads) − (arch control) on all-age pAUC or pAUC_histo above the V5 noise floor, Holm across the 5 non-ConvNeXt-T architectures |
| G3 | **Six-architecture in-domain ensemble with V5 features** (§A3b, V6-3b, V6-10) | Error diversity across inductive biases; the only lever that cleared significance in Phase 1 | V6-3b OOF complete | Primary: uniform soft vote beats the V5 15-model system on folds 1–4 OOF (paired all-age pAUC above the noise floor, Macro-F1 retention ≥ −0.010); <40 pAUC descriptive |
| G3 | **Full-frame eval** (no CenterCrop) sensitivity analysis | V5 audit AU8: periphery clipped at eval | — | Report only |

---

## A5. Decision table: how V5's results reshape V6 (filled in before the V6 freeze)

| V5 result | V6 consequence |
|---|---|
| S01: 384 px beats 224 px on pooled data | V6 finalists at 384 px; else 224 px, and wavelet space-to-depth becomes more important |
| V5 trunk screen: **IN-22k** won | V6-2 starts from IN-22k; DINOv3 is not re-screened; ConvNeXt-V2 (`fcmae_ft_in22k_in1k_384`) becomes the next same-family test |
| V5 trunk screen: **DINOv3** won (IN-22k did not) | V6-2 starts from DINOv3 and N10 (domain-adaptive FCMAE) gets priority |
| V5 trunk screen: neither won | Pretraining is not the lever at ConvNeXt-T scale; V6-12 (foundation LoRA) drops in priority |
| V5 `youngdata` passed (within-band) | N5 is confirmed as a lever; V6-4 extends it (MILK10k after S84, ISIC-2020) |
| V5 `youngdata` failed or only moved all-age pAUC | Young-data expansion is an age prior, not ranking; V6-4 is dropped |
| `look` / `structure` pass | Chromophore stem and DSP are part of every V6 backbone |
| `memory` passes | Multi-proxy contrastive + subtype supervision go into the V6 core |
| `clues` beats `gem` | Any-region escalation logic kept; pAUC-DRO built on it |
| `zoom` > `zoom_random` | Separate-trunk dual-stream and wavelet space-to-depth are promoted |
| M7 / SWAD improve LOAO | N10 and N9 go into the V6 core for G2 |
| DRE-7 sign check passes | MONET-validated concept supervision is promoted |
| V5 composite null | V6 core = pretraining (DINOv3/FCMAE) + data (N5) — the two levers V5 did not test |
| S84 (MILK10k) under-40 CI crosses 0 | The data lever (N5) is prioritised over architecture |
| V6-2b: V5 heads help on ≥ 3 of the 5 other architectures (Holm) | The V5 mechanism is reported as architecture-general; every V6-3b member carries the heads |
| V6-2b: V5 heads help on ConvNeXt only | Reported as trunk-specific; V6-3b members are plain in-domain controls, and the ensemble is judged without heads |
| V6-3b: ensemble beats the V5 system | The distilled V6-10 student is trained from the six-architecture ensemble |
| V6-3b: ensemble ties the V5 system | Report "diversity does not add beyond seeds × folds"; V6-10 distils the smaller system |
| V5 composite null (for the ensemble question) | V6-2b is skipped (no heads to port); V6-3b still runs as plain in-domain controls — the ensemble question stands on its own |

---

## A6. Registry and leakage audit

**Registry (master §21):**
- No V6 item repeats a barred mechanism.
- Foundation models are **fine-tuned**, never frozen-probed (items 4–5).
- No age or metadata input (item 2); N13 uses *other lesions' images*, not tabular data.
- Adversarial invariance is on **archive given class**, never age (item 3).
- Prior correction (N11) is scoped to Macro-F1 only (item 6).
- The decision layer stays S56 (items 7–8).

**Leakage audit (before any V6 read):**
- **DINOv3 LVD-1689M:** web-scale public images. The chance of containing HIBA images is low but
  not zero. Recorded as a limitation.
- **PanDerm / MedSigLIP pretraining corpora** may include public dermoscopy archives. They are
  **not used** unless the corpus list excludes HIBA and the V6 confirmation fallback.
- **MILK10k / ISIC-2020 / DERM12345** are deduplicated against each other and against ISIC-2019
  by ISIC ID and perceptual hash.
- **ISIC-2020** is patient-grouped, and the confirmation fallback subset is patient-disjoint from
  training.
- **HIBA** is deduplicated against every training source and read once (receipt enforced).
- **Frozen V1 six (HAM-only checkpoints):** trained on HAM, whose images sit inside every pooled
  development fold, so they **cannot be scored out of fold** on pooled rows. They appear only as
  the V1 reference system on the confirmation cohort (V6-11), never in an OOF comparison, weight
  fit or calibration.
- **Ensemble fitting:** member weights (Caruana, Nelder–Mead), stacking and the Dirichlet map
  are fitted on cross-fitted OOF only; **uniform soft vote is the pre-declared primary** (the
  Phase-5 rung-6 lesson: ridge stacking had the best test and the worst val_oof). Fold 0 was the
  V5 selection fold and is flagged in every OOF matrix that includes it.
- **Multiplicity:** the portability family (5 non-ConvNeXt-T architectures) and the ensemble
  family (primary + secondaries) are declared in `research/stats/families.py` at V6-0, Holm within
  each family.

---

## A7. What Review 2 shows about V6

- **One slide:** goals G1–G3; the data allocation (MILK10k → V6 training after S84; HIBA → V6
  confirmation); the model priority list (DINOv3 ConvNeXt-T first); the gated phase chart from
  `V6_RUNSHEET.md`; and the statement that V6 will be **frozen with V5's results** before it runs.

---

## A8. Sources

- DINOv3 ConvNeXt-T (timm): https://huggingface.co/timm/convnext_tiny.dinov3_lvd1689m
- MILK10k: https://api.isic-archive.com/doi/milk10k/ ; J Invest Dermatol
  https://www.sciencedirect.com/science/article/pii/S0022202X25022705
- HIBA (Sci Data 2023): https://www.nature.com/articles/s41597-023-02630-0
- ISIC-2020 (Sci Data 2021): https://www.nature.com/articles/s41597-021-00815-z
- DERM12345: https://arxiv.org/pdf/2406.07426
- ISIC-2018 Task 2: https://challenge2018.isic-archive.com/task2/training/ ; Codella et al.
  https://arxiv.org/pdf/1902.03368
- Robust OOD augmentation (ConvNeXt, dermoscopy): https://arxiv.org/abs/2607.26765
- SWAD (NeurIPS 2021): https://arxiv.org/abs/2102.08604
- ECL (MICCAI 2023): https://arxiv.org/abs/2307.04136
- LibAUC / pAUC-DRO: https://docs.libauc.org/api/libauc.losses.html ;
  https://arxiv.org/pdf/2203.00176
- PanDerm (Nat Med 2025): https://www.nature.com/articles/s41591-025-03747-y
- MedSigLIP: https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card
- Dermatology foundation-model benchmark: https://arxiv.org/abs/2601.12382
- MONET / Derm1M context: https://arxiv.org/pdf/2503.14911
- ConvNeXt-V2 + MaxViT ensemble (CXR-LT): https://arxiv.org/abs/2410.10710
- Vessel segmentation in dermoscopy: https://pmc.ncbi.nlm.nih.gov/articles/PMC6236870/
- Pigment network directional filters: https://pubmed.ncbi.nlm.nih.gov/22829364/

---

# Part B — Execution

**Design principle:** a **core path** delivers a complete, reportable V6 on its own. Conditional
phases run only if their V5 or V6 gate fires. V6 can stop at any gate with a result.

## B0. Fixed rules and cost anchors

**Cost anchors (ConvNeXt-T, measured):**
- 224 px fold-0 run: **41.0 min**;
- 384 px fold run: **64–75 min**;
- one 5-fold × 1-seed set: **5.3–6.3 h** at 384 px, or **≈ 3.4 h** at 224 px.

Anything else is **unmeasured** until `scripts/gpu_benchmark.py` has run in the **unfrozen
fine-tune stage** (hardware notes: stage-1 memory understates the peak by up to 6×).

**Fixed for every run:**
- `--num-workers 2`, `--patience 0`, 30 epochs;
- ≥ 5 GB free on C:;
- screens save `_last` only, confirmation saves both;
- smoke first;
- the test lock stays armed.

---

## B1. Core path

| Phase | Work | Inputs | Outputs | GPU (est.) | Stop / continue rule |
|---|---|---|---|---|---|
| **V6-0** Freeze and data | Fill the §A5 decision table from V5. Download and deduplicate MILK10k (training, after S84), ISIC-2020, DERM12345 (if the licence is OK). HIBA eligibility, count and power (metadata only). Declare the allocation. Extend the S71-style lesion-grouped folds to the new rows. **Hash the V6 plan** | V5 results; `perceptual_hashes.npz` | `results/v6/v6_plan_freeze.json`; `data_allocation.json`; `fold_assignments_v6.csv` | CPU | HIBA underpowered → switch to the pre-declared ISIC-2020 patient-held-out fallback |
| **V6-1** Segmenter (if needed) | U-Net-style segmenter on HAM masks (fold-respecting) + Task-2 overlap; human QC on 50 BCN + 50 MSKCC | HAM masks | `ml/checkpoints/seg_v6_f*.pt`; `seg_qc.json` | ≈ 1–2 h (benchmark) | Median Dice ≥ 0.85 and ≤ 5% failures, else the V6-5 dual-stream uses DRE-2 geometry |
| **V6-2** Backbone screen | DINOv3-ConvNeXt-T, ConvNeXt-V2-T, MaxViT-T, EfficientNetV2-S, each with the surviving V5 heads. Fold 0, 224 px, seeds 42/43. Paired against the V5 composite on the V5 trunk (IN-22k ConvNeXt-T, `results/v5/screens/gate_control_in22k_vs_control.json`) | V5 composite spec | `results/v6/screen/*` | ≈ 6–9 h (ConvNeXt pair measured; MaxViT/EffNetV2 benchmark first) | Promote ≤ 2 whose all-age pAUC Δ exceeds the V5 noise floor with Macro-F1 retention ≥ −0.010 |
| **V6-2b** V5-feature portability (§A3b) | Each of the 5 other V1 architectures (ConvNeXt-S, EfficientNet-B0/B3, ResNet-50, DenseNet-121): **arch control** vs **arch + V5 locked heads** (adapter per §A3b). Fold 0, 224 px, seeds 42/43, `_last`. Gate as V5's screen gate against the arch's own control | V5 composite lock; §A3b stage test passes | `results/v6/portability/*`; `portability_holm.json` | 20 runs: ≈ 14 h at the ConvNeXt-T anchor; **unmeasured** for the others (ConvNeXt-S ≈ 1.7× extrapolated); benchmark first (§B3) | Holm across the 5; result sets the §A5 row. No run is repeated if a gate fails |
| **V6-3** Finalists | ≤ 2 backbones, folds 1–4 (+ fold 0 for OOF) × seeds 42/43 at the V5-chosen resolution | V6-2 winners | `results/v6/confirm/*`; OOF matrix | ≈ 21–25 h (384 px, ConvNeXt anchors) | Gates A–D on folds 1–4 vs the V5 composite |
| **V6-3b** Six-architecture OOF (§A3b) | The 5 other V1 architectures × folds 0–4 × seed 42, in domain, with V5 heads if V6-2b made them general (else plain controls). The ConvNeXt-T member **reuses** V5 runs (no retraining): the composite's seed-42 fold runs if the heads are carried, else V5's seed-42 control fold runs, so every member is built the same way | V6-2b outcome | `results/v6/oof_six/*`; common OOF matrix (15,294 rows × members) | 25 runs: ≈ 17 h at the ConvNeXt-T anchor; **unmeasured** for the others | OOF complete for every member before any ensemble is formed |
| **V6-4** Young-data expansion (N5) | Learning curve: add 25/50/100% of the new under-40 histology-confirmed lesions (fold 0, 224 px, 2 seeds), then retrain the best trunk on the full expanded set (5-fold, 1 seed) | V6-0 data | `learning_curve.json` | ≈ 10–12 h | Slope > 0 → keep the data; flat → report "data-saturated at N" |
| **V6-10** Ensemble and deployment | OOF common matrix → pre-declared ensembles, all on folds 1–4 OOF (fold 0 flagged): **E-primary** uniform soft vote of the six in-domain architectures (V6-3b) vs the V5 15-model system; secondaries: Caruana / Nelder–Mead weights (OOF-fitted), morphology-gated (B10, shallow gate), V6-3 finalists added. Diversity report (Yule's Q, disagreement, double-fault, as Phase 1). Each ensemble gets Dirichlet on cross-fitted OOF, per-band ECE / signed gap and an S56 refit. Then **distil the winner into one ConvNeXt-T**; S69 gate; `explain` endpoint in the S60 FastAPI; model card update | V6-3 / V6-3b / V6-4 models | `v6_ensemble.json`; `v6_diversity.json`; distilled checkpoint; `paper/v6/model_card.md` | ≈ 12–20 h | E-primary: paired all-age pAUC above the V5 noise floor **and** Macro-F1 retention ≥ −0.010 (<40 pAUC descriptive). Distilled model within 0.01 (Macro-F1, pAUC) of the chosen ensemble |
| **V6-11** Confirmation | **Single locked read** on HIBA (or the fallback): V6 system vs V5 system vs V1 system, all with S56, at matched referral. Paired under-40 sensitivity (exact McNemar); S70 contract; Macro-F1 non-inferiority | Locked V6 system | `results/v6/v6_confirm_report.json`; receipt | ≤ 1 h | Single read, enforced by a receipt |

**Core-path GPU total ≈ 50–70 h** before the 30 Sep revision; **≈ 81–101 h with V6-2b + V6-3b at the ConvNeXt-T anchor**,
and more in practice, because ConvNeXt-S (≈ 1.7×, extrapolated), EfficientNet-B3 and
DenseNet-121 are slower and all five are unmeasured until §B3. Re-quote after the benchmarks. If the budget binds, V6-3b drops to 3 folds per member and the
ensemble is read on those folds only (declared now, not chosen after results).

---

## B2. Conditional phases (run only if the gate fires; order by expected value)

| Phase | Work | Gate (from V5 / V6) | GPU (est.) | Falsifier / stop |
|---|---|---|---|---|
| **V6-5** Resolution and second view | (a) **Wavelet space-to-depth** (768 px → 12 channels at 384²), BCN/MSKCC-stratified. (b) **Separate-trunk dual-stream** (global + lesion crop from V6-1, late fusion or cross-attention) vs a **capacity-matched ConvNeXt-S** single stream | V5 `zoom` > `zoom_random`; the DRE-8 gain is concentrated on BCN/MSKCC | ≈ 8–12 h (unmeasured; benchmark 768 px decoding) | (a) Gain on BCN/MSKCC, not HAM. (b) Must beat ConvNeXt-S and concentrate in the smallest lesion-size tercile |
| **V6-6** Representation and losses | Multi-proxy contrastive (ECL-style on DRE-6); **subclass Group-DRO** (N6); **subtype supervision** (DERM12345); **pAUC-DRO** | V5 `memory` passed / D4 subclass found / licence OK / V5 `m4` passed | ≈ 6–10 h | Each against its V5 parent: young-mel prototype separation; worst-subclass pAUC; pAUC_histo |
| **V6-7** Domain generalisation | **N10 FCMAE domain pretraining** of ConvNeXt-V2 on unlabelled multi-device dermoscopy (no confirmation images) → fine-tune; **N9** class-conditional archive adversary; **SWAD**; test-time normalisation adaptation; **N11** EM prior (CPU) | V5 M7 / SWAD LOAO results; an acquisition gap remains | N10: **benchmark first, possibly 20–40 h**; others ≈ 6–8 h | LOAO Macro-F1. N9 needs in-distribution retention ≥ −0.01 |
| **V6-8** Concepts | MONET concept probabilities (MILK10k ships them; also run MONET on the rest) **validated against Task-2 expert masks** → concept-supervised DRE-7; Derm7pt 7-point | V5 DRE-7 sign check passed | ≈ 4 h (+ CPU) | Concept AUROC vs expert masks ≥ 0.75 before any use |
| **V6-9** Context and modality | **N13 ugly duckling:** a set encoder over one patient's lesions (ISIC-2020 patient IDs). **N12 smartphone routing:** S69 gate → a clinical expert warm-started from the V6 trunk (PAD patient-grouped + MILK10k clinical pairs) | Patient IDs available / S69 kept | ≈ 8–12 h | N13: young-mel rank among the patient's own lesions. N12: patient-grouped Macro-F1 vs the 0.760 PAD reference |
| **V6-12** Foundation LoRA | PanDerm ViT-L or MedSigLIP-448 with LoRA + gradient checkpointing | V6-2 shows pretraining is the lever **and** the leakage audit clears | Unmeasured (heavy) | Must beat DINOv3-ConvNeXt-T at equal data |
| **V6-13** Full-frame eval | Inference without CenterCrop on OOF (V5 audit AU8) | Always (cheap) | ≈ 0.5 h | Report only |

---

## B3. Benchmarks to run before their phase (unfrozen stage, 2 workers)

```bash
python scripts/gpu_benchmark.py --arch maxvit_tiny_tf_384.in1k --sizes 224 384 --batch 16 --workers 2 --images 12235 --epochs 30 --json-out results/v6/bench_maxvit.json
```

- Run the same benchmark for `tf_efficientnetv2_s.in21k_ft_in1k`,
  `convnext_tiny.dinov3_lvd1689m` and `convnextv2_tiny.fcmae_ft_in22k_in1k_384`.
- `--images 12235` = the fold training size.
- The flags are the script's real CLI (`--arch`, `--sizes`, `--batch`, `--workers`,
  `--cpu-ms-per-image`, `--images`, `--epochs`, `--json-out`). Confirm in V6-0 that `--arch`
  passes a full timm tag with its pretrained suffix. If it strips the suffix, extend the script
  there.
- Also benchmark the V6-5 768 px decode path and V6-7 N10 FCMAE steps before quoting hours.
- **V6-2b / V6-3b (revision 30 Sep):** benchmark the five other V1 architectures at 224 px,
  batch 32, unfrozen stage, with the V5 heads and adapter attached (`convnext_small`,
  `efficientnet_b0`, `efficientnet_b3`, `resnet50`, `densenet121`; the torchvision builders in
  `ml.training.common.build_model`). Confirm in V6-0 that `gpu_benchmark.py` can build them, or
  extend it there. Record the VRAM peak of the fine-tune stage, not the frozen stage.

---

## B4. Suggested order and calendar (no fixed deadline)

1. **Week 1 after Review 2:** V6-0 (CPU) → V6-1 → V6-2 → V6-2b portability screen.
2. **Week 2:** V6-3 finalists (nights + daytime) → V6-3b six-architecture OOF → V6-4 learning curve.
3. **Week 3:** the conditional phases that fired, in the table's order.
4. **Week 4:** V6-10 ensemble and distillation → V6-11 single confirmatory read → write-up.

Every phase ends with a CHANGELOG entry and ledger rows (`research/experiments.csv`, session
`v6_*`). Any change after the V6 freeze goes into a V6 amendment, labelled post-hoc if it follows
any V6 result.
