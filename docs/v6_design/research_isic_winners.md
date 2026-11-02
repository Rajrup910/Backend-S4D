# ISIC challenge winners 2016–2025: what they did, what their own ablations show, and what V6 should take

**Tags.** **[V]** means I read the claim in the primary source: the paper, the official leaderboard, the team's own write-up or repo. **[S]** means the claim comes from a search snippet or a secondary summary. **[M]** means I read the write-up in a third-party mirror because the original Kaggle post would not load.

**Access limits.** These sources would not load: the Kaggle discussion pages (JavaScript only), the Lancet and Wiley full text (HTTP 403), Nature (sign-in redirect), and the Yu et al. TMI 2017 paper (anti-bot wall). Two later searches still did not find MetaOptima's 2018 method write-up, so their architecture and external data are not reported here. Nothing below is filled in from memory.

---

## 1. Per-challenge findings

### 2016 (ISBI; binary melanoma classification, ranked by average precision)
- Training set 900 images, test set 379. [V] [arXiv 1605.01397](https://arxiv.org/abs/1605.01397)
- Top 2016 test score: AP 0.637, AUC 0.804, as quoted in Table 2 of Casio's 2017 paper. [V] [arXiv 1703.03108](https://arxiv.org/abs/1703.03108)
- The winner is usually named as CUHK "CUMED" (Yu et al., TMI 2017, FCRN segmentation then a deep ResNet). This is **[S]**: the leaderboard tab did not render and the paper was blocked. Of little relevance to V6.

### 2017 (Part 3: MEL-vs-rest and SK-vs-rest, ranked by mean AUC)

| Rank | Team | Mean AUC | MEL AUC | SK AUC |
|---|---|---|---|---|
| 1 | Casio / Shinshu (Matsunaga) | 0.911 | 0.868 | 0.953 |
| 2 | Díaz (DermaKNet) | 0.910 | 0.856 | 0.965 |
| 3 | RECOD Titans (Menegola) | 0.908 | 0.874 | 0.943 |
| — | Fusion of all submissions, linear SVM | 0.926 | 0.892 | 0.960 |
| — | Fusion, non-linear SVM | 0.904 | 0.853 | 0.955 |

All rows [V] from [arXiv 1710.05006](https://arxiv.org/abs/1710.05006). All top three teams used extra data.

**1st place, Casio** [V] ([arXiv 1703.03108](https://arxiv.org/abs/1703.03108)):
- Model: an ensemble of ResNet-50s (Keras) fed geometric multi-view inputs. This is both a train-time and a test-time ensemble.
- Colour constancy: the Barata method.
- Extra data: 409 MEL, 66 SK and 969 NV images from the ISIC Archive.

Casio's own ablations, in CV:
- Extra data: SK AUC 0.981→0.992; MEL AUC 0.896→0.899.
- Age/sex thresholding: SK 0.957→0.960. The authors found "no significant increase" for MEL.
- Feeding the SK classifier's output into the MEL score: MEL 0.917→0.924, on the 150-image validation set only.

**3rd place, RECOD** [V], abstract only ([arXiv 1703.04819](https://arxiv.org/abs/1703.04819)): seven models (Inception-v4 ×3, Inception ×3, ResNet-101) stacked by an SVM meta-learner fitted on validation data.

**Lesson.** The simple linear fusion gained +0.015 mean AUC over the best single team. The non-linear SVM fusion lost ground.

**Related factorial study.** RECOD's later experiment ([V], Valle et al., [arXiv 1711.00441](https://arxiv.org/abs/1711.00441)) crossed 9 factors over 5 test sets (2,560 treatments) and ran a multi-way ANOVA. Shares of explained variance, relative:

| Factor | Relative variance | Effect on mean AUC |
|---|---|---|
| Training-data amount | 46% | 85 vs 81 |
| Test-time augmentation | 12% | 84 vs 82 |
| Resolution | 5% | 598 px 84 vs ~300 px 82 |
| SVM decision layer | 4% | Harmful |
| Segmentation | 2% | Harmful |
| Train augmentation | — | Not significant (p=0.17) |

Transfer learning explained 62.8% of variance in a separate experiment. This is the only full factorial ablation in the ISIC literature, but it uses 2017-era backbones and AUC.

### 2018 Task 3 (7-class HAM10000, balanced accuracy; test set 1,512 images)
- Top five entries by BACC: 0.885, 0.882, 0.871, 0.856, 0.845. [V] ([arXiv 1902.03368](https://arxiv.org/abs/1902.03368))
  - The test set had an "internal" part (1,196 images from training sources) and an "external" part (316 images from Turkey, New Zealand, Sweden and Argentina).
  - Most entries did worse on the external part. Some top entries did not.
  - Ranks 4 and 5 had higher plain accuracy (0.896 / 0.884) but lower BACC. The organisers cite this as the reason for using BACC.
- **MetaOptima took places 1–3.** Their entries were "Top 10 Models Averaged" 0.885, "Meta Ensemble" 0.882 and "Best Single Model" 0.871. [V], the team's own announcement ([DermEngine](https://dermengine.com/blog/metaoptima-awarded-isic-2018-challenge)).
  - So on the hidden test, a 10-model average beat their best single model by +0.014 BACC.
  - The architecture and external data are not described on that page. I could not find the method write-up.
- **DAISYLab** (Gessert et al.) [V] ([arXiv 1808.01694](https://arxiv.org/abs/1808.01694)) said it placed second, as the best team using only public data. The 0.856 entry is consistent with this, but that link is inferred, not confirmed.
  - Ensemble: 54 models (SENet154, ResNeXt-101, DenseNet-121/161/169/201, SE-ResNet-101, PolyNet, and others).
  - Input: images kept at native 600×450; training on **unscaled random 224 crops**; 36-crop TTA.
  - Loss: inverse-frequency weighted CE.
  - Extra data: ~13.5k ISIC Archive images added.
  - CV: 5-fold, **grouped by lesion**.
  - Hardware: GTX 1080 Ti.
  - Their single-model ablations (DenseNet-121, CV BACC; the paper calls it "WACC"):

| Change | BACC |
|---|---|
| Baseline | 0.795 |
| No loss weighting | 0.757 |
| Batch oversampling instead | 0.789 |
| No ImageNet pre-training | 0.694 |
| 16 crops / 4 crops at test | 0.785 / 0.662 |
| + ISIC Archive data | 0.804 |
| + SVM over the 36 crop outputs | 0.822 |
| Ensemble | 0.851 (best single SENet154 0.817) |

  - Rotation, scaling and hue augmentation gave no gain.

### 2019 (8 known classes plus an unseen UNK class, BACC; test 8,238 images with hidden labels)

| Rank | Team | BACC (images only) | Method label |
|---|---|---|---|
| 1 | DAISYLab | 0.636 | Ensemble of multi-resolution EfficientNets + SENet154 |
| 2 | DysionAI | 0.607 | EfficientNet-B3/B4 + SE-ResNeXt-101 ensemble |
| 3 | AImageLab & PRHLT | 0.593 | Ensemble with an OOD threshold |
| 4 | DermaCode | 0.578 | 13 models + hierarchical outlier selection |
| 5 | Nurithm Labs | 0.569 | DenseNet-161 with heavy random crops |

All [V], [official leaderboard](https://challenge.isic-archive.com/leaderboards/2019/).

**1st place, DAISYLab** [V] ([arXiv 1910.03910](https://arxiv.org/abs/1910.03910)):
- Backbones and native resolutions: EfficientNet B0 224, B1 240, B2 260, B3 300, B4 380, B5 456, B6 528, plus SENet154 and ResNeXt-WSL at 224.
- Two crop regimes: same-size random crops with 36-crop TTA, and random-resize crops with 16-view TTA.
- Preprocessing:
  - Shades-of-Gray colour constancy (p=6).
  - Long side resized to 600 px.
  - **A heuristic crop of the dark dermoscope field of view**, applied to 6,226 training and 3,864 test images. These are mostly the BCN-style circular images.
- Loss: weighted CE with weights (N/Nᵢ)^k, k=1. Focal loss gave no gain.
- Other training details:
  - AutoAugment gave no gain.
  - CutOut was used.
  - Adam for 100 epochs.
- Extra data: 995 images from the 7-point dataset, plus 1,339 in-house images (353 of them UNK).
- CV: 5-fold, **lesion-grouped**. Patient grouping is not stated.
- Ensemble: an exhaustive subset search on CV picked 9 of 16 configurations, each with 5 fold models (best and last checkpoints).
- Hardware: GTX 1080 Ti for B0–B4; Titan RTX for B5–B6.

DAISYLab's ablations (CV mean sensitivity, Task 1):

| Comparison | Result |
|---|---|
| B0 SS 224 → B6 SS 528 | 65.8 → 68.8, but capacity and resolution change together |
| SS vs RR crops | "Similar"; differences were ±1 in both directions |
| Plain ensemble average | 71.7 |
| Selected ensemble subset | 72.5 (best single 68.8) |
| Metadata (Task 2, per configuration) | +0.2 to +2.8 |
| Official test | **63.6 image-only, 63.4 with metadata**; UNK sensitivity 0.3% |

- The subset was chosen on the same folds it is reported on, so 72.5 is optimistic.
- Even for the winner, metadata did not help on the test set.

**2nd place, DysionAI** [V] (manuscript linked from the leaderboard):
- Backbones: DenseNet-121, SE-ResNeXt-50/101, EfficientNet-B2/B3/B4.
- Resize: short side 600; random crops **without aspect-ratio warping**.
- Shades-of-Gray on the extra images.
- Inverse-frequency weighted loss.
- 16–25 ordered crops at test time.
- Validation: one GTX 1080 Ti and an **80/20 split with no lesion grouping stated**.
- Results:
  - Best 3-model ensemble 0.753 vs best single model 0.739, on validation.
  - DenseNet-121 crop settings "224/224 vs 224/256": 0.626 vs 0.726. The columns are ambiguous, but this suggests the crop-from-larger-image regime matters.
- UNK was assigned whenever the top-1 probability was below 0.35.

**3rd place, AImageLab-PRHLT** [V] (manuscript linked from the leaderboard):
- Inputs: reflect-padded and resized to 512×512.
- Backbones: DenseNet-201, ResNet-152, SE-ResNeXt-101.
- Augmentation variants differ by member: blur, Poisson noise, gamma, hue and cutout.
- Each member saves its top-3 BACC snapshots, giving 21 models.
- Calibration and TTA:
  - Each model is temperature-scaled; they report that the calibrated ensemble was better, with no number given.
  - TTA uses 30 passes.
- A second group adds SE-ResNeXt-50/101 and SENet154, trained as snapshot ensembles with CE + focal loss.
- OOD: an ensemble of 8 leave-one-class-out DenseNets.
- Validation: 2,000 random images, with no lesion grouping stated.

**Post-challenge analysis** [V], Combalia et al., *Lancet Digit Health* 2022 ([PMC9295694](https://pmc.ncbi.nlm.nih.gov/articles/PMC9295694)):
- The top two teams used EfficientNet ensembles; the third used a ResNet ensemble.
- **The top algorithm scored 82.0% BACC on HAM10000 images but 58.8% on BCN20000 images** (AUROC 0.981 vs 0.907). The mean drop across all algorithms was 22.3%.
- Melanoma recall (Table 2): readers 0.62, all algorithms 0.58, top 3 algorithms 0.70.
- Pigmentation:
  - Pigmented vs non-pigmented nevi: 83% vs 35% correct.
  - Pigmented vs non-pigmented melanoma: 71% vs 46% correct.
- Hair reduced accuracy only for AK (36% with hair vs 56% without).
- No melanoma from the s_HAM_molemax source was identified by any algorithm.
- Unseen categories:
  - Only 1.6% were correctly called UNK by the top algorithm.
  - 47.1% were called malignant by the top-25 algorithms.
- Metadata across submissions: mean BACC 50% (Task 1) vs 56% (Task 2). These are unmatched submissions.
- **No age-stratified results are reported.**

### SIIM-ISIC 2020 (Kaggle; binary melanoma AUC; 33,126 training images from 2,056 patients)
Winners: 1st "All Data Are Ext", 2nd "aloe" (Ian Pan), 3rd Deloitte Analytics Spain. 2nd place also won the "without patient context" prize. [V] ([SIIM](https://siim.org/page/melanoma_classification_challenge))

**1st place** [V] ([arXiv 2010.05351](https://arxiv.org/abs/2010.05351), [repo](https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution)):
- Ensemble: 18 models — EfficientNet B3–B7 (mostly noisy-student), SE-ResNeXt-101 and ResNeSt-101.
- Resized inputs from 384 to **896 px**.
- Target: 9 diagnosis classes (the 2019 classes plus a mapped "unknown"). The melanoma probability is used as the score.
- Data: trained on 2020 + 2018/2019 data together.
- Validation: 5-fold CV on all the data. They tracked "cv_all" (stable because 2019 is about 18% melanoma) and **never used the public leaderboard**.
- Augmentation: heavy Albumentations, including distortions, CLAHE and cutout.
- Training: 15 epochs, cosine schedule, batch 64, AMP, V100s with up to 8 GPUs. The repo says 15–45 h per model for all 5 folds.
- Combination: an unweighted average of **probability ranks**.

The 1st-place team's own table (cv_all / private LB):

| Comparison | Result |
|---|---|
| Ensemble | 0.9845 / **0.9490** |
| Best single model | 0.9771 / best single private score 0.9408 |
| B4 resolution sweep: 448 / 640 / 768 / 896 | cv_all 0.9748 / 0.9767 / 0.9771 / 0.9744 (**flat**) |
| B4 640 with vs without metadata | 0.9765 vs 0.9767 cv_all; 0.9302 vs 0.9299 private (**null**) |
| 4-class vs 9-class B5 | 0.9765 vs 0.9771 cv_all; private 0.9362 vs 0.9291 |

- The authors say image-only models beat metadata models overall and that metadata models were kept for diversity.
- Standard deviations of single-model scores: cv_all 0.0012 < cv_2020 0.0043 < private 0.0060 < public 0.0093. They found public-LB scores uncorrelated with cv_all for single models.
- The paper calls the diagnosis target a key to winning but includes **no ablation** of it.
- Grouping is not stated in the paper, and folds differ between models.

**2nd place** [V] (write-up `documentation.pdf` in [i-pan/kaggle-melanoma](https://github.com/i-pan/kaggle-melanoma)):
- Data:
  - The 2019 data was used to train a 3-class model (MEL / NV / other).
  - That model pseudo-labelled the 2020 "unknown" images as nevus, with the threshold at the 5th percentile of known nevi.
  - 2020 melanomas were upsampled ×7 so that half of all melanomas came from each source.
- Models: EfficientNet-B6 at 512 and B7 at 640, with GeM pooling and multisample dropout.
- Training:
  - RandAugment, which he calls "crucial" (no number given).
  - AdamW with warm restarts.
- Pseudo-labels: soft pseudo-labels on the **test set**, then retraining.
- Validation: Deotte's "triple-stratified" folds, described as preventing leakage.
- Metadata: models with metadata did worse on the private LB, so none was used.
- Final entry: 15 models, 0.9485 private.
- **His best single model, a 5-fold B6 at 512 with pseudo-labels, scored 0.9495 — better than his own ensemble.**
- Compute: 4× Quadro RTX 6000. B6 at 512 took about 2 h per fold and B7 at 640 about 4 h.
- With only 77–78 melanomas in the public LB, he judged that many competitors overfit it.

**3rd place** [V] ([README](https://github.com/Masdevallia/3rd-place-kaggle-siim-isic-melanoma-classification)):
- Ensemble: 8 image models at 256/384/512/768, half trained on 2020 only and half on 2019+2020, some with hair augmentation.
- A metadata baseline was blended at weight 0.2–0.3.
- Hardware: Kaggle TPU v3-8.
- Results:
  - With external data: 0.9484 private / 0.9620 public.
  - **Without external data: 0.9489 private.** This entry was not selected.

**Post-challenge analyses:**
- Kurtansky et al., *JEADV* 2025 [V] (abstract; [PubMed 39648687](https://pubmed.ncbi.nlm.nih.gov/39648687/)):
  - The top 50 of 3,308 entries scored AUROC 0.943–0.949.
  - Few used intra-patient patterns.
  - Contextual images did not change readers' sensitivity or specificity (60.0/86.7 → 60.0/85.7).
  - Performance varied by image source.
- PROVE-AI, a prospective test of the 2020 winner [V] ([PMC10338483](https://pmc.ncbi.nlm.nih.gov/articles/PMC10338483/)):
  - 603 lesions with 95 melanomas.
  - AUROC 0.857.
  - At the pre-set threshold: sensitivity 96.8%, specificity 37.4%.
  - Specificity 46% under age 65 vs 24% at 65+; sensitivity 97% in both groups (described as underpowered).
  - By site: head/neck specificity 17%.
  - Lesions under 3 mm: AUC 0.625.
  - The cohort was 96% White with mean age 59, so nothing can be learned about under-40 patients.

### ISIC 2024 SLICE-3D (Kaggle; 3D-TBP tiles, not dermoscopy; pAUC above 80% TPR)
Official summary [V] ([Kurtansky et al., npj Digit Med 2025, PMC12639164](https://pmc.ncbi.nlm.nih.gov/articles/PMC12639164/)):
- Data:
  - 401,059 training tiles.
  - Private set: 370,704 lesions from 935 patients, with 342 malignant (prevalence about 0.09%).
- Field: 2,739 teams.
- Winner: pAUC 0.173, AUC 0.967.
  - The top five scored within 0.001 of each other.
  - **Prize winners jumped 30, 23, 764, 1,137 and 1,100 places from the public LB.**
  - 95.5% of all submissions scored lower on private than on public.

The organisers' ablation of the winner:

| Variant | pAUC | AUC |
|---|---|---|
| Full solution | 0.173 | 0.967 |
| Without patient-context features | — | 0.956 (p<0.001) |
| Without image tiles | — | 0.957 |
| Image-only | **0.142** | 0.922 |
| Vectra metadata only | — | 0.939 |

**1st place** (Ilya Novoselskiy) [M] ([mirror of the write-up](https://github.com/MohammadErfan-Jabbari/isic-2024-skin-cancer); [original repo](https://github.com/ilyanovo/isic-2024), README [V]):
- Image models: EVA02-small (MIM, IN-22k) and EdgeNeXt-base (IN-21k).
  - 5-fold StratifiedGroupKFold.
  - Batches balanced 1:1 between classes.
  - Out-of-fold image predictions feed CatBoost, LightGBM and XGBoost.
- Patient-normalised features: each score divided by the patient's mean, plus LOF features.
- GBDT ensemble: 10 seeds × folds; predictions rank-averaged.
- What did **not** help:
  - Pre-training on earlier ISIC data: "no significant improvement".
  - TTA: no gain.
  - Synthetic data: private 0.140 vs 0.142.
  - Growing the GBDT bag from 45 to 150 models: no significant gain.
- Hardware: one A6000.

**Other places:**
- **3rd place** [V] ([README](https://github.com/kyohei-123/kaggle-isic-2024-3rd-place-solution)): ConvNeXtV2-nano and ViT-tiny/small feeding GBDTs, trained on one RTX 3090.
- **2nd and 5th places** [S]: GBDT ensembles; folds described as "triple-stratified leak-free" and "patient-grouped" respectively.

**Relevance to V6.** The benchmark is dominated by patient context and tabular features, both of which your image-only, single-lesion policy excludes. What carries over:
- Image-only scores fall far short of the full solution (pAUC 0.142 vs 0.173).
- Group CV by patient.
- Do not trust the public leaderboard.

### MILK10k / ISIC 2025 (11 classes; paired clinical and dermoscopic images plus metadata)
- Metric: macro-F1 ("Dice") at a fixed 0.5 threshold. Training set 5,240 lesions; test set 479 lesions. [V] ([landing page](https://challenge.isic-archive.com/landing/milk10k/))
- The benchmark is still open. [V] ([leaderboard](https://challenge.isic-archive.com/leaderboards/milk10k/))

| Rank | Team | Score | Entry label | External data |
|---|---|---|---|---|
| 1 | Allegoria | 0.705 | "3D-ViT v3" | Yes |
| 2 | Tmp | 0.669 | — | Yes |
| 3 | WPI-ECL | 0.662 | "TwinSight v2.0" | Yes |
| 6 | Rohan Iyer | 0.609 | ConvNeXt + Swin-V2 ensemble | No |
| 8 | ENS | 0.604 | SwinV2 SSL-22K + TTA | Yes |

No winning write-up was readable. **Evidence value is low** because the inputs (clinical photos and metadata) are ones you exclude.

---

## 2. Failure analyses and critiques (cross-cutting)
- **Distribution shift is the dominant failure.**
  - 2019: the top system fell from 82.0% to 58.8% BACC on BCN images (Combalia [V]).
  - 2018: entries did worse on the external test partition (Codella [V]).
  - For images from sources not in training, the gap between readers and the top-3 algorithms narrowed, and the authors list lower out-of-distribution performance as a limitation. [V] ([Tschandl, Lancet Oncol 2019, PubMed](https://pubmed.ncbi.nlm.nih.gov/31201137/))
- **Artefacts.** Models trained on images with the lesion information removed still beat a dermatologist benchmark. [V], abstract ([Bissoto 2019](https://arxiv.org/abs/1904.08818))
- **Duplicates and leakage across ISIC years** [V] ([curation repo](https://github.com/mmu-dermatology-research/isic_duplicate_removal_strategy)):
  - 12,038 duplicate training images across releases.
  - 1,592 duplicate test images.
  - 258 overlaps between the 2019 training set and the 2016 test set.
  - The paper that found them [V], abstract ([Cassidy et al., *Med Image Anal* 2022, PubMed 34852988](https://pubmed.ncbi.nlm.nih.gov/34852988/)): duplicates appear within and between the 2016–2020 releases, including across training and test sets, and **14,310 duplicate images** were removed from the combined training set.
- **Age.** No challenge paper reports age-stratified *algorithm* performance below 40.
  - PROVE-AI only splits at 65.
  - One prospective reader study [S] ([arXiv 2401.14193](https://arxiv.org/abs/2401.14193)) found *dermatologist* melanoma sensitivity lower under 35 (0.571 vs 0.737).
  - **The winners' literature gives you no evidence on your headline failure.**
- **CV-to-test optimism.** DAISYLab 2019: CV 72.5 → test 63.6. Part of the gap comes from selecting the ensemble on the reporting folds and from the UNK class (sensitivity 0.3%). [V]

---

## 3. Leakage: public weights you must not use (your evaluation data is ISIC 2019 ⊃ HAM10000)

| Source | Trained with labels on | Usable? |
|---|---|---|
| 2018 Task 3 models, all teams | HAM10000, which is inside ISIC 2019 | **No** |
| 2019 DAISYLab, Dysion, AImageLab | ISIC 2019 | **No** (whether weights were released was not checked) |
| 2020 1st place: weights published on Kaggle (`boliu0/melanoma-winning-models`) [V] | 2018 + 2019 + 2020 | **No** |
| 2020 2nd and 3rd places | 2019 + 2020 (and 2020 test pseudo-labels) | **No** |
| 2024 1st place's 3-class EVA02 | "External" dermoscopy data with bkl/mel/nevus labels (repo not named) [M] | **No; assume contaminated** |
| Any HAM10000, ISIC or DermaMNIST checkpoint on timm or HF Hub | Usually HAM10000 or ISIC | **No** |
| ImageNet-1k/22k, noisy-student, EVA02-MIM backbones | No ISIC labels | Yes |

**Using ISIC 2020 or later archive images as extra *training data*** is allowed in principle. It needs two steps first:
1. Remove image duplicates against your ISIC 2019 pool (the curation repo above gives the method).
2. Check patient overlap, because HAM, MSK and Barcelona sources recur across years. Otherwise the lesion-grouped folds leak.

---

## 4. Synthesis: the ingredients that recur, and their status in this project

| # | Ingredient (winners using it) | Strongest measured evidence | Status here |
|---|---|---|---|
| 1 | **More training data, from multiple sources** (all 2017 top 3; DAISYLab 2018/19; all 2020 top 3) | Valle ANOVA: largest factor (46%). Gessert 2018: +0.009 BACC. Casio: SK +0.011 AUC. 2020 3rd place counter-example: no-external entry 0.9489 vs 0.9484. | **(a) Done** (pooled archives +0.19; young histopathology-confirmed images +0.0225 pAUC). ISIC 2020 and newer archive collections **not yet added**. |
| 2 | **Large, diverse ensembles: architectures × resolutions × folds/snapshots, simple or rank averaging** (every winner 2017–2020) | Ensemble over best single: 2018 test +0.014 (MetaOptima); 2018 CV +0.034; 2019 CV +3.7 (selected) / +2.9 (plain average); 2020 private +0.008. Counter-examples: 2020 2nd place's best single model beat his ensemble; 2024 bag 45→150 no gain. | **(a) partly**: same-architecture seeds and a 6-CNN vote (+0.026 HAM; seed vote +0.008–0.025). **Multi-resolution × multi-architecture ensembles on pooled data: (c) never tried.** |
| 3 | **High resolution, 380–896 px** (2019, 2020) | Single model: Valle +2 AUC points (598 vs ~300 px). 2020 B4 448→896 **flat**. Gessert B0-224 → B6-528 +3.0, confounded with capacity. | **(b) Tested fairly *as a single model*: your null agrees with the winners' own flat curves.** Untested as an *ensemble-diversity* ingredient, which is how winners actually used it. |
| 4 | **Inverse-frequency class-weighted CE** (2018/2019 top 3 all used it) | Gessert 2018 single model: 0.757 → 0.795 BACC. Oversampling 0.789. Focal loss no gain (2019). | **Unknown**: you report LDAM/ASL below "plain CE". **Check whether your CE is class-weighted; if not, it is (c).** |
| 5 | **Dense multi-crop / TTA** (2017–2020) | Valle: 12% of variance, +2 AUC points. Gessert 2018: 36 vs 4 crops 0.795 vs 0.662 (partly a coverage artefact). 2024: no gain. | **(a) Done** (24-view, +0.014). |
| 6 | **Native-scale crops without aspect warping, plus cropping the black dermoscope border** (DAISYLab 2018/19, DysionAI 2019) | DAISYLab cropped 6,226 / 3,864 images; no isolated ablation. DysionAI's crop setting 0.626 vs 0.726 (ambiguous). SS ≈ RR. | **(c) Never tried** as the winners did it. Your "lesion-crop zoom" failure is a different operation. |
| 7 | **Shades-of-Gray colour constancy** (DAISYLab 2019, Dysion 2019, Casio 2017) | **No ablation in any winner's paper**; listed as a claim only. | **(c) Never tried** (dropped). Your measured 18.8 ms/image cost goes away if you precompute it to disk once. |
| 8 | **Fine-grained / diagnosis-level target** (2020 1st: 9-class; 2nd: MEL/NV/other plus pseudo-labelled nevus) | **Claim only.** 1st place's one 4-class model has near-identical cv_all. | **(c) Never tried.** The 8-class split exists; the finer diagnosis subtypes in BCN/MSK/archive metadata are unused. |
| 9 | **Source-balanced sampling** (2020 2nd: 50/50 melanomas per source) | Claim only. | **(c) Never tried.** |
| 10 | **Pseudo-labelling** (2020 2nd) | Best single model 0.9495 included pseudo-labels; no clean ablation. It used the test set (transductive). | **(c)**. The test-set variant violates your touch-once rule. Unlabelled archive images would be the legitimate version. |
| 11 | Metadata branch (2019 Task 2, 2020) | **Null or negative on held-out data in every winner's own numbers** (63.4 < 63.6; 0.9302 ≈ 0.9299; 2nd place worse). | Excluded by policy, and the evidence supports excluding it. |
| 12 | Learned stacker / SVM meta-learner (2017 RECOD, 2018 DAISYLab) | Mixed: Gessert +0.027 on one model. Valle: SVM layer harmful. 2017 non-linear fusion < linear. | **(b) Your Caruana selection overfit**, consistent with the literature. Use plain or rank averaging. |
| — | Patient context / "ugly duckling" (2024) | Large effect (AUC 0.967 vs 0.956; image-only pAUC 0.142 vs 0.173). | Outside policy, and the data is not available. |

### Ranking of the never-tried or unfairly-tested ingredients

**(i) For balanced accuracy / Macro-F1** (expected value against cost):
1. **#4 class-weighted CE, if it is not already on.** This is the largest single-model effect any winner measured (+0.038 BACC) and costs nothing extra.
2. **#2 multi-resolution, multi-architecture ensemble with rank or plain averaging, and no weight fitting.** Winners got +1 to +4 BACC points over their best single model. Ensembling already works here, so this is the most reliable extension. A concrete low-cost version:

| Members | Time |
|---|---|
| ConvNeXt-T at 224 + 384, 5 folds | 5×41 + 5×(64–75) = **525–580 min ≈ 8.8–9.7 h** |
| A third member (e.g. EfficientNet-B3 or B4 at 300–380) | Not yet measured |

   - Basis for the first row: the per-fold times you measured (41 min at 224, 64–75 min at 384), multiplied out. This is an extrapolation and assumes your current 30-epoch schedule.
   - Third-member time: per the GPU timing rule, run `scripts/gpu_benchmark.py` before committing. VRAM is also unmeasured.
3. **#6 + #7 for the BCN rows:** native-scale crops, a dark-border field-of-view crop, and precomputed Shades-of-Gray. This targets the documented 82 → 58.8 HAM-to-BCN collapse and your finding that BCN rows did not gain at 384 px. The winners' evidence here is circumstantial, so treat it as a hypothesis test, not an expected gain.
4. **#1 extension:** add ISIC 2020 and post-2019 archive images after deduplication. More data is the largest factor in the only factorial study.
5. **#8 fine-grained auxiliary target.** Cheap (a head change), but supported by claims only.

**(ii) For under-40 melanoma ranking:**
1. **#1 more young melanomas.** ISIC 2020 has 584 histopathology-confirmed melanomas ([S], snippet of the Rotemberg et al. *Sci Data* 2021 dataset paper). It also has an `age_approx` field ([V], the 2020 1st-place paper uses it as a feature). Later archive collections are a further source, after deduplication. The 81 positives are the binding constraint; data volume dominates every ablation; and your curated young images already moved pAUC +0.0225. **Check age availability and duplicates before planning.**
2. **#8 fine-grained or auxiliary labels** (melanoma in situ vs invasive; nevus subtypes such as Spitz/Reed, dysplastic and congenital where the metadata has them). This gives the network an explicit reason to separate the nevus-like young melanomas from nevi. The mechanism fits your failure, but the evidence is weak.
3. **#9 source- or age-balanced sampling of escalating cases.** Cheap; weak evidence.
4. **#2 multi-resolution ensemble.** Ensembling helps averages, but no winner reports age strata. Your 3-seed vote result should tell you whether diversity reaches the under-40 tail.
5. **#6/#7.** Low expected value for this axis, because the misses are pigmented nevus-like lesions, not colour or border artefacts.

**Not recommended**, based on the winners' own evidence:
- Metadata.
- Learned stackers or ensemble-weight fitting on a small validation set.
- Synthetic lesions (2024: 0.140 vs 0.142).
- Pre-training on prior ISIC labels (it is leakage here, and showed no gain in 2024).
- Chasing resolution beyond 384 px in a single model (2020 B4 curve flat).

**Bottom line.** Across ten years of challenges, only three ingredients have measured, repeatable support: **more data, ensembles of diverse models, and TTA**. You have done all three in some form. The untested pieces with real evidence behind them are class-weighted CE (check it), a multi-resolution, multi-architecture ensemble, and adding ISIC 2020 plus archive data with deduplication. None of the winners measured anything about patients under 40, so V6's under-40 work has no prior art from the challenges to lean on.
