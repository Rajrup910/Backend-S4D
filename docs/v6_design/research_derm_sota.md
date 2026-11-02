# V6 literature scan: dermoscopic classification outside competitions (2021–2026)

**Tags.** VERIFIED = I read it in the primary source (arXiv HTML, Europe PMC full text, model card, repository or abstract record). SECONDARY = search snippet or summary only.
**Leakage notation.** [LG] = lesion- or patient-grouped split, or a true external test. [IMG?] = split unit not stated. [LEAKY] = image-level split shown to leak.
**Blocked or not reached.** PMC (reCAPTCHA; worked around via Europe PMC REST where possible), nature.com and Springer (login redirect), medRxiv and MDPI (403). Mehta 2023 AUROC values and MONET's Methods text could not be retrieved.

---

## A. Priority check: pretraining corpora and HAM / BCN / ISIC 2019 numbers

| Model | HAM10000 in pretraining? | BCN20000? | ISIC 2019 (incl. MSKCC) or other ISIC? | Dedup vs evaluation sets described? | Tag |
|---|---|---|---|---|---|
| **PanDerm** (Nat Med 2025) | **No** | **No** | **No ISIC 2019.** The only ISIC source is **ISIC2024 (352,034 TBP tiles)**. Other sources are in-house Australian: MYM and HOP TBP (Brisbane), MMT (>150 clinics in Australia and New Zealand: 316,399 dermoscopic + 310,951 clinical), NSSI (29,832 dermoscopic, Brisbane), Edu1/Edu2 educational clinical photos, TCGA-SKCM, ACEMID and UAH89k pathology | **No.** Only a claim that "Our strategy minimizes this risk", contrasted with SwAVDerm, which "inadvertently incorporat[ed] images from … ISIC" | VERIFIED: published version ([Europe PMC full text PMC12353815](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12353815/fullTextXML)) and [arXiv v3](https://arxiv.org/html/2410.15038v3) |
| **DermFM-Zero / PanDerm-2** (preprint Feb 2026) | Not listed (used only as the HAM-7 evaluation) | Not mentioned | **Indirectly yes.** Stage 1 is "a comprehensive expansion of the dataset used for PanDerm-v1" (so ISIC2024 is likely), **plus 403,563 Derm1M pairs**, and Derm1M contains MSKCC (next row). Stage 2 adds 240,246 Derm1M pairs, 199,600 educational pairs and 574,239 private Molemap images (141 clinics, AU/NZ). The 15 Stage-1 sources are **not named** | **No** | VERIFIED ([arXiv 2602.10624](https://arxiv.org/html/2602.10624)) |
| **DermLIP / Derm1M** (ICCV 2025) | Not listed as a source. HAM is used for evaluation; its labels were used to build the disease vocabulary | Not mentioned | **Yes: MSKCC.** "We incorporated two public datasets including SCIN [37] and MSKCC [6]". Ref [6] is the **ISIC 2017 challenge paper (Codella et al.)**, so these are ISIC's MSK collections, part of the ISIC 2019 pool. **10,619 MSKCC image-text pairs.** The other 90% comes from YouTube, PubMed, forums and education, where reproduced ISIC figures are possible but unquantified | **No** | VERIFIED ([arXiv 2503.14911](https://arxiv.org/html/2503.14911v2)) |
| **MONET** (Nat Med 2024) | Not stated | Not stated | Not stated. Trained on 105,550 image-text pairs "from a large collection of medical literature"; PubMed subset scripts are in the repo. ISIC was used for auditing, not training (SECONDARY). PubMed figures can reproduce ISIC images; no overlap check is described | Not described | VERIFIED README ([repo](https://github.com/suinleelab/MONET)); "PubMed articles and medical textbooks" is SECONDARY |
| DermINO (preprint 2025) | Indirectly yes | Indirectly yes | **Yes: "ISIC-Duplicate" (57,062 images)**, the curated ISIC 2016–2020 union. **Direct overlap with your pool** | No | VERIFIED ([arXiv 2508.12190](https://arxiv.org/html/2508.12190)) |
| MedSigLIP / Derm Foundation | ISIC not listed | — | Not listed. Dermatology sources are PAD-UFES-20, SCIN, Colombian teledermatology, and Australian skin-cancer images (clinical + dermoscopic). HAM10000 also has a Queensland source, so overlap is **possible, unestablished** (my inference) | Not described | VERIFIED ([MedSigLIP card](https://huggingface.co/google/medsiglip-448), [Derm Foundation card](https://developers.google.com/health-ai-developer-foundations/derm-foundation/model-card)) |

**Reported HAM10000 / BCN20000 / ISIC numbers for these models** (all VERIFIED unless marked):
- **PanDerm, linear probe (LP)** ("For all other tasks, we used linear probing", published Methods). Ext. Data Tables 22–23, arXiv v3. W_F1 / AUROC / BACC at 100% labels:

  | Model | HAM-clean (7-class, [LG] 60/20/20) | BCN20000-clean (9-class, 60/20/20) |
  |---|---|---|
  | **PanDerm** | 0.922 / 0.988 / **0.797** | 0.767 / 0.951 / **0.647** |
  | DINOv2 | 0.876 / 0.962 / 0.686 | 0.705 / 0.923 / 0.565 |
  | ImageNet-supervised | 0.872 / 0.967 / 0.652 | 0.698 / 0.910 / 0.590 |

  - HAM-clean "prevents data leakage by ensuring that lesions from the same patient are not split across sets". BCN-clean "addresses data leakage issues"; its construction is not detailed.
  - v1 and v3 agree. A BCN figure of 0.912 / 0.949 / 0.774 that I extracted earlier was a mis-extraction and is retracted.
- **PanDerm, LP vs fine-tune (FT)**, accuracy only (v3 Ext. Data Table 37): HAM_c LP 97.40 vs FT 98.03; BCN LP 95.19 vs FT 97.65. The scale suggests a different, probably binary, task. **No published 7- or 9-class fine-tuned PanDerm number exists for HAM or BCN.**
- **No ISIC 2019 numbers** are reported for any of the four models.
- **Independent, same [LG] HAM split** (Mahbod et al., [arXiv 2505.16338](https://arxiv.org/html/2505.16338v1); ref [20] = PanDerm, n_test 1,232):

  | Model | HAM BACC | MSKCC binary BACC |
  |---|---|---|
  | Frozen PanDerm + MLP | **79.57** | 64.40 |
  | Fully fine-tuned Swin-V2-B | **81.43** | 64.52 |
  | Fusion of the two | **83.25** | 64.36 |

- **DermFM-Zero, HAM-7 LP, 100% labels.** M_F1 0.733 / BACC 0.740, vs PanDerm 0.716 / 0.710 and MONET 0.667 / 0.646. The split is "stratified by class labels" [IMG?]. At 30% and 50% labels, PanDerm beats DermFM-Zero on M_F1 (0.642 vs 0.602; 0.683 vs 0.654).
- **Derm1M paper, HAM LP accuracy at 1/10/100% labels.** Accuracy only, test split not specified [IMG?]:

  | Model | 1% | 10% | 100% |
  |---|---|---|---|
  | DermLIP-PanDerm-B | 0.755 | 0.817 | 0.881 |
  | PanDerm-L | 0.711 | 0.790 | 0.878 |
  | PanDerm-B | 0.703 | 0.774 | 0.854 |
  | MONET | 0.707 | 0.800 | 0.850 |
  | BiomedCLIP | 0.705 | 0.697 | 0.687 |

- **Frozen probes on DERM12345** (patient-level split, light adapters, weighted F1, 40 subclasses; [arXiv 2601.12382](https://arxiv.org/html/2601.12382)):
  - MedSigLIP 69.79, Derm Foundation 69.50, MONET 69.31, DINOv2-G 68.00, DINOv3 66.78, PanDerm-B 64.12, BiomedCLIP 58.78.
  - **PanDerm-L 36.65.** Frozen PanDerm-L is fragile to the extraction pipeline, consistent with your failed probe.

**Implication for you.** PanDerm is the cleanest of the four with respect to your pool (no HAM, BCN or ISIC 2019; ISIC2024 TBP crops come from overlapping institutions but a different modality). **DermLIP and DermFM-Zero saw MSKCC images with captions.** Use them only if MSKCC is excluded from evaluation or reported separately as contaminated.

---

## B. Calibration: where honest numbers sit

| Source | Protocol | Result | Tag |
|---|---|---|---|
| Gessert et al., ISIC 2019 winner ([ar5iv 1910.03910](https://ar5iv.arxiv.org/html/1910.03910)) | [LG] 5-fold ("all images of the same lesion are in the same fold"), 8 classes, BACC, with external data | Single EffNet-B0 224 px **65.8**; B6 528 px **68.8**; image-only ensemble **72.5**; official test **63.6** | VERIFIED |
| Combalia et al., Lancet Digit Health 2022 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/35461690/)) | Challenge test, 129 algorithms | Best BACC **58.8% on BCN20000 vs 82.0% on HAM10000** | VERIFIED (abstract) |
| BCN20000 Sci Data 2024 baselines ([Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11183228/fullTextXML)) | 75/5/20 split, grouping not stated; text and caption disagree on CV vs test [IMG?] | ResNet18–EffNet-B2 BACC **0.41–0.46**. Lesion crops +2.5 points | VERIFIED |
| PLCRD ([arXiv 2607.18773](https://arxiv.org/html/2607.18773)) | [LG] lesion-disjoint HAM10000; external ISIC 2018 test | ConvNeXt-T Macro-F1 **0.739** (external 0.727). Swin-T 0.754. PLCRD 0.773 | VERIFIED |
| Abhishek et al., Sci Data 2025 ([arXiv 2401.14497](https://arxiv.org/html/2401.14497v2)) | DermaMNIST image-level split | **1,006 of 7,470 HAM lesions (13.47%) span more than one partition** | VERIFIED |

**Reading.** Your pooled single-model BACC of 0.64–0.67 sits at the honest single-model level (Gessert 65.8–68.8). Your HAM-only ConvNeXt-T Macro-F1 of 0.746 matches PLCRD's lesion-disjoint 0.739.

Not ranked here, because the protocol is [LEAKY] or [IMG?]:
- DINOv2 HAM accuracy 0.97 (SECONDARY).
- DINOv2-L HAM Macro-F1 0.85, protocol undescribed ([arXiv 2502.02500](https://arxiv.org/abs/2502.02500)).
- An SSL transformer at "97.32% HAM10000" (SECONDARY).

---

## C. Foundation models fine-tuned (not frozen): practicalities

### PanDerm fine-tuning recipe (`PanDerm_Large_FT`, [repo](https://github.com/SiyuanYan1/PanDerm); VERIFIED)
- `--layer_decay 0.65 --drop_path 0.2 --lr 5e-4 --batch_size 128 --epochs 50 --warmup_epochs 10 --weight_decay 0.05 --mixup 0.8 --cutmix 1.0`, weighted sampler, TTA at test.
- `--update_freq` provides gradient accumulation.
- Variants: ViT-L/16 (304M per the DermFM-Zero text) and ViT-B/16 (86M).
- **Licence CC-BY-NC-ND 4.0.** The "ND" may restrict redistributing fine-tuned weights.

### DINOv2 ViT-L/14 adaptation ([unreviewed repo, Oct 2026](https://github.com/pr4jtk8fhj-dotcom/derm-pretraining-adaptation); VERIFIED that the page states this)
- Task: external HIBA+MSKCC binary AUROC, patient-clustered bootstrap.

  | Adaptation | AUROC |
  |---|---|
  | Frozen | 0.802 |
  | **LoRA r16** | **0.885 ± 0.002** |
  | Full fine-tune | 0.879 ± 0.002 |
  | Full fine-tune, 448 px | 0.892 |

- **The frozen-to-tuned gap is +0.08.**
- Full fine-tuning at 448 px with gradient checkpointing used 24.1 GB on an H100.

### Other models
- **MedSigLIP:** 400M vision + 400M text encoders at 448 px, HAI-DEF terms. It is the strongest frozen dermatology model on DERM12345.
- **Derm Foundation:** BiT-M R101x3, now legacy ("Use MedSigLIP").
- **BiomedCLIP:** weakest in every derm benchmark found.
- **EVA-02 and SigLIP 2:** no non-competition dermoscopy result found (searched).

### Memory (arithmetic, unmeasured)
Full AdamW needs about 16 bytes per parameter before activations:

| Model | Optimiser state | Verdict |
|---|---|---|
| ViT-B | ≈ 1.4 GB | Comfortable |
| **ViT-L** | ≈ 4.9 GB | Marginal. Needs gradient checkpointing, micro-batch 8–16 plus accumulation, or 8-bit Adam |
| MedSigLIP vision | ≈ 6.4 GB | **LoRA only** |

**Benchmark the unfrozen stage with `scripts/gpu_benchmark.py` before quoting any time.**

---

## D. Domain-specific self-supervised pretraining

| Study | Setup | Gain vs ImageNet | Tag |
|---|---|---|---|
| Sirotkin et al. ([ar5iv 2112.12086](https://ar5iv.labs.arxiv.org/html/2112.12086)) | ResNet-50 SSL curriculum on ISIC 2019 training images, 8-class BACC, split unit unstated [IMG?] | **73.76 → 75.44 (+1.7)**, with about 5× fewer iterations than ImageNet pretraining | VERIFIED |
| Gröger et al., ML4H 2024 ([arXiv 2411.05514](https://arxiv.org/html/2411.05514v1)) | DINO/iBOT/BYOL/SimCLR on 242k images (107k ISIC), ViT-T and ResNet-50, 8×V100 | **Frozen kNN only.** HAM F1 34.6→46.3. About 2× label efficiency | VERIFIED |
| ExPLoRA, ICML 2025 ([arXiv 2406.10973](https://arxiv.org/html/2406.10973v5)) | Continued DINOv2 SSL, last block unfrozen plus LoRA | Camelyon17 LP **+1.14**, LoRA-FT **+1.24**, at **90 GPU-h** | VERIFIED |
| Chaves et al., ECCV-W 2022 ([arXiv 2106.09229](https://arxiv.org/pdf/2106.09229)) | SimCLR/SwAV/BYOL | Better in- and out-of-distribution in "almost all" datasets; biggest gains at low data | SECONDARY |

**Feasibility.** ViT-L continued pretraining (about 90–100 GPU-h on data-centre hardware) is **not feasible in 20–40 h on an RTX 5050**. ViT-S or ConvNeXt-T on about 100k archive images might be, for an expected +1–2 points. PanDerm already embodies 2.1M-image domain SSL, so fine-tuning it dominates replicating it.

**Leakage.** The ISIC Archive contains your pool. Cassidy's repository lists **12,038 duplicates across the 2016–2020 training sets** ([repo](https://github.com/mmu-dermatology-research/isic_duplicate_removal_strategy), VERIFIED). Exclude with hashing, not filenames.

---

## E. Cross-dataset generalisation (dermoscopy)

| Method | Protocol | Effect | Tag |
|---|---|---|---|
| **Photometric-heavy augmentation** ([arXiv 2607.26765](https://arxiv.org/html/2607.26765v1)) | [LG]. ConvNeXt-L at 224 trained on BCN/Derm12345/HIBA/BALD/MILK10k/Derm7pt. **OOD = HAM + ISIC 2016–2020**, binary AUROC | "mix" policy **0.770 → 0.823 (+0.053, CI +0.045 to +0.061)**; 4-seed mean **+0.047**; in-domain +0.006. Did not persist on a small clinical set | VERIFIED |
| **SWAD** ([arXiv 2402.06150](https://arxiv.org/pdf/2402.06150)) | ResNet-18, leave-one-archive-out over 7 dermoscopy sets, random within-domain splits, accuracy | ERM **0.6287 → 0.7024 (+7.4)** | VERIFIED |
| DG zoo, ISIC 2019 → Derm7pt / PAD / PH2 (EPVT, [arXiv 2304.01508](https://arxiv.org/html/2304.01508v3)) | ViT-B, binary AUROC, 5 seeds | ERM 79.63; DANN 81.87; CORAL 81.55; **IRM 76.64 (hurts)**; **EPVT 84.11** | VERIFIED |
| DANN on age ≤30 domains (Fogelberg, [arXiv 2304.06968](https://arxiv.org/html/2304.06968v2)) | ResNet-50, ISIC domains | Helped 8/10 domains "except for HA (HAM age ≤30) and MA (MSK age ≤30)" | VERIFIED |
| MixStyle ([arXiv 2106.03171](https://arxiv.org/pdf/2106.03171)) | Skin DG | 78.18 vs ERM 78.97 | SECONDARY |
| Test-time adaptation ([Twente thesis](https://essay.utwente.nl/essays/98382)) | Binary | No "stable improvement" | SECONDARY |
| Colour constancy | — | No measured external-test gain found ([Galdran](https://arxiv.org/pdf/1703.03702): validation only) | SECONDARY |

---

## F. Age and young-patient difficulty

- **Heinlein et al., Commun Med 2024** ([arXiv 2401.14193](https://arxiv.org/pdf/2401.14193); text extracted locally). VERIFIED:
  - The system is **ADAE**: 90 models (18 configurations × 5 folds; EfficientNet-B3–B7, SE-ResNeXt-101, ResNeSt-101). Trained on **ISIC 2019+2020, 58,457 lesions, 5,106 melanomas**. Fed **6 real images per lesion**.
  - <35 years: **AUROC 0.974 (0.942–0.997) vs 0.879 for >74**, from **only 14 melanomas among 177 lesions**.
  - Dermatologists in that band: sensitivity **0.571** vs 0.737 elsewhere, specificity 0.963.
  - AUROC is lower for lesions ≤ 6 mm (0.802 vs 0.917).
  - ADAE recipe ([ar5iv 2010.05351](https://ar5iv.labs.arxiv.org/html/2010.05351)): **9-class diagnosis target**, resized to 384–896 px, 4 of 18 configurations with metadata. The authors write that "models with images only perform better than metadata models overall".
- **Mehta et al., JID 2023;143:1423.** VERIFIED abstract via [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=PMCID:PMC10431965&resultType=core&format=json):
  - Adding **1,536 paediatric images** to 37,662 ISIC images "improved algorithm performance on pediatric images" with no adult loss.
  - **Background skin** drove the gain.
  - AUROC values could not be retrieved.
- **Fogelberg et al.** VERIFIED. Public young-melanoma scarcity, age ≤30 on the body, melanoma : nevus image counts:
  - HAM 25 : 532
  - BCN 71 : 808
  - MSK 37 : 427
- **NCT06621810 (AI-MEL, DKFZ)** ([record](https://clinicaltrials.gov/study/NCT06621810)). Under-30 focus, about 3,000 participants in Germany, Italy and Spain, completion around 11/2026. SECONDARY. A future external test for your failure mode.
- **External young-data sources** (SECONDARY):

  | Source | Size | Diagnosis | Licence |
  |---|---|---|---|
  | **MILK10k** ([ISIC](https://challenge.isic-archive.com/landing/milk10k/)) | 5,240 lesions, paired clinical + dermoscopic | 95.7% histopathology | CC-BY-NC |
  | **DERM12345** (Türkiye; [Sci Data](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11604664/)) | 12,345 images, 1,627 patients | — | CC-BY |
  | **HIBA** ([ISIC](https://api.isic-archive.com/doi/hiba-skin-lesions/)) | 1,616 images, 1,246 lesions, 623 patients | — | CC-BY |
  | **ISIC 2020** ([arXiv 2008.07360](https://arxiv.org/pdf/2008.07360)) | 33,126 images, 2,056 patients, 584 melanomas | — | — |

  ISIC 2020 shares Barcelona, Vienna and MSKCC as sources with your pool, so it **needs patient-level deduplication**. Age distributions were not verified for any of these.
- **Dysplastic nevi and featureless melanoma:** no dermoscopy deep-learning study with histopathology-confirmed atypical-nevus results was found.

**What the systems that did well in young patients did differently:**
1. Much more data: 58k lesions, including ISIC 2020's benign mass.
2. A multi-class target.
3. High resolution and large ensembles.
4. Multiple real views per lesion.
5. Age-targeted enrichment with surrounding skin kept in frame.

Every number rests on 14–71 young melanomas.

---

## G. Synthesis: top 8

How to read the table:
- EV = my judgement, not a measurement. (i) is pooled Macro-F1/BA; (ii) is under-40 escalation pAUC.
- "Fits?" is arithmetic only.

| # | Technique | Best evidence (protocol) | Fits 8.55 GB? | Leakage risk | Tried here? Fair? | EV (i) | EV (ii) |
|---|---|---|---|---|---|---|---|
| 1 | **Age-targeted histopathology-confirmed enrichment** (young melanomas and young nevi from MILK10k, DERM12345, HIBA, ISIC 2020) | Mehta 2023 (paediatric add-in, no adult loss); your own +0.0225 pAUC; Fogelberg (≤30 is a real shift) | Yes | **High without hash and patient deduplication** | Yes, positive. Fair but small | Low–mod | **High** |
| 2 | **Fine-tune PanDerm-B, then -L, with LLRD 0.65** + drop-path + mixup/cutmix, as an **ensemble member** | LP BACC 0.797 vs 0.652 ImageNet ([LG] HAM-clean, author-run); 0.647 vs 0.590 on BCN-clean; independent fusion **+1.8 BACC** | B yes; L marginal | **Low** (no HAM, BCN or ISIC 2019) | **No.** Only a frozen probe (fragile for PanDerm-L) and DINOv3 without LLRD | **Mod–high** | Mod |
| 3 | **LoRA r16 on DINOv2-L or MedSigLIP** | LoRA 0.885 vs frozen 0.802 external AUROC (unreviewed); MedSigLIP best frozen on DERM12345 | Yes with LoRA plus checkpointing | DINOv2 none stated; MedSigLIP possible AU overlap | No | Mod | Mod |
| 4 | **Photometric-heavy augmentation** | +0.047 OOD AUROC, [LG], held-out HAM/ISIC | Yes | None | Standard augmentation only | Low | Low–mod |
| 5 | **SWAD** (beyond EMA) | +7.4 accuracy on a 7-archive skin DG benchmark | Yes | None | EMA only | Low–mod | Low |
| 6 | **ADAE-style scale-up** (ISIC 2020 benign mass, 9-class target, real multi-view TTA) | ADAE <35 AUROC 0.974 (14 melanomas) | Yes | **High** (shared institutions) | Pooling yes; 2020 no | Mod | Mod |
| 7 | **EPVT / artifact-environment DG** | +4.5 AUROC over ERM (binary ViT-B) | Yes | None | No | Low | Low |
| 8 | **Continued domain SSL** on a small backbone, pool excluded | +1.1–1.7 points; 90 GPU-h for ViT-L | ViT-S/B only | **High** if pool images enter SSL | No | Low | Low–mod |

**De-prioritise.**
- **DermLIP and DermFM-Zero as backbones:** MSKCC is in pretraining.
- **DermINO:** ISIC-Duplicate is in pretraining.
- **BiomedCLIP.**
- **MixStyle and IRM:** both hurt.
- **Test-time adaptation.**
- **Colour constancy:** no external evidence, and it costs 18.8 ms/image.
- **DANN / age-adversarial training:** failed on young domains, matching your result.
- **384 px alone:** your null result. Gessert's gain is confounded with capacity.

**Key caution.** None of your failed frozen probes tests whether a *tuned* domain foundation model helps. Under 40, the evidence points at data and context, not architecture. Pre-register a two-sided test.
