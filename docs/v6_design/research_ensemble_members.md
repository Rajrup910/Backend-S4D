# V6 ensemble-member candidates: what could beat or diversify the planned roster

**Tags.** VERIFIED = I read it in the primary source (paper HTML, model card, repo README). SECONDARY = search snippet or summary only. [LG] = lesion- or patient-grouped split. [IMG?] = split unit not stated. [LEAKY] = image-level HAM-family split. Results at 0.95 or above on HAM are not ranked.
**Scope.** This file adds to `docs/v6_design/research_derm_sota.md` and `research_isic_winners.md` and does not repeat them. PanDerm-B's linear-probe numbers, the Mahbod frozen-PanDerm fusion, the DINOv2-L LoRA repo and the DERM12345 frozen benchmark are already there.
**VRAM and time.** I found no measurement on an 8 GB-class GPU for any candidate. Every VRAM and time entry below is "unmeasured, benchmark needed" (`scripts/gpu_benchmark.py`, unfrozen stage). The only measured anchors are your ConvNeXt-T figures.

---

## 1. The finding that should reshape the roster: diversity comes from the objective and the data, not the architecture

**General evidence (ImageNet, not medical).** Gontijo-Lopes, Dauphin & Cubuk (ICLR 2022), [ar5iv 2110.12899](https://ar5iv.labs.arxiv.org/html/2110.12899), VERIFIED. They measured error inconsistency against a ResNet-50 (the fraction of images where only one model is right):

| How the pair differs | Error inconsistency |
|---|---|
| Different seed | 0.085 |
| Hyper-parameters | 0.086–0.106 |
| Architecture | 0.103 (MobileNet-V2) to 0.147 (ViT-B/16) |
| Objective (SimCLR, same data) | 0.128 |
| Dataset and objective (CLIP, ALIGN) | 0.169–0.230 |

- A zero-shot CLIP-S at 63.25% top-1 still raised a 76.20% ResNet-50 to 78.01%.
- The authors say low-accuracy members help only when they are trained differently enough, with a different loss, dataset or both.

**Same direction elsewhere.**
- Mustafa et al. 2020 ([arXiv 2010.06866](https://arxiv.org/pdf/2010.06866)), abstract and intro VERIFIED: "the nature of pre-training itself is a performant source of diversity" for low-data transfer ensembles.
- Pathology (SECONDARY, [Nat Biomed Eng 2025, DOI 10.1038/s41551-025-01516-3](https://rd.springer.com/article/10.1038/s41551-025-01516-3)): an ensemble of CONCH (image-text) and Virchow2 (SSL) beat either model alone in 55% of tasks.

**Your own data fits this.** SwinV2-T and MaxViT-T are different architectures but share ConvNeXt-T's supervised ImageNet objective. Adding them to the 6-CNN vote was a wash (0.7981 vs 0.7986). That is the "architecture" row above, the weakest source of diversity.

**Implication for the current roster** (my inference from the sources below):
- **PanDerm and EVA-02 are the same kind of model.** Both are ViTs pretrained by masked modelling that regresses *CLIP* features.
  - PanDerm: CAE-v2 codebase with a CLIP-large teacher, described as "pretrained on the natural domain" ([arXiv 2410.15038 v3](https://arxiv.org/html/2410.15038v3), Ext. Data Tables 36 and 40, VERIFIED).
  - EVA-02: "masked image modeling (using EVA-CLIP as a MIM teacher)" ([timm card](https://huggingface.co/timm/eva02_base_patch14_224.mim_in22k), VERIFIED).
- **EfficientNetV2-S is the same kind of member as the anchor.** It is an ImageNet-supervised CNN, the category your Swin/MaxViT result already showed adds little.
- **The roster has no member trained with self-distillation (DINO family) or with image-text contrastive learning (SigLIP/CLIP).** Those are the two objectives with the highest measured error inconsistency against a supervised CNN.

---

## 2. General backbones (about 20–90M parameters)

### 2a. DINOv2 ViT-B/14 (or DINOv3 ViT-S/B), fine-tuned with layer-wise LR decay (LLRD)
- **Pretraining data.**
  - DINOv2: self-supervised. Apache-2.0, 86.6M parameters ([card](https://huggingface.co/facebook/dinov2-base), VERIFIED). The card does not name LVD-142M; the corpus being retrieved from web images, not medical data, is SECONDARY.
  - DINOv3: LVD-1689M, about 1.7B images curated from public Instagram posts. ViT-S/B/L and ConvNeXt-T/S/B/L are all distilled from a 7B teacher. Custom "DINOv3 License", gated access. The card calls fine-tuning a "last resort" ([card](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m), VERIFIED).
  - No dermatology labels were used. No ISIC overlap is stated. Web-scale corpora cannot be audited for stray ISIC images; this is unestablished, not shown.
- **Dermoscopy evidence.**
  - Huix et al., WACV 2024 ([arXiv 2310.19522](https://arxiv.org/html/2310.19522), VERIFIED): ISIC 2019, 80/10/10 split, split unit not stated [IMG?]. End-to-end fine-tuned macro recall:

    | Model (fine-tuned end to end) | ISIC 2019 recall |
    |---|---|
    | **DINOv2 ViT-B** | **0.859 ± 0.007** |
    | SEEM | 0.835 |
    | ImageNet-1k supervised | 0.823 |
    | OpenCLIP | 0.818 |
    | ResNet-152 | 0.817 |
    | DINOv1 | 0.775 |

    Under linear probing DINOv2 scored only 0.569, so the gain comes from tuning, not from frozen features. This matches your failed frozen probes.
  - MedMNIST benchmark ([arXiv 2501.14685](https://arxiv.org/html/2501.14685), VERIFIED; DermaMNIST is image-level, [LEAKY], accuracy only): fine-tuned DINOv2 ViT-B/14 scored 92.02, best of 12 models. EVA-02-B scored 88.79, CLIP-B 89.64, EfficientNet-B4 88.13. **The best ViT encoder LR was about 1e-5, against 1e-4 for CNNs.**
  - **The recipe decides the ranking.** Doerrich et al. ([arXiv 2404.15786](https://arxiv.org/html/2404.15786), VERIFIED, also [LEAKY]) trained every model with one AdamW LR of 1e-4 and did not mention LLRD. Under that recipe, CNNs beat ViTs end to end: DenseNet-121 scored 84.74, EVA-02-B 77.94, CLIP-B 75.31. Your DINOv3-ConvNeXt-T collapse (no LLRD) is the same failure. It is not evidence against DINO-family features.
- **Fit.** About 86M parameters, similar to PanDerm-B. VRAM unmeasured, benchmark needed.
- **Verdict.** This is the best-supported "new objective" member: SSL self-distillation on non-medical data with no labels. It has the only dermoscopy fine-tuning result in which a general foundation model beats ImageNet-supervised, and it is clean under your leakage rule.

### 2b. SigLIP 2 ViT-B/16 (image-text, sigmoid loss), fine-tuned with the FT-CLIP recipe
- **Pretraining data.** WebLI, Apache-2.0 ([card](https://huggingface.co/google/siglip2-base-patch16-224), VERIFIED). Contents are not auditable for ISIC (unestablished).
- **Dermoscopy evidence.** None found; I searched. ImageNet-level evidence only:
  - CLIP-family models make "fewer mistakes relative to their ImageNet accuracy", are overconfident (ECE 0.073–0.085 vs 0.025–0.043 supervised), and are less texture-biased ([arXiv 2311.09215 v3](https://arxiv.org/html/2311.09215v3), VERIFIED).
  - **How to fine-tune an image-text model** (FT-CLIP, [arXiv 2212.06138](https://arxiv.org/html/2212.06138), VERIFIED): CLIP ViT-B/16 went from 82.3 to 85.7 on ImageNet with:
    - a low LR, or 6e-4 with LLRD 0.6;
    - **no MixUp or CutMix** (removing them: 85.3 → 85.7);
    - no drop path;
    - EMA, which helped only at some LRs.
  - **Naively fine-tuned image-text models underperform.** Fine-tuned OpenCLIP was *below* ImageNet-supervised on ISIC 2019 (0.818 vs 0.823, Huix [IMG?]).
- **Why it might matter here.** Image-text contrastive training gives the largest measured error inconsistency against a supervised CNN (Section 1). Its calibration bias (overconfident) is the opposite of your ensemble's under-confidence, which may help the averaged probabilities. That last point is inference, not measured.
- **Fit.** ViT-B/16 vision tower, about 86M (my estimate; the card gives about 0.4B for the whole two-tower model). VRAM unmeasured.

### 2c. ConvNeXt-V2-T (FCMAE)
- **Pretraining.** The available timm checkpoint is FCMAE on IN-1k, **then supervised IN-22k and IN-1k fine-tuning**. 28.6M parameters, 4.5 GMACs, IN-1k top-1 83.89 (85.11 at 384). Weights are **CC-BY-NC-4.0** ([card](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k), VERIFIED).
- **Evidence.** No dermoscopy or medical fine-tuning result found (searched twice). ImageNet only.
- **Verdict.** It is a probable drop-in upgrade of the anchor, not a diversity member: same macro-architecture, and the final stage is supervised on IN-22k. A pure-FCMAE checkpoint would be more diverse but needs a longer fine-tune. Cost and gain are both unmeasured.

### 2d. EfficientNet-B4/B5 noisy-student (instead of EfficientNetV2-S)
- **Pretraining.** IN-1k labels plus pseudo-labels on 300M unlabelled JFT images ([arXiv 1911.04252](https://arxiv.org/pdf/1911.04252v1), SECONDARY).
- **Evidence.** It is the family behind the ADAE / 2020 winner (EfficientNet B3–B7, mostly noisy-student) and Gessert's 2019 ensemble (local `research_isic_winners.md`). No ablation of noisy-student against supervised weights on dermoscopy exists (searched).
- **Verdict.** Its error correlation with ConvNeXt-T should be close to a supervised CNN's. It adds a dataset axis (JFT) but not an objective axis.

### 2e. Not recommended for lack of evidence
- MambaOut, RDNet, CAFormer/ConvFormer, TinyViT, FastViT, Hiera: no dermoscopy evaluation of any of them found (searched). All are ImageNet-supervised or MAE ImageNet models, so they fall in the "architecture" diversity bin your Swin/MaxViT result already covers.
- Perception Encoder B/16 (Apache-2.0, about 94M, [timm](https://huggingface.co/facebook/pe_core_base_patch16_224_timm/blob/main/README.md)): no medical result. It is a reserve image-text candidate if SigLIP 2 fails.
- An unreviewed DINOv3-ConvNeXt-B repo claims 97.74 BACC (backbone only) on HAM with "lesion-grouped" 5-fold CV ([repo](https://github.com/Mohammad-Aman-Ullah-Khan/dinov3-skin-lesion), VERIFIED that the README says this). That is incompatible with every honest HAM number (PLCRD's lesion-disjoint ConvNeXt-T Macro-F1 0.739; challenge best 0.885). **Not credible; not ranked.**

---

## 3. Dermatology and medical foundation models: leakage audit

| Model | Pretraining data vs ISIC/HAM/BCN | Status | Source |
|---|---|---|---|
| **DermFM-Zero / PanDerm-2** (Feb 2026), including **DermFM-Zero-Open** | Model card lists **ISIC (38,404 images), BCN20000 and MSKCC** among its 517k image-text pairs. The Open checkpoint's data list is not published | **EXCLUDED.** Open stays excluded until its data list proves otherwise | [HF card](https://huggingface.co/redlessone/DermFM-Zero), [README](https://github.com/SiyuanYan1/DermFM-Zero/blob/main/README.md), VERIFIED |
| **MedImageInsight** (Microsoft) | **ISIC 2019 in pretraining**: 20,268 training images, random hold-out | **EXCLUDED** | [arXiv 2410.06542](https://arxiv.org/html/2410.06542), VERIFIED |
| DermLIP, DermINO | MSKCC; ISIC 2016–2020 union | EXCLUDED | local `research_derm_sota.md` |
| **MedSigLIP-448** | SigLIP-400M further trained on medical pairs. Derm sources: PAD-UFES-20 plus an **internal set of 51,049 images, 210 conditions**. The card also lists "skin cancer from Australia". ISIC, HAM and BCN are not mentioned | **Unestablished.** HAM has a Queensland source; overlap can be neither shown nor ruled out | [card](https://huggingface.co/google/medsiglip-448), [MedGemma report 2507.05201](https://arxiv.org/html/2507.05201), VERIFIED |
| Derm Foundation | BiT-M R101x3. Web image-text pretraining, then fine-tuning on US/Colombia teledermatology, Australian skin cancer and "additional public images" (unnamed). **Legacy** ("Use MedSigLIP") | Unestablished; low value | [card](https://developers.google.com/health-ai-developer-foundations/derm-foundation/model-card), VERIFIED |
| MONET, BiomedCLIP | PubMed/textbook figures with captions. Reproduced ISIC figures *with diagnoses in the caption* are possible and unquantified | Unestablished; BiomedCLIP is the weakest in every derm benchmark found | local doc |
| **PanDerm-B / -L** | No HAM, BCN or ISIC 2019; ISIC 2024 TBP tiles only | **Clean** | local doc |

### PanDerm-L (ViT-L/16, about 304M)
- **Evidence.** Derm1M HAM linear probe, [IMG?], accuracy: L 0.878 vs B 0.854.
  - **No fine-tuned 7-class number exists** for L or B.
  - Frozen PanDerm-L is fragile: 36.65 weighted F1 on DERM12345 vs B 64.12.
  - The paper's own fine-tuning table uses **layer decay 0.75**, batch 256, 50 epochs, on a single 49 GB GPU ([arXiv v3](https://arxiv.org/html/2410.15038v3) Ext. Data Table 41, VERIFIED). The repo example uses 0.65.
  - The repo mentions no LoRA support ([repo](https://github.com/SiyuanYan1/PanDerm), VERIFIED).
- **Fit.** By arithmetic, full AdamW needs about 4.9 GB before activations, so it would need gradient checkpointing, a small micro-batch and gradient accumulation (`--update_freq` exists), or LoRA. Unmeasured.
- **Verdict.** Run it only if PanDerm-B fine-tuned clearly beats the anchor.

### MedSigLIP-448 with LoRA
- **Evidence.**
  - It is the best frozen model on DERM12345 (69.79 weighted F1, [LG] patient split; peak adapter, possibly chosen on test) ([ar5iv 2601.12382](https://ar5iv.labs.arxiv.org/html/2601.12382), VERIFIED).
  - On a 79-condition derm set, linear probe AUC 0.881 vs Derm Foundation 0.843.
  - **No MedSigLIP derm fine-tuning result exists.** The MedGemma report fine-tunes only the LLMs.
- **Costs.** About 400M vision parameters at 448 px under HAI-DEF terms. It is LoRA-only on 8.55 GB (arithmetic).
- **Risks.** Leakage is unestablished, and it would be the most expensive member.

---

## 4. Hard melanoma vs nevus and pigmented lesions

- **No study compares backbone types on melanoma vs nevus, on young patients, or on dysplastic nevus vs melanoma in situ** (searched twice).
- The closest evidence:
  - **PanDerm, longitudinal early melanoma** (SDDI-Alfred, 10-fold CV): melanomas flagged at the first time point, 77.5% (69/89) vs 32.6% for human reviewers. This uses image *sequences*, not a single image, so it does not carry over to your single-image task. VERIFIED, arXiv v3.
  - **DERM12345, frozen:** MedImageInsight confused dysplastic compound nevi with banal compound nevi in 27.93% of cases. No model was robust at that boundary. VERIFIED.
  - **ISIC 2019, all challenge algorithms** (Combalia, local doc): pigmented melanoma 71% correct vs pigmented nevus 83%.
- **Inference to test, not a finding.** Gontijo-Lopes report that contrastive and CLIP models "specialize in data subdomains". Whether a SigLIP or DINO member ranks young nevus-like melanomas differently from ConvNeXt-T is exactly what the OOF under-40 pAUC must decide.
- **No backbone in the literature has evidence of fixing your failure mode.** The local doc's conclusion that data and context matter more than architecture still stands.

---

## 5. Recommended roster change

1. **Replace EVA02-S with DINOv2 ViT-B/14** (or DINOv3 ViT-B/16), fine-tuned with:
   - LLRD 0.65–0.75;
   - base LR about 1e-5 to 5e-5 (the MedMNIST optimum);
   - drop path 0.1–0.2.
   This removes the second CLIP-teacher MIM model and adds the self-distillation objective.
2. **Replace EfficientNetV2-S with SigLIP 2 B/16**, fine-tuned with the FT-CLIP recipe: LLRD about 0.6, no MixUp or CutMix, EMA. This adds the image-text objective, the largest measured error-diversity axis.
3. **Keep** the anchor ConvNeXt-T 224, PanDerm-B (LLRD, B first), and ConvNeXt-T 384. The 384 run is resolution diversity, the way the winners used it.
4. **Gated extras**, only if the budget allows after steps 1–3: PanDerm-L, then MedSigLIP LoRA.
5. **Pre-registered admission test for each new member**, on OOF folds 1–4 against the anchor:
   - error inconsistency against the anchor must exceed EfficientNetV2-S's;
   - adding the member to the vote must give ΔMacro-F1 > 0;
   - report the under-40 escalation pAUC two-sided.

---

## 6. Ranked table

| # | Candidate | Replaces / adds | Best evidence (number, protocol, URL) | Leakage status | Licence | VRAM feasibility | Main risk | Falsifier |
|---|---|---|---|---|---|---|---|---|
| 1 | **DINOv2 ViT-B/14 FT + LLRD** (DINOv3 ViT-B/16 as alternative) | Replaces EVA02-S | ISIC 2019 FT recall **0.859** vs IN-sup 0.823 [IMG?] ([2310.19522](https://arxiv.org/html/2310.19522)); DermaMNIST FT 92.02, best of 12 [LEAKY] ([2501.14685](https://arxiv.org/html/2501.14685)); DINOv2-L LoRA external AUROC 0.885 (local doc) | Clean: SSL, no labels; web corpus not auditable | DINOv2 Apache-2.0; DINOv3 custom, gated | ViT-B about 86M; unmeasured, benchmark needed | Collapse like your DINOv3-ConvNeXt run if LR is too high; patch-14 at 224 gives 256 tokens | OOF Macro-F1 < anchor − 0.03, **or** error inconsistency against the anchor ≤ EfficientNetV2-S's, **or** vote Δ ≤ 0 |
| 2 | **SigLIP 2 B/16 FT (FT-CLIP recipe)** | Replaces EfficientNetV2-S | No derm result. Diversity: CLIP/ALIGN error inconsistency 0.169–0.230 vs 0.103–0.147 for architecture changes (ImageNet, [2110.12899](https://ar5iv.labs.arxiv.org/html/2110.12899)); FT-CLIP 82.3→85.7, MixUp hurts ([2212.06138](https://arxiv.org/html/2212.06138)); naive OpenCLIP FT 0.818 < 0.823 [IMG?] | Clean under the rule (no derm labels); WebLI not auditable | Apache-2.0 | Vision tower about 86M (my estimate); unmeasured | Weak single-model accuracy on fine-grained dermoscopy; recipe sensitivity | Vote Δ ≤ 0 on OOF, **or** single model < anchor − 0.05 |
| 3 | **PanDerm-B FT** (already planned; keep) | Keep | LP HAM-clean BACC 0.797 vs IN 0.652 [LG]; frozen PanDerm + Swin fusion +1.8 BACC [LG] (local doc; [2505.16338](https://arxiv.org/html/2505.16338v1)) | Clean | CC-BY-NC-ND 4.0 | 86M; unmeasured | Shares a CLIP-latent MIM objective with EVA-02 (why EVA-02 goes) | FT Macro-F1 ≤ anchor **and** vote Δ ≤ 0 |
| 4 | **PanDerm-L FT** (checkpointing or LoRA) | Upgrades PanDerm-B, gated | HAM LP accuracy 0.878 vs B 0.854 [IMG?] (local doc); frozen L fragile (36.65 vs 64.12, [2601.12382](https://ar5iv.labs.arxiv.org/html/2601.12382)) | Clean | CC-BY-NC-ND 4.0 | About 304M; optimiser about 4.9 GB by arithmetic; unmeasured | OOM or long fold times; no FT evidence | Fold-0 Macro-F1 ≤ PanDerm-B |
| 5 | **MedSigLIP-448 LoRA** | Adds, gated | Best frozen on DERM12345, 69.79 wF1 [LG] ([2601.12382](https://ar5iv.labs.arxiv.org/html/2601.12382)); LP AUC 0.881 ([card](https://huggingface.co/google/medsiglip-448)) | **Unestablished** (internal Australian skin-cancer set vs HAM Queensland) | HAI-DEF terms | 400M at 448 px: LoRA only; unmeasured | Leakage cannot be cleared; cost | Any provenance evidence of HAM overlap, or vote Δ ≤ 0 |
| 6 | ConvNeXt-V2-T (FCMAE→IN-22k) | Possible anchor upgrade | ImageNet only: 83.89 top-1 ([card](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k)); no medical evidence | Clean | CC-BY-NC-4.0 | 28.6M, same FLOP class as ConvNeXt-T; unmeasured | Little diversity from the anchor | OOF Macro-F1 not above anchor ConvNeXt-T |
| 7 | EfficientNet-B4/B5 noisy-student | Alternative to EfficientNetV2-S | Winners' family (ADAE, Gessert; local doc); no NS-vs-supervised derm ablation | Clean | Apache-2.0 (timm) | Unmeasured | Same diversity bin as a supervised CNN | Error inconsistency ≈ EfficientNetV2-S |
| — | DermFM-Zero(-Open), MedImageInsight, DermLIP, DermINO | — | — | **Excluded**: ISIC/BCN/MSKCC (DermFM-Zero card), ISIC 2019 (MedImageInsight) | — | — | — | — |
| — | MONET, BiomedCLIP, Derm Foundation | — | Weak or legacy | Unestablished (PubMed captions; unnamed public images) | — | — | — | — |

**Bottom line.** No candidate has dermoscopy evidence of beating ConvNeXt-T under lesion-grouped evaluation. The defensible "game-changer" is structural: replace one CLIP-teacher MIM member (EVA02-S) and one ImageNet-supervised CNN (EfficientNetV2-S) with a DINO-family self-distillation ViT and a SigLIP image-text ViT. Fine-tune both with LLRD and a lower LR. The FT-CLIP recipe for SigLIP also drops MixUp and CutMix. Admit each one only if it raises OOF error inconsistency against the anchor and improves the vote.
