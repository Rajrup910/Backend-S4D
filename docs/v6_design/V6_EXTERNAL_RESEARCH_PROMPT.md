# V6 external research prompt (paste everything below the line into ChatGPT, Antigravity or another model)

Written 2026-11-02. It is self-contained: the receiving model sees none of the repository. Attach `docs/V6_RUNSHEET.md`
(Part 0 is the current plan) if the tool accepts files. Bring the answer back as a file, and check every claim before
anything changes in the plan (the 3 Oct ADAE proposal had 3 wrong or unverifiable numbers out of 12).

---

## Role
You are an expert machine-learning researcher in medical image classification, reviewing the plan for the next phase
("V6") of a dermoscopy skin-lesion research project. Your job is to **find what would make this plan better, cheaper or
more novel, and what would make it fail**, using published evidence. Be adversarial: a confirmation of the plan is
worth little; a well-sourced reason to change it is worth a lot.

## The project in brief
- **Task:** 7-class dermoscopic lesion classification (MEL, NV, BCC, AKIEC = AK + SCC, BKL, DF, VASC); an 8-class split
  (SCC separate) exists. **Image-only:** no age, sex, site or other metadata may be a model input (a fixed policy;
  age is used for evaluation only).
- **Data:** the ISIC 2019 training set (HAM10000 + BCN20000 + MSKCC), de-duplicated (perceptual hash) into a 15,294-image
  development pool with **lesion-grouped 5-fold CV**. Fold 0 is the selection fold; folds 1–4 are confirmatory. A
  4,733-image BCN/MSKCC reserved partition and the HAM test rows are already used up (read several times; never again).
  Plus 2,078 extra young-patient (≤ 35 y) histopathology-confirmed ISIC Archive images in training only.
- **Hardware:** one laptop GPU, **RTX 5050, 8.55 GB VRAM**, 15 GB RAM, Windows. Measured: ConvNeXt-Tiny at 224 px,
  batch 32, 30 epochs ≈ 41–50 min per fold run (≈ 59 min with the young extras); 384 px ≈ 64–75 min; 384 px fine-tune
  peaks 6.27 GB. **Any cost you quote for another model is a guess unless you cite a measurement on similar hardware.**
- **Rules:** lesion-grouped splits always; the test set is read once; nothing is selected on accuracy (67% of images are
  nevi); every weight, threshold and temperature is fitted on out-of-fold or validation data only. Primary metrics:
  Macro-F1, balanced accuracy, escalation sensitivity, and escalation partial AUC at FPR ≤ 0.20 ("pAUC").

## What has been measured (V1 → V5; numbers are from the project's result files)
**Current level** (pooled folds 1–4, 12,235 images): Macro-F1 ≈ 0.676, balanced accuracy ≈ 0.666, all-age escalation
pAUC ≈ 0.82, **under-40 escalation pAUC ≈ 0.77**. On the HAM10000 test set alone, the V1 HAM-only system reached
Macro-F1 0.805, but the same system scored **0.411 on BCN/MSKCC** images it had not seen.

**Worked:**
- pooled multi-archive training (0.411 → 0.603 Macro-F1 off-distribution);
- ensembling (6-CNN soft vote +0.026 Macro-F1 on HAM, McNemar p 3.4e-05; a 3-seed vote +0.020 on fold 0);
- 24-view TTA (+0.014);
- Dirichlet calibration (fixes calibration, but lowered escalation sensitivity 0.786 → 0.731 on HAM);
- IN-22k pretraining (+0.008 Macro-F1 on folds 1–4, not significant);
- **young histopathology data plus two escalation heads: under-40 pAUC +0.027 [+0.004, +0.049]**, the only under-40
  gain that held on the confirmatory folds (target was +0.050).

**Failed or falsified:**
- frozen probes of PanDerm and DINOv2 (worse under 40);
- unfrozen DINOv3-ConvNeXt-T without layer-wise LR decay (collapsed to 0.38–0.44 Macro-F1);
- LDAM-DRW and ASL losses (below class-weighted CE);
- a single 384 px model (null on pooled data: HAM rows gain, BCN rows do not);
- ten hand-built dermatology modules (chromophore stem, border geometry, lesion zoom, prototypes, any-region clues, GeM,
  concept logic and others);
- adversarial age removal (under-40 AUC −0.10);
- age-specific decision rules (rejected on cost);
- frozen-feature specialist heads and last-layer retraining;
- Caruana ensemble selection on a 1.5k validation set (overfit);
- frequency-domain and persistent-homology features (they decode the archive, not the lesion).

**The core failure:** 33 of 81 under-40 escalating lesions are missed at FPR 0.20. All 33 are called nevus, and 19 are
in the most pigmented tercile. Escalating prevalence is 3–12% under 40 vs 31–75% at 60+, by archive. The project has
closed this as a **representation/ranking limit**, not a head, loss, threshold or weighting problem.

**Already in place:** class-weighted cross-entropy (effective-number weights) with label smoothing 0.05; S56 selective
abstention as the deployed safety layer.

## The current V6 plan (Part 0, 2 Nov 2026)
**Headline:** a leak-audited, data-maximal ensemble across **pretraining families**, image-only, selected for the
under-40 band by **nested leave-one-fold-out selection**, with the full measured stack (uniform soft vote + 24-view TTA,
always applied and reported with and without + Dirichlet after combining + S56 abstention).
- **Members**, each trained in-domain with layer-wise LR decay where the architecture needs it:
  1. ConvNeXt-T IN-22k at 224 px (the anchor);
  2. **PanDerm-B** (dermatology self-supervised ViT; its pretraining reportedly contains no HAM10000, BCN20000 or ISIC
     2019 images; never fine-tuned here);
  3. EVA02-S (masked-image modelling);
  4. EfficientNetV2-S;
  5. ConvNeXt-T IN-22k at 384 px with 384-pretrained weights.
- **Data added for training only**, all deduplicated at lesion and patient level: the unused ISIC 2019 Test set (BCN
  part), ISIC 2020 rows with a specific diagnosis, DERM12345, and MILK10k after a separate external read. **A lesion-link
  audit found 985 MILK10k lesions that are the same lesions as ISIC 2019 images** under new image ids.
- **Stages:** recipe screen (EMA + photometric augmentation + offline shades-of-grey colour constancy; 8-class head) →
  expanded-pool screen at **matched optimiser steps** → member screen on fold 0 → ≤ 4 finalists × folds 1–4 →
  ensemble, TTA and calibration → one external read on **HIBA** (1,270 images, Argentina, contact-polarised).
- **Endpoints (fixed):** (a) co-primary vs the V5 system on folds 1–4: all-age pAUC above the noise floor **and**
  Macro-F1 Δ > 0 with CI lower bound > 0; (b) under-40 pAUC Δ ≥ +0.050 with CI > 0; (c) secondary: does the domain
  self-supervised member (PanDerm) help the under-40 band more than all ages?
- **Budget:** ≈ 35–55 GPU h (extrapolated), hard cap 80 h.
- **Dropped:** GroupDRO / DFR / adversarial invariance, metadata, learned stackers, synthetic lesions, pAUC-DRO as a
  headline, single models above 384 px, public ISIC-trained checkpoints (trained with labels on the evaluation data).

## What we need from you
Answer each item with **sources (URL, venue, year)**. Tag every claim **VERIFIED** (you read it in the primary source)
or **UNVERIFIED** (memory, snippet or summary). **Never invent a number**; write "not found" instead.

1. **Members.** Is this the best 4–5-model roster for an 8.55 GB GPU and ~50 GPU h? Name better candidates with
   *published dermoscopy results under lesion- or patient-grouped evaluation*, with params, VRAM/time evidence,
   licence, and whether their pretraining used ISIC 2016–2020, HAM10000 or BCN20000 images or labels (that is leakage
   for us).
2. **PanDerm specifically.** The best published fine-tuning recipe for PanDerm-B/L (or PanDerm-2 / DermFM-Zero) on a
   multi-class dermoscopy task, any fine-tuned (not linear-probe) multi-class results, and any independent replication.
3. **The under-40 hard mimics** (pigmented, nevus-like melanomas in young patients). Any method, in any field, with
   *controlled* evidence of closing a gap for a minority subgroup whose positives resemble the majority negative class,
   **without** using the subgroup attribute as input, that is not on our failed list above. Include negative
   replications.
4. **Cross-archive shift (HAM → BCN).** Methods with measured gains on BCN20000 or other held-out dermoscopy sites:
   colour constancy, photometric/style augmentation, field-of-view cropping, SWAD/EMA, test-time adaptation. Numbers and
   protocols.
5. **Ensembling.** Evidence on member diversity by pretraining family vs architecture, ensemble size vs gain,
   multi-resolution members, and whether ensembles help or hurt subgroup gaps (Ko et al. 2023; Schweighofer et al.
   2025, or newer).
6. **Data.** Any dermoscopy dataset released 2024–2026 with histopathology-confirmed melanoma in patients under 40 that
   is **not** in the ISIC Archive, with access route and licence; and any public external cohort that could confirm an
   under-40 result.
7. **Risks.** The three most likely ways this plan fails or produces a misleading result (leakage, selection
   optimism, statistical power with ~65 under-40 escalating lesions, distribution shift), and the cheapest check for
   each.
8. **Novelty.** Is any of the plan's novelty already published? Specifically: age-stratified evaluation of an ensemble
   on lesion-grouped ISIC data; domain-SSL members helping a hard subgroup; nested subgroup-aware ensemble selection;
   cross-dataset lesion-link audits between public dermoscopy datasets.

## Output format
1. A one-paragraph verdict: what you would change first, and why.
2. One section per item 1–8: claims tagged VERIFIED/UNVERIFIED, each with its source.
3. A ranked table: change → expected effect on (i) Macro-F1/balanced accuracy and (ii) under-40 pAUC → GPU cost on 8.55
   GB → evidence strength → the result that would show it did not work.
4. A list of everything you could not verify.
