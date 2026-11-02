# V6 transfer-technique research: hard-mimic minority subgroup (melanoma in patients under 40)

**Scope and caveats.** Web research done 2026-11-01. **VERIFIED** means I read the claim in the primary source (the paper's full text on arXiv/ar5iv/PMC/PMLR, or its abstract when only the abstract is cited), through a fetch tool that summarises the page. **SECONDARY** means the claim came from a search snippet, a blog or a review, or the primary text was paywalled or unreadable. The web tools hit a session limit partway through, so a few checks are still open (listed in section 9). I have not entered any number I did not see. Where I give an expected effect it is my judgement, not a measurement, and I say so.

---

## 0. Bottom line

1. **The controlled literature agrees with your own null results.** In large benchmarks, fairness and robustness algorithms (GroupDRO, resampling, adversarial removal, disentanglement) do not significantly beat ERM. Where a debiasing method closes a gap in-distribution, the gain does not carry over to new sites, and removing demographic encoding costs F1 and AP. On ISIC specifically, GEORGE, subclass-GDRO with the true subclass labels, and last-layer retraining (DFR) give no clean gain.
2. **Two levers have reproduced support:** (a) **more real data from the weak subgroup**, and (b) **model selection by worst-group or Pareto criteria, combined with weight averaging (SWAD)**. Your +0.0225 pAUC from about 2,000 real young-patient images matches (a).
3. **Synthetic data helps most when real data is scarce**, and the benefit fades as real data grows. In dermatology, the strongest paper (Ktena, Nat Med 2024) reports that "real data were more essential". With about 81 training positives, memorisation and label fidelity are the binding risks.
4. **pAUC losses (LibAUC):** the gains come only from the authors' own group, against weak cross-entropy baselines. I found no independent replication.
5. **Distilling an ensemble can remove exactly the gain you want.** Ensembling helps the hardest groups most, and distillation is documented to hurt rare subgroups. Any distillation step needs subgroup-aware safeguards and *nested* cross-fitting.

---

## 1. Hard-mimic subgroups and attribute shortcuts: what closed gaps and what did not

| # | Claim | Number | Protocol | Source | Status | Replicated? |
|---|---|---|---|---|---|---|
| 1.1 | No bias-mitigation method significantly beats ERM | "no method outperforms ERM significantly for any metric", both in- and out-of-distribution | MEDFAIR: 11 algorithms (Resampling, DomainInd, LAFTR, CFair, LNL, EnD, ODR, GroupDRO, SWAD, SAM, ERM) × 10 datasets, **including HAM10000 and Fitzpatrick17k**; Friedman + Nemenyi tests, p<0.05 | [ar5iv 2210.01725](https://ar5iv.labs.arxiv.org/html/2210.01725) | VERIFIED | Yes (see 1.2, 1.3) |
| 1.1b | SWAD ranks first for worst-case and overall AUC, but is not significantly better than ERM; it raises every subgroup and may *widen* the gap | Ranks only | same | same | VERIFIED | — |
| 1.1c | **Model selection matters more than the algorithm** | For ERM, Pareto selection has an average rank of about 1.5 on worst-case AUC vs about 2.5 for overall-AUC selection (significant), with no significant loss of overall AUC | same | same | VERIFIED | Consistent with 1.6 and 1.7 |
| 1.2 | Methods that target worst-group performance do not beat simple data balancing; group-fairness methods reach parity by making every group worse | 9 methods, chest X-ray | Zhang et al., CHIL 2022 | [PMLR v174](https://proceedings.mlr.press/v174/zhang22a.html) | VERIFIED (abstract) | Yes (1.6) |
| 1.3 | Debiasing closes in-distribution gaps but does not transfer to new sites; models with *less* demographic encoding generalise fairness better; DANN gives the lowest OOD gap but worse F1, AP and ECE gaps | 3,456 models (ERM, ReSample, GroupDRO, DANN, CDANN, MA). Encoding vs gap: R=0.82 (No Finding, age). "Minimum attribute prediction accuracy" selection: P=9.6e-94. **ISIC age disparities "small and statistically insignificant"**, ISIC gaps <2% | Yang et al., Nat Med 2024 | [PMC11485237](https://pmc.ncbi.nlm.nih.gov/articles/PMC11485237/) | VERIFIED | Partly (1.1) |
| 1.4 | **In dermatology, age encoding is not the cause of the age-related performance gap** | Correlation of age encoding with unfairness ρ=−0.177, p=0.0746 (not significant); the authors conclude the cause "does not appear to be shortcut learning" | ShorT: frozen encoder, multitask variation of age encoding, Nat Commun 2023 | [PMC10354021](https://pmc.ncbi.nlm.nih.gov/articles/PMC10354021/) | VERIFIED | — |
| 1.5 | On ISIC, inferring hidden subclasses (GEORGE) and even GDRO with the *true* subclass labels give nothing over ERM | AUROC on the non-patch subclass: ERM .922, GEORGE .912, subclass-GDRO .923 | Sohoni et al., NeurIPS 2020 | [ar5iv 2011.12945](https://ar5iv.labs.arxiv.org/html/2011.12945) | VERIFIED | — |
| 1.6 | Last-layer retraining (DFR) on ISIC: the worst group improves but another group collapses | Malignant without patch 64.38→85.84%; benign without patch **94.29→77.72%**; average 90.12→87.43% | Le et al. 2023, a reproduction of DFR | [arXiv 2308.00473](https://arxiv.org/html/2308.00473v2) | VERIFIED | Independent reproduction |
| 1.7 | Simple class/group balancing matches state-of-the-art worst-group methods; group labels matter most **for model selection** | 4 benchmarks | Idrissi et al., CLeaR 2022 | [PMLR v177](https://proceedings.mlr.press/v177/idrissi22a.html) | SECONDARY | Yes (1.2) |
| 1.8 | Algorithms help only some kinds of subgroup shift; worst-class-accuracy selection works without group labels | 20 algorithms, 12 datasets, >10k models | "Change is Hard", ICML 2023 | [arXiv 2302.12254](https://arxiv.org/abs/2302.12254) | VERIFIED (abstract) | — |
| 1.9 | Fine-tuning on a diverse curated set closed the light-vs-dark skin-tone gap | DDI, 656 images; dermatology models lost 27–36% ROC-AUC vs their original tests (numbers by skin type not retrieved) | Daneshjou et al., Sci Adv 2022 | [PubMed 35960806](https://pubmed.ncbi.nlm.nih.gov/35960806/) | SECONDARY (publisher returned 403) | Consistent with your +2k-image result |
| 1.10 | Reinforcing multi-task learning (sex as an auxiliary target) does not remove sex bias; adversarial removal helps only in female-only training | ResNet, ISIC-type data, controlled sex ratios | Raumanns et al., FAIMI 2024 | [TU/e PDF](https://pure.tue.nl/ws/files/378074039/978-3-031-72787-0_2.pdf) | VERIFIED | — |
| 1.11 | Hidden stratification: performance differences of more than 20% on clinically important subsets | ML4H 2019 | Oakden-Rayner et al. | [arXiv 1909.12475](https://arxiv.org/abs/1909.12475) | VERIFIED (abstract) | — |
| 1.12 | A chest X-ray *foundation model* was biased and underperformed a reference model | 127,118 CXRs; pleural effusion dropped 10.7–11.6% for Black patients | Glocker et al., Radiol AI 2023 | [arXiv 2209.02965](https://www.doi.org/10.48550/arXiv.2209.02965) | SECONDARY | — |

**How this maps onto your results.** Your three nulls (Group-DRO-style reweighting, adversarial removal of age with AUC −0.10, and last-layer retraining) are what 1.1–1.6 predict. Items 1.3 and 1.4 add a mechanism: in dermatology, an age-linked gap is mostly *different presentation plus low prevalence*, not a shortcut. Removing age information therefore takes away real signal and fixes nothing. That fits your "representation/ranking limit" conclusion. The only interventions with reproduced support are **data** and **selection**.

---

## 2. Synthetic data for under-represented subgroups

| # | Claim | Number | Protocol | Source | Status |
|---|---|---|---|---|---|
| 2.1 | Diffusion augmentation improves high-risk sensitivity and fairness under shift | Dermatology, label+attribute conditioning: high-risk sensitivity **+27.3% in-distribution, +63.5% OOD** (vs JFT-pretrained baseline); OOD fairness gap **7.5× smaller** (absolute 0.044). Age was among the attributes, with gains "improved on or maintained" | Cascaded DDPM 64→256 px, conditioned on label (± sex/age/skin tone), trained on labelled in-distribution images plus *unlabelled* in-distribution and OOD images; downstream data **75% synthetic / 25% real**; TPUs; 16,530 teledermatology cases (clinical photos, **not dermoscopy**) | [PMC11031395](https://pmc.ncbi.nlm.nih.gov/articles/PMC11031395/) | VERIFIED |
| 2.1b | Negatives from the same paper | In dermatology "real data were more essential"; in the less-skewed setting generated data gave "a minor decline"; only **50.0 ± 12.6%** of generated images were judged diagnostic-quality; memorisation of rare conditions listed as open | same | same | VERIFIED |
| 2.2 | Gains are confined to data-poor regimes and smaller than gains from real images | EfficientNetV2-M: up to **+13.2 pts** at 32 real images per class; at **228 per class, nothing significant** (inpainting +2.0, CI −1.2 to 5.3; text-to-image **−2.2**); gains saturate above 10:1 synthetic:real; 458,920 SD images | Sagers et al. 2023, Fitzpatrick17k-9 and DDI | [ar5iv 2308.12453](https://ar5iv.labs.arxiv.org/html/2308.12453) | VERIFIED |
| 2.3 | ISIC 2024: the 1st-place team added Stable Diffusion images | No gain reported | Kaggle write-up summary (Japanese blog) | [zenn.dev](https://zenn.dev/ottantachinque/articles/2024-09-12_isic2024-top-solution) | SECONDARY |
| 2.4 | Synthetic malignant lesions raised pAUC on SLICE-3D | EVA02 0.1516→0.1633; EdgeNeXt 0.1401→0.1576 (pAUC above 80% TPR) | Post-competition paper; protocol and seeds not visible | [arXiv 2506.03420](https://arxiv.org/abs/2506.03420) | SECONDARY (numbers from the search snippet only) |
| 2.5 | Dermoscopy diffusion (DiDGen, MICCAI 2025): "modest" gains on a single dataset (ISIC 2018); reviewers flagged no dermatologist validation and possible leakage | p-values only | Review page | [MICCAI 2025](https://papers.miccai.org/miccai-2025/0230-Paper4243.html) | VERIFIED (review page) |
| 2.6 | Diffusion models memorise medical training images more than GANs do, especially on small datasets | Brain MRI and CXR | Akbar et al. | [arXiv 2305.07644](https://arxiv.org/pdf/2305.07644) | SECONDARY |
| 2.7 | Compute: diffusers SD LoRA runs "in as low as 11 GB" without 8-bit tricks; about 5 h on a 2080 Ti; LoRA file about 3 MB | HF blog | [huggingface.co/blog/lora](https://huggingface.co/blog/lora) | VERIFIED |
| 2.8 | DreamBooth with gradient checkpointing and 8-bit Adam: 9.92 GB on a T4 | README of a fork | [GitHub](https://github.com/ShivamShrirao/diffusers/blob/main/examples/dreambooth) | SECONDARY |

**Feasibility on 8.55 GB.** I found no *verified* figure for SD-1.5 LoRA fitting in 8 GB. Fitting needs gradient checkpointing, 8-bit Adam, fp16 and 512 px at batch 1, and has to be **measured**. The basic problem is that a generator conditioned on "melanoma, under 40" learns from about 81 positives. That is the regime where 2.6 predicts memorisation and 2.2 predicts that real data dominates. Ktena got around this by training the generator on large *unlabelled* pools from all domains.

---

## 3. Semi-supervised learning and self-training

| # | Claim | Number | Protocol | Source | Status |
|---|---|---|---|---|---|
| 3.1 | With realistic tuning, semi-supervised and self-supervised gains are inconsistent | Median / worst gain in balanced-accuracy points over the best labelled-only baseline: **MixMatch +4.7 / −0.2**; FixMatch +1.6 / **−4.5**; DINO +1.6 / −3.8; BYOL +3.5 / −6.0. Labelled sets were **400–1,660** images, unlabelled 89k–354k | Huang et al., CVPR 2024, 13 methods × 4 medical datasets | [ar5iv 2307.08919](https://ar5iv.labs.arxiv.org/html/2307.08919) | VERIFIED |
| 3.2 | FixMatch gave no substantial gain on CheXpert; helped on retinal images only with imbalance corrections | Matched tuning budget | Zenk et al., BVM 2022 | [Springer](https://www.springerprofessional.de/en/realistic-evaluation-of-fixmatch-on-imbalanced-medical-image-cla/20283342) | SECONDARY |
| 3.3 | Noisy Student: 88.4% ImageNet top-1, ImageNet-A 61.0→83.7% | 300M unlabelled JFT images, about 230× the labelled set | Xie et al., CVPR 2020 | [arXiv 1911.04252](https://arxiv.org/abs/1911.04252v4) | SECONDARY |
| 3.4 | Self-supervised pretraining on unlabelled medical images (plus ImageNet SSL) | **+6.7% top-1** on dermatology (27 classes, clinical photos); MICLe vs SimCLR +1.18 ± 0.09% | Azizi et al., ICCV 2021 | [arXiv 2101.05224](https://export.arxiv.org/abs/2101.05224v2) | SECONDARY |
| 3.5 | PanDerm: dermatology foundation model, ViT-L/16 and ViT-B/16 weights, **CC-BY-NC-ND**, >2M images, 28 benchmarks | No HAM10000 numbers on the README | [GitHub](https://github.com/SiyuanYan1/PanDerm) | VERIFIED (README); benchmark numbers not retrieved |

**What is missing.** I found no controlled medical study at your scale (about 15k labelled images, 10–30× unlabelled). Every rigorous benchmark is low-label. A specific risk for you: a teacher that misses young melanomas will pseudo-label them as nevi. Self-training would then reinforce the very error you are trying to fix.

---

## 4. Partial-AUC and ranking losses

| # | Claim | Number | Protocol | Source | Status |
|---|---|---|---|---|---|
| 4.1 | SOPA beats CE on one-way pAUC for **Melanoma (Kaggle/ISIC 2020, 1.76% positive)** | FPR≤0.3: **CE 0.7651 (0.0135) → SOPA 0.8093 (0.0248)**; AUC-M 0.7788; SOPA-s 0.7775. FPR≤0.5: 0.8151 → 0.8585 | ResNet18; **CE pretraining, then head re-initialised and all layers fine-tuned** per method; "multiple" seeds and splits | [ar5iv 2203.00176](https://ar5iv.labs.arxiv.org/html/2203.00176) | VERIFIED |
| 4.2 | LibAUC benchmark | CheXpert pAUC(FPR<0.3): CE 0.769±0.011, AUC-M 0.820±0.003, SOPA-s 0.823; AUROC CE 0.853 vs 0.890/0.894 | End-to-end, 3 seeds; the CE AUROC of 0.853 is weak for CheXpert | [ar5iv 2306.03065](https://ar5iv.labs.arxiv.org/html/2306.03065) | VERIFIED |
| 4.3 | AUC-M placed 1st on CheXpert; 33/3,314 on Kaggle melanoma | — | ICCV 2021 | [arXiv 2012.03173](https://arxiv.org/pdf/2012.03173) | SECONDARY |

**Caveat.** All of the evidence comes from the LibAUC group. I found **no independent replication** against a strong, pretrained, tuned CE baseline. Minimax group-AUC objectives exist ([AAAI 2023](https://arxiv.org/pdf/2208.10451)), but I found no medical results for them.

---

## 5. Training-recipe levers

| # | Lever | Number | Source | Status |
|---|---|---|---|---|
| 5.1 | SWAD | First rank on worst-case and overall AUC in MEDFAIR, not significant (1.1b); about +1.6 pts over the previous DG best on DomainBed | [MEDFAIR](https://ar5iv.labs.arxiv.org/html/2210.01725), [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html) | VERIFIED / SECONDARY |
| 5.2 | Model soups in dermoscopy (7 fine-tuning resolutions averaged) | Better generalisation and calibration on external data, holdout unchanged; **numbers paywalled** | [Maron et al., EJC 2022](https://cris.fau.de/publications/281402901) | SECONDARY |
| 5.3 | Greedy soup, ViT-G | 90.78 → 90.94% (+0.16) | [Wortsman, ICML 2022](https://proceedings.mlr.press/v162/wortsman22a.html) | SECONDARY |
| 5.4 | LP-FT | About +1 pt ID and +10 pts OOD over FT (10 non-medical benchmarks) | [Kumar, ICLR 2022](https://arxiv.org/pdf/2202.10054) | SECONDARY |
| 5.5 | LP-FT on ISIC 2020 | Mixed: ResNet-50 AUROC 73.54→82.03; DenseNet −0.3; VGG **−5.0**. 3 seeds, no tests. The full-FT baselines (AUROC 0.71–0.83) look under-tuned | [Davila et al. 2024](https://arxiv.org/html/2406.10050v1) | VERIFIED (low quality) |
| 5.6 | Augmentation in dermoscopy | Best of 13 scenarios: AUC 0.882 on ISIC 2017 melanoma; "augmentation can lead to more performance gains than obtaining new images" | [Perez et al. 2018](https://arxiv.org/abs/1809.01442) | VERIFIED (abstract) |
| 5.7 | RandAugment in dermatology | Helped ID sensitivity, not OOD, and **harmed OOD fairness** (in Ktena) | [PMC11031395](https://pmc.ncbi.nlm.nih.gov/articles/PMC11031395/) | VERIFIED |
| 5.8 | Multiple real images per lesion at inference | AUROC 0.905 → **0.930**; ECE 0.131→0.072 (significant); 656 lesions | [arXiv 2306.02800](https://arxiv.org/abs/2306.02800) | VERIFIED (abstract) |

I found no controlled medical evidence with numbers for layer-wise LR decay, label smoothing, progressive resizing or FixRes. Mixup/CutMix papers on ISIC report small gains without matched seeds (not cited).

---

## 6. Distilling an ensemble into one model, with a cross-fitted teacher

| # | Claim | Number | Source | Status |
|---|---|---|---|---|
| 6.1 | "Function matching": the teacher and student see the **same** augmented view, inputs are mixed with mixup, and training is very long. A **fixed (precomputed) teacher overfits** | ImageNet R50: fixed teacher peaks at 78.11% (1,200 ep) and falls to 76.95% at 9,600 ep; function matching reaches **82.31%** at 9,600 ep (82.8% best). Pets: student 91.06 vs teacher 91.03 | [Beyer et al., CVPR 2022](https://ar5iv.labs.arxiv.org/html/2106.05237) | VERIFIED |
| 6.2 | **Cross-fitting** (teacher predictions out-of-fold) plus loss correction counteracts teacher overfitting | Up to **+4 AUC pts** (tabular, overfit teacher); up to +1.5% on CIFAR-10 when the teacher overfits | [Dao et al., ICLR 2021](https://ar5iv.labs.arxiv.org/html/2104.09732) | VERIFIED |
| 6.3 | **Distillation harms rare subgroups**: teacher errors are amplified | About −2% absolute on the rarest classes; worst-10 ImageNet classes about −3%; fix = soften teacher weight where it is unreliable | [Lukasik et al., TMLR 2022](https://arxiv.org/pdf/2106.10494) | SECONDARY |
| 6.4 | Ensembles help the **bottom-k groups most**, and keep improving up to about 50 members while average accuracy plateaus at 4–5 | [Ko et al. 2023](https://ar5iv.labs.arxiv.org/html/2303.00586) | SECONDARY |
| 6.5 | Students rarely match the teacher closely, even with enough capacity | [Stanton et al., NeurIPS 2021](https://arxiv.org/pdf/2106.05945) | SECONDARY |

**Taken together:** 6.3 and 6.4 say the subgroup gain from your ensemble is the part most likely to be lost in distillation.

**A leakage point for your protocol.** Inside a lesion-grouped 5-fold evaluation, the teacher for outer fold *k* must be trained **without fold *k***. The usual 5-fold OOF teacher has seen fold *k*'s labels, and that information reaches the student, which breaks Hard Rule 2. Correct cross-fitting is nested: about 4 inner teachers per outer fold. The alternative is one sealed test split with plain OOF teachers.

---

## 7. Multi-task and auxiliary targets

| # | Claim | Number | Source | Status |
|---|---|---|---|---|
| 7.1 | Training on **757 fine-grained taxonomy classes** and summing their probabilities into coarse classes beats direct training on 3 or 9 classes | Partitioned model: 72.1 ± 0.9% (3-way) and 55.4 ± 1.7% (9-way) over 9 folds, 127,463 images. Direct-training values are in an image table I could not extract | [Esteva et al., Nature 2017](https://www.Gwern.net/doc/ai/nn/2017-esteva.pdf) | VERIFIED (partitioned numbers only) |
| 7.2 | Theory: coarse labels teach common features, **fine-grained labels also teach rare features**, which matters most on hard samples | [Hong et al., TMLR 2024](https://arxiv.org/pdf/2410.23129) | SECONDARY |
| 7.3 | SIIM-ISIC 2020 1st place used a 9-class diagnosis target with no ablation; 4-class vs 9-class B5 CV 0.9765 vs 0.9771 (private LB reversed) | [ar5iv 2010.05351](https://ar5iv.labs.arxiv.org/html/2010.05351) | VERIFIED |
| 7.4 | Auxiliary demographic targets: no effect on bias (1.10); age heads are used as a *diagnostic* of bias (MELBA 2026) | [arXiv 2606.03214](https://arxiv.org/abs/2606.03214) | VERIFIED (abstract) |
| 7.5 | ISIC-DX consensus lexicon: 362 terms, 3 super-categories, 41 differentiation categories, giving a label hierarchy you could use | [JEADV 2024](https://experts.arizona.edu/en/publications/international-skin-imaging-collaboration-designated-diagnoses-isi/) | SECONDARY |

Your gain from IN-22k pretraining is consistent with 7.1–7.2: finer pretraining labels taught features that 1k-class pretraining did not.

---

## 8. Ranked transfers for V6

Cost anchor: your stated **≈41 min per 30-epoch 224 px run**, so a 5-fold CV run is about **3.4 h** (extrapolated from that figure). Effects are **my judgement, not measured**. Your only measured positive anchor is **+0.0225** under-40 pAUC from about 2,000 real images.

| Rank | Transfer | (i) Under-40 pAUC | (ii) Macro-F1 | Cost on one 8.55 GB GPU | Main risk | Falsifier |
|---|---|---|---|---|---|---|
| 1 | **Subgroup-aware checkpoint selection plus weight averaging** (SWAD or EMA; greedy soup across seeds or resolutions). Select on the OOF *Pareto* of {Macro-F1, under-40 pAUC}, never on the evaluation set (1.1c, 1.7, 1.8, 5.1–5.3) | Small + (the most reproduced lever) | Neutral to + | ≈0 extra VRAM (EMA copy ≈0.1 GB, extrapolated); one CV run ≈3.4 h | With about 65 subgroup lesions, selection overfits the selection set | OOF-selected checkpoint shows no gain on the untouched evaluation folds; or the averaged model's subgroup pAUC is within the CI of the last-epoch ERM model |
| 2 | **Dermatology-pretrained initialisation** (PanDerm-Base ViT-B/16, or continued SSL on an unlabelled dermoscopy pool), fine-tuned with LP-FT to protect the features (3.4, 3.5, 5.4) | Moderate +, uncertain. This targets the limit you identified (representation) | + or − | Unmeasured. Run `scripts/gpu_benchmark.py` for ViT-B at 224 px first. Expect about 1–2× ConvNeXt-T cost (guess) | **Leakage**: PanDerm's pretraining may include ISIC/HAM images (not verified). CC-BY-NC-ND licence. Foundation-model bias (1.12) | Under-40 pAUC no better than the IN-22k ConvNeXt at matched compute; or pretraining-set overlap cannot be ruled out |
| 3 | **Fine-grained or hierarchical diagnosis head** (melanoma in situ vs invasive; nevus subtypes such as dysplastic, Spitz/Reed, blue, congenital) trained jointly with the 7-class head (7.1–7.3) | Small to moderate +, *if labels exist* | Neutral to + | One CV run ≈3.4 h | Fine labels are missing for HAM10000 (7 classes only), so the head is trained on a biased subset | No change in under-40 pAUC vs the same run without the auxiliary head (≥3 seeds) |
| 4 | **Two-head pAUC fine-tune**: keep the CE 7-class head; add an escalation head trained with SOPA-s / AUC-M on your CE checkpoint. Age is used **only to sample** pairs (under-40 escalating vs under-40 benign), never as an input (4.1–4.2) | Uncertain; directly optimises the metric | Protected by the CE head | About 1/3 run per fold plus a margin/temperature grid ≈ 5–8 h (extrapolated) | Single-group evidence; similar in spirit to your failed hard-case reweighting; pAUC heads are poorly calibrated | Under-40 pAUC(FPR≤0.2) within noise of CE over ≥3 seeds; or overall Macro-F1 falls |
| 5 | **Synthetic young-melanoma images**: SD-1.5 LoRA, label+age conditioning, generator trained on *all* images (Ktena's unlabelled trick), nearest-neighbour memorisation filter, dermatologist spot-check (2.1–2.8) | Small; expect **below +0.0225** (2.2: synthetic < real) | Neutral to slightly − | LoRA training ≈5 h at 11 GB per HF; 8.55 GB **unverified**, measure first. Generation time unmeasured. Plus one CV run | Memorisation from about 81 positives; mislabelled synthetic "melanomas" right on the nevus boundary | No gain over a matched control (same number of extra *augmented real* positives); or near-duplicates of training positives |
| 6 | **Nested cross-fitted ensemble distillation** with consistent views and within-fold mixup; teacher weight lowered where the teacher is wrong on the subgroup (6.1–6.3) | Retains *part* of the ensemble gain at best | Likely + vs a single model | High: about 2× step cost for online teachers plus long schedules ≈ 6 h per outer fold (guess). Cheaper variant: cache about 8 views (no mixup), ≈ one normal run | 6.3 and 6.4: the subgroup gain is what gets lost; leakage if not nested | Student's under-40 pAUC ≤ best single model; or subgroup fidelity is much worse than overall fidelity |
| 7 | **Lesion-level multi-image aggregation** at inference, if the protocol allows a lesion as the unit (5.8) | Small +, only for lesions with more than one image | Small + | ≈0 | Changes the unit of evaluation; must be pre-registered | No change among multi-image under-40 lesions |
| 8 | **Semi-supervised (MixMatch or Noisy Student)** on an unlabelled dermoscopy pool (3.1–3.3) | Uncertain to − | Small + possible | Heavy: unlabelled-to-labelled ratio μ≈7 multiplies data per step (guess 5–8×) | Pseudo-labels repeat the under-40 miss; evidence only in low-label regimes | Under-40 pAUC ≤ supervised baseline; pseudo-label recall on held-out young melanomas below the teacher's |

**Do not retry (the literature predicts a null):** GroupDRO, GEORGE, DFR, adversarial or invariance methods (1.1–1.6), and auxiliary age prediction (1.10, 1.4).

---

## 9. Open items (not verified)

- PanDerm's pretraining-data overlap with HAM10000/ISIC/BCN, and its HAM benchmark numbers. The fetch was blocked by the session limit.
- Esteva's direct-training accuracies, Maron's soup numbers (paywalled), and Daneshjou's per-skin-type AUROCs (403).
- Seyyed-Kalantari underdiagnosis rates by age ([PMC8674135](https://pmc.ncbi.nlm.nih.gov/articles/PMC8674135/)): the page returned a CAPTCHA, which I did not attempt.
- ISIC 2024 synthetic-data gains (2.3, 2.4), which are secondary.
- A measured VRAM figure for SD-1.5 LoRA on 8 GB.
