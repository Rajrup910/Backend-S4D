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
**Revision 1 Oct — CG-DM becomes the V6 headline (owner decision, pre-freeze, before any V6
result).** V6 is reorganised around one representation hypothesis, **Counterexample-Guided
Differential Morphology (§A0)**, with two additions: **R1 signature-anchored counterexamples**
(§A0.12, upgrades N13) and **R2 pseudo-benign twin** (§A0.13, conditional). The young-data curve,
N13 and DINOv3-LLRD stay independent parallel branches. The six-architecture programme (§A3b,
V6-2b/3b), N10, plain RML (now an ablation) and broad ensembling are **demoted**. Status:
**finalised, NOT frozen** — V5's results can still change §A0/§A5; the hash is taken at V6-0
under §A0.14.
**Revision 1 Oct, evening — the in-domain ensemble is restored (owner request, pre-freeze, before
any V6 result).** A full-history review (V1 → V5, CHANGELOG + `results/`) found that the
1 Oct demotion used the wrong endpoint for the ensemble: S64's +0.0037 is an **under-40 AUC** on
HAM OOF (G1), while ensembling's measured value is **Macro-F1** (G2/G3: Phase 1 McNemar
p = 3.4e-05; +0.026 HAM test). A multi-architecture ensemble **trained in domain on the pooled
corpus has never been tested** (S64: "an ensemble of the six V4 models is untested; it is S67
stage 0"; S67 stage 0 never ran, `results/v4/final_verdict_v4.json`). So: **V6-3b and the V6-10
E-primary are back on the core path** as a parallel G2/G3 branch, with plain in-domain members;
**V6-2b (portability of the V5 heads) stays demoted**. V1's **24-view TTA** (+0.014 Macro-F1,
rung A6) is carried into V6-10 conditionally on V5's Q9 TTA read. Under-40 stays descriptive for
the ensemble — ensembling is not claimed as a G1 lever. Budget: + V6-3b, 25 runs ≈ 17 h at the
ConvNeXt-T 41-min anchor (**extrapolated; the five other architectures are unmeasured until §B3**).
**Same evening, second pass (owner request):** the E-primary gate is **co-primary** — all-age
pAUC **and** Macro-F1 must both pass (owner decision); a full V2–V4 review adds §A9: eight
carried items (C1–C8: bipartite conformal + FRR, label-free budget lock, per-band calibration arm,
eval-transform parity, drift hooks, the S48 endpoint read as declared, TTA, the ensemble) and a
13-row standing-issues table with an honest expectation per issue.
**Revision 2 Oct (owner-approved, pre-freeze, before any V6 result):** Stage 0 becomes a
head-to-head — partial CG-DM vs a nearest-patch (PatchCore-style) and a sparse benign-dictionary
residual, built identically — with a fair combiner baseline, an archive-stratified gate condition,
Bonferroni-interval selection and routing to the carried statistic (§A0.8); log-domain fp32 Sinkhorn
and one top-10% operator (§A0.5); a void-run rule (§A0.9); HIBA's primary = the mechanism endpoint
(§A2); fp16-castable engineered statistics (§B0). External proposals checked and not adopted, two
with CPU probes (`results/v6/{nrfp,topology}_probe.json`): §A0.8 "considered". A same-night full
audit corrected stale cross-references, a calibration misreport, budget anchors and a citation
(CHANGELOG, 2 Oct).
**Revision 3 Oct (owner request, pre-freeze, before any V6 result) — §A10 added.** Headline
benchmark (ISIC 2019 winner's lesion-grouped CV ensemble, BA 71.7), the same-footing table, a
V5-derived pool profile, and the accumulation principle, recorded as **input to the combined
brainstorm after V5 is final and after Review 2**. No other section changes until then; §A10.5
lists what that brainstorm must settle (including whether CG-DM stays the headline).
**Same evening — §A11 added (owner request):** ensemble candidate roster across pretraining families
(Tier 1 open-licence backbones, Tier 2 larger, Tier 3 dermatology foundation models), technique stack
with evidence and repo priors, and a pre-declared OOF-only selection protocol. Brainstorm input; costs
unmeasured until `scripts/gpu_benchmark.py --timm` runs.
**Same evening — §A12 added (owner request, "pay heavy attention"):** the ADAE angle (SIIM-ISIC 2020
winner; Heinlein et al. 2024 under-35 BA 0.890) verified claim by claim; public ADAE weights **excluded**
(trained with labels on our whole corpus); the recipe adopted leak-free as an in-domain teacher ensemble
(+ ISIC-2020, EfficientNet B3–B5 / SE-ResNeXt-101 / ResNeSt-101 at 384–456 px, image-only) distilled into
one student with fold-matched teachers. Priority candidate for the brainstorm; nothing frozen.
**Revision 4 Oct — §A13 added (owner request):** resolution and module-fairness study (RMF). The V5 224-vs-384 null
was measurement + dilution (HAM rows gain, BCN rows do not), not a bug; A13 tests resolution properly (384-pretrained
weights, 5-fold OOF, per-archive readout, EMA) and re-tests the best V5 modules under fair training. Brainstorm input.
**Revision 2 Nov — Part 0 redesign (owner request, pre-freeze, before any V6 result).** A review of this file on
1 Nov (CHANGELOG) scored it 6/10: rigour high, but four competing headlines (§A0, §A10, §A12, §A13), 88–147+ GPU h
and the lowest-prior lever as the headline. Part 0 below was rebuilt from the V1–V5 record and four literature
reviews (`docs/v6_design/`), and **supersedes §A0 as the headline and Part B's core path**. Nothing in Parts A/B is
deleted; Part 0.9 lists what each older section becomes.

**This is the only V6 file.** Part 0 is the current plan. Part A is the earlier plan and the reference detail (goals, data, models, techniques, decision table,
audit). Part B is the earlier execution (ordered, gated phases, costs, stop rules); where it conflicts with Part 0, Part 0 wins.

---

# Part 0 — V6 redesign (2 Nov 2026): data, pretraining diversity, nested subgroup-aware selection

## 0.1 What the evidence says (sources in `docs/v6_design/`)
**Our own record** (`docs/v6_design/internal_evidence.md`, every number with its `results/` file):
- What moved numbers: pooled training (reserved Macro-F1 0.411 → 0.603), young histopathology data plus V5 heads
  (under-40 pAUC +0.027 [+0.004, +0.049] over the same IN-22k trunk, Q11), ensembling (HAM 6-CNN +0.026 Macro-F1,
  McNemar p 3.4e-05; IN-22k 3-seed vote on fold 0 0.662 → 0.682), TTA (+0.014).
- What did not: ten V5 image-feature modules; DRO-style reweighting; adversarial age removal (−0.105); decision-layer
  age rules; frozen-feature heads and probes; loss swaps; a single 384 px model; Caruana selection on a small val set.
- The failure: 33 of 81 under-40 escalating lesions are missed at FPR 0.20, all called nevus, 19 of 33 in the high-melanin
  tercile (`results/v5/diagnostics/b1_hard_core.json`). V5 closed it as a representation limit.

**External record** (four reviews; VERIFIED = read in the primary source):
- *ISIC winners 2016–2024* (`research_isic_winners.md`). Only three ingredients recur with measured support: more
  data (the largest factor in the only factorial ablation, Valle et al.), diverse multi-architecture × multi-resolution
  ensembles with **plain** averaging (Gessert CV 68.8 best single → 71.7 average), and TTA. Single-model resolution is
  flat; learned stackers, metadata and synthetic lesions are null or negative on held-out data. Class-weighted CE (+0.038
  BACC, Gessert 2018) is **already on here** (`research/v5/train_v5.py:694`). **No winner reports any under-40 result.**
- *Dermoscopy beyond competitions* (`research_derm_sota.md`). PanDerm's pretraining corpus contains **no HAM10000,
  BCN20000 or ISIC 2019** (VERIFIED, Nat Med text; its only ISIC source is ISIC-2024 TBP tiles) and it was never
  fine-tuned here (S51 was a frozen probe). Licence CC-BY-NC-ND: research use, no release of fine-tuned weights.
  DermLIP/Derm1M contains ISIC MSKCC and DermINO contains the ISIC 2016–2020 union → excluded. Photometric augmentation
  +0.047 OOD AUROC (lesion-grouped) and SWAD +7.4 on a 7-archive skin DG benchmark. Our single-model BACC 0.64–0.67 sits
  at Gessert's honest lesion-grouped single-model level (65.8–68.8).
- *Other fields* (`research_transfer_techniques.md`). Controlled benchmarks (MEDFAIR; Yang et al., Nat Med 2024)
  find no fairness/robustness algorithm significantly beats ERM; in dermatology an age gap is not shortcut learning
  (ShorT, Nat Commun 2023), which matches our V3/V4 nulls. What does help: **more real subgroup data** and
  **subgroup-aware model selection with weight averaging**. Ensembles help hard groups most; distillation is
  documented to hurt rare subgroups. pAUC losses lack an independent replication; synthetic data is null at ≥ 228
  real images per class.
- *Data* (`research_data_sources.md`, ISIC API metadata pulled 1 Nov). The public young histopathology pool is
  nearly used (5,790 total; our extras took 2,078). Large **unused, non-young** pools exist: ISIC 2019 **Test**
  (8,238; never used by the corpus — `research/v4/build_corpus.py` reads the Training files only), ISIC 2020
  (33,126; 0 recorded image/lesion/patient links to ISIC 2019), DERM12345 (12,345, CC-BY, no age field).

**Leakage found while checking (verified on 2 Nov in this session against `ml/data/manifest_v4.csv`):**
- **616 MILK10k lesions share an ISIC `lesion_id` with development training rows**, and **2 more with the V5 young
  extras** (618). Units, reconciled 2 Nov (`results/v5/s84_lesion_links.json` joined to the manifest through
  `results/v5/isic2019_archive_lesion_ids.csv`): **985** MILK10k lesions link to anything (union of 859 ISIC 2019
  Training, 126 ISIC 2019 Test, 2 extras). The 859 split by manifest role as **616 train + 113 ham_val + 130 ham_test**;
  the 126 Test-linked lesions are in no manifest row. D6 excludes all 985. They are new photographs, so S84's image-level filter
  (ISIC id or dHash/pHash, `research/v5/s84_milk10k.py`) does not remove them. **Consequence for V5 S84:** exclude by
  `lesion_id` before the read, declared as a deviation. Consequence for V6: every source is deduplicated at **lesion and
  patient** level, never by image alone.
- The V5 young extras share **0** images and **0** archive lesions with any manifest split (train, val, reserved,
  ham_val, ham_test): no fold leakage in V5. 530 of them are ISIC 2019 Test (BCN) images.

## 0.2 Headline and claims
**V6 = a leak-audited, data-maximal ensemble across pretraining families, image-only: the headline is general performance
and one deployable system (G2/G3). Under-40 is read at its pre-declared gate and reported as the measured boundary of public
data and model diversity; V6 does not claim to fix it** (owner decision 3 Nov, before any V6 result; the reasons are in 0.8c).
A new under-40 hypothesis belongs to V7. The optional nested selection guards the under-40 band against regression (0.5); it
does not optimise for it. One headline, three claims, each tied to a pre-declared endpoint:
1. **G2/G3 (performance):** an ensemble of in-domain members from different pretraining families (supervised IN-22k
   CNN at two resolutions, domain self-supervised dermatology ViT, self-distillation ViT, image–text contrastive ViT), trained on the expanded pool,
   beats the V5 system on folds 1–4. The endpoint is the owner's E-primary, **unchanged**: co-primary all-age pAUC above
   the noise floor **and** Macro-F1 Δ > 0 with CI lower bound > 0.
   **Anchor recipe (declared 2 Nov audit):** the anchor is the V5 `youngdata` recipe: IN-22k ConvNeXt-T + young
   extras, **without** the V5 heads (`twostep`, `m4`). Reasons:
   - On fold 0, the composite vs `youngdata` alone was all-age pAUC +0.0003, under-40 +0.0077 and Macro-F1 −0.0127
     (`results/v5/screens/gate_composite_in22k_vs_youngdata.json`).
   - Both heads' gains were readout-only on escalation mass.
   - The heads are ConvNeXt-specific, so no other member can carry them.

   This is also why V6-1's control is the banked `youngdata` fold-0 runs.
2. **G1 (under-40):** the same comparison on under-40 escalation pAUC, **Gate A unchanged** (MCID +0.050, CI > 0).
   The bar is V5's, but the comparator is now the **V5 system**, which already sits +0.0351 above the in1k control
   that V5's Gate A was read against. So this is a stricter test than V5's. For continuity, V6 − in1k control is
   reported beside it (descriptive, the same rows and bootstrap as `results/v5/confirm_s42_s43_s44.json`).
   Expectation, stated now: unlikely to pass. V5's best was +0.035, and no lever here targets the hard mimics directly.
   A miss is the expected result, and is reported as the measured boundary of data + diversity, not as a failure of V6's
   headline. A pass would be reported as such, with its CI.
3. **Mechanism (secondary, its own Holm family):** does adding the domain-SSL member (PanDerm) to the supervised
   ensemble help the under-40 band more than all ages? This is the novel question: no ISIC winner reports age strata,
   and none of the PanDerm results the review found does.

**Comparability read (descriptive, not a gate):** 8-class balanced accuracy on the 5-fold OOF beside Gessert's 71.7.
The protocol differences are stated (last epoch vs best epoch, our deduplicated 15,294-row partition, their 2,334
external images and 16 configurations up to 528 px).

**Novelty, stated honestly:** (a) an age-stratified evaluation of a winner-style ensemble on lesion-grouped pooled
ISIC data — none found in the reviewed record, but "first" is not claimed without a systematic search; (b) the
pretraining-family question for a hard-mimic subgroup (claim 3); (c) nested cross-fitted ensemble selection with an
under-40 non-inferiority constraint — a known procedure applied here, which removes the optimism of selecting on the
reporting folds (Gessert 72.5 vs 71.7; our Caruana overfit);
(d) the cross-dataset lesion-link audit (0.1). None of these is a new architecture. CG-DM stays a CPU diagnostic (0.9).

## 0.3 Data plan (allocation declared before any download; every source lesion- and patient-deduplicated)
| Source | Role | Rule |
|---|---|---|
| S71 development pool (15,294) + V5 young extras (2,078) | Training; **the only evaluation rows** (S71 folds) | Unchanged from V5 |
| ISIC 2019 Test, BCN part, minus the 530 already in the extras | Training (adds BCN, the hardest archive: LOAO BA 0.391) | **Verified 2 Nov:** 0 of its lesion ids appear in any manifest split (train, val, reserved, ham_val, ham_test; `results/v5/isic2019_archive_lesion_ids.csv`). Diagnoses are public for **7,082 of 8,238** images; the 1,156 without a diagnosis are excluded. **Lesion-level only:** the archive has `patient_id` for 0 of 8,238 images and `lesion_id` for 8,238 (ISIC facets API, 2 Nov), so patient-level deduplication is impossible for it (the same holds for BCN rows already in the manifest) |
| ISIC 2020, rows with a specific diagnosis | Training (benign diversity; **1 new young melanoma**: 73 of its 74 are already V5 extras) | Patient-grouped; "unknown" benign rows excluded from the 7/8-class head |
| DERM12345 | Training | Subclass → class map declared at V6-0; no age, so it never enters an age readout |
| MILK10k, after V5's S84 read | Training | The 985 lesion-linked lesions excluded (D6; 618 of them link to development training rows, see 0.1) |
| HIBA | **Confirmation**, single receipted read | All-age; under-40 descriptive (20 lesions) |
| SCP2, MRA-MIDAS | Optional external young-band read | Data on request; never on the critical path |

**Added rows are training-only.** Any added row linked by lesion id, patient id or hash to any manifest row or to **HIBA**
(the confirmation cohort, added by the 3 Nov audit) is dropped. Against the V5 young extras, which are also training-only,
only exact image or hash copies are dropped: that is double counting, not leakage, and a patient-level drop would discard
148 labelled ISIC 2020 rows, 81 of them young (`results/v6/data_allocation.json`).
**Limit, stated:** only 417 of 25,331 manifest rows have a patient id (HAM 0, BCN 0, MSKCC 417), so "0 shared patients" with
the evaluation rows is close to empty. A patient with different lesions in an added source and an evaluation fold cannot be
ruled out, most plausibly in ISIC 2020's BCN and MSKCC images. Lesion ids (23,664 of 25,331) and image hashes are the real check. A **subclass → class map is
declared at V6-0 for every added source**, not only DERM12345: ISIC 2019 Test holds classes the head does not have (Scar
314, Melanoma metastasis 633, `results/v6/archive_meta/isic2019_test.csv`), and each is mapped or excluded before any
training. MILK10k has no patient ids either, so it is lesion-deduplicated, patient unknown. Added images are stored as a
resized cache at download: nothing on disk yet except MILK10k, C: has 38.9 GB free, and full-size images already cost
the young extras about 1.68× per step (`results/v5/queue_Q4.txt`). Evaluation stays on the S71 folds, so every V6 number is paired with V5's on identical rows.
`data_allocation.json` reports, per source, separate counts for: patient-id matches, lesion-id matches, exact/perceptual
hash matches, and rows with **no patient id** or **no lesion id**. A row is labelled patient-deduplicated only if it has
a patient id; otherwise "lesion-deduplicated, patient unknown".

## 0.4 Stages, gates and stop rules
**Fixed for every run** (adds to §B0): the optimiser-step budget of the V5 `youngdata` run (30 epochs × its row count).
When the pool grows, epochs shrink to hold steps constant, so a data gain is never a compute gain. **The frozen head stage
scales with them** (3 of 30 epochs today, `HEAD_EPOCHS = 3` in `research/v4/train_v4.py:91`): it stays one tenth of the step
budget, otherwise at about 10 epochs it would eat 30% of the steps and V6-2 would not be a matched comparison. `--num-workers 2`,
`_last` primary, test lock armed, ≥ 5 GB free on C:.

| Stage | Work | Runs / GPU h | Gate → consequence |
|---|---|---|---|
| **V6-0** (mostly CPU) | V5 finished and §A5 filled; the S84 lesion-level fix declared first. `scripts/run_v6_benchmarks.ps1` plus PanDerm-B. Engineering in `research/v6/`: a timm/PanDerm member builder with layer-wise LR decay, epoch-level weight averaging for every run. This reuses `SwadAverager` (`research/v5/m7.py`), unlocked from LOAO-only. It averages one snapshot per epoch, so it is SWA, **not** iteration-dense SWAD. Its window is the last two-thirds of the epochs, so it scales when the step budget shortens epochs. BatchNorm members (only EfficientNetV2-S, if it enters as SigLIP 2's fallback) get a running-statistics re-estimation pass on training images, because averaging BN statistics as if they were weights is wrong, a photometric-augmentation option, an offline shades-of-grey image cache (removes its measured 18.8 ms/image cost), 8-class head with 7-class read-out by summing AK + SCC. Lesion/patient/hash audit of every source → `data_allocation.json`. IN-22k noise-floor seeds 45/46/47 on the **anchor recipe** (`youngdata`), plus the §A0.9 control seeds for the recipe comparison (`results/v5/queue_V6N0.txt`, 0.8c). **Gate A power statement** (CPU): for true effects of +0.035 and +0.050, from V5's hierarchical bootstrap SD on the same 65 lesions, three probabilities reported separately: the point estimate reaches +0.050; the CI lower bound is > 0; both (Gate A itself); recorded before any V6 run. **PanDerm checkpoint pin:** the original PanDerm-B weights, SHA-256 recorded; the repo's newer "PanDerm-2" (DermFM-Zero) is excluded, because its training data include BCN20000, MSKCC and HIBA (model card, checked 2 Nov). **Hash Part 0.** | 6 runs + benchmarks ≈ 5.1–5.6 h (benchmarks ≈ 5–10 min for 25 configurations, from two measured at 12 s each; `youngdata` 58.5–58.7 min and control 40.5–49.9 min, both measured). Tag `v6nf`, so V5's `v5scr` record stays closed | Any member whose benchmarked run exceeds 2.5× the anchor is dropped before screening |
| **V6-1** Recipe | Fold 0, seeds 42/43/44, ConvNeXt-T IN-22k on the V5 pool. Arm **R** = weight averaging + photometric augmentation + shades-of-grey (the averaged `_swad` checkpoint is the arm's read-out; `_last` reported beside it, so the averaging's own contribution is visible); arm **T** = 8-class head. Control = the banked V5 `youngdata` fold-0 runs (same pool, trunk, recipe; no retraining) | 6 runs ≈ 6 h (58.6 min measured for `youngdata`) | V5 screen gate, IN-22k bars. R passes → carried. T is adopted if Macro-F1 retention ≥ the bar (non-inferiority: it exists for comparability, not gain) |
| **V6-2** Pool | Fold 0, 3 seeds, matched steps, carried recipe. Arm **P** = expanded pool (0.3), plain sampling. Arm **Y** = the same pool with a sampler that holds the expected under-40 share of each batch at the V5 pool's own fold share (27.6% on the whole development pool, against 13.1% if the pool is left to dilute; `results/v6/data_allocation.json`, `young_share`). Rows with no age count as not young. Both are compared with the same recipe on the V5 pool. Y adds no data and changes no loss weight; it only prevents dilution | 6 runs ≈ 6 h (extrapolated from 58.6 min) | Gate as V6-1 against the V5 pool, plus BCN-row Macro-F1 Δ reported. **Y is carried only if** it passes that gate **and** beats P on under-40 pAUC by more than `bar_k3` on `pauc_u40`, with Macro-F1 no worse than P's by more than the retention bar. Otherwise P if it passes; otherwise V6 continues on the V5 pool |
| **V6-3** Members | Fold 0, seed 42, carried recipe and pool: **PanDerm-B** (LP-FT; layer decay 0.65 from its repo), **DINOv2 ViT-B/14** (`vit_base_patch14_dinov2.lvd142m`, Apache-2.0; self-distillation; LP-FT, layer decay, backbone LR ≈ 1e-5, positional embedding interpolated to 224 px; DINOv2 ViT-S/14 if B exceeds the 2.5× cost rule), **SigLIP 2 B/16** (`vit_base_patch16_siglip_224.v2_webli`, Apache-2.0; image–text contrastive; FT-CLIP recipe: layer decay ≈ 0.6, no Mixup/CutMix; **EfficientNetV2-S** is its fallback if SigLIP 2 fails the cost rule or diverges). Members are chosen by **training objective**, not architecture: error inconsistency with a supervised ResNet-50 rises from 0.085 (another seed) and 0.103–0.147 (another architecture) to 0.169–0.230 (another objective and data, CLIP/ALIGN) (Gontijo-Lopes et al., ICLR 2022), and our own architecture-only additions (SwinV2-T, MaxViT-T) were a wash. EVA02-S is dropped: like PanDerm it is masked-image modelling against CLIP-feature targets, the same objective family twice. The DINO-family risk is the LR: frozen DINOv2 lost under 40 (S51) and DINOv3-ConvNeXt without layer decay collapsed, so the recipe above is the test, not the backbone. Augmentation and step budget are the V6 recipe's for every member. Also **ConvNeXt-T IN-22k at 384 px** with its 384-pretrained weights. The 384 px member has the best in-repo record of any candidate: V4 S53r +0.0297 Macro-F1 in 3/3 seeds, and V5 S01 HAM rows gained in 3/3 seeds. Its known costs are read on its gate: 1.5× run time (measured), the BCN-row null, and S01's under-40 −0.0192 as a single model. MaxViT-T and SwinV2-T are **not planned**: on HAM val they trailed the anchor (0.7176 and 0.7412 vs 0.7482, `results/review2/v1_member_metrics.csv`), and added to the 6-CNN vote they were a wash (0.7981 vs 0.7986) | 4 runs, cost from V6-0 benchmarks (unmeasured until then) | Promote ≤ 3: single-model all-age pAUC within the anchor's noise floor (`bar_k1`, one seed: z80 × the pair SD of the six anchor seeds) **or** fold-0 soft-vote all-age pAUC above it with Macro-F1 retained. Fold 0 is the selection fold. **Report only, never a gate** (CPU): each candidate's marginal contribution when added to the anchor, on all-age Macro-F1, all-age pAUC, under-40 pAUC and the under-40 high-melanin tercile, plus disagreement, double-fault and **error inconsistency** (the share of rows only one of the two gets right) with the anchor, overall and on the under-40 escalating misses. Fold-0 under-40 counts are small, so these are descriptive |
| **V6-4** Finalists | Anchor + ≤ 3 members × folds 1–4, seed 42 (fold 0 exists from V6-1/2/3) | ≤ 16 runs; ≈ 16–35 h (anchor 58.6 min measured; members from benchmarks) | OOF complete for every member before any ensemble is formed |
| **V6-5** Ensemble (CPU + TTA inference) | **E-primary = uniform soft vote** of the finalists vs the V5 15-model system **with its own 24-view TTA** (`results/v5/preds_tta/`, Macro-F1 0.7005 on folds 1–4), folds 1–4 OOF, so TTA is never credited to V6 alone. **24-view TTA always applied** to every member before combining (owner decision 2 Nov; supersedes the §A5 TTA row's condition). The E-primary system is reported **with and without TTA** on the same rows, each against V5 in the same state (V5 without TTA: 0.6911), so TTA's contribution is measured, not assumed. Each member's views are built at **its own input size** (the 384 px member gets 384 px views). The repo's `build_tta_views(224)` call sites are parameterised in V6-0, and the 384 member's TTA costs ≈ 2.9× per image (pixel ratio; unmeasured). Dirichlet fitted after combining (cross-fitted), S56 refit, C1 conformal as an output field, C3 per-band calibration arm, C5 drift baseline (§A9.1). **Secondary:** nested subgroup-aware selection (0.5). Claim 3 contrast | TTA ≈ 2.8–4.2 h (extrapolated: 8.49 min per 224 px model on one fold, `results/v5/benchmarks/tta_benchmark.json`; the 384 px member's ≈ 2.9× is unmeasured) | E-primary and Gate A as in 0.2; claim 3 under its own Holm family |
| **V6-6** Confirmation | Single receipted HIBA read: V6 vs V5 vs V1 systems with S56 at matched referral (C2 label-free budget lock, C6 S48 endpoint). Optional SCP2/MIDAS pooled young-band read, Clopper–Pearson | ≤ 1 h | One read, enforced by receipt |

**Core total, 3 Nov: ≈ 42–48 GPU h (extrapolated), inside the 80 h cap; re-quoted after V6-0's full benchmarks.** Inputs: the anchor's measured 58.6 min (≈ 262 ms/batch end to end), the 384 member's measured 1.5×, and two step-time benchmarks measured 3 Nov on synthetic batches: the PanDerm-B proxy 224 ms/batch (3.45 GB VRAM), so data-bound at ≈ 1× the anchor, and DINOv2-B 302 ms/batch (4.41 GB), ≈ 1.3×. SigLIP 2 is unmeasured and assumed 1–1.3×. Real runs of each member confirm these.
- The earlier "≈ 35–55 GPU h" was set when EVA02-S and EfficientNetV2-S (≈ 20M parameters each) were members.
- The roster now holds three ViT-B members (PanDerm-B, DINOv2-B, SigLIP 2, ≈ 86M each) and the 384 px member, so the
  total is **likely above 55 h**. That is unmeasured, so no figure is quoted.
- The anchor's 58.6 min (measured) and the 384 member's 1.5× (measured) are the only anchors.
- PanDerm-B is not a timm model: its cost is benchmarked through the architecture-identical `vit_base_patch16_224` as a
  compute proxy (declared), then confirmed on its first real run. **Hard cap 80 h:** if a re-quote exceeds it, the cut order is
SigLIP 2 (weakest dermoscopy evidence), then the 384 px member, then DINOv2-B → DINOv2-S, then V6-4 drops to folds 1–2 for all members (declared now).
**Every stage ends reportable:** a failure at V6-1, V6-2 or V6-3 still leaves a complete, gated V6 on the V5 recipe,
pool or the anchor alone.

## 0.5 Nested subgroup-aware selection (secondary; CPU; never the primary)
For each outer fold k ∈ {1, 2, 3, 4}: run bagged forward selection with replacement (20 bags) over the finalists on
the OOF rows of the **other** folds (fold 0 included). The objective is Macro-F1, subject to under-40 pAUC not falling
below the uniform vote's by more than the noise floor. Apply the selected weights to fold k. Pool the four held-out
reads. The estimate is therefore a held-out estimate of the selection **procedure**, never of a subset chosen on its own
evaluation rows. It replaces uniform only if it beats it on the pooled nested read by more than its lesion-bootstrap SD.
**What it is:** a Macro-F1 optimiser with an under-40 **non-inferiority guard**. It cannot produce an under-40 gain by
design, and it is never reported as subgroup optimisation.

## 0.6 Not repeated, with the reason (V1–V5 nulls agree with the controlled literature)
GroupDRO / subclass-DRO, DFR and last-layer retraining, adversarial or invariance training, auxiliary age heads (V3/V4
nulls; MEDFAIR; ShorT); learned stackers and small-set Caruana (rung-6 lesson; Valle); metadata (registry; null on
every winner's held-out data); single models above 384 px (2020 winner's flat B4 curve; S01); synthetic lesions
(Sagers 2023 null at 228 per class; ISIC 2024 winner 0.140 vs 0.142); pAUC-DRO as a headline (no independent
replication; m4 already tests a ranking head); public ISIC-trained checkpoints (trained with labels on our evaluation
data), including DermFM-Zero / PanDerm-2 (trained on ISIC, BCN20000, MSKCC and the HIBA confirmation cohort).

## 0.7 Honest expectations (recorded before any V6 result)
| Endpoint | Expectation | Why |
|---|---|---|
| E-primary Macro-F1 | Plausible pass | Diversity and data are the only levers with repeated support; our 3-seed vote alone gave +0.020 on fold 0 |
| E-primary all-age pAUC | Uncertain | V5's heads moved all-age pAUC only on their own readout |
| Gate A under-40 | Unlikely | Best measured +0.035; the misses are hard mimics no lever here targets directly; 15 new young melanoma lesions (0.8c) |
| V6-2 arm Y (young-share sampler) | Unlikely to be carried | Hard-case reweighting moved under-40 by −0.000 (S67). It is a safeguard against dilution, not a fix |
| Claim 3 (PanDerm, under-40) | Open | No prior data; the frozen probe lost, a fine-tune was never run |
| Best pooled Macro-F1 of the programme (Review 3) | Plausible | The full stack (ensemble + TTA + Dirichlet + data) is the first system to combine every measured lever on the pooled ruler. It is compared with V5 on identical rows, never with V1's HAM-test 0.805. **Corrected 3 Nov:** ≈ 0.676 was a single seed; the V5 *system* scores **0.6911, and 0.7005 with TTA** (`results/v5/v5_oof_common_matrix.json`), so the bar is higher and V5 already holds the seed-vote and TTA gains, which is a different ruler. Escalation sensitivity is reported beside it, because Dirichlet cost 0.786 → 0.731 on HAM |
| 8-class BA vs 71.7 | Likely short | 4–5 members at 224/384 px vs 16 configurations up to 528 px plus external data |

## 0.8 Freeze checklist (replaces §A0.14 for Part 0)
(1) V5 final and §A5 filled; (2) the S84 lesion-level exclusion declared and applied; (3) `data_allocation.json` with
lesion/patient/hash audits for every source, missing-identifier rows counted separately (0.3); the PanDerm-B
checkpoint hash recorded; (4) benchmarks recorded and the 80 h cap re-checked; (5) the IN-22k bars
recomputed from six seeds; (6) the member builder's eval-transform parity test passes (C4). **Geometry** (resize,
crop, interpolation) is the repo's for every member. **Normalisation** is each member's own pretraining config (ImageNet
mean/std for ConvNeXt, PanDerm and DINOv2; 0.5/0.5 for SigLIP 2), asserted per member, because forcing ImageNet
statistics on SigLIP would be the S58-type transform defect. (7) `research/stats/families.py` declares the V6 families:
E-primary (an intersection-union test), Gate A, claim 3 (its own Holm family), and the nested-selection secondary.
(8) The Gate A power statement is recorded (V6-0). (9) The member builder, the weight averager (BN re-estimation,
scaled window) and the TTA image-size parameter have CPU unit tests. (10) Every approximate recipe value in V6-3 (DINOv2 backbone LR "≈ 1e-5", SigLIP 2 layer decay "≈ 0.6") is replaced by one exact
value from its cited source, with no tuning on S71 folds. (11) The owner approves, then **Part 0's text** and the V6 code
(`research/v6/` and the runner) are hashed into `results/v6/v6_plan_freeze.json`. Hashing only Part 0 means status lines
elsewhere can change without breaking the hash. After the hash, any change is a dated amendment. **No V6-1 or later run before
the hash.** The V6-0 benchmarks and noise-floor seeds run before it because they are inputs to it and test no V6 hypothesis;
the rule turning seeds into bars was committed before any of them was read (0.8c).

## 0.8b Freeze-readiness audit (2 Nov, before any V6 run; Part 0 NOT frozen)
**Verdict: the design is complete and internally consistent; the freeze is blocked only by work, not by open design
questions.**

| # | Checklist item | Status | What unblocks it |
|---|---|---|---|
| 1 | V5 final, §A5 filled | **Blocked** | Q9 TTA → OOF table → Dirichlet and S56 refits → S84 (the V5 run order); then §A5 is filled from V5's final files |
| 2 | S84 lesion-level exclusion | **Ready** | D6 declared; code + tests pass (38); `--dry-run` loads the links file |
| 3 | `data_allocation.json` | Partly done | Done 2 Nov: ISIC 2019 Test lesion-disjoint from every manifest split; 7,082/8,238 labelled; HIBA (ISIC collections 175/176/251) shares 0 images and 0 lesions with the extras and the manifest. **To do:** ISIC 2020 and DERM12345 audits, the MILK10k post-S84 row set, the PanDerm-B hash |
| 4 | Benchmarks, 80 h cap | **Blocked** | `scripts/run_v6_benchmarks.ps1` (DINOv2-B added; SigLIP 2 B and the 384 member present; PanDerm via the ViT-B/16 proxy) |
| 5 | IN-22k six-seed bars | **Blocked** | Seeds 45/46/47 at fold 0 (≈ 2.0–2.5 h, measured anchor) |
| 6 | C4 parity (geometry + per-member normalisation) | **Blocked** | The member builder in `research/v6/` |
| 7 | V6 statistical families | **Blocked** | `research/stats/families.py` entries |
| 8 | Gate A power statement | **Blocked** | CPU script on V5's bootstrap (minutes) |
| 9 | Unit tests (builder, averager, TTA size) | **Blocked** | With item 6 |
| 10 | Hash | Waits for 1–9 | — |

**Found and fixed by this audit (2 Nov):**
- **The anchor's V5 heads were undeclared.** The anchor is now stated as the `youngdata` recipe without heads, with
  the fold-0 evidence (0.2). Before this, V6-1's control and the anchor could silently differ from the V5 system.
- **Gate A's comparator changed silently** from the in1k control to the V5 system. It is now stated as stricter, with
  the in1k contrast reported beside it for continuity (0.2).
- **C4 parity conflicted with members' own normalisation** (SigLIP uses 0.5/0.5). Geometry stays fixed and
  normalisation is per member (0.8).
- **The 384 px member's TTA would have run at 224 px.** TTA views are now built at each member's input size (V6-5).
- **The "35–55 GPU h" total was stale** after the ViT-B swaps. It is withdrawn until the benchmarks run, and PanDerm
  gets a declared timm compute proxy (0.4).
- **Missing freeze items added:** the statistical families, the power statement and the unit tests (0.8 items 7–9).
- **PanDerm's NC-ND licence** now constrains deployment to internal research use (0.9).

**Checked, not a problem:** HIBA vs extras and manifest (0 overlap); ISIC 2019 Test vs every split (0 lesions); the
V6-1 control's banked predictions exist (`results/v5/preds/youngdata_f0_s4{2,3,4}_in22k_v5scr.csv`); the V5 S84
preflight (`--dry-run`) is missing only the refit outputs the V5 run order produces.

## 0.8c Delivery audit (3 Nov, before any V6 run of record; Part 0 NOT frozen)
**Question: does the plan as written deliver what V6 is for?** Each claim was traced to its endpoint, data and code. This
supersedes the status column of 0.8b.

**Verdict by goal:**
- **G2/G3 (Macro-F1, a deployable system): yes, testable and fair** once the fixes below are in. A pass is plausible, not
  assured: V5 already banks the seed-vote and TTA gains, so V6 must win on data and objective diversity alone.
- **G1 (under-40): the plan measures it fairly but does not attack it.** Gate A's probability is **0.106 at a true +0.035
  and 0.50 at a true +0.050** over the V5 system (`results/v6/gate_a_power.json`; `research/v6/gate_a_power.py`). A true
  +0.050 means about +0.085 over the in1k control, against V5's measured +0.035. **If an under-40 gain is the goal, this plan
  will probably report a well-measured miss.** No technique change here alters that; it needs young melanoma data (SCP2,
  MRA-MIDAS, both on request) or a new hypothesis, by amendment.
- **The expanded pool barely adds the data G1 would need.** New under-40 melanoma lesions beyond V5's pool: ISIC 2020 **1**,
  ISIC 2019 Test **0**, MILK10k **14** after D6, DERM12345 unknown (no age field). That is **15**, while the pool grows from
  14,313 to about 43,000 rows, nearly all of them older or age-unknown (`results/v6/data_allocation.json`,
  `net_training_contribution`, before class maps). V6-2 is therefore a test of general data, not of young data, and it
  dilutes the young share. Its under-40 pAUC must be read with that in mind.

**Found and fixed:**
1. **The noise floor used the wrong recipe.** §A0.9 (1 Oct) trains `control` seeds; the anchor became `youngdata` on 2 Nov.
   On the three banked seeds their all-age pAUC pair SD differs (0.0071 vs 0.0153; rough, 3 pairs). Screens use bars from
   the six **anchor** seeds; the control seeds run too, so the difference is measured on 15 pairs each
   (`research/v6/anchor_noise_floor.py`). This was chosen because the anchor is the comparator, before any V6 run.
2. **The stated V5 bar was a single seed** (0.676). The system scores 0.6911, and 0.7005 with TTA (0.7).
3. **TTA was not matched on the comparator** (V6-5).
4. **The frozen head stage did not scale with the step budget** (0.4).
5. **The PanDerm compute proxy was not in the benchmark script**, though 0.4 and 0.8b relied on it. Added
   (`vit_base_patch16_224`), and the V6 roster now runs first in `scripts/run_v6_benchmarks.ps1`.
6. **HIBA was not a link target for the added sources** (0.3); only extras and manifest had been checked. Checked 3 Nov
   by identifier (live API): ISIC 2020, ISIC 2019 Test and MILK10k share 0 images, 0 lesions and 0 patients with HIBA
   (1,635 images, 1,257 lesions, 628 patients). DERM12345 (patient ids only) and image hashes are still to do.
7. **Bars had no stated k.** Declared: `bar_k1` for the V6-3 single-seed gate and for the E-primary all-age pAUC, because
   each V6 system is one realisation; `bar_k3` for three-seed screens. **Confirmed by the owner 3 Nov, before any new seed
   was read.** Bars are z80 × the six anchor seeds' pair SD ÷ √k (`research/v6/anchor_noise_floor.py`), so they follow by
   formula and cannot be chosen after the fact.
8. **Class maps were declared for DERM12345 only** (0.3).
9. **V6 statistical families declared** (`research/stats/families.py`, four entries).
10. **§A0.14 still required CG-DM items** (Stage 0 script, segmenter gate and others) that V6 no longer runs, so a freeze could
    not have satisfied it. 0.8 now replaces it for Part 0.
11. **Headline changed and dilution addressed** (0.2, V6-2): the expanded pool would halve the young share, from 27.6% to
    13.1% (`results/v6/data_allocation.json`).

**Engineering gaps the freeze review must see closed (nothing else blocks):**
- `scripts/run_v5_queue.ps1` marks a run complete at `epochs_run == 30`, and shortened runs would never complete. V6 needs its
  own runner rule.
- `research/v5/train_v5.py` builds ConvNeXt arms only, and allows image sizes 224/384 and batch sizes 16/32. Members go
  through the new builder.
- `results/v6/benchmarks/` is empty until tonight's run. Benchmarks use synthetic batches, so data-bound cost needs the
  first real run of each member.

| # | 0.8 item | Status (3 Nov) |
|---|---|---|
| 1 | V5 final, §A5 filled | **Done** (S84 receipt completed; §A5 final status added) |
| 2 | S84 lesion-level exclusion | **Done** (D6 applied) |
| 3 | `data_allocation.json` | **Identifier links done for all sources and all 25,331 manifest rows** (`research/v6/data_audit.py`): 0 links to HIBA; 0 to the manifest except MILK10k's 859 lesions (D6). Image hashes, class maps, PanDerm hash to do |
| 4 | Benchmarks, 80 h cap | Queued tonight |
| 5 | Six-seed bars | Queued tonight (`results/v5/queue_V6N0.txt`) |
| 6 | C4 parity | Waits for the member builder |
| 7 | Statistical families | **Done** |
| 8 | Gate A power statement | **Done** |
| 9 | Unit tests | Waits for the member builder |
| 10 | Exact recipe values | To do with the member builder |
| 11 | Owner approval, then hash | Waits for 3, 4, 5, 6, 9, 10 |

## 0.9 What the older sections become
- **§A0 CG-DM, §A0.12 R1, §A0.13 R2:** CG-DM Stage 0 survives as a **CPU-only diagnostic** run after V6-5 on the finalists'
  features, if time allows. Stage A/B, R1 and R2 need a new amendment. Reason: the misses are nevus-like in the very
  feature space CG-DM measures distance in (review, 1 Nov).
- **§A3b / V6-2b / V6-3b (the V1 six):** replaced by the pretraining-family roster (V6-3).
- **§A10:** absorbed (pool design, accumulation, 8-class read).
- **§A11:** its Tier 1 roster is the V6-3 candidate source; its selection protocol is replaced by 0.5.
- **§A12 ADAE:** the recipe is adopted as data + diversity. The large members (B5 at 456 px, SE-ResNeXt-101,
  ResNeSt-101) enter only if benchmarked at ≤ 2.5× the anchor. Distillation is **out of the core: the deployed V6 system is the ensemble itself** (distillation is documented to hurt rare subgroups, Lukasik et al. TMLR 2022). Consequences:
  - G3 (§A1) becomes "one deployable *system*".
  - V6-5 measures the ensemble + 24-view TTA inference time per image on this GPU and on CPU (about 5 members × 24
    views = 120 forward passes).
  - The S60 service is extended to serve the ensemble, with the anchor alone as its single-model fallback.
  - PanDerm's CC-BY-NC-ND licence allows the service only as an internal, non-commercial research deployment. The
    fine-tuned PanDerm weights are never published or shared.

  Distillation returns only by amendment, nested, and must keep
  the under-40 pAUC of the ensemble within the noise floor (distillation hurts rare subgroups).
- **§A13 RMF:** A13.0 done (FixRes null); A13.1's 384-pretrained arm becomes the V6-3 384 px member; A13.3 dropped.
- **DINOv3-LLRD, N10, N13, V6-5 to V6-9, V6-12:** deferred; each needs an amendment with new evidence.
- **§A9.1 C1–C8:** kept (V6-5, V6-6).

## 0.10 External review log
**2 Nov, Antigravity** (prompt `docs/v6_design/V6_EXTERNAL_RESEARCH_PROMPT.md`; it scored Part 0 8.0/10). Each claim
was checked against our record and the four reviews before anything changed.
- **Adopted:**
  - **Weight averaging for every run** in place of EMA (V6-0, V6-1). It is kept, but rechecked and corrected the same
    day:
    - The repo's averager is epoch-level SWA. The cited evidence (MEDFAIR rank 1; +7.4 on a skin DG benchmark) is
      iteration-dense SWAD, so that evidence transfers only in part.
    - Measured here on V5's LOAO runs (`results/v5/runs/*_loao-*_s42_in22k_v5scr.json`, averaged vs last-epoch
      Macro-F1, one seed): control BCN +0.013, HAM −0.002, MSKCC −0.002; m7 BCN +0.013, HAM −0.000, MSKCC −0.003.
      That is near-neutral with a BCN-only gain. The older line "V5 SWAD secondary positive on LOAO" (§A4, §A11.3)
      overstates it.
    - It costs no GPU time and never hurt by more than 0.003, so it stays in arm R. The arm's gate and the `_last`
      side-by-side decide it.
    - Two defects are fixed: BatchNorm statistics are now re-estimated rather than averaged, and the window scales
      with the step budget.
  - **Distillation out of the core** (0.9).
  - **A Gate A power statement before any run** (V6-0).
- **Not adopted:**
  - **Replacing the 384 px member with SwinV2-T or MaxViT-T.** This was briefly adopted, then reverted on owner
    recheck the same day:
    - The cited support is SwinV2-**B** (Mahbod 2025), not Tiny.
    - In this repo both transformers trailed the anchor on HAM val (MaxViT 0.7176, SwinV2 0.7412 vs 0.7482) and were
      a wash added to the 6-CNN vote.
    - ConvNeXt-T 384 px gained in V4 S53r (+0.0297, 3/3 seeds) and on S01's HAM rows (3/3).
  - **JTT and Correct-n-Contrast** (error-set upweighting or contrast): near-repeats of S67 hard-case reweighting (−0.000
    under 40) and of the rejected morphology-conditioned hard-negative mining (§A0.8). MEDFAIR finds no such method
    beating ERM.
  - **Its per-change effect sizes and VRAM/time figures:** labelled VERIFIED but not measured on this hardware. Costs
    stay with `scripts/run_v6_benchmarks.ps1`.
- **Corrected:**
  - The +0.047 OOD AUROC from photometric augmentation is arXiv 2607.26765, not Ktena et al. (Nat Med 2024, a diffusion
    paper).
  - The MILK10k link count is **985** (859 Training + 126 Test), not 859; it is already excluded by D6.
  - Its colour-constancy evidence (Perez 2018, 0.814 → 0.842) conflicts with our review, which found no external-test
    gain. Shades-of-grey stays in arm R only because it is free once cached; arm R's gate decides.

**2 Nov, ChatGPT** (same prompt; scored 8.1/10). It scored an intermediate Part 0 in which MaxViT-T replaced the 384 px
member as core; that swap was reverted the same day (above), so its roster comments are read against the current
roster. Each claim was checked against our files and the primary source first.
- **Adopted:**
  - **Missing-identifier accounting** (0.3, 0.8). VERIFIED live on the ISIC facets API: ISIC 2019 Test has `patient_id`
    for 0/8,238 images. Its other claim, `lesion_id` for only 3,232/8,238, is **refuted**: the archive now has 8,238/8,238.
  - **MILK10k units reconciled** (0.1, 0.3): 985 linked in all, 618 of them to development training rows. §0.3 said
    "618 excluded" while D6 excludes 985; fixed.
  - **The nested selector named as a non-inferiority guard** (0.2, 0.5), and "first" dropped from the novelty claims.
  - **Gate A power reports the point-estimate, CI and joint probabilities separately** (V6-0).
  - **Per-member subgroup contribution report**, descriptive, never a gate (V6-3).
  - **DermFM-Zero / PanDerm-2 excluded and the PanDerm-B checkpoint pinned** (V6-0, 0.6). VERIFIED on its Hugging Face
    model card: training data include ISIC, BCN20000, MSKCC and HIBA. The PanDerm repo now ships it as "PanDerm-2",
    so loading "latest" weights would silently leak the confirmation cohort.
  - **PanDerm recipe** VERIFIED in its repo (50 ep, lr 5e-4, 10 warm-up, layer decay 0.65, drop path 0.2, wd 0.05,
    Mixup/CutMix, weighted sampler). Its example is PanDerm-**Large** on PAD-UFES (clinical, 6-class), so V6-3 treats it
    as a starting point, not a dermoscopy-validated recipe.
- **Not adopted:**
  - **Swin-T as a fold-0 screening candidate.** VERIFIED (J Clin Med 2026, PMID 42881884, Tables 4/8): Swin-T external
    BCN BA 0.476 vs ConvNeXt-T 0.451. But that study trains on HAM only, uses one seed per fold, gives no external CI
    and no test for the ranking reversal, and its BCN read is 6-class. V6 trains on pooled data that already contains
    BCN rows, so a HAM→BCN transfer advantage is not what V6 measures. Our in-repo record (SwinV2-T trailed the anchor
    on HAM val; a wash in the vote) stands. Revisit only by amendment if V6-3 leaves a slot under the 80 h cap.
  - **Blinded dermatologist phenotype audit of the 33 under-40 misses.** A sound idea, but it needs dermatologists
    the project does not have, and its cited 2026 age-phenotype study could not be found (search, 2 Nov). It is
    recorded as a future clinical-collaboration study, off the critical path.
  - **UQ longitudinal dataset, MCR-SL.** UQ images already feed ISIC 2020 (8,449 UQ images, `research_data_sources.md`),
    so its independence is doubtful. MCR-SL has 29 excised lesions. MRA-MIDAS was already listed (0.3, optional).
  - **SFPL prototypes, test-time adaptation.** Its own recommendation was not to add them.

**2 Nov, ensemble-member search** (subagent; report `docs/v6_design/research_ensemble_members.md`, every claim tagged).
- **Adopted (V6-3):**
  - **DINOv2 ViT-B/14 replaces EVA02-S, and SigLIP 2 B/16 replaces EfficientNetV2-S** (now its fallback). The reason
    is the objective-diversity evidence above:
    - PanDerm and EVA-02 are both masked modelling against CLIP features (PanDerm arXiv 2410.15038 v3 Ext. Data
      Tables 36/40; EVA-02 timm card).
    - EfficientNetV2-S is a second supervised CNN.
  - Support for DINOv2 on dermoscopy is weak but positive:
    - Full fine-tune on ISIC 2019, recall 0.859 vs 0.823 for ImageNet-supervised. The split unit is not stated
      (WACV 2024, arXiv 2310.19522).
    - With a swept ViT LR (≈ 1e-5), best of 12 on DermaMNIST, which is a leaky image-level split, so it is not ranked
      (arXiv 2501.14685).
  - SigLIP 2 has **no dermoscopy result**. It enters on diversity evidence alone, so it is first in the cut order.
  - Error inconsistency is added to the descriptive member report.
- **Not adopted:**
  - **PanDerm-L:** no published fine-tuned result, about 304M parameters, and its optimiser state alone is about
    4.9 GB by arithmetic.
  - **MedSigLIP:** its "Australian skin cancer" pretraining source may overlap HAM's Queensland images, which cannot
    be cleared.
  - **ConvNeXt-V2-T:** CC-BY-NC, ImageNet evidence only, and the same objective group as the anchor.
  - **EfficientNet-B4/B5 noisy-student:** the same diversity group as the anchor.
- **Newly excluded for leakage:**
  - **MedImageInsight:** ISIC 2019 (20,268 images) is in its pretraining.
  - The DermFM-Zero "Open" checkpoint publishes no data list.
- **Engineering note (V6-0):** DINOv2 needs `img_size=224` with positional-embedding interpolation in the member
  builder. `scripts/gpu_benchmark.py` already passes `img_size` to timm (line 59), and
  `scripts/run_v6_benchmarks.ps1` gains `vit_base_patch14_dinov2.lvd142m`.
- **Not found anywhere:** a backbone comparison on melanoma vs nevus, on young patients, or on dysplastic nevus vs
  melanoma in situ. Claim 3 stays open.

---

**Inherits from V5:**
- the standing rules: test lock; reserved cohort exhausted; lesion-grouped statistics; `_last`
  primary; 2 workers; ≥ 5 GB free on C:;
- the V5 dermatologist-reasoning heads that survive V5 (`docs/v5_design/V5_DERM_REASONING_ENGINE.md`);
- V5's measured noise floor.

**Contents:**
- **Part 0 — Redesign (2 Nov, current plan):** 0.1 evidence · 0.2 headline and claims · 0.3 data · 0.4 stages and
  gates · 0.5 nested selection · 0.6 not repeated · 0.7 expectations · 0.8 freeze checklist · 0.8b freeze-readiness audit · 0.8c delivery audit · 0.9 older sections · 0.10 external review log.
- **Part A — Plan:** **A0 CG-DM headline (+ R1, R2)** · A1 goals · A2 data and allocation · A3 models (A3b V1 architectures: in-domain ensemble restored, portability demoted) · A4 CV techniques · A5 decision
  table · A6 registry and leakage audit · A7 Review-2 slide · A8 sources · **A9 standing issues →
  V6 and the V2–V4 carry-forward (1 Oct evening)** · **A10 headline benchmark, pool design,
  accumulation principle (3 Oct; brainstorm input)** · **A11 ensemble roster, technique
  stack, selection protocol (3 Oct; brainstorm input)** · **A12 ADAE angle: leak-free teacher
  ensemble + distillation (3 Oct; priority brainstorm candidate)** · **A13 resolution & module-fairness study (4 Oct)**.
- **Part B — Execution:** B0 fixed rules · B1 core path · B2 conditional phases · B3 benchmarks ·
  B4 calendar.

---

# Part A — Plan

## A0. Headline: Counterexample-Guided Differential Morphology (CG-DM) — finalised, not frozen

### A0.1 Hypothesis and claim
Melanoma ranking improves when the model learns the **local morphological evidence that
distinguishes a lesion from visually similar, histopathology-confirmed benign melanocytic
counterexamples**, not only its absolute resemblance to a melanoma class. No age, sex, archive,
site, diagnosis-source or other metadata enters any input, rule or retrieval key (patient identity
is the single declared exception, R1 only, §A0.12).

**Claim progression (the only claims the evidence can support):** V5 identifies the failure →
Stage 0 tests whether differential evidence exists in the current representation → Stage A/B test
whether learning it improves the target ranking → confirmation decides whether the gain is real.
CG-DM is never described as "solving" the under-40 problem.

**Formulation and novelty (revised 1 Oct — partial explanation).** CG-DM reframes melanoma–nevus
discrimination as a **partial-explanation problem**. Rather than requiring a lesion to be fully
matched to benign counterexamples, one-sided partial optimal transport lets a population of
biopsy-proven nevi explain only part of the lesion and isolates the remaining local morphology as
differential evidence. The formulation is **motivated** by nevus-associated melanoma (NAM), in
which benign and malignant components coexist histologically (≈ 29% of melanomas in a meta-analysis
of 38 studies, I² = 99%; NAM patients ≈ 4.9 years younger — Pampena et al., JAAD 2017; a nevus
component is dermoscopically visible in only 45.6% of NAMs — Spadafora et al., Exp Dermatol 2026),
while recognising that **NAM cannot be reliably identified from dermoscopy** (Bellinato et al.,
Eur J Dermatol 2023). NAM is motivation, never an asserted explanation of the model's misses; the
current data carry no NAM labels. Novelty wording, conditional on the V6-0 audit: *"Our targeted
literature search did not identify prior dermoscopic work applying one-sided partial optimal
transport against histopathology-confirmed nevus counterexamples for low-FPR melanoma ranking."*
Never "the first". The claim "partial explanation beats balanced matching" additionally requires the
M3 success criterion (§A0.8). R1 adds the same differential against the patient's lesion signature;
R2's generated twin is **not** part of the headline claim unless it passes its gate. Not claimed:
partial / unbalanced / robust OT (Chapel et al., NeurIPS 2020; Phatak et al., ICLR 2023; Balaji et
al., NeurIPS 2020; semi-relaxed Sinkhorn; MADPOT, ICIAP 2025), dustbin-augmented Sinkhorn for
unmatched local features (SuperGlue, CVPR 2020 — related, not the same construction), OT (DeepEMD, CVPR 2020), prototype
/ case reasoning (ProtoPNet; Deformable ProtoPNet), skin prototype+concept systems (CARE-MD, 2026),
retrieval-augmented melanoma diagnosis (Melan-Dx, npj Digit Med 2026 — histopathology;
arXiv 2509.08338 — dermoscopy), counterfactual auditing, patient-context ugly-duckling models
(§A0.12), and **residual / memory-bank anomaly detection against normal-only data** *(added 2 Oct)*:
PatchCore (Roth et al., CVPR 2022, arXiv 2106.08265 — nearest-neighbour patch distance to a bank of
normal patches), sparse-coding reconstruction residuals (Zhao, Fei-Fei & Xing, CVPR 2011; Lu, Shi &
Jia, ICCV 2013), deep-feature sparse coding for medical anomaly detection (arXiv 2201.11506) and the
PCA-residual SubspaceAD (CVPR 2026, arXiv 2602.23013). CG-DM's residual is their instance-retrieved,
mass-constrained relative; §A0.8 tests it against the first two families directly, so any novelty
sentence must survive that comparison. A **targeted prior-art audit is a V6-0 deliverable** before
any novelty sentence is written.

### A0.2 Endpoints
| | Endpoint |
|---|---|
| **Clinical primary** | Under-40, histology-confirmed, pAUC@FPR≤0.20 |
| **Mechanism primary** | Melanoma vs histopathology-confirmed nevus pAUC@FPR≤0.20, all ages |
| Secondary | All-age pAUC_histo; under-40 high-melanin-tercile pAUC; melanoma F1; escalation sensitivity at matched referral; Macro-F1; per-archive mechanism readouts; counterfactual faithfulness |

Holm across the two primaries (declared in `research/stats/families.py` at V6-0). Under 40 stays
the clinical question; the mechanism endpoint carries the power (fold 0: 505 mel vs 671 histo nv).
The mechanism control pAUC (0.692–0.722 over control seeds 42–47) **quantifies the current
melanoma-vs-biopsied-nevus difficulty in this control**; it does not by itself show this is the
uniquely limiting boundary. *(Source: `results/v6/mechanism_noise_floor.json`.)*

### A0.3 Trunk and recipe — inherited, never assumed
Stage A/B use **V5's final composite-lock configuration** (trunk, resolution, recipe, `_last`,
2 workers, AMP), whatever V5 selects. Nothing in §A0 hard-codes IN-22k or 224 px. DINOv3-LLRD is an
independent branch and becomes the CG-DM trunk only if it wins its own screen.

### A0.4 Reference population and retrieval
- **Population reference class:** `class_7 == "nv"` AND `confirmation.histo_confirmed` (the
  corrected V5 definition). No bkl/df/vasc, follow-up-only or consensus-only row. Stage 0 also
  reports the historical `dx_type == histo` definition, as a definition audit only.
- **Retrieval encoder [declared]:** the S72 in1k ConvNeXt-T fold-f seed-42 `_last` model's pooled
  768-d embedding, cosine similarity, **K = 3**; top-3 references precomputed once per image per
  fold and **frozen during training**. The fold-f encoder never saw fold f, so held-out queries are
  leakage-safe; Stage 0 uses the same encoder.
- **Exclusions, in training AND evaluation:** same `group_id` (S71 group_id already merges lesion
  IDs and S49 perceptual-hash duplicates). Without this, a biopsied-nevus training query retrieves
  itself, D ≈ 0, and L_D learns bank membership instead of morphology.
- **Banks:** evaluation bank = that fold's training rows only; deployment bank = the full training
  set only. Retrieval is image-embedding-only.
- **Patient-level limitation:** the S71 manifest has no patient ID (`lesion_id`,
  `effective_lesion_id`, `group_id` only), so same-patient references across lesions cannot be
  excluded or audited on the pooled corpus. Declared limitation; auditable only where patient IDs
  exist (ISIC-2020, §A0.12).

### A0.5 Local alignment (OT) and the differential — partial (primary) and balanced (ablation)
**Common to both.** Query F3 tokens {f_i}, reference tokens {g_j}, L2-normalised; cost
C_ij = 1 − f_iᵀg_j. **Token masses** a_i, b_j ∝ lesion probability of each token from the **V6-1
out-of-fold learned segmenter** (DRE-2 chromophore mask as the declared fallback if V6-1 fails its
gate), normalised to 1 per image. Entropic Sinkhorn **ε = 0.05, 50 iterations, tolerance 1e-3**.
**Precision (revision 2 Oct):** every Sinkhorn solve — training and Stage 0 — runs in the **log domain
(logsumexp updates) in fp32 with autocast disabled**. At ε = 0.05 the kernel exp(−C/ε) reaches
e⁻⁴⁰ ≈ 4 × 10⁻¹⁸ at C = 2, and every C > ≈ 0.83 underflows to 0 in fp16, which turns the scaling
updates into 0/0. Every D statistic must pass the fp16-castability rule of §B0 before it enters a head.
**Top-10% operator (revision 2 Oct; one definition for every statistic):** T(x; a) = the smallest set
of highest-x query tokens whose cumulative mass Σ a_i ≥ 0.10; top10(x; a) = the median of x over
T(x; a). Every "top-10%" token set in §A0.5, §A0.8 and §A0.10 is T(·; a) and every top-10% value is
top10(·; a), so tokens outside the lesion (a_i ≈ 0) cannot fill the set. (§A0.13's R2 top-10% is
unchanged.)
Transport cost always means ⟨P, C⟩ over real reference tokens (no entropy term).

**Balanced CG-DM (m = 1.0, the ablation).** Balanced plan P^bal with marginals a, b.
u_i = Σ_j (P^bal_ij / a_i) C_ij; g̃_i^bal = Σ_j (P^bal_ij / a_i) g_j.
- D_k = top10(u; a); **D_bal = median_k D_k** over the 3 references.
- d_k = Σ_{i∈T_k} u_i (f_i − g̃_i^bal) / Σ_{i∈T_k} u_i over the top-10% tokens; d_bal = mean_k d_k,
  L2-normalised. Differentiable: w_i = softmax(u_i / 0.05); D_train,bal = mean_k Σ_i w_i u_i.

**Partial CG-DM (primary, declared — never selected from results).** **One-sided partial OT with a
fixed unexplained-mass sink:** the query may leave mass unexplained; the trusted benign reference is
the explaining distribution. Augment the reference with one sink of mass (1 − m); real reference
tokens carry mass m·b_j; the query keeps mass a. The sink cost is a constant: with the unexplained
mass fixed at 1 − m it adds a constant to the objective and cannot change which tokens are
selected, so it is not a parameter. **m = 0.8** is a predeclared engineering value — **not** a
biological estimate of a nevus fraction. The grid **m ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}** is a
diagnostic profile only, never a tuning sweep; m = 1.0 is exactly the balanced plan.
- c(m) = ⟨P^(m), C⟩ (real references only), mean over the 3 references.
- **Ranking statistic:** **D_part = [c(1.0) − c(0.8)] / 0.2** — the price of explaining the last
  20% of the lesion. (c(m) itself is never a score: a nevus-plus-melanoma mixture explains its
  benign part cheaply and would look benign.)
- **Phenotype statistic (M1 only):** **κ = [c(1.0) − c(0.8)]/0.2 − [c(0.8) − c(0.5)]/0.3**
  (late slope − early slope), the **late-slope phenotype**. It is *intended* to capture lesions
  that are cheap to explain early and expensive late, but a high κ does **not** establish a mixture
  (a de-novo melanoma can also have a non-linear cost curve), and a histological NAM need not show
  it in this feature space.
- **Residual differential:** r_i = a_i − Σ_j P^(0.8)_ij (unexplained mass of token i; Σ_i r_i = 0.2);
  **d_res = Σ_i r_i (f_i − g̃_i^bal) / Σ_i r_i**, mean over the 3 references, L2-normalised.
  No extra top-k, threshold or temperature.
- The partial arm's head sees **z = [LN(f_pool), LN(d_res)]**; the balanced arm's sees
  [LN(f_pool), LN(d_bal)]. Same dimensions, so the two arms are directly comparable.
- Phatak et al. (ICLR 2023) analyse **two-sided classical** partial OT; the one-sided entropic
  profile here is a different object, so its shape is treated as **empirical**. Phatak is cited only
  as motivation that transport-cost profiles carry information.

### A0.6 Losses
- **L_D (pairwise logistic):** for every melanoma query m and **histopathology-confirmed nevus**
  query b in a micro-batch, log(1 + exp(−(D(m) − D(b)))), with D = **D_part** in the partial arm and
  D = D_train,bal in the balanced arm. Other benign rows never form pairs (they would reintroduce the
  follow-up shortcut). No valid pair → zero contribution. No margin.
- **L_OT (Stage B only):** C_OT = c(1.0), the full (balanced) transport cost, mean over the 3
  references, in both arms — never c(0.8), which would score nevus-plus-melanoma mixtures as benign;
  same pairwise logistic form on C_OT.
- **Weights [declared, not tuned]:** λ_D = 0.10, λ_OT = 0.05. L_A = L_CE + 0.10 L_D;
  L_B = L_CE + 0.10 L_D + 0.05 L_OT.
- **Declared score** for all screens and endpoints: the classifier's `escalation_mass`. D is a
  feature, never the score.
  *(Audit addition 1 Oct evening — V5 Q3 lesson.)* If V5's composite lock carries an escalation
  head (`twostep`), the arm, the capacity-matched control and the V5 system all expose `s_esc`
  too. Q3 showed a dedicated escalation head can beat the same model's escalation mass by
  +0.009 to +0.018 all-age pAUC (`results/v5/screens/q3_verify_rescore.json`), i.e. a **readout**
  gain, not a representation gain. So every Stage A/B and confirmation contrast is reported on
  **both** scores, and the paper may call a CG-DM gain a representation gain only if it holds on
  `escalation_mass` (the declared score above); a gain on `s_esc` alone is reported as a readout
  gain. When the arms carry `s_esc`, the D feature enters the head that produces it as well.

### A0.7 Reference token cache (compute)
Bank ≈ 2,650–2,733 histo-nv images per fold; F3 token maps **0.41 GB fp16**, kept on the GPU;
uint8 images 0.41 GB in RAM. *(Measured: `results/v6/mechanism_noise_floor.json`.)* References use
the deterministic eval transform and stop-gradient. **The cache is refreshed once per epoch and
before every evaluation; one-epoch staleness is treated as a compute approximation and is not
assumed to be bias-free.** A **K = 3 smoke benchmark** (peak VRAM, batch time, commit headroom)
decides whether the approximation is practically acceptable; if K = 3 does not fit, the CG-DM
family stops as computationally infeasible — K is not tuned afterwards.

### A0.8 Stage 0 — CPU budget gate (one GPU feature pass, then CPU)
- **Features:** F3 tokens + pooled embeddings from the five S72 in1k fold models over the
  development rows (≈ 2.3 GB on disk, one fold at a time). One GPU pass, in a gap between queues.
  **Runs after V6-1**, whose out-of-fold masks give the token masses (§A0.5).
- **Three scores, plus two challengers below** (references from each query's **outer-fold training rows**, mirroring deployment;
  both combiners fitted only inside the **same** 5 inner lesion-grouped folds, so they are paired):
  S_0 = e (the S72 fold model's **escalation mass**); S_bal = logistic([e, D_bal]); S_part =
  logistic([e, D_part]). *(Audit fix 1 Oct evening: this line read `s_esc`, but the S72 R0 fold
  models have no escalation head — `s_esc` exists only in twostep-family arms. e is also the score
  M1's 0.0458 threshold is defined on.)* Both
  combiners are fitted on the mechanism rows only (melanoma = 1 vs histopathology-confirmed nevus = 0).
  Training computes one partial solve (m = 0.8) and one balanced solve (m = 1.0) per reference;
  Stage 0 solves the full grid.
  Δ_part,0 = pAUC(S_part) − pAUC(S_0); Δ_part,bal = pAUC(S_part) − pAUC(S_bal).
  **Comparison baseline (correction, 2 Oct ~22:00):** S_0 in every Δ_X,0 is **logistic([e]) fitted in the
  same 5 inner folds** as the other combiners, not raw e. Passing e alone through that combiner already
  costs **−0.0152 [−0.0276, −0.0047]** pooled pAUC on the fold-0 mechanism rows (per-fold refits shift
  calibration between folds; `results/v6/nrfp_probe.json`), so a raw-e baseline would handicap every
  statistic by ≈ 0.015 against the +0.010 bar. Raw-e Deltas are reported alongside, never gated.
  This is the known pooling bias of cross-validated AUC under class imbalance (Forman & Scholz,
  SIGKDD Explor. 2010, doi 10.1145/1882471.1882479); as a reported, not gated, sensitivity every Δ is
  also given as the mean of the five per-inner-fold Δs.
- **Two challengers (revision 2 Oct, owner-approved; declared before any Stage 0 run).** Each
  replaces only the way the lesion is explained; everything else is identical to D_part — the same
  cached F3 tokens and fold models, the same K = 3 retrieved references with the same same-group
  exclusion, the same token masses a and b, the same top10 operator (§A0.5), the same inner folds for
  the combiner.
  - **D_nn (nearest patch, PatchCore-style, no transport):** for reference k, v_ik = min_j C_ij over
    the reference's tokens with b_j > 0; D_nn,k = top10(v_·k; a); **D_nn = median_k D_nn,k**.
    S_nn = logistic([e, D_nn]).
  - **D_dict (population benign dictionary, sparse reconstruction, no retrieval):** per outer fold f,
    one dictionary A_f of **256 atoms** fitted by mini-batch dictionary learning (scikit-learn, version
    pinned, fixed seed) on the L2-normalised F3 tokens with a_i ≥ 1/196 of that fold model's
    **training-fold histopathology-confirmed nevi only** (≤ 200,000 tokens, fixed-seed subsample);
    codes by OMP with **5 non-zero** coefficients; residual ρ_i = ‖f_i − A_f α_i‖²;
    **D_dict = top10(ρ; a)**. Only fold-f queries are coded with A_f, so no query reconstructs itself.
    S_dict = logistic([e, D_dict]). Atoms and sparsity are declared, never tuned.
  - Cost: D_nn reuses the cost matrices (minutes); D_dict is **unmeasured** — benchmark on fold 0
    first; tokens streamed per fold in fp16 (commit headroom).
- **Budget gate (applied to each of S_part, S_nn, S_dict against S_0):** on the mechanism endpoint,
  lesion-bootstrap 95% CI of Δ_X,0 > 0 **and** Δ_X,0 ≥ +0.010 (a pre-declared minimum effect); under-40
  histology-confirmed Δ_X,0 ≥ 0 (direction only); **and (revision 2 Oct) archive-stratified
  Δ_X,0 ≥ 0** — the melanoma-count-weighted mean of within-archive Δ pAUC over the archives with ≥ 20
  melanomas and ≥ 20 histopathology-confirmed nevi, CI reported. Reason: archive identity alone scores
  pAUC@0.2 **0.539** on the fold-0 mechanism rows (melanoma share HAM 0.33 / BCN 0.49 / MSKCC 0.64), so a
  score that only detects atypical acquisition could clear +0.010 without morphology. Δ_bal,0 is
  reported, not gated. **If no statistic passes, the CG-DM family stops** (budget stop, below).
- **Selection among the statistics that pass (declared order; mechanism endpoint, lesion
  bootstrap; Bonferroni over the two contrasts Δ_part,nn and Δ_part,dict — each read on its
  two-sided **97.5%** interval, familywise 95%. *(2 Oct audit: this read "Holm-adjusted bounds";
  Holm is a step-down p-value procedure with no simultaneous-interval form.)*):**
  1. **Equivalence first:** a challenger X replaces D_part if the 97.5% upper bound of
     Δ_part,X < **+0.005** — CG-DM is then shown to add less than 0.005 over the simpler statistic. If
     both challengers qualify, the simpler one is carried (**nn < dict < part**).
  2. Otherwise, if both 97.5% lower bounds of Δ_part,X > 0, **CG-DM is carried**.
  3. Otherwise (inconclusive) **CG-DM stays primary**, and D_nn and D_dict are reported as ablation rungs.
  If only one statistic passes the gate, it is carried.
- **Routing:** Stage A trains the **carried statistic** in place of D_part in L_D (D_nn via a soft-min
  with the §A0.5 temperature 0.05; D_dict with codes recomputed under no-grad each step and the gradient
  through f_i). The balanced arm stays the ablation. The route is **never** to V6-5(a) or to resolution:
  384 px (under-40 Δ −0.0192, `results/v5/s01_decision.json`) and zoom (−0.0033 vs clues, −0.024 vs
  twostep, `results/v5/screens/q4_verify.json`) left young ranking flat or worse, and HAM — 14 of the
  33 young misses — is natively 600 × 450.
- **M1 — partial-explanation enrichment (representation hypothesis; does NOT identify NAM).**
  Lesion level, as B1: a melanoma lesion is **missed** if its maximum S72 OOF escalation mass is
  below the locked FPR-20 operating threshold **0.0458** (`results/v5/diagnostics/b1_hard_core.json`),
  else **detected**; lesion κ = max over its images. **Threshold τ_f** = 90th percentile of the
  out-of-fold κ of histopathology-confirmed nevi from the **other four outer folds**, where, for
  computing τ_f, each of those κ values is **recomputed with fold-f rows removed from its reference
  bank**. So no fold-f lesion enters τ_f as a query or as a reference, and every feature is
  out-of-sample (each row scored by the outer model that never saw it). **Declared residual
  dependence:** the other folds' models were trained on fold-f images; that is inherent to pooled
  K-fold OOF (B1's 0.0458 threshold shares it). A fully nested τ would need inner-fold models
  (extra GPU), so it is not done. Fold-f-internal κ (fold-f model on outer-training rows) is
  **rejected**: those features are in-sample.
  The phenotype is called the **late-slope phenotype**, never "mixture-type".
  Statistic: P(κ > τ | missed) − P(κ > τ | detected); lesion-bootstrap 95% CI > 0, all ages;
  under-40 direction only.
- **M2 — spatial structure (no direction assumed).** Per lesion: residual-mass-weighted distance of
  token centres to the lesion boundary (V6-1 out-of-fold mask, eval-crop frame), divided by the
  equivalent radius √(area/π). Compare late-slope-phenotype (κ > τ) vs other melanomas, two-sided
  lesion-bootstrap CI; direction reported descriptively. Sensitivity: HAM expert masks, HAM rows.
- **M3 — partial vs balanced (the decisive methodological test).** Report Δ_part,bal with its
  lesion-bootstrap CI. **Success criterion for the claim "partial explanation beats balanced
  matching":** (1) Stage 0 CI of Δ_part,bal > 0 on the mechanism endpoint **and** (2) at
  confirmation (folds 1–4), the trained partial arm beats the trained balanced arm on the mechanism
  endpoint (V5 hierarchical bootstrap, CI > 0). Otherwise the paper may say only that partial CG-DM
  adds information, not that it beats balanced matching.
- **Exact audit:** a **predeclared list of 1,000 query/reference pairs** (fixed seed, fold 0, written
  before Stage 0 runs) solved exactly (POT, version pinned) vs entropic; report agreement of c(m),
  D_part and κ. If exact turns out cheap, it may replace entropic for all of Stage 0; validity never
  depends on the exact run.
- **Biological validation (optional):** only with **independent histopathology-confirmed NAM
  labels** for dermoscopic images (none in the current data; check ISIC metadata at V6-0, metadata
  only): does κ > τ enrich for NAM?
- **Considered and not adopted (2 Oct; CHANGELOG):** interior–exterior cross-attention (S50: the
  peri-lesional region encodes **age** more than the lesion, +0.0297 [+0.0174, +0.0424], while its
  escalation AUC is not higher, 0.8709 vs 0.8744 — an age-proxy risk in the barred family);
  morphology-conditioned hard-negative mining (near-repeat of S67 hard-case reweighting, −0.000 under 40,
  and of pAUC-DRO / subclass Group-DRO; clusters built from out-of-fold misses would leak the evaluated
  fold); **native-resolution frequency channel (NR-FP)** — tested on CPU with its own proposed falsifier,
  corrected (`research/v6/nrfp_probe.py` → `results/v6/nrfp_probe.json`, fold 0, 505 mel / 671 histo-nv):
  image high-frequency statistics decode the **archive at macro AUC 0.960** (e alone 0.566) and add
  **−0.0043 [−0.0247, +0.0153]** over the fair baseline (archive-stratified −0.0014; band 112–192
  cycles +0.0024, band > 192 −0.0054; under-40 +0.038 [−0.011, +0.092] on 21 melanomas, direction only,
  archive one-hot alone +0.014) — gate FAIL; its learned form is V6-5(a)'s family (Xu et al., CVPR 2020,
  arXiv 2002.12416); **topology / persistent homology** — `research/v6/topology_probe.py` →
  `results/v6/topology_probe.json` (fold 0, 224 px view, luminance-sublevel and gradient-superlevel Betti
  curves at 16 per-image quantiles, exact β₀/β₁): archive decodability **0.816** despite quantile
  normalisation (median dark components at q47: BCN 42, HAM 100, MSKCC 112 — not camera-invariant), Δ over the
  fair baseline **−0.0266 [−0.0513, −0.0005]**, archive-stratified −0.0288, under-40 +0.0011 — FAIL; the
  multiparameter form (medRxiv 10.1101/2025.11.25.25340992, gains unverified — paper behind a bot check) is
  untested; **hypergraph token routing** — no probe (needs the Stage 0 F3 cache; may run there as a
  descriptive diagnostic only); its cited evidence (Sci Rep 2026, HAM10000, stratified image-level splits, no
  lesion grouping, accuracy only) is leak-prone and builds sample-level, not token-level, hyperedges; global
  token interaction is already covered by CG-DM's transport and V6-5(b); **concept anchoring (PACO)** — is
  V6-8 as planned (MONET's concept scores are themselves image–text-prompt similarities, Kim et al., Nat Med
  2024), and its proposed probe reads MILK10k labels before S84 (C8 — barred). WSD-based contrastive learning
  is not a separate arm: WSD stays V6-5(a) under its own gate,
  and its loss is the D_nn family tested here.
- **Failure is a budget stop, not falsification:** "the CG-DM family is stopped under the
  preregistered compute-allocation rule because the current representation does not demonstrate
  incremental differential information sufficient to justify end-to-end training." It is not
  evidence that L_D-trained features cannot learn the signal.

### A0.9 Stage A / Stage B screens and confirmation
- **Stage A:** L_A for **two arms — partial CG-DM (primary; the carried statistic's arm if §A0.8
  replaced D_part) and balanced CG-DM (ablation)** — each
  seeds 42/43/44, fold 0, vs one **capacity-matched control** (same head; the 384-d
  differential slot is filled with GAP(F3); no references, no L_D). **Gate = the V5 screen gate
  unchanged** (`research/v5/screen_gate.py`: 3-seed mean of same-seed differences; pass if
  ΔpAUC_all > z80·SD/√3 (0.0035) **or** ΔpAUC_histo > (0.0065), strict `>`, and ΔMacro-F1 ≥ −0.0153;
  *these three values are the in1k bars — on the IN-22k trunk they are replaced as stated below*).
  The mechanism endpoint is read alongside as **Δ_mech = (1/3) Σ_{s∈{42,43,44}} [pAUC_mech(arm, s) −
  pAUC_mech(control, s)]**, compared with **T_mech = 0.0075** (pair SD 0.0153, V5's RMS
  definition) as a descriptive readout, not an extra veto. T_mech is a 3-seed screen bar and is
  distinct from the Stage 0 rule.
  *(Audit fix 1 Oct evening — the numbers above are in1k numbers; the gate's **form** stays
  unchanged, its **noise floor** must match the trunk the screens run on.)* The bars 0.0035 /
  0.0065 / −0.0153 come from the in1k control seeds. On the IN-22k trunk V6 inherits, the three
  Q3 control seeds vary **2.15×** as much on all-age pAUC (pair SD 0.0153 vs 0.0071), **1.48×** on
  pAUC_histo and **1.35×** on Macro-F1, but only **1.02×** on the mechanism endpoint
  (`results/v6/noise_floor_in22k_check.json`, `research/v6/noise_floor_trunk_check.py`). So
  **T_mech's in1k proxy holds; the gate bars do not** — at in1k bars a null IN-22k arm would pass
  far more often than the gate's stated 20–36%. Rule: **V6-0 trains IN-22k control seeds 45/46/47
  at fold 0** (V5 recipe, `--no-save-best`; 3 runs ≈ 2.0–2.5 h, from the measured IN-22k control
  anchor 40.5–49.9 min, `results/v5/logs/`), and every V6 screen on that trunk (Stage A/B, V6-2,
  the E-primary noise floor) uses bars recomputed by the frozen `screen_gate.null_model` formula
  from the six IN-22k seeds (15 pairs). On a different inherited trunk, the same rule applies to
  that trunk. Provisional 3-pair IN-22k bars, for orientation only: 0.0074 (all-age), 0.0097
  (histo), retention −0.0207.
- **Run validity (revision 2 Oct; V5 Q4 lesson).** A screen or confirmation run is **void** — rerun
  once under the same seed after the cause is fixed, never scored as PASS or FAIL — if any epoch logs a
  non-finite training loss, or if `front_token_backstop_hits` (or the equivalent count for any
  engineered statistic) exceeds **10 per 10,000 image passes** (≈ 10× the tail rate measured after the
  2 Oct fix, 1–3 per 24,470). A voided run is listed with its cause; it never silently disappears.
- **Stage B:** only if Stage A passes; L_B; same seeds, control, references and gate.
- **Confirmation:** the selected stage of the **carried statistic's** arm (§A0.8; D_part unless
  replaced), the balanced arm (needed for M3) and the control × folds 1–4 × seeds 42/43/44, V5's
  hierarchical bootstrap (seeds, then lesions). **Fold 0 is the selection fold and is never pooled
  into a confirmatory estimate.** If D_nn or D_dict was carried, M3's claim ("partial explanation
  beats balanced matching") is not made; Stage 0's Δ_part,bal is reported descriptively.

### A0.10 Counterfactual replacement — evaluation only
For a trained model: replace the top-10% query tokens — by u_i in the balanced arm, by unexplained
mass r_i / a_i in the partial arm — with their balanced-plan barycentric benign counterparts g̃_i^bal, propagate the edited F3 through the rest of the network (d recomputed against the
same cached references), and compare with **20 matched random draws** that change only *which*
tokens are replaced (same count, references and barycentric operation). **Signs are fixed:**
Δ_CF = s(original) − s(differential tokens replaced); Δ_RAND,r = s(original) − s(random tokens
replaced), with s = the declared escalation score. E_CF = Δ_CF − mean_r Δ_RAND,r, computed on
melanoma queries. **E_CF > 0 means replacing the model-identified differential evidence with its
benign counterpart lowers melanoma evidence more than a matched random replacement.** Pass if the
lesion-bootstrap 95% CI of E_CF > 0 on the mechanism endpoint (under 40: direction only). A faithfulness diagnostic, not causal proof. Pixel deletion is excluded (ROAR,
NeurIPS 2019: deletion creates an out-of-distribution input).

### A0.11 Ablation ladder (the paper's mechanism table)
V5 control → DRE-6 `memory` (references as prototypes) → **RML** = [LN(f_pool),
LN(f_pool − mean_k f_pool(b_k))], same retrieval, no L_D → **balanced CG-DM** (Stage A, m = 1.0)
→ **partial CG-DM** (Stage A, m = 0.8, primary — or the carried statistic's arm, §A0.8) → Stage B
on that arm (→ + R2). Both CG-DM
arms are trained, so M3's trained-arm criterion (§A0.8) is testable.
Q3's `memory` runs are reused only if trunk, recipe and data match exactly; no duplicate retraining.

### A0.12 R1 — Signature-anchored counterexamples (upgrades N13; independent gate; optional mode)
**Why:** CG-DM's differential is computed from the network's own features and the training set, so
it cannot add information a single image lacks. R1 adds new information: **the patient's lesion
signature** — the appearance of that person's *other* lesions. Because those lesions are used
unlabelled, the signature can contain another suspicious lesion; it is a lesion signature, not a
guaranteed nevus phenotype. Its intended clinical analogue is the signature-nevus pattern (Suh &
Bolognia); the ugly-duckling sign reached 0.9 sensitivity across observers (Gachon et al., Arch
Dermatol 2005); patient-wise feature normalisation featured in top ISIC-2024 solutions on the same
pAUC metric. Nevus counts peak in the third decade, so the signature is richest in the failing band
**without any age input**.
**Counter-evidence, stated up front:** in the 2020 SIIM-ISIC challenge analysis, patient-contextual
images **did not improve AI algorithms or human readers** (median reader sensitivity 60.0% → 60.0%,
specificity 86.7% → 85.7%; Kurtansky et al., JEADV 2025, doi 10.1111/jdv.20479). *(2 Oct audit: the
figures previously quoted here, 85.6% → 82.1%, are not in the abstract and were replaced.)* The authors
note that seven contextual images and no total-body image may have been insufficient; see also Wen et
al., JEADV 2026 (doi 10.1111/jdv.70061) on design and data biases in patient-context AI. R1 is
therefore a genuine test, not an expected win.
**Mechanism (fixed K):** for a query whose patient has **≥ 3 other lesions**, take the **top 3**
by image-embedding similarity (no replacement) → d_patient by the same one-sided partial operator
(§A0.5); d_population from the §A0.4 bank (K = 3) as in CG-DM; the head sees
[LN(f_pool), LN(d_patient), LN(d_population)].
**Contamination-tolerant variant (robustness, reported, not primary):** two-sided partial OT
(Chapel et al., NeurIPS 2020) on the patient tier, so reference mass may also go unused. This
*reduces sensitivity* to an unlabelled suspicious lesion in the patient's signature; it does **not**
guarantee that lesion is excluded.
**No patient-availability indicator** (it would be a dataset-structure proxy). Queries with < 3
other lesions are scored by the **population-only CG-DM model**, so the R1 head never sees them.
**Control:** an R1 capacity-matched control with the identical head (same dimensions and parameter
count) where the d_patient slot is filled with GAP(F3) and no patient references are used. N13's set
encoder is a second comparator, its exact configuration **locked at V6-0**.
**Deployment:** R1 is an **optional patient-context mode** for multi-lesion workflows (total-body /
comparative examination). The single-lesion core deployment is CG-DM alone.
**Rules:** the patient's other lesions are used **without their labels** (unbiopsied at
deployment — no label filtering, so training matches deployment); patient identity is the only
grouping key permitted (same reasoning as N13: other lesions' *images*, no tabular data); folds are
**patient-held-out**; age/sex stay banned.
**Prior art:** patient-specific / ugly-duckling context is **not** novel — DMT-Quadruplet
(arXiv 2309.09689), the end-to-end ugly-duckling transformer (MICCAI 2021), Soenksen et al.
(Sci Transl Med 2021), ISIC-2024 tabular patient normalisation. R1's narrow novelty is only the
**patient-contextual local OT differential against the patient's lesion signature, with a
biopsied-nevus population fallback.**
**Data:** ISIC-2020 (33,126 dermoscopic images, 2,056 patients, ≈ 16 lesions/patient, 584
histology-confirmed malignant; median age 50); ISIC young-data candidates carry `patient_id` for
1,208 images / 537 patients (243 escalating) *(`results/v5/young_data/candidates.csv`)*. **Count
under-40 malignancies in ISIC-2020 at V6-0 (metadata only) before committing GPU.**
**Evaluation:** ISIC-2020 patient-held-out folds. HIBA has ≈ 2 images/patient (1,270 / 623), so
HIBA can test only the population-fallback mode — declared now.
**Gate:** its own Stage 0 on ISIC-2020 with S72 features — does the patient-tier differential add
information beyond the S72 escalation mass e (audit fix: not `s_esc`, see §A0.8) **and** the population differential (patient-grouped bootstrap; same
pass-rule form as §A0.8)? Independent of CG-DM's Stage 0. N13's set encoder remains the comparator.

### A0.13 R2 — Pseudo-benign twin (conditional, high risk; NOT in the headline claim)
**Why:** models the benign class instead of the scarce one (81 under-40 escalating lesions in the
corpus). **Training population = the §A0.4 reference class only** (histopathology-confirmed nevi,
2,650–2,733 per fold of 6,425–6,433 nevi; `results/v6/mechanism_noise_floor.json` and the fold
counts in the CHANGELOG). A twin learned from all nevi would be pulled toward easy follow-up nevi
and reintroduce the verification shortcut.
**Operation (defined, not assumed):** "benign projection at noise level t*" = diffuse the query
to a **single declared noise level t* = 0.40 of the schedule** and denoise with the nevus-only
model (DDIM, **20 steps, fixed seed**). This is not claimed to be the "nearest" benign lesion; it
is a fixed, reproducible projection whose usefulness the gate tests. t* is not tuned.
**Cross-fitted generator (no self-reconstruction shortcut):** a generator applied to a nevus it
was trained on would reconstruct it unusually well ("seen by the generator"), while training
melanomas are always unseen — the same defect as self-retrieval. So within each training fold the
reference class is split **2-way by `group_id`** (fixed seed); two generators are trained, and every
training query (nevus **and** melanoma, identically) gets its twin from the generator that did
**not** train on its `group_id` half. Held-out queries use a generator trained on the whole
training fold. (Doubles generator training; covered by the feasibility gate's benchmark.)
**Separate evidence channel (K = 3 unchanged):** the three retrieved references stay exactly as in
§A0.4–A0.5. The twin gives its own channel: per-token discrepancy **q_i = 1 − cos(f_i, f_i^twin)**
on the pixel-aligned F3 tokens (no OT); T = the top-10% tokens by q_i;
**d_twin = Σ_{i∈T} q_i (f_i − f_i^twin) / Σ_{i∈T} q_i**, L2-normalised; the head sees
[LN(f_pool), LN(d_res), LN(d_twin)] against a capacity-matched control whose d_twin slot is GAP(F3).
**Feasibility gate (fold 0 only, before any real budget):**
(1) **128 px is feasibility only**; any positive result must be re-established at V6's final
resolution before it counts; generator training time is benchmarked first.
(2) **Benign identity preservation:** on histo-confirmed nevi, the frozen control's escalation
score changes by a median |s(x) − s(T(x))| ≤ 0.02 and pooled-feature cosine(x, T(x)) has median
≥ 0.90 [declared].
(3) **Melanoma-directed score reduction:** with Δs = s(x) − s(T(x)) from the frozen control, the
lesion-bootstrap 95% CI of [mean Δs(melanoma) − mean Δs(histo nevus)] > 0. (A generator that
merely changes melanomas more cannot pass (2)+(3) together.)
(4) The §A0.8 incremental-information test on the twin residual statistic.
Any failure → R2 dropped; CG-DM unaffected. Runs only after CG-DM Stage A.
**Prior art (crowded):** diffusion "healthy" counterfactuals (arXiv 2207.12268; brain
arXiv 2308.02062; PHANES); dermatology counterfactual auditing of melanoma classifiers (DeGrave et
al., Nat Biomed Eng 2023); skin generative models (DermaFlux 2026, controllable dermoscopy
synthesis); benign-first reverse-exclusion screening. A diffusion benign counterfactual is **not**
a novelty claim; at most, its use as an additional low-FPR differential channel inside CG-DM is.

### A0.14 Freeze rule
The V6 plan is hashed only after: (1) V5 is final and §A5 is filled; (2) §A0 inherits V5's
composite-lock configuration; (3) T_mech is recorded (done: `results/v6/mechanism_noise_floor.json`,
kept as provenance until the freeze); (4) the K = 3 loader benchmark is recorded; (5) the Stage 0
script and output schema are frozen; (6) retrieval and leakage rules are frozen; (7) the prior-art
audit is written; (8) the 1,000-pair exact-audit list is written (fixed seed) and the POT version
pinned; (9) the V6-1 segmenter gate result is recorded (learned out-of-fold masks, or the declared
DRE-2 fallback); *(2 Oct)* (10) the Stage 0 script implements and unit-tests the fair combiner
baseline, the archive-stratified condition, the Bonferroni selection and the per-fold sensitivity;
(11) the D_dict CPU fit is benchmarked on fold 0; (12) every engineered statistic (D_part, D_bal,
D_nn, D_dict, fronts) carries a backstop counter, so the §A0.9 void rule is checkable. No V6 model
result may alter these choices.

### A0.15 Order (parallel where independent)
V6-0 (freeze, data, counts, prior-art audit) → **V6-1 segmenter (mandatory for CG-DM,
out-of-fold)** → in parallel: **CG-DM Stage 0** · **R1 Stage 0**
(ISIC-2020) · **young-data learning curve** · **DINOv3-LLRD screen**. CG-DM pass → K = 3 benchmark
→ Stage A → Stage B → (R2 gate) → counterfactual evaluation → folds 1–4. Conditional afterwards:
pAUC-DRO (only after CG-DM works), topology (**deprioritised 2 Oct**: single-parameter Betti curves fail
the CPU probe, §A0.8 "considered"; only a multiparameter form with new evidence may return), MONET/DermFM
concept teachers (DermFM barred near HIBA
until its training data is audited). Then finalists → distillation → HIBA / ISIC-2020 confirmation.
**Budget (extrapolated from the 41-min anchor; retrieval overhead unmeasured):** CG-DM ≈ 51 runs
≈ 35 GPU-h (screens: capacity control, RML, balanced, partial, Stage B × 3 seeds = 15; confirmation:
partial, balanced and control × folds 1–4 × 3 seeds = 36) + V6-1 (≈ 1–2 h) + the Stage 0 feature
pass; R1 and R2 are benchmarked before any hours are quoted. *(2 Oct audit)* The per-run anchor is
the **V5 composite-lock arm's measured run time**, not the control's: if the lock carries `zoom`, a
fold-0 run measured 73–81 min (V5 Q4, `results/v5/logs/queue_Q4.log`), so the 51 runs ≈ 62–69 GPU-h;
plus ≤ 1 rerun per voided run (§A0.9) and the unmeasured D_dict CPU fit.

## A1. What V6 is for

| Goal | Where V5 leaves it | What V6 adds |
|---|---|---|
| **G1 · Under-40 escalation ranking** | V4 proved a representation ranking ceiling. V5 tests biology-derived heads on the same data and ConvNeXt-T trunk | **CG-DM** (the headline, §A0: differential evidence against biopsied-nevus counterexamples, with the Stage 0 head-to-head), **more young histology-confirmed data**, R1 patient context, DINOv3-LLRD; a second view and subtype objectives only if their gates fire *(2 Oct: this row predated the 1 Oct headline)* |
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
| **HIBA** | 1,270 contact-polarised dermoscopy images, 623 patients (Argentina) | **V6 confirmation cohort** (a new population and device): single read. **Primary on HIBA = the mechanism endpoint** (melanoma vs histopathology-confirmed nevus pAUC@0.2); under-40 and the S70 contract (label-free budget lock, C2) are **descriptive** there (revision 2 Oct) | Licence; deduplication; eligible under-40 escalating count and power statement before the lock |
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

## A3b. The six V1 architectures: V5-feature portability and ensemble members (revision 30 Sep) — **ensemble RESTORED 1 Oct evening; portability demoted**

> **Demoted (1 Oct):** not on the primary path. Under-40 AUC moved only +0.004 with ensembling
> (S64), and §A0 needs the budget. V6-2b / V6-3b run only if budget remains after §A0.15; the
> design below is kept unchanged for that case.
>
> **Restored (1 Oct, evening; owner request, pre-freeze, no V6 result exists):** the demotion
> above stands for **V6-2b only**. **V6-3b and the V6-10 E-primary return to the core path** as a
> parallel G2/G3 branch that uses GPU time the CG-DM path leaves idle (its Stage 0 is CPU work).
> Why the S64 reason does not cover the ensemble:
> - S64's +0.0037 [−0.0421, +0.0516] is **under-40 escalation AUC** on HAM OOF (G1). The ensemble
>   is a **G2/G3** lever: Macro-F1 +0.026 on HAM test with McNemar p = 3.4e-05 (`results/mcnemar_delong.json`),
>   the only Phase-1 intervention that cleared significance.
> - **It was never tested in domain.** S64 itself: "an ensemble of the six V4 models is untested;
>   it is S67 stage 0"; S67 stage 0 never ran (`results/v4/final_verdict_v4.json`). The V1 six are
>   HAM-only (0.411 Macro-F1 on reserved), so the question "does architecture diversity add on the
>   pooled corpus?" is open.
> - **What is already known, and binds the design:** adding SwinV2-T + MaxViT-T to the six CNNs was
>   a **wash** on HAM (0.7981 vs 0.7986) and Caruana selection over 8 **overfitted** (0.7640/0.7839
>   vs 0.7945 uniform) (session 6, CHANGELOG §13) → **six members, uniform soft vote primary**, no
>   member-count expansion without OOF evidence. The soft-vote ensemble is **under-confident**
>   (CHANGELOG §11; also Wu & Gales 2021) → calibrate **after** combining (Dirichlet on the
>   ensemble's cross-fitted OOF — already in V6-10), never per member only.
> - **External evidence:** the ISIC-2019 and SIIM-ISIC-2020 winners were both diverse-backbone
>   ensembles (Gessert et al. 2020; Ha et al. 2020); more divergent training methods give less
>   correlated errors and larger ensemble gains (Gontijo-Lopes et al., ICLR 2022). Whether the
>   under-40 band benefits is **not** settled by the literature: homogeneous ensembles can
>   disproportionately help minority groups (Ko et al., NeurIPS 2023) but ensemble benefits can
>   also be **disparate** across groups (Schweighofer et al., ICML 2025) → under-40 pAUC stays
>   **descriptive**, reported beside the per-band calibration.
> - **Members are plain in-domain controls** (V6-2b is not run, so no heads are ported; the §A5
>   row "V6-2b: V5 heads help on ConvNeXt only" applies by default). The ConvNeXt-T member already
>   exists: the banked S72 `ml/checkpoints/convnext_tiny-v4_R0_kfold_f{0-4}_s42_{best,last}.pt`
>   (the V5 in1k control recipe, 224 px, pooled S71 folds; OOF in `results/v4/kfold/oof_predictions.csv`).
>   V6-3b therefore trains **25 runs** (5 architectures × folds 0–4 × seed 42).
> - **Partition lock (audit, 1 Oct evening):** every V6-3b member trains on the **S71 partition only**
>   (15,294 rows, `results/v4/kfold/`), never on V6-0/V6-4's expanded data, so the banked S72 member is
>   like-for-like and the E-primary contrast against the V5 system (also S71) compares diversity, not
>   data. A model trained on expanded data may join only as a declared V6-10 secondary.

**Why.** V5 screens and confirms every mechanism on one architecture (ConvNeXt-T). A V5 gain is
therefore shown for ConvNeXt-T only; whether it transfers, and whether it adds on top of an
architecture-diverse ensemble, is untested. The V1 six are the repo's own diverse set, and Phase 1
found ensembling the only intervention that cleared significance (McNemar p = 3.4e-05, Holm-surviving
DeLong on bkl, `results/mcnemar_delong.json`), on HAM. But the V1 checkpoints are **HAM-only** and transfer poorly (V1 stack on
BCN/MSKCC reserved Macro-F1 0.411, `results/v4/s54/s54_marginals.csv`), so they are **retrained in
domain**, not reused.

**"V5 features" carried to every architecture** (**V6-2b only** — V6-3b members are plain controls, see
the box above; fixed at V6-0 from V5's results, nothing re-tuned):
- the V5 recipe: pooled S71 lesion-grouped folds, `_last` primary, 30 epochs, two-stage head /
  fine-tune, class-weighted CE with label smoothing 0.05, `--num-workers 2`;
- the V5-chosen resolution from the final composite lock (224 px only if that is the frozen V5
  outcome; S01 failed → 224 px as of 1 Oct, `results/v5/s01_decision.json`);
- the V5 heads in the **composite lock** (`results/v5/composite_lock.json`), each with its V5
  parameter card, and DRE-7 only if it passed its sign check;
- the V5 system's post-hoc layers: Dirichlet on cross-fitted OOF, S56 refit, per-band calibration.

**Pretrained weights:** the V1 weights family (torchvision ImageNet-1k, `ml.training.common.build_model`)
so the contrast isolates *in-domain training + V5 heads*. *(2 Oct audit — corrected: this line said the
ConvNeXt-T member uses the IN-22k trunk, contradicting the V6-3b row and its 25-run count.)* The
ConvNeXt-T member is the **banked S72 IN-1k** fold set (§B1 V6-3b), so all six members share the IN-1k
weights family; the IN-22k trunk enters through the V5-system comparator, not through a member.

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
| G1 | **Learned segmenter** (legacy label "A11a"; unrelated to §A11) | Gives exact lesion geometry for the periphery, asymmetry and context statistics on every archive | Needed unless DRE-2 passed Q1 with median Dice ≥ 0.85 | Dice ≥ 0.85 on held-out HAM, ≤ 5% failures on the 50 + 50 BCN/MSKCC human QC |
| G2 (all-age; under-40 descriptive, 2 Oct) | **Separate-trunk dual-stream** (global + lesion crop; late fusion or cross-attention) vs a **capacity-matched ConvNeXt-S** | Optical zoom on small lesions; context kept separately | DRE-8 / 8b showed "where to look" matters **on escalation mass** (§A5 zoom row) | Gain must beat the capacity-matched single stream and concentrate in the smallest lesion-size tercile (A01 B10) |
| G2 (all-age; under-40 descriptive — 2 Oct: 384 px −0.0192 and zoom −0.024 under 40) | **Wavelet space-to-depth** (768 px → Haar LL/LH/HL/HH × RGB = 12 channels at 384²) | Recovers fine structure lost when downsampling (BCN 2.67×, HAM 1.17×) at 384 px compute | DRE-8 gain larger on BCN/MSKCC **on escalation mass** | Gain on BCN/MSKCC, not on HAM |
| G1 | **Multi-proxy contrastive** loss (ECL-style, on DRE-6 prototypes) | Keeps subtypes separate while pulling rare classes together; avoids SupCon's class collapse | DRE-6 passed | Young mel prototype separation (χ²) improves |
| G1 | **Subclass Group-DRO** (N6) | Worst-subtype risk instead of tiny age groups (V5 audit AU6) | D4 found a low-pAUC, <40-enriched subclass | Worst-subclass pAUC rises; others retained |
| G1 | **Subtype supervision** (DERM12345 subclasses) | Explicit melanoma/nevus subtypes (e.g. superficial spreading, Spitz/Reed, congenital) | Licence OK; DRE-6 passed | Subtype-labelled young mel ranked better |
| G1 | **pAUC-DRO** loss (LibAUC) | Optimises low-FPR partial AUC directly, focusing on the hardest benign mimics. Hard negatives come from **training-fold rows only** — never from out-of-fold misses of the fold being evaluated (2 Oct) | M4 passed | pAUC_histo improves over M4 |
| G1 | **N5 young-data expansion** with a learning curve | Only 81 under-40 escalating lesions in V5 | V5 `youngdata` within-band pass (§A5 row; 2 Oct: was "always", contradicting §A5) | OOF under-40 pAUC slope over 25/50/100% of added young data > 0 |
| G2 | **N10 FCMAE domain-adaptive pretraining** (ConvNeXt-V2 on unlabelled multi-device dermoscopy; no confirmation images) | The trunk learns every device's image statistics without labels | M7/LOAO showed an acquisition gap | LOAO Macro-F1 over the same fine-tune from ImageNet/IN-22k |
| G2 | **N9 class-conditional acquisition adversary** | Archive predicts the label only through prevalence (archive alone pAUC 0.539 on the mechanism rows, §A0.8), so class-conditional invariance removes acquisition cues without removing class signal; age is never a target | M7 lowered decodability but did not close the gap | Within-class decodability falls **and** LOAO rises; in-distribution retention ≥ −0.01 |
| G2 | **SWAD** (dense weight averaging) | Flat minima reduce the domain gap (+1.6% OOD on DG benchmarks) | V5 SWAD secondary positive on LOAO | LOAO Macro-F1 |
| G2 | **Test-time normalisation adaptation** | Adapts normalisation statistics to a new site from unlabelled images | — | LOAO Macro-F1; no label use |
| G2 | **N11 Saerens EM prior** (CPU) | New sites have different class priors; changes the argmax only (Macro-F1), never claimed as an under-40 fix. **Near-repeat warning:** registry item "Global prevalence prior correction (S58 null)" — run only as a LOAO Macro-F1 analysis, lowest priority | — | LOAO Macro-F1 with vs without |
| G2 | **N12 smartphone routing** | S69 gate AUROC 0.9994 → route to a clinical-photo expert instead of refusing (HAM→PAD warm start reached 0.760) | — | Patient-grouped PAD CV and MILK10k clinical pairs |
| G1 | **N13 ugly duckling** (set encoder over one patient's lesions) | In young patients with many nevi, the outlier lesion is the strongest clinical sign | ISIC-2020 patient IDs available | Young-mel rank among a patient's own lesions improves |
| G1/G3 | **MONET concept labels**, validated against Task-2 expert masks before use | Scales concept supervision (DRE-7) to every image; the policy forbids unvalidated pseudo-labels | DRE-7 sign check passed | Concept AUROC vs Task-2 masks ≥ a pre-declared bar; then concept-supervised DRE-7 |
| G3 | **Heterogeneous ensemble → morphology-gated ensemble (B10) → distillation** into one model | Deployable single model with ensemble accuracy | ≥ 2 finalists | Distilled model within 0.01 Macro-F1 / pAUC of the ensemble |
| G1/G3 | **V5-feature portability** across the six V1 architectures (§A3b, V6-2b) | A mechanism that only works on one trunk is a trunk artefact, not biology | V5 composite lock exists | Per architecture: (arch + V5 heads) − (arch control) on all-age pAUC or pAUC_histo above the V5 noise floor, Holm across the 5 non-ConvNeXt-T architectures |
| G3 | **Six-architecture in-domain ensemble** (plain in-domain members by default; with V5 features only if V6-2b runs) (§A3b, V6-3b, V6-10) | Error diversity across inductive biases; the only lever that cleared significance in Phase 1 | V6-3b OOF complete | Primary (**co-primary, owner decision 1 Oct evening**): uniform soft vote beats the V5 15-model system on folds 1–4 OOF on **both** paired all-age pAUC (above the noise floor) **and** paired Macro-F1 Δ > 0 (lesion-grouped CI lower bound > 0); <40 pAUC descriptive |
| G3 | **Full-frame eval** (no CenterCrop) sensitivity analysis | V5 audit AU8: periphery clipped at eval | — | Report only |

---

## A5. Decision table: how V5's results reshape V6 (filled in before the V6 freeze)

| V5 result | V6 consequence |
|---|---|
| S01: 384 px beats 224 px on pooled data | V6 finalists at 384 px; else 224 px *(fired: S01 failed)*. *(2 Oct)* The earlier clause "wavelet space-to-depth becomes more important" is withdrawn: 384 px moved under-40 −0.0192 and zoom −0.024; WSD stays conditional on the zoom row only |
| V5 trunk screen: **IN-22k** won | V6-2 starts from IN-22k; the V5 DINOv3 trunk screen (V4 recipe, no layer-wise decay) is not repeated — **V6's DINOv3-LLRD branch is a new fine-tuning experiment, not a duplicate trunk screen**, and it becomes a CG-DM trunk only if its own V6 screen wins (§A0.3); ConvNeXt-V2 (`fcmae_ft_in22k_in1k_384`) becomes the next same-family test |
| V5 trunk screen: **DINOv3** won (IN-22k did not) | V6-2 starts from DINOv3 and N10 (domain-adaptive FCMAE) gets priority |
| V5 trunk screen: neither won | Pretraining is not the lever at ConvNeXt-T scale; V6-12 (foundation LoRA) drops in priority |
| V5 `youngdata` passed (within-band **on the mean and in ≥ 2 of 3 seeds**; revision 2 Oct) | N5 is confirmed as a lever; V6-4 extends it (MILK10k after S84, ISIC-2020) |
| V5 `youngdata` failed or only moved all-age pAUC | Young-data expansion is an age prior, not ranking; V6-4 is dropped |
| `look` / `structure` pass | Chromophore stem and DSP are part of every V6 backbone |
| `memory` passes | Multi-proxy contrastive + subtype supervision go into the V6 core |
| `clues` beats `gem` | Any-region escalation logic kept; pAUC-DRO built on it |
| `zoom` > `zoom_random` **on the declared score and on escalation mass** (revision 2 Oct: zoom's edge over twostep is +0.0071 declared but −0.0001 on mass, `q4_verify.json` re-score) | Separate-trunk dual-stream and wavelet space-to-depth are promoted; a declared-score-only win is a readout gain and promotes nothing |
| M7 / SWAD improve LOAO | N10 and N9 go into the V6 core for G2 |
| DRE-7 sign check passes | MONET-validated concept supervision is promoted |
| V5 composite null | V6 core = pretraining (DINOv3/FCMAE) + data (N5) — the two levers V5 did not test |
| S84 (MILK10k) under-40 CI crosses 0 | The data lever (N5) is prioritised over architecture *(fired 2 Nov: V5 49/50 vs V1 47/50 at matched referral 0.52, McNemar p = 0.31, both at ceiling; global Macro-F1 +0.100 [+0.074, +0.127]; `results/v5/s84/s84_report.json`)* |
| V6-2b: V5 heads help on ≥ 3 of the 5 other architectures (Holm) | The V5 mechanism is reported as architecture-general; every V6-3b member carries the heads |
| V6-2b: V5 heads help on ConvNeXt only | Reported as trunk-specific; V6-3b members are plain in-domain controls, and the ensemble is judged without heads |
| V6-3b: ensemble beats the V5 system | The distilled V6-10 student is trained from the six-architecture ensemble |
| V6-3b: ensemble ties the V5 system | Report "diversity does not add beyond seeds × folds"; V6-10 distils the smaller system |
| V5 composite lock (any outcome) | §A0 inherits its trunk, resolution and recipe (§A0.3) |
| CG-DM Stage 0 passes / fails | Stage A runs **with the carried statistic** (D_part, D_nn or D_dict, §A0.8 selection) / the family stops by budget rule when none passes (§A0.8); R1, young data, DINOv3 continue |
| R1 Stage 0 passes / fails (ISIC-2020) | R1 trained as the N13 CG-DM variant / N13 set encoder only |
| R2 feasibility gate passes / fails | Twin added as a separate synthetic evidence channel (K = 3 real references unchanged) / R2 dropped |
| ISIC-2020 under-40 malignancy count too small to power R1 | R1 reported on all ages only; under-40 direction-only |
| V5 composite null (for the ensemble question) | V6-2b is skipped (no heads to port); V6-3b still runs as plain in-domain controls — the ensemble question stands on its own |
| V6-2b not run (demoted 1 Oct; the default) | V6-3b members are plain in-domain controls; the ConvNeXt-T member is the banked S72 seed-42 fold set; the E-primary contrast is "architecture diversity vs seeds × folds of one architecture" |
| V5 Q9 TTA: the 24-view TTA improves the V5 system on OOF (paired, above the V5 noise floor) | V6-10 applies the same TTA protocol to every member before combining (inference only); otherwise single view, stated |

**Final status (3 Nov, V5 final; the rows Part 0 still uses).** S01: failed, 224 px stays the base and 384 px is a member
only (`results/v5/s01_decision.json`). Trunk: IN-22k won. `youngdata`: passed, fragile, and is the anchor recipe (0.2).
Composite: locked as `twostep + m4 + youngdata` (`results/v5/composite_lock.json`); Gate A missed at +0.0351 [+0.0125,
+0.0596] (`results/v5/confirm_s42_s43_s44.json`). S84: fired, see its row. Q9 TTA: Macro-F1 0.6911 → 0.7005 on folds 1–4
(`results/v5/v5_oof_common_matrix.json`), applied always. SWAD on LOAO: near-neutral, BCN-only gain (0.10). Rows about
V6-2b, V6-3b, CG-DM, R1 and R2 are superseded by 0.9.

**Provisional status after the V5 Q3 read (1 Oct; final only when V5 is final — not a fill-in of
the table above).** Sources: `results/v5/screens/gate_*_in22k_*.json`, `falsifier_*_in22k.json`,
`q3_verify_rescore.json`.
- `twostep` **screen PASS** (falsifier PASS), but the gain is the escalation head's **readout**:
  on escalation mass it is +0.0025 / −0.0014 (below both bars) → §A0.6's dual-score rule.
- `m4` **screen PASS**, fragile (seed 42 negative) → the pAUC-DRO row's "V5 `m4` passed" is
  provisionally met; §A0.15 still runs pAUC-DRO only after CG-DM works.
- `memory` **FAIL** (falsifier 0/3: the 60+ melanomas' top prototype is inside the <40 top-2 in
  every seed) → the "`memory` passes" row does **not** fire: multi-proxy contrastive and subtype
  supervision stay out of the V6 core; the A0.11 ladder keeps `memory` only as a reused ablation
  rung.
- `clues` FAIL vs twostep and `gem` FAIL vs clues → the "`clues` beats `gem`" row does **not** fire.
- **Q4 (2 Oct, provisional):** `zoom` screen PASS vs clues (+0.0085, all three seeds) and falsifier
  part 1 holds (BCN/MSKCC +0.0147, HAM −0.0032); vs twostep its edge is a **readout** gain (−0.0001 on
  mass), under-40 −0.024, cost ~1.8× per run; 8r × 3 seeds runs before the lock. `youngdata` PASS on the
  frozen bars, fragile (seed-42-driven; within-band fails in seed 44). `look` / `geometry` Q4 runs are
  **void** (NaN-skipped steps, two front-token overflows, fixed 2 Oct) and are re-screened (`v5fix`).
- Pending: Q5 (look/geometry re-screen, 8r, LOAO / m7), the composite lock,
  confirmation, S84.

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
  each family. The E-primary's two co-primary endpoints (all-age pAUC, Macro-F1; 1 Oct evening)
  form an **intersection-union** test — both must pass — so they are not Holm-corrected against each
  other; the family's Holm step applies to E-primary as one hypothesis. *(2 Oct)* Two further families:
  the CG-DM primaries (clinical, mechanism; Holm, §A0.2) and the Stage 0 selection contrasts
  (Δ_part,nn, Δ_part,dict; Bonferroni intervals, §A0.8). The Stage 0 gate on each statistic is a
  compute-allocation rule, not a reported hypothesis test.

---

## A7. What Review 2 shows about V6

- **One slide:** goals G1–G3; the data allocation (MILK10k → V6 training after S84; HIBA → V6
  confirmation); the headline (CG-DM, with the Stage 0 head-to-head against nearest-patch and
  dictionary residuals); the trunk inherited from V5 (IN-22k ConvNeXt-T; DINOv3-LLRD a parallel
  branch — *2 Oct: the earlier "DINOv3 first" predates V5's trunk screen, where DINOv3 collapsed*);
  the gated phase chart from
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
- *Added 1 Oct evening (ensemble restoration, §A3b):* ISIC-2019 winner, ensembles of
  multi-resolution EfficientNets (Gessert et al., MethodsX 2020): https://arxiv.org/abs/1910.03910
- SIIM-ISIC-2020 winner, diverse-backbone EfficientNet ensemble (Ha, Liu & Liu 2020):
  https://arxiv.org/abs/2010.05351
- Diverging training methods → uncorrelated errors → larger ensemble gains (Gontijo-Lopes et al.,
  ICLR 2022): https://arxiv.org/abs/2110.12899
- Homogeneous ensembles disproportionately help minority groups (Ko et al., NeurIPS 2023):
  https://arxiv.org/abs/2303.00586
- Ensemble benefits are disparate across groups; Hardt post-processing mitigates (Schweighofer et
  al., ICML 2025): https://arxiv.org/abs/2410.13831
- Ensembling calibrated members gives an under-confident ensemble; calibrate after combining
  (Wu & Gales 2021): https://arxiv.org/abs/2101.05397
- Vessel segmentation in dermoscopy: https://pmc.ncbi.nlm.nih.gov/articles/PMC6236870/
- Pigment network directional filters: https://pubmed.ncbi.nlm.nih.gov/22829364/

---

## A9. Standing issues → V6, and what V2–V4 hand forward (full-history review, 1 Oct evening)

Owner request, pre-freeze, before any V6 result. Every issue the project has recorded as open is
listed with its evidence and the V6 phase that targets it. **"Targeted" is not "solved"**: where the
evidence says an issue is a measured limit, the expected outcome is written down now so a null is
not reported as a surprise.

### A9.1 Carried forward from V2–V4 (not in V6 before this revision)

| ID | Lever | Measured where | Why carry | Where in V6 |
|---|---|---|---|---|
| **C1** | **Bipartite (age band × escalation) RAPS conformal**, refit on the V6 system's cross-fitted OOF, plus the per-group **False Reassurance Rate** report | V2 F3: under-40 FRR 6/29 → **1/29** at α = 0.10, −0.1724 [−0.321, −0.036], Holm p 0.044 (`results/v2/conformal_subgroup_safety.csv`) — V2's strongest confirmatory positive. V2 H4 GO: marginal coverage nominal (95.07%) while FRR in powered escalating groups exceeds 1 − coverage (`results/v2/frr_by_group.csv`) | V6 had no conformal step at all; the deployed sets are still the V1-era fit. Conformal stays an **output field, never a referral layer** (S59: the conformal layer lost −0.0692 at matched workload) | V6-10 (fit), V6-11 (FRR by band, Clopper–Pearson, reported beside sensitivity) |
| **C2** | **Label-free referral-budget lock on the confirmation cohort**: the referral threshold for the contract's budget term is set from the *unlabelled* score quantile of the locked V6 system on the confirmation images, declared before the read; labels are read once, afterwards. *(Audit clarification:)* the S56 per-band thresholds keep their OOF-fitted relative structure and are moved by **one common scalar** on the uncertainty scale, solved so that total referral equals the declared budget — not new per-band offsets (registry item "per-band decision threshold offsets"). Every system compared at V6-11 gets the same treatment at the same budget | HAM-fitted budgets do not transport: S56 floor transfer 0 of 15 cells met, nominal 20% → ≈ 39% realised (`results/v4/s56/s56_report.json`); S68's train-row quantiles cut the overrun only +0.0426 (< MCID 0.05, `results/v4/s68/s68_report.json`) | Referral was one of the six terms S59 failed (0.386 vs ≤ 0.25). Budget is a property of the unlabelled score distribution, so it can be met by construction without any label; sensitivity at that budget then becomes the real test. Transductive, label-free, and declared — the same matched-referral logic S84 already uses | V6-11 |
| **C3** | **Per-band (S55) Dirichlet as a pre-declared calibration arm**, judged on per-band signed gap and ECE — never stacked with λ or per-band offsets | Global map leaves the gap **sign-flipped** across bands (<40 −0.027, 60+ +0.041, `research/stats/results_oof/band_calibration.csv`); *(2 Oct audit — corrected; the numbers were mislabelled)* per-band Dirichlet shrinks the across-band **signed-gap spread** from 0.0679 (global map) to **0.0084**, while the across-band **ECE gap** widens slightly, 0.0150 → 0.0206 (`research/multical/results_oof/gap_summary.json`); S65 showed stacking it into the policy costs workload | It is the only measured fix for the calibration-fairness issue; the barred items are the *policy* layers (λ(age), per-band offsets), not calibration | V6-10 |
| **C4** | **Eval-transform parity test** for every new trunk (timm checkpoints default to **bicubic**; the deployed eval transform is **bilinear**) | S58: a bicubic extraction sat 22% (relative L2) from the deployed features and cost 0.021 HAM-val Macro-F1 (0.748 → 0.727) before it was caught (CHANGELOG, S58 entry, "Transform defect caught before the freeze") | V6 adds four timm trunks (DINOv3, ConvNeXt-V2, MaxViT, EfficientNetV2) and five torchvision ones | V6-0 unit test: each trunk's eval transform is the repo's, asserted before any run |
| **C5** | **S61 drift hooks wired to the V6 system** (archive probe, Mahalanobis, per-band coverage, score-curve), baseline recomputed on the V6 OOF | `research/v4/s61_drift.py`, `results/v4/s61/drift_baseline.json` (V1 baseline only) | Deployment requirement for G3; costs CPU only | V6-10 |
| **C6** | **Read the S48 endpoint in its declared form** at V6-11: under-40 escalation sensitivity at matched referral, **lesion unit**, MCID 0.10 | V4 final audit: the S48 endpoint was never read exactly as declared (S54 gated on pAUC + Macro-F1; S56/S59 read image-unit sensitivity) | Closes a recorded logic caveat | V6-11 |
| **C7** | **24-view TTA** (V1 rung A6, +0.014 Macro-F1) | `results/ablation_table.csv` | Carried earlier tonight (§A5 TTA row) | V6-10 |
| **C8** | **In-domain six-architecture ensemble**, co-primary pAUC + Macro-F1 | §A3b | Carried earlier tonight | V6-3b, V6-10 |

**Reviewed and deliberately not carried** (each has a measured reason; see
`results/v5/v5_no_repeat_registry.json` and the CHANGELOG): V2 N1–N5 heads and Track B (sign-flipped
between val and BCN, S37); uncertainty-score rescue (F4, no member survives Holm; 24 of 29 under-40
misses rescued by nothing); frozen mass-threshold transport (V2 H3 NO-GO, oracle-only); V3 archive
pooling as a gain claim (H4/H5 falsified — V4 then showed in-domain training lifts Macro-F1 +0.18,
which V5/V6 already do); adversarial age removal (V3 H6, under-40 AUC −0.1049); frozen specialist
heads (V3 H1/H2, S67); S58 pooled head on frozen V1 features (+0.107 Macro-F1 on reserved, but it
recovers only ~63% of full in-domain retraining, which V6 does); routers and the pooled Mahalanobis
gate (S58; the S69 modality classifier replaces it); λ(age), per-band offsets, combined policies
(S57b/S65/S66); 384 px as a default (S53r's HAM-val lever did not replicate on the pooled corpus,
V5 S01); metadata fusion (registry).

### A9.2 Standing issues and their V6 target

| # | Standing issue (evidence) | V6 target | Success means | Honest expectation |
|---|---|---|---|---|
| 1 | **Under-40 escalation ranking ceiling.** Every V4 model 0.711–0.750 under-40 pAUC on reserved, V1 0.7335; flat across backbones (S51), recipes (S54), heads (S58, S67), decision rules (S64) | §A0 CG-DM, R1, V6-4 young data (+ MILK10k / ISIC-2020 after their gates), DINOv3-LLRD | Gate A on folds 1–4: point Δ <40 pAUC ≥ +0.050 and CI > 0 | **Not expected** — V4's largest move was 0.011 and the lesion-bootstrap SD is 0.033 (V5 runsheet §6). V6 tests the levers V4 never had (new data, counterexample mechanism); a null is a reportable boundary |
| 2 | **S59 contract fails every term** on reserved (<40 sens 0.763 vs 0.855; referral 0.386 vs 0.25; retained Macro-F1 0.465 vs 0.888) | V6-11 with the V6 system + C2 + C6 | Contract terms read on the fresh cohort with the budget locked label-free | Referral term met by construction (C2). Sensitivity floors depend on issue 1. The 0.888 retained-Macro-F1 target was set in the HAM era; reported as is, never relaxed after the read |
| 3 | **Referral budget does not transport** (20% → ≈ 39%; 0/15 floors) | C2 | Realised referral = declared budget ± sampling error | Expected to pass (construction) |
| 4 | **Cross-domain Macro-F1** (V1 0.411 on reserved; in-domain V4 0.57–0.62) | C8 ensemble (Macro-F1 co-primary), C7 TTA, MILK10k training, V6-7 SWAD / N11 on LOAO | E-primary passes; LOAO Macro-F1 above the noise floor | Plausible: ensembling is the one lever with a significant Macro-F1 record (Phase 1), never tested in domain |
| 5 | **Calibration sign-flips across age bands**; the soft-vote ensemble is under-confident | C3; Dirichlet after combining (§A3b) | Per-band signed gaps share a sign and shrink | Spread shrink expected (S55 on OOF: signed-gap spread 0.0679 → 0.0084); the across-band ECE gap may widen slightly (0.0150 → 0.0206) — 2 Oct correction |
| 6 | **Conformal coverage ≠ protection** (V2 H4); deployed sets under-cover on reserved (0.854 / 0.825 / 0.779 by band vs 0.95, `results/v4/s59/s59_report.json`) | C1 | Under-40 FRR at or below the V5 system's, coverage audited per band | Plausible (V2 F3 replicated on the V6 OOF), small n — Clopper–Pearson, stated |
| 7 | **No evidence for Fitzpatrick V/VI**; ITA invalid on dermoscopy; PAD covers I–IV only (V 8 / VI 1, S8b) | V6-0: check whether HIBA / MILK10k / ISIC-2020 carry `fitzpatrick_skin_type` (metadata only); report every powered group (≥ 30), I–IV spread separate from "unknown" (S8b lesson) | A powered slice exists and is reported | **Cannot be solved by V6 alone**: it needs dermoscopy with V/VI labels, which no planned source is known to provide. The model card keeps the mandatory notice until such data exists |
| 8 | **Smartphone images out of scope** | S69 modality gate kept (AUROC 0.9994, `results/v4/s69/s69_report.json`); V6-9 N12 clinical-photo expert | Patient-grouped PAD Macro-F1 above the 0.760 reference | Conditional phase |
| 9 | **Under-40 data scarcity** (81 escalating lesions in V4 train) | V5 `youngdata`, V6-4 curve, MILK10k (after S84), ISIC-2020 | Learning-curve slope > 0 | Depends on the V5 `youngdata` screen |
| 10 | **Evidence from one architecture only** | V6-3b (diversity) now; V6-2b portability if budget remains | E-primary; V6-2b Holm | Partly addressed |
| 11 | **Deployment**: S60 serves a softmax pass-through by default; drift hooks not monitoring a live model | V6-10 distillation → one model behind S60 + `explain`; C5 | Distilled model within 0.01 of the ensemble; hooks computed on the V6 baseline | Engineering, expected |
| 12 | **Exhausted cohorts**: reserved read 8×, HAM test receipt 2 | V6-11 single receipted read on HIBA (fallback ISIC-2020 patient-held-out) | One read, power stated before the lock | — |
| 13 | **S48 endpoint never read as declared** | C6 | Read at V6-11, lesion unit | — |

---

## A10. Headline benchmark, pool design and the accumulation principle (revision 3 Oct, owner request — input to the post-Review-2 brainstorm, NOT decided)

**Why this section exists.** The owner asked (3 Oct) for V6 to (1) target the best published
balanced accuracy on a like-for-like protocol, (2) bring back ensembling with pooled training on
the right data, (3) use V5 to design that pool so under-40 sensitivity improves, and (4) be the
accumulation of the models and techniques that proved helpful across V1–V5, with novelty in how
the issues are fixed. The full redesign happens in the combined brainstorm **after V5 is final and
after Review 2**. This section records the research done on 3 Oct so the brainstorm starts from
evidence. It changes no frozen V5 item, no gate and no other section of this file yet.

### A10.1 Same footing first (`results/v5/same_footing.md`, `research/v5/same_footing.py`)
The project's numbers come from three rulers. On the same rows:
- **HAM10000 test:** V1 deployed (6-CNN + TTA + Dirichlet) Macro-F1 **0.805**, BA 0.794; with 20%
  abstention 0.896 on the kept 78% only. Under-40 escalation sensitivity 0.143 (3/21).
- **Reserved BCN20000 + MSKCC (unseen archives):** the same V1 system **0.411**; V4 pooled single
  model 0.603 [0.588, 0.611]. Pooling is worth **+0.19** Macro-F1 off-distribution — the largest
  gain the programme has measured.
- **Pooled fold 0 (development):** single models 0.634–0.664 Macro-F1, BA 0.637–0.651; 3-seed
  soft-votes 0.658–0.682 Macro-F1, BA 0.644–0.672. Under-40 escalation sensitivity 0.62–0.69
  (n = 35 escalating images; descriptive).

### A10.2 The published anchor (verified 3 Oct)
| Source | Protocol | Balanced accuracy |
|---|---|---|
| Gessert et al., ISIC 2019 winner (arXiv 1910.03910; MethodsX 2020) | **5-fold CV on the ISIC 2019 training set, folds grouped by lesion**, 8 classes, + 2,334 external training images, best-epoch checkpoints | single models, images only **65.3–68.8** (EN-B0 224 px 65.8 ± 1.7; EN-B6 528 px 68.8 ± 0.7); ensemble average of 16 configurations **71.7 ± 1.7**; optimal subset (selected on the same CV) 72.5 ± 1.7; + metadata 74.2 ± 1.1 |
| Same team, official ISIC 2019 test (9 classes incl. unknown) | hidden test, one submission | **63.6** (task 1) |
| ISIC 2019 leaderboard, task 1 top five | hidden test; all used external data | 63.6 / 60.7 / 59.3 / 57.8 / 56.9 |
| Combalia et al., Lancet Digit Health 2022 (challenge analysis) | top algorithm, by test-image source | 82.0 on HAM-sourced vs **58.8** on new BCN images (−23.2 points) — *from the abstract via search; full text not retrieved (403); verify before citing* |

Read: **at 224 px our single models (BA 0.637–0.651) sit at the winner's single-model level
(EN-B0 224 px 65.8 ± 1.7). The gap to their 71.7 is the ensemble**: 16 configurations (SENet154,
two ResNeXt-WSL, EfficientNet B0–B6 at 224–528 px, two crop strategies) added ≈ +3 to +6 points over their single
models. Combalia's 82 → 59 drop is the published twin of our own 0.805 → 0.411.
**Not like-for-like yet:** (a) our models predict the **7-class collapse** (AK + SCC → `akiec`);
Gessert's BA is over 8 classes, and merging the hardest pair inflates ours; (b) they select best
checkpoints on the CV fold, we read the last epoch; (c) they add external data; (d) our development
partition is 15,294 de-duplicated images (reserved + HAM val/test removed), theirs 25,331.
**Brainstorm item:** train and report **8-class** BA (the corpus already carries `class_8`;
`ml/configs/class_mapping.json` declares SCC as the planned 8th class) so the headline is
comparable.

**Candidate headline target (for the brainstorm):** *on lesion-grouped 5-fold CV of the pooled
ISIC 2019 corpus, images only, reach the ISIC 2019 winner's CV ensemble (8-class BA ≥ 0.717) with a
pre-declared ensemble, while raising under-40 escalation sensitivity and reporting the external
(reserved / HIBA) drop honestly.* The novelty is not the BA number; it is parity with the winner
**plus** a measured fix for the under-40 failure, explained by pool design.

Context found the same day: an AI-for-young-patients melanoma protocol (ClinicalTrials.gov
NCT06621810) states that available algorithms were trained mostly on melanomas from patients over
60; a dermoscopy-by-age study (PMC12346493) reports growth-related structures (atypical globules,
pseudopods) under 40 vs regression-related features at ≥ 40; a Sept 2026 preprint (arXiv
2609.02111) finds disease-distribution shift outweighs skin tone in the dermatology generalisation
gap (BA 0.62 → 0.21). Foundation models (PanDerm, Nat Med 2025) are the obvious trunk comparison;
their dermoscopy BA protocol was not verified on 3 Oct.

### A10.3 What V5 says about the pool (provisional — V5 is not final; `results/v5/pool_profile.json`)
- **The development pool has 81 under-40 escalating lesions** (BCN 35, HAM 34, MSKCC 12) against
  **1,435 at 60+**. Escalating prevalence under 40 is 2.9–11.9% by archive; at 60+ it is 31–75%
  (BCN 60+ = 74.7%). A model can learn "older → escalate" from the image without any metadata input.
- **Young data is the strongest under-40 lever measured in V5:** the 2,078 extra images add **291
  escalating lesions (236 melanoma)** — 3.6× the in-corpus count — and moved under-40 pAUC by
  +0.0225 with Macro-F1 retained. But stacked on m4 it removed ≈ 60% of m4's all-age ranking gain
  (Q6). The folds 1–4 read (Q7 + D1) decides whether that trade is real.
- **The public young histopathology pool is nearly tapped:** 5,790 ISIC images at age ≤ 35, 2,562
  already in the corpus, 2,353 candidates, 2,078 used; **747 candidates lack a lesion ID** (usable
  only with image-level grouping and the S49 dedupe).
- **BCN is the hardest archive and holds 43% of the young escalating lesions:** LOAO with BCN held
  out, BA 0.391 (HAM held out 0.519; MSKCC 0.577 on its 3 classes).
- Pool levers for the brainstorm, cheapest first: (a) MILK10k young rows after S84 (358 at age
  ≤ 35, histopathology), already allocated to V6 training in §A2; (b) the 747 no-lesion-ID
  candidates; (c) **band-balanced sampling** that equalises escalating prevalence across age bands
  (attacks the age prior without feeding age to the model); (d) young benign rows confirmed by
  consensus / follow-up, for under-40 specificity; (e) ISIC-2020 benign diversity (§A2).

### A10.4 The accumulation principle — candidate V6 system
Every component below has a measured gain; nothing else is carried by default.
| Component | Measured evidence | Source |
|---|---|---|
| Pooled multi-archive training | +0.19 Macro-F1 off-distribution (0.411 → 0.603) | `results/v4/s54/s54_marginals.csv` |
| IN-22k trunk | +0.028 Macro-F1, fold 0 (0.634 → 0.662) | `results/v5/same_footing.md`; trunk gate |
| Ensembling | 3-seed vote +0.008 to +0.025 Macro-F1 (fold 0); 6-CNN vote +0.026 (HAM); winner's CV ensemble ≈ +3 to +6 BA | same-footing; `results/ablation_table.csv`; Gessert Table 1 |
| 24-view TTA | +0.014 Macro-F1 (HAM, A5 → A6) | `results/ablation_table.csv` |
| Dirichlet calibration | +0.019 Macro-F1 (HAM, A6 → A7) **but** escalation sensitivity 0.786 → 0.731 | same |
| Young histopathology data | under-40 pAUC +0.0225, F1 retained | `results/v5/screens/gate_youngdata_in22k_vs_control.json` |
| twostep / m4 escalation heads | all-age pAUC +0.0155 / +0.0195 (fold 0; held-out read pending Q7/Q7b) | `results/v5/screens/gate_*` |
| S56 selective abstention | the deployed safety layer (λ and conformal lose at matched workload, S65) | `docs/RESEARCH_LOG.md` |
Not carried (failed or falsified): look, geometry, zoom, clues, gem, memory, m7, structure, m5,
logic, the λ rule, conformal as a deployed layer.

### A10.5 Questions the brainstorm must settle
1. **Headline:** does CG-DM (§A0) stay the headline, or become one screened arm inside the
   accumulation system with pool design as the headline? V5's record matters here: ten image-feature
   modules failed or were falsified, and the only large gains came from data and ensembling.
2. **8-class** training and reporting for comparability (A10.2).
3. **Ensemble diversity:** architectures and resolutions (the winner used 224–528 px; our S01 found
   384 px no better for a *single* model, which does not settle its value as ensemble diversity).
   Every new architecture or resolution needs a `scripts/gpu_benchmark.py` measurement before it is
   costed.
4. **Pool design experiments** (A10.3 levers), each with a declared under-40 and all-age endpoint.
5. **Compute:** N configurations × 5 folds at the measured anchors, quoted only after benchmarks.

---

## A11. Ensemble candidate roster, technique stack and selection protocol (research 3 Oct, owner request — brainstorm input, NOT decided)

**Goal (owner, 3 Oct):** test a wide, deliberately diverse set of models and techniques in V6, then
choose the best ensemble, the best techniques and the best pooled data **on out-of-fold evidence
only**. This section is the research behind that. §A10.5's questions are still open; nothing here is
frozen, and every GPU cost below is **unmeasured** until `scripts/gpu_benchmark.py --timm` has run.

### A11.1 What the evidence says an ensemble needs
1. **Diversity of training method matters more than diversity of architecture name.** Models
   pretrained differently (supervised, masked-image, self-distillation, image–text) make less
   correlated errors and ensemble better (Gontijo-Lopes et al., ICLR 2022). The roster below is
   therefore built across **pretraining families**, not only backbones.
2. **Every recent ISIC winner was a large, diverse ensemble with a training trick in the target:**
   - ISIC 2019 (Gessert et al.): 16 configurations of SENet154, ResNeXt-WSL and EfficientNet B0–B6
     at 224–528 px, two crop strategies; ensemble average +2.9 BA over the best single model.
   - SIIM-ISIC 2020 (Ha et al., arXiv 2010.05351): diverse EfficientNets at several input sizes;
     the stated keys are a stable validation scheme, **a fine-grained training target (9 diagnosis
     classes, read out as melanoma)** and very diverse members. Metadata models scored worse alone
     but added diversity.
   - ISIC 2024 (npj Digit Med 2025; arXiv 2506.03420): two **EVA-02** models and one **EdgeNeXt**,
     fused by gradient-boosted trees. Stable-Diffusion synthetic malignant lesions added
     +0.012 / +0.014 pAUC.
3. **This repo's own lessons bind the protocol:** SwinV2 + MaxViT added to the six CNNs was a wash
   on HAM, and Caruana selection over 8 members **overfitted** a 1.5k validation set (CHANGELOG
   §13). So members are chosen on the **15,294-row 5-fold OOF**, never on a small validation set,
   and uniform soft-voting is the default the selector must beat.
4. **Under-40 is not a hopeless band for diverse ensembles:** in a prospective multicentre study, an
   open-source ISIC-trained ensemble (ADAE) reached BA **0.890 in patients under 35** vs 0.767 for
   dermatologists (Heinlein et al., Commun Med 2024). That is a binary melanoma task on another
   population, so it is motivation, not a target.
5. **Leakage is the main risk of "out-of-the-box" weights.** Any public model trained **with labels
   on ISIC 2019/2020** (ADAE, Kaggle solution weights) has seen our reserved and HAM-test labels —
   **excluded as a member**. Dermatology foundation models pretrained **without labels** on archive
   images (PanDerm, DermFM-Zero, DermLIP, MONET, Derm Foundation, MedSigLIP) need a corpus audit
   against reserved, HIBA and MILK10k before any external number is reported (already required by
   §A3 for PanDerm / MedSigLIP).

### A11.2 Candidate roster (parameters and licences read from timm 1.0.28 on CPU, 3 Oct)
**Tier 1 — cheap, open licence, high diversity (full fine-tune expected to fit; VRAM unmeasured):**
| Candidate (timm tag) | Params | Pretraining family | Why | Licence |
|---|---:|---|---|---|
| `convnext_tiny.fb_in22k_ft_in1k` (anchor) | 27.8M | supervised IN-22k, CNN | V5 trunk; +0.028 Macro-F1 over IN-1k | Apache-2.0 |
| `eva02_small_patch14_224.mim_in22k` | 21.6M | masked-image modelling, ViT | ISIC 2024 winner's family; least like a supervised CNN | MIT |
| `tf_efficientnetv2_s.in21k_ft_in1k` | 20.2M | supervised IN-21k, CNN (native 300 px) | EfficientNet family won ISIC 2019 and 2020 | Apache-2.0 |
| `caformer_s18.sail_in22k_ft_in1k` | 24.3M | supervised IN-22k, conv + attention MetaFormer | Hybrid inductive bias | Apache-2.0 |
| `vit_small_patch14_reg4_dinov2.lvd142m` | 22.1M | self-distillation, ViT | SSL family. Prior: *frozen* DINOv2 lost under 40 (S51); test **fine-tuned with LP-FT** | Apache-2.0 |
| `seresnext50_32x4d.racm_in1k` | 25.5M | supervised IN-1k, classic CNN | Gessert's SENet/ResNeXt lineage; old-style errors | Apache-2.0 |
| `edgenext_small.usi_in1k` | 5.3M | distilled conv–attention hybrid | ISIC 2024 winner's family; very cheap | MIT |
| `maxvit_tiny_tf_224.in1k` | 30.4M | supervised, multi-axis attention | Already §A3 priority 3 | Apache-2.0 |

**Tier 2 — larger or licence-restricted (benchmark before costing):**
`eva02_base_patch14_224.mim_in22k` (85.8M, MIT); `vit_base_patch16_siglip_224.v2_webli` (92.9M,
Apache-2.0; image–text family); `vit_base_patch16_dinov3.lvd1689m` (85.6M) and
`convnext_small.dinov3_lvd1689m` (49.5M) (DINOv3 licence to verify); `convnextv2_tiny.fcmae_ft_in22k_in1k`
(27.9M; **CC-BY-NC** — academic use only, already §A3 priority 2); `swinv2_tiny_window8_256` (27.6M, MIT);
`hiera_small_224.mae_in1k_ft_in1k` (34.2M, **CC-BY-NC**).

**Tier 3 — dermatology / medical foundation models (frozen-embedding probe first, then LP-FT or LoRA):**
| Model | Weights / licence | Evidence |
|---|---|---|
| PanDerm-Base ViT-B/16 and PanDerm ViT-L/16 (Nat Med 2025) | released; **CC-BY-NC-4.0** | Linear probe ≈ full fine-tune on HAM10000 (paper); PanDerm-MLP ≈ fine-tuned Swin, and their fusion beat both on HAM10000/MSKCC (arXiv 2505.16338) |
| DermFM-Zero (PanDerm-2, Feb 2026; arXiv 2602.10624) | repo released; licence to verify | Image–text FM on > 4M images; zero-shot SOTA on 20 benchmarks (authors) |
| DermLIP (Derm1M, ICCV 2025) | Hugging Face; licence to verify | CLIP on 1.03M dermatology image–text pairs |
| MedSigLIP-448 (Google HAI-DEF, Jul 2025) | HAI-DEF terms | 400M dual tower; dermatology linear probe AUC 0.881 (79 classes) |
| MONET (Nat Med 2024), Derm Foundation (Google) | open / HAI-DEF | Frozen-embedding benchmark on DERM12345 40-class (arXiv 2601.12382): MedSigLIP 69.8, Derm Foundation 69.5, MONET 69.3, DINOv2-G 68.0, PanDerm-B 64.1, **PanDerm-L 36.7** weighted F1 — results are protocol-sensitive |
Protocol: one frozen feature pass per image (GPU minutes, then CPU heads on the S71 folds). Only a
model whose probe is competitive with the fine-tuned anchor, **or** whose OOF errors are least
correlated with it, gets LP-FT (Kumar et al., ICLR 2022: linear-probe-then-fine-tune preserves
out-of-distribution features) or LoRA with gradient checkpointing on the 8.55 GB card.

**Not prioritised:** MambaOut, TinyViT, FastViT, EfficientViT, RepViT (no dermoscopy evidence; ImageNet
efficiency is not our bottleneck); full fine-tuning of any ViT-L (VRAM).

### A11.3 Technique stack — candidates, evidence and prior
| Stage | Technique | Evidence | Repo prior / gate |
|---|---|---|---|
| Target | **Fine-grained target, collapsed at read-out** (ISIC `diagnosis_1…5` hierarchy, metadata-only fetch) | Ha et al. 2020 key #2 | Untested here; one ConvNeXt-T fold-0 screen |
| Target | **8-class** (SCC separate) | Comparability with Gessert (§A10.2) | Corpus already has `class_8` |
| Training | LP-FT for pretrained FM / SSL trunks | Kumar et al., ICLR 2022 | Tier 2–3 members only |
| Training | Layer-wise LR decay (ViTs) | Standard ViT fine-tuning practice | ViT members |
| Training | EMA / SWA / **SWAD** | Cha et al., NeurIPS 2021 | V5 SWAD secondary positive on LOAO |
| Training | Mixup / CutMix | Standard | Unmeasured here; fold-0 screen |
| Loss | Logit adjustment / balanced softmax (Menon et al., ICLR 2021) | Long-tail theory | **Negative prior on loss swaps** (LDAM-DRW 0.7256, ASL 0.7305 < CE 0.7459, HAM) → post-hoc logit adjustment on OOF only (CPU) |
| Resolution | Multi-resolution members (224 / 288 / 384); FixRes test-time size (Touvron et al., NeurIPS 2019) | Gessert 224–528 px | S01 null is single-model only |
| Data | §A10.3 pool levers; **band-balanced sampling** | — | Primary V6 lever per §A10 |
| Data | **Diffusion-synthesised young escalating lesions**, train-only | Ktena et al., Nat Med 2024 (accuracy in under-represented groups improved, esp. out of distribution); ISIC 2024 +0.012–0.014 pAUC | **Memorisation risk** (latent diffusion memorises training images, Nat Biomed Eng 2025): nearest-neighbour audit, never in evaluation |
| Ensemble | Uniform soft-vote (default) vs rank-average vs **bagged Caruana selection with replacement** (Caruana et al. 2004, 2006) vs ridge stacking (nested) | Repo: Caruana overfit on 1.5k val | Selection on 15k OOF; uniform wins ties |
| Ensemble | Weight-space soups / WiSE-FT for same-init members (Wortsman et al., ICML 2022 / CVPR 2022) | One model's cost at inference | Same-init members only |
| Post-hoc | Dirichlet on cross-fitted OOF **after** combining; 24-view TTA | +0.019 / +0.014 Macro-F1 (HAM) | Dirichlet cost escalation sensitivity 0.786 → 0.731 on HAM → read both |
| Deploy | Distillation into one student (Beyer et al., CVPR 2022) | §A4 G3 | Student within 0.01 of the ensemble |
| Safety | S56 selective abstention | Deployed V4 layer | Carried |
Considered, prior against: last-layer group re-balancing (DFR, Kirichenko et al., ICLR 2023) —
V4 closed under-40 as a representation limit, not a head or weighting problem (S51–S67).

### A11.4 Selection protocol (pre-declared before any V6 result)
1. **Benchmark** every Tier 1–2 candidate: `scripts/gpu_benchmark.py --timm --arch <tag>
   --sizes 224 [288 384] --batch 32 --images 15294 --json-out results/v6/benchmarks/<tag>_<size>.json`
   (full fine-tune step, AMP, 2 workers projected). Peak > 7.5 GB at batch 32 → batch 16 × 2
   accumulation, or `--grad-checkpointing`; cost quoted only from these files.
2. **Fold-0 screen, one seed:** keep a candidate if its single-model BA is within the V5 noise floor
   of the anchor **or** adding it to the anchor's soft-vote raises fold-0 BA (marginal
   contribution). Fold 0 is the flagged selection fold, as in V5.
3. **Finalists complete folds 1–4** on the declared pool (§A10), lesion-grouped S71 folds.
4. **Ensemble selection on the full 5-fold OOF (15,294 rows):** bagged forward selection with
   replacement (20 bags) on **8-class BA**, subject to constraints — Macro-F1 retained, all-age
   escalation pAUC retained, **under-40 escalation pAUC not lowered beyond the noise floor**. The
   selected subset must beat uniform-over-finalists by more than its lesion-bootstrap SD, or uniform
   is used.
5. **Headline read:** 8-class BA on the 5-fold OOF vs Gessert's ensemble average 71.7 (protocol
   differences stated: best-epoch vs last-epoch, external data, de-duplicated partition). External:
   the V6 confirmation cohort (HIBA) single read; the **MILK10k Benchmark** (ISIC 2025, blind 479-lesion
   test, live leaderboard, 11-class Macro-F1) is a possible extra blind read — its label space does
   not match ours (3 extra categories), so it is optional and descriptive.

---

## A12. The ADAE angle — a leak-free in-domain teacher ensemble distilled into one model (revision 3 Oct, owner request: "pay heavy attention" — brainstorm input, priority candidate, NOT decided)

**Origin.** An external research pass (pasted by the owner, 3 Oct) proposed rebuilding V6 around
**ADAE** ("All Data Are Ext", the SIIM-ISIC 2020 winning ensemble; Ha et al., arXiv 2010.05351) and
its prospective validation (Heinlein et al., Commun Med 2024, doi 10.1038/s43856-024-00598-5,
PMC11387610). Every claim was checked on 3 Oct against the papers, the solution repository and this
repo's `results/` before anything was adopted. The **recipe** is adopted below. The **public
weights** cannot be used, for the reason in A12.2.

### A12.1 What was verified
| Claim in the proposal | Verified value | Status |
|---|---|---|
| ADAE = 18 CNNs × 5 folds = 90 models: 16 EfficientNet B3–B7, 1 SE-ResNeXt-101, 1 ResNeSt-101; 4 of 18 use metadata (sex, age, site) | as stated (Heinlein et al.) | **read** |
| Training data: SIIM-ISIC 2020 + 2019 (+2018), 58,457 lesions, 5,106 melanomas | as stated; the solution repo trains on "2020 and 2019 data (including 2018)" | **read** |
| Input sizes 384–768 px | solution repo: image sizes **384–896**, data resized to 512 / 768 / 1024 | **read (corrected)** |
| Under 35: BA 0.890 [0.859, 0.920], AUROC 0.974 [0.942, 0.997]; dermatologists 0.767 | as stated — but on **177 lesions with 14 melanomas**; ADAE sensitivity **1.000 (14/14; exact 95% CI [0.768, 1.000])**, specificity **0.779**; dermatologists 0.571 / 0.963 | **read; small n** |
| Dermatologists "missed over 42%" of young melanomas | 6 of 14 | read; small n |
| Test inputs | **six real dermoscopic photos per lesion** (varied angle, position, polarisation mode), outputs aggregated ("R-TTA") | **read — not reproducible from our single-image data** |
| Operating point | threshold pre-set so sensitivity exceeds 85%; overall sensitivity 0.922 | read |
| "Image-only vs metadata ablation, AUROC 0.971 vs 0.974 under 35 (Suppl. Figs 6–7)" | not in the main text; the supplement was not retrieved | **unverified — do not cite** |
| Weights public | `github.com/ISIC-Research/ADAE`; solution repo MIT, weights on Kaggle | read |
| "0.855 under-40 sensitivity at ≤ 25% referral is mathematically defensible; ≥ 0.88 sens ⇒ ≤ 15% referral" | ADAE's own under-35 point: 14 TP + 0.221 × 163 ≈ 36 FP → **≈ 28% referral** | **incorrect** — even ADAE's young result does not meet ≤ 25% referral; S70 stays as declared |
| "IN-22k control under-40 pAUC 0.782" | 0.742 / 0.779 / 0.804, mean **0.775** (`results/v5/screens/gate_control_in22k_vs_control.json`) | **corrected** |
| "Unfrozen DINOv3 collapsed Macro-F1 to 0.38–0.44" | 0.445 / 0.404 / 0.383 (`gate_control_dinov3_vs_control.json`) | **verified** (V5 trunk screen, not LLRD) |
| "V5 escalation heads lost −0.067" | that is **V4** H1 / S40 (N2 head out-of-sample, Δ −0.0670, CI [−0.245, +0.072]); V5's twostep / m4 **passed** | **misattributed** |
| GPU times (2.5 / 3.5 / 5.5 / 6.0 / 11.0 h; 55 / 72 / 85 min; 4.1 / 6.4 / 6.8 GB) | none measured on this machine | **unmeasured guesses — replaced by §A11.4 benchmarks** |

### A12.2 Why the public ADAE weights are excluded
ADAE was trained **with labels** on ISIC 2019, which is our entire corpus: every development fold,
the reserved BCN/MSKCC partition and the HAM test rows. So:
- a "zero-shot ADAE probe on fold 0" would score models on images whose labels they were trained
  on — an in-sample number, not a transfer test;
- distilling from those weights would carry reserved / test labels into our student.
This is the §A11.1 leakage rule, and it applies to any public ISIC-trained checkpoint. **The recipe is
reproducible; the checkpoints are not usable.**

### A12.3 Adopted design (leak-free): the in-domain ADAE-recipe teacher
| Element | ADAE | V6 adoption | Why / constraint |
|---|---|---|---|
| Data scale | 58k lesions, 2020 + 2019 + 2018 | S71 development pool + young extra + MILK10k (after S84) + **ISIC-2020 training rows** (patient-grouped, deduplicated; its patient-held-out subset stays reserved as the confirmation fallback, §A2) | ISIC-2020 brings the large young/middle-aged benign-nevus population our pool lacks (§A10.3). **Constraint to verify at V6-0:** most ISIC-2020 benign rows carry no specific diagnosis, so they fit a **binary escalation / melanoma auxiliary head**, not the 7/8-class head (Ha et al. mapped them to an "unknown" target) |
| Members | EfficientNet B3–B7, SE-ResNeXt-101, ResNeSt-101 | §A11 roster **plus** `tf_efficientnet_b3/b4/b5.ns_jft_in1k` (10.7 / 17.6 / 28.4M; native 300 / 380 / 456 px), `seresnext101_32x4d.gluon_in1k` (46.9M), `resnest101e.in1k` (46.2M) — all Apache-2.0, all resolve in timm 1.0.28 | ADAE's diversity came largely from size and depth variation within one family plus two different families |
| Resolution | 384–896 | members at **384 and up to 456** where VRAM allows (benchmark; gradient checkpointing or batch 16 × 2) | S01's 384 px null is single-model only; the ADAE evidence is ensemble-level |
| Target | 9 diagnosis targets, melanoma read out | fine-grained target (§A11.3) + 8-class head + binary melanoma / escalation auxiliary head | Ha et al.'s key #2 |
| Metadata | 4 of 18 models | **none** (image-only; no age, sex or site input) | V6 rule. ADAE's metadata contribution is unquantified in the paper, so our image-only system is a conservative replication |
| Test-time | six real photos per lesion | digital TTA (24-view, V1) | We have one image per lesion. ADAE's under-35 result includes R-TTA, so it is **not** an expected value for us |

### A12.4 Distillation into one deployable student — cross-fitted so no held-out label leaks
The student (ConvNeXt-T or the best single §A11 member, 7/8-class) learns from cross-entropy plus a
KL term toward the teacher ensemble's softened class distribution (Hinton et al. 2015; "consistent"
teacher views, Beyer et al., CVPR 2022).

**Leakage trap and the fix.** The naive target — the teacher's out-of-fold prediction for each
training row — leaks. A row in fold *j* is predicted by teacher models trained on every fold except
*j*, which includes the student's held-out fold *k*. The declared rule is:
**the student for fold *k* is distilled only from the fold-*k* teacher models**, which trained on
exactly the student's training rows. Teacher targets are computed on the same augmented view the
student sees (online), or cached over several augmented views. The alternative, nested 4-fold teachers
per outer fold, is costed only if the online teacher does not fit in VRAM.

### A12.5 Gates and endpoints
- **Teacher gate (fold 0 screen, then folds 1–4):** the teacher ensemble against the ConvNeXt-T
  IN-22k control, on the five-fold OOF where the power is (81 under-40 escalating lesions in
  development, not fold 0's 35): **all-age escalation pAUC** above the V5 noise bar **and** 8-class BA
  ahead. Under-40 escalation pAUC and sensitivity are the declared **G1 readout**, with lesion-grouped
  intervals. They are not a pass condition while the under-40 count stays this small.
- **Student gate:** student within **0.01 BA** and within the noise floor on all-age and under-40
  pAUC of its teacher (§A4 G3); any gain visible only on a private head and not on escalation mass is
  discarded (the V5 readout-artefact lesson).
- **S70 contract** (0.855 under-40 sensitivity at ≤ 25% referral) is unchanged and reported
  honestly; A12.1 shows that even ADAE's young subset sits near 28% referral.

### A12.6 What is not decided here (post-Review-2 brainstorm)
- Whether A12 **replaces CG-DM** as the headline (the proposal's recommendation) or runs beside it.
  The evidence favours data + diverse ensembles + distillation (§A10, §A11). CG-DM's Stage 0 is cheap
  CPU work, so it can still run as a screen.
- The proposal's cuts (R2 pseudo-benign twin; DINOv3-LLRD). The unfrozen DINOv3 collapse (A12.1)
  supports deferring DINOv3-LLRD; R2 is already conditional (§A0.13).
- Compute: **unmeasured** until `scripts/run_v6_benchmarks.ps1` (which now includes the ADAE-family
  members at 384 / 456 px) has produced `results/v6/benchmarks/`.

---

## A13. Resolution and module-fairness study (RMF) — revision 4 Oct, owner request (brainstorm input, priority mechanism, NOT decided)

**Why.** The V5 audit (`results/v5/fairness_audit.md`) found three handicaps against the V5 modules (no tuning, the
backbone's learning rate for new modules, 224 px), and the investigation `results/v5/resolution_investigation.md`
found the 224-vs-384 null was a **measurement and dilution** problem, not a pipeline bug:
- resolution helps HAM-type images (V4 S53r +0.0297 Macro-F1, 3/3 seeds; V5 S01 HAM rows +0.015, 3/3 seeds);
- BCN rows (45% of the pool) show no gain, which dilutes the pooled estimate to +0.004;
- S01 had one fold, 3 seeds, a last-epoch endpoint with ≈ 0.006 epoch-to-epoch noise, and 224-pretrained weights.
So V6 tests resolution and the modules **properly before** building on either.

### A13.0 Free check first (inference only, after Q8b)
Re-score existing 224-trained `_last` checkpoints (folds 1–4) at test sizes 256 / 288 / 320 (FixRes, Touvron et al.
2019: a model trained with random-resized crops sees objects larger at train time than at a centre-crop test). Zero
training cost; minutes of GPU per checkpoint (unmeasured). Development folds only.

### A13.1 Resolution study done right
| Element | Declared choice | Why |
|---|---|---|
| Arms | IN-22k ConvNeXt-T at **224** (224-pretrained) vs **384** with the **384-pretrained** weights `convnext_tiny.fb_in22k_ft_in1k_384` | removes the S01 pretraining mismatch |
| Rows | full lesion-grouped **5-fold OOF** (15,294 rows), not fold 0 | ≈ 5× the rows; folds 1–4 confirmatory, fold 0 flagged |
| Seeds | 3 (2 if the budget forces it, stated) | — |
| Endpoints | **8-class balanced accuracy** (headline, §A10.2) and all-age escalation pAUC; under-40 pAUC descriptive | comparability with the ISIC 2019 anchor |
| **Per-archive readout, pre-declared** | HAM / BCN / MSKCC deltas plus a resolution × archive interaction | the effect is archive-dependent; pooled-only reading hides it |
| Noise control | evaluate **EMA weights** (or the mean of the last-k checkpoints) beside `_last`, both declared | last-epoch noise ≈ the effect size |
| Power | per-seed fold-0 Δ SD ≈ 0.008; pooled over 5 folds and 3 seeds, the SE is ≈ 0.002 (extrapolated, assumes fold-independent noise) → ≈ +0.006 detectable | S01 could only see ≥ +0.015 |
| Cost | 224 arm: folds 1–4 × 3 seeds = 12 runs at 40–50 min (**measured** IN-22k 224 anchor) ≈ 9 h; 384 arm: 15 runs at 64–75 min (**measured** in1k 384 anchor; IN-22k 384 extrapolated, same architecture) ≈ 17 h | ≈ 26 h total; the 224 arm doubles as Q11 (trunk-only control) |

### A13.2 If HAM gains and BCN does not: archive-aware detail, not a global resize
Test the three BCN hypotheses on existing predictions before any training (resolution investigation §4): lesion-area
fraction by archive (masks), Δ by lesion size, Δ on non-segmentable / acral / mucosal cases. Then pick one:
**lesion-centred native-resolution crop** as a second view (V5's zoom showed random-location crops already help —
+0.0085 vs clues — so a multi-view member is a candidate, judged as an *engineering* arm, not a mechanism),
**multi-resolution ensemble members** (224 + 384, §A11 diversity), or archive-specific TTA scales.

### A13.3 Fair re-test of the V5 biology (only if A13.1 finds a resolution gain)
Re-run the modules with the strongest V5 evidence — `twostep` / `m4` (heads), `geometry` (real gain, wrong
mechanism), `look` (chromophore) — at the winning resolution, with:
- **separate parameter groups:** new modules at 1e-3 with their own cosine for the whole run, trunk at 1e-4;
- **no zero-initialised input channels frozen in the head stage:** small random init, and the stem unfrozen;
- **an equal tuning budget** to the control's: a pre-declared 3-point learning-rate grid per module, selected on fold 0
  only, confirmed on folds 1–4;
- the V5 falsifiers unchanged for any **mechanism** claim; an arm that gains but fails its falsifier may still enter the
  §A11 ensemble as an **engineering** member (the falsified-vs-useful split the audit recommended).

### A13.4 Order and gates
A13.0 (free) → A13.1 (≈ 26 h) → A13.2 (CPU diagnostics, then one arm) → A13.3 only if A13.1 shows a resolution gain on
the pre-declared endpoint. All of it is declared and hashed at V6-0 before any result. The brainstorm decides its
priority against §A10–A12.

---

# Part B — Execution

**Design principle:** a **core path** delivers a complete, reportable V6 on its own. Conditional
phases run only if their V5 or V6 gate fires. V6 can stop at any gate with a result.

## B0. Fixed rules and cost anchors

**Cost anchors (ConvNeXt-T, measured):**
- 224 px fold-0 run: **41.0 min**;
- 384 px fold run: **64–75 min**;
- one 5-fold × 1-seed set: **5.3–6.3 h** at 384 px, or **≈ 3.4 h** at 224 px.
- *(Added 1 Oct evening, measured in V5 Q3, `results/v5/logs/queue_Q3.log`)* IN-22k ConvNeXt-T,
  224 px, fold 0, full run wall time: control **40.5–49.9 min**; twostep / clues / gem / m4
  **37.8–43.4 min**; memory **46.2–46.9 min**.

Anything else is **unmeasured** until `scripts/gpu_benchmark.py` has run in the **unfrozen
fine-tune stage** (hardware notes: stage-1 memory understates the peak by up to 6×).
*(1 Oct evening)* `gpu_benchmark.py` times a bare backbone on random tensors; for an arm with extra
heads, views or CPU transforms, time the real arm with the `research/v5/arm_benchmark.py` pattern
(real loader, model and loss, unfrozen, scaled against a measured reference run).
*(2 Oct, V5 E8/Q4 lesson)* Arm benchmarks report **absolute seconds per epoch**; a ratio to a reference
arm is used only if that reference ran warm in the same process and was not timed first — the
first-timed arm ran 22–29% slow (cold cache), which under-projected every V5 Q4 run by ~20%.

**Fixed for every run:**
- `--num-workers 2`, `--patience 0`, 30 epochs;
- ≥ 5 GB free on C:;
- screens save `_last` only, confirmation saves both;
- smoke first;
- the test lock stays armed;
- **(2 Oct)** every engineered token or statistic (fronts, D_part / D_bal / D_nn / D_dict, concept
  scores) is computed in fp32 and must be **fp16-castable after standardisation**: |z| ≤ 50 with a
  counted saturation backstop, not only `isfinite` (V5 Q4: finite values of 10⁷–10⁸ overflowed fp16 in
  the classifier); its standardisation statistics are checked for heavy-tail contamination before use.

---

## B1. Core path

> **1 Oct revision — read first.** The **primary path is §A0.15** (CG-DM with R1/R2, plus the
> young-data, DINOv3-LLRD and N13 branches). In the table below:
> - **V6-0** additionally: count ISIC-2020 under-40 malignancies (metadata only); pull ISIC API
>   patient IDs (metadata only) where available; write the prior-art audit; freeze the Stage 0
>   script; record the K = 3 loader benchmark.
> - **V6-2** is trimmed to **DINOv3-LLRD and ConvNeXt-V2** (MaxViT / EfficientNetV2 only if budget
>   remains), and is a parallel branch, never upstream of CG-DM.
> - **V6-2b is demoted; V6-3b is restored (1 Oct evening, §A3b)** with plain in-domain members,
>   25 runs, filling GPU gaps the CG-DM path leaves. **V6-3** runs at **V5's chosen resolution** (S01 failed
>   → 224 px unless V5's composite lock says otherwise); its "384 px" hours estimate is superseded.
> - **V6-10** ensembles only the §A0 / branch winners; distillation unchanged. *(1 Oct evening:
>   the **E-primary six-architecture uniform soft vote is restored** beside the §A0 / branch
>   winners, and the V5 TTA protocol is carried per the §A5 TTA row.)*
> - **V6-6**: ECL → the RML/CG-DM ablation ladder (§A0.11); subclass discovery → a Stage 0
>   diagnostic; pAUC-DRO only after CG-DM works. **V6-7 N10 is demoted.** **V6-8** concepts become
>   offline teachers (DermFM barred near HIBA until audited). **V6-9 N13** gains R1 (§A0.12).

| Phase | Work | Inputs | Outputs | GPU (est.) | Stop / continue rule |
|---|---|---|---|---|---|
| **V6-0** Freeze and data | Fill the §A5 decision table from V5. Download and deduplicate MILK10k (training, after S84), ISIC-2020, DERM12345 (if the licence is OK). HIBA eligibility, count and power (metadata only). *(1 Oct evening, §A9: check `fitzpatrick_skin_type` availability on HIBA / MILK10k / ISIC-2020, metadata only (issue 7); C4 eval-transform parity test for every trunk.)* Declare the allocation. Extend the S71-style lesion-grouped folds to the new rows. **Hash the V6 plan** | V5 results; `perceptual_hashes.npz` | `results/v6/v6_plan_freeze.json`; `data_allocation.json`; `fold_assignments_v6.csv` | CPU | HIBA underpowered → switch to the pre-declared ISIC-2020 patient-held-out fallback |
| **V6-1** Segmenter (**mandatory for CG-DM, out-of-fold**: fold-f masks come from a segmenter trained without fold f) | U-Net-style segmenter on HAM masks (fold-respecting) + Task-2 overlap; human QC on 50 BCN + 50 MSKCC | HAM masks | `ml/checkpoints/seg_v6_f*.pt`; `seg_qc.json` | ≈ 1–2 h (benchmark) | Median Dice ≥ 0.85 and ≤ 5% failures, else the V6-5 dual-stream uses DRE-2 geometry |
| **V6-2** Backbone screen | DINOv3-ConvNeXt-T, ConvNeXt-V2-T, MaxViT-T, EfficientNetV2-S, each with the surviving V5 heads. Fold 0, 224 px, seeds 42/43. Paired against the V5 composite on the V5 trunk (IN-22k ConvNeXt-T, `results/v5/screens/gate_control_in22k_vs_control.json`) | V5 composite spec | `results/v6/screen/*` | ≈ 6–9 h (ConvNeXt pair measured; MaxViT/EffNetV2 benchmark first) | Promote ≤ 2 whose all-age pAUC Δ exceeds the V5 noise floor with Macro-F1 retention ≥ −0.010 |
| **V6-2b** V5-feature portability (§A3b) | Each of the 5 other V1 architectures (ConvNeXt-S, EfficientNet-B0/B3, ResNet-50, DenseNet-121): **arch control** vs **arch + V5 locked heads** (adapter per §A3b). Fold 0, 224 px, seeds 42/43, `_last`. Gate as V5's screen gate against the arch's own control | V5 composite lock; §A3b stage test passes | `results/v6/portability/*`; `portability_holm.json` | 20 runs: ≈ 14 h at the ConvNeXt-T anchor; **unmeasured** for the others (ConvNeXt-S ≈ 1.7× extrapolated); benchmark first (§B3) | Holm across the 5; result sets the §A5 row. No run is repeated if a gate fails |
| **V6-3** Finalists | ≤ 2 backbones, folds 1–4 (+ fold 0 for OOF) × seeds 42/43 at the V5-chosen resolution | V6-2 winners | `results/v6/confirm/*`; OOF matrix | ≈ 21–25 h (384 px, ConvNeXt anchors) | Gates A–D on folds 1–4 vs the V5 composite |
| **V6-3b** Six-architecture OOF (§A3b) | The 5 other V1 architectures × folds 0–4 × seed 42, in domain, with V5 heads if V6-2b made them general (else plain controls). The ConvNeXt-T member **reuses** V5 runs (no retraining): the composite's seed-42 fold runs if the heads are carried, else V5's seed-42 control fold runs, so every member is built the same way. *(1 Oct evening: V6-2b is demoted, so members are plain controls and the ConvNeXt-T member is the banked S72 `convnext_tiny-v4_R0_kfold_f{0-4}_s42` set, on disk, verified 1 Oct)* | V6-2b outcome (default: not run → plain controls) | `results/v6/oof_six/*`; common OOF matrix (15,294 rows × members) | 25 runs: ≈ 17 h at the ConvNeXt-T anchor; **unmeasured** for the others | OOF complete for every member before any ensemble is formed |
| **V6-4** Young-data expansion (N5) | Learning curve: add 25/50/100% of the new under-40 histology-confirmed lesions (fold 0, 224 px, 2 seeds), then retrain the best trunk on the full expanded set (5-fold, 1 seed) | V6-0 data | `learning_curve.json` | ≈ 10–12 h | Slope > 0 → keep the data; flat → report "data-saturated at N" |
| **V6-10** Ensemble and deployment | OOF common matrix → pre-declared ensembles, all on folds 1–4 OOF (fold 0 flagged): **E-primary** uniform soft vote of the six in-domain architectures (V6-3b) vs the V5 15-model system; secondaries: Caruana / Nelder–Mead weights (OOF-fitted), morphology-gated (B10, shallow gate), V6-3 finalists added. Diversity report (Yule's Q, disagreement, double-fault, as Phase 1). *(1 Oct evening: members are scored with the V5 TTA protocol if the §A5 TTA row fires, else single view; paired Macro-F1 Δ — the Phase-1 endpoint — is now **co-primary** with all-age pAUC, see the stop rule. §A9 adds to every ensemble: C1 bipartite RAPS refit on its OOF (output field), C3 per-band Dirichlet arm, C5 drift baseline.)* Each ensemble gets Dirichlet on cross-fitted OOF, per-band ECE / signed gap and an S56 refit. Then **distil the winner into one ConvNeXt-T**; S69 gate; `explain` endpoint in the S60 FastAPI; model card update | V6-3 / V6-3b / V6-4 models | `v6_ensemble.json`; `v6_diversity.json`; distilled checkpoint; `paper/v6/model_card.md` | ≈ 12–20 h | E-primary, **co-primary (owner decision 1 Oct evening)** — passes only if **both** hold on folds 1–4 OOF: (1) paired all-age pAUC@0.20 Δ above the V5 noise floor; (2) paired **Macro-F1 Δ > 0**, lesion-grouped bootstrap CI lower bound > 0. An intersection-union test: no multiplicity correction between the two, but power is the lower of the two (stated in the report, with each endpoint's own result). <40 pAUC descriptive. Distilled model within 0.01 (Macro-F1, pAUC) of the chosen ensemble |
| **V6-11** Confirmation | **Single locked read** on HIBA (or the fallback): V6 system vs V5 system vs V1 system, all with S56, at matched referral. Paired under-40 sensitivity (exact McNemar); S70 contract; Macro-F1 non-inferiority. *(1 Oct evening, §A9: the contract's budget term uses the C2 label-free budget lock; the S48 endpoint is read in its declared form, lesion unit (C6); C1 per-band FRR and coverage reported with Clopper–Pearson; Fitzpatrick slice if issue 7's check finds labels.)* | Locked V6 system | `results/v6/v6_confirm_report.json`; receipt | ≤ 1 h | Single read, enforced by a receipt |

**Core-path GPU total ≈ 50–70 h** before the 30 Sep revision; **≈ 81–101 h with V6-2b + V6-3b at the ConvNeXt-T anchor**,
and more in practice, because ConvNeXt-S (≈ 1.7×, extrapolated), EfficientNet-B3 and
DenseNet-121 are slower and all five are unmeasured until §B3. Re-quote after the benchmarks. If the budget binds, V6-3b drops to 3 folds per member and the
ensemble is read on those folds only (declared now, not chosen after results).

**Consolidated core-path budget (audit, 1 Oct evening — supersedes the totals above).** Per-run
anchor = IN-22k ConvNeXt-T at 224 px, **38–50 min** (measured, §B0). Every row except the anchor
itself is an **extrapolation**; "unmeasured" means no number is quoted.

| Phase | Runs | GPU h | Basis |
|---|---:|---:|---|
| V6-0 IN-22k noise-floor seeds 45/46/47 (§A0.9) | 3 | 1.9–2.5 | measured anchor × 3 |
| V6-1 segmenter | — | 1–2 | runsheet estimate, unmeasured |
| §A0.8 Stage 0 feature pass (5 fold models × 15,294 rows) | — | ≈ 0.25–0.5 | S58 measured 19,096 images in 2.8 min per model (CHANGELOG S58); F3 token writes unmeasured |
| CG-DM screens + confirmation (§A0.15) | 51 | 32–43; **62–69 if the V5 lock carries `zoom`** (2 Oct) | control anchor, or the measured zoom run (73–81 min); **+ retrieval/OT overhead unmeasured** (K = 3 benchmark) |
| V6-2 trimmed: DINOv3-LLRD + ConvNeXt-V2, seeds 42/43 | 4 | 2.5–3.3 | same-family anchor |
| V6-3 finalists (≤ 2 × folds 0–4 × seeds 42/43, 224 px) | ≤ 20 | ≤ 12.7–16.7 | anchor |
| V6-3b five V1 architectures × 5 folds × s42 | 25 | 15.8–20.8 at the anchor | **the five architectures are unmeasured** (§B3); ConvNeXt-S ≈ 1.7× |
| V6-4 young-data curve + retrain | — | 10–12 | runsheet extrapolation |
| V6-10 ensemble, TTA inference, distillation | — | 12–20 | runsheet estimate; TTA unmeasured |
| V6-11 confirmation | — | ≤ 1 | — |
| **Core total** | | **≈ 88–121 h** (≈ 118–147 h if the lock carries `zoom`) | **plus** OT overhead, void-rule reruns, R1/R2 (benchmarked first), the five architectures, TTA |

R1, R2 and every §B2 conditional phase are outside this total. At the measured anchors the core
path is roughly one week of continuous GPU; the calendar (§B4) is bound by gates and reads, not by
GPU hours.

---

## B2. Conditional phases (run only if the gate fires; order by expected value)

| Phase | Work | Gate (from V5 / V6) | GPU (est.) | Falsifier / stop |
|---|---|---|---|---|
| **V6-5** Resolution and second view | (a) **Wavelet space-to-depth** (768 px → 12 channels at 384²), BCN/MSKCC-stratified. (b) **Separate-trunk dual-stream** (global + lesion crop from V6-1, late fusion or cross-attention) vs a **capacity-matched ConvNeXt-S** single stream | V5 `zoom` > `zoom_random` on the declared score **and on escalation mass** (§A5); the DRE-8 gain is concentrated on BCN/MSKCC | ≈ 8–12 h (unmeasured; benchmark 768 px decoding) | (a) Gain on BCN/MSKCC, not HAM. (b) Must beat ConvNeXt-S and concentrate in the smallest lesion-size tercile |
| **V6-6** Representation and losses | Multi-proxy contrastive (ECL-style on DRE-6); **subclass Group-DRO** (N6); **subtype supervision** (DERM12345); **pAUC-DRO** | V5 `memory` passed / D4 subclass found / licence OK / V5 `m4` passed | ≈ 6–10 h | Each against its V5 parent: young-mel prototype separation; worst-subclass pAUC; pAUC_histo |
| **V6-7** Domain generalisation | **N10 FCMAE domain pretraining** of ConvNeXt-V2 on unlabelled multi-device dermoscopy (no confirmation images) → fine-tune; **N9** class-conditional archive adversary; **SWAD**; test-time normalisation adaptation; **N11** EM prior (CPU) | V5 M7 / SWAD LOAO results; an acquisition gap remains | N10: **benchmark first, possibly 20–40 h**; others ≈ 6–8 h | LOAO Macro-F1. N9 needs in-distribution retention ≥ −0.01 |
| **V6-8** Concepts | MONET concept probabilities (MILK10k ships them; also run MONET on the rest) **validated against Task-2 expert masks** → concept-supervised DRE-7; Derm7pt 7-point | V5 DRE-7 sign check passed | ≈ 4 h (+ CPU) | Concept AUROC vs expert masks ≥ 0.75 before any use |
| **V6-9** Context and modality | **N13 ugly duckling:** a set encoder over one patient's lesions (ISIC-2020 patient IDs). **N12 smartphone routing:** S69 gate → a clinical expert warm-started from the V6 trunk (PAD patient-grouped + MILK10k clinical pairs) | Patient IDs available / S69 kept | ≈ 8–12 h | N13: young-mel rank among the patient's own lesions. N12: patient-grouped Macro-F1 vs the 0.760 PAD reference |
| **V6-12** Foundation LoRA | PanDerm ViT-L or MedSigLIP-448 with LoRA + gradient checkpointing | V6-2 shows pretraining is the lever **and** the leakage audit clears | Unmeasured (heavy) | Must beat V6's incumbent trunk (IN-22k ConvNeXt-T, or DINOv3-LLRD if it won its screen) at equal data |
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

1. **Week 1 after Review 2:** V6-0 (CPU + prior-art audit) → Stage 0 feature pass → CG-DM Stage 0 ·
   R1 Stage 0 · young-data learning curve · DINOv3-LLRD screen (parallel). **§B3 benchmark of the
   five V1 architectures, then V6-3b runs in the GPU gaps** while Stage 0 is on the CPU
   (1 Oct evening restoration; ≈ 17 h extrapolated, re-quoted from the benchmark).
   **V6-3b runs fold-major** (fold 0 for all five architectures first, then fold 1, …): after the
   first five runs a fold-0 diversity report (Yule's Q, disagreement, double-fault vs the banked S72
   fold-0 member) exists as an early **descriptive** read — fold 0 is the selection fold, so it never
   decides anything; the E-primary is read only on folds 1–4 once all are complete.
2. **Week 2:** K = 3 benchmark → CG-DM Stage A → Stage B → (R2 feasibility gate) → folds 1–4
   confirmation. V6-3b continues in any remaining GPU gaps. (V6-2b only if budget remains.)
3. **Week 3:** the conditional phases that fired, in the table's order.
4. **Week 4:** V6-10 ensemble and distillation → V6-11 single confirmatory read → write-up.

Every phase ends with a CHANGELOG entry and ledger rows (`research/experiments.csv`, session
`v6_*`). Any change after the V6 freeze goes into a V6 amendment, labelled post-hoc if it follows
any V6 result.
