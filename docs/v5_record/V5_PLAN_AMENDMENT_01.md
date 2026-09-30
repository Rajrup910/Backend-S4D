# V5 Plan Amendment 01 — pre-S01 review

**Date:** 2026-09-23
**Status:** **ADOPTED 2026-09-23 — all items accepted by the owner (A1–A13, B1–B11), none
rejected.** Hashed into `results/v5/v5_plan_freeze.json` as `amendments[0]` before any V5 GPU
run. Where an item below conflicts with the master plan, runsheet, session plan or
`v5_external_analysis_spec.md`, this amendment wins.
**Amends:** `docs/V5_MASTER_RESEARCH_PLAN_REVISED.md`, frozen at sha256 `B7542DAB…` in
`results/v5/v5_plan_freeze.json`. That file is **not edited**, so its freeze hash stays valid.
**Why it is still legitimate:** no V5 GPU run has happened yet. An amendment adopted and hashed
before `V5-S01` counts as part of the pre-registration. It is not a post-hoc change.

**Adoption record.** Owner decision 2026-09-23: *adopt all items*. Decisions per item:
A1–A13 ACCEPTED; B1–B11 ACCEPTED. The two items that need an owner choice at execution time are
adopted as obligations, not yet decided: A11 (choose (a) learned segmenter or (b) HAM-only
before S02) and B6 (reader study, subject to ethics sign-off). This file is immutable from here
on. Any further change goes into a new file, `V5_PLAN_AMENDMENT_02.md`.

Every number below is taken from an existing repo artifact, and the source is named. "Computed
here" means a read-only computation on `results/v4/kfold/oof_predictions.csv`, done on the date
above.

---

## Summary

| # | Severity | Issue | Fix |
|---|---|---|---|
| A1 | **Critical** | Gates quote HAM-only counts, but V5 evaluates the pooled S71 partition. Fold 0 has only **16** under-40 escalating lesions | Two-level gate: all-age screening on fold 0; the under-40 gate is read on cross-fitted OOF only |
| A2 | **Critical** | Fold 0 is used both to select candidates and to confirm them | Select on fold 0; confirm finalists on folds 1–4 |
| A3 | Major | Best-epoch checkpoints are chosen on the same fold that is then scored | `_last` is primary and `_best` is a sensitivity check (S54 convention) |
| A4 | **Critical** | Absolute gates (0.7650–0.8050) are imported from HAM val/test, but pooled folds score 0.65–0.70 | Replace them with paired Δ gates against the 384 px control |
| A5 | Major | Measured seed SD (0.0230) is larger than the MCID (0.015) and the warning threshold (0.015) | Use a hierarchical (seed × lesion) CI and set the noise floor from S01 itself |
| A6 | Major | S01 has one 224 px seed, and seeds 43/44 are not in the budget | Add two 224 px runs to S01 |
| A7 | Minor | The runtime estimate applies a GPU-bound multiplier to a decode-bound run | Use the measured 64–75 min, confirmed by the smoke epoch |
| A8 | **Critical** | Stage 7 reinstates conformal and groupwise layers that V4 rejected, and leaves out the actually deployed S56 | Rebuild Stage 7 around S56 refit on V5 OOF |
| A9 | **Critical** | Gate F's null is already beaten by the V1 stack. It also drops the S70 contract, and its primary endpoint cannot be measured on the candidate cohorts | Paired V5-vs-V1 primary endpoint plus the contract; re-sized |
| A10 | **Critical** | BCN20000 and MSKCC are listed as "external", but they are in the training data | Remove them; restate the candidate list |
| A11 | Major | The segmentation gate (≥90% masks) cannot pass: HAM masks cover 45.6% of development rows | Validated learned segmenter, or restrict to HAM and say so |
| A12 | Major | B4's premise (fine structure lost at native resolution) mostly does not apply to HAM's 600×450 images | Archive-stratified readout; fix the QC assertion |
| A13 | Minor | Internal inconsistencies (Macro-F1 definition, CI method, pAUC form, grids) | Line edits listed below |
| B1–B11 | Additions | Error anatomy, case-mix endpoint, noise floor, reader study, in-domain attributes, S56 on V5, arm pruning, … | See Part B |

---

# Part A — Corrections

## A1. Gate sample sizes come from the wrong population (critical)

Gates A and C (§18) and §20 quote **HAM-only** counts: 64 images / ≈34 lesions under 40, and
397 / 893 in the 40–59 and 60+ bands. But S01 onward train and score on the **pooled S71
partition** (HAM + BCN20000 + MSKCC, 15,294 rows). Counts computed here:

| Quantity | Plan says | Pooled S71 actual |
|---|---:|---:|
| Under-40 escalating, all folds | 64 img / ~34 lesions | **193 img / 81 lesions** (BCN 35, HAM 34, MSKCC 12) |
| Under-40 escalating, **fold 0 only** | — | **35 img / 16 lesions** |
| Under-40 escalating, folds 1–4 | — | 158 img / 65 lesions |
| 40–59 escalating | 397 | 1,463 img / 544 lesions |
| 60+ escalating | 893 | 3,624 img / 1,436 lesions |
| All-age escalating, fold 0 | — | 1,069 img / 419 lesions |
| Age unknown | not handled | 251 rows (47 escalating) |

**Consequence.** Every Tier-1 screen (S02–S12) applies Gate A (Δ under-40 pAUC ≥ +0.050) to
**16 lesions**. At that size, pAUC moves by more than 0.05 when a single lesion changes rank, so
the gate becomes a coin flip that runs about 12 times. The S70 note that "V4 val holds 0 under-40
escalating lesions" (CHANGELOG §S70) is why OOF is the only development source at all.

**Fix: two-level gate.**
- **Screening, fold 0.** The primary is **all-age escalation pAUC@0.20** (419 lesions), plus
  Gate B. Under-40 pAUC and under-40 rescue count (B1) are reported **descriptively only**.
- **Finalist, cross-fitted OOF.** Gate A (under-40 pAUC) is read here, on the 65 lesions in
  folds 1–4 (see A2), with a lesion-grouped CI.
- Define the unknown-age policy: exclude those rows from banded endpoints and keep them in
  global ones.

## A2. Selection and confirmation reuse fold 0 (critical)

Stages 1–5 select every candidate on fold 0 (N_val = 3,059). Then §19 Tier 3 and S81 "confirm"
finalists on all 5 folds, and that set includes fold 0 again. With roughly 15 screened arms, the
winner's fold-0 score is biased upward, and including fold 0 in the confirmation carries that
bias forward.

**Fix.** Fold 0 is the **selection fold**. Finalist gates (A–E) are read on **folds 1–4 only**
(12,235 rows; 65 under-40 escalating lesions). Fold 0 is still generated so that the OOF matrix
is complete for the ensemble and for S56, but it is flagged `selection_fold=true` and left out of
every confirmatory statistic.

## A3. Checkpoint selection on the scored fold

S72 fold 0 picked its best epoch on the fold it reports: best 0.6508 at epoch 20/30 against last
0.6394 at epoch 30 (`results/v4/kfold/logs/f0.w2.b32.20260918-044511.out.log`). If S01 compares
the best checkpoint of each arm, it rewards whichever arm is noisier. The S54 plan already settled
this: **`_last` primary, `_best` sensitivity** (patience off, fixed 30 epochs;
`results/v4/s54_plan.json`). Adopt that convention for every V5 paired comparison.

## A4. Absolute thresholds imported from other populations (critical)

The runsheet gates S14 ≥ 0.7700, S16 ≥ 0.7650, S17 ≥ 0.7950 and S18 ≥ 0.8050. Master §18 Gate F
requires ≥ 0.7500. The pooled 224 px control scores **0.6508–0.6974 per fold, mean 0.6703**
(CHANGELOG §Phase Y execution). The 0.77 figure is the S53r **HAM-val** number, and 0.805 is the
V1 **HAM-test** number. Those are different populations with different class mixes. As written,
these gates would most likely reject every backbone because of the population mismatch, not
because the backbones are worse.

**Fix.** Every development gate becomes a **paired Δ against the 384 px control on the same
rows**, as Gates A–C already are. Absolute numbers may appear only as labelled reference context.

## A5. Seed noise exceeds the MCID

S53r measured the control's seed SD at **0.0230** Macro-F1 (CHANGELOG §S53r readout). The plan sets
the MCID at +0.015 and Gate D's instability warning at s_seed > 0.015, so the warning will fire
on pure noise. The S01 hurdle "95% bootstrap CI of paired Δ excludes 0" resamples images.
That captures sampling noise but **not seed noise**, which is the dominant term here.

**Fix.**
- Build the CI for the seed-paired mean Δ with a hierarchical bootstrap: resample seed pairs,
  then lesions within each. Also report the t-interval over the 3 paired deltas.
- Set the warning threshold from the 224 px three-seed spread that S01 itself produces (see B3),
  not from a fixed number.
- The S53r precedent (+0.0297, seed range +0.0097 to +0.0466) would still pass. That is expected
  and correct.

## A6. S01's 224 px comparator is one seed

§8 V5-A1 names a "multi-seed 224 px Fold 0 control for seeds 42, 43, 44". Only
`R0_kfold_f0_s42` is banked. The runsheet's 4.6 h for S01 is exactly 3 × 92.6 min, so it covers
the 384 px arm only. **Add two 224 px fold-0 runs (seeds 43, 44).** At the measured 41.0 min
each, that is about 1.4 h. Without them, the design is not seed-paired.

## A7. Runtime extrapolation

- The plan's figure of "42.0 min × 2.26" is a typo. The S72 fold-0 log says **41.0 min**, and
  41.0 × 2.26 gives the 92.6.
- The ×2.26 multiplier is itself the wrong anchor. It comes from HAM-only 600×450 runs, which are
  GPU-bound. The pooled 224 px run is **decode-bound**: it measured 41 min against a 22 min
  synthetic benchmark, because BCN images are 1024×1024 (CHANGELOG §S70).
- The measured 384 px anchors are **64 min** (S70 synthetic benchmark, batch 32, 6.26 GB) and
  **~75 min** (banked R1+R4 pooled runs scaled to one fold). The best estimate is 64–75 min per
  seed, not 92.6. Confirm it with the smoke epoch before quoting nightly blocks.
- EfficientNetV2-M, SwinV2-384 and the dual-stream model have **no measured anchor**. Their VRAM
  figures in the runsheet (~5.4 / ~6.4 GB) are extrapolated. Run `scripts/gpu_benchmark.py` in
  the **unfrozen fine-tune stage** for each one before it enters a block.

## A8. Stage 7 contradicts V4's frozen safety conclusions (critical)

| Plan says | Repo record |
|---|---|
| §4: groupwise Dirichlet "SUPPORTED … KEEP". Stage 7: "S55 groupwise multi-calibration" | S55 was **not adopted** as the deployed calibrator. Under S65, the per-band map **undoes the frozen <40 λ** (<40 decision sensitivity 0.409→0.262 on reserved) (RESEARCH_LOG §S65) |
| §4: "Equalized bipartite conformal — SUPPORTED SAFETY LEVER". Stage 7 keeps it | S59: "the λ and conformal layers both lose at matched workload"; **deployed stack is S56@0.20 alone** |
| Stage 7 does not mention S56 | S56 per-band selective abstention **is** the deployed system |

**Fix.** Stage 7 becomes:
1. 24-view TTA.
2. A calibrator. Global Dirichlet is the default; S55 per-band is a pre-declared alternative,
   chosen on OOF and never stacked with λ.
3. **S56 per-band abstention, refit on V5 cross-fitted OOF.**
4. S69 modality gate.

Conformal is reported descriptively. The plan should also state what this unlocks: *"V1 is the
base only because S56 needs cross-fitted OOF predictions"* (RESEARCH_LOG, V4 final audit). V5
finalists get 5-fold OOF, so **V5 is the first point at which a non-V1 base is possible**. Make
that an explicit deliverable (B5).

## A9. Gate F does not test V5 (critical)

1. **The null is already beaten.** H0 is "under-40 sensitivity ≤ 0.50". The incumbent
   V1 + S56 system already reaches **0.763** under-40 system sensitivity on reserved
   (RESEARCH_LOG §S59), and V1 argmax reaches 0.547 on HAM OOF (CLAUDE.md, S7). Rejecting 0.50
   would say nothing about V5.
2. **It drops the S70 decision.** S70 D2 keeps the **0.855 floor**, the **0.25 referral limit**
   and a **relative term against V1** (`results/v4/s70/s70_decisions.json`). Gate F has none of
   the three.
3. **The primary endpoint probably cannot be measured.** A 7-class Macro-F1 needs all 7 classes
   in the external cohort. ISIC-2020 is overwhelmingly nevus / "unknown" / melanoma, with little
   or no BCC, AKIEC, DF or VASC (verify against its ground-truth file). And because the testing
   is fixed-sequence, the endpoint V5 exists for sits behind an unrelated one.
4. **Wrong interval method.** Wilson with continuity correction contradicts the project rule:
   Clopper–Pearson for small-count proportions (`research/stats/intervals.py`, S7).

**Fix.**
- **Primary:** the paired difference in under-40 escalation sensitivity, V5 system vs V1 system
  (both with S56), at **matched referral**. Lesion unit, exact McNemar on discordant lesions.
- **Key secondary:** the S70 contract (0.855 floor, referral ≤ 0.25).
- **Global:** Macro-F1 non-inferiority if the cohort has all 7 classes, otherwise escalation
  AUROC non-inferiority. Declare which one before unblinding.
- **Sample size.** Computed here with exact binomial, one-sided α = 0.05. To show that the
  sensitivity exceeds the incumbent's 0.763 when the truth is 0.855, power is **0.399 at n = 50,
  0.608 at n = 81, 0.864 at n = 150**. The plan's "≥ 40 lesions" floor is sized against the wrong
  null. The paired design needs fewer lesions, so size it from the V1-vs-V5 discordance rate
  observed on OOF once finalists exist.

## A10. "External" cohort candidates include training archives (critical)

§16 Role A lists "BCN20000/MSKCC/ISIC-2020 unread partition". `manifest_v4.csv` covers **all
25,331 ISIC-2019 images** (CHANGELOG §S70), and BCN20000 + MSKCC make up **8,313 of the 15,294**
OOF rows. **Remove BCN20000 and MSKCC from Role A.**
- **ISIC-2020** partly shares institutions with ISIC-2019. Label it "cohort-external,
  same-institution". Before any read it needs the duplicate / patient / perceptual-hash screen
  already listed, run against all 25,331 ISIC-2019 images, not only HAM.
- **Other candidates for S75 to verify** (availability and licence not checked here): HIBA
  (Hospital Italiano de Buenos Aires, ISIC Archive, has age); MILK10k (ISIC).
- **Derm7pt** is already assigned to Role B (attribute supervision), so it cannot also serve as
  Role A.

## A11. The segmentation gate fails by construction

§9 B5 requires valid masks on **≥ 90%** of the development split. The only masks on disk are
HAM's (`data/ham10000/HAM10000_segmentations_lesion_tschandl/`), and HAM is **45.6%** of the
pooled rows. BCN and MSKCC have none, and they hold **129 of the 193** under-40 escalating images.
As written, A2 (dual-stream), B4, B5 and B9 can never run.

**Fix (choose one before S02):**
- **(a) Learned segmenter.** Train it on HAM masks, fold-respecting, so no fold-k image's mask
  trains the segmenter used for fold k. Validate by Dice on held-out HAM masks **plus a
  human-annotated QC sample of 50 BCN + 50 MSKCC images**, with a pre-declared pass rule such as
  median Dice ≥ 0.85 and ≤ 5% failures. This is a validated mask, not the "unreliable heuristic"
  §9 forbids.
- **(b) HAM-only restriction.** Restrict the mask-dependent arms to HAM rows and state that they
  then see only 34 of the 81 under-40 lesions.

## A12. The B4 native-resolution premise is archive-dependent

Native sizes (sampled here): **HAM 600×450**, **BCN 1024×1024**, **MSKCC 1024×680–768**. Going
from HAM's 450 px short side to 384 px is only a **1.17×** downsample, against **2.67×** for BCN.
Native-resolution micro-patches can therefore add information mainly on non-HAM rows. Archive
identity is decodable at 0.97 even after colour normalisation (S49 re-run), so any B4 gain is
confounded with archive.

**Fix.**
- Pre-register an archive-stratified B4 readout.
- Correct §9's QC assertion from "1024×768 or native ISIC dimensions" to "crop coordinates are in
  the **native** pixel frame of that image, whatever its size".

## A13. Line-level fixes

- **§20:** "Macro-F1 (7-class unweighted harmonic mean)" is wrong. Macro-F1 is the unweighted
  **arithmetic** mean of the per-class F1 scores.
- **Gate B:** "stratified bootstrap" should read **lesion-grouped bootstrap** (Hard Rule 1;
  `research/ablation/bootstrap.py`).
- **pAUC:** state the form. The S54 plan used **McClish-standardised pAUC@0.20**; use the same so
  V4 and V5 numbers are comparable.
- **§24 S03:** lists λ_esc ∈ {0.25, 0.5, 1.0}, but §19 Tier 1 says the default of 0.5 comes
  first. Keep §19.
- **B12:** its 3×3 "candidate grid" contradicts Tier 2's margin-only ablation. Keep Tier 2.
- **Runsheet S13:** "worst-group risk improves ≥ 10%" does not say relative or absolute, or on
  which fold. Restate it as Δ under-40 pAUC on folds 1–4.

---

# Part B — Additions

## B1. Stage 0b: under-40 error anatomy (CPU, existing S72 OOF, no new read)

Characterise the 81 under-40 escalating lesions before building any new mechanism.
- **What to record per lesion:** caught or missed at the control's matched-referral threshold;
  class (images: mel 133, bcc 52, akiec 8 — computed here); archive; mask area fraction where a
  mask exists; anatomic site; native resolution.
- **Output:** the **"hard core"**, meaning the lesions the pooled control misses.
- **How to use it:** every Stage 2/3 mechanism declares in advance which part of the hard core it
  targets (small lesions → dual-stream; peripheral-growth morphology → B9/B12; high-frequency
  structure → B8). Each readout reports **rescue count on its declared subset**, with a
  Clopper–Pearson CI.

This turns a noisy 16-lesion pAUC into a test of whether the mechanism does what it claims. It is
the most useful single addition in this amendment.

## B2. Case-mix-standardised under-40 endpoint

The V4 audit found the under-40 AUC gap is **partly case mix**: melanoma-vs-benign 0.873 (<40)
against 0.940 / 0.909 (RESEARCH_LOG §V4 audit). The under-40 band is 2,405 / 2,721 nevi (computed
here). Report two things beside every under-40 pAUC:
- **MEL-vs-NV AUC within band.**
- A **class-standardised escalation pAUC**, with the <40 class mix reweighted to the pooled mix.

That way a shift in which escalating classes are caught cannot pass for a ranking gain.

## B3. Measured noise floor before Stage 2

S01 (with A6) produces three 224 px and three 384 px seeds on fold 0. From the 224 px
seed-vs-seed differences, publish the **null distribution** of Δ for Macro-F1, all-age pAUC and
under-40 pAUC as `results/v5/s01_noise_floor.json` **before** any Stage-2 run. Any MCID below
the 95th percentile of that null is raised to it. This replaces the historical σ = 0.072 with a
number measured on the actual partition and recipe.

## B4. Archive-stratified reporting for every under-40 claim

The under-40 escalating lesions are split BCN 35 / HAM 34 / MSKCC 12. Every under-40 result
reports the per-archive breakdown and a pooled estimate with archive fixed effects. A gain that
exists only in BCN rows is an archive finding, not a morphology finding, until shown otherwise.

## B5. Deliverable: the S56 system on a V5 base

Once finalists have 5-fold OOF, refit S56 per-band thresholds on it and compare the **V5 + S56**
system against **V1 + S56** at matched referral, using OOF folds 1–4 only. This does two things:
it resolves the open "V4/V5 base" decision (RESEARCH_LOG, V4 final audit), and it makes the
Gate F comparator (A9) available before the external read.

## B6. Small dermatologist reader study (owner decision; needs ethics sign-off)

The null pathway (§26) names "intrinsic visual ambiguity" as an alternative explanation, but
nothing in the plan can test it.
- **Design:** two dermatologists, blind to label and model output, read the 81 under-40
  escalating lesions plus about 160 age-matched benign mimics (NV/BKL).
- **What it measures:** reader sensitivity at comparable specificity, and whether readers also
  miss the model's hard core (B1).
- **Cost and value:** CPU-free, about a day of reader time, and it is the only direct evidence
  on ambiguity. It is also the kind of evidence TMI and MedIA reviewers ask for.

## B7. In-domain morphology attributes: ISIC-2018 Task 2

Before transferring Derm7pt attributes, check whether the **ISIC-2018 Task 2** expert attribute
masks (pigment network, negative network, streaks, milia-like cysts, globules; ~2,594 images)
overlap `manifest_v4.csv`. If they do, B11/MORPH-ATTR gets dermoscopy attribute supervision
**from the same source as the training data**. That avoids the cross-dataset definition audit in
§10 and keeps Derm7pt free for other roles.
- Any overlapping image inherits its S71 fold, so no leakage is possible.
- Count and licence to be verified.

## B8. Pre-register where scarce under-40 data goes

The binding limit is 81 lesions. Any newly sourced under-40 escalating data can be used for
training (more ranking signal) **or** for confirmation (Gate F), **never both**. Record the
allocation rule in S75 before data arrives. Otherwise the temptation will be to decide after
seeing how the model does.

## B9. Prune overlapping arms

- **B6 (MEL-vs-NV branch) and B12 (MEL-vs-NV ranking loss)** attack the same boundary through
  the same end-to-end mechanism. Merge them into one Stage-3 slot, with B12 as the default and
  the branch as its Tier-2 variant.
- **B3 (hierarchy)** nests B2 (escalation head). Run B3 only if B2 passes.
- Together these cut Stage 2/3 from 12 arms to 10. That means fewer selection degrees of freedom
  on fold 0 (A2), and about 3 GPU hours saved.

## B10. Lesion-size interaction for dual-stream

Dermoscopy lesions often fill most of the frame, and in that case a "lesion-centred crop" nearly
duplicates the global view. Using mask area fraction from B1 / A11, **pre-register** that the
dual-stream gain should concentrate in the bottom tercile of lesion size. If the aggregate gate
passes but the gain is flat across terciles, the mechanism claim, "optical zoom on small
lesions", fails. The capacity explanation (§23b) then stays open.

## B11. Governance

Adoption is recorded in `CHANGELOG.md` and in `v5_plan_freeze.json` (`amendments[0]`, with
per-item decisions). Any later amendment after S01 starts must be labelled **post-hoc** in the
paper.

---

## Budget effect (ConvNeXt-Tiny anchors only; others unmeasured)

| Change | GPU time |
|---|---:|
| A6: two 224 px fold-0 seeds | +1.4 h (measured 41.0 min each) |
| A7: re-anchored 384 px per-seed time | −0.9 to −1.4 h per 3-seed block (64–75 min vs 92.6) |
| B9: two fewer Stage-2/3 arms | about −2.5 to −3 h |
| A11(a): segmenter training | small; measure before quoting |
| B1, B2, B3, B4, B5 | CPU only |

Net effect: roughly neutral to slightly cheaper than the runsheet's expected path. Treat the
EfficientNetV2-M, SwinV2-384 and dual-stream lines as **unknown** until `gpu_benchmark.py` has run.
