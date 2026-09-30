# V5 Plan Audit — pre-adoption review of Amendment 02, the DRE and the runsheet

**Date:** 2026-09-29
**Scope:**
- `docs/v5_design/V5_PLAN_AMENDMENT_02.md` (A02);
- `docs/v5_design/V5_DERM_REASONING_ENGINE.md` (DRE);
- `docs/V5_RUNSHEET.md`;

all checked against the frozen master plan (`B7542DAB…`), Amendment 01 (`219634DB…`) and the
repository record.
**Status of the audited files:** PROPOSED and **not hashed**. `results/v5/v5_plan_freeze.json`
holds only `amendments[0]`. Every fix below is therefore part of the pre-registration, not a
post-hoc change.
**Method:**
- read every cross-reference between the three files;
- re-sum each night's GPU hours from measured anchors;
- check each arm against the 12 barred mechanisms (master §21);
- check each number against its source file.

---

## 1. Findings and resolutions

| # | Severity | Finding | Evidence | Resolution (where) |
|---|---|---|---|---|
| **AU1** | Critical | Two different night schedules existed. A02 §7 put M3 and G on night 3; DRE §6 put Clues on night 2 and Zoom on night 3 | A02 §7 vs DRE §6 | One authoritative table in the runsheet §8. A02 §7 and DRE §6 now only point to it |
| **AU2** | Critical | Confirmation nights were planned at 8.5–10 h (composite + control, 8 × 64–75 min). With DRE-8 in the composite they reach about 10.7–11.7 h, which exceeds a night | 384 px anchor 64–75 min (A01 A7); zoom 1.5–2× unmeasured | The 12 control fold runs are independent of screening, so they move to **daytime windows** (owner decision 29 Sep). Nights carry the composite only (4.3–5.0 h, ×≈1.34 with zoom). Fallback: 2 seeds (runsheet §8.1, §8.3) |
| **AU3** | Major | Confirmation assumed 384 px, but 384 px was only shown to help on **HAM-only** (S53r R1 +0.0297); S01 has not yet tested the pooled partition | CHANGELOG §S53r; A01 A6 | **Resolution branch:** if S01 fails its two-hurdle rule, the composite and confirmation run at 224 px (runsheet §8.3) |
| **AU4** | Major | **Score-function confound.** The control ranks by `escalation_mass` (`research/v4/recipe.py:641`); arms with an escalation head would rank by their head logit. A Δ could come from the scoring rule, not the representation | recipe.py `escalation_mass` | Each arm has a **declared escalation score**, fixed in the arm registry before its run. Escalation mass is logged for every arm as a secondary (runsheet §2, §6) |
| **AU5** | Major | CPU pre-checks (Q1, Q2/D6, Q4, Q5, M1-QC) decide which arms run. Hashing the plan after running them would make the gates post-hoc | runsheet §8 | **Hash first** on the morning of 30 Sep, then run the pre-checks (runsheet §1, then §4) |
| **AU6** | Major | Group-DRO over age × escalation: fold 0 trains on folds 1–4, which hold only **65 under-40 escalating lesions (158 images)**. The master plan (§8 V5-A3) itself says "do not create tiny unstable groups" | A01 A1 counts | **G moves to V6** as subclass-DRO, gated on D4 (A02 §4, §5) |
| **AU7** | Major | No external cohort was sourced, so S84 could not run | A02 §5 | **MILK10k pre-registered** with eligibility, deduplication, allocation and analysis locked before download (runsheet §9). Owner decision 29 Sep |
| **AU8** | Minor | The eval transform `Resize(size·256/224) + CenterCrop(size)` clips 6.25% of each side (`recipe.py:557`), which can remove periphery that DRE-4 analyses | recipe.py `RESIZE_RATIO = 256/224` | Kept identical to the control for comparability, and reported as a limitation. A full-frame eval sensitivity analysis is scheduled in V6 |
| **AU9** | Minor | M7's CPU augmentations (CLAHE, optical/grid distortion) may make LOAO runs data-bound (the pooled 224 px run is already decode-bound: 41 min vs 22 min synthetic) | A01 A7 | Time them in the smoke epoch. If the epoch time grows by more than 25%, move them to the GPU (kornia) or drop the slow component and note it |
| **AU10** | Minor | Nested arms waited a whole night for their parent | A02 §7 | Nested arms run **on the same night** as their parent; the rule is applied at the morning read |
| **AU11** | Minor | Control seeds 45/46 are an A02 item, but A02 cannot be hashed before night 1 | A02 §6.1 | They run in the 30 Sep daytime window, after the hash and before the noise floor is needed (1 Oct morning) |
| **AU12** | Info | About 10 screened arms is a multiplicity risk | — | Restated: a screen is a **compute-allocation filter, not a test** (V4 S52 precedent). Confirmatory claims come only from folds 1–4 × seeds with the hierarchical CI |
| **AU13** | Minor | The composite is screened at 224 px but confirmed at 384 px, with no 384 px check before confirmation | Runsheet §8 | The fold-0 384 px composite (N8, selection fold) is reported descriptively. An all-age pAUC drop of more than 0.02 against its 224 px screen is flagged in the results |
| **AU14** | Major | The frozen pre-amendment `docs/v5_record/V5_FINAL_RUNSHEET.md` and `docs/v5_record/V5_SESSION_PLAN.md` still list the S78–S85 queue (three backbones, dual-stream, SupCon, wavelets, Group-DRO). The new runsheet's first name (`V5_RUNSHEET_FINAL.md`) was nearly identical to the frozen file's, an easy source of execution error | grep of `docs/` | The new file is renamed **`V5_RUNSHEET.md`**. Its header states that it supersedes both frozen files where they conflict (A01 preamble precedence). The frozen files stay unedited so their hashes remain valid |
| **AU15** | Major | Several module values were left open in the design files: DRE-2 β and mask width; palette τ; DSP Gabor wavelengths, Frangi constants and veil thresholds; DRE-6 scale/temperature and the orthogonality weight. An unfixed value is a tuning degree of freedom after results | DRE §3 | All fixed in runsheet §7 and marked **[declared here]**, before the hash |
| **AU16** | Major | Eleven V5/V6 plan files sat side by side in `docs/`, so it was unclear which one to execute | `docs/` listing | **One execution file per version:** `docs/V5_RUNSHEET.md` and `docs/V6_RUNSHEET.md` (the V6 master plan was merged in as Part A). Rationale files are in `docs/v5_design/`. The four frozen files moved **byte-identical** to `docs/v5_record/`, the freeze file paths were updated, and all 10 frozen hashes were re-verified. `docs/README.md` states which file to open |


### 1b. Pre-hash audit for the best case (2026-09-30; runsheet revision)

Aimed at raising the chance of a large, detectable effect **without** raising expectations. Every
number below is from `results/`, the ISIC API (metadata only) or a calculation shown here.

| # | Severity | Finding | Evidence | Resolution |
|---|---|---|---|---|
| **AU17** | Major | Every arm sat on the IN-1k ConvNeXt-T that V4 closed as a representation ceiling | S51/S54/S58/S64/S67; every V4 model's <40 pAUC 0.720–0.750 | Trunk screen first (`control --trunk in22k / dinov3`), same gate; winner carries every arm (`research/v5/trunks.py`, exact-forward test) |
| **AU18** | Major | Screen null did not match the statistic: a 2-seed mean Δ was compared with the 95th percentile of **single-seed** pair differences (≈ √2 too wide) | Illustration with seed SD 0.023 (S53r): a real +0.030 passed ≈ 15% of the time | 3 seeds; null SD = pooled pair SD / √k; 80th percentile (≈ 78% for +0.030). A null arm passes 20–36% (two endpoints OR-ed) instead of ≈ 1% — accepted because the screen is a filter (AU12) |
| **AU19** | Major | All-age pAUC is inflated by HAM follow-up nevi (device + label shortcut); the real differential is melanoma vs histology-confirmed benign | D0: histo-only pAUC 0.744–0.751 in every band; fold 0 pAUC_histo **0.7412** (1,965 rows; 896 confirmed benign) vs all-age 0.8105 (S72 OOF, corrected definition AU27) | Screen passes on all-age pAUC **or** pAUC_histo |
| **AU20** | Minor | DSP, zoom and M5 need fine detail but are screened at 224 px | Design | 384 px rescue (2 seeds, arm + parent) if S01 picks 384 and the 224 px Δ is positive |
| **AU21** | Major | ≈ 56–68 GPU h booked out of ≈ 130 h realistically available (judgement) | §8 re-sum | One continuous queue (§8) |
| **AU22** | Minor | N2 depended on same-day pre-checks | §8 old table | Queue ordered by dependency |
| **AU23** | Minor | S84's V5 system used 5 of the 15 models confirmation trains | §9 | 15-model system; no extra GPU |
| **AU24** | Major | The largest lever ever measured (pooling, +0.18) was absent; the corpus has 81 <40 escalating lesions | `young_data --count` (`results/v5/young_data/count.json`): pool 5,790 young histopathology dermoscopic images; minus 2,562 already in the V4 corpus, MILK10k 358, HIBA 185, and **20 lesions / 41 images whose other photos are in the corpus or those cohorts** (per-lesion API check; lesion-id joins are meaningless across namespaces) → **2,353 candidates, 535 escalating (457 mel; 195 escalating lesions with an id + 114 images without one)**, 1.37 GB; licences CC-0 1,103 / CC-BY 465 / CC-BY-NC 785. Corpus today: 81 <40 escalating lesions | Train-only `youngdata` arm; lesion-level leak check (every archive image of each candidate lesion); perceptual dedup after download |
| **AU25** | Info | Calibration invisible per arm | Owner request | Raw ECE and signed gap, overall and per band, for every arm |
| **AU26** | Info | With a new trunk the confirmatory contrast mixes trunk and modules | — | Descriptive split from the screens; optional trunk-only control (Q11) |
| **AU27** | **Critical** | M4's `confirmed_benign` (and the pAUC_histo draft) counted **every BCN and MSKCC benign image as confirmed**. It is not: BCN20000 has **5,101 histopathology of 7,831 benign (65%)**; a 40-image MSKCC benign sample had **6** histopathology. **D5 over the whole corpus** (`results/v5/diagnostics/d5_acquisition.json`): histopathology among benign = HAM 3,386/8,061, BCN 3,713/5,579, MSKCC 338/2,351 (1,899 missing); 4,370 of 25,331 images have no confirm type; polarisation is missing for almost every image, so D5 cannot inform M7's targets. Under the old rule fold-0 pAUC_histo read 0.7629; corrected it is 0.7412 | ISIC API, 2026-09-30 | One definition, `research/v5/confirmation.py`: histopathology in any archive, from D5 (`research/v5/d5_acquisition.py`, metadata only). `m4` and pAUC_histo now need D5 |
| **AU28** | Major | **Disk:** the revised queue writes ≈ 16–18 GB (≈ 59 screen/LOAO checkpoints × 0.111 GB + ≈ 33 confirmation/fold-0/control runs × 0.222 GB + S01 1.6 GB + young images 1.37 GB + SWAD 0.7 GB) against **26 GB free** now; C2 needs ≥ 10 GB free before each run, and §11.4 forbids deleting checkpoints during V5 | `df` 2026-09-30 | **Owner frees ≥ 10 GB** (Downloads / `.ollama`, per the hardware notes) **before Q7**, or moves finished screen checkpoints off C:. The queue's C2 check stops the run rather than failing it |
| **AU29** | Minor | S01's hurdle CI: a t-interval over 3 seed pairs needs a mean ≳ 0.08 to exclude 0 (t₂ = 4.30 × 0.0325/√3), so read literally it can never pass | A01 A5 | The **hierarchical bootstrap** (seed pairs, then lesions) is decisive; the t-interval is descriptive |
| **AU30** | Major | Gate D requires **mean Macro-F1 > 0** and every seed ≥ −0.010. For a composite that improves ranking and leaves Macro-F1 flat, that fails about half the time on the mean alone, contradicting Gate B's *retention* | Master §Gate D vs Gate B | Gate D's direction rule applies to the ranking endpoint; Macro-F1 is judged by Gate B (retention) only |
| **AU31** | Info (expectation) | Gate A (Δ <40 pAUC ≥ +0.050 on 65 lesions) is large against the record: the biggest V4 move was 0.011, and a single model's <40 pAUC on folds 1–4 has a lesion-bootstrap SD of **0.033** (S72 OOF, 1,000 resamples; point 0.725) | computed 2026-09-30 | Gate A kept as written; read as point ≥ +0.050 **and** hierarchical CI lower bound > 0. A Gate-A pass is not the expected outcome; all-age and histo-only pAUC are where a positive result is most likely to be demonstrable |
| **AU32** | Major | Screen retention (Macro-F1 Δ ≥ −0.010) is tighter than seed noise: with SD 0.023 a 3-seed mean Δ of a do-nothing arm falls below −0.010 about 30% of the time | calculation | Retention fails only below both −0.010 and −1.645 × pair SD / √k |
| **AU33** | Major | `youngdata` raises the young escalation prior (≈ 7% → ≈ 14% of young training images). A gain could be a learned age prior — the barred λ(age) mechanism (S57b/S65/S66) — rather than ranking | registry item 7 | Falsifier: band-stratified pAUC (mean of within-band pAUCs) must rise and <40 within-band pAUC must not fall |
| **AU34** | Info (expectation) | DINO-family features lost to the in-repo ConvNeXt under 40 in S51 (frozen DINOv2 −0.0725 [−0.118, −0.029]). Fine-tuning is not barred (registry bars only frozen probes), but the prior is negative | CHANGELOG §S51 | IN-22k (never tested here) is the stronger candidate; DINOv3 is kept only because it is the same architecture and cheap to screen |
| **AU36** | Info | IN-22k is the primary trunk candidate, DINOv3 secondary. Weights downloaded 30 Sep; remap on the real weights matches timm exactly (max abs diff 0.0) | `research/v5/trunks.py`; owner approval 30 Sep | Runsheet §6.0. IN-22k has never been screened here; the only relevant prior is S51's negative frozen-DINO result |
| **AU35** | **Critical (schedule)** | `train_v5` refuses `m5`, `zoom`, `logic` and `m7`: **not implemented** (`train_v5.run`) | code | Q4 (zoom, m5), Q5 (m7) and Q6 (logic) need that code, smoke-tested, before their block. If not ready, the arm is skipped and recorded, not rescheduled after results |

---

## 2. Technical corrections already applied in A02 (revision of 2026-09-29)

| Item | Problem | Correction |
|---|---|---|
| M1 OD transform | The earlier draft wrote −log(I + 1/255), which is negative at I = 1. Gamma-encoded sRGB breaks Beer–Lambert additivity | sRGB linearisation → clamp [1/255, 254/255] → OD = −ln(I); glare pixels excluded from the basis fit |
| M1 depth channel | The proposed OD_B/OD_R ratio had its physics reversed (red is *absorbed* by deep melanin; lower ratio = deeper). It is unbounded where OD_R ≈ 0 and driven by haemoglobin | Bounded log-ratio, centred on the skin ring and **melanin-gated**. Explicitly not a vascular channel |
| M1 stem | A 1×1 6→3 adapter forces a rank-3 per-pixel bottleneck | Widened 6-channel patchify stem, new slices zero-initialised (≈ 4.6k parameters) |
| M3 | Pooling not specified | Normalised LSE, r = 4, on the **stride-16** map; patch maps saved |
| M4 | Pair starvation at batch 16; follow-up masking implementation unspecified | Per-micro-batch pairs, 0 when empty, mean \|P\| logged; per-sample `esc_weight` |
| M5 | Masks would be misaligned by image-only augmentation; no unlabelled-row handling | Paired `transforms.v2` with `tv_tensors.Mask` plus a unit test; `has_attr` indicator with a safe denominator |
| M6 fallback | Averaging D₄ axes blurs the Menzies "any axis" vs Kittler "some axis" distinction | min and max over the 4 axes, contrast-normalised; the chromophore asymmetry is pigment-load-normalised |

---

## 3. Compute re-sum (measured anchors; unmeasured items flagged)

| Block | Runs | Hours |
|---|---|---|
| N1 S01 | 3 × 384 + 2 × 224 | 4.6–5.1 |
| Day 30: control s45/46 | 2 × 224 | 1.4 |
| N2 | ≤ 10 × 224 | ≤ 6.8 (+ ≤ 10% heads, smoke-timed) |
| N3 | 8 × 224 + 2 zoom | 7.6–8.2 (zoom unmeasured) |
| N4 | LOAO 6 runs + 4 × 224 composite | 6.1–8.9 (LOAO extrapolated; up to 5.5 h if zoom is in the composite) |
| N5–N7 | 4 × composite each | 4.3–5.0 each (×≈1.34 with zoom, unmeasured) |
| Daytime control fold runs | 12 | 12.8–15.0 (384) or ≈ 8.2 (224) |
| N8 | fold 0 × 2 + TTA + 8b/8r | ≈ 5–7 (TTA unmeasured) |
| N9 | S84 | ≤ 1 |

Every night fits under 9 h at the measured anchors. The two nights at risk are N3 and N4, and
both depend on unmeasured zoom cost. Their fallback is written down (runsheet §8.3).

---

## 4. Registry check (master §21, 12 barred mechanisms)

| New V5 item | Nearest barred or historical item | Why it is not a duplicate |
|---|---|---|
| M1 chromophore stem / depth | 11 colour constancy (buggy); S53r corrected colour null | Adds physically defined channels; does not normalise the illuminant |
| DRE-10 DSP | 2 tabular metadata fusion | Image-derived structure maps; no metadata |
| DRE-2/3/4 geometry | R3 mask-guided crop (dropped at S50) | No crop and no exterior dropout; geometry drives the asymmetry/periphery statistics |
| DRE-6 memory | 1 frozen specialist head | Trained end-to-end with the trunk |
| GeM | B11 (a frozen-plan arm) | Used as the **mechanism control** for Clues, not as a standalone candidate |
| M4 | S67 hard-case reweighting; R4 class weighting | Uses clinical verification status, not model errors or class frequency |
| M7 (+ arXiv 2607.26765 components) | HSV-jitter ban; R5 RandAugment null (in-distribution) | Physically plausible illuminant/chromophore perturbation; judged on a **new endpoint** (LOAO), which the registry allows |
| SWAD secondary | R6 EMA (null, in-distribution) | New endpoint (out-of-archive); no extra training |
| DRE-8 zoom | A2 dual-stream (deferred) | Shared trunk, evidence-driven crop, no masks; compute-matched random-crop control |
| S84 on MILK10k | 12 repeated reserved reads | A different, fresh cohort; single read enforced by a receipt |

No new item is materially equivalent to a barred mechanism.

---

## 5. Residual risks accepted

1. **Power under 40.** 65 under-40 escalating lesions in folds 1–4. Only a large under-40 gain can
   be certified; smaller effects are reported as descriptive, with mechanism evidence.
2. **MILK10k is partly same-institution** (Vienna, MSKCC). It is labelled as such. Deduplication
   by ISIC ID and perceptual hash removes known overlaps, but it cannot rule out patient overlap.
3. **Unmeasured costs** (zoom, DSP, M5 head, M7 CPU augmentation, TTA) are timed before use, and
   each has a written fallback.
4. **Literature cited from memory** (A02 §11, DRE §9) must be verified before the paper, not
   before the runs.

---

## 6. Adoption steps (morning of 30 Sep, before any pre-check) — executable version in runsheet §1

1. The owner accepts or rejects each item of A02, the DRE and the runsheet.
2. Compute the SHA-256 of each of the three files; on Windows, `certutil -hashfile <file>
   SHA256`.
3. Append `amendments[1]` to `results/v5/v5_plan_freeze.json` with the three hashes, the adoption
   date and the per-item decisions.
4. Record the adoption in `CHANGELOG.md`. From then on, any change goes into Amendment 03 and is
   labelled post-hoc if it comes after any V5 result.
