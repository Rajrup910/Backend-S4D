# V5 Runsheet — the single V5 execution document

**Date:** 2026-09-29
**Status:** **ADOPTED 2026-09-30** by the owner (every item accepted), hashed **together with**
`docs/v5_design/V5_PLAN_AMENDMENT_02.md` and `docs/v5_design/V5_DERM_REASONING_ENGINE.md` as
`amendments[1]` in `results/v5/v5_plan_freeze.json`, before any CPU pre-check (audit AU5). From
here on, any change goes into an Amendment 03, labelled post-hoc if it follows any V5 result.

**Supersedes:**
- the night tables in Amendment 02 §7 and DRE §6, which now point here (AU1);
- **where they conflict, the frozen pre-amendment documents `docs/v5_record/V5_FINAL_RUNSHEET.md` and
  `docs/v5_record/V5_SESSION_PLAN.md`**. Those files are hash-frozen and **not edited**. Their
  S78–S85 queue (dual-stream, wavelets, SupCon, GeM arm, Group-DRO, the three backbones, the
  X-queue) is replaced by §6–8 below, as recorded in Amendment 02 §5. Amendment 01's preamble
  gives an amendment precedence over those files.

The reasons behind every rule are in `docs/v5_design/V5_PLAN_AUDIT.md`.

**Revision 2026-09-30 (before the hash; audit AU17–AU26).** Pre-hash audit aimed at the best
case: a trunk screen (IN-22k / DINOv3 ConvNeXt-T) before the arms, a screen gate whose null
matches its statistic (3 seeds, 80th percentile, all-age pAUC **or** pAUC_histo), a 384 px rescue
for fine-structure arms, a train-only young-data arm, a 15-model V5 system, per-arm calibration
reporting, and one continuous GPU queue. These replace the corresponding text in §2, §4–§6, §8,
§9 and §11 below. The confirmation folds still hold only 65 under-40 escalating lesions; nothing
here changes that.

**Owner decisions recorded here (2026-09-29):**
- the GPU may train in free **daytime** windows as well as nights;
- **MILK10k** is V5's single confirmatory external cohort (S84), and becomes V6 training data
  only after that read;
- **Review 2 (9 Oct)** shows whatever V5 has finished by 8 Oct, plus the V6 plan. The rest of V5
  runs afterwards, in the order below, **under the same hash**, and is reported as a V5 addendum.

---

## 0. How to use this file (file map)

**Execute V5 from this file only.** The other V5 files hold the rationale or the frozen record:

| File | Role | May it change? |
|---|---|---|
| `docs/V5_RUNSHEET.md` (this file) | **Execution:** every rule, check, arm, parameter, night and command; the **session plan is §11** | Until the hash (§1); afterwards only through an Amendment 03 |
| `docs/V6_RUNSHEET.md` | The single V6 plan and execution document (draft for Review 2) | Yes, until the V6 freeze |
| `docs/v5_design/V5_PLAN_AMENDMENT_02.md` | Why: skin optics, class biology, M1–M7 rationale, protocol changes | Hashed **with** this file |
| `docs/v5_design/V5_DERM_REASONING_ENGINE.md` | Why: the dermatologist-reasoning modules (DRE-0…10) | Hashed **with** this file |
| `docs/v5_design/V5_PLAN_AUDIT.md` | The audit behind the rules here (AU1–AU15) | Reference |
| `docs/v5_design/V5_IDEAS_BIOLOGY_FIRST.md` | Idea pool and D0 diagnostics | Reference |
| `docs/v5_record/` (master plan, frozen runsheet, session plan, Amendment 01) | **Frozen pre-registration record** (sha256 in `results/v5/v5_plan_freeze.json`) | **Never.** Moving or editing breaks the record |
| `docs/RESEARCH_LOG.md` | Programme history | Append-only |

**Precedence when two texts disagree:**
this file → Amendment 02 → DRE → Amendment 01 → the frozen record. Amendment 01's preamble gives
amendments precedence over the frozen files, and this file is adopted together with Amendment 02
and the DRE.

---

## 1. Adoption and hashing (Wed 30 Sep morning, before any check or run)

1. The owner accepts or rejects each item in this file, Amendment 02 and the DRE. Record the
   per-item decisions.
2. Compute the three hashes **after the last edit**:

```bash
python -c "import hashlib,sys;[print(hashlib.sha256(open(f,'rb').read()).hexdigest().upper(),f) for f in sys.argv[1:]]" docs/V5_RUNSHEET.md docs/v5_design/V5_PLAN_AMENDMENT_02.md docs/v5_design/V5_DERM_REASONING_ENGINE.md
```

3. Append `amendments[1]` to `results/v5/v5_plan_freeze.json` with:
   - the three paths and sha256 values;
   - the adoption timestamp and `adopted_before_first_v5_gpu_run`. This is `true` for the
     Amendment 02 items; S01 already ran under the master plan and Amendment 01;
   - the item decisions.
4. Re-verify that every existing hash in the freeze file still matches. Then add a CHANGELOG
   entry.
5. **Only then** start §4 (the CPU pre-checks). Their pass rules decide which arms run, so they
   must be fixed before the checks run (audit AU5).

---

## 2. Fixed rules for every run

| Rule | Value | Source |
|---|---|---|
| Workers | `--num-workers 2` | Error 1455 at 4 workers; 2 = 3 at 384 px (306 vs 307 ms/batch) |
| Schedule | 30 epochs, `--patience 0` | A01 A3 / S54 convention |
| Primary checkpoint | `_last` | A01 A3 |
| Checkpoints saved | Screens/LOAO: `_last` only (`--no-save-best`). Confirmation (folds 0–4): both (`--save-best`) | A02 §6.5 |
| Disk guard | ≥ 5 GB free on C: before each run; trainer refuses below it | Hardware notes 2026-09-16 |
| Smoke | `--smoke` (2 batches, 1 epoch) before each new arm or night | V4 practice |
| 384 px batch | 16 × `--grad-accum 2` (effective 32) | Banked R1+R4 runs |
| 224 px batch | 32 × 1 | S72 |
| Test / reserved | **Never read.** Receipts stay at test 2 / reserved 8 | Standing rule |
| Declared escalation score | Head logit s_esc if the arm has an escalation head; otherwise `escalation_mass` (sum of softmax on mel/bcc/akiec). Escalation mass is logged for **every** arm as a secondary | AU4 |
| Screen status | A screen is a **compute-allocation filter, not a test**. Only folds 1–4 carry confirmatory claims | AU12, V4 S52 |

**Anchors:**
- 224 px fold run = **41–42 min (measured, S72 `results/v4/kfold/run.log`: folds 42.0 / 41.0 /
  42.0 / 42.0 / 42.0 min; fold 0 was 42.0)**;
- new trunks (IN-22k, DINOv3) are the same architecture; their cost is **unmeasured** until the
  trunk smokes, and 41–42 min is assumed until then;
- 384 px fold run = **64–75 min (measured range)**;
- LOAO runs ≈ 28 / 29 / 46 min (**extrapolated** from training-set size; time the first);
- DRE-8 zoom ≈ 1.5–2× per step (**unmeasured**: run `scripts/gpu_benchmark.py` in the unfrozen
  stage);
- DSP / M5 / M7 overheads **unmeasured**: timed in the smoke epoch.

---

---

## 3. Commands for night 1 (S01, existing `train_v4`)

Smoke first:

```bash
python -m research.v4.train_v4 --rungs R1 --corpus pooled --fold 0 --seed 42 --batch-size 16 --grad-accum 2 --num-workers 2 --patience 0 --device cuda --smoke
```

**384 px:** run for seeds 42, 43 and 44.

```bash
python -m research.v4.train_v4 --rungs R1 --corpus pooled --fold 0 --seed 42 --batch-size 16 --grad-accum 2 --num-workers 2 --patience 0 --device cuda --run-tag v5s01
```

**224 px:** run for seeds 43 and 44. Seed 42 is the banked S72 fold-0 run.

```bash
python -m research.v4.train_v4 --rungs R0 --corpus pooled --fold 0 --seed 43 --batch-size 32 --num-workers 2 --patience 0 --device cuda --run-tag v5s01
```

---

---

## 4. CPU pre-checks and their pass rules (30 Sep, after §1; development rows only)

None of these reads test, reserved or external labels. Outputs go to `results/v5/diagnostics/`.

| ID | Question | Pass rule (fixed) | If it fails | Output |
|---|---|---|---|---|
| **M1-QC** | Does the chromophore decomposition behave? (20 HAM images per class) | vasc/bcc mass lies mainly in c_hb; blue-grey regions sit low on c_depth (visual sheet reviewed by the owner) | `look` and `structure` do not run | `m1_chromophore_qc.png` |
| **Q1** | Does the chromophore lesion mask (DRE-2) match the HAM expert masks? | Median Dice ≥ 0.75 **and** centroid error ≤ 10% of the diagonal | `geometry` uses the D₄ fallback (§7) | `q1_geometry_qc.json` |
| **Q2 / D6** | Do handcrafted colour count, colour asymmetry and segmental index separate <40 mel from <40 histo-nv? (lesion-grouped AUC) | ≥ 1 feature with AUC lower CI bound > 0.5 | `geometry` does not run; reported as a finding | `d6_chaos_probe.json` |
| **Q5** | DSP: textbook signatures (tubular:blob highest in bcc; network {mel, nv} > rest; regularity nv > mel; Clark–Evans nv > mel; veil mel > nv) **and** young differential | ≥ 3 of 5 signatures (CI excludes 0, right direction) **and** ≥ 2 tokens with <40 AUC lower bound > 0.5 | `structure` does not run; reported as a finding | `q5_dsp_probe.json` |
| **Q4** | Task-2 masks: overlap by split; 12 under-40 MSKCC mel (reported only); pixel agreement on 50 images | ≥ 1,000 **train** rows; **0** val/reserved rows used; pixels agree | `m5` does not run | `m5_overlap.json` |
| **B1** | Hard core: which of the 81 under-40 escalating lesions the control misses, and their anatomy | Descriptive | — | `b1_hard_core.json` |
| **D2** | Does the perilesional ring alone predict age band or escalation? | Descriptive (feeds V6 B5) | — | `d2_context_probe.json` |
| **D4** | Is there a low-pAUC, under-40-enriched melanoma subclass? | Descriptive (gates V6 subclass-DRO) | — | `d4_subclasses.json` |
| **D5** | ISIC polarisation / verification metadata per archive (API, if available) | Descriptive (M4 benign pool; M7 targets) | M4 keeps the default benign pool | `d5_acquisition.json` |
| **S75** | MILK10k eligibility (§9) | Counts only; no predictions | — | `s75_milk10k_eligibility.json` |
| **B3 noise floor** | Control seeds **42–47** at 224 px (in1k): pooled pair SD for all-age pAUC, pAUC_histo, Macro-F1, under-40 pAUC (`research/v5/screen_gate.py --null-only`) | Written **before** any screen is read | — | `results/v5/screens/noise_floor.json` |
| **YD** (AU24; count done 30 Sep: 2,353 candidates, 535 escalating, 1.37 GB, 20 leaking lesions removed) | Young-data pool: ISIC Archive, age ≤ 35, histopathology, dermoscopic; minus the V4 corpus, MILK10k, HIBA, PAD-UFES-20; 7-class mapping. **Metadata only** (`research/v5/young_data.py --count`) | Owner approves the download from the count; rows ready **by Thu 1 Oct 20:00**, else the arm moves to V6 | `youngdata` does not run | `results/v5/young_data/count.json` |

---

## 5. Planned `train_v5` interface (implemented on 30 Sep; the arm registry is fixed by this document)

```
python -m research.v5.train_v5 --arm <ARM> --fold <k> --seed <s> --image-size {224|384}
       --batch-size {32|16} --grad-accum {1|2} --num-workers 2 --patience 0 --epochs 30
       [--no-save-best | --save-best] [--loao-holdout {ham|bcn20000|mskcc}]
       [--swad-window 10 30] [--trunk {in1k|in22k|dinov3}] [--extra-train <csv>]
       --run-tag <tag> --device cuda [--smoke]
```

- **`--trunk`** (AU17): the pretrained weights of the same torchvision ConvNeXt-T
  (`research/v5/trunks.py`; timm `convnext_tiny.fb_in22k_ft_in1k` [`_384` at 384 px] or
  `convnext_tiny.dinov3_lvd1689m`, remapped key by key and loaded strictly). Every module is
  unchanged. The trunk is part of the run id (`<arm>_f<k>_s<s>_<trunk>_<tag>`).
- **`--extra-train`** (AU24): train-only rows for the `youngdata` arm. They are appended to the
  training frame only; the trainer refuses any row sharing an image or lesion group with the
  development partition.

- **Arm registry:** `research/v5/arms.py`, one frozen spec per arm (modules, loss weights, declared
  score).
- **Parity:** `--arm control` must reproduce `train_v4 --rungs R0` (same partition, recipe and
  seed). The pass rule is a 2-epoch smoke whose per-step loss matches within 1e-3.
- **Outputs:**
  - `results/v5/runs/<arm>_f<k>_s<s>_<tag>.json`;
  - per-image predictions `results/v5/preds/<arm>_f<k>_s<s>_<tag>.csv` (7 probabilities, declared
    score, escalation mass);
  - patch maps `results/v5/patchmaps/<arm>_f<k>_s<s>.npz` (float16);
  - ledger rows in `research/experiments.csv` (session `v5_*`).

---

---

## 6. Arm inventory (screens: 224 px, fold 0, seeds 42, 43 and 44)

**6.0 Trunk screen (AU17, AU36), first.** `control --trunk in22k` and `control --trunk dinov3`, fold 0,
224 px, seeds 42/43/44, each gated against the in1k control with the screen gate below.
**IN-22k is the primary candidate** (`convnext_tiny.fb_in22k_ft_in1k`, 22k-class supervised then
1k fine-tune; never tested in this repo, and the DINO family lost under 40 in S51); **DINOv3 is
the secondary** (same architecture, so it costs nothing extra to screen). At 384 px the trunk is
`convnext_tiny.fb_in22k_ft_in1k_384`. Both weight sets are downloaded and their remap into the
torchvision trunk reproduces timm's forward pass exactly (max abs diff 0.0, 30 Sep). If **both**
pass, the larger mean Δ all-age pAUC wins; a tie (difference below the noise floor) goes to IN-22k. If one
passes, the one with the larger mean Δ all-age pAUC becomes **the trunk for every arm, the
composite and the composite's confirmation**; if both fail, the trunk stays in1k. The control
fold runs (§8.1) stay on in1k, the pre-registered V4 comparator, so the confirmatory contrast is
the V5 system (trunk + modules) against the V4 control; the screens give the descriptive split
between trunk and modules (AU26).

| Arm | Modules | Comparator | Pre-condition (CPU, 30 Sep) | Declared score | Declared hard-core subset (A01 B1) | Falsifier (from A02 / DRE) |
|---|---|---|---|---|---|---|
| `control` | V4 R0 recipe | — (seeds 42 banked, 43/44 S01, 45/46 day 30) | parity smoke | escalation mass | — | — |
| `look` | M1 chromophore 6-channel stem + DRE-1 palette tokens | control | M1-QC | escalation mass | <40 esc in the lowest melanin tercile; bcc/vasc F1 | Haemoglobin shuffle removes the gain; depth ablation hits blue-grey lesions |
| `structure` | look + **DRE-10 DSP** (maps + tokens) | look | **Q5** | escalation mass | <40 mel missed as nv | Token ablation removes the gain; rescued <40 mel score higher on GLOBULES_IRREGULAR / NETWORK_ATYPICAL / VEIL |
| `geometry` | DRE-2/3/4 + border abruptness | control | Q1 (else D₄ fallback); Q2 / D6 | escalation mass | <40 mel vs histo-nv with small control score gap | Per-sector rotation of c(φ) removes the gain |
| `twostep` | M2 | control | — | s_esc | <40 bcc/akiec → nv/bkl | Fewer melanocytic ↔ non-melanocytic confusions |
| `clues` | M2 + DRE-5 (M3 LSE r = 4 on F3 + eccentricity) | twostep | — | s_esc | <40 mel missed as nv | Rescued lesions show eccentric evidence (HAM masks) |
| `gem` | M2 + GeM (p = 3, learnable), all heads | clues (mechanism control) | — | s_esc | as clues | — (it is the control) |
| `m4` | M2 + M4 masking + hinge ranking m = 0.20 | twostep | — | s_esc | <40 mel vs histo-nv | pAUC_histo must improve |
| `m5` | Task-2 expert structure heads | control | **Q4** | escalation mass | MSKCC rows | Held-out Dice > trivial; gain on MSKCC first; mask-shuffle removes it |
| `memory` | DRE-6 multi-prototype head | control | — | escalation mass | <40 mel | <40 mel concentrate on ≤ 2 prototypes that are not the 60+ one; deleting them costs <40 sensitivity |
| `zoom` | clues + DRE-8 | clues | benchmark | s_esc (noisy-OR) | BCN/MSKCC small focal | Gain larger on BCN/MSKCC; random location shrinks it |
| `control` LOAO / `m7` LOAO | V4 recipe / + M7 (with arXiv 2607.26765 components) | control LOAO | first LOAO run timed | escalation mass | per-archive <40 (H-link) | Archive decodability falls **and** LOAO Macro-F1 rises. SWAD is a secondary |
| `composite`, `composite_nologic` | Passing arms (+/− DRE-7) | each other; control | composite lock | per lock | union of the component subsets | DRE-7 sign check in both seeds |
| `zoom_lesion` (8b), `zoom_random` (8r) | zoom with a DRE-2 lesion crop / a random crop | zoom | zoom passed | s_esc | — | Descriptive, 1 seed |
| `youngdata` (AU24, AU33) | control + train-only young histology-confirmed rows (`--extra-train`) | control | YD count + owner-approved download | escalation mass | <40 escalating, all archives | The gain must be **within-band**: the band-stratified pAUC (mean of the <40, 40–59, 60+ within-band pAUCs) must rise, and the <40 within-band pAUC must not fall. A gain that exists only in all-age pAUC is a learned age prior (the barred λ(age) mechanism, S57b/S65/S66), not ranking |

- **Nested arms** (structure/look, clues/twostep, gem/clues, m4/twostep, zoom/clues) run **on the
  same queue block as their parent**. The comparison is read at the next read (AU10).
- **Screen gate** (AU18, AU19; replaces A02 §6.2; `research/v5/screen_gate.py`):
  - statistic = the mean over seeds 42/43/44 of (arm − comparator) on the same seed;
  - null = the k-seed-mean difference under no effect: SD = pooled pair SD of the control seeds
    42–47 / √k, normal approximation;
  - **pass** if the statistic exceeds the **80th percentile** of that null on **all-age
    escalation pAUC@0.20 or pAUC_histo** (escalating rows + **histopathology-confirmed** benign
    rows in any archive, from D5; AU27), **and** Macro-F1 is retained, **and** the falsifier
    passes. Retention fails only if the mean Macro-F1 Δ is below both −0.010 and
    −1.645 × pair SD / √k (AU32): with seed noise near 0.023, a fixed −0.010 alone would reject
    about 30% of arms that change nothing.
  - Why 80th, not 95th: the screen is a compute filter (AU12). Illustration only, taking the
    HAM-val Macro-F1 seed SD 0.023 as the per-seed noise (the fold-0 pAUC noise is measured by
    `screen_gate.py --null-only`): a real +0.030 effect on one endpoint passed the old gate
    (2 seeds, 95th percentile of a single-seed null) about 15% of the time and passes this one
    about 78% of the time. The price: a null arm passes 20–36% of the time (two endpoints OR-ed)
    instead of about 1%. Confirmation on folds 1–4 remains the only test, and the stacking check
    (Q6) catches arms that do not add up.
  - **384 px rescue** (AU20): if S01 picks 384 px, and `structure`, `zoom` or `m5` fails at 224
    px with a positive point estimate on either endpoint, it is re-screened at 384 px (seeds
    42/43, arm and parent, same gate). Fine structure is what 224 px discards.
  - **Descriptive for every arm** (AU25): under-40 pAUC; raw ECE and signed gap (confidence −
    accuracy), overall and per age band. Calibration after the Dirichlet map is read only on the
    composite, from cross-fitted OOF (a fold-0 map would be fitted on the rows it scores).

  Under-40 values on fold 0 are descriptive only.

**Composite rule (fixed):**
- Include every arm that passes its gate and its falsifier, on the trunk chosen in §6.0.
- `youngdata` is included if it passes; its rows then enter the composite's training on every
  fold (still train-only).
- For a nested pair, include the child only if it beats its parent.
- `gem` is included only if it beats `clues`.
- `m7` is included only if LOAO Macro-F1 exceeds the null and fold-0 retention is ≥ −0.010.
- DRE-7 (logic) is included only if its sign check holds in both seeds and `composite` beats
  `composite_nologic`.
- If nothing passes, confirmation runs control vs control (the noise floor), and V5 reports a
  mechanism-level null.

**Gate definitions, fixed before the hash (AU29–AU31):**
- **S01 hurdle:** the hierarchical bootstrap (resample seed pairs, then lesions) decides; the
  t-interval over the 3 paired deltas is descriptive only (with 2 df it would need a mean ≳ 0.08).
- **Gate A** (folds 1–4, 65 <40 lesions): point Δ <40 pAUC ≥ +0.050 **and** hierarchical CI lower
  bound > 0. Expectation, stated now: V4's largest move was 0.011 and a single model's <40 pAUC
  has a lesion-bootstrap SD of 0.033, so a Gate-A pass is not the expected outcome.
- **Gate B:** Macro-F1 retention, point Δ ≥ −0.010 (hierarchical CI reported).
- **Gate D:** the direction rule (mean > 0, ≥ 2 of 3 seeds > 0, no seed < −0.010) applies to the
  ranking endpoint (all-age pAUC; <40 pAUC descriptive). Macro-F1 is judged by Gate B only; a
  composite need not *raise* Macro-F1 to pass.

**Composite lock:** after the Q6 stacking check (target Fri 2 Oct), write `results/v5/composite_lock.json` (arm
spec, declared score, resolution) and record its sha256 in the CHANGELOG.

---

---

## 7. Module parameter cards (every value fixed; nothing is tuned)

**About these values:**
- The rationale is in the design files (§0).
- Values marked **[declared here]** were left open in the design files. They are fixed now, before
  the hash (audit AU15).
- F3 = ConvNeXt stage-3 map (stride 16, 384 channels: 14×14 at 224 px, 24×24 at 384 px).
- Pixel sizes are quoted at 224 px and scale linearly with resolution.

**M1 / DRE-0 — Chromophore stem**
- **Colour transform:** sRGB → linear. Clamp to [1/255, 254/255]. OD = −ln(I).
- **Basis:** a per-fold ICA basis (melanin, haemoglobin, shading) fitted on 10⁶ training-fold
  pixels. Glare (any sRGB channel ≥ 250/255) is excluded from the fit.
- **Depth channel:** c_depth = tanh(ln((OD_B + 0.02)/(OD_R + 0.02)) − μ_skin) · σ((c_mel − t_mel)/(0.1·t_mel)).
  μ_skin is the median of the outer 10% ring; t_mel is the fold's median lesion melanin. Lower
  means deeper.
- **Stem:** [RGB, c_mel, c_hb, c_depth] with the chromophores standardised per fold. The stem is
  widened 3 → 6 channels, and the new slices are zero-initialised.

**DRE-1 — Palette**
- K = 6 OD prototypes per fold (k-means on the fold's lesion pixels), then frozen.
- a_k = softmax(−‖x − c_k‖²/τ) with τ = 0.05 **[declared here]**.
- Tokens:
  - area fraction per colour;
  - soft colour count Σ σ((area_k − 0.05)/0.01);
  - eccentricity per colour.

**DRE-2 — Geometry (no gradient)**
- ℓ = max(0, c_mel − m_skin) + β·max(0, c_hb − h_skin), with β = 1.0 **[declared here]**. The skin
  baseline is the median of the outer 10% ring.
- s = σ((ℓ − t)/w), with t from per-image Otsu and w = 0.1·t **[declared here]**.
- Take moments of s → μ, principal axes, and r = 2√λ₁.
- **Fallback** (image centre, image axes, flag token) if the mask covers < 3% or > 95%, or if Q1
  failed.

**DRE-3 — Chaos**
- F3 and the palette maps are rotated into the lesion frame (`grid_sample`) and mirrored about both
  principal axes.
- Tokens: min and max over the axes of the structure asymmetry A_s and the colour asymmetry A_c,
  both normalised.
- Pattern count: the entropy of assignment to P = 8 learned pattern prototypes. The orthogonality
  penalty weight is 0.01 **[declared here]**.
- **Fallback (Q1 failed):** the D₄ reflections g_k = rot_{90k}∘flip_h, k = 1…4; min_k and max_k of
  the normalised distance.
- **Chromophore asymmetry:** A_chrom = Σ s·(|c_mel − g(c_mel)| + |c_hb − g(c_hb)|) / Σ s·(c_mel + g(c_mel) + c_hb + g(c_hb) + 1e-6),
  taking min and max over the axes.

**DRE-4 — Periphery and border**
- Polar-warp F3 over the annulus ρ ∈ [0.6r, 1.2r], with 16 sectors × 3 radial bins.
- A 1×1 conv gives the peripheral channel c(φ).
- Tokens: S = 1 − H(c/Σc)/ln 16; rim coverage; radial gradient.
- **Border abruptness:** the radial derivative of c_mel over ρ ∈ [0.85r, 1.15r] per sector,
  normalised by the lesion median. A sector is abrupt if it is above the fold's training 75th
  percentile. Tokens: abrupt fraction and angular variance.

**DRE-10 — DSP (fixed filters, GPU)**
- **Vessels:** Hessian of c_hb at σ ∈ {1, 2, 4} px → Frangi-type tubularity (β_F = 0.5,
  c_F = half the max Hessian norm per image) and blobness |λ₁|/|λ₂| **[declared here]**.
- **Network:** Gabor on c_mel, 4 orientations × wavelengths {6, 10} px, σ = 0.56·λ
  **[declared here]**.
- **Dots/globules:** LoG on c_mel at σ ∈ {1.5, 3}, 3×3 max-pool NMS, threshold = lesion mean +
  2 SD.
- **Veil:** c_depth < the fold's lesion-pixel 25th percentile ∧ c_mel > the fold median ∧ Gabor
  energy < the fold's 25th percentile **[declared here]**.
- 5 maps are standardised per fold and fed as zero-init stem channels (6 → 11). About 18 tokens go
  to the head and to DRE-7.

**M2 — Two-step**
- BCE heads for melanocytic {mel, nv} and escalating {mel, bcc, akiec}, weight 0.5 each. The
  7-class CE is unchanged.

**DRE-5 / M3 — Clues**
- P = Conv1×1(F3). s_esc = (1/4)·ln((1/K)Σ exp(4·pᵢ)).
- Eccentricity token = ‖μ_e − μ‖/r.
- P is saved at inference (float16).

**GeM**
- p initialised at 3, learnable, on the pooled feature for all heads.

**M4 — Clinical differential**
- `esc_weight` = 0 for HAM `follow_up` rows, in the escalation BCE and in ranking. The 7-class CE
  is unchanged.
- Confirmed benign = **histopathology-confirmed** in any archive (D5, `research/v5/confirmation.py`;
  AU27). Not "all BCN/MSKCC benign": 35% of BCN benign are not histopathology.
- Hinge ranking: margin 0.20 on s_esc, weight 0.5, all (mel, confirmed-benign) pairs per
  micro-batch; 0 if there is no pair. Mean |P| is logged.

**M5 — Expert structures**
- Conv3×3(384→128) → GELU → Conv1×1(128→5) on F3.
- BCE + soft Dice, weight 0.20. Area-downsampled targets.
- `has_attr` indicator with denominator max(1, Σ has_attr).
- Paired geometric transforms (`transforms.v2` + `tv_tensors.Mask`).

**DRE-6 — Subtype memory**
- K: mel 8, nv 8, bcc 4, bkl 4, akiec/df/vasc 2 each.
- Features and prototypes are L2-normalised: logit_c = s·τ·LSE_k(cos(f, w_{c,k})/τ), with s = 16
  and τ = 0.1 **[declared here]**.
- Orthogonality penalty 0.01 **[declared here]**.

**DRE-7 — Logic residual**
- Rules as in DRE §3 (R_benign, R_chaos_clue, R_sk, R_nonmelanocytic), product t-norm.
- α initialised at 0. Missing concepts are held at the neutral value.
- Sign check: α(R_chaos_clue) > 0, α(R_benign) < 0 and α(R_sk) < 0 in both seeds.

**DRE-8 — Zoom**
- A 448 px short-side copy is kept.
- p* = argmax P (stop-grad). The crop side is max(0.35·2r, 0.25 × the short side), resized to
  224, through the shared trunk.
- P_esc = 1 − (1 − P_g)(1 − P_z).
- 8b: DRE-2 box × 1.2. 8r: a random location of the same size.

**M7 — Acquisition randomisation** (LOAO only; each p = 0.5 unless stated):
- OD exposure Δ ~ U[−0.08, 0.08];
- chromophore scale ×(1 ± U[0, 0.10]);
- Planckian illuminant 6,500 ± 1,500 K;
- aperture radius U[0.45, 0.60] × the short side, plus vignette U[0, 0.3];
- CLAHE p = 0.4 (clip 4, 8×8, luminance only);
- OneOf(optical, grid distortion);
- OneOf(Gaussian blur ≤ 5, median blur ≤ 5, noise σ 0.02–0.11);
- resample U[0.4, 1.0] then JPEG quality U[70, 95];
- RRC scale U[0.6, 1.0];
- 360° rotation + transpose.

**SWAD (secondary, LOAO only)**
- Dense weight average over epochs 10–30, saved as `_swad.pt`.

---

## 8. Schedule: one continuous GPU queue (AU21, AU22)

The GPU runs day and night through `scripts/run_v5_queue.ps1`. It stops only for smokes, the
morning reads and the owner's own use; `results/v5/PAUSE` stops it between runs. Order is by
dependency, so a slipped pre-check never idles the GPU. Hours use the measured anchors (224 px
41–42 min; 384 px 64–75 min); new-trunk, zoom, DSP, LOAO and TTA costs are **unmeasured** until
their smokes, and §8.3's fallbacks apply.

| # | GPU block | Runs | Hours | Needs | Read that follows |
|---|---|---|---|---|---|
| **Q0** | **S01** (§3; registered by master + A01) | 384 px × s42/43/44; 224 px × s43/44 | 4.6–5.2 | C2 | S01 two-hurdle → resolution (E2) |
| **Q1** | Control 224 px **s45/46/47** (`train_v4 --rungs R0`, tag `v5s01`), then **last-epoch scoring** of every fold-0 control run (`research.v5.infer_last` on the `_last.pt` of R0 s43–47 and R1 s42–44; the screen and S01 read the last epoch, A01 A3) | 3 + 8 inference | 2.1 + inference (minutes each, unmeasured) | **E1 hash** | — |
| **Q2** | Trunk smokes, then **trunk screen** (§6.0): `control --trunk in22k` (primary) then `--trunk dinov3` × s42/43/44 | 6 | ≈ 4.1–4.2 (unmeasured) | Q1, parity smoke | Noise floor (s42–47) → trunk decision |
| **Q3** | Arms with **no pre-check**, on the chosen trunk: `twostep`, `clues`, `gem`, `m4`, `memory` × s42/43/44 | 15 | ≈ 10.3–10.5 (+ ≤ 10% heads) | Q2 read | Screen reads |
| **Q4** | Pre-check-gated arms: `look`, `structure`\*, `geometry`\*, `m5`\*, then `zoom`, then `youngdata`\* × s42/43/44 | ≤ 18 | ≈ 12.3–12.6 + zoom 1.5–2× (unmeasured) | §4 verdicts; YD rows; zoom benchmark | Screen reads; 384 px rescue if it fires (≤ 9 h) |
| **Q5** | LOAO `control` + `m7` (3 hold-outs each; `--swad-window 10 30`) | 6 | ≈ 6.2 (extrapolated; time the first) | m7 smoke | LOAO read |
| **Q6** | Stacking check: `composite`, `composite_nologic` × s42/43/44, 224 px | 6 | ≈ 4.1–4.2 | Q3–Q5 reads | **Composite lock** (target Fri 2 Oct) |
| **Q7** | `composite` folds 1–4 × s42 **and** `control` folds 1–4 × s42, at the S01 resolution (`--save-best`) | 8 | 8.5–10.0 (384) | Lock | **Seed-42 confirmation** — Review-2 minimum |
| **Q8** | Same for s43, then s44 | 16 | 17.1–20.0 (384) | Q7 | Seed-43/44 confirmation |
| **Q9** | Fold 0 `composite` + `control` × s42/43/44 (selection fold, flagged; completes the 15-model OOF), TTA (benchmark first), `zoom_lesion` / `zoom_random` if zoom passed | 6 + TTA + 2 | ≈ 6.4–7.5 + TTA | Q8 | V5 OOF → **S56 refit** (CPU) |
| **Q10** | **S84 MILK10k single read** (§9) | — | ≤ 1 | S56 refit, S75 | S84 report |
| **Q11** | If time allows: trunk-only control (chosen trunk, no modules) folds 1–4 × s42 (AU26) | 4 | 4.3–5.0 | Q8 | Descriptive decomposition |

\* only if its pre-check passed. Projected total ≈ 80–95 GPU h (+ ≤ 9 h rescue). About 130 GPU h
are realistically available between 30 Sep 09:00 and 8 Oct 00:00. **8 Oct: no GPU. 9 Oct: Review
2.**

### 8.1 Daytime control fold runs (they do not depend on any screen, AU2)

- 12 runs: `control` (in1k, the V4 recipe), folds 1–4 × s42/43/44, at the S01-chosen resolution,
  with `--save-best`. 12.8–15 h in total (384 px) or 8.2–8.4 h (224 px).
- They now run inside the continuous queue (Q7/Q8, paired seed by seed with the composite), so
  each completed seed gives a complete paired confirmation.

### 8.2 Review-2 cut line

- **Must have by 8 Oct:**
  - S01 and the measured noise floor;
  - the CPU biology diagnostics (M1-QC, Q1, Q2/D6, Q5, B1, D2, D4);
  - every screen and the composite lock;
  - the trunk screen and its decision;
  - confirmation at **seed 42** (composite and control, folds 1–4).
- **If time allows:** seeds 43/44, fold 0, TTA, the S56 refit, S84.
- **After Review 2:** whatever is unfinished runs in this table's order, under the same hash, and
  is reported as a V5 addendum. Nothing is re-planned after results.

### 8.3 Pre-declared fallbacks

| Trigger | Action |
|---|---|
| S01 two-hurdle fails (384 not better on the pooled partition) | Confirmation and composite at **224 px**; the fold-run cost becomes 41 min |
| Control fold runs not complete by the end of 5 Oct | Confirmation on **2 seeds** (s42, s43), stated in the results |
| A night lost | N9 buffer absorbs it; S84 moves after Review 2 |
| Zoom benchmark > 2× | `zoom` moves after Q5; LOAO `m7` moves to Q9 |
| New-trunk smoke > 1.25× the in1k time | The trunk screen still runs; its hours are re-quoted from the smoke before Q3 |
| YD rows not ready by Thu 1 Oct 20:00 | `youngdata` moves to V6 (recorded); nothing else changes |
| DSP / M7 smoke epoch > +25% time | Move the filters to the GPU (kornia), or drop the slow component and note it (AU9) |
| A pre-check fails (Q1/Q2/Q4/Q5) | The arm is not run. The failure is reported as a finding |

### 8.4 Checklist before every night

1. ≥ 10 GB free on C: (C2); the block's checkpoints fit (screens 0.11 GB per run, confirmation
   0.22 GB per run). **AU28:** the revised queue writes ≈ 16–18 GB against 26 GB free on 30 Sep,
   so the owner frees ≥ 10 GB (Downloads / `.ollama`) **before Q7**.
2. `--num-workers 2`, `--patience 0`, 30 epochs.
3. `--no-save-best` on screens and LOAO; `--save-best` on confirmation.
4. `--smoke` passes for every arm in the night.
5. Queue written to `results/v5/queue_<block>.txt`, so an interrupted block can resume.

**Projected disk use:**
- S01: 1.6 GB;
- control seeds 45/46: 0.4 GB;
- screens and LOAO: ≈ 28 × 0.111 ≈ 3.1 GB;
- SWAD copies: 0.7 GB;
- confirmation: 26 × 0.222 ≈ 5.8 GB;
- DRE-8b/8r: 0.2 GB.

Total **≈ 11.9 GB** of 31 GB free (pre-revision). **Revised (AU28):** ≈ 16–18 GB including the
young-data images (1.37 GB); C: had **77 GB free** on 30 Sep after the owner moved 52.5 GB to D:.

---

---

## 9. S75 / S84 — MILK10k, pre-registered before download

**Source:** MILK10k (https://api.isic-archive.com/doi/milk10k/; Tschandl et al., *J Invest
Dermatol*). 5,240 lesions / 10,480 images (paired
clinical + dermoscopic), 95.7% histopathology-confirmed, age in 5-year bins, CC-BY-NC, 344.7 MB
zip.

**Allocation (A01 B8):**
- **V5 confirmation only.** No MILK10k image enters V5 training, calibration, S56 fitting or
  model selection.
- After the S84 read, MILK10k becomes eligible for V6 **training**, and V6 declares its own
  confirmation cohort.

**S75 eligibility (CPU; metadata and images only; no model is run):**
1. Use the dermoscopic image of each lesion. Clinical images are not used in V5.
2. Map diagnoses to the 7 classes: mel, nv, bcc, akiec (= AK + SCC/KA), bkl (benign
   keratinocytic), df, vasc. *Inflammatory/infectious* and *other* go to an out-of-scope stratum,
   excluded from Macro-F1 and reported separately.
3. **Exclude** any lesion whose recorded ISIC identifier, or whose perceptual hash
   (`results/v4/perceptual_hashes.npz`, same hash and threshold as S49), matches any of the 25,331
   ISIC-2019 images.
4. Label the cohort **"cohort-external, partly same-institution"** (Vienna and MSKCC contribute).
5. Report the eligible counts by class × age band, including the **under-40 escalating lesion
   count**, in `results/v5/s75_milk10k_eligibility.json`. Counts only; no predictions.

**S84 analysis (A01 A9, locked with this file):**
- **Systems compared:**
  - **V5 system** (AU23) = composite, **3 seeds × 5 folds = 15-model ensemble** (mean of the
    model probabilities; if s43/s44 are incomplete at S84 time, the completed seeds, stated), TTA
    as Stage 7, a Dirichlet calibrator refit on the seed-averaged V5 OOF, and **S56 refit on that
    OOF**. No extra training: the models are the confirmation runs;
  - **V1 system** = the frozen V1 stack with deployed S56@0.20.
- **Primary:** paired difference in **under-40 escalation sensitivity** at **matched referral**
  (V5's threshold set so its overall referral equals V1's on MILK10k). Lesion unit; exact
  McNemar on discordant lesions; one-sided α = 0.05.
- **Key secondary:** the S70 contract (under-40 system sensitivity ≥ 0.855 and referral ≤ 0.25),
  Clopper–Pearson.
- **Global:** 7-class Macro-F1 of V5 vs V1, non-inferiority margin 0.02 (lesion-grouped
  bootstrap CI).
- **Descriptive:** escalation AUROC and pAUC@0.20 overall and by age band; per-institution
  breakdown; out-of-scope stratum referral rate.
- **Power:** computed on 30 Sep from the eligible under-40 count and the V1-vs-V5 discordance
  observed on OOF folds 1–4 (A01 A9 method). Reported whatever it is. **Underpowered is stated,
  not hidden.**
- **Single read:** the script refuses a second execution (receipt `results/v5/s84_receipt.json`,
  `n_executions ≤ 1`).

---

---

## 10. Deliverables for Review 2 (whatever is complete on 8 Oct)

1. The biology atlas and the dermatologist-reasoning design (A02 §1–3; DRE).
2. CPU biology findings: whether chromophore geometry, asymmetry, periphery and structure
   primitives separate young melanoma from young biopsied nevi (Q1/Q2/Q5/D6), and the hard-core
   anatomy (B1).
3. The noise floor, screen results and composite lock, with each mechanism's falsifier outcome.
4. Confirmation on folds 1–4 (at least seed 42), with gates A–D and the case-mix and per-archive
   readouts.
5. If done: the S56 system on the V5 base, and the MILK10k external read.
6. The V6 plan (`docs/V6_RUNSHEET.md`).

---

## 11. Session plan — execution order, CPU work in parallel with GPU work, model per session

**Host measured 2026-09-29 22:10:**
- AMD Ryzen 9 8940HX (16 cores / 32 threads), 15.2 GB RAM;
- commit limit 30.4 GB, **commit headroom 5.1 GB with apps open**;
- C: has 31 GB free;
- RTX 5050 idle.

Commit headroom, not VRAM, is the binding constraint on this machine (error 1455). So every rule
below is about memory first.

### 11.1 Concurrency rules (hard; they apply to every session)

| # | Rule | Why |
|---|---|---|
| C1 | **Exactly one GPU job at a time.** Smokes, benchmarks, TTA and training never overlap. Every CPU job sets `CUDA_VISIBLE_DEVICES=""` so it cannot create a CUDA context | Protects VRAM (8.55 GB) and keeps timings valid |
| C2 | **Before any training starts:** commit headroom **≥ 6 GB** (close the browser and other heavy apps) and C: **≥ 10 GB** free, so the pagefile can grow. The trainer's own guard is 5 GB. **Optional, owner action (system setting):** raise the pagefile's *initial* size so the commit limit does not depend on growing under load: System → Advanced → Performance → Virtual memory → C: custom, initial 24,576 MB, maximum 32,768 MB. That raises the limit from 30.4 to about 39 GB and uses about 9 GB more disk; C: stays above 20 GB. Reboot required | 2 workers at 384 px need several GB of commit; headroom was 5.1 GB when measured |
| C3 | **Before a CPU job starts while training runs:** commit headroom **≥ 3 GB**. The CPU job must stay **≤ 2 GB** RAM: stream images one at a time, never hold the dataset in memory | Keeps training's workers alive |
| C4 | CPU jobs run at **BelowNormal priority** with `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` = **8** (**4** during 224 px training). They use no multiprocessing pool above 4 | Training uses about 3–4 cores; 224 px runs are decode-bound, so CPU theft slows them |
| C5 | **Heavy CPU jobs** (ICA basis fit, Q5 filter extraction, D4 GMM, MILK10k hashing) run only **while a 384 px run is going** (GPU-bound, so the CPU has slack) **or while the GPU is idle**. During 224 px runs, only light work: code, reading results, small statistics | The measured decode bottleneck at 224 px |
| C6 | GPU above **87 °C** sustained (`nvidia-smi`) → let the current run finish, pause the queue for 20 min. Laptop plugged in, on a stand, performance power mode | Laptop throttling |
| C7 | Smokes and benchmarks run **only in a GPU gap**. Pause the queue with `results/v5/PAUSE`; the runner stops between runs, never mid-run | C1 |
| C8 | No script reads test, reserved or MILK10k labels, except S84, once | Standing rules |

**Status check (PowerShell): commit headroom, free space on C: and GPU state:**

```powershell
$cl=(Get-Counter '\Memory\Commit Limit').CounterSamples[0].CookedValue/1GB; $cb=(Get-Counter '\Memory\Committed Bytes').CounterSamples[0].CookedValue/1GB; "headroom_GB=$([math]::Round($cl - $cb, 1)) C_free_GB=$([math]::Round((Get-PSDrive C).Free/1GB, 1))"; nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader
```

**CPU-job environment (PowerShell).** Set this first in the same window, then start the job and
lower its priority:

```powershell
$env:CUDA_VISIBLE_DEVICES=""; $env:OMP_NUM_THREADS="8"; $env:MKL_NUM_THREADS="8"; $env:OPENBLAS_NUM_THREADS="8"
```

```powershell
$p = Start-Process -FilePath "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe" -ArgumentList "-m","research.v5.<script>" -NoNewWindow -PassThru; $p.PriorityClass = "BelowNormal"; $p.WaitForExit()
```

### 11.2 Model and thinking level per kind of work (token optimisation)

| Tier | Model | Thinking | Use for |
|---|---|---|---|
| T1 | **Haiku 4.5** | none | Launching queues, status checks, hashing, downloads, git, disk/headroom checks |
| T2 | **Sonnet 5.5** | medium | Writing pre-check scripts, `train_v5`, the arm registry, unit tests, the queue runner, the S75 script, the S56 refit script |
| T3 | **Opus 5.5** | high | Reading results against the pre-declared gates, the S01 resolution decision, the composite lock, fallback decisions, **any failing parity/smoke or crash** |
| T4 | **Opus 5.5** | extended | 8 Oct final synthesis: gates A–D, the Review-2 narrative and slides |

### 11.3 Sessions (strict order; a session starts only when its "needs" are met)

> **Revision 2026-09-30:** the GPU column of E2–E16 below is replaced by the §8 queue (Q1–Q11).
> The CPU/Claude work, the "needs" and the model tiers stand, with these additions: E1 also hashes
> the arm registry sha (`python -m research.v5.arms --show`); D5 (`research.v5.d5_acquisition`)
> must exist before any `m4` run or screen read; the YD count goes to the owner at E1 and the
> download (if approved) runs under C3–C5; `m5`, `zoom`, `logic` and `m7` must be implemented and
> smoke-tested before Q4/Q5/Q6 (AU35).

- Interpreter: `C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe`.
- Working directory: `C:\Users\RAJ\Downloads\Capstone review 1`.
- "Daytime queue" = the 12 control fold runs of §8.1 (folds 1→4, s42 → s43 → s44).
- **Slip recorded 2026-09-30 (before the hash):** S01 was **not** launched on 29 Sep; at 30 Sep
  morning headroom was 2.9 GB, the GPU idle, and there were no `v5s01` runs. The dates below
  shift as follows, and nothing else changes:
  - **E0 runs Wed 30 in daytime**, in parallel with E1 (hash) and E4 (coding). Heavy E3
    pre-checks may run during E0's 384 px runs (C5), but only with headroom ≥ 3 GB (C3).
  - **E2's S01 read** happens after E0 ends (about 5 h). Control s45/s46 follow E0 on the GPU.
  - The **daytime queue starts Thu 1**.
  - **N2 stays Wed 30 night** if E5's parity smoke and arm smokes pass in the evening gap.
    Otherwise N2 moves to Thu 1 night, and every later night shifts by one. N9's buffer absorbs
    one shift; §8.2's cut line absorbs a second.

| Session | When | GPU (background) | CPU / Claude work in parallel | Needs | Done when | Model |
|---|---|---|---|---|---|---|
| **E0** | **Tue 29 Sep, now** | **S01**: smoke → 384 px s42/43/44 → 224 px s43/44 (4.6–5.1 h) | None (overnight) | C2 passes | 5 run JSONs tagged `v5s01` | T1 Haiku |
| **E1** | Wed 30, morning | idle | Owner adopts §0–11, A02 and the DRE → **hash** (§1) → append `amendments[1]` → re-verify all hashes → CHANGELOG | E0 finished | 13 hashes verify (10 old + 3 new); the freeze file has 2 amendments | T1 Haiku |
| **E2** | Wed 30, after E1 | **Control 224 px s45, s46** (`train_v4 --rungs R0 --corpus pooled --fold 0`, 1.4 h, decode-bound) | **S01 read:** two-hurdle rule (CI lower bound > 0 **and** mean paired gain ≥ +0.015) → `results/v5/s01_decision.json` (sets the §8 resolution). Light CPU only (C5) | E1 | Decision file written | T3 Opus high |
| **E3** | Wed 30, after s45/s46 | **Daytime queue starts** via `train_v4` (the V4 recipe *is* the V5 control), at the S01 resolution. **Pause by ~19:30** for E5 | **Write and run the §4 pre-checks:** M1-QC, Q1, Q2/D6, **Q5**, B1, D2, D4, D5, under C3–C5 (heavy ones only while a 384 px run is going). **Downloads** (owner-approved): Task-2 GT → Q4; MILK10k → S75 | E2 | Every §4 output written; each arm's go/no-go in `results/v5/precheck_verdicts.json` | T2 Sonnet medium (scripts); T3 Opus high (verdicts) |
| **E4** | Wed 30, parallel with E3 | (same queue) | **Implement** `research/v5/train_v5.py`, `research/v5/arms.py` (§5–7), and `scripts/run_v5_queue.ps1`. The runner reads `results/v5/queue_<name>.txt`, skips completed runs, checks C2 before each run, honours `PAUSE`, and logs. **Unit tests (CPU):** OD clamp, mask/image alignment, `has_attr`, empty-pair hinge, zero-init stem equals RGB | E1 | Unit tests pass | T2 Sonnet medium |
| **E5** | Wed 30, ~19:30 (GPU gap) | **Parity smoke** (`train_v5 --arm control` vs `train_v4 R0`, 2 epochs, per-step loss within 1e-3) → `--smoke` for each N2 arm → **zoom benchmark** (`gpu_benchmark.py`, unfrozen) | Apply the §8.3 time fallbacks from the smoke timings; write `queue_N2.txt` | E3 paused; E4 done | Parity passes; every N2 arm smokes | T1 Haiku; **T3 Opus high if parity fails** |
| **E6** | **Wed 30 night (N2)** | `run_v5_queue.ps1 N2`: look, structure\*, geometry\*, twostep, memory × s42/43 (\* per E3 verdicts; ≤ 6.8 h) | None | E5 | Run JSONs present | T1 Haiku |
| **E7** | Thu 1, morning | Resume the daytime queue | **1.** Noise floor from control s42–47 → `results/v5/screens/noise_floor.json` (`screen_gate.py --null-only`). **2.** Then read N2 against the §6 gate and falsifiers → `results/v5/screen_N2.json` | E6 | Both files, in that order | T3 Opus high |
| **E8** | Thu 1, ~19:30 (GPU gap) | Pause → smokes for the N3 arms (clues, gem, m4, m5\*, zoom) | Write `queue_N3.txt` | E7 | Smokes pass | T1 Haiku |
| **E9** | **Thu 1 night (N3)** | N3: clues, gem, m4, m5\*, zoom × s42/43 (7.6–8.2 h) | None | E8 | Run JSONs present | T1 Haiku |
| **E10** | Fri 2, morning | Resume the daytime queue | Read N3 (clues vs twostep, gem vs clues, m4 vs twostep, zoom vs clues, m5 vs control) → composite candidate per §6 → `queue_N4.txt` (LOAO control + m7 with SWAD; composite ± logic) | E9 | `screen_N3.json`, `queue_N4.txt` | T3 Opus high |
| **E11** | **Fri 2 night (N4)** | Evening gap: smokes for m7 / composite. Then N4 (6.1–8.9 h). Time the first LOAO run; apply §8.3 if it runs over | None | E10 | Run JSONs present | T1 Haiku |
| **E12** | Sat 3, morning | Resume the daytime queue | Read N4 → **composite lock** (`composite_lock.json`, sha256 in the CHANGELOG) → `queue_N5.txt` (composite folds 1–4 × s42, `--save-best`) | E11 | Lock written **before** any N5 run | T3 Opus high |
| **E13** | **Sat 3 / Sun 4 / Mon 5 nights (N5–N7)** | Composite folds 1–4 × s42 / s43 / s44 (4.3–5.0 h each; ×≈1.34 with zoom) | Days: the daytime queue continues; light monitoring only | E12 | 12 composite runs | T1 Haiku |
| **E14** | Sun 4, morning | Daytime queue | **Seed-42 confirmation readout** (composite vs control, folds 1–4). This is the **Review-2 minimum** | Control s42 folds 1–4 + N5 | `confirm_s42.json` | T3 Opus high |
| **E15** | Mon 5, evening | — | **Fallback check:** are all 12 control fold runs done? If not → confirm on 2 seeds (§8.3), recorded | — | Decision recorded | T3 Opus high |
| **E16** | **Tue 6** | Night (N8): fold 0 composite + control → TTA benchmark → TTA inference → zoom_lesion / zoom_random if zoom passed | Day: write the **S56 refit** script and the **receipt-guarded S84** script | E13 | V5 OOF complete (folds 0–4) | T2 Sonnet medium |
| **E17** | Wed 7 | Day: after the S56 refit, **S84 MILK10k single read** (≤ 1 h). N9 = buffer | S56 refit on V5 OOF (CPU) **before** S84 | E16 + S75 | `s84_receipt.json` n = 1; report | T3 Opus high (verify the locked §9 spec); T1 to launch |
| **E18** | Thu 8 | **No GPU** | Gates A–D (hierarchical seed × lesion CI), B2 case mix, B4 per archive, pAUC_histo, falsifiers, DRE explanation panels, Review-2 slides (V5 + the V6 plan) | E14 at minimum | Slides done | **T4 Opus extended** |
| **E19** | After 9 Oct | Remaining V5 items in §8 order, same hash | V5 addendum | — | — | as above |

**E0 commands (tonight, PowerShell, from the working directory).** First run the status check in
§11.1: it needs headroom ≥ 6 GB (close the browser) and C: ≥ 10 GB. Then the smoke. Stop if it
fails:

```powershell
$py="C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"; & $py -m research.v4.train_v4 --rungs R1 --corpus pooled --fold 0 --seed 42 --batch-size 16 --grad-accum 2 --num-workers 2 --patience 0 --device cuda --smoke
```

Then the S01 queue. It stops at the first failure:

```powershell
$py="C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"; $ok=$true
foreach ($s in 42,43,44) { & $py -m research.v4.train_v4 --rungs R1 --corpus pooled --fold 0 --seed $s --batch-size 16 --grad-accum 2 --num-workers 2 --patience 0 --device cuda --run-tag v5s01; if ($LASTEXITCODE -ne 0) { "STOP R1 seed $s"; $ok=$false; break } }
if ($ok) { foreach ($s in 43,44) { & $py -m research.v4.train_v4 --rungs R0 --corpus pooled --fold 0 --seed $s --batch-size 32 --num-workers 2 --patience 0 --device cuda --run-tag v5s01; if ($LASTEXITCODE -ne 0) { "STOP R0 seed $s"; break } } }
```

**E2 / E3 commands before the queue runner exists (PowerShell).**

E2, control seeds 45/46 at 224 px (only after the E1 hash):

```powershell
$py="C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"; foreach ($s in 45,46) { & $py -m research.v4.train_v4 --rungs R0 --corpus pooled --fold 0 --seed $s --batch-size 32 --num-workers 2 --patience 0 --device cuda --run-tag v5s01; if ($LASTEXITCODE -ne 0) { "STOP seed $s"; break } }
```

E3, the daytime control fold runs:
- Use **R1 with `--batch-size 16 --grad-accum 2`** if `s01_decision.json` says 384 px, or **R0
  with `--batch-size 32`** if it says 224 px. `train_v4` always writes `_best` and `_last`, as
  confirmation requires.
- The loop has no pause: **start a fold only if it can finish before the evening GPU gap**
  (64–75 min at 384 px, 41 min at 224 px). From Thursday, use `run_v5_queue.ps1`, which honours
  `PAUSE`.

```powershell
$py="C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"; foreach ($f in 1,2,3,4) { & $py -m research.v4.train_v4 --rungs R1 --corpus pooled --fold $f --seed 42 --batch-size 16 --grad-accum 2 --num-workers 2 --patience 0 --device cuda --run-tag v5ctl; if ($LASTEXITCODE -ne 0) { "STOP fold $f"; break } }
```

### 11.4 Break-nothing list

1. Running any post-S01 run (control seeds 45/46/47, the trunk screen, pre-checks, any arm) **before E1's hash**.
2. Reading a screen **before** `s01_noise_floor.json` exists (E7 order).
3. Launching N5 **before** `composite_lock.json` exists.
4. Running S56 or S84 before folds 0–4 OOF is complete. Running S84 twice.
5. Two GPU jobs at once; a CPU job started with headroom < 3 GB; training started with headroom
   < 6 GB or C: < 10 GB.
6. Editing any file in `docs/v5_record/`, or any file hashed at E1 (changes go in an Amendment 03).
7. Deleting checkpoints or run JSONs during V5.
