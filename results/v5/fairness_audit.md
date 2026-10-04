# V5 fairness audit — was the dermatologist-reasoning programme given a fair test? (2026-10-04)

Question (owner): a biologically grounded, dermatologist-style pipeline produced almost nothing. Was the
training fair, were the screens and falsifiers too strict, and why did it fail? Every number below is read
from the file named beside it.

## 1. What actually happened
| Stage | Outcome | Source |
|---|---|---|
| CPU premise checks | M1-QC, Q1 (geometry, Dice 0.781), Q2/D6 pass; **Q4 fail** (416 mask overlaps < 1,000 → `m5` never run); **Q5 fail** (1/5 dermoscopic-structure signatures, **0 young tokens** → `structure` never run) | `results/v5/precheck_verdicts.json` |
| Fold-0 screens (3 seeds, 80th-percentile bar) | twostep, m4, youngdata pass; look, clues, gem fail the gate; memory, geometry, zoom pass the gate but **fail their falsifier** | `results/v5/screens/gate_*.json`, `falsifier_*.json`, `q5_read.json` |
| Stacking (Q6) | composite passes vs control but is sub-additive (vs m4 −0.012 all-age pAUC) | `gate_composite_in22k_vs_m4.json` |
| Confirmation, folds 1–4, seeds 42+43 | all-age pAUC −0.0007 [−0.008, +0.007]; **under-40 pAUC +0.036 [+0.007, +0.067]**; Macro-F1 +0.013; Gates A fail, B pass, C pass, D cannot pass | `results/v5/confirm_s42_s43.json` |

## 2. Were the screens too strict? — No; they were lenient
- The screen bar is the **80th percentile** of a measured seed-pair null (pair SD: all-age pAUC 0.0071,
  histo 0.0134, Macro-F1 0.0161, under-40 0.0191; `noise_floor.json`). A null arm passes 20–36% of the time
  (runsheet §6, AU12). Retention fails only below −0.0153.
- The arms that failed did so on point estimates near or below zero, not narrowly: clues −0.0014 vs its
  parent, gem −0.0012, look (fixed) −0.0018 all-age pAUC. A more lenient bar would not have rescued them.

## 3. Were the falsifiers too strict? — Fair as mechanism tests; strict as an engineering filter
- **geometry:** rotating the lesion frame by 45–135° kept ≈ 75% of the gain (rule: ≤ 50%) → the gain does not
  come from lesion-axis geometry. A fair test of the claimed mechanism.
- **zoom:** a random-location crop beat the evidence-guided crop in 3/3 seeds → the gain (+0.0085 vs clues)
  is a generic multi-view / scale effect, not "looking where the evidence is". Fair.
- **memory:** young-melanoma prototypes were the same prototypes used for 60+ melanoma (condition b failed in
  every seed) → no young-specific memory. Fair.
- **The strictness is in the consequence, not the test:** a falsified arm was excluded from the composite even
  when its gain was real (geometry +0.0049, zoom +0.0085 all-age pAUC). That is correct for a *mechanistic*
  claim ("biology helps") and too strict for an *engineering* goal ("anything that helps"). V6 should keep the two
  separate: mechanism claims stay falsifier-gated; a performance ensemble may include any arm with an OOF gain.

## 4. Was Gate A too strict? — It is a clinical bar, and it was known to be hard
- Gate A needs a point gain ≥ +0.050 **and** a CI above zero, on 65 under-40 escalating lesions. The plan said in
  advance that a pass was "not the expected outcome": V4's largest move was 0.011 (AU31).
- With the observed SE (≈ 0.015 from the two-seed CI), a true effect of +0.036 would reach a +0.050 point estimate
  only ~18% of the time. So the honest reading is **a real, statistically detectable, sub-clinical gain** — not
  "nothing". It should be reported that way.

## 5. Training fairness — four real handicaps against the modules
1. **No tuning for modules, a tuned recipe for the control.** Every module parameter was fixed in advance
   (runsheet §7 "nothing is tuned") to prevent p-hacking, while the control recipe was refined over V1–V4. One
   shot of hyperparameters per module is a structural disadvantage. Deliberate, defensible, and a limitation.
2. **New modules learn at the backbone's fine-tuning rate.** `train_v5.set_stage`: modules train at 1e-3 only
   during the 3 frozen-trunk epochs, then at **1e-4** with the trunk for 27 epochs. `look`'s extra chromophore input
   channels are **zero-initialised inside the frozen stem**, so they train only at 1e-4. Freshly initialised
   components usually need their own (higher) learning rate; whether this cost anything is untested.
3. **The biology was tested at 224 px.** S01 chose 224 px on single-model Macro-F1 (`s01_decision.json`), and the
   384 px rescue for structure / zoom / m5 could only fire if S01 had picked 384. Fine dermoscopic structure is
   what 224 px discards — and Q5 found 0 young structure tokens at that resolution.
4. **Bugs, caught and fixed before the verdict.** The first look/geometry runs had a front-end NaN problem; after
   the 2 Oct fix both were re-screened (`*_v5fix`): geometry +0.0164 → +0.0049, look +0.0044 → −0.0018 all-age
   pAUC. The verdicts used the fixed runs, so this is fair — but it shows how fragile these streams were.

Nothing favoured the modules unfairly; one thing favoured the **composite**: it uses the IN-22k trunk while the
pre-registered comparator is the in1k control (runsheet §6.0). On fold 0 the trunk alone gave +0.0072 all-age pAUC,
+0.0275 Macro-F1 and **−0.0155 under-40 pAUC** (`gate_control_in22k_vs_control.json`). Read descriptively (fold 0 vs
folds 1–4, not a paired test), the modules + young data therefore **removed** the trunk's all-age gain and part of its
Macro-F1 gain, while adding ≈ +0.05 under-40 ranking on top of a trunk that had lowered it. Q11 (trunk-only control,
folds 1–4) is the run that would test this directly.

## 6. Why did a biologically grounded design yield so little? — The honest answer
1. **The failure case is the hardest problem in dermoscopy, not a missing feature.** The control's under-40 misses
   are **nevus-like melanomas**: 51 of 55 missed images called `nv`, 19 of 33 lesions in the highest melanin tercile
   (`b1_hard_core.json`). Dermatologists in a prospective trial caught only 8 of 14 such melanomas under 35
   (sensitivity 0.571; Heinlein et al. 2024).
2. **The biological signal was not measurable in our images at our resolution.** The two most "dermatological"
   modules never ran because their premise failed on data (Q5: 0 young structure tokens; Q4: too few expert masks).
3. **A pretrained CNN already encodes most hand-crafted cues.** Colour/chromophore, asymmetry and attention-to-clues
   added nothing over the trunk; where small gains appeared, the falsifiers showed they came from generic effects
   (extra views, extra capacity), not from the biology.
4. **Statistical power.** 65 under-40 escalating lesions in confirmation, 35 on fold 0; the under-40 pair SD (0.019)
   is as large as the effects the modules could plausibly produce.
5. **What did move the under-40 needle was data:** young histopathology-confirmed images (+0.0225 on fold 0, and the
   composite's +0.036 on folds 1–4). The successful external system (ADAE) used 58k lesions, 384–896 px, 90 models
   and six photos per lesion.

## 7. Verdict
V5 was a **fair and, if anything, lenient test** of whether hand-built dermatological modules improve escalation
ranking, with three real handicaps (no module tuning, shared fine-tune learning rate, 224 px). Within those limits the
answer is **no** for all-age ranking and **a small, real, sub-clinical yes** for under-40 ranking, carried by young data
rather than by the reasoning modules. What V5 did *not* test: the same biology at 384–512 px, with tuned modules or
module-specific learning rates, or as auxiliary supervision on a foundation-model trunk. Those are the fair V6
follow-ups, and they rank below data and ensembling on expected value.
