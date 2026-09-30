# V5 Ideas — Biology First

**Date:** 2026-09-23
**Status:** CANDIDATES, **not pre-registered**. Nothing here may run until it has been adopted into
`docs/v5_design/V5_PLAN_AMENDMENT_02.md` and hashed into `results/v5/v5_plan_freeze.json`, following the
same procedure as Amendment 01.
**Goals:**
1. Raise **under-40 escalation sensitivity**.
2. Raise **cross-domain Macro-F1**, meaning (a) across dermoscopy archives/sites and (b) across
   modality, to smartphone clinical photos.

**Numbers:** every number here comes from a file in `results/` or `ml/results/`. The new ones are
from `results/v5/diagnostics/d0_brainstorm.json`, written by
`research/v5/d0_brainstorm_diagnostics.py`. That script uses the S72 cross-fitted OOF matrix,
development rows only, with the test lock armed.

---

## 0. What the data says before any new idea

Three measurements, taken 2026-09-23 on the pooled 224 px control's OOF predictions. All pAUC
values are McClish-standardised at FPR ≤ 0.20.

### 0.1 The under-40 gap largely reflects which archive those lesions come from

| Archive | <40 escalating lesions | <40 pAUC | 40–59 pAUC | 60+ pAUC |
|---|---:|---:|---:|---:|
| HAM | 34 | 0.795 | 0.847 | 0.813 |
| BCN20000 | 35 | 0.696 | 0.734 | **0.643** |
| MSKCC | 12 | **0.507** | 0.701 | 0.764 |

Within HAM, the under-40 band trails 60+ only slightly (−0.018, CI [−0.104, +0.061]). Within BCN,
60+ is the worst band. But **58% of under-40 escalating lesions (47 of 81) come from BCN and
MSKCC**, the archives the model ranks worst, and in MSKCC the under-40 ranking is at chance (only
12 lesions).

**Implication:** the two goals are partly **one problem**. Anything that improves cross-site
robustness should also lift under-40 sensitivity. That becomes a testable link hypothesis
(**H-link**, §4).

### 0.2 In HAM, verification status explains most of the remaining band gap

- In HAM's OOF rows, **every escalating image is histopathology-confirmed**, while the 2,592
  "follow-up" images (digital monitoring) are **100% nevus**.
- Restricting both bands to histopathology-confirmed rows moves the <40-minus-60+ pAUC gap from
  **−0.018 to +0.006** (CI [−0.081, +0.094]).
- On those same rows, every band scores about **0.74–0.75**.

So the hard problem at every age is **melanoma vs. a benign lesion suspicious enough to
excise**. HAM's <40 band has 368 histo-confirmed benign lesions.

Two consequences:
- The follow-up images give the model a **device-plus-label shortcut** ("monitoring-camera look
  → nevus"). That shortcut does not exist at other sites, which is a plausible contributor to the
  cross-domain loss.
- Among clinically suspicious lesions, the model ranks all ages about equally (and about equally
  poorly). The lever is the melanoma-vs-suspicious-nevus boundary itself.

### 0.3 Pooling several images of one lesion adds nothing

- 3,014 lesions have more than one image, and 51 of the 81 under-40 escalating lesions do.
- Lesion-level max vs mean pooling differs by **[−0.006, +0.008]**, and image-level pAUC is
  0.739 vs 0.736 at lesion level.
- The views are near-duplicates, so **multi-view fusion is dropped** before it costs any GPU time.

### 0.4 The cross-domain numbers the ideas are aiming at

| Setting | Macro-F1 | Source |
|---|---:|---|
| HAM-only V1 stack on BCN/MSKCC reserved | 0.411 | `results/v4/s54/s54_marginals.csv` |
| Pooled V4 224 px control on reserved (3 seeds, `_last`) | 0.588–0.611 | same |
| Pooled control, in-distribution fold val | 0.651–0.697 | CHANGELOG §Phase Y |
| HAM-only models on PAD smartphone (zero-shot) | 0.124–0.188 | CLAUDE.md, S8b |
| HAM→PAD warm-started ConvNeXt-Tiny (5-class) | 0.760 | `ml/results/PAD_MACRO_F1_PUSH.md` |
| Under-40 pAUC on reserved, every arm incl. V1 | 0.720–0.750 | `s54_marginals.csv` |

The largest cross-domain gain the project has ever measured came from **more diverse training
data**: +0.18 Macro-F1 from pooling (S54). No head, loss or decision rule has come close to that.

---

## 1. The biological model of the failure

Each line is tagged **[lit]** (published, citation in §6, to verify before the paper),
**[data]** (measured in this repo) or **[hyp]** (hypothesis to test).

1. **The model's "melanoma prototype" is an older patient's pigmented melanoma.**
   - 60+ holds 1,436 of the 2,061 age-known escalating lesions (70%) [data]. So what the model
     learned as "melanoma" is largely what melanoma looks like on chronically sun-damaged skin:
     regression, grey structures, lentigo-maligna patterns [lit].
   - Young and cross-site malignancies are **off-prototype** [hyp].
2. **Young-patient melanoma is often nevus-like, lightly pigmented, or focal.**
   - It is more often superficial spreading, thinner, and either **nevus-associated** (arising in
     part of an existing nevus) or de novo [lit].
   - It shows growth-phase structures: pseudopods and asymmetric peripheral globules (the 2025
     age-stratified study already cited in the plan) [lit].
   - In the youngest patients it is more often **amelanotic or nodular**, which is why paediatric
     guidance uses a modified ABCD: Amelanotic, Bleeding/Bump, Colour uniformity, De novo [lit].
3. **Young benign mimics have the same structures, arranged symmetrically.**
   - Growing nevi in young people carry a *symmetric* rim of peripheral globules.
   - Spitz/Reed nevi show a *symmetric* starburst [lit].
   - So what separates them from melanoma is the **symmetry and focality** of a structure, not
     whether it is present [lit/hyp].
4. **Colour is chromophore physics.**
   - Brown or black means epidermal melanin, and blue-grey means dermal melanin (Tyndall
     scattering). Red or pink means haemoglobin in vessels, and white means fibrosis/regression
     [lit].
   - Low-pigment melanomas carry their evidence in the **haemoglobin/vascular** signal, which a
     network trained mostly on pigmented melanomas under-uses [hyp].
5. **The skin around a lesion shows the patient's age.** Photoaging (solar lentigines,
   telangiectasia, elastosis) can let a model infer age from the background and apply a learned
   "young → benign" prior. That would produce the observed *confidently wrong* misses [hyp].
   Models trained with the lesion removed still score above chance on ISIC data [lit].
6. **Sites differ in acquisition, not in disease.** Sites differ in device, polarisation, field of
   view, borders/vignettes, resolution (HAM 600×450, BCN 1024×1024) and verification workflow.
   Archive identity stays decodable at 0.97 even after colour normalisation (S49 re-run) [data].
   Whatever carries it is not global colour.

---

## 2. Diagnostics to run first (CPU only)

These decide which GPU ideas are worth running.

| ID | Question | Method | Decides |
|---|---|---|---|
| **D1** | Does §0.2 hold outside HAM? | Fetch `diagnosis_confirm_type` (or equivalent) for BCN/MSKCC images from the ISIC Archive API. Availability not yet verified | Whether N4 and the histo-only endpoint extend to all archives |
| **D2** | Does the background carry age and malignancy? | Using HAM masks, fit probes on features of the **perilesional ring only** (lesion blacked out) to predict (a) age band, (b) escalation | If (a) is strong and (b) is above chance, B5 (context α) and N8 are justified |
| **D3** | What carries archive identity? | Archive probe on variants: border/vignette region only, lesion only, grey-scale, resampled to a common resolution, re-JPEG'd | The target list for N8's randomisation |
| **D4** | Is there a hidden melanoma subclass? | Cluster OOF melanoma embeddings (per-class GMM, k chosen by BIC); report per-cluster pAUC and <40 share | Whether N6 is worth a GPU run |

---

## 3. The ideas

Each idea lists: **Biology → CV mechanism → why it is not a repeat (no-repeat registry) → which
goal it targets → cost → the result that would prove it wrong.**

- **Cost anchor:** a 384 px ConvNeXt-Tiny fold-0 run takes **64–75 min** (S70 benchmark; banked
  R1+R4 runs scaled to one fold).
- **Screening:** every GPU idea is screened under Amendment 01's two-level gate (fold-0 all-age
  pAUC plus Macro-F1 retention). It must also report rescue on its **declared hard-core subset**
  (Amendment B1).

### Tier 1 — highest expected value

#### N1. Leave-one-archive-out (LOAO) development endpoint (enabler for goal 2)

- **Problem:** there is currently **no way to select for cross-domain Macro-F1 during
  development**. The reserved cohort is used up (8 reads), and the external cohort is single-read
  and confirmatory.
- **Design:** train on two archives and score the third, three times per candidate.
  - This gives a cross-site Macro-F1 for every candidate without touching reserved or external
    data.
  - It also gives a per-archive under-40 readout (BCN 35, HAM 34, MSKCC 12 lesions; descriptive).
- **Not a repeat:** evaluation infrastructure, not a model change.
- **Cost:** 3 runs per candidate, used only for goal-2 candidates. Screen at 224 px. Training-set
  sizes differ per hold-out (8.3k–13.7k images), so **time the first LOAO run** rather than
  extrapolating.
- **Falsifier:** none (it is an instrument). **Pre-register:** a candidate's cross-domain claim
  is judged on LOAO Macro-F1, never on fold-0 in-distribution Macro-F1.

#### N2. Chromophore channels: melanin and haemoglobin density maps (goals 1 and 2)

- **Biology:** §1.2 and §1.4. Hypo- and amelanotic young melanomas, BCC's arborising vessels and
  vascular lesions all put their evidence in haemoglobin. Dermal (blue-grey) versus epidermal
  (brown) melanin encodes depth.
- **CV mechanism:**
  - Convert RGB to optical density (OD = −log(I/I₀)).
  - Decompose each pixel into **melanin** and **haemoglobin** densities. Use either a fixed
    Beer–Lambert basis (analogous to H&E colour deconvolution) or a per-image ICA in OD space
    (Tsumura's skin-colour decomposition).
  - Add both maps as **two extra input channels**. Use a 5-channel stem whose new weights are
    initialised to zero, so the run starts exactly at the RGB model.
  - Optional third map: a blue-grey "dermal melanin" ratio.
- **Why it can also help across sites:** in OD space a multiplicative illuminant or camera gain
  becomes an additive offset. Per-image OD centring therefore removes most device colour gain
  **without** discarding colour, unlike HSV jitter. The same decomposition was developed on
  clinical skin photographs, so it also applies to smartphone images (N12).
- **Not a repeat:** colour constancy (buggy version barred; corrected Shades-of-Grey NULL)
  *normalised* the illuminant and kept RGB as the only input. N2 *adds* physically meaningful
  per-pixel channels and keeps RGB. Registry field `new_mechanistic_difference`: "adds chromophore
  density inputs; does not normalise".
- **Cost:** GPU cost is essentially that of one fold-0 run.
  - ⚠️ **Precompute the maps to disk.** Do not compute them in the dataloader. A float64
    per-pixel transform measured 18.8 ms/image for shades-of-grey and made a 224 px run as slow as
    a 384 px one (hardware anchors).
- **Falsifiers:**
  - The gain must concentrate in the **bottom tercile of melanin density** and in bcc/vasc F1.
  - Shuffling the haemoglobin channel across images must remove it.
  - If the gain is uniform across pigment levels, it is capacity or normalisation, not
    chromophores.

#### N3. Asymmetric "any-region" (noisy-OR) multiple-instance head (goal 1)

- **Biology:** §1.2. In a nevus-associated melanoma, most of the image *is* a nevus and the
  malignant component is focal and eccentric. Global average pooling lets the benign majority
  outvote the focal evidence, producing a confident "nv". That matches the observation that only
  11% of under-40 misses get referred.
- **CV mechanism:**
  - A 1×1-conv **patch-level escalation logit map** on the final feature map.
  - Aggregate it **class-asymmetrically**: malignant evidence by top-k / log-sum-exp ("malignant
    if any region is"), benign evidence by mean ("benign only if every region is"). This is the
    clinical logic, built in.
  - The 7-class head stays unchanged. The escalation logit enters the escalation loss (and B2's
    head, if B2 survives).
  - It produces a localisation map for free.
- **Not a repeat:**
  - GeM (B11) pools *features*, symmetrically for all classes.
  - Micro-patches (B4) change the *input resolution*.
  - Attention MIL treats classes symmetrically.
  - N3 changes the **decision logic of aggregation**.
- **Cost:** negligible parameters; one fold-0 run.
- **Falsifier:** for rescued malignant lesions, the top-k patches must be **eccentric** (centroid
  off the lesion centroid, covering a minority of the mask; measurable with HAM masks). Diffuse
  evidence on rescued cases means the nevus-associated mechanism is not what is working.

#### N4. Verification-aware hard negatives and label-source de-shortcutting (goals 1 and 2)

- **Biology/clinic:** §0.2. The real differential is melanoma vs. excised, suspicious nevi
  (dysplastic, atypical, growing). The follow-up nevi are an easy, device-specific negative.
- **CV mechanism:**
  - (a) Draw B12's ranking pairs and B7's hard negatives as **melanoma vs histo-confirmed
    nevus**.
  - (b) Treat `dx_type` as a nuisance domain. Balance sampling so that "follow-up camera look"
    stops predicting the label, or hold out follow-up images from the escalation loss.
  - (c) Adopt **histo-only pAUC** as a co-primary development readout.
- **Not a repeat:**
  - Class reweighting (NULL) weighted *classes*.
  - S67's hard-case reweighting used *model errors*.
  - N4 uses *clinical verification status*, which is outside knowledge the model cannot see.
- **Cost:** a sampler change plus one fold-0 run.
- **Falsifier:** histo-only pAUC must improve. An all-rows gain without a histo-only gain is a
  case-mix effect, not a better boundary.
- **Limit:** `dx_type` exists only for HAM; D1 decides whether N4 can extend to other archives.

#### N5. Targeted training-data expansion for young melanoma (goals 1 and 2 — probably the biggest lever)

- **Evidence:** pooling was worth +0.18 cross-site Macro-F1, and only **81** under-40 escalating
  lesions exist in the whole development corpus.
- **Action:** source additional **histopathology-confirmed, age-annotated** dermoscopy with an
  over-sample of under-40 melanoma and nevus. Candidates to verify: ISIC Archive collections
  outside ISIC-2019, HIBA, MILK10k, ISIC-2020.
  - Screen for duplicates against all 25,331 ISIC-2019 images (perceptual hashes already exist
    in `results/v4/perceptual_hashes.npz`).
  - Allocate per Amendment B8: each new source goes to **training or confirmation, never both**,
    and that is decided before download.
- **Not a repeat:** S75 sources *evaluation* data; this sources *training* data.
- **Cost:** mostly data engineering. GPU cost is the retrain.
- **Falsifier:** plot OOF under-40 pAUC against the number of under-40 training lesions (a
  learning curve made by subsampling). If the curve is already flat at 81, more data will not
  help. Run this subsampling check **first**; it is 2–3 fold-0 runs.

### Tier 2 — strong mechanisms, gated on a diagnostic

#### N6. Hidden-stratification discovery plus subclass Group-DRO (goal 1; gated on D4)

- **Biology:** §1.1–1.2. "Melanoma" is several visual diseases (superficial spreading, nodular,
  lentigo maligna, amelanotic, nevus-associated), and the young subtypes are a minority subclass
  of the label.
- **CV mechanism:** GEORGE-style. Cluster each class's embeddings to find subclasses, then run
  Group-DRO over the discovered subclasses. Age is never used, so it works when age is missing
  (251 dev rows now; external cohorts later).
- **Not a repeat:** A3/S13 defines groups by age × escalation. N6 uses *visual* subclasses.
  Propose it as the S13 variant, not an extra arm.
- **Falsifier:** D4 must first find a low-pAUC melanoma cluster that is enriched for <40.
  Without one, N6 does not address goal 1.

#### N7. Lesion self-asymmetry encoder (goal 1)

- **Biology:** §1.3. The discriminating cue is symmetry of peripheral structure, not its
  presence.
- **CV mechanism:**
  - Align each lesion to its principal axes (mask moments).
  - Run the shared trunk on the image and on its reflections about both axes.
  - The asymmetry token = |F − reflect(F)|, pooled over the lesion rim, and it goes into the head.
  - A cheap approximation reflects the feature map instead of the image. It is inexact, because
    convolutions are not reflection-equivariant, but it works as a screen.
- **Not a repeat:** B9 (polar) encodes *radial distribution*; N7 is explicit
  *self-comparison*.
- **Cost:**
  - Exact version: about 2× forward, roughly 130–150 min per fold-0 run (extrapolated, so
    benchmark it).
  - Feature-map version: about 1×.
- **Falsifier (CPU, before any GPU):** an asymmetry score computed from *existing* OOF features
  alone must separate <40 melanoma from histo-confirmed nevus above chance. If it cannot, skip
  N7.

#### N8. Physically plausible acquisition randomisation (goal 2; target list from D3)

- **CV mechanism:** randomise the nuisance factors that D3 shows carry archive identity:
  - circular aperture and vignette masks (MSKCC/BCN) vs rectangular crops (HAM);
  - synthetic hair, ruler and gel-bubble overlays;
  - resample-and-re-JPEG to mimic each archive's resolution and compression;
  - field-of-view (zoom) jitter;
  - **colour jitter in chromophore/OD space** (±10% melanin/haemoglobin scale, illuminant offset).
    This replaces HSV jitter, which the plan bans because it breaks colour semantics.
- **Not a repeat:** Mixup/CutMix mixes *labels*, and colour constancy *normalises*. N8
  randomises the *measured* nuisance factors.
- **Falsifier:** archive decodability from features must fall *and* LOAO Macro-F1 must rise. If
  decodability falls but LOAO does not move, archive identity was not the cause.

#### N9. Class-conditional acquisition invariance (goal 2; ceiling test after N8)

- **CV mechanism:** a gradient-reversal adversary that predicts **archive given class**
  (conditional-invariance domain generalisation).
- **Why this is not the barred age-invariance:**
  - Age is *diagnostic*, so removing it hurt (V3/S45).
  - Archive is not biologically diagnostic.
  - Conditioning on class keeps legitimate case-mix differences.
- **Risk:** archive is 0.97-decodable, so the adversary may be unstable. Run only after N8, with
  in-distribution Macro-F1 retention ≥ −0.01 as a hard stop.
- **Falsifier:** within-class archive decodability must fall, and LOAO Macro-F1 must rise.

#### N10. Domain-adaptive self-supervised pretraining, then full fine-tune (goal 2)

- **Mechanism:** continue ConvNeXt-V2's FCMAE (masked-autoencoder) pretraining on **unlabelled**
  dermoscopy from many devices, then fine-tune end-to-end.
  - Sources: ISIC Archive images outside the training corpus, plus ISIC-2020's ~27k
    "unknown"-diagnosis images *if* ISIC-2020 is not the confirmation cohort.
  - The trunk sees the target domains' image statistics without any labels.
- **Not a repeat:** the barred items are *frozen* foundation-model probes. Here the trunk keeps
  training and is fine-tuned.
- **Leakage rule:** no image from the confirmation cohort may enter pretraining. Use the same
  duplicate screen as N5.
- **Cost:** the largest item here and **unmeasured**. Run `scripts/gpu_benchmark.py` on an FCMAE
  step before quoting hours.
- **Falsifier:** LOAO Macro-F1 gain over the identical fine-tune from ImageNet/IN-22k weights.

### Tier 3 — cheap, narrow or data-gated

#### N11. Unsupervised label-shift adaptation for a new site (goal 2 Macro-F1 only)

- **Mechanism:** estimate the new site's class priors from unlabelled predictions with the
  Saerens EM algorithm, then reweight posteriors before the argmax. CPU only.
- **Evidence it matters:** priors differ sharply between sites. BCN is melanoma/BCC-heavy, and
  PAD is about 77% escalating (S8b), where a prior shift mechanically moved the argmax.
- ⚠️ **Registry:** S58 barred "global prior correction" **as an under-40 ranking fix** (it cannot
  change ranking, and it does not claim to). N11's endpoint is cross-site **Macro-F1** (the argmax
  changes). Register it with that endpoint only, and never report it as an under-40 intervention.
  S68 (NULL) recalibrated referral budgets, which is a different target.
- **Falsifier:** LOAO Macro-F1 with the EM prior versus without it.

#### N12. Route, don't reject: modality-routed smartphone expert (goal 2b, cross-modality)

- **Evidence:** the S69 gate detects smartphone images at AUROC 0.9994 and currently *refuses*
  them. A HAM→PAD warm-started ConvNeXt-Tiny reached **0.760** 5-class Macro-F1, against
  0.12–0.19 zero-shot.
- **Mechanism:** the gate *routes* smartphone images to a clinical-photo expert, warm-started
  from the V5 dermoscopy trunk, instead of refusing them. N2's chromophore channels transfer to
  this modality.
- **Requirements:** the PAD checkpoints are lost, so this needs retraining. Use a fresh
  **patient-grouped** CV protocol, because the old PAD test split has already been read.
- **Optional sub-arm:** PAD's patient-reported history (grew / changed / bled — the "E" of
  ABCDE and the paediatric "Bleeding/Bump") is *information*, not demographics. It needs its own
  registry justification against the barred "tabular metadata fusion", whose tested variables
  were age/sex/site.

#### N13. Ugly-duckling patient context (goal 1; blocked by data)

- **Biology:** in young patients with many nevi, the strongest clinical signal is a lesion that
  differs from the patient's *own* nevi [lit].
- **Mechanism:** a set encoder over all lesions of one patient, scoring each lesion's deviation
  from that patient's nevus distribution.
- **Blocked:** `manifest_v4.csv` has no patient ID. ISIC-2020 has one, which conflicts with using
  ISIC-2020 as the external cohort. Owner decision; this is a candidate headline for **V6**.

### Rejected in this brainstorm

- **Multi-view lesion fusion:** measured today, no gain (§0.3).
- **Age-conditioned FiLM / feature modulation:** a form of the barred metadata fusion.
- **Diffusion-synthesised young melanomas:** a generator cannot invent morphology it has not
  seen, and it risks artifact shortcuts. Revisit only as augmentation with real-only validation,
  and only after N5.

---

## 4. Link hypothesis (pre-register with the ideas)

> **H-link:** under-40 escalating lesions are concentrated in the archives the model ranks worst
> (§0.1), so goal-2 (cross-site) interventions will also raise under-40 pAUC, mostly on BCN/MSKCC
> rows.

**Test:** every goal-2 candidate reports under-40 pAUC per archive on its LOAO hold-outs. If
cross-site gains do not carry over to under-40, the two goals are separate problems and the paper
should say so.

---

## 5. Recommended portfolio and order

| Step | Items | Compute |
|---|---|---|
| 0 | D1–D4 (CPU); N7's CPU pre-check; N5's learning-curve check (2–3 fold-0 runs) | ~1 CPU day + ~2–4 GPU h |
| 1 | N1 LOAO infrastructure. Time one LOAO run | measure |
| 2 | **N2, N3, N4** fold-0 screens beside Stage 2 (cheapest, most mechanistically distinct) | 3 × 64–75 min |
| 3 | N6 (if D4 finds a subclass) as the S13 variant; N7 (if its pre-check passes) | 1–2 runs |
| 4 | Goal-2 track on LOAO: N11 (CPU) → N8 → N9 | 3 runs per candidate |
| 5 | N5 data expansion and N10 SSL: the big levers, long lead time, start sourcing now | N10 unmeasured |
| 6 | N12 routed smartphone expert | PAD retrain |

**Qualitative expected impact** (judgement, not measurement):

| Idea | Goal 1 (under-40) | Goal 2 (cross-domain) |
|---|---|---|
| N5 data | High | High |
| N2 chromophores | Medium | Medium |
| N3 noisy-OR MIL | Medium | Low |
| N4 verification-aware negatives | Medium | Medium |
| N8, N9, N10 | Low–Medium via H-link | Medium–High |
| N12 routing | Low | High on smartphone |

N1 is not a model; it is what makes a goal-2 claim possible at all.

**Adoption:** chosen items go into `docs/v5_design/V5_PLAN_AMENDMENT_02.md`. Each gets a no-repeat
registry entry (`new_mechanistic_difference`, `new_endpoint`, `why_not_duplicate`), a declared
hard-core subset, and a falsifier. The amendment is hashed before any of its GPU runs.

---

## 6. Literature to verify before any of it reaches the paper

These are cited from memory. Confirm each reference before relying on it.

- Cordoro KM et al., *J Am Acad Dermatol* 2013 — paediatric melanoma cohort and modified ABCD
  criteria.
- Pampena R et al., *J Am Acad Dermatol* 2017 — meta-analysis of nevus-associated vs de novo
  melanoma (the association with younger age and thinner tumours).
- Age-stratified dermoscopy of melanoma, 285 cases, 2025 — PubMed 40805292 (already in the plan).
- Tsumura N et al., *SIGGRAPH* 2003 — melanin/haemoglobin decomposition of skin colour by ICA.
- Ruifrok AC, Johnston DA, *Anal Quant Cytol Histol* 2001 — colour deconvolution (the OD-basis
  analogue).
- Bissoto A et al., *CVPR Workshops (ISIC)* 2019 — "(De)Constructing Bias on Skin Lesion
  Datasets", above-chance performance with the lesion removed.
- Oakden-Rayner L et al., *CHIL* 2020 — hidden stratification.
- Sohoni N et al., *NeurIPS* 2020 — "No Subclass Left Behind" (GEORGE).
- Ilse M et al., *ICML* 2018 — attention-based deep multiple-instance learning.
- Li Y et al., *ECCV* 2018 — conditional invariant adversarial domain generalisation.
- Saerens M et al., *Neural Computation* 2002 — adjusting classifier outputs to new priors (EM).
- Woo S et al., *CVPR* 2023 — ConvNeXt V2 / FCMAE.
