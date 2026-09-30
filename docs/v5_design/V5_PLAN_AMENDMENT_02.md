# V5 Plan Amendment 02 — Biology-derived arms and a 10-night schedule to 9 October

**Date:** 2026-09-29
**Status:** **ADOPTED 2026-09-30** (owner accepted every item), hashed into
`results/v5/v5_plan_freeze.json` as `amendments[1]` with the runsheet and the DRE, **before** the
first run of any item it introduces. Where this file and `docs/V5_RUNSHEET.md` differ (revision of
30 Sep, audit AU17–AU36), the runsheet wins. S01 (§7, night 1) is already registered by the frozen plan plus Amendment 01 and
does not wait for this file.
**Amends:** `docs/v5_record/V5_MASTER_RESEARCH_PLAN_REVISED.md` (freeze `B7542DAB…`) as already amended by
`docs/v5_record/V5_PLAN_AMENDMENT_01.md` (`219634DB…`). Neither file is edited.
**Adopts from `docs/v5_design/V5_IDEAS_BIOLOGY_FIRST.md`:** N1, N2, N3, N4, N7 (screen version), N8. It adds
M2 (two-step head) and M5 (expert structure supervision), which are new here.
**Revision 2026-09-29 (pre-adoption, so not post-hoc):**
- M1: sRGB linearisation, OD clamp, glare exclusion, and a depth channel (bounded, melanin-gated),
  fed through a 6-channel stem. A 1×1 6→3 adapter was considered and rejected (§4 M1).
- M3: explicit normalised-LSE formula; patch maps saved at inference.
- M4: hinge ranking with m = 0.20, per-sample loss masking, pAUC_histo defined.
- M5: head specified, has_attr indicator, joint augmentation, overlap pre-flight.
- M6 fallback: D₄ min/max plus a chromophore asymmetry token.
- M7: parameters fixed.
- §6/§7/§10: checkpoint, worker and timeline constraints.

**Revision 2 (2026-09-29, pre-adoption):**
- New arms: GeM mechanism control and a SWAD secondary. M7 extended with the evidence-based OOD
  augmentations.
- Group-DRO moved to V6. §5 split table updated.
- §6 gains the declared-score rule, same-night nesting and a multiplicity note. §7 now points to
  `docs/V5_RUNSHEET.md`.
- §8 gains Q5 and S75/MILK10k. §10 gains the audit risks.
- Audit: `docs/v5_design/V5_PLAN_AUDIT.md`. **Adopted and hashed together** with the DRE and the runsheet.

**Companion:** `docs/v5_design/V5_DERM_REASONING_ENGINE.md` (the Dermatologist Reasoning Engine) is adopted and
hashed with this file. It replaces M6 with the mask-free geometry block, adds the colour palette,
periphery analysis, subtype memory, clinical-logic head and second look, and supersedes §6/§7
screening blocks where they differ.

Counts marked *(computed 2026-09-29)* are read-only tabulations of `ml/data/manifest_v4.csv`, on
the 15,294 `split == train` rows. Every other number names its source.

---

## 0. Bottom line

1. **What V5 currently is.** It is a pre-registered, hash-frozen programme of 10 sessions
   (S76–S85). Its stages are:
   - integrity audit;
   - a 384 px control;
   - about 12 representation arms (dual-stream, escalation head, hierarchy, micro-patches, context
     decomposition, SupCon, wavelets, GeM, ranking loss, and others);
   - Group-DRO;
   - three new backbones;
   - an ensemble, the safety stack, and one locked external read.

   Amendment 01 fixed its statistics: two-level gates, fold 0 used for selection and folds 1–4 for
   confirmation, paired Δ gates, a hierarchical seed CI, and S56 as the safety layer.
2. **Its problem for 9 October.** The plan's own estimates add up to about 50 GPU hours before any
   finalist cross-fitting. Most arms are generic computer-vision capacity changes. The biology sits
   in §6 as prose, but no arm is *derived from* a specific biological mechanism, and the D0 findings
   (the under-40 gap is largely archive plus case mix, and the real differential is melanoma vs a
   histologically confirmed suspicious nevus) are not in it.
3. **This amendment:**
   - replaces the Stage 2/3 menu with **biology-derived arms** (M1–M7 here, plus the Dermatologist
     Reasoning Engine modules in the DRE companion), each built from one step of how a
     dermatologist actually decides (§3);
   - screens them cheaply (224 px, 2 seeds, fold 0);
   - stacks the survivors into **one composite**;
   - confirms that composite on folds 1–4 × 3 seeds against a matched control, at the resolution
     S01 chooses (384 px, or 224 px if S01 fails; audit AU3).

   It runs on 9 nights plus daytime control runs on the RTX 5050 (`docs/V5_RUNSHEET.md`).
   Review 2 (9 Oct) shows whatever is complete. The rest runs afterwards under the same hash.
4. **Realistic expectation.** V4 proved that decision layers cannot manufacture ranking. Every arm
   here changes the **representation**, which is the right layer. Even so, with 65 under-40
   escalating lesions in folds 1–4, only a large under-40 gain can be *certified*. The all-age
   pAUC, the histo-only pAUC and the mechanism checks are where a positive result is most likely to
   be demonstrable by 9 October.

---

## 1. Skin optics: why dermoscopy colours are biology

A dermatoscope removes surface glare (by immersion fluid or cross-polarised light), so the image
is light that has entered the skin, been absorbed by **chromophores** and scattered back. What
colour a structure shows is set by *what* absorbs and *how deep* it sits.

| Colour | Chromophore / tissue | Depth | Seen in |
|---|---|---|---|
| Black | Melanin | Stratum corneum / upper epidermis | Melanoma (peripheral black dots), pigmented SK, lentigo |
| Light to dark brown | Melanin | Epidermis / dermo-epidermal junction (DEJ) | Nevus, melanoma, lentigo, pigmented BCC |
| Grey | Melanin in melanophages | Papillary dermis | Regression (melanoma), LPLK |
| Blue / blue-grey | Melanin | Deeper dermis; long wavelengths escape, blue is back-scattered (Tyndall) | Blue nevus, BCC ovoid nests, invasive melanoma (blue-white veil) |
| Red / pink | Oxy-haemoglobin | Vessels in papillary dermis | BCC, AK/SCC, amelanotic melanoma, vascular, Spitz |
| Purple / blue-black | Deoxy-haemoglobin, thrombosis | Vascular spaces | Angiokeratoma, thrombosed haemangioma |
| White (dull) | Fibrosis / scar, reduced melanin | Dermis | Regression, DF central patch |
| White (shiny, polarised only) | Altered collagen birefringence | Dermis | Melanoma, BCC, DF, Spitz |
| Yellow / white dots | Keratin, sebum | Epidermis / infundibulum | SK milia-like cysts, AK/SCC scale |

**Pigment network** is the DEJ seen from above. The brown lines are melanin-rich rete ridges, and
the holes are the dermal papillae between them. A *regular* network means orderly junctional
melanocytes (nevus). A *thick, irregular or broken* network means disordered proliferation
(melanoma). A *negative* (inverse) network means pale rete between pigmented nests (Spitz,
melanoma). Facial skin has flat rete ridges, so there is no network there, only a
*pseudo-network* around follicles.

**Two physical facts matter for the model:**
- **Colour is a mixture of two absorbers plus shading.** In optical density (OD = −log(I/I₀)),
  melanin and haemoglobin add linearly, and illumination or camera gain becomes a constant offset
  along (1,1,1) (Tsumura's decomposition). So an RGB network has to *learn* an unmixing that
  physics gives for free. It learns it from a training set that is 70% older-patient pigmented
  melanoma, and in doing so it under-learns the haemoglobin axis on which low-pigment young
  melanoma and superficial BCC carry their evidence.
- **Polarisation changes which structures exist in the image.** Shiny white lines/strands and
  rosettes appear only under polarised dermoscopy. Milia-like cysts and blue-white veil are more
  conspicuous without polarisation. Archives differ in device and mode, so a structure the model
  relies on can be *physically absent* at another site. That is a biological route to
  cross-domain loss that colour normalisation cannot fix (see D5).

---

## 2. The seven classes: biology → dermoscopy → where it sits in our data → what the model must see

Dev counts are images in the 15,294-row pooled train split *(computed 2026-09-29)*.

### 2.1 Melanoma (mel) — 2,525 images; <40: 133

- **Biology.** A malignant proliferation of melanocytes.
  - **Radial growth phase:** atypical melanocytes spread along the DEJ, singly and in irregular
    nests. Dermoscopically this gives an atypical, thick or broken network, irregular dots and
    globules, and, where nests reach the edge, **segmental radial lines / pseudopods**.
  - **Vertical growth phase:** invasion into the dermis gives blue-white veil, blue-black areas,
    shiny white lines and atypical polymorphous vessels.
  - **Immune regression:** grey peppering (melanophages) and white scar-like areas.
- **The organising principle is disorder:** asymmetry of *structure and colour*, and ≥ 3 colours.
- **Age:**
  - **Older patients, chronic sun damage (face, 60+):** lentigo maligna (asymmetric follicular
    pigmentation, rhomboidal structures, annular-granular pattern) and regression.
  - **Under 40:** mostly superficial-spreading, thin, and often **nevus-associated** (the malignant
    part is focal and eccentric inside a benign nevus). Growth-phase structures predominate:
    pseudopods and *asymmetric* peripheral globules (PubMed 40805292, 2025).
  - **Youngest patients:** more amelanotic, nodular and spitzoid. The evidence is vascular (dotted
    or polymorphous vessels, milky-red areas), not pigment.
- **Our data.**
  - 60+ contributes 1,500 of 2,525 mel images. The model's melanoma prototype is an old,
    pigmented, regressing melanoma.
  - The 133 under-40 mel images: BCN 70 / HAM 51 / MSKCC 12.
  - Sites of under-40 mel: lower extremity 36, anterior torso 28, head/neck 28, upper extremity
    26, posterior torso 13.
- **What the model must see:** (i) *focal* malignant evidence inside an otherwise benign lesion;
  (ii) *asymmetry* of peripheral structures, not their mere presence; (iii) the haemoglobin
  channel when pigment is low.

### 2.2 Melanocytic nevus (nv) — 8,038 images; <40: 2,405

- **Biology.** Benign nests of melanocytes, organised and symmetric: one or two patterns (reticular,
  globular, homogeneous) and a uniform colour.
- **Nevus morphology is age-dependent:**
  - **Children and young adults:** globular pattern, and a **symmetric peripheral rim of globules**
    in growing nevi. Spitz/Reed nevi show a **symmetric starburst** of streaks.
  - **Adults:** reticular.
  - **Elderly:** fewer nevi, dermal/homogeneous, involuting.
- **Our data.**
  - **2,405 of the 2,721 under-40 images (88%) are nevi.** The young band is overwhelmingly benign,
    and its benign lesions show the same *growth* structures (globules, streaks) as young melanoma.
  - HAM's 2,592 `follow_up` images are all nevi, captured by the monitoring camera (D0 §0.2).
- **Model consequence.** On young skin, "has peripheral globules/streaks" does *not* separate
  melanoma from nevus. **Symmetry and focality do.** The easy follow-up nevi teach a device
  shortcut rather than this boundary.

### 2.3 Basal cell carcinoma (bcc) — 1,928 images; <40: 52

- **Biology.** A tumour of basal keratinocytes. It is **not melanocytic**, so there is **no pigment
  network**. Its structures:
  - **Arborising vessels:** thick, branching, in sharp focus, because the tumour thins the overlying
    epidermis.
  - **Pigmented tumour nests** in the dermis, which show as blue-grey ovoid nests and globules.
  - **Leaf-like and spoke-wheel areas** where nests touch the epidermis.
  - **Ulceration, and shiny white blotches/strands** (polarised).
  - **Superficial BCC**, common in younger patients and on the trunk, is pink-white with fine
    short telangiectasia and multiple small erosions: almost no melanin.
- **Our data.** 40 of the 52 under-40 BCC images are BCN.
- **What the model must see:** *absence* of network plus haemoglobin morphology.

### 2.4 Actinic keratosis / intraepidermal carcinoma / SCC (akiec) — 874 images; <40: 8

- **Biology.** Dysplastic keratinocytes on chronically sun-damaged skin (face, scalp, hands), on a
  continuum from AK through Bowen's disease to invasive SCC. Its structures:
  - **Strawberry pattern:** a red pseudo-network around follicles with white-yellow follicular
    plugs.
  - Scale, and rosettes (polarised only).
  - **Glomerular/dotted vessels in clusters** (Bowen's).
  - In SCC, white circles and keratin masses.
- **Our data.** Almost entirely an older-patient class, with 8 young images. It matters for Macro-F1
  and for the older bands, not for the under-40 gap.
- **What the model must see:** red plus white/yellow keratin; a haemoglobin-plus-keratin signature.

### 2.5 Benign keratosis (bkl: SK, solar lentigo, LPLK) — 1,657 images; <40: 72

- **Biology.** Proliferation of epidermal keratinocytes with keratin retention. Its structures:
  - **Seborrhoeic keratosis:** milia-like cysts (keratin pearls), comedo-like openings,
    fissures/ridges ("brain-like"), a sharp moth-eaten border.
  - **Solar lentigo:** fingerprint pattern.
  - **Lichen-planus-like keratosis:** *grey granular* pattern, an immune regression that imitates
    a regressing melanoma.
- **Why it matters.** It is the main mimic of the *older* melanoma prototype: grey structures and
  blotches of regression.
- **What the model must see:** keratin structures (milia-like cysts, comedo openings), which are
  melanin-independent and polarisation-sensitive.

### 2.6 Dermatofibroma (df) — 114 images; <40: 18

- **Biology.** A dermal fibrohistiocytic proliferation. The classic pattern is a central white
  scar-like patch (fibrosis) with a delicate peripheral network (the overlying basal layer is
  hyperpigmented). Variants are frequent (PubMed 18209171).
- **Why it matters.** A central white area plus peripheral network can imitate regression, and a
  young, leg-predominant population overlaps with young melanoma sites.

### 2.7 Vascular (vasc) — 158 images; <40: 33

- **Biology.** Ectatic or proliferating vessels (haemangioma, angiokeratoma, pyogenic granuloma).
  Red/blue/purple **lacunae** separated by septa, no melanin network, and a whitish veil in
  angiokeratoma.
- **What the model must see:** a nearly pure haemoglobin (oxy/deoxy) signal. This is the easiest
  class for chromophore inputs to help and a clean positive control for M1.

### 2.8 The under-40 problem, restated biologically

- **The young band is 88% nevus.** Its escalating lesions are 133 mel, 52 bcc and 8 akiec images
  (81 lesions; BCN 35, HAM 34, MSKCC 12).
- **The young melanomas most likely to be missed** are those a pigment-centric, globally pooled,
  old-melanoma-prototype network is worst placed to see:
  - **nevus-associated:** focal and eccentric;
  - **growth-phase:** structures shared with growing nevi, where only their asymmetry differs;
  - **low-pigment:** the evidence is haemoglobin.
- **On top of that,** 58% of them come from the two archives the model ranks worst (D0 §0.1). The
  HAM follow-up nevi teach a device cue that does not exist at those sites.

---

## 3. Reverse-engineering the dermatologist's decision into the network

Dermoscopy teaching reduces diagnosis to a small number of steps: the two-step algorithm and
"chaos and clues" (Kittler/Rosendahl; Argenziano consensus). Each step is a computation the model
can be given explicitly. This is the organising idea of the amendment.

| Step | What the clinician computes | Biological basis | Module | Arm |
|---|---|---|---|---|
| 0 | See the colours as tissue: pigment vs blood vs keratin, and depth | Chromophore optics (§1) | Fixed optical-density unmixing into melanin / haemoglobin / shading channels, fed with RGB | **M1** |
| 1 | Is it melanocytic? (network, globules, streaks, homogeneous blue, parallel pattern) | Cell of origin | Auxiliary *melanocytic vs non-melanocytic* head plus a dedicated *escalation* head | **M2** |
| 2 | Chaos: is structure/colour asymmetric? | Disordered vs orderly proliferation | Asymmetry about the lesion's own axes (DRE-3), with a D₄-reflection fallback, plus chromophore asymmetry | **M6 / DRE-3** |
| 3 | Clues: is there *any* focal malignant feature (eccentric structureless zone, grey/blue, peripheral black dots, segmental radial lines, polymorphous vessels)? | Focal, eccentric malignancy (nevus-associated) | Asymmetric noisy-OR (top-k / log-sum-exp) escalation pooling | **M3** |
| 3b | Know what the clue structures *are* | Expert morphology | Auxiliary dense heads supervised by ISIC-2018 Task 2 expert masks (network, negative network, streaks, globules, milia-like cysts) | **M5** |
| 4 | Train on the real differential: melanoma vs a nevus suspicious enough to excise | Clinical verification | Verification-aware negatives plus a MEL-vs-confirmed-benign pairwise ranking loss | **M4** |
| 5 | Ignore the camera: the same disease looks the same at every site | Acquisition is not disease | Physically plausible acquisition randomisation in OD space | **M7** |

```
RGB ──► OD unmix (M1, fixed) ──► [RGB | melanin | haemoglobin] ──► ConvNeXt trunk ──► F (feature map)
                                                                          │
      ┌──────────────────────┬────────────────────────┬──────────────────┼────────────────────────┐
      ▼                      ▼                        ▼                  ▼                        ▼
 7-class head (CE)   melanocytic head (M2)   escalation map ─► noisy-OR (M3)   asym token (M6)   attribute heads (M5)
      ▲                                               ▲                                     (Task-2 masks, where present)
      └──────────── M4: ranking loss on escalation score, mel vs confirmed benign ──────────┘
```

**What is kept from the frozen plan:**
- the escalation head (B2) and the hierarchy (B3), merged into M2 (Amendment 01 B9 already nests
  them);
- the MEL-vs-NV ranking loss (B12), absorbed into M4;
- Group-DRO (S80), **later moved to V6** as subclass-DRO (rev. 2, audit AU6);
- the 384 px control (S01);
- the whole statistics, integrity and safety apparatus of Amendment 01.

---

## 4. The arms

Every arm has a no-repeat registry entry (`new_mechanistic_difference`, `new_endpoint`,
`why_not_duplicate`), a declared hard-core subset (Amendment 01 B1) and a falsifier. Hyper-
parameters are **fixed here** and none is tuned. Screening is 224 px on fold 0 with seeds 42 and
43, paired against the 224 px control.

### M1 — Chromophore input channels (N2)

- **Mechanism (rev. 2026-09-29):**
  1. Inside the model, un-normalise the RGB tensor to I ∈ [0,1].
  2. **Linearise sRGB** with the inverse sRGB gamma. Beer–Lambert additivity holds for linear
     intensity only. Gamma-encoded values make melanin and haemoglobin mix non-linearly and bias
     the basis.
  3. **Clamp** I_lin to [1/255, 254/255]. The lower bound removes the log(0) singularity; the upper
     bound keeps OD strictly positive, so the depth ratio in step 6 stays finite.
  4. **Exclude glare from basis fitting:** pixels with any channel ≥ 250/255 in sRGB (specular
     reflections from immersion fluid or bubbles) carry no chromophore information. They are left
     out of the ICA fit and of DRE-2's geometry, and at inference they pass through clamped.
  5. Compute OD = −ln(I_lin), natural log throughout. Project it onto a **fixed 3×3 basis**
     (melanin, haemoglobin, shading = (1,1,1)). The basis is estimated **once per fold, from that
     fold's training pixels only**, by ICA on about 10⁶ sampled non-glare OD pixels (Tsumura).
  6. **Third channel, dermal-melanin depth.** Epidermal melanin absorbs blue much more than red, so
     brown pigment has OD_B ≫ OD_R. When melanin lies in the dermis, the overlying collagen
     back-scatters short wavelengths before they reach it, while red penetrates and is **absorbed**
     by the deep melanin. OD_B therefore falls relative to OD_R, and the lesion looks blue-grey
     (Tyndall effect). The blue-to-red OD ratio thus encodes depth, and **lower means deeper**:
     - c_depth = tanh( ln((OD_B + ε)/(OD_R + ε)) − μ_skin ) · σ((c_mel − t_mel)/w), with ε = 0.02
       OD. μ_skin is the median log-ratio of the image's outer border ring, t_mel is the fold's
       median lesion melanin, and w = 0.1·t_mel.
     - The log makes the ratio symmetric, tanh bounds it, and the **melanin gate** zeros it where
       there is no melanin. Without the gate, the ratio explodes on bright skin and glare (OD_R ≈ 0)
       and is driven by haemoglobin, which absorbs blue at 415 nm. It is **not a vascular channel**:
       vascular evidence belongs to c_hb.
  7. **6-channel stem.** Standardise [c_mel, c_hb, c_depth] with the fold's training mean and SD,
     concatenate them to RGB, and widen ConvNeXt's 4×4 patchify stem conv from 3 to 6 input
     channels. The pretrained RGB filters are kept, and the 3 new input slices are initialised to
     **0**, so the epoch-0 output is bit-identical to the RGB model.
     - A 1×1 6→3 adapter placed *in front of* an unchanged stem is **rejected**. It also starts
       identical, but it forces all six channels through a rank-3 per-pixel bottleneck, so every
       chromophore signal the model uses must displace RGB information. The widened stem costs
       96 × 3 × 16 ≈ 4.6k extra parameters and has no bottleneck.
- **Cost.** A fixed 1×1 linear layer on the GPU is effectively free. This **replaces** the ideas
  doc's disk precompute: the 18.8 ms/image warning applied to float64 CPU transforms in the
  dataloader, not to a GPU matmul.
- **Hard core:** under-40 escalating lesions in the bottom melanin tercile, plus bcc and vasc F1.
- **Falsifiers:**
  - The gain must concentrate in the low-melanin tercile and in bcc/vasc.
  - Shuffling the haemoglobin channel across images at inference must remove the gain.
  - A uniform gain across pigment levels means capacity, not chromophores.
  - **Depth channel:** zeroing c_depth at inference must cost more on lesions with blue-grey
    areas (DRE-1 palette) than on lesions without.
- **QC (CPU, before the run):** on 20 HAM images per class, the decomposition must put vasc and
  bcc mass in c_hb, and blue nevus / blue-white-veil regions low on c_depth. Visual QC sheet saved
  to `results/v5/diagnostics/m1_chromophore_qc.png`.
- **Not a repeat:** colour constancy *normalised* the illuminant and kept RGB as the only input.
  M1 *adds* physically defined channels and keeps RGB.

### M2 — Two-step head: melanocytic gate plus escalation head (B2 + B3 merged)

- **Mechanism:** two auxiliary BCE heads on the pooled feature:
  - *melanocytic* = {mel, nv} vs the rest;
  - *escalating* = {mel, bcc, akiec} vs the rest.

  Each has loss weight 0.5 (Amendment 01 A13 keeps the default of 0.5). The 7-class CE is
  unchanged. The escalation logit is the score used for the escalation pAUC.
- **Hard core:** under-40 bcc/akiec misclassified as nv or bkl (the melanocytic/non-melanocytic
  confusion).
- **Falsifier:** the gain must show up as fewer melanocytic ↔ non-melanocytic confusions in the
  confusion matrix. A gain that sits only within melanocytic (mel ↔ nv) is not step 1 at work.
- **Why merged:** B3 nests B2 (Amendment 01 B9). The clinical algorithm asks both questions
  together, and merging saves an arm.

### M3 — Focal-clue noisy-OR escalation pooling (N3)

- **Mechanism:**
  - Replace M2's globally pooled escalation logit with a patch-level logit map
    P = Conv1×1(F3) ∈ ℝ^{B×1×H′×W′}. It sits on the **stride-16** stage-3 map (14×14 at 224 px,
    24×24 at 384 px), not the 7×7 final map, which is too coarse to localise a focal clue.
  - Aggregate with normalised log-sum-exp, K = H′W′ and r = 4.0 (fixed):
    s_esc = (1/r) · ln( (1/K) Σᵢ exp(r·pᵢ) ).
    Because of the 1/K, s_esc always lies between mean(p) and max(p). It approaches the max
    ("malignant if any region is") as r grows, and r = 4 keeps gradients on more than one patch.
    It is computed with `torch.logsumexp` for numerical stability.
  - The benign and 7-class paths keep mean pooling.
  - **At inference, P is saved** for every OOF image as float16 (≈ 1.1 KB per image at 384 px,
    ≈ 17 MB per fold set). It is the heat map for the eccentricity falsifier below and for DRE-5/8.
  - It is screened **against M2**, so the pair isolates the aggregation rule.
- **Hard core:** under-40 mel missed by the control whose prediction was nv (the nevus-associated
  candidates).
- **Falsifier:** on rescued lesions with a HAM mask, the top-k patches must be **eccentric**
  (centroid off the mask centroid, and covering less than half of the mask). Diffuse evidence
  means the nevus-associated mechanism is not what is working.

### M4 — Train on the clinical differential (N4 + B12)

- **Mechanism:**
  - **(a) Loss masking.** The HAM `follow_up` nevi (2,592 dev images, all nv; D0 §0.2) **stay in
    the 7-class CE**, so the class priors are unchanged against the control. They get a per-sample
    weight of **0** in the auxiliary escalation BCE (M2) and are **never drawn into ranking
    pairs**. This is implemented as a per-sample `esc_weight` column in the dataset, not by
    dropping rows, so batch composition matches the control.
  - **(b) In-batch pairwise margin ranking loss** (weight 0.5) on the escalation logit s_esc.
    P is the set of all in-batch (mel_i, confirmed_benign_j) pairs:
    L_rank = (1/|P|) · Σ_{(i,j)∈P} max(0, m − (s_esc(x_i) − s_esc(x_j))), with m = 0.20 on the logit
    scale (fixed).
    - The hinge only produces gradient on pairs that are ordered wrongly or nearly tied. That makes
      it hard-pair focused, which suits a pAUC at low FPR.
    - The escalation BCE anchors the logit scale, so the margin cannot be met just by inflating
      logits.
    - **Pairs are formed per micro-batch.** At 384 px that is batch 16 × accumulation 2, so a
      micro-batch holds only about 2–3 mel. A micro-batch with |P| = 0 contributes 0 (no NaN).
    - The sampler is **not** changed, to keep the comparison with the control clean. The mean |P|
      per micro-batch is logged, so a starved loss is visible.
    - **Confirmed benign** = **histopathology-confirmed** benign in any archive. D5 (30 Sep,
      `results/v5/diagnostics/d5_acquisition.json`) narrowed the pool as foreseen here: among
      benign images, HAM 3,386/8,061, BCN 3,713/5,579, MSKCC 338/2,351 are histopathology (audit
      AU27; `research/v5/confirmation.py`).
  - **(c) Co-primary endpoint:** **histology-only escalation pAUC@0.20 (pAUC_histo)** is logged on
    every screen and confirmation. Its population is every escalating row plus the
    histopathology-confirmed benign rows in any archive (D5; AU27), and the HAM-histo-only subset
    is reported separately (D0's definition).
- **Falsifier:** histo-only pAUC must improve. An all-rows gain without a histo-only gain is case
  mix, not a better boundary.
- **Not a repeat:** class reweighting weighted *classes*, and S67 reweighted *model errors*. M4 uses
  *clinical verification status*, which the model cannot see.

### M5 — Expert structure supervision from ISIC-2018 Task 2 (new)

- **Data.** ISIC-2018 Task 2 provides expert masks for five dermoscopic attributes on 2,594 images:
  pigment network (58.7%), globules (23.2%), milia-like cysts (26.3%), negative network (7.3%) and
  streaks (3.9%) ([ISIC 2018 Task 2](https://challenge2018.isic-archive.com/task2/training/);
  Codella et al., [arXiv:1902.03368](https://arxiv.org/pdf/1902.03368)).
  - Their image IDs lie in the ISIC_0000000–ISIC_0016072 range. In our manifest that range holds
    **2,903 MSKCC images: 1,609 train, 417 val, 877 reserved** *(computed 2026-09-29)*.
  - The 1,609 train rows include **all 12 under-40 MSKCC melanomas**, the archive where under-40
    ranking is at chance (D0 §0.1).
  - **The actual overlap must be verified** against the downloaded Task 2 ID list before adoption.
- **Mechanism:**
  - **Head:** on the stride-16 map F3 (384 channels), apply Conv3×3(384→128) → GELU →
    Conv1×1(128→5). This gives five mask logits at H′×W′ (pigment network, negative network,
    streaks, globules, milia-like cysts). About 0.44 M parameters.
  - **Loss:** BCE plus soft Dice per attribute, with total weight 0.20. Targets are the expert
    masks, **area-downsampled** to H′×W′ (soft targets in [0,1], not nearest-neighbour, so thin
    streaks survive).
  - **Sample indicator:** the dataset returns `has_attr ∈ {0,1}` and a zero mask for rows without
    Task 2 labels. The loss is Σ_b has_attr_b · ℓ_b / max(1, Σ_b has_attr_b), so unlabelled
    images add exactly 0 and a batch with no labelled image cannot divide by zero.
    - An attribute that is **absent** in a labelled image is a valid all-zero target (has_attr = 1).
    - An image **not in Task 2** is unlabelled (has_attr = 0).
  - **Joint geometric augmentation (required):** every spatial transform applied to the image
    (random resized crop, flips, rotation) is applied identically to its 5 masks. Colour and
    optical-density transforms are applied to the image only. Without this the supervision is
    spatially misaligned. `train_v4`'s image-only torchvision pipeline cannot do this, so
    `train_v5` uses paired transforms (`torchvision.transforms.v2` with `tv_tensors.Mask`). A unit
    test must assert mask/image alignment on a synthetic image.
  - The loss applies **only on train-split rows that have masks**, and each row keeps its S71
    fold.
  - **Masks for val or reserved rows are never loaded.**
- **Biology it encodes:** network means melanocytic (step 1); negative network and streaks are
  melanoma/Spitz clues (step 3); milia-like cysts point to SK; globules separate nevus from
  melanoma by *distribution*.
- **Policy check:** these are expert annotations, as §10 requires. They are not Grad-CAM,
  attention or pseudo-labels.
- **Falsifiers:**
  - The attribute head's held-out Dice must exceed a trivial predictor (so the structures are
    actually learned).
  - The gain must appear on MSKCC rows first.
  - Shuffling masks across images during training must remove it.
- **Owner action:** download the Task 2 ground-truth zip only (the images are already on disk),
  and verify the IDs and licence (CC-BY-NC). This needs your go-ahead.
- **Pre-flight verification (day of 30 Sep, before adoption; §8 Q4):** after the download, write
  `results/v5/diagnostics/m5_overlap.json` with:
  - (i) overlapping IDs by split. **Pass rule: ≥ 1,000 train rows**, and **0** rows used from
    val or reserved;
  - (ii) how many of the **12 under-40 MSKCC mel** train images have Task 2 masks, reported and
    **not** a pass condition. M5's value does not rest on 12 images, and making it a gate would
    select on the under-40 subgroup;
  - (iii) byte-level agreement between Task 2 images and our copies on a 50-image sample, so the
    masks align with our pixels (same resolution and orientation).

### M6 — Chaos token: feature self-asymmetry (N7, screen version)

- **Superseded in the primary path by DRE-3** (`docs/v5_design/V5_DERM_REASONING_ENGINE.md`). DRE-3
  reflects about the **lesion's own centroid and principal axes**, derived mask-free from the
  chromophore maps (DRE-2). The version below is the **fallback**, used if DRE-2 fails its Q1
  geometry QC.
- **Fallback mechanism (rev. 2026-09-29): D₄ reflections instead of 2 flips.**
  - On F3, compute the four reflections of the dihedral group D₄: g_k = rot_{90k}∘flip_h,
    k = 1…4 (vertical, horizontal and the two diagonal axes).
  - d_k = mean_p ‖F3(p) − g_k(F3)(p)‖₁ / mean_p ‖F3(p)‖₁, **normalised** so a darker or
    higher-contrast lesion does not look more asymmetric just because it has more signal.
  - **Tokens: min_k d_k and max_k d_k, not the mean.** min is "symmetric about at least one
    axis" (Menzies' negative feature); max is "asymmetric about some axis" (Kittler's chaos).
    Averaging the four would blur exactly the distinction the clinical rules draw.
  - **Limits:** D₄ axes are 45° apart and pass through the *image* centre. A lesion whose symmetry
    axis is at about 22° is off by up to 22.5°, and an off-centre lesion looks asymmetric. The
    whole block only goes ahead if Q1 fails; that is why DRE-3 is primary.
- **Chromophore asymmetry token (both paths).** Using the same axes (DRE-2 when it passes, D₄
  otherwise), at image resolution:
  - A_chrom = Σ_p s(p)·(|c_mel − g(c_mel)| + |c_hb − g(c_hb)|) / Σ_p s(p)·(c_mel + g(c_mel) + c_hb + g(c_hb) + ε)
  - s is the DRE-2 soft mask (all ones in the fallback). The result is a relative asymmetry in
    [0,1] that does not scale with pigment load. Take min and max over the axes, as above.
  - These scalars go into the head. They are image-derived (not metadata), and they are exactly
    the quantities Q2/D6 tests on CPU first.
- **CPU pre-check first (D6).** A handcrafted asymmetry and colour-count score, computed from M1's
  chromophore maps on HAM rows with masks, must separate *<40 mel* from *histo-confirmed nv* at an
  AUC lower CI bound above 0.5. If it does not, **drop M6 without a GPU run**.
- **Falsifier:** the gain must concentrate on under-40 mel vs histo-nv pairs whose control score
  gap is small.

### M7 — Acquisition randomisation in OD space (N8; goal 2)

- **Mechanism (fixed parameters; each applied with p = 0.5 per image, independently).**
  Randomise the factors that D3 finds carry archive identity:
  - **Optical-density perturbations** (on linearised, clamped OD as in M1; natural log):
    - additive exposure jitter along the shading axis (1,1,1): OD ← OD + Δ·(1,1,1), with
      Δ ~ U[−0.08, +0.08]. That is about ±8% exposure (e^{±0.08}).
    - chromophore scaling: c_mel ← c_mel·(1 + u_m), c_hb ← c_hb·(1 + u_h), with u ~ U[−0.10, +0.10].
      The image is then reconstructed through the M1 basis and re-encoded to sRGB.
    - This replaces the banned HSV jitter. It changes *amount* of pigment and blood, never hue
      semantics.
  - **Synthetic aperture and vignette:** a circular aperture with radius U[0.45, 0.60] × the short
    side, black outside (BCN/MSKCC style), and a radial vignette falloff of U[0, 0.3].
  - **Resample and re-JPEG:** downsample to U[0.4, 1.0] × native size, then upsample, then JPEG
    quality U[70, 95].
  - **Field-of-view jitter:** random resized crop scale U[0.6, 1.0], replacing the control's range
    only in this arm.
  - **Evidence-based OOD components (rev. 2).** A ConvNeXt dermoscopy study with separate policy
    selection and evaluation raised held-out-institution ROC-AUC from **0.787 to 0.826** with
    in-domain AUC unchanged (arXiv 2607.26765). Its transfer-helpful components are adopted in
    physically plausible form:
    - **Planckian illuminant jitter:** blackbody colour temperature 6,500 K ± 1,500 K, p = 0.5.
      This is an illuminant change, not hue rotation, so it respects the HSV-jitter ban.
    - **CLAHE:** p = 0.4, clip 4, 8×8 tiles, on luminance only.
    - **OneOf(optical distortion, grid distortion)** at p = 0.5.
    - **OneOf(Gaussian blur ≤ 5, median blur ≤ 5, Gaussian noise σ 0.02–0.11)** at p = 0.5.
    - **Full 360° rotation plus transpose.** Dermoscopy has no canonical orientation, while the
      control uses ±20°.
  - Order: geometric → OD → Planckian → aperture → CLAHE → blur/noise → re-JPEG. M1's chromophore
    channels are computed **after** augmentation, so they see the perturbed image.
  - **Cost:** CLAHE and distortions run on the CPU. Time them in the smoke epoch. If the epoch
    time grows by more than 25%, move them to the GPU (kornia) or drop the slow component and
    note it (audit AU9).
- **SWAD secondary (free rider; rev. 2).**
  - Both LOAO `control` and LOAO `m7` keep a dense running average of weights over epochs 10–30
    (fixed window; one extra model copy in VRAM; `_swad.pt`, 111 MB per run).
  - It is evaluated on the held-out archive as a **pre-registered secondary**, with no extra
    training.
  - **Registry:** R6 (EMA) was null *in-distribution*. SWAD targets out-of-domain flat minima
    (Cha et al., NeurIPS 2021) and is read on a new endpoint (LOAO).
- **Readout:** it is screened **exclusively on LOAO (N1) Macro-F1** (three hold-outs, mean and
  per-archive), never on fold-0 in-distribution Macro-F1. Fold-0 retention (≥ −0.010) is a guard,
  not evidence.
- **Falsifier:** archive decodability must fall *and* LOAO Macro-F1 must rise.
- **H-link:** it also reports per-archive under-40 pAUC on its hold-outs.

### GeM — mechanism control for Clues (rev. 2)

- **Mechanism:** M2 plus GeM pooling, f = ((1/|R|) Σ x^p)^{1/p} with p initialised at 3 and
  learnable, on the pooled feature for **all** heads. That is *symmetric* max-like pooling.
- **Role:** Clues (M2 + DRE-5) claims its gain comes from the clinical **asymmetric any-region**
  rule (malignant if any region is, benign only if all are).
  - If `clues` beats `twostep` but **not** `gem`, the gain is generic max-like pooling, and the
    clinical-logic claim fails.
  - GeM enters the composite only if it beats `clues`.
- **Cost:** about 1.4 h (2 × 41 min); runs the same night as `clues`.
- **Registry:** B11 appears in the frozen plan as a standalone arm. Here it is a declared
  mechanism control, not a candidate.

### G — Group-DRO: **moved to V6** (rev. 2; audit AU6)

- The fold-0 training set (folds 1–4) holds only **65 under-40 escalating lesions (158 images)**.
  An age band × escalation group of that size is exactly the "tiny unstable group" that master
  plan §8 V5-A3 forbids.
- G moves to V6 as **subclass-DRO over visual subclasses (N6)**, gated on D4. See
  `docs/V6_RUNSHEET.md`.

---

## 5. What leaves the 9-October window, and why

| Frozen-plan item | Decision | Reason |
|---|---|---|
| S78 dual-stream (A2) | **V5 lean + V6 full** | V5: DRE-8 zoom on a **shared trunk** (Δparams ≈ 0, no masks), with the lesion-crop (8b) and random-crop (8r, the §23b compute-matched control) ablations. V6: separate-trunk dual-stream with a learned segmenter and a capacity-matched ConvNeXt-S control. The original needs masks on 54% of rows (A11), and V4's R3 crop was a no-op on 18.9% of lesions |
| B4 micro-patches | V6 (as wavelet space-to-depth) | HAM is only a 1.17× downsample at 384 px (A12). Archive-confounded |
| B5 context decomposition | V6, pending D2 | Needs masks. D2 tells us whether the background carries age at all |
| B8 Haar wavelets | **V6, reformulated** | A DWT of the same image adds nothing ConvNeXt's 4×4 stem cannot learn. The real mechanism is **wavelet space-to-depth** of a 2× image (768 → 12 channels at 384²), which recovers downsampling loss on BCN/MSKCC. It is gated on DRE-8's archive-stratified result |
| B7 SupCon | **V6, as multi-proxy contrastive** | Plain SupCon collapses a class into one cluster, which erases the young-melanoma subtypes DRE-6 keeps. It needs 2 views (≈ 2×) and large batches. Its minority-class benefit (ECL, MICCAI 2023) is covered in V5 by DRE-6 prototypes |
| B11 GeM | **V5, as a mechanism control** | See §4 GeM |
| S80 Group-DRO | V6 (subclass-DRO) | Audit AU6 |
| S81 three new backbones (about 16 h) | **V6, list revised** | DINOv3-ConvNeXt-T first; ConvNeXt-V2-T; **MaxViT-T** (the repo's best single model) replaces SwinV2-T; EfficientNetV2-S replaces -M. See `docs/V6_RUNSHEET.md` |
| S85 X-queue, N10 SSL, N12 routing, N13 | V6 | Unmeasured cost, or blocked by data |
| N5 young-data sourcing | V6 | MILK10k becomes V6 training data after S84 |
| S84 external read | **V5 on MILK10k** (owner decision 29 Sep) | Pre-registered in `docs/V5_RUNSHEET.md` §9 before download (audit AU7) |

---

## 6. Protocol changes (all amend Amendment 01 only where stated)

> **Revision 2026-09-30 (before the hash):** items 1–2 below (2 seeds, 95th-percentile gate,
> all-age pAUC only) and the M4 confirmed-benign definition are superseded by
> `docs/V5_RUNSHEET.md` §6 (3 seeds, k-seed-mean null, 80th percentile, all-age pAUC or
> pAUC_histo, noise-aware retention, trunk screen, `youngdata`) and by audit AU17–AU35 in
> `docs/v5_design/V5_PLAN_AUDIT.md` §1b.

1. **Screen at 224 px, 2 seeds** (42, 43) on fold 0, paired against the 224 px control's seeds.
   - The 224 control seeds are 42 (banked S72), 43 and 44 (S01, A01 A6), and 45 and 46 (new; run
     in the 30 Sep daytime window after the hash, audit AU11).
   - Two seeds at 224 px (2 × 41.0 min measured) cost about the same as one seed at 384 px
     (64–75 min) and halve the seed noise that A5 identified.
   - The control at the S01-chosen resolution is the comparator for the composite (audit AU3).
2. **Screen gate (fold 0, selection only):**
   - Δ all-age escalation pAUC@0.20 (McClish, 419 lesions, lesion-grouped) > the 95th percentile
     of the S01 224 px seed-vs-seed null (B3);
   - Macro-F1 Δ ≥ −0.010;
   - the arm's mechanism falsifier passes.

   Under-40 figures on fold 0 (16 lesions) are descriptive only (A1).
3. **Composite rule (fixed now):** the authoritative version is in `docs/V5_RUNSHEET.md` §6.
   It covers nested pairs, GeM, M7, DRE-7 and the null case.
4. **Confirmation:** composite vs the control at the **S01-chosen resolution** (384 px, or 224 px
   if S01 fails its two-hurdle rule; audit AU3), **folds 1–4 × seeds 42/43/44**, `_last`
   primary.
   It uses Amendment 01's gates A–D and the hierarchical seed × lesion CI, and it adds the B2
   case-mix standardisation and the B4 per-archive readout.
5. **Checkpoint and disk rule (hard constraint).** A ConvNeXt-Tiny checkpoint is **111 MB**
   (measured: `ml/checkpoints/convnext_tiny-v4_R0_kfold_f0_s42_*.pt`).
   - **Screening and LOAO runs (nights 2–4) save `_last.pt` only.** `train_v5` gets a
     `--save-best/--no-save-best` flag, **defaulting to off**. A3 makes `_last` primary, so a
     screen never needs `_best`.
   - **Only confirmatory runs (nights 5–8: folds 0–4, composite and control) save both** `_best`
     and `_last`, because `_best` is A3's pre-declared sensitivity analysis.
   - Night 1 (S01) runs on `train_v4`, which always writes both. That is 7 runs × 2 × 111 MB
     ≈ 1.6 GB, and it is accepted.
   - **Projected total:** S01 1.6 GB + screens/LOAO about 26 × 0.111 ≈ 2.9 GB + confirmation
     26 × 2 × 0.111 ≈ 5.8 GB, so **≈ 10.3 GB of the 31 GB free**. That leaves about 20 GB, above
     the 5 GB guard.
   - Disk is a commit-limit setting on this machine: the pagefile grows into free disk, and the
     2026-09-16 error-1455 failure was a full C: drive. `train_v4.py` refuses to start below 5 GB
     free, and `train_v5` must keep that guard.
6. **Dataloader workers (hard constraint): `--num-workers 2` on every training and inference
   script,** including `train_v5`, LOAO and TTA inference.
   - 4 workers die at 384 px with Windows error 1455 (ERROR_COMMITMENT_LIMIT), because each worker
     is a full spawned torch import against about 7.5 GB of commit headroom.
   - At 384 px, 2 and 3 workers measure the same (306 vs 307 ms/batch), so 2 costs nothing.
   - DRE-8's extra 448 px copy per sample raises per-worker memory. Do **not** raise workers to
     compensate; benchmark it at 2.
7. **Declared escalation score (audit AU4).** The control ranks by `escalation_mass`. An arm with
   an escalation head ranks by its head logit s_esc. Each arm's score is fixed in the arm registry
   before its run, and escalation mass is logged for every arm as a secondary. Otherwise a Δ could
   reflect the scoring rule rather than the representation.
8. **Same-night nesting (audit AU10).** A nested arm (structure/look, clues/twostep, gem/clues,
   m4/twostep, zoom/clues) runs on the same night as its parent. The nesting rule is applied at
   the morning read.
9. **Multiplicity (audit AU12).** About 10 screened arms. A screen is a compute-allocation filter,
   not a test (V4 S52 precedent). Confirmatory claims come only from folds 1–4 with the
   hierarchical seed × lesion CI.

---

## 7. Schedule: 9 nights plus a write-up day

Anchors:
- **224 px fold-0 run: 41.0 min, measured** (S72 log; decode-bound because of BCN's
  1024×1024 images).
- **384 px fold run: 64–75 min, measured range** (S70 benchmark; banked R1+R4 runs at batch 16 ×
  accumulation 2).
- **LOAO runs: extrapolated** by training-set size (8,313 / 8,590 / 13,685 images → about 28 / 29
  / 46 min at 224 px). Time the first one.
- **New heads (M2–M6):** expected to add little. M5's dense head is **unmeasured**, so run a
  `--smoke` epoch plus `scripts/gpu_benchmark.py` in the **unfrozen** stage before it enters a
  night.

**The night-by-night schedule now lives in `docs/V5_RUNSHEET.md` §8 (audit AU1).** It is
the only authoritative table. The earlier version here put the control and composite fold runs
on the same nights (8.5–10 h, beyond a night once zoom is in; audit AU2). The runsheet moves the
12 control fold runs into daytime windows and keeps nights 5–7 for the composite.

The per-night checklist, the Review-2 cut line and the pre-declared fallbacks are in the runsheet
(§8.2–8.4).

**Timeline:**
- Training nights **29 Sep – 7 Oct**, plus daytime windows for the control fold runs.
- **8 Oct:** no GPU.
- **9 Oct:** Review 2 shows whatever V5 has completed, plus the V6 plan. Unfinished V5 runs
  continue afterwards under the same hash.

---

## 8. CPU diagnostics (day of 30 Sep; none reads test, reserved or external data)

| ID | Question | Output | Gates |
|---|---|---|---|
| B1 | Which of the 81 under-40 escalating lesions does the control miss, and what are they (class, archive, pigment tercile, mask size, site)? | `results/v5/diagnostics/b1_hard_core.json` | Every arm's declared subset |
| D2 | Does the perilesional ring alone predict age band or escalation? (HAM masks) | `d2_context_probe.json` | B5 (deferred) |
| D4 | Is there a low-pAUC, under-40-enriched melanoma subclass? (per-class GMM on OOF embeddings, BIC) | `d4_subclasses.json` | G → N6 variant |
| **D5** (new) | What does ISIC record for polarisation (`dermoscopic_type`) and verification per archive? (ISIC API; availability unverified) | `d5_acquisition.json` | M4 benign pool; M7 targets |
| **D6** (new) | Does a handcrafted asymmetry and colour count (from the chromophore maps) separate <40 mel from histo-nv? | `d6_chaos_probe.json` | M6 / DRE-3/4 go or no-go |
| **M1-QC** (new) | Do c_mel / c_hb / c_depth put vasc/bcc mass in haemoglobin and blue-grey areas low on depth? (20 HAM images per class) | `m1_chromophore_qc.png` | M1 adoption |
| **Q4 / M5 pre-flight** (new; after owner-approved download) | Task 2 ID overlap by split; the 12 under-40 MSKCC mel (reported); pixel agreement | `m5_overlap.json` | M5 adoption (≥ 1,000 train rows, 0 val/reserved) |
| **Q1** (DRE) | Does chromophore geometry match the HAM expert masks? | `q1_geometry_qc.json` | DRE-2 used, else D₄ fallback |
| **Q5** (rev. 2) | Do the DSP structure primitives show the textbook class signatures and separate <40 mel from <40 histo-nv? (DRE §3 DRE-10) | `q5_dsp_probe.json` | `structure` arm go or no-go |
| **S75** (rev. 2; after owner-approved download) | MILK10k eligibility: 7-class mapping, deduplication against the 25,331 ISIC-2019 images by ISIC ID and perceptual hash, counts by class × age band. **No predictions** | `s75_milk10k_eligibility.json` | S84 power statement |

**Order (audit AU5):** these checks run **after** the hash of A02 + DRE + runsheet, because their
pass rules decide which arms run.

---

## 9. What 9 October can show, whatever happens

- **Positive composite:**
  - a biology-derived representation that beats a matched control on folds 1–4 × 3 seeds;
  - a mechanism check per component (M1 low-pigment rescue, M3 eccentric evidence maps, M5 learned
    structures);
  - V5 + S56 vs V1 + S56 at matched referral (B5).
- **Null composite:** still a result.
  - Each biological mechanism is tested with a declared falsifier.
  - The noise floor is measured on the real partition.
  - The under-40 gap is decomposed into archive, case mix, verification and morphology (D0, B1,
    D4, D6).
  - Evidence-map figures show *where* the model looks on young melanomas.
- **Either way:**
  - the skin-optics and class-biology atlas (§1–2), with the model design mapped to the
    dermatologist's algorithm (§3);
  - the pre-registration trail (freeze plus two hashed amendments);
  - the S69 modality gate and the S60 service from V4.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| New code breaks a night | `--smoke` before every block, and a parity smoke of `train_v5` (no arms) against the `train_v4` R0 loss curve |
| M5 overlap smaller than the ID range suggests | Verify before adoption. M5 is optional in the composite rule |
| Seed noise swamps the screen | 2 seeds per arm, 5-seed 224 px control, gate from the measured null |
| The composite's gains don't add up | Night-4 stacking check at 224 px before committing nights 5–7 |
| Commit-limit crash (error 1455) | `--num-workers 2` everywhere (§6.6); ≥ 5 GB free on C:; `_last`-only checkpoints on nights 2–4, both only on 5–8 (§6.5); projected use ≈ 10.3 GB of 31 GB |
| OD transform instability (glare, black borders) | sRGB linearisation, clamp [1/255, 254/255], glare excluded from basis fitting, bounded and melanin-gated depth channel (M1) |
| M5 masks misaligned with augmented images | Paired geometric transforms plus a unit test on a synthetic image (M5) |
| M4 ranking loss starved of pairs at batch 16 | Pairs per micro-batch, 0 when empty, mean \|P\| logged (M4) |
| Fold-0 selection bias | Fold 0 is excluded from every confirmatory statistic (A2) |
| Confirmation nights overflow (audit AU2) | Control fold runs in daytime windows; nights carry the composite only; fallback to 2 seeds |
| 384 px not better on pooled data (audit AU3) | S01 resolution branch: composite and confirmation at 224 px |
| Scoring-rule confound (audit AU4) | Declared score per arm; escalation mass logged for all |
| Zoom or DSP cost above estimate | Benchmark and smoke first; written reschedule in runsheet §8.3 |

---

## 11. Literature to verify before the paper (cited from memory)

- Kittler H, Rosendahl C, Cameron A, Tschandl P — *Dermatoscopy: pattern analysis* / "chaos and
  clues" (≈2011–2016).
- Argenziano G et al., *J Am Acad Dermatol* 2003 — dermoscopy consensus, two-step algorithm.
- Zalaudek I et al., *Arch Dermatol* ≈2006 — nevus dermoscopic patterns by age and body site.
- Pan Y et al. / Benvenuto-Andrade C et al., *Arch Dermatol* 2007–2008 — polarised vs
  non-polarised dermoscopy differences (shiny white structures, milia-like cysts).
- Lallas A et al. — dermoscopy of BCC subtypes (superficial BCC).
- Zalaudek I et al. ≈2012 — AK progression model (strawberry pattern).
- Plus every item in `V5_IDEAS_BIOLOGY_FIRST.md` §6 (Tsumura 2003, Pampena 2017, Cordoro 2013,
  PubMed 40805292, Bissoto 2019, Sohoni 2020, Ilse 2018).
- Codella N et al., ISIC 2018 challenge — [arXiv:1902.03368](https://arxiv.org/pdf/1902.03368)
  (verified 2026-09-29: Task 2 = 2,594 training images, 5 attributes).
