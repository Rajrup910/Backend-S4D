# V5 Dermatologist Reasoning Engine (DRE) — companion to Amendment 02

**Date:** 2026-09-29
**Status:** **ADOPTED 2026-09-30** (owner accepted every module), hashed **together with**
`docs/v5_design/V5_PLAN_AMENDMENT_02.md` and `docs/V5_RUNSHEET.md` as `amendments[1]`. Nothing here
ran before that hash. Where this file and the runsheet differ, the runsheet wins.
**Revision 2 (2026-09-29, pre-adoption):**
- adds **DRE-10 Dermoscopic Structure Primitives**, border abruptness (DRE-4), and the DRE-8b/8r
  ablations;
- DSP concepts are added to the DRE-7 rules;
- §6 now points to `docs/V5_RUNSHEET.md`. The audit is in `docs/v5_design/V5_PLAN_AUDIT.md`.

**Relation to Amendment 02:**
- It keeps M1 (chromophores), M2 (two-step head), M3 (noisy-OR clues), M4 (clinical-differential
  training), M5 (expert structure masks) and M7 (acquisition randomisation).
- It **replaces M6** with the mask-free geometry block DRE-2/3/4.
- It adds four new mechanisms: the **colour palette** (DRE-1), **periphery analysis** (DRE-4),
  **subtype memory** (DRE-6) and the **second look** (DRE-8), plus the **clinical-logic residual
  head** (DRE-7).

Every mechanism is written as: **how the expert thinks → what that computes → the module →
biology it encodes → falsifier → cost → registry check.** Costs are anchored to measured runs
(224 px fold 0 = 41.0 min; 384 px fold = 64–75 min). Anything else is marked **unmeasured**.

---

## 1. How an expert actually reads a dermoscopic image

Seen from the inside, an experienced dermoscopist's read is a sequence of distinct operations,
not one judgement:

1. **Gestalt (≈1 s).** "This is a nevus / SK / BCC." Holistic pattern recognition. *The CNN
   already does this, and it is where it fails on young melanoma: the gestalt of a young
   nevus-associated melanoma is "nevus".*
2. **Tissue reading.** Colours are read as tissue: brown means junctional melanin, blue-grey means
   dermal melanin, red means vessels, white means fibrosis or regression, yellow means keratin.
   The expert **counts colours**; three or more is suspicious.
3. **Orientation.** Find the lesion, its axes and its edge. The periphery is where growth happens.
4. **Step 1: melanocytic or not?** A network, aggregated globules, streaks, homogeneous blue or a
   parallel pattern means melanocytic. Otherwise, which keratinocytic or vascular entity is it?
5. **Chaos.** Is the arrangement of *structure or colour* asymmetric? Asymmetry of shape does not
   count. Is it one pattern or many (multicomponent)?
6. **Clues.** Is there *any* focal malignant clue? Examples: an eccentric structureless zone,
   grey/blue structures, peripheral black dots, segmental radial lines or pseudopods, white lines,
   polymorphous vessels. **One clue is enough; clues are not averaged.**
7. **Periphery, the young-patient question.** Peripheral globules or streaks are *normal* in a
   growing nevus when they form a **complete, symmetric rim**. They are suspicious when they are
   **segmental**, meaning present in only part of the circumference.
8. **Second look.** "It looks benign, but that corner…" The expert **zooms in on the one worrying
   spot**. This is the "little red riding hood" lesion: benign at first glance, suspicious on
   closer look.
9. **Differential recall.** "This reminds me of the nevus-associated melanoma I saw last year."
   Case-based memory of **subtypes**, not one average picture per diagnosis.
10. **Rule check before signing off.** Menzies: a lesion is not melanoma if it has **symmetry of
    pattern AND a single colour**. Otherwise one positive feature is enough to escalate. This is
    explicit Boolean logic.
11. **Context.** Age, site, the patient's other nevi ("ugly duckling"), and change over time.

Steps 1, 2, 3, 5 and 6 are partly inside any CNN. **Steps 3, 7, 8, 9 and 10 are not.**
- Global average pooling cannot express "only in part of the circumference" (step 7).
- A 224 px downsample cannot zoom (step 8).
- A single weight vector per class cannot hold several subtypes (step 9).
- A linear head cannot express "AND of two benign features, OR of nine malignant ones" (step 10).

This is the gap the engine fills.

---

## 2. The engine

```
           native image ─────────────────────────────────────────────┐
                │ resize                                              │ (DRE-8 crop)
                ▼                                                     ▼
RGB ─► DRE-0 chromophore unmix ─► melanin m, haemoglobin h ─► DRE-2 geometry (mask-free): lesion soft-mask s,
 │                               │                              centroid μ, axes R, radius r
 │                               └► DRE-1 colour palette: 6 clinical colours per pixel
 ▼
ConvNeXt-Tiny trunk ─► F3 (stride 16) ─┬─► DRE-3 chaos: axis-aligned asymmetry (structure + colour), pattern count
                                       ├─► DRE-4 periphery: polar warp → segmental index, rim coverage
                                       ├─► DRE-5 clues: patch evidence map e → noisy-OR, eccentricity
                                       │         └─► DRE-8 second look at argmax(e) on native pixels → noisy-OR
                     F4 ─► pooled f ───┼─► DRE-6 subtype memory: K prototypes per class, max over subtypes
                                       └─► 7-class head (CE), M2 melanocytic + escalation heads
                         concepts ─► DRE-7 clinical-logic residual (Menzies / chaos-and-clues rules) ─► + escalation logit
```

- **Trunk:** ConvNeXt-Tiny as in V4.
- **Feature maps for spatial reasoning:** stage-3 features F3 (stride 16: 14×14 at 224 px, 24×24
  at 384 px). The final 7×7 map is too coarse for angular analysis.
- **Size:** every head is small. The engine adds <2% of parameters.
- **Compute:** only DRE-8 adds real compute (a second forward pass).

---

## 3. Modules

### DRE-0 · Chromophore unmixing (= M1)

As in Amendment 02 M1 (rev. 2026-09-29):
- sRGB linearisation, then an OD clamp to [1/255, 254/255] and glare exclusion;
- a fixed per-fold optical-density ICA basis, fitted on non-glare training pixels only;
- melanin, haemoglobin and a bounded, melanin-gated **dermal-depth** channel (lower means deeper),
  fed with RGB through a widened, zero-initialised **6-channel** stem.

c_depth also sharpens DRE-1's blue-grey colour, which is the depth cue clinicians read.

It is also the **sensor** for DRE-1 and DRE-2, which reuse m and h without needing a
segmentation mask.

### DRE-1 · Clinical colour palette (new) — step 2

- **Expert:** "light brown, dark brown, black, blue-grey, red, white. How many, and where?"
  Three or more colours is a melanoma criterion (ABCD, 7-point, Menzies). An *eccentric* black or
  blue-grey area is a clue.
- **Module:**
  - Fit K = 6 colour prototypes c_k in OD space **once per fold** by k-means on lesion-region
    pixels of that fold's training images. **Freeze them.**
  - Name each by its chromophore coordinates: high m = dark brown/black; m with a raised blue
    fraction = blue-grey; high h = red; low m and low h with high reflectance = white.
  - Per pixel, a soft assignment a_k(p) = softmax_k(−‖x_p − c_k‖² / τ), with τ fixed.
- **Tokens (into the head and DRE-7):**
  - area fraction of each colour inside the lesion soft-mask s;
  - a **soft colour count** Σ_k σ((area_k − 0.05)/0.01);
  - the **eccentricity of each colour**: ‖centroid_k − μ‖ / r.
- **Biology:** colour = chromophore × depth (Amendment 02 §1). The count and position of colours
  are exactly what clinicians score.
- **Falsifier:** colour count and blue-grey or black eccentricity must differ between <40 mel and
  <40 histo-confirmed nv on CPU (Q2, §5) before any GPU run. On the GPU, zeroing the palette tokens
  at inference must remove the gain.
- **Cost:** negligible (a fixed 6-prototype distance on a 28×28 grid).
- **Registry:** new. Not colour normalisation, not metadata.

### DRE-2 · Orientation without masks (new) — step 3; removes the A11 blocker

- **Expert:** find the lesion, its long axis and its edge.
- **Module (no learning, no gradient):**
  - Lesion density ℓ = max(0, m − m_skin) + β·max(0, h − h_skin). The skin baseline is the median
    of the outer 10% border ring.
  - Soft mask s = σ((ℓ − t)/w), with t set per image by Otsu on ℓ.
  - Take image moments of s: centroid μ, covariance → principal axes R (angle θ), radius
    r = 2√λ₁.
  - **Fallback:** if the mask covers less than 3% or more than 95% of the image (for example an
    amelanotic lesion with weak h), use the image centre and axis-aligned axes, and set a flag
    token.
- **Why this matters:** A11 blocked every mask-dependent arm (dual-stream, B5, B9, exact N7),
  because only HAM (45.6% of rows) has masks. Chromophore geometry gives **approximate lesion
  geometry on every row, from physics**, with no segmenter to train.
- **QC gate (CPU, before adoption):** compare s with the HAM expert masks on HAM train rows.
  - Pre-declared pass: median Dice ≥ 0.75 and centroid error ≤ 10% of the image diagonal.
  - If it fails, DRE-3/4 use image-centred geometry and say so.
- **Cost:** CPU/GPU trivial. Computed on the fly at 224 px on the GPU (no disk cache).

### DRE-3 · Chaos: axis-aligned asymmetry and pattern count (replaces M6) — step 5

- **Expert:** "Is structure or colour asymmetric about either axis? One pattern or several?"
- **Module:**
  - Resample F3 and the palette maps a_k into the lesion frame (rotate by −θ about μ with
    `grid_sample`; cheap).
  - Mirror about each principal axis.
  - **Structure asymmetry:** A_s(axis) = Σ_p s(p)·‖F3(p) − F3(mirror(p))‖ / Σ s.
  - **Colour asymmetry A_c(axis):** the same computation on a_k.
  - **Tokens:** the min and max over the two axes, for both. Menzies scores symmetry "on all
    axes"; Kittler's chaos is asymmetry on any.
  - **Pattern count:** soft-assign each lesion patch of F3 to P = 8 learned pattern prototypes,
    with an orthogonality regulariser. Report the entropy of the lesion's pattern histogram
    (multicomponent means high entropy).
- **Fallback:** if Q1 fails, Amendment 02 M6's D₄ reflections are used (min and max over 4 axes,
  normalised). The chromophore asymmetry token A_chrom is computed on whichever axes are active.
- **Why the axes matter:** the original N7 flipped about the *image* axes. A lesion rotated by 30°
  then looks asymmetric when it is not. DRE-2's axes make the comparison clinically correct.
- **Falsifier:** on CPU, a handcrafted colour asymmetry (from the palette on HAM rows) must separate
  <40 mel from <40 histo-nv (Q2). On the GPU, the gain must concentrate on under-40 mel vs
  histo-nv pairs whose control score gap is small.
- **Cost:** about 1× (one extra `grid_sample` on a 14×14×384 map).

### DRE-4 · Periphery: segmental vs circumferential (new) — step 7, the young-patient discriminator

- **Expert:** in a young patient, peripheral globules or streaks are **benign when they form a
  complete symmetric rim** (a growing nevus; Spitz/Reed starburst). They are **suspicious when
  segmental**. This is the single most age-specific rule in dermoscopy, and it is exactly the
  "same structure, different arrangement" failure in Amendment 02 §2.2.
- **Module:**
  - Polar-warp F3 about μ over a **peripheral annulus**, ρ ∈ [0.6r, 1.2r], with 16 angular sectors
    × 3 radial bins (`grid_sample`).
  - A learned 1×1 conv produces a *peripheral structure* channel c(φ), pooled over ρ.
  - **Segmental index:** S = 1 − H(c/Σc) / log 16. This is angular concentration: 0 means
    uniformly around the rim, 1 means one sector.
  - **Rim coverage:** the fraction of sectors with c(φ) > mean(c).
  - **Radial gradient:** outer-bin activation minus inner-bin activation (structures at the edge,
    i.e. growth).
  - **Border abruptness (rev. 2):**
    - Along each of the 16 sectors, take the radial derivative of c_mel across DRE-2's boundary
      (ρ ∈ [0.85r, 1.15r]), normalised by the lesion's median melanin.
    - Tokens: the **fraction of abrupt sectors** (derivative above the fold's 75th percentile of
      training values) and the **angular variance** of abruptness.
    - Biology: an abrupt pigment cut-off (the "B" of ABCD, "sharp demarcation" in Menzies) marks
      melanoma. Nevi fade gradually at their edge. Abruptness confined to *some* sectors is the
      segmental form. It becomes the DRE-7 concept **BORDER_ABRUPT**.
- **Biology:**
  - Growing nevus: high coverage, low S.
  - Reed/Spitz starburst: high coverage, low S.
  - Radial-growth melanoma (pseudopods / streaks in one segment): low coverage, high S.
  - Nevus-associated melanoma: an eccentric focus that is off-centre and segmental.
- **Falsifiers:**
  - **CPU (Q2):** a handcrafted S computed from the melanin map's local structure energy in the same
    annulus must separate <40 mel from <40 histo-nv on HAM rows (AUC lower CI bound > 0.5).
  - **GPU:** the rescued under-40 lesions must have a higher S than the missed ones, and randomly
    rotating c(φ) *per sector* (which destroys the arrangement but keeps the amount) at inference
    must remove the gain.
- **Registry:** B9 ("boundary/radial representation") encoded the radial *distribution*. DRE-4 is
  a specific angular-*concentration* statistic tied to one clinical rule. Declare it as the B9
  implementation, not as an extra arm.
- **Cost:** about 1×.

### DRE-5 · Clues: focal evidence plus eccentricity (= M3, extended) — step 6

- M3's patch evidence map e (1×1 conv on F3) is aggregated by normalised log-sum-exp,
  s = (1/r)·ln((1/K)Σ exp(r·eᵢ)) with r = 4 and K = H′W′: "malignant if any region is".
- The map is saved at inference (Amendment 02 M3).
- **New eccentricity token:** ‖μ_e − μ‖ / r, where μ_e is the softmax(e)-weighted centroid.
- **Clinically:** an *eccentric* structureless area or hyperpigmentation is a clue, while a
  *central* blotch in a nevus is benign.
- **Falsifier:** as M3 (rescued lesions show eccentric evidence), plus the eccentricity token's
  inference ablation.

### DRE-6 · Subtype memory: multi-prototype class head (new) — step 9

- **Expert:** "melanoma" is not one picture. The clinician holds *several*: superficial spreading,
  nodular, lentigo maligna, amelanotic, nevus-associated, spitzoid. The same holds for nevi
  (reticular, globular, Spitz/Reed, blue, congenital, dysplastic).
- **The failure it targets:** a linear head has **one weight vector per class**. With 1,500 of the
  2,525 mel images from 60+, that vector points at *old melanoma*, and the 133 young melanomas are
  a minority mode that gets averaged away. Amendment 02 §2.1 calls this the prototype problem.
- **Module:**
  - Each class c holds K_c learned prototypes w_{c,k}: mel 8, nv 8, bcc 4, bkl 4, akiec/df/vasc 2.
  - logit_c = τ·logsumexp_k(⟨f, w_{c,k}⟩ / τ), a **soft max over subtypes**.
  - An orthogonality penalty keeps a class's prototypes distinct.
  - This replaces the linear 7-class head. The trunk is trained end-to-end.
- **Why it can help under 40:** a minority subtype gets its own prototype instead of being pulled
  toward the majority. This is GEORGE's idea (N6) without clustering and without age, so it also
  works on the 251 age-unknown rows and on external cohorts.
- **Falsifiers:**
  - Under-40 mel OOF embeddings must concentrate on ≤ 2 mel prototypes that are **not** the
    60+-dominant one (χ² on the prototype × age-band table).
  - Deleting those prototypes at inference must drop under-40 mel sensitivity more than deleting
    a random mel prototype.
- **Interpretability for free:** each prototype's nearest training images form a "subtype atlas"
  that can be shown in a review.
- **Cost:** negligible.
- **Registry:** not a frozen head (the trunk is trained). Not an age mechanism.

### DRE-7 · Clinical-logic residual head (new, neuro-symbolic) — step 10

- **Expert:** Menzies, 7-point and chaos-and-clues are *rules*: AND of benign features, OR of
  malignant ones. A linear head cannot compute "benign only if symmetric AND one colour".
- **Concepts** c ∈ [0,1], each a sigmoid of a small linear map from its tokens:
  - **SYM** (DRE-3), **ONE_COLOUR** and **BLUE_GREY** (DRE-1);
  - **CLUE** (DRE-5 noisy-OR), **SEGMENTAL** (DRE-4), **MULTICOMPONENT** (DRE-3 entropy);
  - when M5 is trained, **NETWORK, GLOBULES, STREAKS, NEG_NETWORK, MILIA**. These are the
    **supervised concepts**, from expert Task 2 masks, image-level presence.
  - when DRE-10 passes Q5 (rev. 2): **NETWORK_ATYPICAL, GLOBULES_IRREGULAR, VEIL,
    VESSEL_POLYMORPHOUS** and **BORDER_ABRUPT** (DRE-4). These are fixed-filter structure
    concepts, available on every image.
- **Rules** (fixed structure, product t-norm, soft-OR = 1 − Π(1 − x)):

  | Rule | Formula | Clinical source |
  |---|---|---|
  | R_benign | SYM · ONE_COLOUR · (1 − CLUE) · (1 − NETWORK_ATYPICAL) | Menzies negative features (+ typical network) |
  | R_chaos_clue | (1 − SYM) · OR(CLUE, SEGMENTAL, BLUE_GREY, STREAKS, NEG_NETWORK, NETWORK_ATYPICAL, GLOBULES_IRREGULAR, VEIL, VESSEL_POLYMORPHOUS, BORDER_ABRUPT) | Kittler chaos and clues; 7-point major criteria (atypical network, blue-white veil, atypical vessels) |

  Concepts whose module is not in the composite are held at 0 inside the OR and at 1 inside a
  (1 − x) term, so a missing module never changes the rule's structure.
  | R_sk | MILIA · (1 − NETWORK) | Keratinocytic certificate |
  | R_nonmelanocytic | (1 − NETWORK) · (1 − GLOBULES) | Two-step algorithm, step 1 |

- **Output:** the escalation logit becomes the trunk escalation logit + Σ_j α_j·logit(R_j). The α_j
  are initialised at **0**, so the head starts as the plain model and only earns weight if the
  rules help. It is a **residual** because concept bottlenecks alone usually cost accuracy.
- **Falsifiers:**
  - **Sign check, pre-declared:** α(R_chaos_clue) > 0, α(R_benign) < 0 and α(R_sk) < 0 must be
    learned in all seeds. If the network learns the rules backwards, the head is fitting noise.
  - Removing the logic head at inference must remove the gain.
- **Bonus:** every prediction decomposes into doctor-readable terms, for example: "chaos: yes
  (colour asymmetry 0.71); clue: segmental radial structures at 2–4 o'clock; benign certificate:
  fails → escalate".
- **Cost:** negligible.

### DRE-8 · Second look: evidence-driven zoom on native pixels (new) — step 8

- **Expert:** zooms in on the one worrying spot.
- **Why dual-stream was the wrong way to zoom:** the frozen plan's dual-stream crops the *lesion*,
  which needs masks and often duplicates the global view (A01 B10). The expert zooms on the
  *suspicious region*.
- **Module:**
  - The dataloader also returns a 448 px short-side copy.
  - Take p* = argmax of e (DRE-5) on the 224 view, with stop-grad. Crop a window of side
    max(0.35·2r, 25% of the image) around p* from the 448 copy, and resize it to 224.
  - Pass it through the **shared trunk** to get a zoom escalation logit.
  - P_esc = 1 − (1 − P_global)(1 − P_zoom), i.e. noisy-OR. Both logits receive the escalation
    loss.
- **Biology:** focal clues (pseudopods, peripheral black dots, polymorphous vessels) are smaller
  than a 224 px downsample of a 1024 px BCN image can resolve (a 2.67× downsample, A12).
- **Falsifiers:**
  - The gain must be larger on BCN/MSKCC than on HAM, since the HAM downsample is only 1.17×
    (archive-stratified readout, A12).
  - The gain must shrink when p* is replaced by a random location of the same size.
- **Cost (unmeasured):** a second forward and backward pass, so roughly 1.5–2× per step. Run
  `scripts/gpu_benchmark.py` in the **unfrozen** stage first.
  - At 384 px, run it at batch 16 × accumulation 2. The measured 6.27 GB at batch 32 plus a second
    224 px pass (2.41 GB) would exceed the 8.55 GB card.
- **Ablations (rev. 2; the lean form of the frozen plan's dual-stream).** Only if `zoom` passes,
  1 seed each, descriptive:
  - **DRE-8b `zoom_lesion`:** the second view is the **lesion-centred** crop (DRE-2 box, 1.2× the
    lesion's extent). This is the original dual-stream idea, without masks.
  - **DRE-8r `zoom_random`:** a random crop of the same size. It is the **compute-matched control**
    that §23b requires: same trunk, same FLOPs, no evidence.
  - Reading: evidence > lesion > random means *where* the second look goes matters. Evidence ≈
    random means the gain is extra compute. The separate-trunk dual-stream with a
    capacity-matched ConvNeXt-S control is V6.

### DRE-10 · Dermoscopic Structure Primitives (DSP) (new, rev. 2) — structure identification

- **Expert:** before any gestalt, the dermoscopist *names structures*. Is the network regular? How
  are the dots and globules arranged? What shape are the vessels? Is there a veil?
- **Why fixed filters:**
  - M5 supervises five structures, but only on MSKCC rows.
  - DSP computes clinically defined structure maps on **every** image, from physics-based filters
    on the DRE-0 chromophore channels.
  - The filters are fixed, not learned, so the biology defines them. Pigment-network detection
    with directional/Gabor filters and vessel detection are established in the dermoscopy
    literature (Barata et al. 2012; Jaworek-Korjakowska et al. 2018).

| Primitive | Channel | Fixed filter | Biology it encodes | Tokens → concept |
|---|---|---|---|---|
| Vessel morphology | c_hb | Multi-scale Hessian, σ ∈ {1, 2, 4} px at 224 px (scaled with resolution); Frangi-type **tubularity** and **blobness** maps | Arborising (thick branching tubular) → BCC; dotted/glomerular (blobs) → Spitz, melanoma, Bowen; **polymorphous** → melanoma; hairpin → SK; comma → dermal nevus | Coverage; tubular:blob ratio; calibre entropy over σ; polymorphism = entropy of {thin-tubular, thick-tubular, blob} → **VESSEL_POLYMORPHOUS** |
| Network regularity | c_mel | Gabor bank, 4 orientations × 2 wavelengths; local energy, dominant orientation and wavelength | Regular rete-ridge mesh → nevus; thickened, irregular or broken network → melanoma | Coverage; regularity = 1 − CV(dominant wavelength) over network pixels; orientation coherence; periphery-minus-centre line darkness → **NETWORK_ATYPICAL** |
| Dots and globules as a point pattern | c_mel | LoG (σ = 1.5, 3) with 3×3 max-pool non-maximum suppression; threshold = lesion mean + 2 SD | Regular or symmetric-rim distribution → (growing) nevus; clustered, eccentric or irregular → melanoma; blue-grey dots (peppering) → regression | Density; **Clark–Evans ratio R** (>1 regular, <1 clustered); fraction peripheral (0.6–1.2 r); sectors occupied /16; size CV; mean c_depth of the dots → **GLOBULES_IRREGULAR** |
| Veil / structureless | c_depth, c_mel, Gabor energy | Low depth ratio ∧ high melanin ∧ low texture energy | Blue-white veil and eccentric structureless areas mean dermal invasion (melanoma) | Area fraction; eccentricity → **VEIL** |

- **Integration:**
  - The 5 maps (tubularity, blobness, network energy, blob response, veil) are standardised per
    fold and added as **zero-initialised** stem channels (6 → 11), so epoch 0 equals the `look`
    arm.
  - About 18 scalar tokens go to the head and to the DRE-7 concepts.
  - Everything is computed on the GPU with fixed kernels; there is no disk cache.
- **CPU pre-check Q5** (skimage/scipy, development rows, lesion-grouped CIs):
  - (i) **Five textbook signature checks**, each in the expected direction with a CI excluding 0:
    - tubular:blob ratio is highest in bcc;
    - network coverage is higher in {mel, nv} than in non-melanocytic lesions;
    - network regularity is higher in nv than in mel;
    - Clark–Evans R is higher in nv than in mel;
    - veil fraction is higher in mel than in nv.
  - (ii) **Young differential:** at least 2 tokens with <40 mel vs <40 histo-nv AUC lower CI
    bound > 0.5.
  - **Pass needs ≥ 3 of (i) and (ii).** Otherwise the `structure` arm is dropped without a GPU run,
    and the failed signatures are reported as a finding about what these filters can see in our
    images.
- **Screen:** the arm `structure` = `look` + DSP, nested on `look` and run the same night.
- **Falsifiers:**
  - Token ablation at inference removes the gain.
  - On rescued under-40 mel, the pre-declared concepts (GLOBULES_IRREGULAR, NETWORK_ATYPICAL,
    VEIL) are higher than on missed ones.
- **Cost:** a few fixed convolutions per batch. Expected ≤ 10%; **timed in the smoke epoch**
  (unmeasured).
- **Registry:** image-derived structure maps, not metadata (barred item 2), and not colour
  normalisation (item 11). New.

### DRE-9 · Context: deferred, and why (step 11)

- **Ugly duckling** (compare with the patient's other nevi) needs patient IDs, which
  `manifest_v4.csv` does not have. That makes it a V6 headline (N13).
- **Age as a *rule selector*** ("in a young patient, discount a symmetric rim"): DRE-4 already
  encodes that rule *without* reading age. Feeding age into the head would fall under barred
  item 2 (tabular fusion) or the FiLM rejection. **Not proposed.**
- **Change over time:** HAM follow-up images are all nevi and cannot teach change. Not available.

---

## 4. Registry check against the 12 barred mechanisms

| Barred item | Engine exposure | Verdict |
|---|---|---|
| 1 Frozen specialist head | All heads trained end-to-end with the trunk | Clear |
| 2 Tabular metadata fusion | No age/sex/site input anywhere | Clear |
| 3 Adversarial age invariance | None | Clear |
| 4–5 Frozen foundation probes | None | Clear |
| 6 Global prior correction | None | Clear |
| 7–8 λ(age), per-band offsets | None; decision layer stays S56 | Clear |
| 9 Mahalanobis admissibility | None | Clear |
| 10 Ridge stacking | None | Clear |
| 11 Buggy colour constancy | Chromophores *add* channels; palette is fixed per fold; nothing normalises | Clear |
| 12 Reserved reads | Development rows only | Clear |

---

## 5. CPU pre-checks (day of 30 Sep; development rows only)

| ID | Check | Pass rule (pre-declared) | Controls |
|---|---|---|---|
| **Q1** | DRE-2 chromophore mask vs HAM expert masks | Median Dice ≥ 0.75, centroid error ≤ 10% of diagonal | DRE-2 used vs image-centre fallback |
| **Q2** | Handcrafted colour count, colour asymmetry and segmental index: <40 mel vs <40 histo-nv (HAM rows with masks; lesion-grouped AUC) | Each feature's AUC lower CI bound > 0.5 | Which of DRE-1/3/4 get GPU time |
| **Q3** | D4 subclass check on S72 OOF embeddings | Descriptive; informs how DRE-6 is read, not its K | — |
| **Q4** | Task 2 ID overlap and licence (M5), after owner-approved download | ≥ 1,000 overlapping **train** rows | Supervised concepts in DRE-7 |
| **Q5** (rev. 2) | DSP textbook signatures and the young differential (DRE-10) | ≥ 3 of 5 signatures **and** ≥ 2 tokens with <40 AUC lower bound > 0.5 | `structure` arm; DSP concepts in DRE-7 |

Q2 matters most. If the young-melanoma vs young-nevus signal is not in the pixels by these
clinical measures, the DRE-3/4 hypothesis fails **for the cost of a CPU afternoon**, and that
failure is itself a finding worth reporting.

---

## 6. How the engine fits the Amendment 02 nights

Screening follows Amendment 02 §6: 224 px, fold 0, seeds 42/43, paired against the 5-seed 224 px
control. The pre-registered gate is on all-age escalation pAUC plus Macro-F1 retention, with each
module's falsifier required.

**The arm inventory, the composite rule and the night-by-night schedule are in
`docs/V5_RUNSHEET.md` §6–8, the only authoritative version (audit AU1).** The engine
maps onto these runsheet arms:

| Engine block | Runsheet arm | Comparator |
|---|---|---|
| DRE-0 + DRE-1 | `look` | control |
| + DRE-10 | `structure` | look |
| DRE-2/3/4 (+ border abruptness) | `geometry` | control |
| M2 | `twostep` | control |
| M2 + DRE-5 | `clues` | twostep (and `gem` as the mechanism control) |
| DRE-6 | `memory` | control |
| DRE-5 + DRE-8 | `zoom` (+ `zoom_lesion`, `zoom_random`) | clues |
| DRE-7 | `composite` vs `composite_nologic` | each other |

Group-DRO has moved to V6 (audit AU6).

---

## 7. What this gives the review on 9 October

1. **A model that reasons in the clinician's vocabulary.** For any lesion it can output:
   - the chromophore maps;
   - the colour palette and colour count;
   - the lesion axes and an asymmetry score;
   - a polar plot of the periphery with its segmental index;
   - the evidence map and its eccentricity;
   - the zoomed region;
   - the nearest subtype prototype;
   - which clinical rule fired.

   This is a strong live demo whatever the metrics say. It can also be exposed through the S60
   FastAPI service as an `explain` endpoint (one extra route, after 9 Oct if time is short).
2. **A falsifiable biological account of the under-40 gap.** Q2 tests whether arrangement cues
   separate young melanoma from young nevus. DRE-4/6 test whether giving the network those cues
   and subtype capacity changes ranking. Each has a pre-declared falsifier and a sign check.
3. **The confirmatory numbers** from folds 1–4 × 3 seeds against a matched 384 px control
   (Amendment 01 gates, hierarchical CI, per-archive and case-mix readouts).

---

## 8. Honest expectations

- **Most likely to move the numbers:**
  - **DRE-6 (subtype memory)**, because the prototype problem is measured (70% of mel is 60+);
  - **DRE-8 (zoom)** on BCN/MSKCC, which is where 58% of young escalating lesions are;
  - **M4 (clinical differential)**.
- **Most likely to be explanatory rather than a large metric gain:** DRE-1/3/7. Rule and concept
  heads rarely beat a strong black box on AUC. Their value is that the model's reasoning becomes
  *inspectable and checkable against dermatology*.
- **DRE-4 is the highest-risk, highest-meaning module.** If Q2 passes and DRE-4 rescues young
  melanomas with high segmental index, that is a clinically specific, publishable mechanism.
- **Power:** 65 under-40 escalating lesions in folds 1–4. Only a large under-40 gain can be
  *certified* by 9 October. Smaller ones will be reported as descriptive, with mechanism evidence.

---

## 9. Literature to verify (cited from memory)

- Menzies SW et al., *Arch Dermatol* 1996 — the Menzies method (negative and positive features).
- Argenziano G et al., *Arch Dermatol* 1998 — the 7-point checklist; *J Am Acad Dermatol* 2003 —
  consensus two-step algorithm.
- Kittler H, Rosendahl C et al. — "chaos and clues" / pattern analysis (≈2011–2016).
- Mascaro JM Jr, Mascaro JM, *Arch Dermatol* 1998 — "little red riding hood" sign.
- Grob JJ, Bonerandi JJ, *Arch Dermatol* 1998 — ugly duckling sign.
- Zalaudek I et al. — age- and site-dependent nevus patterns; symmetric peripheral globules in
  growing nevi.
- Kawahara J et al., *IEEE JBHI* 2019 — Derm7pt, multitask 7-point learning.
- Chen C et al., *NeurIPS* 2019 — ProtoPNet, "This looks like that".
- Koh PW et al., *ICML* 2020 — concept bottleneck models.
- Fu J et al., *CVPR* 2017 — RA-CNN, recurrent attention zoom ("look closer").
- Sohoni N et al., *NeurIPS* 2020 — GEORGE, hidden subclasses.
- Tsumura N et al., *SIGGRAPH* 2003 — melanin/haemoglobin decomposition.
- Codella N et al., ISIC 2018 — [arXiv:1902.03368](https://arxiv.org/pdf/1902.03368).
