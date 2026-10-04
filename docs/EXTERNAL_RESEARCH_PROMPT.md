# Research brief for an external AI assistant: V6 planning for a skin-lesion triage model

**How to use:** give the assistant this file, and optionally the GitHub repository (`Rajrup910/Backend-S4D`).
The brief is self-contained: every number carries the repository file it comes from. **The public
snapshot may not contain `research/v5/`, `research/v6/` or most of `results/v5/` (not yet committed)**;
where a path is missing, rely on the numbers quoted here. Written 1 Oct 2026; **revised 2 Oct 2026**
with the V5 screen results (Q3, Q4), the V6 audit corrections, and the defects found on 2 Oct.

---

## 0. What I want from you, and how to answer

I am planning **V6** after five rounds of work (V1–V5). I want **evidence-based ideas and critiques**,
ranked by expected value on my hardware, with honest odds. The specific questions are in §9. For
every idea give:

1. **Mechanism** — why it should change *this* failure (not "it improves accuracy in general").
2. **Evidence** — papers or results from comparable settings (dermoscopy, long-tailed medical
   imaging, subgroup ranking). Say when evidence is thin.
3. **Near-repeat check** — which item in §6 it resembles, and why it is not the same thing.
4. **Cost on my hardware** (§7) — GPU hours at 224 px on ConvNeXt-T scale, VRAM, CPU and RAM.
5. **Falsifier** — the result that would show it did not work, stated before running it.
6. **Endpoint** — which of my metrics (§4) it should move, and by how much, realistically.

**Citation rules (important — I verify every reference before using it):**
- Give each source as authors, year, venue and a **DOI or arXiv ID**. If you cannot give an
  identifier, say "unverified" next to it.
- Mark each claim **[read]** if you have the source's text in front of you, or **[recalled]** if it is
  from memory. Never invent a paper, a number from a paper, or a dataset statistic.
- Quote effect sizes only with their source; say "no comparable evidence found" when that is the truth.

Do **not** propose anything barred in §5 (it invalidates the study), and do not re-propose anything in
§6 without a specific reason it would behave differently. "Train longer", "use a bigger model" or
"ensemble more" without a mechanism is not useful: §6 shows why.

---

## 1. The project in one paragraph

A skin-lesion **triage** system: 7-class dermoscopy classification (akiec, bcc, bkl, df, mel, nv, vasc)
on the pooled ISIC-2019 corpus (HAM10000 + BCN20000 + MSKCC; 15,294 development images), with
smartphone clinical photos (PAD-UFES-20) as a second modality. The clinical output is **escalation**:
refer mel, bcc and akiec. Deployed stack (V1): a 6-CNN soft-vote ensemble, 24-view TTA, a Dirichlet
calibrator fitted on out-of-fold predictions, and **S56 selective abstention**. V4 moved training
in-domain (pooled corpus, 5-fold lesion-grouped CV). **V5** screens dermatology-inspired modules on a
ConvNeXt-T (ImageNet-22k) trunk; **V6** is the next plan (§3). Targets: IEEE TMI / MedIA / MICCAI.
Python 3.12, PyTorch 2.11, Windows, one laptop GPU.

---

## 2. The hard walls (measured; every number has a source)

### Wall 1 — Under-40 melanoma ranking (the main one)
- V1, HAM test: under-40 escalation sensitivity **0.143 (3/21)** vs ~0.78 in older bands; only **11%** of
  misses are referred by abstention. Better-powered OOF estimate **0.547** (35/64).
  *(`research/selective/results/session4_report.md`, `research/stats/results_oof/age_gap_intervals.csv`)*
- It is a **ranking** limit, not a threshold one: under-40 AUC **0.878** vs **0.948 / 0.929** in older
  bands; a constant age-bonus rule needs **25.9%** under-40 referral for 80% sensitivity.
  *(`results/v4/s64/`, `docs/RESEARCH_LOG.md` S64)*
- What the misses are: of 33 young escalating lesions missed at the 20%-FPR edge, **32 are melanomas**,
  **51 of their 55 images are called nevus**, and **19 of 33** are in the highest melanin tercile.
  *(`results/v5/diagnostics/b1_hard_core.json`)*
- Data: **81** under-40 escalating lesions in the whole corpus; escalating prevalence **4.9%** under 40
  vs **35.5%** at 60+. Fold-0 validation has only **35** under-40 escalating rows, so under-40 screen
  results are descriptive. *(`results/age_band_prior.csv`; `results/v5/screens/q4_verify.json`)*
- **Every V5 module so far that raised all-age ranking left under-40 flat or worse** (§2a): IN-22k
  pretraining −0.0155; zoom vs twostep −0.024.

### Wall 2 — Cross-domain collapse
- HAM-trained V1 stack on held-out BCN/MSKCC: Macro-F1 **0.411** vs **0.784** in-domain.
  *(`results/v4/s54/`)*
- Smartphone photos (PAD-UFES-20, 2,106 images): members **0.124–0.188** Macro-F1; the soft vote
  (**0.167**) is worse than its best member, and the HAM-fitted calibrator makes it worse (**0.133**).
  *(`research/xdomain/results/`)*

### Wall 3 — The deployment contract cannot be met
S70 contract: under-40 sensitivity **≥ 0.855** at total referral **≤ 0.25**. On the reserved cohort **no
age band meets it**. Age-conditional thresholds and conformal layers **lose** to plain selective
abstention at matched workload (S65). *(`docs/RESEARCH_LOG.md` S63, S65)*

### Wall 4 — The signal is the size of the noise (worse on the stronger trunk)
- Seed-to-seed pair SD, fold 0, **ImageNet-1k** control (6 seeds): Macro-F1 **0.0161**, all-age pAUC
  **0.0071**, histology pAUC **0.0134**, under-40 pAUC **0.0191**. *(`results/v5/screens/noise_floor.json`)*
- On the **ImageNet-22k** trunk that V5 actually screens on (3 seeds): all-age pAUC pair SD **0.0153
  (2.15×)**, histology **0.0199 (1.48×)**, Macro-F1 **0.0218 (1.35×)**; the mechanism endpoint
  (melanoma vs biopsied nevus) is unchanged (**1.02×**). The frozen V5 screen bars (from in1k noise)
  are therefore too lenient for in22k arms: at in22k noise they would be **0.0074 / 0.0097** instead of
  0.0035 / 0.0065. *(`results/v6/noise_floor_in22k_check.json`, `results/v5/screens/q4_verify.json`)*
- 384 px (+0.030 on HAM only) gives **+0.0034 [−0.0163, +0.0245]** on pooled data.
  *(`results/v5/s01_decision.json`)*

### Wall 5 — A label/device shortcut inflates the easy metric
All-age pAUC ≈ 0.80–0.81, but escalating vs **histopathology-confirmed** benign ≈ 0.72–0.74. HAM's
non-biopsied follow-up nevi are easy negatives. *(`results/v5/screens/noise_floor.json`)*

### Wall 6 — Calibration errors cancel in the aggregate
The ensemble is **under**-confident (mean confidence 0.705 vs accuracy 0.861). After the global
Dirichlet map the per-age signed gap **flips sign** (under-40 −0.027, 60+ +0.041), hidden in an
aggregate ECE of 0.025. *(`research/stats/results_oof/band_calibration.csv`)*

### Wall 7 — Biology-derived modules give small, fragile or readout-only gains (V5 results)
See §2a for the numbers. Summary: the dermatology-inspired modules either fail, pass only through a
dedicated readout head, pass in one seed, or (look, geometry) could not be read because the training
was numerically broken (§2b). Filter-based dermoscopic structure maps showed **1 of 5** textbook
signatures and were not run; expert structure masks overlap only **416** training rows (needed 1,000).
*(`results/v5/diagnostics/{q5_dsp_probe,m5_overlap}.json`)*

### Wall 8 — External confirmation will be underpowered
MILK10k: **5,220** eligible lesions, only **71** under-40 escalating. *(`results/v5/s75_milk10k_eligibility.json`)*

---

## 2a. V5 screen results (fold 0, seeds 42/43/44, ImageNet-22k trunk, 224 px)

Gate: mean over seeds of (arm − comparator) on all-age pAUC@FPR0.2 **or** histology pAUC must exceed
the 80th percentile of the seed-noise null (frozen bars 0.0035 / 0.0065), Macro-F1 retained (bar
−0.0153), and a pre-declared falsifier must pass. Per-seed values in brackets.
*(Sources: `results/v5/screens/gate_*_in22k_*.json`, `falsifier_*_in22k.json`, `q3_verify_rescore.json`,
`q4_verify.json`)*

| Arm (comparator) | What it adds | Δ all-age pAUC | Δ histo pAUC | Δ Macro-F1 | Verdict |
|---|---|---|---|---|---|
| IN-22k trunk (in1k control) | Pretraining | +0.0072 | +0.0130 | +0.0275 | trunk chosen; under-40 **−0.0155** |
| twostep (control) | melanocytic + escalation heads | +0.0155 | +0.0213 | −0.0093 | PASS — but on plain escalation mass only **+0.0025**: a **readout** gain from its own head |
| m4 (twostep) | ranking mel vs biopsied benign | +0.0040 | +0.0132 | −0.0054 | PASS, fragile (seed 42 negative on mass) |
| memory (control) | multi-prototype head | +0.0030 | +0.0085 | −0.0114 | FAIL (falsifier 0/3 seeds) |
| clues (twostep) | focal log-sum-exp clue map | −0.0014 | +0.0045 | −0.0048 | FAIL |
| gem (clues) | GeM pooling control | −0.0012 | −0.0022 | +0.0013 | FAIL |
| **zoom** (clues) | evidence-driven 2× zoomed second look | **+0.0085** [+0.0080, +0.0082, +0.0091] | +0.0065 | −0.0007 | PASS, consistent; falsifier part 1 holds (gain on BCN/MSKCC +0.0147, HAM −0.0032); part 2 (random-location control) running |
| **youngdata** (control) | +2,078 young histology-confirmed ISIC images | +0.0073 [+0.0251, −0.0053, +0.0021] | +0.0075 | +0.0023 | PASS on frozen bars, **fragile**: seed-42-driven; within-band falsifier holds on the mean (band-mean +0.0111, <40 +0.0225) but fails in seed 44 |
| look (control) | chromophore input channels + palette tokens | +0.0044 | +0.0044 | +0.0005 | **void** — trained with NaN steps (§2b); re-screen running |
| geometry (control) | lesion-axis asymmetry, border abruptness | +0.0164 | +0.0296 | −0.0323 | **void** — NaN in 9–18 of 30 epochs (§2b); re-screen running |

**Readout vs representation (the key V5 lesson).** Re-scoring arms on the control's score (escalation
mass) separates "a better readout head" from "a better image representation":
- twostep vs control: +0.0155 declared → **+0.0025** on mass.
- **zoom vs twostep** (the arm it would join): **+0.0071** declared [all seeds positive] → **−0.0001** on
  mass; under-40 −0.024 (descriptive); and zoom costs **~1.75–1.95×** the GPU time (73–81 vs ~42 min/run).
So V5's passing heads mostly improve how escalation is *read out*, not what the trunk *sees*.

**Still running (2–3 Oct):** look/geometry re-screen after the fix, zoom random-location control
(3 seeds), leave-one-archive-out control vs M7 (physically based acquisition randomisation + SWAD).
Then a composite lock, and confirmation on folds 1–4 (hierarchical bootstrap, seeds then lesions).

## 2b. Defects found and fixed on 2 Oct (lessons for any V6 engineered feature)
- **Finite-but-huge features overflow fp16 under AMP.** The chromophore front computed tokens in fp32 and
  passed them through `nan_to_num`, which only catches NaN/inf. Two tokens produced *finite* values of
  10⁷–10⁸, which overflowed fp16 (max 65,504) in the classifier → NaN loss → GradScaler skipped the step.
  (a) colour eccentricity divided by a lesion radius of 0 on augmented crops with **one valid pixel**
  (coverage was measured over valid pixels, so the fallback never fired); (b) border abruptness divided
  by a lesion median melanin of ≈ 0. Fixed at source (geometry fallback on a too-small valid region;
  abruptness undefined below 0.1·t_mel), plus a logged ±50 SD saturation backstop.
- **Heavy-tailed engineered features poisoned their own standardisation**: the abruptness-variance token
  had fold mean 5.0 / SD **333** (from blow-ups), so every ordinary image standardised to ≈ 0 — the token
  was dead except when it overflowed. After the fix: 0.0013 / 0.0057; a legitimate tail still reaches
  z ≈ 98 (raw 0.56).
- **Benchmark ratios against a cold reference** under-projected run times by ~20%; absolute per-epoch
  seconds were accurate. *(CHANGELOG, 2 Oct)*

---

## 3. The V6 plan as it stands (finalised, not frozen) — `docs/V6_RUNSHEET.md`

- **Headline: Counterexample-Guided Differential Morphology (CG-DM).** Melanoma ranking should improve when
  the model learns the *local* evidence that distinguishes a lesion from visually similar,
  **histopathology-confirmed nevus** counterexamples. Formulated as **partial explanation**: one-sided
  partial optimal transport lets K = 3 retrieved biopsied nevi explain part of the lesion's F3 tokens; the
  unexplained remainder is the differential evidence. Motivated (not asserted) by nevus-associated
  melanoma. No metadata in any input, rule or retrieval key.
- **Endpoints:** clinical primary = under-40 histology-confirmed pAUC@0.2; mechanism primary = melanoma vs
  biopsied nevus pAUC@0.2, all ages (fold 0: 505 mel vs 671 histo nv; control 0.692–0.722). Holm across both.
- **Stage 0 (budget gate, CPU after one GPU feature pass):** does the differential exist in the current
  representation? Δ(pAUC) of logistic([e, D_part]) over e alone, CI > 0 and ≥ +0.010.
- **Branches:** R1 (signature-anchored counterexamples using patient IDs in ISIC-2020); R2 (generated
  pseudo-benign twin, high risk); DINOv3 with layer-wise LR decay; a **six-architecture in-domain
  ensemble** (V6-3b) with distillation (V6-10), whose E-primary is **co-primary**: paired all-age pAUC
  above noise **and** paired Macro-F1 Δ > 0 (intersection-union test).
- **Carried from V2–V4 (§A9):** bipartite RAPS conformal with per-group false-referral control; a label-free
  referral-budget lock; per-band Dirichlet as a calibration arm; eval-transform parity test; drift hooks;
  TTA; the ensemble. 13 standing issues mapped to phases, including Fitzpatrick V/VI (**not solvable by
  V6 alone** — no known dermoscopy source with V/VI labels).
- **Audit corrections already applied (1 Oct):** (1) Stage 0 scored a head the reference models do not
  have → uses escalation mass; (2) screen bars came from the wrong trunk → V6 trains three extra
  IN-22k control seeds and recomputes bars over 15 pairs; (3) readout vs representation → every CG-DM
  contrast reported on both escalation mass and the head's score, and only a mass gain may be called a
  representation gain; (4) budget consolidated to **≈ 88–121 GPU-h** (extrapolated from 38–50 min/run).
- **Corrections from 2 Oct, to be applied before the freeze:** the decision row "zoom > zoom_random →
  promote dual-stream / wavelet space-to-depth" must also require a gain on escalation mass (Defect 3);
  every engineered-token module must check **fp16-castability and bounded standardisation**, not only
  finiteness; the youngdata decision row needs a per-seed reading (it holds on the mean, fails in 1 of 3);
  zoom's cost anchor is ~1.8×, not the assumed 1.34×.

---

## 4. Metrics and how results are judged

- Primary: **Macro-F1**, **balanced accuracy**, **escalation sensitivity**, **pAUC@FPR≤0.20**
  (McClish-standardised) all-age, on histology-confirmed rows, and under 40.
- Accuracy is **never** a selection criterion (nv is 67% of HAM).
- Screens: fold 0, 3 seeds, paired against a comparator, 80th percentile of measured seed noise.
  Confirmation: folds 1–4, hierarchical bootstrap over seeds then lesions. Gate A (young endpoint):
  Δ under-40 pAUC ≥ **+0.05** with CI lower bound > 0 — not expected; V4's largest move was 0.011.

---

## 5. Hard rules — anything that breaks these is unusable

1. All splits grouped by lesion; no lesion in two partitions.
2. The HAM test split is locked; the BCN/MSKCC reserved cohort is exhausted; MILK10k is read once (S84).
3. Every weight, threshold, temperature or calibrator is fitted on out-of-fold or validation data only.
4. Every reported number comes from a file in `results/`.
5. **Barred mechanisms** (12; `docs/v5_record/V5_MASTER_RESEARCH_PLAN_REVISED.md` §21): frozen specialist
   heads; tabular metadata fusion; adversarial invariance to **age** (invariance to archive given class is
   allowed); **frozen** foundation-model probes (fine-tuning is allowed); global prevalence prior
   correction; age-dependent decision rules (λ(age), learned age priors) and changes to the S56 decision
   layer; colour constancy; repeated reads of a reserved cohort. **Age may not be an input or a
   threshold**: a fix must improve ranking *within* the young band.

---

## 6. Already tried — with the result (do not re-propose without a new reason)

| Tried | Result | Source |
|---|---|---|
| Ensembling 6 architectures (HAM) | Only overall significant gain (McNemar p=3.4e-05); under-40 AUC **+0.004 [−0.042, +0.052]** | `results/mcnemar_delong.json`, S64 |
| Pooled training (34 → 81 young escalating) | Under-40 flat | S54 |
| Young-specialist head / hard-case reweighting / metadata | **−0.039 / −0.000 / +0.007** under-40 pAUC | `results/v4/s67/` |
| Frozen foundation probes (DINOv2, PanDerm) | DINOv2 **−0.0725** under 40 | S51 |
| Escalation heads (MLP/GBM) | Δ pAUC **−0.067** | `results/v5/v5_no_repeat_registry.json` |
| λ(age) rule; conformal layers | Help the young least; lose to S56 at matched workload | S5, S65 |
| 384 px | **+0.0034** pooled (CI includes 0) | `results/v5/s01_decision.json` |
| Class-balanced sampling, RandAugment, EMA, SupCon/CosFace | Null | CHANGELOG S53r |
| Three-stage routed front end | Under-40 pAUC **+0.007** (flat) | S58 |
| DINOv3 fine-tune, single LR | Collapses when the trunk unfreezes (Macro-F1 0.38–0.44) | `gate_control_dinov3_vs_control.json` |
| **V5 modules** (§2a) | twostep/m4 readout gains; memory, clues, gem fail; zoom readout-only vs twostep; youngdata fragile; look/geometry pending | §2a |

---

## 7. Constraints

- GPU: RTX 5050 Laptop, **8.55 GB** VRAM. RAM 15.2 GB; Windows commit headroom (~7.5 GB) is the real
  limit — 4 DataLoader workers crash (error 1455); 2 workers is the rule.
- Measured per fold-0 run (30 epochs, 12,235 images, 224 px, IN-22k): control/heads **38–47 min**; look,
  geometry **42–43**; youngdata (+2,078 images) **58.6**; zoom (448 px view, second trunk pass) **73–81**.
  Fine-tune VRAM 2.4 GB (zoom 4.4 GB); 384 px 6.3 GB. *(`results/v5/logs/queue_Q*.log`)*
- Budget: ~125 GPU-hours/week at best; V6 core ≈ 88–121 GPU-h.
- Data on hand: pooled ISIC-2019 (15,294), HAM10000, PAD-UFES-20, ISIC-2018 Task-2 masks (2,594), MILK10k
  (reserved until S84), 2,078 young histology-confirmed ISIC images (all <40, 510 escalating). V6 adds
  ISIC-2020 (patient IDs), DERM12345, HIBA (confirmation only).

---

## 8. Repository map

| Topic | Files |
|---|---|
| Rules, history, findings | `CLAUDE.md`, `docs/RESEARCH_LOG.md`, `CHANGELOG.md` (V5 entries at the end) |
| V5 plan, arms, gates | `docs/V5_RUNSHEET.md`, `docs/v5_design/` (M1–M7, DRE modules, audit) |
| V6 plan | `docs/V6_RUNSHEET.md` (§A0 CG-DM, §A5 decision table, §A9 carry-forward, §B budget) |
| Barred / tried registry | `results/v5/v5_no_repeat_registry.json`, `docs/v5_record/V5_MASTER_RESEARCH_PLAN_REVISED.md` §21 |
| Code (may be absent from the snapshot) | `research/v5/` (train_v5, modules, front, chromophore, zoom, screen_gate), `research/v6/` |

---

## 9. The questions (where I need help)

**A. Why biology adds so little — and when it would add more.**
1. Several modules re-express information already in the 224 px pixels (chromophore unmixing, asymmetry,
   border abruptness), while the two that passed most convincingly add new information (zoom: pixels lost
   to downscaling; youngdata: new images). Is there evidence, in dermoscopy or medical imaging, on when
   hand-crafted or physics-derived features improve a *fine-tuned, strongly pretrained* CNN (low-data
   regimes, distribution shift, resolution limits) versus when they are redundant? What effect sizes are
   reported, and on which metric?
2. Propose a **cheap test** (CPU, or a single GPU pass) that separates "the feature carries information the
   trunk lacks" from "the feature only adds inductive bias", before spending screen GPU time.
3. For heavy-tailed engineered features (ratios, variances): which representation (log, rank/quantile,
   robust scaling, binning) is standard for feeding them into a deep classifier, with evidence?

**B. Readout vs representation.**
4. twostep's and zoom's gains largely vanish when scored on the control's escalation mass. Is a gain that
   comes from a dedicated escalation head (with the same trunk) a legitimate clinical improvement or a
   calibration/readout artefact? How do comparable papers evaluate this (linear probe vs fine-tune
   protocols, head-swap tests)? Should the V6 composite count readout gains?

**C. CG-DM (the V6 headline).**
5. **Prior-art audit:** any published work using partial / unbalanced optimal transport, or retrieval of
   biopsy-proven references, for melanoma-vs-nevus discrimination or low-FPR ranking? Give identifiers.
   What is the closest work, and is the novelty sentence in §3 defensible?
6. **Failure modes:** retrieval leakage via near-duplicates, references that are themselves mislabelled,
   K = 3 too small, OT on 14×14 F3 tokens too coarse. Which is most likely to sink it, and what Stage 0
   result would reveal it early?
7. Is there evidence that **nevus-associated melanoma** is over-represented among *young* melanomas in
   dermoscopy datasets, which would make the partial-explanation framing apt for Wall 1?

**D. Young data and fairness data.**
8. Learning curves: from comparable evidence, is +2,078 young images (510 escalating) expected to move
   under-40 pAUC measurably, or is an order of magnitude more needed? The V5 result is fragile (§2a).
9. Public dermoscopy sources with **age**, **histopathology confirmation**, and ideally **Fitzpatrick V/VI**
   labels (licences, overlap with ISIC-2019, deduplication). Please state plainly if none exist for V/VI.

**E. Statistics.**
10. Why would a stronger pretrained trunk (IN-22k) show **~2× more seed variance** on all-age pAUC than
    IN-1k, but the same variance on melanoma-vs-biopsied-nevus? Known results on seed variance vs
    pretraining? What does that imply for screen design (more seeds vs more folds; paired lesion-level
    tests; screening on the lower-noise mechanism endpoint)?
11. With 35 young escalating rows per fold and 71 externally, what is the most powerful honest design for
    a young-band claim (pooling folds, hierarchical models, equivalence framing)?

**F. Ensembles, domain shift, the contract.**
12. Architecture diversity vs seeds × folds of one architecture, for in-domain dermoscopy: what does the
    evidence say about gains on ranking metrics at low FPR?
13. Most cost-effective route to smartphone robustness (Wall 2), given that post-hoc calibration and
    ensembling made PAD worse.
14. Is 0.855 young sensitivity at ≤ 25% referral achievable from a single dermoscopic image at all? If
    not, what claim should replace it?

**G. Triage of the V6 plan.**
15. Which V6 items (§3: CG-DM, R1, R2, DINOv3-LLRD, the ensemble, carried-forward items) do you expect to
    fail, and why? What would you cut first under a fixed ~100 GPU-h budget?

Please say plainly when the honest answer is "this is probably not fixable with the available data",
and what evidence would change that view.
