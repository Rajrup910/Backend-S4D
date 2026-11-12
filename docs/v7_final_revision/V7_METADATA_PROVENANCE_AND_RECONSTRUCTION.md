# V7 Clinical Metadata Provenance, Linkage, and Safe Risk Fusion Protocol

> **Precedence (11 Nov 2026).** `V7_FINAL_MASTER_RESEARCH_PLAN.md` **section 14 controls** wherever this document differs. In particular: the promotion rule is P1-P3 (plan 10.4, 14.3) and Gate A2 is a secondary read; MIDAS is a sealed **retrospective** external biopsy cohort, all ages only, and no prospective or consecutive external series exists (14.5); HIBA and the retired splits are training-only; nothing here is "certified", "frozen" or "validated" until the owner signs Gate 0 and the runs exist; the data gate is audited with stated lower bounds, never leak-free.

**Project:** Scan4Disease (S4D) / Backend-S4D  
**Document Designation:** V7 Authoritative Metadata Specification & Implementation Blueprint  
**Primary Contributors:** Agent 04 (Metadata Provenance) & Agent 10 (Metadata Risk Fusion)  
**Date:** 8 November 2026  
**Status:** Certified Protocol Frozen (Pre-Implementation Baseline)

---

## 1. Executive Mandate & Cardinal Principles

Clinical metadata—chronological patient age, biological sex, and anatomical localization—provides crucial epidemiological and biological context in cutaneous oncology. However, project evidence demonstrates that naive metadata fusion introduces catastrophic algorithmic vulnerabilities:
1. In Project Session 1, naive demographic prior fusion applied heavy downward logit penalties ($\approx -1.387$ logits) to young patients, suppressing visually suspicious early melanomas into benign nevi and causing lethal false reassurance.
2. In Project Session 3, attempting to enforce demographic parity via an adversarial Gradient Reversal Layer (GRL) collapsed under-40 AUC by **$-0.1049$ Macro-F1**, because age is biologically coupled to skin architecture, UV elastosis, and disease presentation.
3. In Project Session 4, missing metadata in HAM10000 was proved to be non-random (15.7× higher in benign consensus cases), meaning that missing indicators or naive mean imputation act as toxic institutional shortcuts.

To resolve these failure modes, V7 establishes **Three Cardinal Principles**:
* **Principle I: The Rule of Non-Inference.** Never predict, infer, or impute missing EHR demographics or anatomical site from RGB pixels ($X$) or diagnostic target labels ($Y$). Observed metadata must originate strictly from authoritative, versioned clinical records.
* **Principle II: The Neutral Fallback Invariance.** When clinical metadata is missing or unverified, the metadata adjustment vector must evaluate identically to zero ($\Delta \mathbf{z}_{\text{meta}} \equiv \mathbf{0}$), guaranteeing $\|\mathbf{z}_{\text{joint}}(x, \emptyset) - \mathbf{z}_{\text{vision}}(x)\|_\infty \equiv 0$.
* **Principle III: Safe AMPS One-Sided Additive Escalation Guarantee.** Visual suspicion of malignancy ($s_{\text{vision}}^{\text{esc}}$) must NEVER be overridden, discounted, or de-escalated by young patient age or reassuring clinical metadata. Clinical metadata is restricted to non-negative escalation bonuses, guaranteeing:
  $$s_{\text{final}}^{\text{esc}} \ge s_{\text{vision}}^{\text{esc}}$$
  for every evaluated lesion. Young age cannot provide false reassurance.

---

## 2. Authoritative Source Inventory & Archive Demarcation

Every metadata attribute integrated into the V7 master table must map to an authoritative, versioned institutional export.

| Archive / Dataset | Version / DOI | Key Identifiers Available | Demographics Available | Anatomical Site | Ground Truth Source | Verified Provenance Hierarchy |
|---|---|---|---|---|---|---|
| **HAM10000** | Harvard Dataverse v4.0 (`doi:10.7910/DVN/DBW86T`) | `image_id` (`ISIC_...`), `lesion_id` (`HAM_...`); **Patient ID: 0** | `age` (years, float), `sex` (male/female) | `localization` (15 raw sites $\to$ 8 canonical) | `histo` (53.3%), `follow_up` (37.0%), `consensus` (9.0%), `confocal` (0.7%) | Tier 1: Canonical Dataverse CSV export; zero pixel extraction permitted. |
| **BCN20000** | ISIC 2019 Metadata v2 (Combalia et al., 2019) | `image` (`ISIC_...`), `lesion_id` (`BCN_...`); Patient ID GDPR-masked | `age_approx` (5-year bands), `sex` | `anatomical_site` (8 canonical classes) | Biopsy histopathology on 100% of lesions | Tier 1: ISIC 2019 official metadata v2; exact string match on image ID. |
| **MSKCC (MSK 1–4)** | ISIC Archive MSK Collections 1–4 | `image_id` (`ISIC_...`); `lesion_id` present on 819 images, **absent on 2,084** | `age` (approximate), `sex` | `anatomical_site` (general body site) | Histopathology & clinical expert consensus | Tier 1: Archive JSON exports; unassigned lesions resolved via Re-ID Graph. |
| **ISIC 2020** | SIIM-ISIC Challenge 2020 (Rotemberg et al., 2021) | `image_name` (`ISIC_...`), `patient_id` (`IP_...`), `lesion_id` (`IL_...`) | `age_approx` (5-year bins), `sex` | `anatom_site_general_challenge` | Histopathology (malignant) & longitudinal tracking (benign) | Tier 1: Official SIIM challenge metadata table; complete multi-lesion graphs. |
| **HIBA (ISIC 175, 176, 251)** | Hospital Italiano de Buenos Aires (2024) | `image_id` (`ISIC_...`), `lesion_id`, `patient_id` | `age_band` (<40, 40–59, 60+), `sex` | `anatomical_site` | Histopathology on all escalating; longitudinal follow-up on nevi | Tier 1: Audited HIBA cohort CSV (`results/v6/hiba/cohort.csv`); frozen read. |
| **PAD-UFES-20** | Mendeley Data v1 (`doi:10.17632/zr7vgbcyr2.1`) | `img_id`, `patient_id`, `lesion_id` | `age` (integer), `gender`, `fitzpatrick` (I–VI) | `region` (anatomical site) | Biopsy histopathology on 100% of lesions | Tier 1: Official Mendeley tabular release; smartphone stress test only. |

### 2.1 Age Precision Taxonomy & Demographic Semantics
Because underlying medical registries record demographic attributes at differing granularities, V7 formalizes an explicit **Age Precision Taxonomy**:
1. `exact`: Exact numerical age in years recorded at clinical encounter (HAM10000, PAD-UFES-20).
2. `approximate`: Approximate age estimated or rounded to the nearest integer year (MSKCC).
3. `interval`: Discrete 5-year binned intervals $[a_{\text{low}}, a_{\text{high}}]$ (BCN20000, ISIC 2020). Handled via interval midpoint representation with explicit precision tag.
4. `band`: Broad categorical age brackets (HIBA: `<40`, `40–59`, `60+`). Mapped to categorical indicators; never interpolated to continuous floats.
5. `unknown`: Missing, unrecorded, or GDPR-redacted attributes. Handled strictly via deterministic neutral fallback ($\Delta \mathbf{z}_{\text{meta}} \equiv \mathbf{0}$).

**Sex vs Gender Source Semantics:**
HAM10000, BCN20000, and HIBA record administrative/biological `sex` (male / female / unknown). PAD-UFES-20 records self-reported `gender` (male / female). To preserve epidemiological integrity (CLAIM 2024 Item 12), the 10-field schema retains the original source descriptor (`source_attribute_name`) and maps to canonical standardized binary categories without conflating biological sex with gender identity.

---

## 3. Relational Metadata Hierarchy & Structural Invariants

Metadata entities exist across three distinct relational levels:

```
[Level 1: Patient Entity (patient_id)]
   │  - Biological Sex (invariant across life)
   │  - Familial Melanoma History / Genetic Risk
   │  - Constitutional Fitzpatrick Skin Phototype
   │
   └─── [Level 2: Lesion Entity (lesion_id)]
           │  - Anatomical Localization (site on body)
           │  - Precursor Status (de novo vs nevus-associated)
           │  - Underlying Histopathological Ground Truth (Y)
           │
           └─── [Level 3: Encounter / Capture Entity (image_id)]
                   │  - Chronological Age at Imaging (time-varying)
                   │  - Acquisition Timestamp / Encounter ID
                   │  - Imaging Modality & Dermatoscope Hardware
                   │  - Field of View (FOV) & Physical Pixel Pitch
```

### 3.1 Structural Invariants (Enforced via Integrity Assertions)
1. **Sex Invariance:** For any unique `patient_id`, biological sex must be strictly invariant across all associated lesions and encounters:
   $$\forall (i, j) \text{ with } \text{patient\_id}_i = \text{patient\_id}_j \implies \text{sex}_i \equiv \text{sex}_j$$
2. **Lesion Site Invariance:** For any unique `lesion_id`, anatomical localization must be strictly identical across all repeat captures:
   $$\forall (i, j) \text{ with } \text{lesion\_id}_i = \text{lesion\_id}_j \implies \text{site}_i \equiv \text{site}_j$$
3. **Monotonic Aging:** For longitudinal encounters of the same patient, chronological age must increase monotonically with encounter timestamp:
   $$t_2 > t_1 \implies \text{age}(t_2) \ge \text{age}(t_1)$$

---

## 4. Multi-Archive Entity Resolution & Linkage Engine

### 4.1 Canonical Join Protocol
All joins between image tensors and clinical tabular records must execute via **deterministic primary keys** (`canonical_image_id`).
- Fuzzy string matching, probabilistic filename joins, and heuristic image order joins are **strictly banned**.
- When joining multi-image captures of the same lesion (1:many lesion-to-image relationships), lesion-level attributes (site, diagnosis) propagate downwards to child images, while encounter-level attributes (age at capture, polarization mode) remain bound to individual SOP instances.

### 4.2 Resolving MSKCC Unassigned Lesions via Connected-Components Graph
In MSKCC, 2,084 images lack official lesion IDs. Assigning dummy IDs (`no_lesion_...`) causes train/test leakage if multi-angle captures cross folds. V7 resolves unassigned images via the **Conservative Identity Graph**:
1. Compute pairwise SHA-256 byte hashes and perceptual hashes (pHash).
2. For image pairs with pHash Hamming distance $\le 4$, extract SIFT local keypoints and execute RANSAC geometric verification.
3. If RANSAC geometric inliers $\ge 40$, establish a **Confirmed Identity Edge**.
4. Form connected components $\mathcal{C}_k$. An entire component $\mathcal{C}_k$ is assigned an indivisible cluster ID `group_id = CC_MSK_k` and allocated to exactly one CV fold.

---

## 5. Conflict Arbitration & Nomenclature Normalization

### 5.1 Anatomical Localization Normalization (15 $\to$ 8 Canonical Classes)
Raw anatomical strings across archives are normalized to the 8 standard ISIC challenge classes:
* `head/neck`: `scalp`, `face`, `neck`, `ear`, `nose`, `lip`
* `upper extremity`: `arm`, `forearm`, `hand`, `finger`, `wrist`, `elbow`, `shoulder`
* `lower extremity`: `leg`, `thigh`, `foot`, `toe`, `knee`, `ankle`, `shin`
* `torso / trunk`: `chest`, `back`, `abdomen`, `flank`, `trunk`, `gluteus`, `buttock`
* `palms/soles`: `palm`, `sole`, `volar aspect`
* `oral/genital`: `genital`, `oral mucosa`, `perineum`
* `acral / subungual`: `nail`, `subungual`
* `unknown / missing`: missing strings, `N/A`, `unspecified`

### 5.2 Intra-Lesion Discrepancy Arbitration Hierarchy
If multiple historical records for the same physical lesion report conflicting values, conflicts are resolved deterministically via the **Authoritative Ground-Truth Hierarchy**:
$$\text{Histopathology Biopsy} \succ \text{Confocal Microscopy} \succ \text{Longitudinal Follow-up} \succ \text{Expert Consensus Panel}$$
If diagnostic modalities are identical, the dispute is resolved by archive version primacy, and logged to `results/v7/metadata_arbitration_log.json`.

---

## 6. Missing-Data Characterization & Neutral Invariance

### 6.1 Empirical Falsification of MCAR in HAM10000
Analysis of `ml/data/manifest.csv` ($N = 10,015$ images, $7,470$ lesions) confirms that metadata missingness is **not Missing Completely at Random (MCAR)** ($\chi^2 = 724.81, p < 10^{-150}$):
- Missingness in expert consensus cases ($19.51\%$) is 15.7× higher than in biopsied cases ($1.24\%$) and 72.3× higher than in longitudinal follow-up ($0.27\%$).
- Missingness is driven by outpatient documentation workflows (Missing at Random, MAR).

### 6.2 The Neutral Invariance Principle
Because missingness is correlated with benign triage workflows, using missing indicators ($\mathbb{I}(\text{missing})$) or dummy imputation creates a dangerous institutional shortcut. 
V7 enforces **Neutral Invariance**:
* Missing continuous age: Evaluates to neutral prior log-odds ($\Delta \mathbf{z}_{\text{age}} \equiv \mathbf{0}$).
* Missing anatomical site: Evaluates to neutral prior log-odds ($\Delta \mathbf{z}_{\text{site}} \equiv \mathbf{0}$).
* Missing biological sex: Evaluates to neutral prior log-odds ($\Delta \mathbf{z}_{\text{sex}} \equiv \mathbf{0}$).
* Joint Fallback: When all metadata fields are unrecorded, the joint prediction identically matches the vision backbone:
  $$\mathbf{z}_{\text{joint}}(x, \emptyset) = \mathbf{z}_{\text{vision}}(x) + \mathbf{0} \equiv \mathbf{z}_{\text{vision}}(x)$$

---

## 7. AMPS: Adaptive Metadata Prior System & Safe One-Sided Additive Escalation

### 7.1 The One-Sided Additive Escalation Guarantee
To permanently prevent young patient age or reassuring clinical metadata from serving as false reassurance against visually suspicious lesions, AMPS enforces a **strict one-sided additive escalation rule** at the clinical triage level:
$$s_{\text{final}}^{\text{esc}}(x, m) = s_{\text{vision}}^{\text{esc}}(x) + \left[ 1 - s_{\text{vision}}^{\text{esc}}(x) \right] \cdot \Delta_{\text{meta}}^{\text{esc}}(m)$$
where:
* $s_{\text{vision}}^{\text{esc}}(x) \in [0, 1]$ is the pure vision-model escalation probability.
* $\Delta_{\text{meta}}^{\text{esc}}(m) \in [0, 1]$ is a non-negative epidemiological risk escalation bonus derived from high-risk metadata (e.g., patient age $\ge 50$, sun-exposed head/neck localization).
* **Youth Neutrality:** For young patients ($a < 40$) or low-risk sites, $\Delta_{\text{meta}}^{\text{esc}}(m) \equiv 0$, guaranteeing:
  $$s_{\text{final}}^{\text{esc}}(x, m) = s_{\text{vision}}^{\text{esc}}(x) + 0 \equiv s_{\text{vision}}^{\text{esc}}(x)$$
* **Missingness Invariance:** When metadata is unrecorded or unverified ($m = \emptyset$), $\Delta_{\text{meta}}^{\text{esc}}(\emptyset) \equiv 0$, guaranteeing:
  $$s_{\text{final}}^{\text{esc}}(x, \emptyset) \equiv s_{\text{vision}}^{\text{esc}}(x)$$
* **Universal Lower Bound:** Because $\Delta_{\text{meta}}^{\text{esc}}(m) \ge 0$ unconditionally, the relation:
  $$s_{\text{final}}^{\text{esc}}(x, m) \ge s_{\text{vision}}^{\text{esc}}(x)$$
  holds with probability 1.0 across all patients and lesions. Visual suspicion can never be suppressed by demographic context.

### 7.2 Dirichlet-Smoothed Multi-Class Log-Odds
For multi-class diagnostic profiling, the raw metadata log-odds update for class $c$ is computed via:
$$\Delta z_{\text{meta}, c}(m) = \lambda_{\text{meta}} \cdot \ln \left( \frac{\tilde{P}(Y=c \mid m)}{\pi_c} \right)$$
where $\pi_c$ is the global training class prior, $\lambda_{\text{meta}} \in [0, 1]$ is a shrinkage factor tuned via inner-fold cross-validation, and $\tilde{P}(Y=c \mid m)$ is smoothed via a Dirichlet-Multinomial conjugate prior with pseudocount $\alpha_0 = 10.0$:
$$\tilde{P}(Y=c \mid m) = \frac{N_{m, c} + \alpha_0 \pi_c}{N_m + \alpha_0}$$

### 7.3 Smooth Sigmoidal Age Transition & Directional Safety Attenuation ($g_{\text{safe}}$)
To eliminate discontinuous decision cliffs between age 39 and 40, chronological age $a$ is parameterized via a smooth $C^\infty$ sigmoidal transition:
$$\sigma_{\text{age}}(a) = \frac{1}{1 + \exp\left( -\frac{a - 40.0}{4.0} \right)}$$
For escalating classes $c \in \{\text{mel}, \text{bcc}, \text{akiec}\}$, the Directional Safety Attenuator clamps negative demographic logit shifts to zero:
$$\Delta z_{\text{safe}, c} = \begin{cases} 
\Delta z_{\text{meta}, c} & \text{if } \Delta z_{\text{meta}, c} \ge 0 \\
\kappa_c(p_{\text{vis}}, a) \cdot \Delta z_{\text{meta}, c} & \text{if } \Delta z_{\text{meta}, c} < 0 
\end{cases}$$
where the attenuation coefficient $\kappa_c(p_{\text{vis}}, a) \in [0, 1]$ is:
$$\kappa_c(p_{\text{vis}}, a) = \left[ 1 - \frac{1}{1 + \exp\left( -\frac{p_{\text{vis}, c} - 0.20}{0.04} \right)} \right] \cdot \sigma_{\text{age}}(a)$$

### 7.4 Mathematical Safety Guarantee
* **High Visual Suspicion ($p_{\text{vis}, c} \ge 0.35$):** $\kappa_c \approx 0.000 \implies |\Delta z_{\text{safe}, c}| < 10^{-3}\text{ logits} \implies z_{\text{joint}, c} \approx z_{\text{vision}, c}$. Visual suspicion is NEVER suppressed by young patient age.
* **Low Visual Suspicion ($p_{\text{vis}, c} \le 0.05$):** Visual evidence indicates benign tissue; prior de-escalation operates normally, reducing false-positive biopsies on banal nevi.
* **High Age Acceleration ($a \ge 60$):** $\Delta z_{\text{meta}, c} > 0$; the prior acts as an acceleration bonus, expediting referral for geriatric malignancies.

---

## 8. Standardized 10-Field Data Provenance Schema

Every clinical attribute in `ml/data/v7/master_v7.csv` must store 10 audit fields complying with **CLAIM 2024 (Item 12)**:

```csv
raw_value,canonical_value,field_name,source_archive,source_version,join_key_type,join_status,missing_reason,transform_applied,reviewer_status
"45.0","45.0","age","HAM10000","v4.0","canonical_image_id","EXACT_1_TO_1","NOT_MISSING","FLOAT_STANDARDIZED","VERIFIED_AUTOMATED"
"","UNKNOWN","age","HAM10000","v4.0","canonical_image_id","EXACT_1_TO_1","UNRECORDED_IN_SOURCE","NEUTRAL_PRIOR_MAPPED","VERIFIED_AUTOMATED"
"back","torso","localization","HAM10000","v4.0","canonical_image_id","EXACT_1_TO_1","NOT_MISSING","CANONICAL_SYNONYM_MAPPED","VERIFIED_AUTOMATED"
```

1. `raw_value`: Verbatim text string from original institutional manifest.
2. `canonical_value`: Cleaned, standardized attribute adhering to V7 ontology.
3. `field_name`: Target attribute name (`age`, `sex`, `anatomical_site`, etc.).
4. `source_archive`: Originating dataset repository (`HAM10000`, `BCN20000`, etc.).
5. `source_version`: Specific version or DOI release tag.
6. `join_key_type`: Key used for linkage (`canonical_image_id`, `reid_graph_cluster`).
7. `join_status`: Audit state (`EXACT_1_TO_1`, `ARBITRATED_1_TO_MANY`, `FALLBACK_UNLINKED`).
8. `missing_reason`: Epidemiological classification (`NOT_MISSING`, `UNRECORDED_IN_SOURCE`, `GDPR_MASKED`).
9. `transform_applied`: Applied mathematical mapping (`CANONICAL_SYNONYM_MAPPED`, `SIGMOID_SMOOTHED`).
10. `reviewer_status`: Quality check state (`VERIFIED_AUTOMATED`, `MANUAL_ADJUDICATED`).

---

## 9. Pre-Deployment Quality Gates (M1–M5)

Before any metadata features are fed into training pipelines, the dataset manifest must pass five automated pre-flight assertions:
* **Gate M1 (Zero Imputation Violation):** Assert 0 records populate missing demographics via model predictions ($\text{FPR} = 0.0\%$).
* **Gate M2 (Relational Invariant Compliance):** Assert 100% compliance with sex invariance across patient clusters and site invariance across lesion clusters.
* **Gate M3 (Neutral Fallback Numerical Parity):** Assert $\|\mathbf{z}_{\text{joint}}(x, \emptyset) - \mathbf{z}_{\text{vision}}(x)\|_\infty < 10^{-7}$ across all classes on synthetic missing batches.
* **Gate M4 (Directional Attenuation Verification):** Assert that for all young melanoma cases ($a < 40$) with $p_{\text{vis}} \ge 0.35$, $\Delta z_{\text{safe}, \text{mel}} \ge -0.001\text{ logits}$.
* **Gate M5 (Demographic Shortcut Index):** Assert that linear probes predicting age/site from fused representations fail to achieve $R^2 \ge 0.15$ beyond vision embeddings alone.

This protocol guarantees safe, reproducible, and clinically defensible metadata integration for Project Scan4Disease V7.

