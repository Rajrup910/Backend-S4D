# V6 data sources: public dermoscopy inventory, under-40 escalating supply, ISIC 2019 overlap, external cohorts

Compiled 2026-11-01/02.

**Tags**
- **VERIFIED**: read in the primary source (dataset paper full text, the official page, or a live ISIC API response).
- **SECONDARY**: from a search summary, a compilation or a mirror.
- **DERIVED**: my own arithmetic or join on VERIFIED data. This includes joins on `isic_id`/`lesion_id`/`patient_id` across about 90k metadata records pulled from `api.isic-archive.com/api/v2/images/search/`. No images were downloaded.

**Definitions**
- "Young" means ISIC `age_approx` ≤ 35. ISIC rounds age to the nearest 5 years, so this covers ages below about 37.5. Ages 38–39 sit in the 40 bucket, which I report separately as an upper-bound add-on.
- "Escalating" means histopathology-confirmed melanoma, BCC or SCC, plus AK where stated.
- "ISIC 2019" means archive collections 65 (Training, 25,331) and 72 (Test, 8,238).

## 0. Headline findings

1. **The archive's young histopathology pool is nearly exhausted.** There are 5,790 young dermoscopic histopathology images in total (your 5,790). Of these, 3,656 are in ISIC 2019/BCN/HAM and **2,134 outside**. That matches the ≈2,078 extras you already used. *VERIFIED facets; DERIVED.*
2. **Young escalating lesions, archive-wide** (histopathology-confirmed malignant, dermoscopic):
   - **2019 Training: 147 lesions** (107 MEL, 39 BCC, 1 SCC), plus 26 images with no `lesion_id`. Your 81 training lesions plus the reserved BCN/MSKCC partition should add up to this.
   - **2019 Test: 47 lesions** (36 MEL, 10 BCC, 1 SCC): 42 from BCN, 5 from HAM val/test.
   - **Outside ISIC 2019: 234 lesions** plus 100 unlinked images. 28 of those lesions carry a `lesion_id` that also appears in 2019 Training.
   - *DERIVED from the API.*
3. **Correcting a common assumption: the BCN20000 train/test split is lesion-disjoint.** The 3,576 train lesions and 1,864 test lesions share **0** `lesion_id`s, and 2019 Train vs 2019 Test also share 0. So the **47 young escalating lesions in 2019 Test do not share a lesion with your reserved BCN rows.** They are the largest clean pool you have not touched. *DERIVED.*
4. **MILK10k is lesion-linked to HAM10000.** **859 of its 5,240 lesions (16.4%) share a `lesion_id` with ISIC 2019 Training, and 126 more with 2019 Test, all of them HAM10000 ("MILK study team")**. Of its 72 young escalating lesions, 21 are HAM lesions. Before the external read, and before V6 training, drop these lesions or route them by `lesion_id` against your HAM test rows. *DERIVED.*
5. **HIBA has only 20 young escalating lesions** (in 18 patients), so it cannot confirm an under-40 claim on its own. It is fine as an all-age confirmation cohort. *DERIVED/VERIFIED.*
6. **There are two fully disjoint, non-ISIC external cohorts:**
   - **SCP2** (8 German university hospitals; 1,910 lesions; 750 melanomas; 6 dermoscopic images per lesion; available on request). Its test set has 14 melanomas under age 35.
   - **MRA-MIDAS** (Stanford; 1,290 lesions). Its under-40 count is not published.
   - No public cohort found has ≥ 50 under-40 histopathology-confirmed melanomas. *VERIFIED / SECONDARY.*

---

## 1. The ISIC Archive and how to query it

### 1.1 Query mechanics (VERIFIED by live calls)
- **Count:** `GET https://api.isic-archive.com/api/v2/images/search/?query=<Lucene>&collections=<ids>&limit=1` returns `count`. Pages hold at most 100 records, and `next` is a cursor.
- **Facets:** `GET https://api.isic-archive.com/api/v2/images/facets/?query=...&collections=...` returns buckets for `diagnosis_1…5`, `diagnosis_confirm_type`, `age_approx`, `image_type`, `dermoscopic_type`, `fitzpatrick_skin_type`, `copyright_license` and `anatom_site_special`, plus present/missing counts for `lesion_id` and `patient_id`.
- **Collections:** `https://api.isic-archive.com/api/v2/collections/?limit=200` lists 62 public collections.
- **`collections=a,b` returns the UNION** (65 → 451, 294 → 572, 65+294 → 675), so |A∩B| = |A| + |B| − |A∪B|.
- **Working query:** `age_approx:[0 TO 35] AND image_type:dermoscopic AND diagnosis_confirm_type:histopathology AND diagnosis_1:Malignant`. AK is `diagnosis_3:"Solar or actinic keratosis"` and is not under `Malignant`.
- **CLI:** `isic image download --search '…'` ([isic-cli](https://pypi.org/project/isic-cli/)). For bulk work, ISIC recommends its AWS Open Data snapshot ([notice](https://api.isic-archive.com/api/docs/swagger/)).
- **Caveat:** `lesion_id` links exist only where a contributor supplied them. "0 shared lesion IDs" means no *recorded* link, not proven independence. The MILK10k↔HAM links show that the archive does record links when they are given.

### 1.2 Totals (VERIFIED facets, 2026-11-01)
- **553,019 public images.** Dermoscopic 124,961; TBP tiles 401,059 (+16,990 overviews); clinical 9,748; RCM 260.
- **Dermoscopic confirmation:** histopathology 42,218, serial imaging 31,536, consensus 28,394.
- **Dermoscopic licences:** "CC-BY-NC" 55,986, "CC-BY" 49,024, "CC-0" 19,951.
- **Dermoscopic metadata:** Fitzpatrick on 9,255 images (V 614, VI 573).
- **Young dermoscopic images:** 27,650 (histopathology 5,790).
- **Young histopathology malignant images:** **1,047** (834 MEL, 194 BCC, 18 SCC) covering 399 distinct lesions, with 126 images unlinked. Young AK: 35 images (27 histopathology).

### 1.3 What ISIC 2019 is made of (DERIVED from union counts and joins)
- **Training (25,331 images, 12,264 lesions):**
  - HAM10000 train: 10,015
  - BCN20000 train: 12,413 images / 3,576 lesions
  - 2,903 images from the 2016/2017 challenge sets (MSK-1…5/UDA: 2,722; SONIC: 181)
- **Test (8,238 images, 3,232 lesions):** BCN20000 test 6,533 images / 1,864 lesions, plus HAM10000 val+test 1,705. Diagnoses are visible in the archive, although the BCN paper describes the test labels as private.
- **ISIC 2018 Task 1–2 train:** 2,593 of its 2,594 images are in 2019 Training (MSK). Its 1,100 val/test images are not.
- **Image overlap with 2019 Train+Test is 0** for ISIC 2020, MILK10k, HIBA, DERM12345, BALD, SLICE-3D, MSKCC Skin Tone, MEL-SELF, PROVe-AI, PAD-UFES-20 and iToBoS. SONIC shares 181 images.
- **Meta-collections overlap heavily:** ISIC-DICM-17K 6,147 of 17,060; ISIC Balanced 6,147 of 9,810; IMA++ 2,746 of 14,967; collection 294 11,665 of 18,133.

---

## 2. Dermoscopic dataset inventory

"Young esc" counts young histopathology-confirmed malignant images or lesions. "Overlap" is with ISIC 2019.

| Dataset | Size | Confirmation | Age; young esc | IDs | Device / polarisation | Population; FST | Licence (exact) | Access | Overlap with ISIC 2019 | Tag / source |
|---|---|---|---|---|---|---|---|---|---|---|
| ISIC 2016 / 2017 / 2018 T1–2 | 900+379; 2,000 (+150/600); 2,594+100+1,000 | mixed | yes | partial | mixed | US/EU | "CC-0" | open | 2,903 of 2,975 (2016/17); 2,593 of 2,594 (2018 T1–2 train) | VERIFIED [challenge data](https://challenge.isic-archive.com/data/), API |
| HAM10000 (2018 T3) | 10,015 + 1,511 test (+193 val) | HP, follow-up, consensus, confocal | yes | lesion_id | Vienna, Queensland | AT/AU | "Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)" | open, [Dataverse v4.0 (2023-02-07)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T); includes lesion masks | train in 2019 Train; val/test in 2019 Test | VERIFIED |
| BCN20000 (full) | 18,946 images / 5,583 lesions (archive links 5,440) | all melanoma and other malignant HP; benign by HP, confocal or 2-dermatologist consensus | 5-yr bins; **young: 517 images** (309 train / 208 test) | lesion_id; no patient_id | "three high-resolution cameras"; polarisation not stated | Barcelona 2010–16 | paper: "Creative Commons Attribution 4.0 International (CC-BY 4.0)"; archive tags all images "CC-BY-NC" (**conflict**) | figshare 10.6084/m9.figshare.24140028; ISIC 249 | train in 2019 Train, test in 2019 Test; **0 shared lesions** | VERIFIED ([EBI](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11183228/fullTextXML)), DERIVED |
| ISIC 2020 | 33,126 images / 32,701 lesions / 2,056 patients (+10,982 test, no GT) | 584 HP melanoma; benign HP (1,424) or ≥ 6-month follow-up / consensus | **young: 74 melanoma (74 lesions, 63 patients)**; 7,655 young images in 476 patients | patient_id, lesion_id | MoleMax HD; SLR/phone attachments; not recorded per image | MSKCC 11,108; UQ 8,449; BCN 7,311; Vienna 4,374; Sydney 1,884 | "Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)" (archive mix CC-BY / CC-BY-NC) | open | 0 images, 0 recorded lesions, 0 recorded patients; only Vienna is stated to have excluded earlier lesions; 425 internal duplicates | VERIFIED ([EBI](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7843971/fullTextXML)), DERIVED |
| SLICE-3D (ISIC 2024) | 401,059 tiles / 1,042 patients; Permissive subset 217,477 | 393 malignant, pathology within 3 months; most benign are weak labels | young: 10 malignant tiles | patient_id; lesion_id only on manual tags | **3D-TBP crops (~133 px), not dermoscopy**; 71% cross-polarised | 7 centres; no FST | full "CC-BY NC"; Permissive "CC BY 4.0" | open | 0 | VERIFIED ([EBI](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11324883/fullTextXML)) |
| MILK10k | 5,240 lesions × (dermoscopic + clinical); Benchmark 479 lesions (no GT) | 95.7% biopsied or excised | young: **72 escalating lesions** (MEL 33, BCC 35, SCCis 1, AK 2, other 1); 427 young lesions in total | lesion_id; prior-ISIC-image field; skin tone 0–5 | not stated | Vienna, Ankara, Queensland, MSKCC, Skopje | "CC-BY-NC" | open | 0 images, but **859 lesions linked to 2019 Train and 126 to 2019 Test (all HAM10000)**; 21 of the 72 young escalating lesions | VERIFIED ([DOI page](https://api.isic-archive.com/doi/milk10k/)), DERIVED |
| HIBA | 1,616 images / 1,246 lesions / 623 patients (1,270 dermoscopic) | 58.2% of lesions biopsied; all malignant biopsied | 5-yr bins, mean age 62.1; **young: 21 images, 20 lesions, 18 patients** | lesion and patient IDs | **contact polarised**: Scalar VL-7EX II, Canfield Mirror 7, FotoFinder, phone attachments | Buenos Aires; FST for 566 patients (I 44, II 451, III 66, IV 5) | "Creative Commons Attribution 4.0 International License" | open (ISIC 10.34970/587329) | 0 | VERIFIED ([EBI](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10584927/fullTextXML)) |
| DERM12345 | 12,345 images / 1,627 patients | 2-dermatologist consensus; "all malignant biopsy proven" (MEL 400, BCC 423, SCC 303, AK 58) | **no age**, so young count not derivable | patient_id only | MoleMax 3/HD, FotoFinder (contact polarised); DermLite DL4 + iPhone | Türkiye | "Creative Commons Attribution 4.0 International License" | open (Dataverse 10.7910/DVN/DAXZ7P; ISIC 399) | 0 | VERIFIED ([EBI](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11604664/fullTextXML)) |
| BALD (acral) | 666 images (144 MEL, 522 nevi) | all melanoma HP; 193 nevi excised | young: 10 melanoma | — | — | AT, GR, FR, IT, **JP**; European and Asian patients | "CC-BY-NC" | open (ISIC 410; [GitHub](https://github.com/kittler/BALD)) | 0 | VERIFIED |
| MSKCC Skin Tone | 4,879 images (3,678 dermoscopic) | benign only, clinical assessment | ages ≥ 30 | — | — | **FST I–VI, about 750–950 each** | "CC-BY" | open (ISIC 413) | 0 | VERIFIED (API) |
| SONIC | 9,251 nevi | consensus or serial imaging | ages 10–15 | — | — | paediatric | "CC-0" | open (ISIC 293) | 181 | VERIFIED (API) |
| PROVe-AI / MEL-SELF | 603 / 3,008 images | all HP / mostly consensus | 63 / 110 young images | — | — | — / AU | "CC-0" / "CC-BY" | open | 0 / 0 | VERIFIED (API) |
| UQ Longitudinal (2025) | 480 participants; 35,909 dermoscopic images; 250,162 tiles | HP for 1,267 lesions (30 MEL, 80 BCC, 48 SCC) | banded; 30 participants ≤ 35 | yes | VEOS, Canon T6i | Brisbane | "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)" | UQ eSpace 10.48610/a13deaf | **29.7% of dermoscopic images overlap ISIC 2020/2024** | VERIFIED ([EBI](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12484743/fullTextXML)) |
| **SCP2** (Germany) | 1,910 lesions / 1,716 patients; 6 dermoscopic images per lesion; 750 MEL, 885 nevi | HP; a 792-lesion subset has an 8-pathologist panel label plus WSIs | mean age 60.9; **test-set < 35: 177 lesions, 14 MEL**; panel subset < 35: 100 patients (4 invasive MEL, 2 MIS) | patient | 4 camera/dermatoscope set-ups; polarised and non-polarised | 8 German university hospitals, 2021–23 | not stated; panel subset is non-commercial only | **on request**: corresponding author or SCP Data Protection Committee | **not in ISIC** | VERIFIED ([Commun Med 2024](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11387610/fullTextXML), [Nat Commun 2025](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11742048/fullTextXML)) |
| **MRA-MIDAS** | 796 patients / 1,290 lesions / 3,830 images | HP for biopsied lesions (3 dermatopathologists) | age and FST (16.8% FST III–VI); invasive MEL n = 46 (SECONDARY); under-40 not published | patient | digital dermatoscope + iPhone at 15/30 cm | Stanford (public part) + Cleveland Clinic | not stated on page | AIMI → Redivis login | **not in ISIC** | AIMI page VERIFIED ([AIMI](https://aimi.stanford.edu/datasets/mra-midas-Multimodal-Image-Dataset-for-AI-based-Skin-Cancer)); counts SECONDARY ([NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIdbp2400732), [medRxiv](https://www.medrxiv.org/content/10.1101/2024.06.27.24309562.full.pdf)) |
| BOSQUE Test Set | 151 dermoscopic images | HP, ISIC 7-class | binned | — | — | Bogotá; FST II 11, III 94, IV 34, V 7, VI 5 | "CC0 1.0" | open ([Dataverse](https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/AQEPIN)) | not in ISIC | VERIFIED |
| PH2 | 200 images (40 MEL) | clinical + histological diagnosis | not stated | — | 768×560 | Portugal | not verified (site down) | registration | not in ISIC | SECONDARY ([page](https://www.fc.up.pt/addi/ph2%20database.html)) |
| Derm7pt | 1,011 cases (dermoscopic + clinical) | 7-point criteria + diagnosis | no age | — | — | likely the EDRA atlas | CC BY-NC-SA 4.0 (SECONDARY) | [derm.cs.sfu.ca](https://derm.cs.sfu.ca) | not in ISIC | SECONDARY ([arXiv 2309.06961](https://arxiv.org/pdf/2309.06961)) |
| SKINL2 / MCR-SL | 250–376 light fields / 240 lesions, 60 subjects (29 HP) | — | — | — | Raytrix plenoptic / — | Portugal / Norway | not found / CC-BY | — / [Zenodo](https://zenodo.org/records/17306338) | not in ISIC | SECONDARY |
| Masks | ISIC 2018 T2 attribute masks (2,594 images; pigment network, negative network, streaks, milia-like cysts, globules); HAM10000 lesion masks; ISIC 2018 Novice masks (11,720); IMA++ (14,967 images) | — | — | — | — | — | HAM "CC BY-NC 4.0"; others not stated | open | **T2 images are 2019 Training MSK rows**; IMA++ shares 2,746 images | VERIFIED ([T2](https://challenge.isic-archive.com/landing/2018/46/)), API |

Small or secondary items:
- Mexican BCC-only set: 460 images, 304 dermoscopic ([BMC 2025](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-025-02706-y)).
- DermPK, Pakistan: 157 images ([Mendeley](https://data.mendeley.com/datasets/fdhyjjypbd/2)).

No 2024–2026 public **East Asian, Indian or African dermoscopy** release was found. BALD (Japanese acral) and MSKCC Skin Tone (FST I–VI, benign only) come closest.

### Non-dermoscopic (routing, OOD or fairness only)

| Dataset | Key facts | Licence | Tag |
|---|---|---|---|
| PAD-UFES-20 | 2,298 images / 1,641 lesions / 1,373 patients; smartphone; Brazil | "CC BY 4.0" ([Mendeley](https://data.mendeley.com/datasets/zr7vgbcyr2/1)) | VERIFIED |
| Dermofit | 1,300 SLR clinical images (76 MEL) | £75 academic licence; no commercial or teaching use ([Edinburgh](https://homepages.inf.ed.ac.uk/rbf/DERMOFIT/datasets.htm)) | VERIFIED |
| MED-NODE | 170 non-dermoscopic images | — | SECONDARY |
| SD-198 / SD-260 | 6,584 clinical images / SD-260 unverified | — | SECONDARY |
| Fitzpatrick17k | 16,577 clinical images | "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License" | SECONDARY ([mirror](https://github.com/standardgalactic/fitzpatrick17k)) |
| DDI | 656 images / 570 patients; 171 malignant; FST balanced | terms not shown | VERIFIED counts ([AIMI](https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images)) |
| PASSION, SCIN, DDI-2, DermaCon-IN | — | — | SECONDARY ([compilation](https://github.com/AxDante/skin-lesion-dataset-analysis)) |

---

## 3. Under-40 escalating lesions after removing ISIC 2019 overlap (DERIVED from API joins)

| Source | Young esc images | **Lesions** | Lesions recorded in ISIC 2019 | **New lesions** | Status for you |
|---|---:|---:|---:|---:|---|
| ISIC 2019 Training (reference) | 451 | 147 (+26 unlinked images) | — | — | your 81 + reserved partition |
| **ISIC 2019 Test** | 215 (BCN 208, HAM 7) | 47 (BCN 42, HAM 5) | 0 in Train | **47** | **probably unused**; the 5 HAM lesions are official HAM test |
| MILK10k (young malignant + AK) | 72 | 72 | 21 (HAM) | **51** | reserved for the external read |
| ISIC 2020 | 74 | 74 (63 patients) | 0 recorded | 74 | likely already in your extras |
| HIBA | 21 | 20 (18 patients) | 0 | 20 | reserved for confirmation |
| Coll. 294 residual / MSK-1…5/UDA / other | 217 / 133 / 55 (overlapping) | about 65 new lesions in total, plus about 100 unlinked images | 12 | ~65 + ≤ 100 images | likely in your extras; the unlinked images match your "747 without lesion ID" |
| BALD / SLICE-3D | 10 / 10 | 10 / 10 | 0 | 10 / 10 | acral; TBP modality |
| Young AK, archive-wide | 35 images (21 in 2019 Train, 6 in Test, 2 in MILK10k) | — | — | ~6–8 | — |

**Ages 38–39 add-on (upper bound).** These sit in the 40 bucket, of which roughly 40% is under 40. Bucket-40 histopathology malignant dermoscopic images: 2019 Test 113, MILK10k 93, HIBA 11, ISIC 2020 25, 2019 Train 346.

**Bottom line.** Clean, unused young escalating lesions:
- About **47 from 2019 Test**.
- **51 from MILK10k** and **20 from HIBA**, once those leave their evaluation roles.

That is about **118 lesions**, against today's 81. Counting 38–39-year-olds adds roughly 50 more: about 10 from 2019 Test, about 37 from MILK10k (before removing HAM links) and about 4 from HIBA. This is a rough DERIVED estimate that takes 40% of the 40 bucket and assumes ages are spread evenly. Archive-wide, about 250 lesions are new relative to ISIC 2019, but most are probably already in your 2,078 extras. Outside ISIC, SCP2 adds about 14+ young melanomas (on request) and MIDAS an unknown number. **No public route gives 10× more under-40 escalating data**; collaboration or private data is the only other lever.

---

## 4. Unlabelled pools for self-supervised pretraining, with leakage

| Pool | Size | Contains your evaluation images? |
|---|---|---|
| Full archive / AWS snapshot | 553,019 total (124,961 dermoscopic) | **Yes**: HAM test, the reserved BCN partition, MSKCC rows, HIBA, MILK10k |
| Archive dermoscopic minus 2019 Train + Test | 91,392 (= 124,961 − 33,569) | **Yes, by lesion**: MILK10k has 985 HAM-linked lesions; 7,080 MSK/UDA same-institution images; 1,100 ISIC 2018 T1–2 val/test images. Remove HIBA and MILK10k if they are confirmation reads |
| ISIC 2020 train + test images | 33,126 + 10,982 | 0 images and 0 recorded lesions or patients shared with 2019 |
| SLICE-3D | 401,059 tiles | no (different modality) |
| UQ Longitudinal | 35,909 dermoscopic + 250,162 tiles | no 2019 overlap reported; ND licence (check whether derived models count as adapted material) |
| SONIC / MSKCC Skin Tone / DERM12345 / MEL-SELF | 9,251 / 4,879 / 12,345 / 3,008 | 181 / 0 / 0 / 0 |
| ISIC-DICM-17K / ISIC Balanced / IMA++ / coll. 294 | 17,060 / 9,810 / 14,967 / 18,133 | **Yes**: 6,147 / 6,147 / 2,746 / 11,665 images shared |
| Derm1M | 403,563 images, 10.1% from "public datasets" | probably (SECONDARY, [arXiv](https://arxiv.org/pdf/2503.14911)) |

**Rule:** exclude any image whose `isic_id`, **`lesion_id`** or `patient_id` touches a reserved, test or confirmation row. Removing by `isic_id` alone misses the 985 MILK10k↔HAM lesions.

---

## 5. External confirmation: different institution and not in the ISIC Archive

| Cohort | Size | Under-40 MEL | Access | Verdict |
|---|---|---|---|---|
| **SCP2** (8 German university hospitals) | 1,910 lesions, 750 MEL, dermoscopic | test < 35: 14 MEL (177 lesions) | on reasonable request; panel subset non-commercial | **best all-age external cohort**; fully disjoint |
| **MRA-MIDAS** | 1,290 lesions, prospective, paired | not published | AIMI/Redivis; terms not shown | strong US cohort; ask for age-stratified counts |
| BOSQUE | 151 images, HP, FST II–VI | unknown, tiny | CC0 1.0 | skin-tone probe |
| PH2 / Derm7pt | 200 / 1,011 | no age | registration / site | all-age sanity checks |
| SKINL2 / MCR-SL / Mexico BCC | small | unknown | open or mixed | negligible |

HIBA, DERM12345 and BALD are **disjoint from ISIC 2019 but inside the archive**. MILK10k is **not** disjoint, because it is HAM-linked. **Every option is underpowered for under-40 on its own.** Pre-register a pooled or meta-analysed young-band read with exact Clopper–Pearson intervals across HIBA (20) + MILK10k (51 clean) + SCP2 (≥ 14) + MIDAS (unknown).

---

## 6. Ranked data plan

| # | Source | Adds (images; young escalating) | Confirmation | Overlap / leakage | Licence | Effort | Role |
|---|---|---|---|---|---|---|---|
| 1 | **ISIC 2019 Test, BCN part** | 6,533 images / 1,864 lesions; **42 young lesions** (+5 HAM) | HP malignant | **Low**: 0 lesions shared with Train; keep the 5 HAM-test lesions out if they are your eval | archive "CC-BY-NC" (paper CC-BY 4.0) | low (API) | **Training** |
| 2 | MILK10k | 5,240 lesions; 51 clean young | 95.7% HP | **Medium**: 985 HAM-linked lesions to drop | "CC-BY-NC" | low | external read on the **4,255 unlinked lesions**, then training |
| 3 | SCP2 (Germany) | 1,910 lesions; ≥ 14 young MEL | HP + panel | **None** | request; non-commercial | high | **External confirmation** |
| 4 | MRA-MIDAS | 1,290 lesions; unknown | HP | None | terms to verify | medium | **External confirmation** |
| 5 | ISIC 2020 | 33,126; 74 young MEL; 7,609 young benign lesions in 476 patients | HP / follow-up | low recorded overlap | "CC BY-NC 4.0" | low | **Training**: benign diversity, patient context (check against extras) |
| 6 | DERM12345 | 1,167 HP malignant; no age | HP malignant | none | "Creative Commons Attribution 4.0 International License" | low | training (escalating diversity) or all-age confirmation |
| 7 | HIBA | 20 young lesions | 58% HP | none | "Creative Commons Attribution 4.0 International License" | low | **all-age confirmation**; under-40 only when pooled |
| 8 | Coll. 294 / MSK / UDA residual | ~65 lesions + ≤ 100 unlinked images | HP | medium (12 linked; same institution as reserved MSKCC) | CC-0 / mixed | low | training only if not already in extras; group by patient_id |
| 9 | BALD | 10 young acral MEL | HP | none | "CC-BY-NC" | low | training (acral) |
| 10 | SLICE-3D, SONIC, MSKCC Skin Tone, archive residual | 401k / 9k / 4.9k / ~91k | weak | strip linked lesions | mixed | low | **SSL pool**; Skin Tone set for FST slicing |
| 11 | UQ Longitudinal | 35,909 dermoscopic | partial HP | 29.7% overlap ISIC 2020/2024 | "CC BY-NC-ND 4.0" | medium | SSL only if ND permits |
| 12 | BOSQUE / PH2 / Derm7pt | small | HP / mixed | none | CC0 / ? / NC-SA | low | small probes |
| 13 | T2 / HAM / IMA++ masks | — | — | T2 images are 2019 Training MSK rows | mixed | low | auxiliary supervision on **training rows only** |
| 14 | PAD-UFES-20, Fitzpatrick17k, DDI, SD-198, Dermofit, MED-NODE, PASSION, SCIN | — | — | — | — | — | **Not usable** for dermoscopy |

### Not verified
- BCN20000's licence conflict.
- MIDAS licence and age-stratified counts.
- PH2, Derm7pt and SKINL2 licences.
- SD-260.
- Whether SCP2's full 1,910-lesion set (not just its test split) has more young melanomas.
- All lesion-independence claims rest on the archive's recorded `lesion_id`s.

The joins can be reproduced with `scratchpad\isic_fetch.ps1` (metadata CSVs in `scratchpad\isic_meta\`).
