<div align="center">

# Scan4Disease · Backend-S4D

### Pre-registered research on safe skin-lesion triage in dermoscopy, and on why it keeps failing patients under 40 ([duplicate-image disclosure](docs/LEAK_DISCLOSURE.md))

<img src="docs/assets/readme/hero.gif" width="100%" alt="Animated intro: accuracy is misleading on HAM10000; the best system reaches Macro-F1 0.8047 but catches 3 of 21 serious under-40 cases; five versions narrowed the cause to a ranking limit.">

<p>&nbsp;</p>

<p>
  <img src="docs/assets/readme/icons/python.svg" width="56" height="56" alt="Python" title="Python">&nbsp;&nbsp;&nbsp;
  <img src="docs/assets/readme/icons/pytorch.svg" width="56" height="56" alt="PyTorch" title="PyTorch">&nbsp;&nbsp;&nbsp;
  <img src="docs/assets/readme/icons/numpy.svg" width="56" height="56" alt="NumPy" title="NumPy">&nbsp;&nbsp;&nbsp;
  <img src="docs/assets/readme/icons/pandas.svg" width="56" height="56" alt="pandas" title="pandas">&nbsp;&nbsp;&nbsp;
  <img src="docs/assets/readme/icons/scikitlearn.svg" width="56" height="56" alt="scikit-learn" title="scikit-learn">&nbsp;&nbsp;&nbsp;
  <img src="docs/assets/readme/icons/matplotlib.svg" width="56" height="56" alt="Matplotlib" title="Matplotlib">&nbsp;&nbsp;&nbsp;
  <img src="docs/assets/readme/icons/opencv.svg" width="56" height="56" alt="OpenCV" title="OpenCV">&nbsp;&nbsp;&nbsp;
  <img src="docs/assets/readme/icons/powershell.svg" width="56" height="56" alt="PowerShell" title="PowerShell">&nbsp;&nbsp;&nbsp;
  <img src="docs/assets/readme/icons/git.svg" width="56" height="56" alt="Git" title="Git">&nbsp;&nbsp;&nbsp;
  <img src="docs/assets/readme/icons/latex.svg" width="56" height="56" alt="LaTeX" title="LaTeX">
</p>

**Lesion-grouped splits · HAM10000 test locked · SHA-256 pre-registered plans · V6 complete · V7 duplicate gate**

[V1 Baseline](#v1) · [V2 Rigour](#v2) · [V3 Validity](#v3) · [V4 Measurement](#v4) · [V5 Mechanism](#v5) · [V6 Result](#v6) · [V7 Duplicate gate](#v7)

**Team 193 · Project Exhibition · School of Computing Science and Engineering, VIT Bhopal University**

[Programme](#the-programme-at-a-glance) · [What was established](#what-five-versions-established) · [Deployed system](#the-deployed-system) · [Integrity](#research-integrity) · [Repository](#repository-layout) · [Reproduce](#getting-started) · [Team](#team) · **[Navigate the repo →](NAVIGATION.md)**

</div>

---

## The question

Skin-lesion classifiers are usually judged on accuracy, which misleads here: **66.9%** of HAM10000's 10,015 images are benign nevi, so a model that always answers "benign mole" scores about 67% while missing every melanoma. This project asks whether a classifier can be trusted as a **triage tool**: does it send the lesions that need a specialist (melanoma, basal-cell carcinoma, actinic keratosis) to one, for every patient group?

| | |
|---|---|
| **Task** | 7-class dermoscopy diagnosis; escalate {MEL, BCC, AKIEC} |
| **Metrics** | Macro-F1, balanced accuracy, escalation sensitivity, under-40 escalation pAUC. **Never accuracy** |
| **Cohorts** | HAM10000, BCN20000, MSKCC, PAD-UFES-20 (smartphone), ISIC Archive (train-only young data), MILK10k (external, read once) |
| **Central finding** | Under-40 escalation sensitivity of **0.143** (3 of 21 images, from only 10 lesions) against about 0.78 in older bands, with *confident* misses. Five versions narrowed the cause to a **ranking limit in the learned representation** |

---

## The programme at a glance

<img src="docs/assets/readme/programme.gif" width="100%" loading="lazy" alt="Animated timeline V1 to V5: the question each version asked and the answer it got.">

Each version was started by the question its predecessor left open, and each plan was frozen and hashed before the runs that tested it. The record, session by session, is in [`CHANGELOG.md`](CHANGELOG.md).

<sub>The GIFs on this page are of two kinds: charts and step strips recorded from the project's Review 2 infographic, and big-type scenes whose numbers are read directly from `results/` by [`scripts/readme_media/`](scripts/readme_media/).</sub>

---

<a id="v1"></a>

## V1 · The baseline

<img src="docs/assets/readme/v1.gif" width="100%" loading="lazy" alt="V1 reel: six-step pipeline, the ablation ladder, calibration raising Macro-F1 while missed serious cases rise from 62 to 78, the abstention simulator, and the under-40 gap.">

- **Only ensembling cleared significance:** the 6-CNN soft vote beats ConvNeXt-Tiny with McNemar χ² 17.20, p = 3.4 × 10⁻⁵ (85 vs 38 discordant cases).
- **The ensemble is under-confident, not over-confident,** which is why the class-wise Dirichlet map beats temperature scaling (soft-vote test ECE 0.1575 → 0.0206).
- **Better calibration cost escalations:** on the TTA stack, Macro-F1 rises 0.7859 → 0.8047 while missed serious cases rise 62 → 78.
- **Under 40, the misses are confident:** abstention refers only 11.1% of them.

<details>
<summary><b>V1 ablation ladder and age bands</b> (HAM10000 test, 1,502 images, 290 escalating cases)</summary>
<br>

Lesion-grouped 1,000× bootstrap CIs. Rungs were **selected on validation**.

| Rung | System | Macro-F1 [95% CI] | Balanced acc. | Escalation sens. | Missed serious |
|---|---|:---:|:---:|:---:|:---:|
| A1 | ResNet-50 | 0.7058 [0.647, 0.750] | 0.7217 | 0.7379 | 76 |
| A2 | **ConvNeXt-Tiny** (best single CNN on val) | 0.7459 [0.693, 0.781] | 0.7792 | 0.7759 | 65 |
| A3 | SwinV2-Tiny | 0.7273 [0.668, 0.777] | 0.7116 | 0.6621 | 98 |
| A4 | Gated metadata fusion | 0.7411 [0.684, 0.787] | 0.7433 | 0.6862 | 91 |
| A5 | **6-CNN uniform soft vote** | 0.7718 [0.720, 0.812] | 0.7923 | 0.7828 | 63 |
| A6 | + 24-view TTA | 0.7859 [0.734, 0.824] | 0.8102 | **0.7862** | **62** |
| A7 | + Dirichlet calibration | **0.8047** [0.757, 0.841] | 0.7939 | 0.7310 | 78 |

MaxViT-Tiny scores 0.7525 on test but 0.7176 on val, so it was not selected.

| | < 40 | 40–59 | ≥ 60 |
|---|:---:|:---:|:---:|
| Escalating share of training images | **4.9%** | 12.8% | 35.5% |
| Test escalation sensitivity | **0.143** (3 / 21; 10 lesions) | 0.786 (55 / 70) | 0.774 (154 / 199) |
| Misses referred by margin abstention | **11.1%** (2 / 18) | 33.3% (5 / 15) | 37.8% (17 / 45) |

The test figure rests on 21 cases. The better-powered OOF estimate is 0.547 (35 / 64).

<sub>Sources: `results/ablation_table.csv`, `results/mcnemar_delong.json`, `results/review2/v1_member_metrics.csv`, `results/age_band_prior.csv`, `research/selective/results/session4_report.md`, `research/stats/results_oof/age_gap_intervals.csv`.</sub>
</details>

---

<a id="v2"></a>

## V2 · Is it the decision threshold?

<img src="docs/assets/readme/v2.gif" width="100%" loading="lazy" alt="V2 reel: six-step pipeline, the compression gap spanning zero in every cohort, coverage versus protection, and the six hypothesis verdicts.">

- **No.** The compression gap spans zero in all three cohorts, so the miss sits upstream of the decision rule.
- **The only GO: coverage is not protection.** Marginal conformal coverage reaches 95.07%, yet false reassurance on serious lesions is 8.64%, above the 4.93% ceiling. Bipartite conformal cuts under-40 false reassurance from 6/29 to 1/29 (Holm p = 0.044).

<details>
<summary><b>V2 numbers</b></summary>
<br>

| Cohort (under 40) | Compression gap C | 95% CI | Escalating |
|---|:---:|:---:|:---:|
| HAM OOF | −0.0312 | [−0.0877, +0.0256] | 64 |
| BCN20000 | +0.0054 | [−0.0058, +0.0137] | 369 |
| PAD-UFES-20 | 0.0000 | [0.0000, +0.0256] | 74 |

<sub>Sources: `results/v2/bootstrap_intervals.json`, `results/v2/final_verdict.json`, `results/v2/frr_by_group.csv`, `results/v2/conformal_subgroup_safety.csv`.</sub>
</details>

---

<a id="v3"></a>

## V3 · Can it be broken?

<img src="docs/assets/readme/v3.gif" width="100%" loading="lazy" alt="V3 reel: six-step pipeline, the in-sample probe collapsing from 0.9930 to 0.7182, same-data seed noise exceeding the effect, removing age making under-40 ranking worse, and six hypothesis verdicts.">

- **An inflated lead was caught:** V2's probe fell from 0.9930 to 0.7182 on out-of-sample features, below the deployed posterior (0.8096).
- **The endpoint was noisier than the effect:** two models on byte-identical data caught 14/22 and 9/22.
- **Age is load-bearing:** removing it moved Macro-F1 from 0.7421 to 0.6000 and under-40 AUC from 0.8249 to 0.7201.

<details>
<summary><b>V3 hypotheses</b> (zero test reads)</summary>
<br>

| # | Claim | Evidence | Verdict |
|---|---|---|:---:|
| H1 | N2's head result survives out-of-sample features | 0.9930 → 0.7182 | Falsified |
| H2 | A better head recovers the gap | overall −0.0131 | Falsified |
| H3 | The representation is age-entangled | age-band AUC 0.6922 [0.6638, 0.7187] | **Certified** |
| H4 | Pooling archives adds ≥ +0.03 Macro-F1 | −0.0088 [−0.0554, +0.0354] | Falsified |
| H5 | Breadth buys unseen-archive robustness | 0 of 2 zero-shot contrasts | Falsified |
| H6 | Removing age helps under 40 | under-40 AUC −0.1049 | Falsified |

Best condition 0.7869: the 0.80 ceiling was not broken.

<sub>Sources: `results/v3/final_verdict.json`, `results/v3/oos_probe_report.json`, `results/v3/s44_control_and_multiplicity.json`, `results/v3/d2_strong_evaluation.json`.</sub>
</details>

---

<a id="v4"></a>

## V4 · Measure it properly

<img src="docs/assets/readme/v4.gif" width="100%" loading="lazy" alt="V4 reel: six-step pipeline, the power audit, the V1 system on different rulers, foundation models, the safety contract failing, and the ranking limit.">

- **The question became measurable:** "3 of 21" was 10 lesions; a reserved cohort of 4,733 images now holds 104 under-40 escalating lesions.
- **In-domain data is the big lever:** on the same reserved rows, Macro-F1 goes from 0.411 (V1) to 0.603 (pooled control), confounded by corpus and labelled so.
- **It is a ranking limit:** under-40 AUC 0.878 against 0.948 and 0.929. Frozen foundation models do not rank under-40 better, and the pre-declared safety contract meets 0 of 9 terms.
- **One layer survives:** S56 per-age-band abstention is deployed.

<details>
<summary><b>V4 numbers</b></summary>
<br>

| Read | Result |
|---|---|
| Gate A: composite vs V1, reserved (confounded by corpus) | Macro-F1 +0.1798 [+0.1297, +0.2254] |
| Gate B: composite vs control, reserved | outcome 4, "neither": Macro-F1 −0.0121 [−0.0321, +0.0098], under-40 pAUC +0.0089 [−0.0200, +0.0367] |
| S51: PanDerm − ConvNeXt, under-40 pAUC | −0.0363 [−0.0828, +0.0127]; falsifier fired |
| S59: safety contract, reserved | 0 of 9 terms; sensitivity 0.7634 / 0.7040 / 0.6744 vs a 0.855 floor; referral 0.386 vs a 0.25 cap |
| S56: per-band vs global abstention at a 20% budget | under-40 system sensitivity +0.154 [+0.104, +0.216]; under-40 referral 0.232 → 0.454 |

<sub>Sources: `results/v4/power_audit.json`, `results/v4/s54/s54_gate.json`, `results/v4/backbone_probe_deltas.csv`, `results/v4/s59/s59_report.json`, `results/v4/s56/contrasts_reserved.csv`, `results/v4/s64/ceiling.json`, `paper/v4/v4_findings.md`.</sub>
</details>

---

<a id="v5"></a>

## V5 · Dermatologist reasoning

<img src="docs/assets/readme/v5.gif" width="100%" loading="lazy" alt="V5 reel: module funnel, fold-0 screens, the held-out confirmation, four frozen gates, seed slopes, and the decomposition of the under-40 gain.">

Each module turned a bedside heuristic into code and had to pass a screen and a falsifier written before its result existed. Three survived: **two-step triage heads, melanoma-vs-look-alike ranking, and young biopsy-proven data**, on an ImageNet-22k trunk.

> **Correction, 5 Nov 2026.** A leak audit found that V5's young extras contained images, or other photos, of lesions in the rows it was scored on: 90 rows, including 26 that are the same image under another name, such as `ISIC_0012503_downsampled` and `ISIC_0012503`. V5 trained on the extras and its control did not, so this favoured V5. The numbers below are corrected, with the original read beside them. The reels show the original read. Sources: [`confirm_s42_s43_s44_without_copies.json`](results/v5/confirm_s42_s43_s44_without_copies.json), [`leak_audit_keypoints.json`](results/v6/screens/leak_audit_keypoints.json).

- **No established under-40 gain:** +0.0136 [−0.0084, +0.0357], p = 0.228, on 58 under-40 escalating lesions, against a +0.050 MCID. The original read was +0.0351 [+0.0125, +0.0596], p = 0.003. Gate A fails either way.
- **The modules and young data no longer stand out:** +0.0068 [−0.0183, +0.0290], p = 0.612 (originally +0.0270 [+0.0045, +0.0488], p = 0.023). The trunk swap alone: +0.0068, p = 0.576 (originally +0.0081, p = 0.432).
- **All-age ranking did not move:** +0.0012, 1 of 3 seeds positive (originally +0.0020).

### How the 12,235 was earned

Every image is scored once, by a model that never saw it or any other photo of the same lesion.

<img src="docs/assets/readme/v5_fold.gif" width="100%" loading="lazy" alt="The fold machine animation: 15,294 images in a cocoon open, split by lesion, into a butterfly whose body is fold 0 and whose wings are folds 1 to 4; each wing is held out in turn, over three seeds.">

### The external read: MILK10k

One execution, one label read, receipted. Before any stage ran, lesions already in ISIC 2019 were excluded (deviation D6).

<img src="docs/assets/readme/v5_milk10k.gif" width="100%" loading="lazy" alt="The S84 MILK10k read: 4,255 lesions analysed; Macro-F1 0.474 vs 0.374; under-40 primary not met at 49 of 50 vs 47 of 50; the safety contract fails on referral.">

- **V5 classifies better than V1 on the outside cohort:** Macro-F1 0.474 vs V1's 0.374, Δ +0.100 [+0.074, +0.127], non-inferior at the −0.02 margin. The gain is relative: absolute Macro-F1 is lower than the 0.7005 measured on development folds.
- **The under-40 primary was not met:** 49/50 vs 47/50 at a matched 0.52 referral, McNemar p = 0.3125. Both systems sit at the ceiling, and only 50 of the planned 71 lesions remained after D6. This is a lack of resolution, not evidence that the systems are equal.
- **Copies in MILK10k (checked 5 Nov 2026):** 89 of the 4,255 analysed lesions match images V5 or V1 trained on, 86 of them in HAM10000. None is an under-40 escalating lesion, so the primary is unchanged. Without them Macro-F1 is 0.450 vs 0.344, Δ +0.106 [+0.078, +0.134]. The labels were re-read for this, by owner decision; every decision is the stored one ([`s84_without_copies.json`](results/v5/s84/s84_without_copies.json)).
- **The safety contract fails on referral:** sensitivity 0.960 clears the floor (lower bound 0.863), but referral is 0.463 against a 0.25 cap. MILK10k is 81.4% escalating, against 34.8% in the development pool.
- **How MILK10k is used now:** it is spent as a test set. It becomes V6 training data without the 985 linked lesions, and HIBA stays untouched as V6's confirmation cohort.

<details>
<summary><b>V5 numbers</b>: confirmation, decomposition, screens, MILK10k</summary>
<br>

**Three-seed confirmation, corrected** (locked composite vs control, held-out folds 1–4, without the 90 rows): 12,145 images, 6,921 lesions, 58 under-40 escalating lesions, 2,000 hierarchical bootstrap resamples.

| Gate | Δ (composite − control) | 95% CI | p | Result |
|---|:---:|:---:|:---:|:---:|
| **A** · under-40 escalation pAUC | **+0.0136** | [−0.0084, +0.0357] | 0.228 | Not met (MCID +0.050) |
| **B** · Macro-F1 retention | +0.0107 | [−0.0012, +0.0233] | 0.070 | Pass |
| **C** · older-band sensitivity at spec 0.80 | +0.0062 (40–59), −0.0014 (60+) | [−0.0112, +0.0222], [−0.0106, +0.0088] | 0.536, 0.907 | Pass |
| **D** · all-age escalation pAUC | +0.0012 | [−0.0061, +0.0082] | 0.790 | Not met |

**The original read** (12,235 images, 7,002 lesions, 65 under-40 escalating lesions, 2,000 hierarchical bootstrap resamples; frozen in [`confirm_s42_s43_s44.json`](results/v5/confirm_s42_s43_s44.json)):

| Gate | Δ (composite − control) | 95% CI | p | Result |
|---|:---:|:---:|:---:|:---:|
| **A** · under-40 escalation pAUC | **+0.0351** (+0.0136 [−0.0084, +0.0357], p 0.228, without the 90 rows with a copy in V5's training data; [disclosure](docs/LEAK_DISCLOSURE.md)) | [+0.0125, +0.0596] | 0.003 | Not met (MCID +0.050) |
| **B** · Macro-F1 retention | +0.0107 | [−0.0015, +0.0225] | 0.095 | Pass |
| **C** · older-band sensitivity at spec 0.80 | +0.0071 (40–59), −0.0007 (60+) | [−0.0127, +0.0234], [−0.0101, +0.0083] | 0.522, 0.872 | Pass |
| **D** · all-age escalation pAUC | +0.0020 | [−0.0056, +0.0093] | 0.596 | Not met |

**Decomposition** (Q11, deviation D4), original read. Corrected: modules + young data +0.0068 [−0.0183, +0.0290], p = 0.612; trunk +0.0068 [−0.0161, +0.0313], p = 0.576.

| Contrast | Under-40 pAUC | p | Macro-F1 | p |
|---|:---:|:---:|:---:|:---:|
| Modules + young data (same IN-22k trunk) | +0.0270 [+0.0045, +0.0488] | 0.023 | +0.0025 [−0.0073, +0.0118] | 0.649 |
| IN-22k trunk (same recipe) | +0.0081 [−0.0127, +0.0301] | 0.432 | +0.0082 [−0.0045, +0.0191] | 0.188 |

**MILK10k** (S84; 5,220 eligible, 965 excluded, 4,255 analysed, 4,148 in scope):

| Endpoint | V5 | V1 | Result |
|---|:---:|:---:|:---:|
| Primary: under-40 sensitivity at matched referral | 0.980 (49 / 50) | 0.940 (47 / 50) | Not met, p = 0.3125 |
| Key secondary: sensitivity ≥ 0.855 and referral ≤ 0.25 | 0.960, referral 0.463 | | Not met |
| Global Macro-F1 | 0.474 | 0.374 | Non-inferior, Δ +0.100 [+0.074, +0.127] |
| Global Macro-F1 without the 89 copies | 0.450 | 0.344 | Non-inferior, Δ +0.106 [+0.078, +0.134] |
| Escalation AUROC, all ages | 0.831 | 0.782 | Descriptive |
| Under-40 escalation pAUC | 0.747 | 0.767 | Descriptive, V1 leads |

**24-view TTA on the V5 composite** (descriptive, common OOF, folds 1–4): Macro-F1 0.6911 → 0.7005.

<sub>Sources: `results/v5/confirm_s42_s43_s44.json`, `results/v5/q11_decomposition.json`, `results/v5/review2_tables.md`, `results/v5/composite_lock.json`, `results/v5/s84/s84_report.json`, `results/v5/s84_receipt.json`, `results/v5/protocol_deviations.json`, `results/v5/v5_oof_common_matrix.json`.</sub>
</details>

---

## What five versions established

| Moved the numbers | Effect | Source |
|---|---|---|
| Training in domain | Macro-F1 0.411 → 0.603 on the same reserved rows (confounded by corpus) | V4 |
| Ensembling | 0.7459 → 0.7718, McNemar p = 3.4 × 10⁻⁵ | V1 |
| Per-band abstention | under-40 system sensitivity +0.154 [+0.104, +0.216], paid for with referral | V4 |

| Ruled out | Evidence | Source |
|---|---|---|
| Decision compression | every interval spans 0 | V2 |
| A removable age shortcut | under-40 AUC −0.1049 when removed | V3 |
| Frozen foundation backbones under 40 | PanDerm −0.0363 [−0.0828, +0.0127] | V4 |
| Hand-built colour, geometry, zoom and prototype modules | failed a screen or a falsifier written in advance | V5 |

---

## The deployed system

<img src="docs/assets/readme/system.gif" width="100%" loading="lazy" alt="The deployed stack: dermoscopy image, six CNNs, soft vote, 24-view TTA, Dirichlet calibration, then S56 per-age-band abstention choosing between automated triage and referral.">

| Layer | Status | Evidence |
|---|---|---|
| Soft vote + TTA + Dirichlet | Deployed since V1 | `results/ablation_table.csv` |
| S56 per-band abstention | Deployed (V4) | `results/v4/s56/contrasts_reserved.csv` |
| Age-conditional λ rule, conformal layer | Not deployed: no gain over S56 alone at matched workload (combined policy +0.004 [−0.014, +0.025]) | `results/v4/s59/s59_report.json`, `results/v4/s65/contrasts_reserved.csv` |

---

<a id="v6"></a>

## V6 · Complete: a better system, and the under-40 gap still open

V6 is a **data-maximal ensemble across pretraining families, image-only**, frozen on 4 Nov 2026: a uniform vote of ConvNeXt-Tiny (IN-22k) at 224 and 384 px, PanDerm-B and DINOv2 ViT-B/14, chosen on fold 0 and read once on folds 1–4 (12,235 rows) against V5's 15-model system, both with 24-view TTA.

| Endpoint (folds 1–4) | V6 vs V5 | Interval | Verdict |
|---|---|---|---|
| Macro-F1 | 0.7433 vs 0.7005, +0.0427 | [+0.0245, +0.0618] | Pass |
| All-age pAUC | 0.8890 vs 0.8553, +0.0337 | [+0.0263, +0.0415] | Pass |
| Escalation sensitivity (no harm) | +0.0115 | [+0.0005, +0.0231] | Pass |
| **Gate A** · under-40 pAUC | +0.0381 | [+0.0107, +0.0711] | **Not met** (MCID +0.050) |

The V6 vote replaces V5 as the deployed system. It passes on all rows, without the 380 at-risk rows and without the 426, under a rule declared before any fold 1–4 prediction existed ([`leak_sensitivity_read.json`](results/v6/leak_sensitivity_read.json)). Its under-40 gain is real but smaller than the clinically meaningful difference fixed in advance, so the under-40 question stays open. On HIBA, a new hospital read once under a receipt ([`hiba_report.json`](results/v6/hiba/hiba_report.json)), it scores Macro-F1 0.7348 against V5's 0.5715 (+0.1633 [+0.1235, +0.2064]) on 1,199 lesions and catches 18 of 20 under-40 escalating lesions; 20 is too few to certify the under-40 band. Sources: [`e_primary.json`](results/v6/e_primary.json), plan [`docs/V6_RUNSHEET.md`](docs/V6_RUNSHEET.md).

**Leak audit, 5 Nov 2026.** 380 development rows (2.5%) have a copy or another photo of the same lesion in training data, found by name and by keypoint matching. A second audit's lesion, patient and pixel links extend that to 426 (2.8%). The frozen training rows stay as they are, because V5 trained on the same extras. Instead, a system must pass on all rows, without the 380 and without the 426, declared before any V6 prediction on folds 1–4 existed ([`leak_sensitivity_declaration.json`](results/v6/leak_sensitivity_declaration.json)). Every fold-0 decision comes out the same without them ([`decisions_without_at_risk_rows.json`](results/v6/screens/decisions_without_at_risk_rows.json)). Plan: [`docs/V6_RUNSHEET.md`](docs/V6_RUNSHEET.md), Part 0.

---

<a id="v7"></a>

## V7 · The duplicate problem, and the gate that now guards every run

**The problem.** Every split in V1–V6 was grouped by `lesion_id`, and no ID crosses a split. But the same image, or another photo of the same lesion, can sit on both sides **under a different name or ID**, and no ID check can see that. A pixel audit found such rows in every version, from 0.01% to 1.47% of the rows read. Each is disclosed with its effect in [`docs/LEAK_DISCLOSURE.md`](docs/LEAK_DISCLOSURE.md); V5's under-40 gain shrank once its copies were removed (see V5 above).

**The fix.** V7 will not train without a receipt that binds one master table of every image it can touch (57,488) to one link table, checked by four layers ([`research/data/`](research/data/), [`research/v7/m1/`](research/v7/m1/)):

1. **Identity:** the same file, lesion ID, patient ID, normalised name, SHA-256 or image hash.
2. **Appearance search:** every image against all others with SSCD, a copy-detection network; a neighbour above a threshold fixed on held-out data becomes an *at-risk* link.
3. **Geometry:** SIFT keypoints must agree (at least 30 RANSAC inliers) for a link to become *confirmed*, so two different detectors agree.
4. **The gate:** any training row linked to an evaluated or never-train row is removed. Today that is 3,424 rows from all training and 305 to 349 more per held-out fold, from 20,370 confirmed and 123,471 at-risk links ([`data_gate_report.json`](results/v7/data_gate_report.json)). Planted copies, eight kinds, are found 384 of 384 times on every build.

**How well it works, measured** ([`verify.json`](results/v7/m1/verify.json), [`baselines.json`](results/v7/m1/baselines.json)). The endpoints and the adoption rule were written to [`protocol.json`](results/v7/m1/protocol.json) before any matcher was scored.

| Check | Before V7 (DINOv2 retrieval) | V7 gate (SSCD + SIFT) |
|---|---|---|
| Known copies, 465 pairs found by other detectors | 72.3% | **89.9%**, exact copies 100% |
| Re-photographs, 1,039 held-out lesions, 1 false match in 100,000 pairs | 22.5% (with SIFT) | **26.3%** |
| Re-photographs in rows no matcher used, 13,489 pairs | 6.0% | **27.7%** [24.1%, 31.5%] |

It found **8 HAM-test images** with a copy that every earlier audit missed, one of them copied in V1 and V4 training data, and **no copy** of a training image in the sealed external cohort (MIDAS, pixels only, labels unread). Two matchers trained here scored higher overall (29.6% and 28.7%) but lower on HAM, so the rule fixed in advance kept SSCD.

**What it does not fix.** Most re-photographs of one lesion, taken on another day or at another angle, are still missed, and an at-risk link is right only about half the time by eye. The gate therefore removes some rows that were never leaks; that is the stated cost of a missed leak being worse than a lost training row. The repository's claim is **"audited, with stated lower bounds"**, never "leak-free".

**What V7 tests next.** With the gate in place, V7's pre-registered plan returns to the under-40 question: resolution above 224 px, learning-rate fixes for foundation backbones, concept guidance and dark young melanomas, each read on all rows and again without every gate-linked development row. The plan is under review and not yet signed: [`V7_FINAL_MASTER_RESEARCH_PLAN.md`](docs/v7_final_revision/V7_FINAL_MASTER_RESEARCH_PLAN.md).

---

## Research integrity

These rules are enforced in code, not left to convention. Breaking any one of them invalidates an experiment.

1. **Lesion-grouped splits.** Every split groups images by `lesion_id` (10,015 images on 7,470 lesions); `assert_no_leakage()` must pass.
2. **Locked test.** The HAM10000 test split was read under one frozen plan on 4 Sep 2026; `results/test_pass_receipt.json` stands at **2 executions** (two stages of that plan), and `research/testguard.py` locks the split process-wide.
3. **Read-once external cohorts.** `results/v5/s84_receipt.json` records 1 execution and 1 label read of MILK10k.
4. **Nothing hand-entered.** Every number comes from `results/`. Manuscript claims and this README are re-checked against it by audit scripts.
5. **No accuracy for selection.** Model, weight, threshold and temperature choices are made on validation or out-of-fold data, never on test, and never by accuracy.
6. **Pre-registration before data.** Plans are SHA-256-hashed before the runs that test them (`results/v3/analysis_plan_v3.json`, `results/v4/analysis_plan_v4.json`, `results/v5/v5_plan_freeze.json`). Deviations are numbered and logged before the data they concern (`results/v5/protocol_deviations.json`).
7. **Honest intervals.** Exact Clopper–Pearson intervals for small-count proportions, a lesion-grouped bootstrap for everything else, Holm correction inside each declared comparison family (`research/stats/`). An interval that contains the null is reported as *not certified*, never as evidence of absence.

---

## Repository layout

```text
Backend-S4D/
├── ml/                     Core training and evaluation engine
│   ├── preprocessing/      Lesion-grouped splitter, transforms, dataset loaders
│   ├── training/           Two-stage fine-tuning (AdamW, cosine schedule, class-balanced loss)
│   ├── evaluation/         Metrics (Macro-F1, balanced accuracy, escalation sensitivity)
│   ├── explainability/     Grad-CAM
│   └── configs/            Class mapping, training config, frozen split_v1.csv
├── research/               One package per research question
│   ├── ensembling/ tta/ calibration/ selective/ conformal/ agerule/ dca/   V1 phases 1–4
│   ├── stats/ ablation/ external/ xdomain/                                V1 statistics, audits, external cohorts
│   ├── v2/ v3/ v4/ v5/ v6/                                                The post-Review-1 programme
│   ├── review2/            Generator for the Review-2 fact tables
│   └── experiments.csv     Append-only ledger of every run
├── results/                Every number and figure (JSON / CSV / Markdown), per version
├── paper/                  Manuscripts (LaTeX), tables and figures
├── docs/                   Research plans, runsheets and the research log, in time order
├── scripts/                Environment check, dataset download, run launchers, GPU benchmark, README media
├── tests/                  Unit and contract tests (V4 service, V5 modules)
└── CHANGELOG.md            Session-by-session record: what was done, why, and the source of every new number
```

Model checkpoints and the image datasets are **not** in the repository (size and licence); see below.

---

## Getting started

### 1 · Environment

```bash
git clone https://github.com/Rajrup910/Backend-S4D.git
cd Backend-S4D
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r ml/requirements.txt
python scripts/verify_env.py
```

PyTorch must come from the CUDA 12.8 index: the default PyPI wheels do not support the RTX 50-series (sm_120) GPU this work was run on.

### 2 · Data

Download the archives from their official sources and place them under `data/` (git-ignored): **HAM10000** (Harvard Dataverse), **BCN20000** and **MSKCC** (ISIC Archive), **PAD-UFES-20** (`python scripts/download_pad_ufes.py`). The frozen V1 split is `ml/configs/splits/split_v1.csv`; the V4/V5 five-fold partition is `results/v4/kfold/fold_assignments.csv`.

### 3 · Verify the record

```bash
python -m research.ablation.audit_manuscript      # V1 manuscript claims vs results/
python -m research.v4.audit_v4 --check            # V4 splits, receipts and numerical claims
python -m research.v5.adopt_e1 --verify           # V5 pre-registration hashes
python scripts/readme_media/audit_readme.py       # every number and link on this page vs results/
pytest tests -q                                   # unit and contract tests
```

None of these read the test split.

---

## Documentation

| Read this | For |
|---|---|
| [`NAVIGATION.md`](NAVIGATION.md) | Fast lookups: where each piece of code, result and number lives |
| [`docs/README.md`](docs/README.md) | Every plan, runsheet and log, in the order it was written |
| [`CHANGELOG.md`](CHANGELOG.md) | The full narrative: each session, its decision and the file behind each number |
| [`results/README.md`](results/README.md) | How the V1 result files are generated |
| [`paper/v4/v4_findings.md`](paper/v4/v4_findings.md) | What V4 settled |
| [`results/v5/review2_tables.md`](results/v5/review2_tables.md) | Generated V5 tables (screens, confirmation, decomposition) |
| [`scripts/readme_media/`](scripts/readme_media/) | How the GIFs on this page are built |

---

## Team

| Member | Registration | Responsibility |
|---|---|---|
| **Prateek Chhabra** | 23BAI10169 | Ensembling, TTA and inference |
| **Rajrup Roy Chowdhury** | 23BAI10213 | CNN backbones, calibration, V5 confirmation (corresponding author) |
| **Aditya Srivastava** | 23BAI10303 | Vision transformers and foundation-model probes |
| **Kanak Pravin Sonare** | 23BAI11369 | Multi-cohort data and the lesion-grouped split guard |
| **Manishka Gupta** | 23BAI11303 | Metadata fusion and the dermatologist decision flow |
| **Uddhav Gupta** | 23BAI10146 | Statistical audits and conformal guarantees |

**Supervisor:** Dr. Paras Jain, Associate Professor, SCSE, VIT Bhopal University.

### Citation

```bibtex
@misc{backend_s4d_2026,
  title  = {Scan4Disease: Pre-Registered Evaluation of Safe Skin-Lesion Triage and the Under-40 Blind Spot},
  author = {Chhabra, Prateek and Roy Chowdhury, Rajrup and Srivastava, Aditya and
            Sonare, Kanak Pravin and Gupta, Manishka and Gupta, Uddhav},
  year   = {2026},
  howpublished = {\url{https://github.com/Rajrup910/Backend-S4D}},
  note   = {VIT Bhopal University, Team 193}
}
```

<sub>Research software for study purposes. It is not a medical device and must not be used for clinical decisions.</sub>
