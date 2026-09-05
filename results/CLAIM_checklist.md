# CLAIM 2024 checklist

**CLAIM** — Checklist for Artificial Intelligence in Medical Imaging. Originally Mongan,
Moy & Kahn Jr., *Radiology: Artificial Intelligence* 2020;2(2):e200029 (an RSNA journal);
this document follows the **2024 update**, Tejani et al., *Radiology: Artificial
Intelligence* 2024;6(4):e240300, which expands the checklist to **44 items**.

**On wording.** Item topics below are paraphrased in one line each. The authoritative
wording is in the cited paper and is not reproduced here. This file records *where in this
project each item is addressed* and, where an item is not met, says so plainly rather than
leaving it blank — a checklist whose only function is to display ticks is worth nothing to
a reviewer.

**Status vocabulary**

| Status | Meaning |
|---|---|
| **Met** | Addressed in the manuscript or a `results/` artifact, cited in the right-hand column. |
| **Partial** | Addressed, but with a stated limitation that a reviewer should see. |
| **Not met** | Not done. The reason is given. |
| **N/A** | Does not apply to a retrospective study on public datasets. |

Generated for the manuscript in `paper/manuscript.tex`. Intended to be included as a
supplementary appendix, not only as a repository file — the point of a checklist is that a
reviewer can read it.

---

## Title and abstract

| # | Item | Status | Where |
|---|---|---|---|
| 1 | Identifies the study as applying AI/ML to medical imaging | Met | Title; Abstract |
| 2 | Structured abstract: aims, methods, results, conclusions | Met | Abstract (247 words) |

## Introduction

| # | Item | Status | Where |
|---|---|---|---|
| 3 | Scientific and clinical background, including the intended use | Met | §I Introduction |
| 4 | Study objectives and hypotheses | Met | §I, contributions list |

## Methods — study design

| # | Item | Status | Where |
|---|---|---|---|
| 5 | Prospective or retrospective study | Met | Retrospective, stated in §III-A |
| 6 | Study goal (model creation, exploratory, feasibility, non-inferiority) | Met | §I — model development and evaluation, explicitly not a clinical trial |
| 7 | Ethical approval / institutional review | N/A | Public de-identified datasets (HAM10000, PAD-UFES-20); no new human-subject data collected. Stated in §III-A |
| 8 | Registration of the study protocol | **Not met** | Not registered. A pre-registered analysis plan is used instead: `results/analysis_plan.json`, frozen before the single test read (S9) and hashed into `results/frozen_artifacts.json` |
| 9 | Sources of funding and role of funder | Met | Acknowledgements (`TODO` marker pending) |

## Methods — data

| # | Item | Status | Where |
|---|---|---|---|
| 10 | Data sources | Met | §III-A; HAM10000 (Tschandl et al.), PAD-UFES-20 (Pacheco et al.) |
| 11 | Eligibility criteria: how, by whom, and when data were selected | Met | §III-A; `ml/preprocessing/split_dataset.py` |
| 12 | Data pre-processing steps | Met | §III *Backbones and training*; `ml/preprocessing/transforms.py`. Colour constancy was planned and **not** implemented — reported as absent rather than omitted (Limitations) |
| 13 | Selection of data subsets, if applicable | Met | §III-A |
| 14 | Definitions of data elements, with references to Common Data Elements | Partial | Class definitions in Table I. No formal CDE mapping — dermoscopy has no widely adopted CDE set for these seven diagnoses |
| 15 | De-identification methods | Met | Inherited from the source datasets, both released de-identified; stated in §III-A |
| 16 | How missing data were handled | Met | Missing `age`/`sex` are kept as an explicit `unknown` level and never folded into a real group — `research/selective/fairness.py:load_attributes`; the excluded counts are footnoted in Table IV |

## Methods — ground truth

| # | Item | Status | Where |
|---|---|---|---|
| 17 | Definition of the reference standard | Met | §III-A; histopathology, follow-up, expert consensus or in-vivo confocal microscopy, per HAM10000's `dx_type` |
| 18 | Rationale for choosing the reference standard | Met | §III-A |
| 19 | Source of ground-truth annotations; qualifications of annotators | Met | §III-A, deferred to the source dataset publications |
| 20 | Annotation tools | N/A | Labels are dataset-supplied; no annotation performed in this work |
| 21 | Measurement of inter- and intra-rater variability | **Not met** | Not available. The source datasets do not release per-rater labels, so variability cannot be computed here. Stated in Limitations |

## Methods — data partitions

| # | Item | Status | Where |
|---|---|---|---|
| 22 | Intended sample size and how it was determined | Partial | Fixed by the public datasets rather than powered in advance. Post-hoc power is reported where it binds: the under-40 subgroup gates in `research/stats/results_oof/intersectional_age_sex.csv` and the conformal per-class certifiability counts |
| 23 | How data were assigned to partitions; specify training, validation, test | Met | §III-A; **all splits grouped by `lesion_id`** with `assert_no_leakage()` enforced — `ml/configs/splits/split_v1.csv`, train 6,981 / val 1,532 / test 1,502 |
| 24 | Level at which partitions are disjoint | Met | Lesion level, not image level. This is Hard Rule 1 of the project and the reason the bootstrap resamples lesions (`research/ablation/bootstrap.py`) |

## Methods — model

| # | Item | Status | Where |
|---|---|---|---|
| 25 | Detailed description of the model | Met | §III *Backbones and training*, *Ensembling*, *Test-time augmentation and calibration*; six CNN backbones, soft-vote ensemble, 24-view TTA, Dirichlet calibration |
| 26 | Software libraries, frameworks, and packages | Met | §III *Backbones and training*, closing paragraph; Python 3.12, PyTorch 2.11 (CUDA 12.8), `timm`, scikit-learn 1.5. Pinned environment in `pyproject.toml` |
| 27 | Initialisation of model parameters | Met | §III *Backbones and training*; ImageNet-pretrained, two-stage schedule (3 frozen head epochs, then fine-tune) |

## Methods — training

| # | Item | Status | Where |
|---|---|---|---|
| 28 | Details of training approach | Met | §III *Backbones and training*; `ml/configs/training_config.yaml`, effective-number class weighting (Cui et al. 2019), seed 42 |
| 29 | Method of selecting the final model | Met | **Macro-F1 on validation, never accuracy** (Hard Rule 3 — 67% `nv` makes accuracy meaningless). The one place this discipline was nearly broken is documented as a counter-example: ridge stacking has the best test and the worst val, and is therefore *not* reported |
| 30 | Ensembling techniques, if used | Met | §III *Ensembling*; uniform soft-vote over six CNNs. Rank-averaging, Nelder–Mead weights, ridge stacking and Caruana greedy selection were all evaluated and are reported as rejected |

## Methods — evaluation

| # | Item | Status | Where |
|---|---|---|---|
| 31 | Metrics of model performance, and why they were chosen | Met | §III *Metrics*; Macro-F1, balanced accuracy, escalation sensitivity, ECE, and the two clinical-utility definitions (Number Needed to Biopsy at a stated reference prevalence, False Reassurance Rate). Accuracy is explicitly excluded as a selection criterion |
| 32 | Statistical measures of significance and uncertainty | Met | Lesion-grouped bootstrap for Macro-F1/ECE/AUC; **exact Clopper–Pearson for subgroup proportions** — the switch rule is stated once in `research/stats/intervals.py` and applied by `research/run_session7_stats.py`. McNemar and DeLong with Holm–Bonferroni |
| 33 | Robustness or sensitivity analysis | Met | 11-rung ablation ladder (`results/ablation_table.csv`) plus five further combination levers reported as negative, two of them pre-registered rungs read in the single test pass (`results/session9/new_rung_comparisons.json`); λ stability under lesion resampling, `research/agerule/results_oof/`; NNB reported across a stated prevalence range 0.01–0.05 |
| 34 | Methods for explainability or interpretability | Partial | Grad-CAM overlays (`paper/figures/gradcam/`, Fig. 7) plus a quantitative lesion-interior attribution fraction over the whole test split against Tschandl's segmentation masks (0.523 [0.509, 0.536], concentration ratio 3.45; `results/session9/attribution_summary_test.json`). Deliberately **not** claimed as causal evidence: for a misclassified case the map is taken w.r.t. the predicted class and lands on the lesion almost by construction, and Grad-CAM fails known sanity checks (Adebayo et al. 2018). The mechanistic claim rests on escalation-mass AUC instead |
| 35 | Validation or testing on external data | Partial | Performed, on a **different modality**. The frozen system was applied unmodified to all 2,106 PAD-UFES-20 smartphone clinical images (`research/xdomain/results/`, §IV *External evaluation*): the six-CNN soft-vote reaches Macro-F1 0.167, below its own best member (0.188), and the frozen calibration map degrades it further to 0.133. This is a dataset-shift benchmark rather than external validation of the intended use; **no independent same-modality dermoscopy cohort has been evaluated**, and that gap is stated in Limitations |
| 36 | Pre-specification of the analysis, and any deviations | Met | `results/analysis_plan.json` (frozen before the single test read) and `results/comparison_families.json` (every comparison family declared before its members are run, with confirmatory and exploratory sets separated) |

## Results

| # | Item | Status | Where |
|---|---|---|---|
| 37 | Flow of participants or cases (a diagram is recommended) | Partial | Counts given in Table I and §III-A; no CONSORT-style flow diagram. Nothing is excluded after ingestion, so the flow is a single step |
| 38 | Demographic and clinical characteristics of each partition | Partial | Age band and sex reported, including the training escalating-class prior by band (`results/age_band_prior.csv`, 4.9% under 40 vs 35.5% at 60+) and an age × sex intersectional slice with its suppressed cells named. **Skin tone is unavailable for the development cohort**: HAM10000 ships no Fitzpatrick labels and the ITA image proxy is documented as invalid on dermoscopy. It is reported for the external cohort, where 1,302 of 2,106 images carry a Fitzpatrick label — types I–IV are powered, V and VI are suppressed, so the analysis says nothing about darker skin |
| 39 | Performance metrics for all partitions | Met | `results/ablation_table.csv`; validation and OOF in `research/stats/results_oof/` |
| 40 | Estimates of diagnostic accuracy and their precision | Met | Every reported proportion carries a 95% interval with its method named — Table IV, `research/stats/results_oof/age_gap_intervals.csv` |
| 41 | Failure analysis of incorrectly classified cases | Met | §V *Confidently wrong is a distinct failure mode*; the under-40 blind spot, and specifically that it is a *confidently*-wrong failure — the abstention gate defers only 6.2% of that band against 17.3% at 60+, and rescues 11.1% [0.014, 0.347] of its misses against 36.2% [0.227, 0.515] at 60+. Across the split only 21 of 78 missed escalating lesions are deferred (`results/session9/orthogonality_test.csv`) |

## Discussion

| # | Item | Status | Where |
|---|---|---|---|
| 42 | Study limitations, including potential bias, statistical uncertainty, and generalisability | Met | §VI Limitations; per-band calibration disparity, the intersectional suppressed cells, the stacking mismatch, the forfeited conformal guarantee under out-of-fold calibration, and the failure of the age rule to be orthogonal to abstention in the very band it targets are all named rather than omitted |
| 43 | Implications for practice, including the intended use and clinical role | Met | §V *Clinical positioning*; framed around deployability — decision rule, abstention, conformal sets and number-needed-to-biopsy rather than headline accuracy |

## Other information

| # | Item | Status | Where |
|---|---|---|---|
| 44 | Registration number and name of registry; where the full protocol can be accessed; sources of funding | Partial | Not registered (item 8). Protocol, code and frozen artifact hashes are in the repository (`results/frozen_artifacts.json`, `results/analysis_plan.json`); repository URL is a `TODO` marker in the manuscript pending release |

---

## Summary

| Status | Count |
|---|---|
| Met | 33 |
| Partial | 7 |
| Not met | 2 |
| N/A | 2 |
| **Total** | **44** |

The 2 unmet items are honest gaps, not oversights, and both are stated in the
manuscript: **no study registration** (mitigated by a frozen, hashed analysis plan) and **no
inter-rater variability** (not derivable from the public label sets). Item 44 is partial for
the same reason as item 8.

Item 35 moved from *Not met* to *Partial* once the external evaluation ran: it is a genuine
external evaluation, but on smartphone clinical photography rather than an independent
dermoscopy cohort, so it tests robustness under domain shift rather than validating the
intended use. Calling it *Met* would overstate what was done.

Section pointers here name subsections rather than letters, because subsection letters shift
whenever the manuscript gains a section and silently rot.
