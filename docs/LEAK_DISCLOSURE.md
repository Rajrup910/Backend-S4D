# Disclosure: duplicate images and linked lesions across splits (V1 to V6), and the V7 fix

**Status: DRAFT for the owner's review. Not committed, not published.**

**Found:** 5 November 2026, by a read-only audit of every split and training pool from V1 to V6. This disclosure is
published at the start of V7, by the owner's decision of 5 November. The commit dates show the gap between finding and
publishing. Until then the results stayed on one machine, because the public README still carried the claims corrected
below.

## 1. What leaked

Every split in this project was grouped by `lesion_id`, and that grouping holds: no `lesion_id` or archive lesion ID crosses
any split or fold, V1 to V6. But the same image, or another photo of the same lesion, can sit in two splits **under
different names or lesion IDs**, and no ID check can see that. A pixel audit found such rows in every version.

| Version | Read | Rows read | Rows with a copy in that model's training data | Share |
|---|---|---:|---:|---:|
| V1 | HAM test, trained on HAM train | 1,502 | 8 (7 + 1 found by M1) | 0.53% |
| V1 | HAM val | 1,532 | 3 | 0.20% |
| V2 | BCN + MSKCC external, trained on HAM train | 15,316 | 2 | 0.01% |
| V3 | HAM val / external test, all-three train | 1,532 / 2,232 | 4 / 10 | 0.26% / 0.45% |
| V4 | val / reserved / ham_val / ham_test | 2,270 / 4,733 / 1,532 / 1,502 | 7 / 16 / 4 / 8 | 0.26% to 0.53% |
| V4 k-fold | each fold against the other folds | about 3,059 | 6 to 11 | 0.20% to 0.36% |
| V5 | each fold against the other folds + young extras | about 3,059 | 18 to 24 | 0.59% to 0.78% |
| V6 | each fold against the other folds + extras + added pool | about 3,059 | 32 to 45 | 1.05% to 1.47% |

These counts are **lower bounds**: the detectors miss some re-photographs.

**Found later by the learned-matcher audit (M1, 12 Nov 2026), all checked by eye:** 8 HAM-test images have a recoloured,
re-cropped or zoomed copy elsewhere that the audit above had not flagged (`results/leak_audit/leak_rows_all.csv` lists them with
no pixel copy). One, `ISIC_0027820`, is copied in the development rows (`ISIC_0030835`), which were HAM training data in V1 and
training data in V4: the V1 and V4 HAM-test counts above rise from 7 to **at least 8**. The other 7 are copied in the V6 added
pool; no version from V1 to V6 trained on that pool and read HAM test, so they change no reported result. V7 removes the
training-side copy of all 8 (`results/v7/m1/m1_links.csv`).

The largest single source was V5's **young extras** (2,078 training rows, all linked rows MSKCC): 35 were the same photo as a
development training row under another name, 4 as a `val` row and 14 as a `reserved` row; 63 / 19 / 37 shared a lesion ID with
those splits and 51 / 18 / 34 a patient ID. 642 extras have no lesion ID, so only pixels can check them.

Also found: V1's PAD-UFES-20 split had **5 exact-byte duplicate pairs across train and test**, and 179 of 1,297 patients
(567 rows) spanning its splits, because it split by lesion, not patient. 16 patients (50 rows) span V4's splits and 11 patients
(27 rows) span the development folds; only 14.4% of MSKCC rows carry a patient ID and HAM and BCN carry none, so patient
overlap is largely unobservable.

## 2. Deviations: what the leak changed

| Claim | As published | Corrected | Source |
|---|---|---|---|
| V5 under-40 escalation pAUC gain (Gate A) | +0.0351 [+0.0125, +0.0596], p 0.003 | **+0.0136 [-0.0084, +0.0357], p 0.228** without the 90 rows with a copy in V5's training data. Gate A failed in both reads (MCID +0.050) | `results/v5/confirm_s42_s43_s44_without_copies.json` |
| V5 S84 (MILK10k) | under-40 0.98 vs 0.94, McNemar p 0.3125 | unchanged without 89 copied lesions; Macro-F1 gain +0.100 to +0.106 | `results/v5/s84/s84_without_copies.json` |
| V6 (deployed vote vs V5) | Macro-F1 +0.0427 [+0.0245, +0.0618] | passes on all rows, without 380 declared at-risk rows and without 426. The rule was declared before any V6 prediction on folds 1-4 | `results/v6/e_primary.json`, `results/v6/leak_sensitivity_read.json`, `results/v6/leak_sensitivity_declaration.json` |
| V1 HAM test figures | as in the V1 paper draft | **not recomputed**: that needs a second test read, which Hard Rule 2 forbids. at least 8 of 1,502 test rows (0.53%) had a copy in training (7 found by the pixel audit, 1 more by M1) | audit table above |
| V1 PAD-only models | as reported | not correctable: the checkpoints no longer exist | - |

The V5 correction matters because V5 trained on the extras and its control did not, so the copies favoured V5. The other
shares are small and spread across both arms of each comparison, so their effect on the other published comparisons is
expected to be small; that has not been measured for every claim.

## 3. Why it happened

| Cause | Where |
|---|---|
| Image IDs compared as exact strings, so `ISIC_x` never matched `ISIC_x_downsampled` | `research/v5/young_data.py` |
| A guard that could never fire: the extras' group IDs carry a `__young__` prefix, so they cannot clash | `research/v5/train_v5.py` (`load_extra_train`) |
| A check that could never fail: the patient column was not in the file it read, so "0 shared patients" was guaranteed | `scripts/audit_leak_ids.py` |
| Only development rows were scanned for copies; evaluation splits were not | earlier keypoint audit |
| `lesion_id` treated as proof of a different lesion; HAM has one lesion under two IDs | every version's split check |

## 4. The fix (V7 data gate)

V7 builds one master table of every image it could touch and one link table, and refuses to train without a receipt that
binds both (`research/data/`, `research/v7/train_v7.py`). From the gate's own report (`results/v7/data_gate_report.json`):

* **57,488 images compared**, every source V7 could touch, including the sealed external cohort; every identity field 100% filled.
* **Planted positives found: 384 of 384**, eight families (renamed copy, downsampled variant, lesion link, patient link,
  re-encoded, resized and recoloured, rotated, centre crop) x 48 real images, 16 per archive. A check that has never found a
  planted copy proves nothing, so this runs with every gate build.
* **Links (receipt of 12 Nov, after M1):** 20,370 confirmed (same lesion by ID, normalised ID, SHA-256, existing groups,
  correlation plus keypoint copies, and matcher hits that SIFT confirms); 123,471 at risk (patient IDs, hash matches,
  single-detector keypoint hits, matcher hits alone). Patient IDs and hash matches alone never merge groups; they mark rows at
  risk. Before M1: 15,233 and 106,839.
* **Removals:** 3,424 rows leave all training because they link to an evaluated row or a never-train set (681 before M1);
  305 to 349 more per held-out fold (31 to 41 before); when an archive is held out, every row from it or linked to it (BCN 8,951,
  HAM 1,323, MSKCC 1,195, net of the rows already removed from all training). V7 keeps V6's number of optimiser steps, so
  removing rows never also shortens training (plan 13.5, decision 7).
* **Learned same-lesion matcher (M1, `research/v7/m1/`):** endpoints and the adoption rule were written to
  `results/v7/m1/protocol.json` before any matcher was scored. On 1,039 held-out lesions (7,648 same-lesion pairs) at a pairwise
  false-match rate of 1 in 100,000, the old pipeline (DINOv2 retrieval plus SIFT) linked **22.5%** of re-photograph pairs and
  the untrained copy detector SSCD **26.3%**. Two matchers trained here scored higher pooled (29.6% and 28.7%) but lower on HAM,
  so the pre-registered rule kept SSCD, with SIFT as the second detector (LightGlue scored lower than SIFT). On rows no matcher
  used (13,489 pairs in val, reserved and HAM val) SSCD links **27.7% [24.1%, 31.5%]** of re-photographs and **89.9%** of
  known copies (100% of exact copies). It found the 8 HAM-test copies above and no copy of a training image in the sealed
  external cohort (MIDAS, pixels only; labels unread).
* **Reads:** every V7 promotion read is repeated on four row sets (all rows, without the 380, without the 426, without every
  development row the gate links), and a gain from added data does not count until it survives that last set.

## 5. What this does not fix, stated plainly

* **Most re-photographs of one lesion are still missed.** With M1 the gate links about 28% of same-lesion re-photographs taken
  on another day or at another angle (HAM about half, BCN about a quarter), while exact and recoloured copies are found about
  90-100% of the time. An at-risk link is right about half the time by eye (24 reviewed), so V7 also removes some rows that
  were never leaks; that is the cost the data plan accepts. **The gate is "audited, with stated lower bounds". This project
  does not claim to be leak-free.**
* Patient overlap in HAM and BCN cannot be measured: those archives carry no patient ID.
* Pretraining corpora of foundation-model members cannot be audited for our evaluation images; every V7 read is repeated with
  ImageNet-pretrained members only (plan 11.3).

## 6. Claims corrected in this repository

* `README.md`: "Leak-free" removed from the title line; the Gate A table row now shows the corrected V5 figure.
* `docs/V6_RUNSHEET.md` lines stating "0 images and 0 archive lesions ... no fold leakage in V5": corrected in place with a dated note.
* `CHANGELOG.md` entries that called the extras' identifiers clean: corrected by a dated entry (earlier entries are not rewritten).

* `README.md` (11 Nov): the credit line "zero-leakage shield" reworded to "lesion-grouped split guard".
* `docs/V6_GLOBAL_BENCHMARK_REVIEW.md` (11 Nov): a correction note added at the top (images, not lesions; no "zero-leakage verified").

## 7. Older documents that still carry the old wording

These files are dated history and are not rewritten (the V6 runsheet is frozen; the manuscripts are earlier versions). They
use "leak-free" or "zero-leakage" in two different senses, and neither is a claim this project makes today.

| Sense | Where | How to read it now |
|---|---|---|
| A procedure: parameters were fitted on out-of-fold rows only, or a teacher never trained on the rows it labels | `docs/RESEARCH_ROADMAP.md:9`, `docs/RESEARCH_LOG.md:437` and `:467`, `docs/V6_RUNSHEET.md:86`, `:747`, `:1792`, `:1828`, `paper/manuscript.tex:1200`, `paper/manuscript_edited.tex:729` | It describes how parameters were fitted. It says nothing about duplicate images between splits, which sections 1-3 of this disclosure cover |
| A property of the data: "zero-leakage multi-cohort architecture", "zero-leakage deduplication", "a leak-free benchmark" | `paper/v4/manuscript_v4.tex:80` and `:149`, `scripts/readme_media/scenes.py:244` | Not supported. The V4 corpus was grouped by lesion id and hashes, which this audit shows miss renamed copies and re-photographs |

Five `CHANGELOG.md` headings also use the phrase; the CHANGELOG is a dated record and is corrected by later entries.
*(Owner decision 22: correct each file, or leave them as history under this section.)*

The full audit (`results/leak_audit/LEAK_AUDIT_REPORT.md`, `PREVENTION.md` and the row lists) is published with this
disclosure so every number above resolves to a file. *(Owner decision: publish the audit folder, or keep it local and cite only
the files in the right-hand column of section 2.)*
