# V7 Audit Report — freeze readiness and the first training run

**Date:** 12 Nov 2026, 17:30–20:00 IST. **Scope:** the V7 workspace as it stands on disk, judged against the
owner's audit brief of 12 Nov 2026.
**Nothing was trained, signed, committed or pushed.** No `ham_test` label, MIDAS label or new prediction was read.
**Every check is reproducible** from scripts in `results/v7/audit_2026-11-12/`. Raw outputs are in that folder's `raw/`.

**Labels:** **PASS**, **FAIL**, **BLOCKED**, **NOT VERIFIED** and **N/A** are used strictly. *Recomputed* means
re-derived here from the files on disk. *Reported* means read from a project file and not re-derived.

---

## 0. Executive verdict

**FREEZE APPROVED WITH EXPLICIT LIMITATIONS (technical).**

> **Addendum, 13 Nov 2026 (section 11).** An external review found a gate defect this audit missed (A11, patient IDs
> never loaded for the young extras and ISIC 2020) and a hole in P2 (A12). Both are fixed before signing; the gate and its
> downstream files were rebuilt and re-verified. "Repairs: none were needed" below is superseded.

**First run: NOT LAUNCHED. It is blocked on two owner-only actions:**
1. Commit the working tree, on the owner's yes.
2. Sign Gate 0: `python -m research.v7.plan_freeze --sign --owner-approved --owner "NAME"`.

The repository's own rule (plan 14.9) forbids any run of record before Gate 0. I may not sign it on the owner's behalf.

- **What passed.**
  - Every technical gate I could test: data-gate receipt bound to the bytes on disk; per-run isolation re-derived
    independently in all 8 run forms; V6 reference reproduced to 1e-9; 595/595 tests; verifier exit 0; chain dry-run clean.
  - The 24 owner decisions in plan 13.5 are all written: 12 Nov, "as recommended", with decision 11 overridden to
    "M1 before runs of record", and M1 is done.
- **Single most serious issue.** This is about evidence, not code. V7's under-40 and dark-lesion builds were designed
  from error analyses of folds 1–4. Those same rows are where V7 is promoted (finding A2).
  - So any subgroup gain V7 shows there is weak confirmatory evidence.
  - No fresh cohort can confirm an under-40 result: HIBA is spent and now in training, and MIDAS has 69 under-40 images.
  - This, not a bug, is what limits the paper.
- **Repairs.** None were needed in V7 code: no demonstrable defect was found in the active path. One regression test was
  added (`tests/test_v7_audit_isolation.py`, 10 tests). It checks isolation independently of the gate's own code, and a
  mutation check confirms it fails when a removal list is emptied.
- **Remaining risks.**
  - Re-photograph recall: about 33% of same-lesion pairs are still unguarded where IDs are missing.
  - Patient identity is unobservable for HAM and BCN.
  - The promotion rule is low-powered: P1 promotes 10–25% of the time and Gate A2 passes 10–16% at a true +0.035.
  - The foundation members' pretraining cannot be audited.
  - The working tree is uncommitted (49 paths).

---

## 1. Repository and environment inventory (Phase 0)

| Item | Value | Source |
|---|---|---|
| Repo root | `C:\Users\RAJ\Downloads\Capstone review 1` (the V7 workspace: `research/v7/`, `docs/v7_final_revision/`, `results/v7/`) | filesystem |
| Branch / HEAD | `v5-confirmation-review2` / `5f14ebc` ("V7 M1 amendment 01 …"); `main` = `33b0f74` | `raw/head_start.txt` |
| Working tree at start | 37 modified tracked, 12 untracked paths (49 lines) | `raw/git_status_start.txt` |
| Uncommitted, scientifically material | `research/data/build_master_v7.py`, `research/data/overlap_guard.py`, `research/v7/m1/{cost,headline,tiers}.py`, `research/v7/plan_freeze.py`, the data-gate receipt/report, data roles, pool receipt, smokes, `docs/LEAK_DISCLOSURE.md`, the plan | `git status` |
| Gate 0 | **unsigned** (`results/v7/plan_freeze.json` absent; `plan_freeze --status` says so) | command output |
| Owner decisions (plan 13.5) | 24 / 24 written (12 Nov) | plan 13.5 |
| Python / OS | 3.12.10, Windows 11 26200 | runtime |
| Torch / CUDA / cuDNN | 2.11.0+cu128 / 12.8 / 91900 | runtime |
| GPU | RTX 5050 Laptop, 8.55 GB, sm_120, driver 610.62; idle at audit (378 MiB used) | `nvidia-smi` |
| Libraries | torchvision 0.26.0, timm 1.0.28, numpy 2.4.4, pandas 3.0.5, scikit-learn 1.9.0, Pillow 12.2.0, OpenCV 5.0.0, SciPy 1.18.0 | `raw/pip_freeze.txt` (87 packages) |
| Disk | C: 27 GB free (94% used; trainer floor 5 GB); D: 64 GB free | `df` |
| Lock file | **none**; `pyproject.toml` has lint config only. Snapshot taken: `raw/pip_freeze.txt` | — |
| Mutating commands found (not run) | `plan_freeze --sign/--amend`, `build_master_v7` (rewrites master/links/receipt), `pool_v7` (rewrites pool), `train_v7` without `--smoke`, `external_read predict/read` (spends MIDAS), `run_v7_first_night.ps1` without `-DryRun` | code read |

**Canonical V7 path:**
- data gate: `research/data/`;
- trainer: `research/v7/train_v7.py` (patches the V6 trainer's seams);
- promotion: `research/v7/promotion.py` + `promotion_run.py`;
- TTA: `research/v7/predict_tta_v7.py`;
- external read: `research/v7/external_read.py`;
- the plan that controls: `docs/v7_final_revision/V7_FINAL_MASTER_RESEARCH_PLAN.md` §14 (with §13.5 decisions).

`docs/V7_MASTER_RESEARCH_PLAN.md` (2,922 lines) is an older plan. Its banners say §14 of the final plan controls.

---

## 2. Dataset integrity, split and leakage (Phase 1)

Script: `results/v7/audit_2026-11-12/independent_data_audit.py` → `independent_data_audit.json`. MIDAS is summarised by
ids and hashes only.

### 2.1 Ledger, recomputed (images; groups = `master_group` after M1; lesion ids = source ids)

| Role (`frozen_split`) | Images | Kept | Master groups | Source lesion ids (rows with id) | Patient ids (rows) | Classes (nv/mel/bcc/bkl/akiec/vasc/df) | Age <40 / 40-59 / 60+ / unknown | <40 escalating images / groups |
|---|--:|--:|--:|--:|--:|---|---|---|
| `train` (development, 5 folds) | 15,294 | 15,223 | 8,673 | 7,590 (14,136) | 61 (77) | 8038/2525/1928/1657/874/158/114 | 2721/5825/6497/251 | 193 / 81 |
| `train_added` | 25,471 | 25,349 | 21,571 | 10,713 (13,543) | 1,605 (11,928) | 16005/2023/3639/1641/1581/336/246 | 1005/5798/6696/11972 | 90 / 73 |
| `train_extra` (young extras) | 2,078 | 2,044 | 1,360 | 897 (1,436) | 0 | not in the joined tables (labels in `results/v5/young_data/extra_train.csv`) | — | — |
| `train_hiba` | 1,199 | 1,199 | 1,192 | 1,199 | 589 (1,199) | 552/159/223/61/125/40/39 | 251/357/587/4 | 20 / 20 |
| `reserved` | 4,733 | 4,723 | 1,985 | 1,371 (4,105) | 41 (44) | 1892/1174/828/420/338/28/53 | 978/1525/2111/119 | 279 / 103 |
| `val` | 2,270 | 2,270 | 941 | 645 (1,972) | 17 (22) | 923/483/414/220/178/24/28 | 343/770/1107/50 | 0 / 0 (verified: no mel/bcc/akiec under 40 in `val`) |
| `ham_val` | 1,532 | 1,510 | 1,119 | 1,120 | 0 | 1018/173/82/160/53/22/24 | 267/699/558/8 | 22 / 10 |
| `ham_test` (never train, never read) | 1,502 | 1,502 | 1,120 | 1,121 | 0 | 1004/167/71/167/52/21/20 | 290/671/532/9 | 21 / 10 |
| `midas` (sealed) | 3,409 | 3,409 | 2,964 | 0 | 685 (3,152) | not summarised (sealed) | not summarised | — |
| **Total compared** | **57,488** | 57,229 | | | | | | |

**Notes:**
- `val` has 326 nevi, 5 bkl, 7 df and 5 vasc under 40, and no escalating lesion. This was checked against
  `manifest_v4.csv`.
- The `train_extra` labels exist in `results/v5/young_data/extra_train.csv`. This audit did not join them.

**Reconciliation with the historical figures in the master prompt:**
- 6,981 images / 5,229 lesions is HAM's part of the development pool, not "training".
- 1,532 / 1,120 is `ham_val`.
- 1,502 / 1,121 is `ham_test`.
- All three match the manifests. They describe the V1–V4 HAM-only split, not V7.
- V7 trains on up to 48,935 rows per fold run (§2.3).

### 2.2 Split isolation at every available unit — recomputed

| Check | Result | Status |
|---|---|---|
| Receipt `master_sha256`, `links_sha256`, 7 gate-code hashes vs bytes on disk | all match | **PASS** |
| Files on disk for all 57,488 rows | 0 missing | **PASS** |
| sha256 of a random 1,500 recomputed from bytes; PIL decode + RGB convert | 0 mismatches, 0 decode failures | **PASS** (sample) |
| Development rows: `group_id`, `src_lesion_id` crossing folds | 0, 0 | **PASS** |
| Development rows: `master_group` crossing folds | 45 groups (M1 merged across V6's frozen folds) | expected: handled per run by removal, see below |
| Development rows: `src_patient_id` crossing folds | 11 of 61 patients | handled per run by removal, see below |
| Kept rows sharing master group / lesion id / patient id between a protected role (`ham_test`, `midas`) and any training role | 0 / 0 / 0 | **PASS** |
| Exact sha256 clusters (52) and exact pHash clusters (181) spanning folds, groups or protected roles | 0 | **PASS** |
| **Per run (folds 0–4, holdouts BCN/HAM/MSKCC):** training rows allowed by the receipt that share a master group, group id, lesion id, patient id, sha256 or pHash with, or have a confirmed or at-risk link to, any row that run evaluates | **0 in every one of 8 run forms**; 0 between any training row and `ham_test`/MIDAS | **PASS** (`independent_fold_isolation.json`) |
| Removal counts | fold 0: 711 = 259 global + 452; folds 1–4: 645/650/599/650; holdouts 9,875/2,269/1,781 (= receipt + 259). The trainer's 697 for fold 0 = 711 − 14 global rows that sit in fold 0's evaluation side | **PASS** (reconciled) |
| Mutation check | with fold 0's removal list emptied: 128 confirmed links, 240 at-risk, 76 shared groups, 6 shared patients, 8 pHash | the check detects leaks |

**What this shows and does not show:** no overlap was detected by the tests run. It does **not** show that leakage is
impossible.
- Re-photograph detection after M1 amendment 01 guards 0.672 [0.605, 0.737] of same-lesion pairs where lesion ids are
  hidden (`results/v7/m1/headline.json`). About a third of pairs that IDs cannot see are still unguarded.
- HAM and BCN carry no patient id. The group-size cap (31) equals the largest observed group, so it has no headroom.

### 2.3 Duplicates and near-duplicates

- **Exact bytes:** 52 clusters (104 rows). **Exact pHash:** 181 clusters (365 rows). All of them sit inside one master
  group and one role. **PASS.**
- **Near duplicates** (M1, SSCD + ConvNeXt-Femto, density-normalised, SIFT where unsure):
  - Held-out pair TPR at FMR 1e-5 is 0.584. **Reported**, `results/v7/m1/amendment_01.json`.
  - Pairs were checked by eye in M1 (`review_new_links.png`). **Reported**.
  - This audit did not re-adjudicate pairs: **NOT VERIFIED** here.

### 2.4 Label consistency inside groups — new finding

- 16 master groups (122 rows; 12 in development) contain more than one diagnosis. Examples: mel+df, nv+vasc, akiec+bkl.
- 15 of them join two or more *source* lesion ids, mostly in BCN through `m1:sscd_femto_sn` links.
- Of 328 master groups that span more than one source lesion id, 15 disagree at lesion level. Random merging would give
  about 63%.
- **Interpretation:** at least roughly 7% of M1's cross-lesion merges (about 24 of 328, after dividing by the 0.63
  chance of a visible conflict) are not the same lesion. They may be the same patient or a false match.
- **Impact:** conservative for leakage, since it removes more rows. It slightly distorts group counts and the bootstrap
  unit if `master_group` is used. The promotion bootstrap uses `group_id`, which is unaffected.
- **Severity:** P2. **No fix:** changing groups is a protocol change. A proposed audit is in §8.

### 2.5 Preprocessing and experiment-construction leakage

| Pathway | Finding | Status |
|---|---|---|
| Augmentation / TTA before split | Augmentation is in the training dataset only; TTA at inference; the V7 TTA writer self-checks the identity view against banked predictions (max \|Δp\| 6.4e-4) | **PASS** (reported, `tests/test_v7_tta.py` passes) |
| Calibration / thresholds on test | Dirichlet, S56 and conformal fit on OOF only (V6); V7 promotion uses argmax and threshold-free pAUC | **PASS** (reported, V6 audit §6) |
| OOF folds grouped | S71 folds by `group_id`: 0 lesion/group crossing; M1 cross-fold groups handled by per-run removal | **PASS** |
| `ham_test` | never read in V7; no code path reads it outside the guard | **PASS** (code read + grep) |
| MIDAS | sealed; `external_read.predict` needs Gate 0 and a pixel-audit receipt; labels hash recorded | **PASS** (reported) |
| **Repeated use of folds 1–4** | Folds 1–4 were read at V6-5; V7's exploratory reads (dark read, shortcut chain, verification strata, xAUC, member contribution, band calibration) ran on the same rows. Those reads *motivated* B5/B8/B7/B2. V7 is promoted on those rows | **finding A2** below |
| Metadata encoding labels | `archive` is a strong label correlate (BCN df/vasc; HAM follow-up nevi) and is not a model input. Melanin index depends on source (HAM follow-up nevi 0.19 vs BCN 0.48–0.50), so any colour build can learn the archive | risk; P3 LOAO addresses it |

### 2.6 Licensing and provenance

**Reported** from `docs/v7_final_revision/V7_MODEL_CANDIDATE_REGISTRY.csv`, the plan and the V6 audit; **not re-verified**
against the licence texts.
- HAM10000, BCN20000, MSKCC, ISIC 2019/2020, MILK10k: CC-BY-NC family.
- DERM12345: CC-BY-NC.
- HIBA: ISIC archive collections 175/176/251.
- MIDAS: research-use agreement; its per-patient `cohort.csv` is git-ignored.
- PanDerm weights: CC-BY-NC-ND.
- The registry still lists every licence cell as unverified and 13 of 17 checkpoint hashes as `NOT_HASHED` (plan 14.2).

---

## 3. Code, model, training and inference (Phase 2)

**Approach:** the V7 path patches the V6 trainer through named seams, and had a six-workstream audit plus a red team on
11 Nov (`results/v7/freeze_audit/`). I did not re-review every line. I re-ran everything, read the risk points, and
re-derived the numbers that decide promotion.

| Area | Evidence | Status |
|---|---|---|
| Class order | Alphabetical 7-class order (akiec, bcc, bkl, df, mel, nv, vasc) = `class_index_7`; banked V6 `y_true` equals `manifest_v4.class_index_7` on all 12,235 rows | **PASS** (recomputed) |
| Head / mapping | 7-class heads for V6 finalists; 8-class anchors exist (V6 audit) | reported |
| Checkpoint loading | strict rebuild in `predict_tta_v7`; B2's concept head dropped by name only | reported (tests pass) |
| Step budget | decision 7 "hold V6": `STEP_BUDGET = "hold_v6"`; fresh smokes show `step_budget 13410` (V6's) with 14,036 pool rows kept and 697 removed | **PASS** (F11 closed in code) |
| Smokes of the six first-night forms | re-run 12 Nov 17:47–17:50, after the 17:09 receipt (R1, R2, R3, R3b, B3, B5); 1–2 optimiser steps each | **PASS** (F13 closed for these six; the other seven smokes are older and are not needed for the first night) |
| R4 learning-rate range checks | rehearsals exist (11 Nov 20:56–21:04, before amendment 01); the check of record runs after Gate 0 | **NOT VERIFIED** on the current receipt |
| NaN/Inf, 1455, disk floor, overwrite | trainer raises on non-finite loss; 5 GB floor; a finished summary refuses a second write; the chain retries 1455 once with one worker | reported + code read |
| Determinism | seeds fixed per member (42); bitwise determinism not claimed; seed variance of V6 members unmeasured (one seed each) | limitation |
| VRAM | B3 at 384 px runs batch 16 × accum 2 (V6 rule); anchors fit (measured 16 Sep: 6.27 GB fine-tune at 384/b32) | reported |
| Inference/app | `backend/` app not in V7 scope; serving parity **NOT VERIFIED** | N/A for this run |

---

## 4. Adversarial failure testing (Phase 3)

The suite already contains targeted failure tests for most of the master prompt's list. I mapped them, ran them, and
added one independent test.

| # | Failure to reveal | Where tested | Result |
|:-:|---|---|---|
| 1–2 | missing/corrupt files; missing/duplicate ids; unknown label | `test_v7_data_gate.py` (45), `data_roles.check_sources` | PASS |
| 3–4 | same lesion / copy across splits | `test_v7_data_gate.py` plants (384/384, 8 families); **new** `test_v7_audit_isolation.py` (10) | PASS |
| 5 | empty class / one-class metric | `test_v7_promotion_run.py`, S1 counted-class rule | PASS |
| 6 | batch size one, last batch | `test_v6_trainer.py` | PASS (reported) |
| 7 | NaN/Inf loss | `test_v7_trainer.py` | PASS |
| 8 | CPU/GPU, AMP | smokes on GPU with AMP; TTA self-check on CPU | PASS (smokes) |
| 9–10 | missing/corrupt/wrong checkpoint; resume; overwrite | `test_v7_tta.py`, `test_v7_trainer.py`, `test_v7_freeze.py` | PASS |
| 11 | deterministic evaluation twice | TTA identity-view self-check (6.4e-4 vs tolerance 0.03) | PASS |
| 12 | class-index permutation | `test_v7_decisions.py` (class order is protected) | PASS |
| 13 | eval transforms deterministic | `test_v7_tta.py` | PASS |
| 14 | TTA aggregation | `test_v7_tta.py` | PASS |
| 15 | OOF fold isolation | new test + gate tests | PASS |
| 16 | calibration/threshold leakage; test lock | `test_v6_lock.py`, `research/testguard.py` | PASS |
| 17 | hand-computable metric fixture | `test_v7_decisions.py` | PASS |
| 18 | promotion fail-closed, Gate 0 tamper, amendment, second candidate | `test_v7_freeze.py` (27), `test_v7_promotion_run.py` (37) | PASS |

**Whole suite:** `pytest tests` gives **595 passed, 0 failed, 0 errors, 1 skipped** (`raw/pytest_all.xml`) plus the 10 new
tests. The process exits 1 because pytest's temp-directory cleanup hits `PermissionError` on
`AppData\Local\Temp\pytest-of-RAJ\pytest-current`, after the session ends. This is environmental, not a test failure.
Run with `--basetemp` pointed at a writable folder to get exit 0.

**Verifier:** `scripts/audit_v7_final_verification.py` exits 0 with 55 `PASS:` lines (`raw/verifier.stdout.txt`).

**Chain:** `scripts/run_v7_first_night.ps1 -DryRun` prints every step and finishes 6 of 6 screens on the simulated clock
(`raw/first_night_dryrun.txt`). Its real step 0 (`plan_freeze.require()`) refuses while Gate 0 is unsigned.

---

## 5. Numerical and claim audit (Phase 4)

Script for the V6 reference: `independent_v6_reference.py`. It is written from scratch with sklearn and imports no
project module.

| Metric / count | Prior reported | Recomputed | Split / cohort | Unit | Evidence | Status |
|---|---|---|---|---|---|---|
| V6 vote Macro-F1, folds 1–4 | 0.743258 | **0.7432581** | folds 1–4, 12,235 img / 6,989 groups | image rows | `independent_v6_reference.json` | **reproduced** |
| 95% CI | — | [0.718, 0.768] | same | group bootstrap, 2,000 | same | new |
| Balanced accuracy / accuracy | — | 0.723 / 0.823 | same | image | same | new |
| Escalation pAUC (McClish, FPR ≤ 0.2), all ages | — | 0.889 | same | image | same | new |
| Escalation pAUC, under 40 | — | **0.833 [0.772, 0.884]** | 2,164 img; 158 escalating images / **65 groups** | image, group CI | same | new |
| pAUC by band <40 / 40–59 / 60+ | — | 0.833 / 0.877 / 0.856 | same | image | same | new |
| Argmax escalation sensitivity, all / <40 | — | 0.826 / **0.589** | same | image | same | new |
| BCN Macro-F1 | 0.638 | 0.6377 | folds 1–4 BCN | image | same, `denominators.json` | **reproduced** |
| MSKCC Macro-F1 (labels 0–6) | — | 0.348 | folds 1–4 MSKCC | image | same | absent classes score 0; use the S1 counted-class rule, never this figure |
| HIBA Macro-F1 (V6 declared) | 0.7348 [0.691, 0.767] | not re-derived (labels sealed in the V6 audit's reconstruction) | 1,199 lesions / 589 patients | lesion | V6 audit §8 | reported (independently reproduced on 12 Nov by the V6 audit) |
| Gate receipt counts | 57,488 rows; 27,162 / 109,280 links; 259 + 452/386/391/340/391 | same | gate | rows | `independent_data_audit.json` | **reproduced** |
| Historical 6,981/5,229; 1,532/1,120; 1,502/1,121 | as given | same | HAM dev / ham_val / ham_test | image / lesion | ledger | reproduced; not V7's split |
| Manuscript: under-40 test sensitivity 0.143, "twenty-one positives", exact CI [0.030, 0.363] | `paper/manuscript.tex:668,755,1221` | 21 images = **10 lesions** (all mel) | `ham_test` (V1–V4) | image | ledger | **FAIL as stated**: a Clopper–Pearson interval on 21 correlated images overstates precision; the honest unit is 10 lesions. The README already says "21 images, from only 10 lesions" |
| Manuscript: val "22 escalating cases <40" | project notes, S5 | 22 images = **10 groups** | `ham_val` | image | ledger | same issue |
| "≈48,000 lesions" | earlier claim | unsupported | — | — | V6 audit §4 | **FAIL** (do not use) |

---

## 6. Findings register

| ID | Severity | Finding | Evidence | Impact | Fix / status | Regression test |
|---|---|---|---|---|---|---|
| A1 | **P0 for launch (governance)** | Gate 0 unsigned; the first run of record cannot start | `plan_freeze --status` | blocks tonight's run | owner signs after commit (plan 14.9) | `test_v7_freeze.py` |
| A2 | **P1 (validity of interpretation)** | Under-40/dark/verification builds were motivated by exploratory reads of V6 folds 1–4, and V7 is promoted on the same rows. Plan 4.1 says Tier A is never independent confirmation, but no rule discounts subgroup gains on hypothesis-generating rows | `shortcut_chain.json`, `verification_strata.json`, `dark_v6_oof_read.json` status fields | a folds 1–4 subgroup gain is exploratory, not confirmatory; under-40 confirmation needs data that does not exist | **not fixable in code**; state it in every report | — |
| A3 | P1 (reproducibility) | The gate code, receipts and plan that the freeze binds are uncommitted | `git status` | the freeze manifest cannot name a commit | commit on the owner's yes, before signing | verifier checks hashes |
| A4 | P2 | 16 master groups carry two or more diagnoses; at least about 7% of M1's cross-lesion merges are not one lesion | §2.4 | conservative for leakage; group counts slightly off | report; propose an adjudication sample (§8) | — |
| A5 | P2 | Manuscript under-40 CI uses images (21) not lesions (10) | §5 | overstated precision in the V1–V5 paper draft | rewrite in the next manuscript; never quote 0.143 bare | — |
| A6 | P2 | No dependency lock file | `pyproject.toml` | environment drift | `raw/pip_freeze.txt` snapshot; propose `requirements-v7.lock` | — |
| A7 | P2 | R4 range checks of record never ran on the current receipt (rehearsals predate amendment 01) | `results/v7/lr_range_rehearsal/` | R2/R3/R3b screens refuse if R4 fails | first-night step 3 | `test_v7_lr_range.py` |
| A8 | P3 | `pytest` exits 1 from a temp-dir `PermissionError` after a fully passing session | `raw/pytest_all.txt` | false red in a naive check | use `--basetemp`; the verifier already counts from JUnit | — |
| A9 | P3 | Per-archive Macro-F1 over labels 0–6 counts absent classes as 0 (MSKCC 0.348) | §5 | misleading if quoted | the S1 rule already handles it; never quote the raw figure | `test_s1_counts_only_classes…` |
| A10 | info | 11 development patients span folds; patient overlap is unobservable for HAM/BCN | §2.2 | removed per run; residual unknown | stated | new test (patient key) |

No defect found here invalidates an existing metric or checkpoint.

---

## 7. Research deadlocks (Phase 5) — what the evidence supports

In short:

- **Under-40.** The gap in V6 is real but estimated on few lesions.
  - Under-40 pAUC 0.833 [0.772, 0.884] vs 0.877 at 40–59.
  - Argmax sensitivity 0.589 (images) under 40 vs 0.826 overall.
  - 65 escalating groups.
  - Gate A (V6) failed: +0.038 [+0.011, +0.071] vs the +0.050 MCID.
  - V4 closed heads, weighting and decision rules as levers.
  - The CI width (±0.056) sets a minimum detectable effect of roughly 0.05–0.06 on these rows.
- **Dark lesions.**
  - The original "young and dark means nevus" shortcut has evidence against it: darker excised nevi are not scored lower
    (ρ +0.137), while darker young melanomas are (ρ −0.252, p 0.084, 48 lesions).
  - Melanin depends on source.
  - Exploratory, n = 48.
- **Leakage.** Controlled as in §2. Residual risk is bounded below, not excluded.
- **Cross-site.** BCN is the weak archive. Part of it is case mix, through verification strata: BCN's pAUC deficit
  disappears on histology-confirmed negatives; its Macro-F1 deficit does not.
- **Calibration.** The global Dirichlet map leaves opposite-signed band errors.

---

## 8. Proposed next actions (none taken; each needs the owner)

1. Commit the working tree (A3), then sign Gate 0 (A1). Then run the first night: `V7_FREEZE_MANIFEST.md` §6.
2. Adjudicate a 40-pair sample of M1 cross-lesion merges with conflicting labels (A4). This needs no GPU. Estimate M1
   precision on cross-lesion merges.
3. Write `requirements-v7.lock` from `raw/pip_freeze.txt` (A6).
4. Add to every V7 report: "folds 1–4 subgroup results are exploratory because the builds were designed on them" (A2).

---

## 9. Freeze gate table

| Gate | Status | Evidence | Remaining action |
|---|---|---|---|
| Correct repo, branch, commit identified | PASS | §1 | — |
| No user changes overwritten | PASS | only new files written (§10) | — |
| Manifests reconcile to files and labels | PASS | §2.1–2.2 | — |
| Split isolation at lesion level; patient where ids exist | PASS (detected-overlap sense) | §2.2, new test | patient ids absent for HAM/BCN (stated) |
| Duplicate / near-duplicate audit adjudicated | PASS for exact; near-duplicate adjudication **reported** (M1) | §2.3 | sample re-adjudication (§8.2) |
| No unresolved critical leakage pathway | PASS for the first run; A2 limits interpretation of folds 1–4 subgroup gains | §2.5 | disclose |
| Label mapping consistent end to end | PASS | §3 | — |
| Train/val/test/calibration/OOF/TTA boundaries | PASS | §2.5 | — |
| Core metrics reproduced or marked unverifiable | PASS | §5 | — |
| Config and dependency versions recorded | PASS with limitation (no lock file) | §1 | A6 |
| Checkpoint save/load/resume tested | PASS (tests) | §4 | — |
| GPU smoke, one forward/backward/step | PASS (12 Nov 17:47–17:50) | §3 | — |
| Deterministic evaluation, hand-computable fixtures | PASS | §4 | — |
| Existing and new tests pass | PASS (595 + 10) | §4 | — |
| Adversarial tests run and recorded | PASS | §4 | — |
| Storage, memory, GPU, paths, naming | PASS (C: 27 GB; GPU idle; unique `--run-tag v70`; summaries refuse overwrite) | §1 | preflight at launch |
| Freeze manifest matches snapshot | **BLOCKED** until commit | `V7_FREEZE_MANIFEST.md` | A3 |
| **Gate 0 signed** | **BLOCKED (owner only)** | §1 | A1 |

---

## 10. Files written by this audit

- `results/v7/audit_2026-11-12/independent_data_audit.py` and `.json`.
- `results/v7/audit_2026-11-12/independent_fold_isolation.py` and `.json`.
- `results/v7/audit_2026-11-12/independent_v6_reference.py` and `.json`.
- `results/v7/audit_2026-11-12/raw/`: start state, verifier output, pytest JUnit, pip freeze, sha256 list, chain dry-run.
- `tests/test_v7_audit_isolation.py`.
- `docs/V7_AUDIT_REPORT.md`, `docs/V7_FREEZE_MANIFEST.md`.
- `.relay/NOW.md`.
- `CHANGELOG.md` (one appended section).

No existing file was modified except `CHANGELOG.md`.

---

## 11. Addendum, 13 Nov 2026: after an external review

An independent review of the research brief (steps 1-2) raised points that were checked against the files on disk.

| ID | Severity | Finding | Evidence | Fix / status | Regression test |
|---|---|---|---|---|---|
| A11 | **P0 (leakage, missed by this audit)** | `build_master_v7.load_rows` read patient IDs only from the leak audit's row list, HIBA and MIDAS. The young extras (1,039 IDs in `results/v5/young_data/candidates.csv`) and ISIC 2020 (5,559 in `results/v6/archive_meta/isic2020.csv`) had none, and 274 of the 417 ISIC 2019 patient IDs were missing. This audit's isolation check compared the builder's own column, so it could not see the gap. On the 12 Nov gate, training kept images of 2/0/1/2/2 patients evaluated in folds 0-4 and of 6 patients (12 images) in the MSKCC hold-out, all young extras | `raw/patient_id_defect_pre_fix.json` | `fill_archive_patients` (fails closed on a missing file or a clashing ID). Gate rebuilt (13 Nov 02:00, passed): at-risk links 109,280 → 137,170; removals 361 global (was 259), 476/413/437/375/457 per fold, HAM 2,097, MSKCC 1,559, BCN 9,616. No row left the removal lists (9 moved to the global list). Pool, roles, eight B5 bases and denominators rebuilt | isolation check now also keys on `archive_patient`, read from the archives' metadata: 0 in all 8 runs; with fold 0's list emptied it finds 14 shared patients |
| A12 | P1 (promotion rule) | P1's all-age pAUC credits cross-band ranking. A candidate that lowers every under-40 score raises P1, leaves the within-band under-40 pAUC delta at exactly 0 and passes the old P2 | `test_p2_vetoes_a_candidate_that_lowers_every_young_score` | Owner decision before signing: the under-40 xAUC joins P2's veto (upper 95% bound ≥ 0); plan 13.5 row 19 and §14.3 amended | same test; `test_p2_fails_closed_without_the_xauc_read` |
| A13 | info | The 40 back-solved HIBA vasc labels | `results/v7/hiba_train/hiba_train.csv` | All 40 are vasc in the separately fetched archive diagnoses: closed | — |
| A14 | info | Wording: MIDAS is prospective, two-site and general dermatology (not retrospective); PanDerm lists HIBA as a benchmark dataset; seed noise measured for the anchor only; README's 0.8047 is V1's pre-fix HAM-test read | public sources named by the review | Corrected in `V7_FREEZE_MANIFEST.md` §7 and README | README audit |

Not changed: the M1 recall residual on cross-archive, ID-less pairs is unmeasured (the review is right that the measured 67.2% is on same-archive pairs with IDs), and bootstrap intervals at 65 under-40 groups under-cover (F4; gain claims already use the wider interval).
