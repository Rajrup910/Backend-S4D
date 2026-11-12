# WS4 + WS6 audit: ID/OOD evaluation design, cohort roles, clinical-safety consistency

Read-only review, 11 Nov 2026. Scope: plan sections 1-12 and companion docs as they stand; section 13 and `plan_freeze.py`, `promotion.py`, `builds.py`, `train_v7.py` are being edited by the lead and are not flagged for in-progress state. Line numbers are as read at audit time and may shift.

## 1. Evidence inspected
- Docs: `V7_FINAL_MASTER_RESEARCH_PLAN.md` (all), `V7_DOMAIN_GENERALISATION_AND_VALIDATION_PROTOCOL.md`, `V7_STATISTICAL_PREREGISTRATION_AND_SAFETY_PLAN.md`, `V7_FINAL_EXECUTION_RUNSHEET.md`, `V7_OPEN_QUESTIONS_AND_APPROVAL_GATES.md`, `V7_RED_TEAM_REVIEW_AND_RESOLUTIONS.md`, `V7_STANDING_PROBLEMS.md`, `V7_INHERITANCE_LEDGER.md`, `README.md`, `docs/V7_MASTER_RESEARCH_PLAN.md` (grep), `docs/V6_RUNSHEET.md` and `docs/RESEARCH_LOG.md` (grep for `reserved`).
- Code (read): `research/v7/external_read.py`, `midas_prepare.py`, `data_roles.py`, `pool_v7.py`, `promotion.py` (as of audit), `builds.py`, `train_v7.py::check_args_v7` (executed against `builds.BUILDS`, CPU only, no data), `plan_freeze.py`, `band_calibration.py`.
- JSON, aggregates/keys only: `seal.json` (keys: sealed, cohort_sha256, labels_sha256, labels_path, rows, lesion_groups, pixel_audit; rows 765, groups 513), `prepare_receipt.json` (counts, hashes, category mapping; no per-image data), `data_roles_v7.json` (bindings_ok true, labels_read false), `verification_strata.json`, `xauc_read.json`, `p4_band_calibration.json`, `pool_v7_receipt.json`.
- `ml/data/manifest_v4.csv`: aggregate count of under-40 escalating groups in `reserved` only (verified 104: BCN 80, MSKCC 24; 56 melanoma; all-fold development 81, folds 1-4 65).
- NOT opened: MIDAS `cohort.csv`, `labels_restricted`, HAM test, HIBA outcomes. No file other than this report was written.

## 2. Track availability

| Track | AVAILABLE NOW | DESIGNED only | READ HISTORICALLY | UNAVAILABLE |
|---|---|---|---|---|
| (a) In-domain folds 1-4 vs banked V6 | Banked V6 vote predictions (`results/v6/preds_tta`, manifest `v6_reference_manifest.json`); `promotion.py` library (P1/P2, E-histo, xAUC) | The V7-candidate read (runner is a plan blocker, lead's area) | Folds 1-4 read at V6-5 and again by every V7 pre-test (strata, xAUC, shortcut chain, member contribution, P4, dark read) on V6 outputs only | Any V7 candidate prediction on folds 1-4 (no V7 run exists) |
| (b) Cross-site LOAO on BCN/HAM/MSKCC | Gate removals per holdout (`data_roles_v7.json` CROSS_SITE) | P3, L0 vote, candidate LOAO runs | `results/v6/m7_loao_binary.json` (one binary SWAD-style LOAO test); per-archive in-domain rereads (not LOAO) | A V6-recipe LOAO baseline; any candidate LOAO run; runnable L0 vote (section 6) |
| (c) PAD-UFES-20 other modality | Images restored (S8b) | "Separate track on patient-grouped splits" (9.3, P6) with no spec, endpoint or code | S8b read of all 2,106 images with frozen HAM-only models; Mahalanobis AUROC 0.913; Fitzpatrick I-IV spread 0.093 | Patient-grouped PAD splits (9.6 says rebuild first); PAD checkpoints (do not exist); V7 PAD results |
| Sealed external: MIDAS (not one of the three tracks; plan treats as Tier C2) | Sealed cohort (765 images, 513 records, 69 under-40 images) | The single read | None (labels unread) | Paired V6-vs-V7 read (section 4, F-04) |

Conflations found: see F-01 to F-03, F-09, F-12.

## 3. Task 2: MIDAS role conflicts

Controlling statement (10.11, line 534): MIDAS is the sole sealed external read, sealed 10 Nov. Section 13 (line 774) says 9.7 item 3 is decided. The target wording ("retrospective biopsy-cohort benchmark, all-age read at most, no prospective workflow claim, under-40 not powered") appears NOWHERE as such; 10.11 and 13.7 say only "under-40 stays unconfirmable".

Passages in conflict (quote; superseded?):

| file:line | Quote | Class | Superseded explicitly? | Action |
|---|---|---|---|---|
| Plan:152-153 | "Untouched Confirmatory Prospective Cohort (New Hospital)" / "Event-driven consecutive presentation (E >= 71" | i | Partly: 9.4 (line 345-347) says BLOCKED, never says "replaced by MIDAS"; 10.11 does not mention Tier C2 | Add pointer to 10.11; relabel MIDAS as retrospective |
| Plan:175-181 | "E >= 71 escalating events in patients < 40" and 4,250/4,666 consecutive | i, iii | 9.1 row 11 (line 294) calls figures placeholders; no pointer to MIDAS's 69 images | Pointer: not applicable to MIDAS |
| Plan:256 | "Phase 9: Independent External Hospital Validation (Tier C2, E >= 71)" | i | No | Pointer |
| Plan:345-347 | "No external cohort is in hand. A new-hospital consecutive series is not available" | ii | By 10.11 and 13.1 (line 774) implicitly; 9.4 header still "BLOCKED" | Edit 9.4 header to "SUPERSEDED by 10.11" |
| Plan:351-356 | MIDAS listed as "candidate", "about 1,290 lesions ... licence terms not stated" | iv | 10.11 gives actual numbers (765/513/69) | Mark table historical |
| Plan:382 | "Tier C2: name a clinical partner, accept a sealed public cohort" | ii | 13.1 (line 774) | Mark decided |
| Plan:959, 974 | "69 MIDAS records under 40" | iv (unit) | No: 69 is IMAGES (`primary_by_age_band` counts cohort rows); under-40 records and positives unknown | Say "69 images" |
| README:9, 31, 87 | status cites "sections 9-12"; "E>=71 sizing"; "new external hospital ... consecutive presentation lesions" | i, ii | No | Update status, add pointer |
| Domain:29-30, 90-93 | "Tier C2: Untouched Confirmatory External Cohort (New Hospital, event-driven E >= 71 ..." "single-pass evaluation for clinical efficacy" | i, iii | No; file has no audit notice | Banner + pointer |
| Domain:108-109 | "V7 External Hospital vs V6 External Hospital (paired confirmatory comparison on Tier C2)" | iii | No | See F-04 |
| Stat:117-119 | "accrual continues prospectively until E >= 71 verified under-40 escalating events" | i, iii | Partly in-text ("planning placeholders", line 117) | Pointer |
| Runsheet:29, 232-239 | "Phase 9: Independent External Hospital Validation (Tier C2 ..." "External consecutive presentation lesions" "external clinical hospital partners unblind" `scripts/run_v7_external_confirmation.py --locked-receipt` | i, iii | Banner (line 3) points to section 9 only, which says BLOCKED; script does not exist (only `research/v7/external_read.py`) | Rewrite Phase 9 as MIDAS sealed read; fix command |
| OpenQ:33, 56-59 | "untouched hospital cohort with consecutive presentation prior"; "Authorize clinical collaboration outreach ... consecutive ... N >= 4,250" | i, ii | No; file has no audit notice | Mark decision 5 resolved |
| RedTeam:90, 125 | "External clinical validation (Tier C2) mandates ... consecutive presentation lesions (N >= 4,250 ...)" | i | No | Historical banner |
| Standing:20 (P12) | "BLOCKED until a cohort exists; V7 then claims internal promotion" | ii | No | Update to MIDAS sealed |
| `external_read.py` docstring l.1-2 | "Tier C2; BLOCKED until a cohort exists" | ii | No | Comment only (lead's area if touched) |
| `docs/V7_MASTER_RESEARCH_PLAN.md`:82, 2735, 2766, 2855, 2888, 2917 | prospective/consecutive Tier C2, "sole confirmatory evaluation" | i, iii | Document has a precedence banner; still unqualified | Accept as superseded; no edit |
| agent03/agent12 files | same | i | Research inputs | none |

Also: Plan 11.2 item 3 (line 605-607) compares MIDAS to a development number (0.865/0.814), a cross-cohort comparison that Plan 4.1 item 3 forbids; it is a sanity anchor, not a test, and should be worded so.

### How the seal works (from code)
- `seal` (own-split midas): refuses if `seal.json` exists; checks unique image ids, no overlap with master table by normalised id or SHA-256 (cohort's own split excluded; `sha256` column absent in the MIDAS cohort so only ids are checked); records SHA-256 of cohort file and of label file, label PATH (absolute, local), row and group counts, and a text note that the keypoint pixel audit is "required before predict". Copies cohort to `results/v7/external/midas_v7/cohort.csv` (gitignored).
- `predict`: needs Gate 0, requires seal, refuses if `predictions.json` exists, checks coverage of cohort ids and the 7 probability columns, hashes predictions. ONE prediction set per cohort name.
- `read`: refuses if `receipt.json` exists or `--owner-approved` missing; verifies predictions hash and label-file hash; writes `receipt.json` (`label_read_count: 1`) BEFORE opening labels; then reads labels once, scores all ages and `<40`. A crash after the receipt burns the read permanently.
- `data_roles_v7.json` agrees with seal and prepare receipt (765 / 513 / {60+ 529, 40-59 167, <40 69} / 39 intermediate; labels hash equal). Minor: plan 10.6 quotes 656 global removals and HAM 1,476 / MSKCC 1,255; the JSON has 681, 1,475, 1,253 (LOW, F-15).

## 4. Task 3: HIBA, retired splits, D2

Passages still treating HIBA / retired splits as evaluation or benchmark (10.11 says training only; "no V7 system is ever scored on HIBA"):
- Domain:27-28 "Tier C1 ... permanent retrospective comparator for V7"; Domain:87-89 "it can only serve as a retrospective benchmark in V7"; Domain:108 "V7 HIBA vs V6 HIBA (paired retrospective comparison on Tier C1)".
- Runsheet:237 "retrospective HIBA benchmark (N=1,199)" as a Phase 9 control; Runsheet:58 lists HIBA, ISIC 2020, PAD-UFES-20 as "External benchmarks" inputs.
- Plan:150-151 (Tier C1: "permanent historical comparator; ZERO V7 model tuning") and Plan:132: wording says benchmark, but V7 trains on HIBA (10.11), so the "zero tuning" statement is false for V7 models.
- OpenQ:33, 88 (HIBA as the anchor of "transportability") and README:47.
- Code is consistent: `pool_v7.py`, `train_v7.load_pool_v7`, `data_roles.py` all train on HIBA and retired splits; no code scores them.
- Factual error: "`reserved` read once at S63" (Plan 10.11 l.535, 12.3 D2 row l.727, 12.5 item 12, `pool_v7.py` docstring) conflicts with `docs/V6_RUNSHEET.md:1568` "reserved read 8x" and RESEARCH_LOG S51-S67 (backbone probes, S54, S58, S59, S64, S67 all used it). `reserved` is a heavily used development set, not a once-read set.

### D2 numbers and status
- 65 = under-40 escalating lesion groups in folds 1-4 (verified, Plan 9.1 row 2). 104 = groups in `reserved` (verified in manifest: BCN 80, MSKCC 24, 56 melanoma, zero HAM). 169 = 65 + 104. Specified only in a table row (12.3, l.727), 12.5 item 12, 13.5 item 13, ledger row 49, standing P18. No code, no test, no split manifest (grep: none; "D2" in `experiments.csv` is an unrelated age-adversary run).
- `added_train_v7.csv` (35,183 rows, hash in `pool_v7_receipt.json` and bound by `data_roles.py`) contains ALL 8,513 retired rows. D2 changes training rows per fold, so it changes the pool file/hash/receipt, `data_roles_v7.json`, the data-gate removals and the step budget (steps follow the core pool).
- Inside the plan: 13.6 says an amendment "may not change ... the row sets", yet D2 changes both training and evaluation row sets; 13.5 "Default at signing" adopts the Recommended column for blank rows, so signing with row 13 blank would put D2 (evaluation on `reserved`) and 10.11 (never scored) in force together, and D2's own row says it "Reverses the owner's decision of 10 Nov".
- Statistical caveats: `reserved` is BCN/MSKCC only (archive-skewed); V7 trained on 4/5 of `reserved` vs V6 never trained on it, so the paired read is a data-gain read, not a recipe read; banked V6 fold checkpoints exist (`ml/checkpoints/anchor-v6_*`) but a V6 re-score on 4,733 images must use identical inference code (9.2 forbids re-scoring under a new transform); fold-to-fold assignment of the reserved fifths is unspecified; cross-fit folds must be lesion-grouped and gated.

| Option | What it requires | Statements that must change | Risk |
|---|---|---|---|
| A. Decline D2 | Strike 12.3 D2 row, 12.4 last row, 12.5 item 12, 13.5 row 13 (set "keep 10 Nov"), F3 list; fix "read once" wording | 12.3, 12.4, 12.5, 13.5, ledger 49, standing P18 (reword as "power limit accepted, 65 groups") | Under-40 stays 65 groups; no change to pool/hashes |
| B. Separate signed amendment, secondary/diagnostic only | Cross-fit split manifest (lesion-grouped, per-fold gate), code, V6 re-score receipt, own signature before first run; never feeds P1-P3 or row sets; labelled "previously accessed development data (read 8x)"; pool for headline runs unchanged OR headline runs also cross-fit (decide once) | 10.11 gets an exception line; 12.3 row rewritten; 13.5 row 13 default must be "decline" until amendment signed; 13.6 wording; pool_v7 docstring | Two pool variants if headline stays all-reserved: D2 read then differs from the promoted system |
| C. Adopt now, full spec | Rebuild pool per fold, new `added_train_v7.csv`/receipt/hashes, rerun gate and smokes, rewrite P1/P2 row sets, power sim | 10.4, 10.8, 10.11, 12.x, 13.5/13.6, `data_roles.py` | Largest change right before freeze; invalidates "step budgets hold"; contradicts 13.6 |

Technical recommendation (not a choice for the owner): A for the signed freeze, with B kept as a post-signing amendment if the owner wants the power gain; in all cases correct "read once at S63" and make sure 13.5's "default at signing" cannot adopt D2 by a blank cell. Reason: D2 gains power only for a secondary under-40 read, is a data-gain not method comparison, rests on a set already read 8 times, and conflicts with the frozen training-only roles.

## 5. Task 4: E-histo and xAUC verification
- Plan 11.2 tables match `verification_strata.json` to 3 decimals in every row (all 0.889/0.865/0.957, -0.092 [-0.110,-0.072]; under-40 0.833/0.814/0.886; HAM, BCN, MSKCC rows; gap table). Denominators: 1,908 rows have a missing confirmation type and sit in neither "histo" nor "other" (mostly MSKCC; MSKCC "all negatives" 1,051 groups vs 156 + 53 classified). 11.2 does not state this next to the MSKCC row.
- Plan 13.3 xAUC table, melanoma-only rows, per-archive and pooled -0.106 [-0.179, -0.032] match `xauc_read.json`. All reported intervals are 95% percentile, although 13.1 F4 found 95% under-covers at 65 groups (0.913) and mandates the 97.5% interval for under-40 gain claims.
- Code: `e_histo` and `xauc_u40` return `decisive: False`; the docs call both secondary and "never promotes". No place found where either is described as a gate.
- Over-generalisation: "BCN's escalation-pAUC deficit against HAM is case mix, not transfer" (11.2) rests on an interval [-0.008, +0.056] that includes a deficit, and on different histo-negative compositions; "The deficit is not young melanomas' archive mix" (13.3) rests on two of three archive intervals spanning zero. Uncontrolled factors ARE listed (archive, subtype, thickness, images per lesion) in 13.3; the HAM follow-up mechanism is labelled "possible, untested" and 12.1 L1 numbers are not applied to other archives.

## 6. Task 5: Safety stack
- Specified: Dirichlet, class-conditional conformal, S56 (Plan 2 diagram l.83-85; 9.2 row l.313 `CORE_SAFETY`; Runsheet Phase 7); P4 band-wise calibration "joins the safety stack" (10.10 l.526); AMPS is in the optional metadata arm (2.3, l.100-104) with no code anywhere in `research/v7` (grep). No passage calls the Dirichlet/S56/conformal stack "optional"; "optional" is applied to the metadata arm and morphology arms only.
- Inconsistency: project CLAUDE.md records that conformal and lambda layers lost at matched workload (S65) and deployed stack is S56 alone; the plan lists conformal as CORE_SAFETY without reconciling.
- P4: `p4_band_calibration.json` drop rule is met only through band 60+; the under-40 band's |gap| falls by about 0.006 (below the 0.01 rule). P4 has no row in the owner decision table and `band_calibration.py` is not hashed by Gate 0 (lead's file; noted only).
- Unsupported-safety language: Stat:137 "full promotion certified"; Stat:156 "to guarantee valid class-conditional coverage" (OOF conformal has no exact guarantee, plan itself says so in Standing P11); Runsheet:193 "guarantees robust clinical deployment"; Runsheet:159, README:81-85, Metadata:21,147-149,212, Plan:104 "mathematically prevented"/"guarantee" (holds per lesion only; 9.6 qualifies it at matched workload but the earlier text does not); OpenQ:93 `THEORETICALLY_BOUNDED`, OpenQ:95 `CLINICALLY_VALIDATED` for Arm D (no clinical validation exists); Domain:50 "proves that selective referral diverts"; "Certified" status lines in Domain:7, Stat:8, OpenQ:6, RedTeam:7, README:122, Runsheet:273 (contradicts Plan 9.1 row 14). Correct handling of zero events exists (Stat:87-88; Plan:234).
- Verification bias: disclosed for development negatives (11.2) and the MIDAS comparator (11.2 item 3), but not as a limitation of the MIDAS read itself: biopsy-only cohort means no non-biopsied benign lesions, so flagged share, specificity-type figures and sensitivity at a referral threshold are not deployment estimates. Not stated where external results will be interpreted (Plan 4, 10.11, Stat 4.3).

## 7. Task 6: P3 like-for-like and L0
- `promotion.p3_holdout` takes a candidate and a baseline on the same holdout rows with the same counted classes: like-for-like by construction. What is NOT specified in docs/registry: the pool policy for L0 (V6's 25,471 added rows vs the 35,183-row V7 pool; 10.8 says runs of record use the V7 pool), the step budget per holdout (decision 7 covers folds only), and the seed count. If L0 uses the V6 pool and the candidate the V7 pool, P3 credits added data (HIBA, retired splits), the very thing the 4th row set is meant to discount.
- Step "5b" (candidate LOAO runs) appears only in 13.8 (l.1008, per audit-time read); it is absent from the 10.8 table and from the runsheet (runsheet Phase 8 still "Promotion Gate A2 Read", Gate A2 as HALT rule, contradicting 10.4's P1-P3 and 9.3).
- L0 as a 4-member vote is NOT runnable under the registry as audited: `check_args_v7` against L0 (member "anchor", 224 px, `loao_holdout_allowed=True`): `--member anchor` 224 OK; `--member dinov2_b` (224 or 392) REFUSED ("build L0 is declared for member 'anchor'"); `--member anchor --image-size 384` REFUSED ("build L0 runs at 224 px"). Only L0 and X2 allow LOAO; R1, R2, R3, B5 are refused for `--loao-holdout`, so candidate LOAO runs cannot start either. (Lead is editing `builds.py`/`train_v7.py`; re-test after their edit.)
- Edge case in `p3_holdout` (as audited): if the Macro-F1 arm is "not estimable" (no class reaches the counted-group floor, plausible for MSKCC with 3 classes), `passed` is decided by the pAUC arm alone.
- Selection contamination: members and preprocessing are chosen on fold 0, whose training rows include the later LOAO holdout archives, so LOAO is internal-external with mild selection leakage (9.6 "scope of nesting" covers it generally, not for P3).

## 8. Task 7: Small-subgroup language
- Plan:855 "The deficit is not young melanomas' archive mix." and Plan:945 "L1 confirms the target"; 13.3 "The deployed under-40 problem is larger than the endpoint the plan reports" (point estimate, 65/48 groups, 95% intervals near zero for melanoma-only); Domain:50 "proves". Zero-event handling is correct (Stat:87-88, 234). No document claims a definitive under-40 result; the above should be hedged.

## 9. Findings

| ID | Finding | Sev | Conf | Evidence | Correction | Files | Tests |
|---|---|---|---|---|---|---|---|
| F-01 | MIDAS role contradicted by Tier C2 "prospective/consecutive/new hospital" text in 14+ places; no passage states the retrospective, all-age-at-most, not-prospective, under-40-not-powered wording | HIGH | High | table in section 3 | Add one canonical MIDAS-role sentence to 10.11 and a pointer banner in Domain, Stat, Runsheet, OpenQ, RedTeam, README, Standing P12 | listed | doc-consistency grep check |
| F-02 | "69 records under 40" (13.7) is 69 IMAGES; under-40 records and positives unknown | MEDIUM | High | `prepare_receipt.json` `primary_by_age_band` counts cohort rows | Say images; state positives unknown while sealed | Plan | none |
| F-03 | Runsheet Phase 9 cites nonexistent `scripts/run_v7_external_confirmation.py`, "hospital partners", consecutive lesions | HIGH | High | `ls scripts` | Rewrite to `research.v7.external_read` MIDAS | Runsheet | doc check |
| F-04 | `external_read.predict` freezes ONE prediction set; V6_REFERENCE cannot be frozen on MIDAS, so Domain:109 paired V7-vs-V6 and Runsheet "V6_REFERENCE evaluated on identical cohort" are unsupported; `score` bootstraps against a zero comparator | HIGH | High | code lines predict/score | Either add a second frozen prediction set (before signing external code) or state MIDAS is single-system, descriptive, compared only to 11.2's anchor | external_read.py, docs | synthetic paired test |
| F-05 | `read` never verifies `cohort_sha256` (recorded at seal); pixel-audit requirement is text only; labels sit unlocked on disk; crash after receipt burns the sole read; intermediate-tumour secondary list from `midas_prepare` is never read by `read` | MEDIUM | High | code | Verify cohort hash; enforce audit receipt in `predict`; pre-register crash rule; rehearse with the real cohort file and synthetic labels | external_read.py | extend external-read tests |
| F-06 | MIDAS score lacks intervals for sensitivity and Macro-F1 (Stat 7.2 requires exact CP for small counts), no S1 class-count rule; under-40 slice has 69 images | MEDIUM | High | `score()` | Add exact CP and S1 rule before signing | external_read.py, Stat | yes |
| F-07 | Retired splits/HIBA still described as benchmark/comparator/evaluation (Domain 27, 87-89, 108; Runsheet 58, 237; Plan 150-151) against 10.11 | HIGH | High | section 4 | Rewrite Tier C1 as "V6 historical result; V7 trains on HIBA" | Domain, Runsheet, Plan | grep |
| F-08 | `reserved` "read once at S63" is false (read 8x per V6_RUNSHEET:1568; used for S51-S67 decisions); repeated in 10.11, 12.3, 12.5, `pool_v7.py` | HIGH | High | grep | Correct everywhere | Plan, pool_v7.py comment | none |
| F-09 | D2 contradicts 10.11 and 13.6; blank decision cells default to adopt; no code/spec/manifest; changes pool hash if adopted | HIGH | High | section 4 | Option A, or B as signed amendment; make default "decline" | Plan 12/13, ledger 49, P18 | split-manifest test if B/C |
| F-10 | P3 pool policy and step budget for L0 vs candidate unspecified; 5b missing from 10.8 and runsheet | HIGH | Medium | section 7 | State pool = same for both arms, or report both | Plan 10.8, Runsheet | none |
| F-11 | L0 vote and candidate LOAO runs refused by current registry | HIGH | High | executed check | Extend registry (lead) | builds.py, train_v7.py | `check_args` tests per member |
| F-12 | Runsheet Phase 8 still Gate A2 primary with HALT; Domain Tier B/D wording vs 9.3 "PAD classification track" with no spec | MEDIUM | High | text | Align to 10.4; PAD reported-only | Runsheet, Domain | none |
| F-13 | P3 Macro-F1 arm "not estimable" lets pAUC alone decide | MEDIUM | Medium | `p3_holdout` | Require both arms or declare rule | promotion.py (lead) | unit test |
| F-14 | 11.2 "case mix, not transfer" and 13.3 "not archive mix" over-state; xAUC intervals are 95% although F4 mandates 97.5% for under-40 gain claims | MEDIUM | Medium | JSON | Hedge; add 97.5% numbers | Plan | none |
| F-15 | Numbers drift: 656/1,476/1,255 (10.6) vs 681/1,475/1,253 (JSON) | LOW | High | JSON | Update text | Plan | doc-vs-JSON check |
| F-16 | Unsupported safety language and "Certified" statuses (section 6 list); conformal as CORE_SAFETY vs S65; P4 under-40 band does not meet drop rule; no decision row for P4; no AMPS code | MEDIUM | High | grep | Replace with measured/exact-interval wording; reconcile conformal | Stat, Runsheet, README, Metadata, OpenQ, Plan | none |
| F-17 | Verification bias not disclosed where MIDAS will be interpreted | MEDIUM | High | grep | Add limitation sentence in 10.11 | Plan | none |
| F-18 | Cross-cohort MIDAS comparator (0.865/0.814) vs Plan 4.1 rule | LOW | Medium | 11.2 | Word as anchor only | Plan | none |
| F-19 | `seal.json` stores absolute local `labels_path` (published per 13.9) | LOW | High | seal.json | Redact before publishing | seal.json | none |
| F-20 | `external_read.py` is not in the Gate 0 hashed list (lead's file) | INFO | High | `plan_freeze.FROZEN` | Lead to decide | - | - |

## 10. Unresolved
- Whether MIDAS under-40 melanoma/escalating positives are enough for anything (unknown while sealed; cannot be checked under the boundaries).
- Whether V6 fold checkpoints for all 5 folds x 4 members exist for a D2 re-score (only some `anchor-v6_*` files confirmed).
- Whether the "keypoint audit before predict" has been run (no receipt seen).
- Section 13 content was not audited (lead editing); sections 1-12 line numbers should be re-checked after the lead's edits.
