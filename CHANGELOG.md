# CHANGELOG

Purpose: a single document from which the entire project can be reconstructed and defended in a
viva. Every number below carries the `results/` (or `research/*/results*/`) file it came from —
per the project's Hard Rule 4, nothing here is hand-entered. Where CLAUDE.md's running log and a
`results/` file disagreed, the `results/` file wins and CLAUDE.md has been corrected alongside
this document.

---

## 1. How to read this project

Four hard rules govern every experiment, and most viva questions collapse into "why does this
rule exist":

1. **All splits are grouped by `lesion_id`.** HAM10000 has multiple images per lesion (different
   angles/magnifications of the same mole). Splitting by image would let the same lesion appear in
   both train and test, so the model could partly memorize test lesions from train — that is
   leakage, not generalization. `assert_no_leakage()` checks this on every split file.
2. **The test set is touched once per experiment.** Every fitted quantity (ensemble weights,
   calibration maps, thresholds, temperatures, abstention cutoffs) is chosen on validation (or, in
   session 6, OOF) and only *evaluated* on test. Reading test twice and keeping the better number
   is test-set selection — the paper's single most repeated failure mode to avoid, and the reason
   rung A5 reports uniform soft-vote (0.7718) instead of ridge stacking (0.7815): ridge has the
   better test score but the *worse* val_oof score (0.7583 vs 0.7911), so picking it would mean
   using test performance to choose the model — see §4.
3. **Never accuracy.** `nv` (benign nevus) is 67% of the data; a classifier that always predicts
   `nv` scores ~67% accuracy while missing every escalating case. Macro-F1, Balanced Accuracy, and
   Escalation Sensitivity are the reported metrics because they weight all seven classes equally.
4. **Every number in the manuscript resolves to a file under `results/`.** `audit_manuscript.py`
   enforces this mechanically (83 checks as of the post-compile pass).

---

## 2. Data and splits

- **HAM10000** (dermoscopy, 7 classes): 10,015 images / 7,470 lesions.
  Train 6,981 images, val 1,532, test 1,502 — lesion-grouped, no overlap.
  Per-class train/val/test: `akiec` 222/53/52, `bcc` 361/82/71, `bkl` 772/160/167, `df` 71/24/20,
  `mel` 773/173/167, `nv` 4683/1018/1004, `vasc` 99/22/21.
- **PAD-UFES-20** (smartphone clinical, 6 classes): 2,106 manifest rows, 1,302 with Fitzpatrick
  skin-tone labels. Images are not present locally in this repo copy — restoring them is
  Workstream B of the current plan.

## 3. Phase 0 — six frozen CNN baselines

Two-stage schedule per architecture (3 head epochs frozen backbone, then full fine-tune),
effective-number class weighting (Cui et al. 2019), seed 42, monitored on val Macro-F1.
ResNet-50 0.7058 (worst) → ConvNeXt-Tiny 0.7459 (best of the six, `results/ablation_table.csv`
rows A1/A2). These six checkpoints are the frozen base for every downstream ensembling result.

## 4. Phase 1 / session 1 — ensembling and diversity (`research/ensembling/results/report.md`)

Diversity audit on val: mean pairwise disagreement 0.1483, mean Yule's Q 0.8986, mean double-fault
0.0989 — the six CNNs disagree enough to be worth combining.

| Method | val_oof Macro-F1 | test Macro-F1 |
|---|---:|---:|
| nonneg_stacking_ridge | 0.7583 | 0.7815 |
| **soft_vote_arithmetic (reported)** | **0.7911** | **0.7718** |
| nelder_mead_simplex | 0.7911 | 0.7718 |
| caruana_greedy | 0.7798 | 0.7688 |
| convnext_tiny_baseline | 0.7482 | 0.7459 |
| rank_average | 0.5047 | 0.5074 |

**Why uniform soft-vote is reported and ridge stacking is not, despite ridge's higher test score**:
ridge's val_oof (the number chosen *before* looking at test) is 0.7583, the *worst* of the
competitive methods — it overfits the OOF fold structure. Soft-vote's val_oof is 0.7911, the best.
Reporting ridge would mean the test read, not the validation-time selection rule, picked the
winner — the project's central teaching example of test-set selection, and reproduced deliberately
here so it can be pointed to directly.

Winner vs. ConvNeXt-Tiny baseline on test: bootstrap 95% CI 0.7697 [0.7319, 0.8073]; McNemar
chi2=17.203, p=3.36e-05 (85 images only the ensemble got right vs. 38 only the baseline got right).

## 5. Phase 2 / session 2 — TTA, calibration, thresholds, DCA (`research/calibration/results/session2_report.md`, `research/tta/results/report.md`)

- 24-view TTA lifts the soft-vote ensemble to test Macro-F1 **0.7859**.
- Calibration (fit on val, TTA ensemble): uncalibrated ECE 0.1547→ temperature 0.0338 →
  matrix scaling 0.0263 → **Dirichlet 0.0201 (best by val ECE)**, test ECE 0.0206, test Macro-F1
  **0.8047**.
- **Why Dirichlet beats a single temperature**: temperature scaling applies one scalar to every
  class simultaneously, but the ensemble's miscalibration is class-dependent (see §11 — the
  ensemble is under-, not over-, confident, and by different amounts per class). Dirichlet fits a
  full linear map in log-probability space and can express that; one scalar cannot.
- Cost-sensitive thresholds (fit on Dirichlet-calibrated val probabilities) were **demoted off the
  Macro-F1 ladder in session 5**: they trade Macro-F1 0.7718→0.7490 (−0.062) to buy higher
  escalation sensitivity, and inside the set of images retained at 20% margin-abstention, they add
  zero effect — abstention already absorbs the risk they were built to manage.
- DCA: net benefit at treatment threshold p_t=0.10 — ensemble 0.1577 vs. ConvNeXt-Tiny baseline
  0.1559 vs. Treat-All 0.1034.

## 6. Phase 3 / session 3 — transformers, losses, fusion (`results/ablation_table.csv` rows A3/A4)

SwinV2-Tiny test Macro-F1 0.7273 (val 0.7176). MaxViT-Tiny test 0.7525 — the best *single* model on
test, but **not the reported single-model pick**, because its val score (0.7176) loses to
ConvNeXt-Tiny's val score (0.7482); picking MaxViT would again be test-set selection (rung A2 in
`results/ablation_table.csv` is `convnext_tiny`, re-confirmed in session 5's Part A specifically to
resist this trap). LDAM-DRW 0.7256, ASL 0.7305, gated fusion val 0.7643 / test 0.7411
(`results/ablation_table.csv` row A4).

## 7. Phase 4 / session 4 — selective classification, conformal, fairness

### Selective (`research/selective/results/session4_report.md`)

Uncertainty scores ranked by val AURC (lower is better; margin selected):

| Score | val AURC | test AURC |
|---|---:|---:|
| **margin (selected)** | 0.02563 | 0.03109 |
| msp | 0.02671 | 0.03084 |
| entropy | 0.02714 | 0.03109 |
| entropy+mahalanobis | 0.02795 | 0.03245 |
| mahalanobis | 0.03558 | **0.03965 (worst)** |

**Why Mahalanobis ranks last**: it answers "is this input unlike anything in training?" — a
distribution-shift question. HAM10000 test is drawn from the same acquisition pipeline as train,
so there is no shift for it to detect; it ends up ranking *in-distribution* difficulty worse than
the probability vector does, because ambiguous-but-in-manifold lesions are exactly what it is
blind to. Combining it with entropy makes entropy *worse*. It would need PAD-UFES-20 (a genuine
domain shift) for a fair test — Workstream B in the current plan.

Risk-coverage (argmax rule, margin abstention): at 10% target abstention (89.2% achieved coverage,
n=1340), Macro-F1 0.8577, escalation sensitivity 0.7621, 54 missed serious.

### Conformal (`research/conformal/results/session4_conformal_report.md`)

Marginal LAC at alpha=0.10 gives 90.4% overall coverage (the guarantee "holds") while covering only
74.8% of genuinely malignant lesions and leaving **63 with a prediction set containing no
escalating class at all** — the average is propped up by the 67% of cases that are `nv`. Mondrian
class-conditional calibration fixes this: **LAC 91.4% coverage on serious / 17 false reassurances,
RAPS 94.1% / 6 false reassurances** (the best configuration by this metric). *(These are the
correct, current numbers — the log below in §14 records that an earlier CLAUDE.md entry had stale
figures of "89.3%/24, 92.4%/12" from before the L1 Dirichlet-half-split fix.)*

`df` and `vasc` cannot certify a class-conditional threshold at alpha=0.05 (≈11 calibration images
each, below the ~19 needed) — their thresholds sit near the max observed score and the class is
included in almost every set regardless of model belief. Documented as a scarcity artifact, not
silently smoothed over.

### Fairness (`research/selective/results/session4_report.md`)

At 10% abstention: sex gap is modest (Macro-F1 gap 0.031, TPR gap 0.085). Age-band gap is severe
(Macro-F1 gap 0.495, TPR gap 0.766) — driven entirely by the `<40` group (§11).

## 8. Phase 5 / session 5 — ablation, statistics, manuscript

11-rung ladder (7 Block A, 4 Block B), 1000× lesion-grouped bootstrap CIs, McNemar, DeLong+Holm
(`results/ablation_table.csv`, `results/bootstrap_cis.json`, `results/mcnemar_delong.json`).

**Why McNemar found the ensembling gain significant (p=3.4e-05) while the Macro-F1 bootstrap CI
still crosses zero**: McNemar tests raw per-image correct/incorrect, which is dominated by the `nv`
majority class (67% of test); Macro-F1 weights all seven classes equally, so the rare-class
variance widens the CI even though the ensemble wins on the vast majority of individual images.
Both are correct; they answer different questions.

**DeLong was broken and is fixed.** `_fast_delong_structural_components` computed pooled midranks
on the original case order while indexing as `[positives|negatives]`, making all 42 per-class AUCs
come out ≈0.5. Fixed by ranking the reordered array; verified against `sklearn.roc_auc_score`
exactly, ties included. Post-fix AUCs range 0.73–0.9999; with Holm–Bonferroni across the 42 tests,
**1 survives**: A5 vs. A2 on `bkl`, AUC 0.9195→0.9572, z=3.61, Holm p=0.0129 — corroborates the
McNemar ensembling result independently.

## 9. Post-compile revision pass (from the Overleaf-compiled PDF)

Four errors found only by reading the *rendered* output — none of them would have been caught by
`audit_manuscript.py`'s numeric checks alone:

1. **Calibration direction was backwards** in an earlier draft — see §11, corrected everywhere.
2. Fig. 6 caption claimed a subgroup breakdown the figure does not contain (it's aggregate-only) —
   rewritten to point at Table IV instead.
3. Fig. 5 caption quoted LAC's worst class (mel 0.671) while the figure actually plots APS
   (mel 0.796) — both now named.
4. Fig. 4 caption said `margin` "dominates" the other scores — it is statistically tied with MSP
   and entropy, and MSP is marginally *better* on test AURC (0.03084 vs 0.03109) — softened.

## 10. Session 6 (current round, in progress) — see the active plan for full detail

### S1 — A.0 bug fixes + fold assignment (2026-09-04)

**Finding, not new work**: on starting S1, `research/oof/make_folds.py`, `research/oof/train_folds.py`,
the five materialized fold files under `ml/configs/splits/oof/`, `ml/configs/splits/split_v1.folds.csv`,
and `ml/results/oof_fold_report.md` already existed on disk (file timestamps 04:01–04:02, before
this session's `CHANGELOG.md` was even created at 04:27) — this round's A.0/A.1 work was done in an
untracked prior pass and never logged. This entry is that missing log, written after independently
re-verifying every claim rather than trusting the artifacts on their face:

- **All three A.0 latent bugs are already fixed**, confirmed by reading the code directly:
  1. `research/tta/extract_tta_predictions.py` has `--out-name` (line 116) plumbed through
     `extract_one` (line 104: `out_dir / (out_name if out_name else f"{arch}_{split}.csv")`) — an
     OOF extraction run can no longer silently overwrite `research/predictions_tta/{arch}_test.csv`.
  2. `ml/training/train.py:397-398` tags the training-history filename with `--checkpoint-tag`
     (`{arch}{history_tag}_training_history.json`) — fold runs no longer clobber each other's or the
     original six checkpoints' history files.
  3. `research/run_session4_selective.py:61-64` — `_load_matrices`'s fallback from TTA to plain
     predictions now hard-fails (re-raises) when `predictions_dir == "research/predictions"` or
     `"oof" in predictions_dir`, so a half-extracted OOF directory cannot silently emit val-fitted
     numbers under an OOF label.
- **A.1 fold assignment verified independently** (not just re-read): re-ran `assert_no_leakage()`
  against all five `ml/configs/splits/oof/split_v1.fold{k}.csv` files fresh in this session, checked
  each fold file's image set against the real test split (`ml/configs/splits/split_v1.csv`) for
  overlap (none), and checked `split_v1.folds.csv` for exactly one fold per lesion and its image-id
  union equalling the train split exactly (6,981) — all passed. `ml/results/oof_fold_report.md`'s
  yields (`df` 71, `vasc` 99, `<40` escalating 64) match the plan's pre-registered expected values
  exactly. `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on `class_index` /
  `lesion_id`, matching the precedent in `research/ensembling/oof.py`.
- **`train_folds.py` (A.2, S2 scope) already exists too** and was dry-run in this session
  (`python -m research.oof.train_folds --dry-run --archs convnext_tiny efficientnet_b0 --folds 0 1 2 3 4`)
  to confirm the pipeline is sane before handing S2 off: it deep-copies `load_training_config()`
  per fold (guards against the `lru_cache` singleton-mutation hazard the plan flagged), applies
  arch-specific batch sizing (16 for `convnext_small`/`efficientnet_b3`, 32 default), sets
  `checkpoint_tag=f"oof_f{fold}"`, and reports all 10 smoke-tested runs as `READY`. No fold
  checkpoints exist yet (`ml/checkpoints/oof/` absent) — S2 training has not been run.
- **One deviation from the plan spotted, not yet fixed (flagged for S2/S3, out of S1's scope):**
  `train_folds.py`'s `--dry-run` estimate table uses a hardcoded `RUNTIME_MINUTES_ESTIMATE` dict,
  not one computed from `ml/results/experiments.csv` as A.2 specifies — a Hard-Rule-4 deviation
  (hand-entered numbers) worth fixing before S2 is run for real, though it does not block a smoke
  test. Also unaddressed: the plan's `--prune-last` ordering requirement (prune only after that
  fold's OOF prediction file exists) — current code prunes immediately after training, before
  `extract_oof.py` (A.3, not yet built) can run. Not a live bug today because `--prune-last`
  defaults off, but will need fixing before it's used.
- Stale `CLAUDE.md` conformal numbers (89.3%/24, 92.4%/12 → 91.4%/17, 94.1%/6) were corrected in S0,
  ahead of schedule — see §7/§14.

S1 is complete: all three A.0 bugs verified fixed, fold assignment independently re-verified against
its own leakage/power checks rather than trusted from the pre-existing report, and one real (but
currently inert) deviation queued for whoever runs S2.

### S2 — training driver fixes + handoff (2026-09-04)

Before handing off the actual GPU commands, fixed the two issues flagged at the end of S1:

- **`train_folds.py`'s runtime estimate is now data-derived**, per A.2 and Hard Rule 4. It was a
  hardcoded `RUNTIME_MINUTES_ESTIMATE` dict (14–25 min/fold, guessed) that turned out **badly wrong
  for the two largest architectures**: real median full-train time on `split_v1.csv`
  (`ml/results/experiments.csv`, `train_time_seconds` grouped by `arch`) is **45.5 min for
  convnext_tiny** (hardcoded said 19) and **35.75 min for resnet50** (hardcoded said 17) — more than
  double for both. Replaced with `_runtime_minutes_by_arch()`, which reads the median
  `train_time_seconds` per arch from the six baselines' original full-HAM10000 runs and uses that as
  the fold-runtime estimate (fold trains on ~80% of the images with the same epoch schedule, so this
  is a deliberate, documented overestimate — conservative for GPU-hour planning, not tuned down).
  **Recomputed total for the full 6-arch × 5-fold run: 693.9 minutes (~11.6 GPU-hours)**, verified by
  re-running `--dry-run` — lower than the plan's original ~18–20h hand-estimate, and now backed by a
  file instead of a guess.
- **`train_folds.py` now logs each fold to `research/experiments.csv`** with `session="session6_oof"`,
  `method=f"{arch}_oof_f{fold}"`, `split="val"` (the fold model's score on the untouched global
  validation split — not an OOF score, since OOF scoring needs `extract_oof.py`, A.3/S3, not built
  yet), and the checkpoint path + wall-clock seconds in `notes`. This was silently missing from the
  otherwise-complete driver and is required by A.2; without it the 30 fold runs would have produced
  checkpoints with no research-ledger provenance at all.
- Both changes verified with `--dry-run` after editing (import sanity, no crash, correct arithmetic).

**Handed off**: the actual training commands (smoke test → ConvNeXt-Tiny stage → two overnight
batches), per the plan's execution order. GPU training itself is run by the user, not this session —
see the chat for the exact PowerShell commands given.

**Runtime-estimate correction, mid-flight (2026-09-04)**: the "data-derived" estimator above still
badly overestimated real GPU-hours -- it assumes this machine trains at the *same* per-epoch speed
as whatever hardware produced the original six baselines. It does not: calibrating against the six
OOF runs actually completed on this machine (`efficientnet_b0` 24.1 s/epoch, `convnext_tiny`
~31–34 s/epoch, at fold image counts) against the matching original-baseline rows, the two machines
are roughly the same speed on average (~1.0x), not ~4x apart as first guessed from a single outlier
row (`convnext_tiny`'s 143 s/epoch original row, which its own sibling row of 55 s/epoch contradicts).
Recalibrated central estimate for the remaining 24 folds: **~7.0 GPU-hours** (range 4.4–8.7h
depending on how many folds hit early stopping early vs. run the full 30 epochs), not the printed
11.6h. The printed dry-run number is now known to be a mild overestimate and is left as-is
(conservative is the right direction for unattended overnight planning) rather than re-tuned again
mid-session on a six-data-point sample.

**Crash isolation added before the unattended overnight run.** `train_folds.py`'s per-fold loop had
no exception handling -- one CUDA OOM or transient fault would kill the entire queue and strand
whatever architectures hadn't started yet, unacceptable for a ~7h unattended run. Added: a `try`
around `train(cfg, args)` that logs a `FAILED` row to `research/experiments.csv` (session6_oof) and
continues to the next fold rather than raising; an end-of-run summary block that globs
`ml/checkpoints/oof/*-oof_f*_best.pt` and prints the failure list so the morning check is a single
glance at the bottom of the log. Not exercised against a real failure (none occurred in this run),
but the control flow was read carefully, not just added on faith.

### S3 prep, written ahead of GPU need (2026-09-04)

While the overnight training queue runs, wrote the two pieces of S3 (OOF extraction) that don't
need a GPU to write or statically verify, plus the independent S8a (PAD-UFES-20 + segmentation mask
download) script -- so tomorrow's session can run them immediately instead of waiting on writing code.

- **`research/oof/extract_oof.py`** (A.3). Two modes: `--mode plain` (single deterministic view,
  mirrors `research/extract_predictions.py`, for the fast pre-TTA diagnostic) and `--mode tta`
  (drives `research.tta.extract_tta_predictions.extract_one` directly -- reused, not
  reimplemented, per the plan's "one TTA implementation" instruction). Stages per-fold predictions
  under `<out-dir>/_folds/fold{k}/{arch}_train.csv`, then assembles into `<out-dir>/{arch}_train.csv`
  with the three assertions the plan specifies: exactly 6,981 rows, `set(image_id)` equals the train
  split exactly (no duplicates), and every row's class probabilities sum to 1 within 1e-4. Output
  goes to **new** directories (`research/predictions_oof(_tta)`), never the frozen
  `research/predictions(_tta)` that `results/frozen_artifacts.json` hashes. Writes
  `results/oof_provenance.json` (per mode/arch/fold: checkpoint path, SHA256, row count, val
  macro-F1 and epochs pulled from `ml/results/experiments.csv`) -- kept separate from
  `frozen_artifacts.json` on purpose, since it documents fold-checkpoint provenance, not a frozen
  result set. Verified with syntax parse, module import, and `--help` (no GPU calls yet -- avoided
  running any inference while the training queue was live, to not contend for the GPU).
- **`research/oof/diagnose_shift.py`** (A.4, "run before trusting anything downstream"). For each
  arch, runs all 5 fold checkpoints over the identical global val split (present unchanged in every
  fold file), probability-averages the 5 folds ("fold-bagged"), and compares macro-F1 / mean
  max-probability / ECE against the frozen full-train model's own plain val predictions
  (`research/predictions/<arch>_val.csv` -- deliberately non-TTA on both sides, so a TTA-vs-plain
  confound doesn't get mistaken for the fold-vs-frozen gap this script exists to measure). Also
  doubles as the health check that every fold checkpoint loads. Writes
  `research/oof/results/diagnose_shift_report.md`. Same verification level as above (syntax, import,
  `--help`) -- not run against real checkpoints yet since that needs the GPU the training queue is
  using.
- **`scripts/download_pad_ufes.py`** (S8a), fully run and verified, not just written -- this one
  doesn't touch the GPU. Both source APIs were queried live (via `WebFetch`) while writing it,
  rather than guessing hardcoded download URLs (the exact risk the plan calls out for this task):
  - **PAD-UFES-20** (Mendeley DOI 10.17632/zr7vgbcyr2.1): the file list, SHA256 hashes, sizes, and
    download URLs are fetched at *run time* from `https://data.mendeley.com/public-api/datasets/zr7vgbcyr2`,
    not hardcoded -- confirmed live: 4 files (`imgs_part_{1,2,3}.zip` + `metadata.csv`), **3.59 GB
    total**, matching the plan's "~3.5 GB" estimate almost exactly.
  - **HAM10000 segmentation masks** (Harvard Dataverse, same DOI as HAM10000, `10.7910/DVN/DBW86T`):
    file id and MD5 resolved at run time from Dataverse's dataset API (`.../api/datasets/:persistentId/`)
    by filename match, not a hardcoded file id -- confirmed live: `HAM10000_segmentations_lesion_tschandl.zip`,
    10,808,743 bytes, MD5 `6e8d252e09cfdb0189199f15985a5b84`.
  - **Actually executed and verified** (not just smoke-tested in isolation): downloaded and
    SHA256/MD5-verified `metadata.csv` and the segmentation-mask zip for real, then ran the full CLI
    path (`python -m scripts.download_pad_ufes --skip-pad`), which extracted **10,016 PNG masks**
    into `data/ham10000/HAM10000_segmentations_lesion_tschandl/` -- one more than HAM10000's 10,015
    images; not investigated further here, flagged for whoever consumes the masks in Workstream D to
    check for a duplicate or an off-by-one before trusting a per-image join. Also caught and fixed a
    real bug during this verification: `.relative_to(REPO_ROOT)` calls in the print helpers would
    raise `ValueError` on any path outside the repo (harmless in normal use since `resolve()` always
    returns repo-relative paths, but crashed the first smoke test that used a real temp directory) --
    replaced with a `_rel()` helper matching `prepare_pad_ufes.py`'s existing pattern, which falls
    back to the absolute path instead of raising.
### S3 — three bugs in the frozen TTA path, found by running it (2026-09-04)

`research/tta/extract_tta_predictions.py` — **the script that generated the entire frozen
`research/predictions_tta/` set that rungs A6/A7 and every downstream result depend on — was
completely non-functional** and had been for some time. It has exactly one caller-side entry point
and that call could not execute at all:

1. **`predict_tta(dataset=...)` — wrong keyword.** The parameter in `research/tta/predict.py` is
   `base_dataset`. `TypeError` on every invocation, for every arch and split.
2. **`predict_tta(description=...)` — parameter did not exist.** The caller had always passed a
   per-arch/split progress label; `predict.py` had a hardcoded `desc="TTA predicting"` instead, so
   the argument was invalid *and* the intended labelling was silently lost.

Both are consistent with `predict.py` having been refactored after the frozen TTA predictions were
generated, without updating its single caller. **The published TTA artifacts could not be regenerated
by their own script** — a reproducibility hole in the frozen-artifact declaration that no numeric
audit would surface, since `frozen_artifacts.json` hashes the output CSVs, not the source that makes
them. Fixed by restoring `description` as an optional parameter on `predict_tta` (default preserves
prior behaviour) and correcting the keyword to `base_dataset`.

3. **`_RawImageDataset` asserted on `.transform` of a `Subset`.** Surfaced only by writing an
   8-image smoke test before committing GPU hours: `--limit` wraps the `LesionDataset` in a
   `torch.utils.data.Subset`, which carries no `.transform`, so the module's *own documented*
   smoke-test usage (`--limit 32 # smoke test` in its docstring) raised `AttributeError`. Does not
   affect full runs (`limit=None`). Fixed by unwrapping nested `Subset`s before the assert.

Verified by running the real TTA path end-to-end on 8 images: 8 rows out, probability rows summing to
1.0, correct per-arch progress label. **Runtime estimate corrected while measuring**: TTA sustains
~3.6 images/s against the plain path's ~74 images/s (24 views per image), so the full 6-arch OOF TTA
extraction is **~3.5–4 hours, not the 2.5 h previously quoted**.

**Reproducibility of the frozen TTA artifacts: verified, not assumed.** Finding bugs 1–2 raised a
fair question the repo could not answer from history (this copy has no git): was the `predict.py`
refactor *behaviour*-preserving, or did the pooling math change after
`research/predictions_tta/` was generated? Settled empirically rather than argued — re-ran the
repaired TTA path with the **frozen full-train** `convnext_tiny` checkpoint over 200 val images
(val, not test, so Hard Rule 2 is not even in question) and compared against the frozen
`research/predictions_tta/convnext_tiny_val.csv`:

| Metric | Value |
|---|---|
| Max absolute per-class probability difference | **3.94e-06** |
| Mean absolute difference | 7.69e-08 |
| Argmax agreement | **100% (200/200)** |

Differences are at float32/cuDNN nondeterminism scale, not behavioural. **The refactor was
behaviour-preserving and the frozen TTA predictions are exactly reproducible by current code.** The
reproducibility claim in Methods §III-G is now verified rather than merely asserted — a stronger
position than before these bugs were found.

**Impact on session work: none.** The `TypeError` fired on the first call before any image was
processed, so no GPU time was lost; the 30 fold checkpoints, the plain OOF matrices, and the
`diagnose_shift` report were never at risk (the plain path uses `predict_split`, an entirely separate
code path). Nothing required retraining or re-extraction.

### S3 complete — OOF extraction, shift diagnostic, PAD restored (2026-09-04)

**Both OOF prediction matrices built and verified through the real loader contract**, not just
written to disk: `research/predictions_oof/` (plain) and `research/predictions_oof_tta/` (24-view
TTA) each load via `load_split_matrix("train", predictions_dir=...)` as `(6981, 6, 7)` with 6,981
unique image ids over 5,229 lesions, probability rows summing to exactly 1.0, zero NaN. Every train
image scored exactly once by the fold model that never saw it. `results/oof_provenance.json` carries
both modes × 6 archs × 5 folds with per-checkpoint SHA256, epochs and fold val macro-F1.

Runtime came in at roughly **1.75 h**, against my measured-from-smoke-test estimate of 3.5–4 h —
the smoke test's 8-image batch was dominated by warm-up and understated steady-state throughput
(sustained 4.4–8.4 img/s, arch-dependent, vs the 3.6 img/s extrapolated).

**A.4 shift diagnostic (`research/oof/results/diagnose_shift_report.md`) — two findings:**

1. **Fold-bagging beats the frozen single model on 5 of 6 architectures** (ΔF1 +0.020 to +0.037;
   only `efficientnet_b0` loses at −0.020; mean +0.023). The plan predicted fold models would be
   "systematically slightly weaker" — on macro-F1 they are not. `convnext_tiny` fold-bagged reaches
   **0.7847** on val, from five folds of a *single* architecture, against the published six-architecture
   soft-vote's 0.7911. Direct val-measured support for rung **A8** (30-member fold-bag), which the
   plan called the one remaining credible Macro-F1 path after four post-hoc levers came back dead.
2. **The stacking mismatch is real and in the predicted direction**: fold-bagged is less confident on
   all 6 archs (Δconf −0.025 to −0.075) and worse calibrated on all 6 (ECE 0.135–0.177 vs frozen
   0.097–0.147).

> **Interpretation caveat to carry into S4 — the report overstates the mismatch.** The fold-bagged
> Δconf conflates (a) fold models trained on 80% of data being individually less confident, which is
> what actually biases an OOF-fitted calibrator, with (b) averaging five models pulling max-prob down
> — the same under-confidence mechanism this paper already documents for the 6-CNN ensemble. But
> `predictions_oof/{arch}_train.csv` holds **single** fold-model predictions per image, not 5-model
> averages, so (b) is absent from the matrices a calibrator will actually be fitted on. From the
> per-fold values the run printed, the single-model gap is ≈ −0.004 (`efficientnet_b3`), −0.009
> (`densenet121`), −0.020 (`resnet50`), −0.045 (`efficientnet_b0`) — averaging about **−0.02**
> against the report's headline **−0.051**. The bagged framing roughly doubles the apparent
> over-sharpening risk. Add a single-model column to that report in S4 so the caveat written into
> Methods reflects the real magnitude. This is a limitation of the diagnostic as A.4 specified it
> (it asked for the 5-fold average), not a bug.

**S8a done in parallel (network-bound, ran alongside the GPU work).** PAD-UFES-20 restored:
3.59 GB over 4 files, each SHA256-verified against the Mendeley manifest, extracted to
`data/pad_ufes_20/`, and `ml/data/manifest_pad.csv` rebuilt — **2,106 rows, 1,302 with Fitzpatrick
labels, image paths resolving**. Class distribution confirms the prior-shift premise the post-S11
plan's Workstream E2 is built on, with real numbers rather than the estimates quoted there:
`bcc` 845 (40.1%), `akiec` 730 (34.7%), `nv` 244 (11.6%), `bkl` 235, `mel` 52 (2.5%) — against
HAM's 67% `nv`. **Note for E2/E5: PAD holds only 52 melanomas**, which bounds any melanoma-specific
claim on PAD regardless of what prior correction recovers.

### S4 — downstream rewiring: `--fit-split` across the runners (2026-09-04)

Workstream A.6. Four runners gained a selectable fitting split so every val-fitted parameter in
the project can also be fitted on the 6,981 OOF training rows S3 produced, **without any existing
default changing and without any published output path being written to**. Verified: the 34
prediction-matrix hashes in `results/frozen_artifacts.json` are unchanged, `run_part_a.py
--table-only` still regenerates the 11-rung ladder, and `audit_manuscript.py` still passes all 83
checks.

**The rule, made uniform rather than per-runner.** Under `--fit-split oof` every *parameter*
(calibrator coefficients, abstention quantiles, decision thresholds, conformal quantiles) is fitted
on OOF, while every *method selection* (which calibrator family, which uncertainty score) stays on
**validation**. The plan stated this for session 2 only; applying it to all four means there is one
sentence to defend rather than four. The reasoning is the same in each case: selecting a method on
the same OOF rows its coefficients were fitted on would relocate the in-sample problem, not fix it.
Implemented once in `research/fitsplit.py` (`FitPlan`, `resolve_fit`, `write_fit_state`).

**Two hazards not in the plan, found while reading the code, that would have cost S5 and S9:**

1. **`run_comparison.py` as specified would have read test twice.** All four runners fit *and*
   read test in a single pass, so "run each downstream script twice" means two extra test reads —
   before S9, whose entire purpose is that nothing reads test until the pre-registered pass. Fixed
   with a `--no-test` fit-only mode on every runner, and the guarantee made **mechanical** rather
   than documentary: `research/testguard.py` is a process-wide lock that
   `research.ensembling.data.load_split_matrix` and `research.selective.features.load_features`
   consult before touching disk, raising `TestSplitLocked` rather than warning. S9's sanctioned
   read goes through `testguard.test_unlocked(...)`. Negative-tested: 11/11 guard cases raise as
   intended, including that the lock is restored after the context manager exits.
2. **The Mahalanobis score is doubly in-sample on the OOF path.** Its Gaussians are fitted on train
   features extracted from the *full-train* checkpoints, so scoring OOF training rows with it means
   the model saw those images in training *and* the Gaussians were fitted on those exact features.
   The resulting abstention quantiles would be meaningless. It is now disabled by default under
   `--fit-split oof` (`--allow-oof-mahalanobis` overrides, and the report states which applied).

**A third hazard the plan named only half of.** A.6 said an OOF run must not wear the published
ledger `session`. The same is true of a *val* fit-only run: it writes rows with no test split
behind them, and under the old default it would have inherited `session2`/`session4` and become
indistinguishable from published rows in `research/experiments.csv`. `resolve_fit` now refuses
both, and the fit-only val arm logs as `session2_valfit` / `session4_valfit`.

**Regression guard, built because the obvious one requires a test read.** Re-running each runner on
its defaults and diffing the report would prove the published path survived the refactor — and
would read test, which this round may not do. `research/oof/check_valfit_regression.py` instead
recomputes every val-side fitted quantity longhand in its pre-refactor form and asserts bit-level
agreement (atol 1e-12) with what the refactored runners write to `fit_state.json` under
`--fit-split val --no-test`. **59 quantities compared, all agree** — calibrator weights and biases,
the ECE-selected calibrator, cost-sensitive threshold vectors, the AURC of all eight candidate
uncertainty scores, all five abstention quantiles, and all 12 conformal quantile vectors plus their
degenerate-class sets. Since test enters those runners only as a straight-line *application* of
these objects, the published path is intact. The check writes under `session6_regress` and prunes
its own ledger rows on exit — a regression check is not an experiment.

**Changed files.** New: `research/testguard.py`, `research/fitsplit.py`, `research/oof/__init__.py`,
`research/oof/run_comparison.py`, `research/oof/check_valfit_regression.py`. Rewired:
`research/run_session2_calibration.py` (argparse built from nothing — it had none),
`research/run_session4_selective.py` (the five hardcoded `"session": "session4"` strings at 193,
219, 256, 272, 291 are now `plan.session`), `research/run_session4_conformal.py` (its own
`--session`/`--out-dir` replaced by the shared flags; the hardcoded caveat that OOF "would roughly
quintuple" rare-class calibration is now computed and correct in both directions).
`research/thresholds/optimize.py`: `ThresholdState.val_cost`/`.val_specificity` →
`fit_cost`/`fit_specificity`, plus `optimize_thresholds_by_group(..., min_group_positives=30)`,
`GroupThresholdState` and `apply_group_thresholds` — the home for S5's age rule. Additive guards in
`research/ensembling/data.py` and `research/selective/features.py`; `ARCHS` untouched, as the plan
requires.

#### What the OOF fit actually changed (`results/oof_vs_val_comparison.csv`, `paper/tables/oof_vs_val.tex`)

Produced by `python -m research.oof.run_comparison`, which runs all three runners in both arms
under the test lock. 26 rows. Every val-fitted diagnostic below is **in-sample** (fitted on val,
measured on val) while every OOF-fitted one is measured on data it never saw — the CSV carries a
`val_fit_in_sample` column so the two are never read as like-for-like.

| Quantity | val-fitted | OOF-fitted | Read |
|---|---:|---:|---|
| rows behind the fit | 1,532 | 6,981 | 4.6x |
| selected calibrator | dirichlet | dirichlet | unchanged |
| val ECE of the selected calibrator | 0.0201 (in-sample) | **0.0210** (held out) | OOF calibration generalises essentially as well |
| val Macro-F1 after calibration | 0.7825 (in-sample) | **0.7719** (held out) | −0.011 — the stacking mismatch, in the predicted direction |
| cost-sensitive fit specificity | 0.8775 | 0.8626 | both clear the 0.85 floor |
| conformal calibration rows | 762 | 3,468 | |
| rarest class's calibration points | 11 | **45** | alpha=0.05 needs >=19 |
| degenerate class-conditional cells at alpha=0.05 | **6** | **0** | the headline OOF win |
| degenerate cells at alpha=0.10 | 0 | 0 | already fine |
| exact finite-sample guarantee | yes | **no** | five score functions, not one |
| selected uncertainty score | `margin` | **`msp`** | see below |
| val AURC of the selected score | 0.0256 (in-sample) | 0.0308 (held out) | |

**Three findings S5/S6/S9 must not be surprised by.**

1. **A7-oof may not be a Macro-F1 win, and should be pre-registered two-sided.** A.5 frames rung
   A7-oof as "the pure benefit of un-overfitted calibration". Measured on val, the OOF-fitted
   Dirichlet costs **0.011 Macro-F1** against the val-fitted one — and that comparison already
   favours the val arm unfairly, since its number is in-sample. ECE is a wash (0.0210 vs 0.0201).
   So the honest prior is that OOF calibration buys *calibration sample size*, not accuracy, and
   `results/analysis_plan.json` should say so before the test read rather than after.
2. **The OOF fit changes which uncertainty score wins on validation**, `margin` → `msp`. The
   OOF-fitted Dirichlet map reshapes the calibrated probability geometry, and the score ranking
   moves with it. Two consequences: the abstention thresholds in the two arms sit on **different
   score scales and are not comparable as numbers** (the comparison CSV says so in its own `note`
   column, generated conditionally rather than hand-written), and S5's argument that "margin is a
   poor escalation detector" now has to be made against whichever score the pre-registered
   configuration actually selects.
3. **The alpha=0.05 rare-class problem is genuinely solved; the alpha=0.01 one is not, and the fix
   has a price.** Six class-conditional cells that could not certify a finite threshold at
   alpha=0.05 on val now certify on OOF, and `df`/`vasc` go from 11 to 45 calibration points. The
   price is stated in the OOF report and in `fit_state.json` (`exact_guarantee: false`): OOF scores
   come from five fold models, none of which is the full-train model that scores test, so split
   conformal's finite-sample guarantee does not transfer. Achieved coverage on val runs 0.876-0.905
   against a 0.90 nominal — close, but it must be **audited empirically and reported**, never
   asserted. CV+ / cross-conformal would restore a (1-2*alpha) guarantee and needs a second test
   read: out of scope under Hard Rule 2, noted as the follow-up.

**Test reads this session: zero.** No S4 run wrote a ledger row with `split=test` — the 32 new rows
carry only `train` and `val`, which is the mechanical check, not a claim. The single exception is
the negative test for the lock itself, which opened the frozen test prediction CSVs inside
`testguard.test_unlocked` to confirm the unlock path works and counted 1,502 rows; no model output
was scored against a test label and no quantity from it enters any artifact.

### S5 — the age-conditional escalation rule (2026-09-04)

Workstream A.7. Session 4 measured the under-40 blind spot but could not act on it, because a
band-conditional operating point has to be fitted somewhere and **validation carries 22 escalating
cases in that band**. The OOF predictions S3 produced carry **64**. That single count is what makes
this session possible, and it is the retrospective justification for the whole S3/S4 OOF detour:
the age rule is not merely *better* fitted out-of-fold, it is **unfittable on validation at all**
under the project's own `min_group_positives = 30` gate.

New: `research/agerule/lambda_rule.py`, `research/agerule/__init__.py`,
`research/run_session5_agerule.py`. Outputs under `research/agerule/results_oof/`
(`age_rule_lambda.json`, `session5_agerule_report.md`, 5 CSVs, `lambda_sweep.png`), 5 ledger rows
under `session6_oof`. **Test reads: zero** — the runner has no published val-fitted arm to
reproduce, so `--no-test` is its intended mode and it arms `research/testguard.py`.

#### The instrument, and why it is one number

The rule is `argmax_c ( p_c + lambda * 1[c escalates] )`, one scalar per age band. That is exactly
`research.thresholds.optimize.apply_thresholds` with `theta_c = -lambda` on the escalating classes,
so it stays inside the decision-rule family the project already defends, restricted to its one
clinically meaningful direction. `optimize_thresholds_by_group` — built in S4 and named there as
"the home for S5's age rule" — was **not** used for the headline: its own docstring warns that a
7-parameter search needs far more positives than 64, and that warning binds. It is fitted anyway,
as an ablation, so the choice is evidenced rather than asserted.

#### The diagnostic came first, and it changed the claim

Escalation-mass AUC (probability mass on `akiec`/`bcc`/`mel`, scored against the binary escalate
label) is threshold-free, so it separates a *decision-rule* failure from a *representation* failure.
Only the first is fixable by a threshold, and reporting a fix without this number would have
assumed the answer. Measured on the cross-fitted OOF ensemble:

| band | n | escalating | prior | argmax sens | escalation-mass AUC (95% CI) |
|---|---:|---:|---:|---:|---|
| `<40` | 1319 | 64 | 4.85% | 0.547 | **0.889** (0.831–0.949) |
| `40-59` | 3112 | 397 | 12.76% | 0.612 | 0.953 (0.936–0.965) |
| `60+` | 2512 | 893 | 35.55% | 0.733 | 0.933 (0.922–0.945) |

The `<40` band has the lowest argmax sensitivity **and** the lowest escalation-mass AUC. So the
under-40 deficit is **not a pure decision-rule failure**: part of it is a threshold a lambda can
move, and part of it is genuinely weaker ranking that no threshold recovers. The AUC interval
overlaps both other bands on 64 cases, so this is suggestive, not established — but it is the
opposite of the convenient result, and it caps what S9 and S14 are allowed to claim. The `<40`
argmax sensitivity here (0.547) also sits far above the published test figure (0.143, 3/21): the
test estimate rests on **21** positives and the two are different splits scored by different
models (fold models vs. the full-train model), so they are not like-for-like — but the OOF estimate
is the better-powered one and the paper should stop treating 0.143 as a precise quantity.

#### Fitted values (cross-fitted Dirichlet, cost-minimising, 0.85 specificity floor)

| band | lambda | fit sens (from) | referral rate (from) | bootstrap 95% CI | resamples selecting lambda=0 |
|---|---:|---|---|---|---:|
| `<40` | **0.26** | 0.625 (0.547) | 0.076 (0.052) | **[0.00, 0.61]** | **10.2%** |
| `40-59` | 0.74 | 0.897 (0.612) | 0.227 (0.102) | [0.67, 0.82] | 0.0% |
| `60+` | 0.33 | 0.833 (0.733) | 0.392 (0.318) | [0.22, 0.44] | 0.0% |
| `unknown` (2 positives) | 0.65 pooled fallback | — | — | not fitted | — |

**The lambda for the band this session exists for is the least certain one.** Its interval contains
zero and a tenth of lesion resamples select no rule at all, while both older bands' intervals
exclude zero. This is the direct consequence of 64 positives and it is the load-bearing caveat on
anything S9 measures: a null result on test is as consistent with this interval as a positive one.
S9 must pre-register it that way.

#### Held-out check on validation (the fit never saw these rows)

| band | sens (argmax → rule) | missed | referral rate |
|---|---|---:|---|
| `<40` | 0.545 → 0.591 | 10 → 9 | 0.064 → 0.097 |
| `40-59` | 0.550 → 0.817 | 27 → 11 | 0.080 → 0.202 |
| `60+` | 0.774 → 0.881 | 51 → 27 | 0.378 → 0.448 |
| **all** | **0.714 → 0.847** | **88 → 47** | 0.185 → 0.272 |

Validation Macro-F1 0.7638 → 0.7713. The rule helps in every band and helps **least** in `<40` —
exactly what the diagnostic predicted, which is the strongest evidence available that the
mechanism story is right. Referral rate is reported beside sensitivity everywhere, because a rule
that catches more melanoma by referring far more patients has moved cost onto that band rather than
removing it.

#### Session-4's abstention finding reproduced under a different score, on a different split

S4 warned that the OOF fit flips the selected uncertainty score `margin` → `msp`, so S5's "margin
is a poor escalation detector" argument had to be remade against whatever the pre-registered
configuration actually selects. Remade, and it survives: under **`msp`** at the pre-registered 10%
operating point (realised 10.6% on val), the share of argmax-missed escalating cases that
abstention refers anyway is **10.0% in `<40`** (1 of 10) against 22.2% and 41.2% in the older
bands. The published test figure under `margin` was 11.1%. The under-40 failure is *confidently*
wrong under both scores, on both splits — it is a property of the failure, not of the score.

#### Four bugs, three of them mine, found by running rather than by reading

1. **The lambda grid was the binding constraint, not the data.** At the initial `[0, 0.6]` grid the
   pooled fit and the `40-59` band both selected 0.59 — the largest value on the grid. Widening to
   `[0, 1.2]` moved them to 0.65 and **0.74**; `<40` and `60+` were interior and unchanged. A
   clipped parameter frozen into a pre-registration is unrecoverable after the test read, so the
   fix is mechanical, not documentary: `fit_lambda` now raises `GridBoundaryError` when the
   optimum sits at the grid maximum. Negative-tested. `bootstrap_lambda` passes `strict=False`,
   since a boundary hit on one resample is a property of that resample.
2. **`msp` was reimplemented instead of imported, inverting it.** `research/selective/scores.py`
   defines every score as an **uncertainty** (`msp` is `1 - max p`) with referral when the score is
   *above* threshold. The first draft of `_referral_of_misses` used `max p` and `<`, which reported
   a 0.7% abstention rate on val and would have supported a dramatic and entirely false claim that
   the pre-registered threshold fails to transfer. It transfers fine (10.6% realised against 10%
   nominal). The function now imports `predictive_scores` and `coverage_threshold`, and records the
   realised rate beside the nominal one so transfer stays a measurement.
3. **The 7-parameter ablation was being read on an incomparable scale.** Raw bootstrap SD is 0.137
   for the theta vector's escalating entries against 0.150 for the scalar lambda — which reads as
   the vector being *more* stable. But theta is searched over a range of 0.6 and lambda over 1.2.
   As a share of the range actually searched, the vector's spread is 0.229 against lambda's 0.125,
   about **1.8x wider from seven free parameters instead of one**. Both figures and the
   normalisation are now in `fit_group_vector_ablation` and the report states plainly that at this
   sample size *both* instruments are unstable — a difference of degree, not a disqualification.
4. **Stability was being reported for bands that never used the fitted value.** The `unknown` band
   falls back to the pooled lambda, but was still bootstrapped, printing a bootstrap mean of 0.00
   beside an applied lambda of 0.59. The bootstrap is now restricted to bands that cleared the
   positives gate.

Also fixed: the applied-rule ledger row logged `escalation_sens` as `nan` from a wrong metric key
(`compute_metrics` nests it at `clinical.binary_sensitivity`). Ledger rows from the pre-fix runs
were pruned before the final run, so `research/experiments.csv` holds exactly the 5 rows the
current artifacts correspond to.

#### Leak discipline

Lambda is fitted on **cross-fitted** calibrated probabilities: the deployed Dirichlet map is fitted
on all 6,981 OOF rows, so fitting lambda on those same calibrated rows would have made it in-sample
with respect to the calibrator — the same hazard S4 caught in the Mahalanobis score, in a different
place. `crossfit_calibration` refits the map five times on four folds each. The optimism turns out
to be small (pooled lambda 0.65 either way; `<40` 0.26 cross-fitted vs 0.24 in-sample) and both
values are recorded in `age_rule_lambda.json`, so the size of the correction is on the record rather
than assumed. Verified after the run: all **34** hashes in `results/frozen_artifacts.json`
unchanged, `audit_manuscript.py` still passes all **83** checks, and the test lock still raises on
`load_split_matrix("test")`.

#### A finding for a later session, not acted on here

The published OOF cost-sensitive threshold vector in `research/selective/results_oof/fit_state.json`
is `[-0.3, -0.3, 0.3, 0.2, -0.3, 0.3, -0.1]` — `akiec`, `bcc` and `mel` all sit at **-0.3, the floor
of `optimize_thresholds`' default `np.linspace(-0.3, 0.3, 13)` grid**. That is the same grid-binding
pathology as bug 1, in a published artifact: the optimizer wanted to push the escalating classes
further and could not. The lambda fit, unconstrained, lands at 0.26–0.74. This is **not** fixed
here, because `optimize_thresholds` feeds published val-fitted numbers that
`results/frozen_artifacts.json` hashes and `audit_manuscript.py` checks. Flagged for a session that
can afford to re-derive them.

### S7 — statistical hardening: the right intervals, the right attribution, the declared families (2026-09-04)

Workstream D plus G.1, G.2 and G.4. **Test was not read.** `research/run_session7_stats.py` arms
`research.testguard` unconditionally rather than via a flag, because every quantity in this session
is a validation/OOF quantity and a runner that *could* read test would make S9's single
pre-registered pass one peek less credible. Run:

```
python -m research.run_session7_stats --fit-split oof --n-boot 2000
```

Artifacts in `research/stats/results_oof/`; 10 ledger rows under `session7_stats_oof`.

**Four new modules.** `research/stats/calibration_slices.py` (per-group ECE / signed gap with
lesion-grouped intervals), `research/stats/families.py` (the multiple-comparison declaration),
`research/stats/intersectional.py` (age × sex with the power gates enforced), and the runner.
`research/stats/intervals.py` already existed and was applied rather than rewritten.

#### G.1 — the interval kind changes the claim, so the rule is now applied, not just written down

`intervals.py` encodes the switch (exact Clopper–Pearson for small-count proportions, lesion-grouped
bootstrap otherwise). S7 applies it to every subgroup proportion. Both intervals are carried in
`age_gap_intervals.csv`, not just the leading one, so a reader can check that the exact interval was
not chosen because it happened to be wider in the convenient direction.

| split | band | k/n | sensitivity | 95% CI | leading method |
|---|---|---|---|---|---|
| val | <40 | 12/22 | 0.545 | [0.322, 0.756] | Clopper–Pearson |
| val | 60+ | 175/226 | 0.774 | [0.696, 0.838] | grouped bootstrap |
| oof | <40 | 35/64 | 0.547 | [0.382, 0.716] | grouped bootstrap |
| oof | 40-59 | 243/397 | 0.612 | [0.547, 0.674] | grouped bootstrap |
| oof | 60+ | 655/893 | 0.733 | [0.697, 0.768] | grouped bootstrap |

Source: `research/stats/results_oof/age_gap_intervals.csv`.

**The under-40 point estimate replicates across splits and the magnitude still is not pinned down.**
val 0.545 (22 positives) and OOF 0.547 (64 positives) agree closely — but these are different splits
scored by different models, so this is *not* a like-for-like replication and the report says so. The
S5 warning stands: stop quoting the test figure 0.143 as if it were precise.

**One pre-specified confirmatory comparison**, `<40` vs `60+`, sole member of its family:
difference **−0.187** [−0.370, −0.016], unpaired lesion-grouped bootstrap, raw p = 0.036, Holm
p = 0.036 (a family of one takes no penalty — that is what pre-specifying buys rather than testing
all three band pairs). Validation arm, direction only: −0.229 [−0.597, 0.078], interval crosses zero
at 22 positives. **A large age gap is established; its magnitude is not.**

#### G.2 — the under-confidence is not uniform, and the global map cannot fix that

Per-band calibration on OOF, before and after the single global Dirichlet map
(`research/stats/results_oof/band_calibration.csv`):

| source | band | n | accuracy | mean conf | signed gap | ECE |
|---|---|---|---|---|---|---|
| uncalibrated | <40 | 1319 | 0.930 | 0.701 | **−0.229** | **0.229** |
| uncalibrated | 40-59 | 3112 | 0.900 | 0.710 | −0.190 | 0.191 |
| uncalibrated | 60+ | 2512 | 0.791 | 0.672 | −0.119 | 0.120 |
| uncalibrated | ALL | 6981 | 0.866 | 0.694 | −0.172 | 0.172 |
| dirichlet | <40 | 1319 | 0.935 | 0.908 | **−0.027** | 0.028 |
| dirichlet | 40-59 | 3112 | 0.904 | 0.908 | +0.005 | 0.026 |
| dirichlet | 60+ | 2512 | 0.789 | 0.830 | **+0.041** | 0.041 |
| dirichlet | ALL | 6981 | 0.869 | 0.880 | +0.012 | 0.025 |

Two findings, and the second is stronger than the plan predicted:

1. **The under-40 band is simultaneously the most "accurate" (0.930) and the worst calibrated
   (ECE 0.229).** It is dominated by easy nevi, so accuracy flatters it while the model is least
   honest about its own confidence exactly where it misses melanoma. The calibration story and the
   fairness story are the same story.
2. **After the global Dirichlet map the residual gap flips sign across bands** — `<40` stays
   under-confident (−0.027) while `60+` is pushed into *over*-confidence (+0.041). Aggregate ECE
   looks excellent (0.025) and hides two opposite-signed errors. One set of coefficients cannot
   serve three bands, which is a direct, measured argument for group-wise calibration
   (Hébert-Johnson et al. 2018, multicalibration) rather than an argument by analogy.

The band ECE spread narrows from 0.109 uncalibrated to 0.015 calibrated, so the map does most of the
work — the point is the sign flip, not the residual size.

#### D — the Macro-F1 drag is `mel`, and the rare-class complaint is half right

Per-class F1 with lesion-grouped intervals (`research/stats/results_oof/per_class_f1.csv`):

| split | worst | second | widest interval |
|---|---|---|---|
| oof | `mel` **0.607** [0.567, 0.645] | `akiec` 0.700 | `df` width 0.229 |
| val | `akiec` **0.600** [0.450, 0.733] | `mel` 0.689 | `df` width 0.387 |

The Limitations paragraph currently blames the ladder's unresolvable upper rungs on `df`/`vasc`
scarcity. That is **half right and needs rewriting, not deleting**: rare classes genuinely do carry
the widest intervals (`df` 0.229–0.387 wide against `nv` 0.013), but the *level* — where Macro-F1 is
actually lost — is melanoma, the clinically decisive class. Width and level are different
complaints and the draft conflates them.

#### D — intersectional age × sex, OOF-only, with suppressed cells named

`research/stats/results_oof/intersectional_age_sex.csv`. 6 of 10 cells clear both gates
(`MIN_GROUP_SIZE=30`, `MIN_POSITIVES=10`, imported from `research/selective/fairness.py` rather than
redefined); 4 are suppressed and listed with their reason rather than dropped. Escalation
sensitivity ranges 0.500 (`<40 x female`, 38 escalating) to 0.748 (`60+ x male`, 599). The
monotone age pattern holds within each sex, and no cell shows a sex effect that survives its own
interval — reported as exploratory, with no significance claim attached.

OOF-only is a design decision, not a convenience: val and test hold 22 and 21 escalating cases under
40, so their sex cells sit at or below the `MIN_POSITIVES=10` gate. OOF's 64 gives cells of ~26–38.

#### G.4 — the comparison family grew, and it is now declared

`results/comparison_families.json`, written before its members are populated. Seven families, three
confirmatory and four exploratory, each with its question, estimand, split, size and correction.
`families.adjust()` **refuses to correct an exploratory family**, so a descriptive result cannot
quietly acquire a significance claim later by someone calling the correction function on it.

Two substantive consequences:

- **The six ladder McNemar tests were reported unadjusted in S5** while the 42 DeLong tests were
  corrected. They are a family by exactly the same argument. Holm-corrected here from the frozen
  `results/mcnemar_delong.json` — a re-derivation, not a new test read, the same standing as
  `run_part_a.py --comparisons-only`. **The ensembling result survives comfortably**: A5 vs A2
  p = 3.36e-05 → Holm **2.02e-04**. Nothing else clears 0.05 (next best: gated fusion, Holm 0.675).
  `results/mcnemar_delong.json` itself is untouched.
- **S9's two new rungs (A7-oof, A8) are declared now, before they exist**, as their own confirmatory
  family of two, so admitting them cannot look retrospective. They are separate from the DeLong
  family because the estimand differs (Macro-F1, not per-class AUC) and folding them in would
  penalise the S5 result for a later session's additions. Both pre-registered **two-sided** — S4
  measured A7-oof *costing* 0.011 val Macro-F1, so it may well be a loss.

#### Table IV, and what S9 inherits

`paper/tables/table4_agegap.tex` is generated with **validation and OOF columns populated and the
test column left as `\textsc{pending}`**. S9 fills it from the same generator during the single
pre-registered pass. `--table-only` re-renders it from the CSV with no bootstrap, no ledger write and
no prediction load, mirroring `run_part_a.py --table-only`.

Two defects were caught by reading the rendered LaTeX rather than the numbers — the same class of
error as the post-compile revision pass in §9:

1. **The `unknown`-age row emitted 6 cells into a 7-column table.** The cell formatter returned a
   single `---` for a missing band while the populated branch returned two columns, so every column
   after it misaligned. No numeric audit can see this. Fixed: the formatter always returns two cells.
2. **A 2-case `unknown` band rendered as a sensitivity of 1.000** next to three real estimates. The
   band is now excluded from the rendered table with its counts stated in a footnote (10 val images /
   0 escalating; 38 OOF / 2 escalating) and retained in the CSV — the same "name what you suppress"
   discipline the intersectional table uses.

#### `results/CLAIM_checklist.md`

CLAIM 2024 (Tejani et al., *Radiology: Artificial Intelligence* 2024;6(4):e240300 — an RSNA
journal), **44 items**, not the 2020 version's 42. Item topics are paraphrased with the source cited
rather than reproduced. Status: **33 met, 6 partial, 3 not met, 2 N/A**. The three unmet items are
stated as gaps rather than left blank — no study registration (mitigated by the frozen, hashed
`analysis_plan.json`), no inter-rater variability (not derivable from the public label sets), and no
external validation at submission (in progress, S8). Belongs in the manuscript as a supplementary
appendix, since the point of a checklist is that a reviewer can read it.

#### Regression guard

`audit_manuscript.py` still passes all **83** checks; `results/mcnemar_delong.json`,
`results/ablation_table.csv` and the 34 hashed prediction matrices are untouched. Ten ledger rows
were written, deleted and rewritten once during the session: the first run inherited `fitsplit`'s
generic `session6_oof` label, which would have buried S7's rows among S3–S6's 67. The runner now
defaults its ledger session to `session7_stats_<fit-split>`.

#### Open items this session did not close

- **`research/stats/intervals.py` is applied but its `SMALL_COUNT = 30` switch is a judgement call**,
  stated once in the module rather than defended empirically. At 64 positives the OOF `<40` band
  lands on the bootstrap side; at 22 the val band lands on the exact side. Both are reported.
- **The PAD-UFES-20 checkpoints are gone, and `CLAUDE.md` claimed otherwise.** Its file map listed
  `*.PAD-only.pt` (6) and `*.PAD-warm.pt` (6); neither exists on disk. The runs happened
  (`ml/results/CROSS_DATASET_COMPARISON.md`, `PAD_MACRO_F1_PUSH.md`) but the weights and histories
  are not recoverable. **S8 must budget for retraining, not reloading** — corrected in `CLAUDE.md`
  with the verified inventory: 6 HAM-only baselines, 30 OOF fold models (60 files), 4 session-3
  variants (8 files), 1 gated-fusion head, and one stray untagged `resnet50_best.pt`.
- **Lesion-interior attribution (Workstream D, third bullet) was not built.** It needs Tschandl's
  `HAM10000_segmentations_lesion_tschandl` masks, which S8a downloads. The existing
  `border_mass_fraction` is an image-frame heuristic, not a lesion boundary, and is not a substitute.
- **There is no `### S6` entry in this changelog.** S6 ran (18 `session6_hier_oof` ledger rows,
  `research/run_session6_conformal.py`, `research/conformal/results_oof/`, and `stats/intervals.py`
  itself is S6's work) but never wrote its section, contrary to the standing rule. Reconstructing it
  from artifacts is a separate task and was not attempted here rather than guessed at.

### S8a — PAD + masks download re-verified, manifest re-ingested (2026-09-04)

Tier 1 (Haiku, no thinking) per the manuscript-closing plan's session table — pure file
verification, no new download needed. `data/pad_ufes_20/`, the mask zip, and
`data/_downloads/` (cached source zips) were already present on disk from earlier S3-adjacent
work; this session re-ran `python -m scripts.download_pad_ufes` to re-verify rather than trust
the prior state blind.

- **PAD-UFES-20**: all 4 files SHA256-re-verified against Mendeley's live manifest (no
  re-download — cache hit), re-extracted, `ml/data/manifest_pad.csv` rebuilt: **2,106 rows,
  1,302 with Fitzpatrick labels**, all 2,106 `path` entries resolve on disk. Class distribution
  unchanged from the earlier S3 pass: `bcc` 845, `akiec` 730, `nv` 244, `bkl` 235, `mel` 52 (192
  SCC rows correctly dropped — no honest mapping into the 7-class scheme).
- **HAM10000 segmentation masks**: MD5-re-verified, re-extracted. **Resolved the "10,016 vs
  10,015" discrepancy flagged unresolved in the earlier S3 entry** — it is not a duplicate or
  off-by-one. The extraction produces a `__MACOSX/` resource-fork junk directory (standard
  artifact of zips built on macOS) containing exactly 1 spurious file; the real mask directory
  (`HAM10000_segmentations_lesion_tschandl/HAM10000_segmentations_lesion_tschandl/`) contains
  exactly **10,015 PNGs**, one per HAM10000 image. **Flag for Workstream D**: whoever writes the
  lesion-interior attribution consumer must glob only that inner directory (or filter to
  `*.png` excluding `__MACOSX`), or a naive recursive glob will pick up the junk file.
- **Side effect noted, not requested by S8a's scope**: `download_pad_ufes.py`'s
  `prepare_pad_ufes` call also writes `ml/data/manifest_combined.csv` (10,015 HAM + 2,106 PAD =
  12,121 rows) — new on this machine, worth knowing about before any downstream script assumes
  only `manifest_pad.csv` exists.

No GPU touched; no images re-downloaded (all cache hits). This closes Workstream B's download
prerequisite — Fitzpatrick slicing and the HAM→PAD ensemble/Mahalanobis benchmark (S8b) can now
proceed without a network dependency.

### S8b — ensemble on PAD, Fitzpatrick slice, Mahalanobis under real shift (2026-09-04)

Tier 3 (Opus, high). Inference-only, external evaluation: the six frozen HAM-only CNNs, the
uniform soft-vote ensemble, the frozen OOF Dirichlet map, and the frozen S5 age-conditional
λ rule, all applied unchanged to the **full 2,106-image PAD-UFES-20 cohort** (not the 314-row
subset the repo had evaluated before — too few Fitzpatrick-labelled/melanoma cases to power
anything). New code: `research/xdomain/extract_pad.py` (the one GPU pass — 6 CNN prediction
matrices + ConvNeXt-Tiny penultimate features, ~15 min) and `research/xdomain/run_session8b.py`
(CPU analysis). Nothing fitted on PAD; HAM test not read (Hard Rule 2) — the Mahalanobis
in-distribution reference is HAM **val**. Outputs: `research/xdomain/results/`
(`ensemble_on_pad.csv`, `fitzpatrick_slice.csv`, `fitzpatrick_gaps.json`,
`mahalanobis_shift.json`, `session8b_report.md`), 4 rows in `research/experiments.csv`
(`session8b`).

**1. The ensemble has now been evaluated on PAD — and does not help.** Individual HAM-only
members: Macro-F1 0.124 (`efficientnet_b3`) to 0.188 (`convnext_small`) — matches the prior
per-model range (`ml/results/CROSS_DATASET_COMPARISON.md`, 0.113–0.172) closely enough to
corroborate it. The uniform 6-CNN soft-vote reaches **0.167**, *below* the best single member
(`convnext_small` 0.188) — the diversity that helps in-distribution does not help when every
member is wrong for correlated reasons (dermoscopy features absent in smartphone photos). The
frozen HAM-OOF Dirichlet map makes it **worse still** (0.167→0.133): a calibration map fitted
to HAM's prior actively distorts probabilities calibrated for a completely different
acquisition modality and class prior.

**2. The age-conditional λ rule redistributes cost toward catching escalating cases here too —
but this is a much weaker result than it first looks, and must not be reported as "the S5
mechanism replicated externally."** Applied on top of the (harmful) Dirichlet map:
escalation sensitivity 0.223→**0.585**, melanoma recall 0.115→**0.327**, missed-serious
1264→**675**, and — unlike the demographic-parity-costing pattern typical of such rules —
Macro-F1 and balanced accuracy *also* rise (0.133→0.177, 0.296→0.347). Two things cut against
reading this as a replication of the HAM under-40 finding:
  - **PAD's escalating prevalence is ~77%** (bcc 40.1% + akiec 34.7% + mel 2.5%), the *inverse*
    of HAM's ~19%. A constant bonus added to the escalating-class logits mechanically helps on
    any cohort this skewed toward the classes it favours — this is much closer to a
    prior-correction effect than to evidence that "argmax discards signal an escalation-mass
    score can recover" (the actual S5 claim, which needs the ranking/AUC decomposition, not
    tested here).
  - **Even with the rule, Macro-F1 (0.177) stays below the best unadulterated single member**
    (0.188) — the absolute numbers describe a model still fundamentally miscalibrated for this
    domain, not a rescued deployment.
  - This is **not** Workstream E1 (`DATASET_REFINING.md`) — E1 is same-modality multi-center
    dermoscopy replication (BCN-20000/MSKCC), a mechanism test with a matched acquisition
    protocol. This is dermoscopy→smartphone-clinical, a much larger shift, and the correct
    frame is "the escalation-bias intervention is not harmful and plausibly helps under
    extreme positive-prevalence shift," not "the hidden-stratification mechanism transfers."
  - **Correct framing for the manuscript**: report as a directional, non-preregistered
    external probe — encouraging that the intervention causes no harm and likely helps under
    a very different prior, explicitly weaker evidence than a same-modality replication.

**3. Fitzpatrick slice run — and the reported gap needs disaggregation before use.** PAD carries
Fitzpatrick for 1,302/2,106 rows: I 137, II 750, III 347, IV 59, V 8, VI 1 (unlabelled: 804).
V/VI fall below `MIN_GROUP_SIZE=30`/`MIN_POSITIVES=10` and are correctly suppressed, not
silently dropped — cohort simply cannot speak to the darkest tones. `fairness.gaps()` computed
`equalized_odds_tpr_gap=0.297` and `demographic_parity_gap=0.291`, but **both are driven almost
entirely by the "unknown" (no Fitzpatrick label) group**, whose escalation sensitivity is 0.051
against 0.256–0.349 for every labelled tone I–IV — a missing-data/selection artifact (unlabelled
rows likely differ systematically, e.g. by lesion type or acquisition site), not a skin-tone
finding. **Restricted to labelled tones I–IV only**, the sensitivity spread is a much smaller
0.093 (I 0.349, II 0.290, III 0.256, IV 0.278) and is **non-monotonic** — it does not reproduce
the literature's typical darker-skin-underperforms pattern, and n=59 at IV is thin. `macro_f1_gap`
(0.064) is *not* confounded this way — "unknown"'s macro-F1 (0.103) sits inside the I–IV range.
**Action for whoever writes this up**: report the I–IV-only spread as the skin-tone finding,
report the unknown-vs-labelled gap as a separate (data-completeness) finding, and do not quote
the pooled 0.297 gap as a skin-tone disparity number — it isn't one. `fairness.gaps()` itself is
not wrong (it has no way to know "unknown" isn't a comparable group); the caller must exclude it
before invoking `gaps()` if a downstream script needs the true skin-tone-only number. Not fixed
here — this session's `slice_attribute` call reports the raw per-group table, which is honest;
the pooled `gaps()` summary is what a reader could misuse.

**4. Mahalanobis distance is redeemed under a genuine shift.** Session 4 ranked it last
(test AURC 0.0397 vs margin's 0.0311) because same-distribution HAM test has no shift to
detect. Fitted on HAM train features (no refit), scored on HAM val (in-distribution, n=1,532)
vs PAD (shift, n=2,106): **AUROC 0.913** separating the two, median score 485 (HAM val) vs
8,717 (PAD) — **18× separation**. This is the fair test the manuscript's Limitations section
already asks for (G.3, Ovadia et al. 2019 / Lakshminarayanan et al. 2017 shift protocol).

No hand-entered numbers — every figure above resolves to a file under `research/xdomain/results/`
or a row in `research/experiments.csv` (`session8b`).

### Session 6 plan — S5 through S9 (written 2026-09-04, resolving a dangling pointer)

Section 10's heading has said "see the active plan for full detail" since S1, and that plan is not
in this repository — it lived in the session that produced it. S5 had to be reconstructed from
side-references in the S4 entry (`optimize_thresholds_by_group` described as "the home for S5's age
rule"; the three findings "S5/S6/S9 must not be surprised by") and a footnote in
`DATASET_REFINING.md` §Pre-S12. That reconstruction could have been wrong, and the next session
should not have to repeat it. What is actually pinned down by artifacts on disk:

| # | Scope | Status |
|---|---|---|
| S1 | A.0 bug fixes, A.1 fold assignment | done |
| S2 | training driver fixes + handoff | done |
| S3 | A.3/A.4 OOF extraction, shift diagnostic, PAD restored | done |
| S4 | A.6 `--fit-split` rewiring, test lock, val-fit regression guard | done |
| **S5** | **A.7 age-conditional lambda rule, fitted OOF** | **done (this entry)** |
| S6–S8 | **not reconstructable from this repository** | see below |
| S9 | the single pre-registered test pass | blocked on `results/analysis_plan.json` |

**S6–S8 are unknown and must be restated from the plan document before they are started.** Nothing
on disk names them. Do not infer them from this file — an invented workstream that reads plausibly
is worse than an admitted gap.

**What S9 inherits, and must pre-register before it reads test:**

1. `results/analysis_plan.json` does not exist yet, and `DATASET_REFINING.md`'s Pre-S12 integrity
   gate asserts on it. S9 creates it.
2. The frozen parameters S9 applies without refitting:
   `research/calibration/results_oof/fit_state.json` (Dirichlet),
   `research/selective/results_oof/fit_state.json` (`msp`, abstention quantiles),
   `research/conformal/results_oof/fit_state.json` (Mondrian quantiles, `exact_guarantee: false`),
   and **`research/agerule/results_oof/age_rule_lambda.json`** (the band lambdas). That last path is
   the one `DATASET_REFINING.md`'s Pre-S12 `required` list was waiting on and should now name —
   its footnote says the filename was undefined because "S4 and S5 produce them". It now exists.
3. Three quantities that must be pre-registered **two-sided**, all measured rather than assumed:
   rung A7-oof (OOF calibration costs 0.011 val Macro-F1 and buys calibration sample size, not
   accuracy — S4); the `<40` lambda (bootstrap CI [0.00, 0.61], 10.2% of resamples select zero —
   S5); and OOF conformal coverage, which has **no** exact finite-sample guarantee and must be
   audited empirically, never asserted (S4).
4. The `margin` → `msp` flip means the two arms' abstention thresholds sit on different score
   scales and are not comparable as numbers (S4). Any S9 table putting them side by side must say so.

### Post-S11 external-validity plan reviewed and rewritten (`DATASET_REFINING.md` v2, 2026-09-04)

Reviewed the post-S11 plan for external validation and rewrote it. v1's strategy was sound
(inference-only external battery, frozen backbones, prior/optical disentanglement) but had four
items that could not execute, two statistical errors, and a viva Q&A that pre-wrote unmeasured
results. 16 changes, fully itemised in the document's own §12. The load-bearing ones:

- **Pre-written results stripped from the Q&A.** v1 asserted the under-40 collapse "replicated
  identically", escalation-mass AUC "≥0.90", and FRR "≤2%" — none measured. This violated its own
  Hard Rule 4 and pre-decided an experiment whose entire value is that it can fail. Rewritten to
  state design + `[RESULT — Sxx]` placeholders.
- **E1 split into mechanism vs. operating point.** λ is known not to transfer stably even across
  half-splits *within* HAM (0.16–0.31), so expecting a HAM-fitted λ to transfer to Barcelona was
  optimistic. Now reports escalation-mass AUC (tuning-free, tests whether the *mechanism* transfers)
  separately from frozen-λ transfer, plus the full λ sweep labelled descriptive. Robust to the most
  likely outcome — "mechanism transfers, operating point needs local recalibration" — which is also
  the more useful clinical statement.
- **Gate 0 power check added** as a hard gate before any GPU time: nobody had counted how many
  under-40 escalating lesions BCN-20000 actually contains, and the whole E1 claim rests on it.
  Pre-committed go/no-go thresholds (≥100 primary / 40–99 supporting / <40 descriptive+pool MSK).
- **E2's prior correction was wrong twice.** It subtracted `log P_HAM`, but `class_weighting:
  effective_number` means the model's implicit prior ≠ empirical HAM prior (over-correction — and the
  reason Session 6's prior lever came back dead in-distribution). And plugging in the true PAD prior
  is an oracle, not a method. Now: implicit source prior estimated from the model's own held-out
  predictions, target prior estimated label-free by **EM (Saerens et al. 2002)**, with the oracle
  demoted to a ceiling reference. Converts an oracle analysis into a deployable method at no extra compute.
- **Hard Rule 6 added (lesion-grouped external intervals).** v1 used Clopper–Pearson over external
  images; external cohorts have multiple images per lesion, so independence is violated and the
  intervals come out too narrow — the exact error the internal analysis was built to avoid.
- **`FRR` renamed to `point-FRR` / `set-FRR`.** v1's point-prediction definition collided with the
  paper's existing conformal-set definition, which `audit_manuscript.py` already checks.
- **TENT cut**: ConvNeXt-Tiny/Small use **LayerNorm and have no BatchNorm**, so it applies to only 4
  of 6 archs; BN affine params are backbone weights, contradicting the document's own Hard Rule 1.
- **PH2 cut** (orphaned — named in the hierarchy and in E4's hypothesis but with no acquisition or
  inference step, making that hypothesis uncomputable) → 3-level shift hierarchy.
- **E3 (3-tier triage) promoted to primary external metric** — the only metric definable consistently
  across cohorts, and it makes SCC scoreable (a 7-class model cannot emit SCC; v1 left this undefined).
- Added: conformal **exchangeability caveat** (external cohorts aren't exchangeable with the HAM
  calibration set → no finite-sample guarantee, empirical audit only), pre-registration Hard Rule 7 +
  separate multiple-comparison family, §10 contingency branches deciding in advance how a null result
  gets reported, TRIPOD+AI as the external-validation reporting standard, DDI flagged as an
  access-gated (Stanford DUA) dependency, and ISIC source-filtering flagged as verify-first with an
  API fallback (ISIC 2019 has no source column; ID preservation across releases is an assumption).
- Fixed a **no-op assertion** in v1's pre-flight gate: `A.exists() or B.exists()` where B was a
  directory that always exists, so it could never fail.

Nothing in this plan has been run; it executes after S11.

**v3 (same day): Workstream E0, the Comparison Arm, added.** Prompted by the question of whether to
retrain everything on ISIC 2019 given 4× GPU capacity. Costed it honestly: ≈26 GPU-hours → ~6.5h at
4×, so compute was *not* the blocker. Rejected it anyway for three reasons now recorded in the plan's
§11: (1) **HAM10000 is a subset of ISIC 2019**, so training on it dissolves the external validation
set — BCN/MSK stop being external and `split_v1.csv` becomes meaningless, forcing a new site-stratified
split design, which is exactly where leakage enters; (2) it risks **destroying the contribution**,
since the under-40 blind spot exists partly *because* HAM's <40 band is 4.9% escalating vs 35.5% at
60+, and with 4× more melanoma the blind spot may simply not exist, leaving the λ rule as a solution
to a problem the paper no longer has; (3) one day of compute implies 1–2 weeks regenerating 34
matrices, the ladder, all CIs, McNemar, DeLong, conformal, selective, fairness, 7 figures, 3 tables
and 83 audit checks.

Replaced with a far cheaper experiment that answers the same reviewer question **causally**: train one
ConvNeXt-Tiny from scratch on BCN-20000 + MSKCC only (no HAM images at all), 3 seeds, ~1.1 GPU-hours,
and ask whether the under-40 argmax failure *re-arises independently* in a model that has never seen an
Austrian image. E1 asks "does our model's failure transfer?"; E0 asks "does the failure re-emerge from
scratch?" — the stronger test, at ~4% of the compute, touching zero frozen artifacts.

Design points that make it valid rather than merely cheap: **Cell A (the HAM control) is recomputed as
ConvNeXt-Tiny alone, uncalibrated, no TTA**, re-sliced from existing frozen predictions at zero compute
— because comparing the published 0.143 (an ensemble+TTA+Dirichlet number) against a single BCN model
would confound the data question with an ensemble question. Recipe parity, class-space parity (SCC
dropped so escalation-mass means the same thing on both sides), and lesion-grouped split discipline are
all mandatory; 3 seeds guard against mistaking one unlucky initialization for a site effect. Gate 0 was
extended to power-check the *projected BCN/MSK test split* specifically, since that count must exceed
HAM's 21 under-40 escalating cases for the comparison to be worth running. Hard Rule 1 was scope-noted
to distinguish "published checkpoints are immutable" from "no new model may ever be trained".

Both outcomes are pre-committed in §10: if the blind spot re-arises, the claim upgrades to a structural
property of argmax under age-stratified prior skew; if it does not, the finding is data-driven and the
λ rule reframes as the mitigation for clinics that cannot fix their sampling — which is the realistic
deployment situation anyway.

  - The 3.59 GB of PAD images were **not** downloaded in this pass (out of caution about doing a
    multi-GB, multi-minute unattended download alongside the GPU queue without the user present to
    notice a problem) -- run `python -m scripts.download_pad_ufes --skip-masks` (masks are already
    done) whenever convenient; it will re-verify and skip the two already-cached small files
    automatically.


Four candidate Macro-F1 levers were tested with lesion-grouped cross-fitting, test untouched, and
all four came back negative or dead:

| Lever | Held-out result | Verdict |
|---|---|---|
| 8-member soft-vote (add SwinV2-Tiny + MaxViT-Tiny) | 0.7981 vs 0.7986 (6-CNN) | wash |
| Caruana greedy selection over 8 | 0.7640 / 0.7839 vs 0.7945 uniform | worse, overfits |
| Global prior correction `p/π^τ` | −0.0035 / −0.0061 | dead, τ_opt≈0 |
| Per-class offsets | mean +0.0009, sign flips across folds | noise |

0.8047 test Macro-F1 is the ceiling of the six frozen checkpoints under any post-hoc reweighting —
reported as a negative result (Workstream E), not hidden. The remaining credible Macro-F1 path is
fold-bagging (K-fold OOF training, rung A8), now underway.

Two findings replace the accuracy chase:

- **The Macro-F1 bottleneck is `mel` and `akiec`**, not the ultra-rare `df` / `vasc` as an earlier
  Limitations draft claimed — melanoma is the clinically decisive class and the actual drag.
  **S7 re-measured this against a file** (`research/stats/results_oof/per_class_f1.csv`) and the
  conclusion holds while the numbers moved: on OOF `mel` **0.607** and `akiec` **0.700**; on val
  `akiec` **0.600** and `mel` **0.689**, against `df` 0.711 and `vasc` 0.850. The figures quoted
  when this bullet was first written (val `mel` 0.693 / `akiec` 0.713 / `df` 0.727 / `vasc` 0.909)
  came from a val-fitted Dirichlet map; S7 scores val with the **OOF-fitted** map, so the two are
  not the same quantity and only the S7 numbers resolve to a `results/` artifact (Hard Rule 4).
  The rare-class complaint is **half right**: `df`/`vasc` do carry the widest intervals (`df` 0.229
  wide on OOF, 0.387 on val, against `nv` 0.013) — width and level are different complaints.
- **The under-40 escalation-sensitivity gap is a decision-rule failure, not an information
  failure.** Escalation-mass AUC (`p(akiec)+p(bcc)+p(mel)` vs. ground truth) is 0.927 for patients
  under 40 — nearly as good as older bands (0.933–0.934). The probability vector *does* carry the
  signal; argmax discards it because `nv` dominates the prior in that band. A one-parameter
  age-conditional escalation-mass rule, cross-fitted over 20 lesion-grouped half-splits, moves
  under-40 sensitivity 0.565→0.733 (mean gain +0.168, improving in 16/20 folds) at matched 60+
  specificity (0.877). This corrects an over-strong Discussion claim that no probability-vector-
  derived rule could fix this failure.

### S9 — the frozen analysis plan and the single pre-registered test pass (2026-09-04)

**Why this session exists.** S4–S8 added a lot of new test-set quantities: two ladder rungs
(A7-oof, A8), Number Needed to Biopsy, False Reassurance Rate with intervals, the age-conditional
rule at a fitted λ, hierarchical conformal coverage, intersectional cells, per-band calibration,
lesion-interior attribution. Computed one at a time, each is a selection opportunity — compute it,
dislike it, change something upstream, compute it again — and nothing in the repository would
record that this had happened. Under Hard Rule 2 they therefore cannot be read incrementally. S9
writes down what will be asked of the test split *before* asking it, then asks all of it at once.

**What was built.**

- **`results/analysis_plan.json`** (`research/session9/plan.py`, runner
  `research.run_session9_plan`) — **19 pre-registered quantities across 7 groups**, frozen
  2026-09-04, sha256 `5c9bebcf255a87b7…`. Each carries its definition, its formula, the split its
  parameters were fitted on, its interval rule, its direction (two-sided vs descriptive) and its
  multiple-comparison family. It also fixes the constants a later choice could otherwise quietly
  move: escalating class set, age-band edges, α values, ECE bins, power gates, and the NNB
  reference prevalence. **The plan hashes its six fitted-input files** (the OOF Dirichlet map, the
  MSP abstention quantiles, the per-band λ, the flat and hierarchical conformal states, the S7
  statistics state) and the six frozen test matrices, so refitting any of them and re-running would
  visibly be a *different plan* rather than the same plan run twice.
- **The plan is recorded in `results/frozen_artifacts.json`** under a new `analysis_plan` key, not
  as a 35th file entry — so `build_paper_artifacts.py` still reproduces the declared 34 prediction
  hashes exactly, which is the regression guard the plan asked for. Verified: `num_files` 34,
  `files` 34, `analysis_plan` present.
- **`research/run_session9_testpass.py`** — the single pass. It refuses to emit any quantity the
  plan does not name (every id goes through `plan.quantity()`, which raises on an unregistered
  one), re-hashes every fitted and test input against the freeze before starting, and aborts if any
  has drifted. Two stages under one plan and one receipt: `--stage tables` (CPU, 18 quantities,
  frozen matrices only) and `--stage attribution` (GPU, Grad-CAM × Tschandl lesion masks).
- **`results/test_pass_receipt.json`** (`research/session9/receipt.py`) — append-only record of
  every execution, carrying the plan hash, the input hashes and the quantity ids. A repeat
  execution is **refused** unless `--rerun-reason` is supplied, and the reason is stored
  permanently. `testguard` stops an *accidental* test read; the receipt is what stops a deliberate
  second one from being invisible.
- **`research/session9/nnb.py`** — NNB formalised, with the prevalence correction that makes it
  reportable. NNB = (TP + w·FP)/TP with benign importance weight w = n_pos(1−π)/(n_neg·π); primary
  π = 0.03 with a pre-registered 0.01–0.05 sensitivity range. A self-check asserts w = 1 and the
  correction is the identity when π equals the observed prevalence.
- **`research/oof/extract_foldbag_test.py`** — scores the test split with all 30 fold checkpoints
  so rung A8 can exist. Refuses to run before the plan is frozen (A8's definition must be fixed
  before its inputs are produced) and refuses `--limit` into the default output directory, because
  a truncated matrix still looks "present" to the availability check.
- **Table IV's test column.** `run_session7_stats._render_table4` now takes an optional
  `test_rows`, so all three columns are rendered by one generator and cannot drift in interval
  choice or rounding. Its caption gains a sentence saying the test column is *not* a like-for-like
  replication of the validation column — different models, different patients.

**Three decisions worth defending at viva.**

1. **A8 is uniform over all 30 members, with no member selection.** The plan originally said
   "member set selected on val". Session 1 already measured Caruana greedy selection over 8 members
   losing to the uniform average on held-out data; selecting among 30 would give the winner's curse
   a far larger surface. Uniform bagging takes no selection decision at all — strictly stronger
   discipline, and it removes the need for a second 30-model extraction over validation.
2. **A8's calibrator carries a stated mismatch.** No out-of-fold matrix can exist for a 30-member
   bag — for any training row, 24 of the 30 members saw it — so A8 reuses the Dirichlet map fitted
   on the 6-member OOF ensemble. Averaging 30 members flattens the maximum probability further than
   averaging 6, so the map is applied to a slightly flatter distribution than it was fitted on and
   will *under*-sharpen A8. The direction is knowable, so it is stated: **A8 is a lower bound** on a
   properly calibrated fold-bag.
3. **A declared family does not shrink.** The rehearsal (below) caught this: `families.adjust`
   refused to Holm-correct the two-member `s9_new_rungs` family with one p-value. Correcting over
   only the tested member would have handed it a *lighter* penalty than it was pre-registered to
   carry — the same failure as a family that grows silently, which is exactly what G.4 warns about.
   If a declared member cannot be evaluated it now enters the correction at p = 1.0, keeping Holm's
   denominator at the declared size, and is named in `members_not_evaluated` so it is not mistaken
   for a null result.

**The rehearsal, and why it mattered.** `--smoke` runs the identical code path on the validation
split, writing to `results/session9_smoke/` with no receipt, no ledger row and no Table IV. The
point is that a crash halfway through the real pass is recoverable — the receipt is only written on
success — but the numbers already printed would have been *seen*, and "I only glanced at it because
it crashed" is not a defence anyone should have to make. The rehearsal found three real defects
before the read was spent: the family-shrinkage bug above, an `AttributeError` from the two
divergent `Proportion` dataclasses (`research.stats.intervals` exposes `.interval`, the older copy
in `research.conformal.hierarchical` does not), and a docstring escape that broke the extraction
driver's import.

It also produced two strong correctness signals:

- The rehearsal reproduces S5's frozen validation figures **exactly** — λ-rule per-band sensitivity
  `<40` 0.545→0.591, `40-59` 0.550→0.817, `60+` 0.774→0.881, ALL 0.714→0.847, macro-F1
  0.7638→0.7713 (`research/agerule/results_oof/age_rule_lambda.json`, `val_holdout`). The pass is
  applying the frozen parameters, not re-deriving different ones.
- **The conformal refit reproduced all 20 frozen OOF states bit-for-bit.** Only α = 0.10 and 0.05
  are frozen, but the FRR sweep needs 0.20/0.15/0.02/0.01 as well, so those states are refitted from
  the OOF matrices with the S6 recipe, seed and lesion-grouped half-split. `assert_matches_frozen`
  checks the refit against the two frozen α before any of the six is used; if it failed, the sweep
  would not be trustworthy and the pass would stop.

**Status.** Complete. The fold-bag extraction ran (30 models × 1,502 test images, 24-view TTA,
~1.6 h GPU at ~7 images/s) so that A8 and A7-oof landed in the same single read. Both stages of the
pass have executed; the test split has now been read, once, under one plan.

#### S9 results — the test read, spent once (2026-09-04)

`results/test_pass_receipt.json` records **two executions of one plan** (`tables` 22:23 UTC, 18
quantities; `attribution` 22:26 UTC, 1 quantity), both carrying plan sha256 `5c9bebcf255a87b7…`,
neither with a `rerun_reason`. All 19 pre-registered quantities emitted. Outputs in
`results/session9/`.

**The pipeline check passed exactly.** Rung A7-val recomputed from the frozen matrices reproduces
the published `results/ablation_table.csv` macro-F1 of **0.8047238352979755 to zero drift**
(`abs_drift` 0.00e+00 against a 5e-4 tolerance). The conformal refit reproduced all **20** frozen
OOF states bit-for-bit before the sweep α values were used.

**1. Both new rungs lose, and the last Macro-F1 lever is dead** (`results/session9/ladder.csv`,
`new_rung_comparisons.json`).

| Rung | Macro-F1 [95% CI] | Esc. sens. | Missed | Δ vs A7-val [95% CI] | Holm p |
|---|---|---|---|---|---|
| A7-val | 0.8047 [0.7548, 0.8420] | 0.7310 | 78 | — | — |
| A7-oof | 0.7871 [0.7315, 0.8295] | 0.7310 | 78 | −0.0176 [−0.0451, +0.0054] | 0.242 |
| A8 (30-member bag) | 0.7810 [0.7207, 0.8258] | 0.7345 | 77 | −0.0237 [−0.0625, +0.0043] | 0.242 |

Neither clears Holm inside the two-member `s9_new_rungs` family, and **both point estimates are
negative**. A7-oof's −0.018 is the price of OOF-fitted calibration, close to the −0.011 S4 measured
on validation and pre-registered two-sided for exactly this reason. A8 is the important one:
fold-bagging was *the one remaining credible Macro-F1 path* after session 6 killed four post-hoc
combination levers, and on test it does not deliver — it is nominally the worst of the three. It
does catch one more escalating case (77 missed vs 78), which is the direction the bag was expected
to help, but nothing here is significant. **Workstream E's exhaustion result is now complete: five
levers tried, five dead.** With the caveat that A8 carries the pre-registered under-sharpening
mismatch (it reuses the 6-member OOF Dirichlet map), so it remains a lower bound, not a refutation
of fold-bagging in principle.

**2. The mechanism claim weakens on test, and this is the most consequential finding of the
session** (`results/session9/age_gap_test.csv`).

| Band | n | Escalating | Sensitivity [95% CI] | Escalation-mass AUC |
|---|---|---|---|---|
| `<40` | 290 | 21 | 0.143 [0.030, 0.363] (CP) | **0.810** |
| `40-59` | 671 | 70 | 0.814 [0.694, 0.918] | 0.975 |
| `60+` | 532 | 199 | 0.764 [0.683, 0.836] | 0.933 |
| All | 1502 | 290 | 0.731 [0.665, 0.793] | 0.950 |

The under-40 escalation-mass AUC is **0.810 on test**, against 0.927 on validation and 0.889 OOF,
and against 0.975/0.933 for the two older bands on the *same* split. The claim the Discussion is
built on — that the under-40 failure is a *decision-rule* failure because the probability vector
still ranks escalating lesions well in that band — **does not replicate at full strength**. On test
the band has genuinely weaker ranking as well. The honest reading is that the failure is *part*
information and *part* decision rule, and the balance between the two is not resolved by this data:
21 positives is thin, the interval on that AUC is correspondingly wide, and the val/OOF/test
estimates disagree. S10 must soften this claim rather than restate it.

**3. The confirmatory age gap is far larger on test than out-of-fold, and the middle band is why.**
`<40` vs `60+`: **−0.621 [−0.786, −0.386]**, two-sided bootstrap p ≈ 0, Holm p < 0.001 as the sole
member of `age_gap_confirmatory` — against −0.187 [−0.370, −0.016] OOF. The gap is established
beyond doubt; its *magnitude* swings enormously across splits, which is the point S7 already made
with Clopper–Pearson and which this read reinforces rather than settles. Note the `40-59` band moved
0.550 (val) → 0.612 (OOF) → **0.814** (test) — a swing no aggregate metric would have exposed, and
a reminder that these per-band estimates rest on 60–70 positives.

**4. The two safety nets are NOT orthogonal in the band that matters**
(`results/session9/orthogonality_test.csv`). This was a pre-registered, directional claim and it
came back mixed:

| Band | Misses (argmax) | Referred by abstention | Caught by λ | Both | Jaccard |
|---|---|---|---|---|---|
| `<40` | 18 | 2 | 2 | 2 | **1.00** |
| `40-59` | 13 | 2 | 11 | 2 | 0.18 |
| `60+` | 47 | 17 | 16 | 15 | 0.83 |
| All | 78 | 21 | 29 | 19 | 0.61 |

In `40-59` the rule and abstention are genuinely complementary — λ rescues 11 of 13 misses that
abstention leaves behind. In `<40` they are **the same two cases**: of 18 missed escalating lesions,
abstention refers 2, λ catches 2, and the overlap is complete. So the argument that the age rule
adds a safety net abstention cannot provide holds in general but **fails precisely in the subgroup
the paper is about**. This must be reported as measured; it is the strongest internal evidence that
the under-40 problem is not fixable from the probability vector alone, and it is consistent with
finding 2.

**5. The λ rule buys sensitivity and pays in Macro-F1 and referrals**
(`results/session9/agerule_test.csv`, `nnb_test.csv`). Overall escalation sensitivity
**0.731 → 0.831**, missed serious **78 → 49**, at Macro-F1 **0.7871 → 0.7492** (−0.038) and referral
rate 0.178 → 0.268. Per band: `<40` 0.143 → 0.238, `40-59` 0.814 → 0.971, `60+` 0.764 → 0.844.
Under-40 remains poor in absolute terms. Priced as NNB at the pre-registered π = 0.03, the whole
cohort moves **3.0 → 6.2**, and `<40` **5.2 → 8.1**; observed-prevalence NNB is 1.26 → 1.67, which is
the number that must never be set beside the 8–15 dermatologist range without the correction.

**6. Conformal: the bipartite calibrator is the clear winner and the marginal one is as misleading
as predicted** (`results/session9/conformal_test.csv`, `frr_bounds_test.csv`). At α = 0.10, LAC
marginal achieves 0.901 overall coverage but only **0.735 on escalating lesions**, FRR **0.228**
[0.169, 0.288]; the bipartite variant at the same α gives 0.907 serious coverage and FRR 0.059. Best
overall is **RAPS bipartite at α = 0.05: FRR 0.0138 [0.0038, 0.0349], serious coverage 0.955, mean
set size 1.87**. The α needed to *bound* FRR (upper confidence limit, not point estimate) below 0.05
is α = 0.05 for RAPS bipartite and α = 0.02 for LAC bipartite; **no α in the sweep bounds FRR below
0.01**, reported as unreachable rather than clipped.

**7. Per-class F1 confirms the S7 attribution correction** (`results/session9/per_class_f1_test.csv`).
The drag is `akiec` **0.697** [0.563, 0.805] and `mel` **0.699** [0.614, 0.772], not `df` 0.791 or
`vasc` 0.842. The rare classes do, however, own the widest intervals (`df` 0.389, `vasc` 0.308 wide),
so Limitations should attribute *level* to akiec/mel and *variance* to df/vasc — they are different
claims and the current draft conflates them.

**8. Per-band calibration: the sign-flip replicates, the "under-40 is worst calibrated" finding does
not** (`results/session9/band_calibration_test.csv`). Uncalibrated signed gaps on test: `40-59`
**−0.197**, `<40` −0.163, `60+` −0.105 — so on test the *middle* band is the worst calibrated, where
S7 found `<40` worst on OOF. What does replicate, and is the load-bearing half of the G.2 argument,
is that the single global Dirichlet map leaves **residuals of opposite sign across bands**: `40-59`
−0.023, `<40` +0.015, `60+` **+0.034**, under an aggregate ECE of 0.017. One global map cannot
correct three bands that are miscalibrated in different directions.

**9. Intersectional cells, with the gates respected** (`results/session9/intersectional_test.csv`).
5 of 8 cells usable; **`<40 × male` is suppressed** (9 escalating < the `MIN_POSITIVES=10` gate) and
`<40 × female` posts 0.250 on 12 positives. Equalized-odds TPR gap across usable cells 0.598. The
suppressed cell is named rather than dropped — and it is the cell an unprincipled analysis would
have reported at 0.000.

**10. Attribution: the maps are on the lesion, and the caveat is visible in the numbers**
(`results/session9/attribution_test.csv`). Mean lesion-interior Grad-CAM fraction **0.523**
[0.509, 0.536] over all 1,502 test images, 0 missing masks, against a mean lesion area of just
**0.256** of the frame — a concentration ratio of **3.45**, so the maps are far more focused on the
lesion than area alone would produce. Errors score *higher* than correct predictions (0.570 vs
0.513), which is exactly the artefact the module's docstring warns about: the map is taken with
respect to the *predicted* class, so it lands on the lesion almost by construction. Reported as a
supporting check; the mechanism argument stays with the escalation-mass AUC.

**What S10 must change.** Three claims in the current draft do not survive this read and must be
revised, not restated: (a) the under-40 failure as a *pure* decision-rule failure (finding 2);
(b) the orthogonality of the age rule and abstention as a general property (finding 4 — it fails in
`<40`); (c) "under-40 is the worst-calibrated band" as a cross-split fact (finding 8 — the sign-flip
replicates, the ranking does not). The exhaustion result (finding 1) gains a rung and becomes
stronger.


### S10 — manuscript revision: the decision-rule story replaces the accuracy story (2026-09-05)

Workstream F. `paper/manuscript.tex` rewritten against the S7/S8b/S9 results. Source grew
908 -> 1,831 lines (55.5 KB -> 121 KB). **No test split was read**; every new number is
re-derived from a frozen `results/` artifact.

**Verification run at the end of the session, all four green:**

| Check | Result |
|---|---|
| `research/ablation/audit_manuscript.py` (pre-existing assertions) | **83/83 pass** — nothing previously asserted was broken |
| Structural validation (cites/labels/refs/graphics/braces/envs/tabular arity) | **PASS** — 47 cites = 47 bibitems, 0 unused, 0 dangling refs, 0 duplicate labels |
| New-number verification (210 literals resolved from `results/`) | **PASS** |
| Row-level reconstruction of the 4 hand-written tables (25 rows) | **PASS** |

All three are staged in the repo as `research/ablation/verify_s10_numbers.py`,
`verify_s10_tables.py` and `validate_structure.py`, runnable from the repo root. **S11 should
fold the first two into `audit_manuscript.py` and delete them** — they are exactly the
difference between its 83 checks and the ~100+ the plan calls for. `validate_structure.py` is
worth keeping permanently: with no LaTeX toolchain on this machine it is the only thing that
catches a dangling `\ref` or an unbalanced environment before Overleaf does.

**Two real transcription errors were caught by the row-level check and fixed** — both FRR
lower bounds rounded the wrong way from the 4-dp S9 markdown report: LAC/bipartite/α=0.10
`0.034` → **0.035** (true 0.0345159) and RAPS/marginal/α=0.10 `0.156` → **0.155** (true
0.1554646). This is the argument for reconstructing table rows from the CSV rather than
eyeballing them against a rendered report.

**The three claims S9 said must change, changed:**

1. **The pure decision-rule reading is gone.** §Results-fairness now reports the escalation-mass
   AUC on all three splits (val 0.927 / OOF 0.889 / **test 0.810**, against 0.975 and 0.933 for
   the older bands on test) and concludes the failure is *part* decision rule and *part* lost
   ranking, with the balance unresolved at 21 positives. The Discussion claim *"No abstention
   rule, no conformal set, and no calibration map ... can detect an error the probability vector
   does not signal"* is explicitly retracted and replaced: the vector **does** signal it; what
   fails is the *summary* each mechanism takes of the vector (margin/MSP are large whenever
   `p(nv)` is large), which is a correctable defect — and the λ rule, which reads the discarded
   direction, recovers part of the gap where abstention recovers almost none.
2. **Orthogonality is reported as failing where it matters.** New `tab:ortho`: Jaccard **1.00**
   in `<40` (both mechanisms reach the same 2 of 18 misses), 0.18 at 40-59, 0.83 at 60+, 0.61
   overall. Written up as a negative result inside the paper's own headline contribution, and
   repeated in Limitations — 16 confidently-wrong `<40` misses are reached by neither mechanism
   and we have no proposal that reaches them.
3. **"Under-40 is the worst-calibrated band" is not asserted.** New `tab:bandcal` reports the
   test slice; the text states the *sign disagreement* replicates on all three splits (60+ ends
   over-confident at +0.030 / +0.041 / +0.034 after Dirichlet) while the *ranking* does not (on
   test the largest uncalibrated gap is 40-59 at −0.197, not `<40` at −0.163).

**New material, by section:**

- **Title** now reads "... Subgroup-Conditional Calibration, Abstention and Conformal
  Guarantees" (was "Calibration, Selective Prediction, and Class-Conditional Conformal
  Guarantees"). **Abstract** rewritten, 247 words (IEEE budget 250).
- **Contributions** 5 items → 7, reordered so the exhaustion result and the subgroup-conditional
  findings lead.
- **Related Work** gains §"Hidden stratification and subgroup underdiagnosis" (Oakden-Rayner,
  Seyyed-Kalantari, Geirhos, Zech, Hardt) with an explicit statement of the delta we claim over
  it — mechanism measured, safety mechanisms shown inoperative, mitigation priced. Calibration,
  imbalance and conformal subsections gain multicalibration, logit adjustment and equalised
  coverage.
- **Methods** gains §Out-of-fold predictions (fold protocol, leakage assertions, rare-class
  yields 71 `df` / 99 `vasc` / 64 escalating `<40`, and the stacking mismatch stated not assumed
  away), §External cohort, NNB and FRR as displayed equations, the bipartite calibrator, the
  OOF-conformal exchangeability caveat (no exact guarantee; CV+ would need a second test read),
  §The age-conditional escalation rule, and a rewritten §Statistical protocol covering the
  fixed interval rule, the seven declared comparison families, the frozen `analysis_plan.json`
  and the append-only test-pass receipt.
- **Results** gains §"Five combination levers, five negative results" (`tab:exhaustion`,
  including A7-oof −0.0176 and A8 −0.0237, both Holm p=0.242), per-band calibration
  (`tab:bandcal`), the OOF-fitted gate that flips margin→MSP, class-conditional-is-not-enough
  + `tab:bipartite` (under-40 escalating coverage **0.238 marginal → 0.571 class-conditional →
  0.952 bipartite**, and *no α in the grid bounds FRR below 0.01*), §"An age-conditional
  decision rule, and what it costs a clinic" (`tab:agerule`, `tab:ortho`, NNB with the
  prevalence caveat stated once for DCA/NNB/FRR together), §Intersectional slices,
  §External evaluation on PAD-UFES-20, and a rewritten §Explainability.
- **Limitations** rewritten: three items closed (K-fold now run, skin tone now measured for
  Fitzpatrick I–IV, CNN+transformer ensemble now evaluated and a wash), four added (stacking
  mismatch + lost conformal theorem, A8 under-sharpening, λ selected by clinical cost rather
  than equality of opportunity, orthogonality failure), and the rare-class attribution
  **corrected**: the *level* drag is `akiec` 0.697 / `mel` 0.699, the *variance* is `df` (width
  0.389) / `vasc` (0.308) — two different claims that the old text conflated.
- **References** 32 → 47 (all cited, none unused).

**Two code fixes made in service of the manuscript** (both real bugs, not cosmetics):

- `research/run_session7_stats.py::_render_table4` emitted `\begin{table}`; Table IV is seven
  columns and overflows a single IEEE column. Now `table*`.
- **`--table-only` silently reverted Table IV's test column to `\textsc{pending}`**, because
  `_render_table_only` never looked for the S9 test rows. Anyone re-rendering the table after
  S9 would have quietly un-filled it. Now reads `results/session9/age_gap_test.csv` when
  present (a frozen artifact, not a test read) and prints which case it took.

**Deliberately left for S11** (per the plan's session split): extending
`audit_manuscript.py` to ~100+ checks, regenerating `results/frozen_artifacts.json`, rebuilding
`paper/manuscript_overleaf.zip`, and attaching `results/CLAIM_checklist.md` as the supplementary
appendix the manuscript now cites (`\cite{tejani2024claim}`).

**Open issue for the user, not resolvable here.** The source roughly doubled. There is no LaTeX
toolchain on this machine, so the compiled page count is unknown, but the 9-page draft will grow
substantially and IEEE TMI charges over 10 pages. If a split is wanted, the cleanest supplementary
boundary is §External evaluation + §Intersectional slices + §Explainability, which are
self-contained and none of which carries a headline claim.


### S11 — audit, artifacts, supplementary, Overleaf bundle (2026-09-05)

Workstream F's remainder. No test read, no re-training, no new results.

**`audit_manuscript.py`: 83 → 300 checks.** The S10 staging verifiers are folded in and
deleted. Two new helpers: `present()` (a literal formatted from a `results/` artifact must
appear in the manuscript or one of its `\input` tables) and `row()` (a whole table row must
reconstruct cell-for-cell — the stricter form, and the one that caught S10's two rounding
errors). All four hand-written tables (`tab:exhaustion`, `tab:bandcal`, `tab:bipartite`,
`tab:agerule`, `tab:ortho`) are now verified at row level, not by substring.

Beyond the literals, ~20 **directional** claims are asserted rather than compared to a digit,
because if any of them flips a *paragraph* is wrong, not a number:

- both new rungs are still negative;
- `<40` still has the worst within-band escalation-mass AUC on test (the softened mechanism claim);
- under-40 conformal coverage still improves marginal `<` class-conditional `<` bipartite;
- post-Dirichlet per-band residuals still disagree in sign;
- `akiec`/`mel` still hold the lowest per-class F1 **and** `df`/`vasc` the widest intervals
  (both halves of the corrected Limitations attribution);
- errors still score *higher* on lesion-interior attribution (the predicted-class artefact);
- the PAD soft-vote still trails its own best member;
- the Fitzpatrick I–IV pattern is still non-monotonic;
- the `<40` orthogonality overlap is still total;
- the OOF conformal arm still reports **no** exact guarantee, and the val/OOF score selection
  is still `(margin, msp)`;
- every test-pass execution still carries the frozen plan's hash and no rerun reason.

**Three bugs found, two of them latent and destructive:**

1. **`build_paper_artifacts.py` would have erased the pre-registration record.**
   `freeze_artifacts()` rewrote `results/frozen_artifacts.json` wholesale, so re-running it
   would have silently deleted the `analysis_plan` key that `run_session9_plan.py` writes —
   the hash of the frozen analysis plan that the manuscript's own statistical-protocol
   section cites. Fixed to carry forward every key it does not own. Verified: after the
   re-run, all **34 prediction-matrix hashes are byte-identical** and `analysis_plan` survives.
2. **The Overleaf bundle was already broken.** `paper/manuscript_overleaf.zip` was assembled
   by hand in the post-compile pass and shipped **one of the three `\input` tables** — it
   predates `oof_vs_val.tex` and `table4_agegap.tex`, so uploading it would have failed to
   compile on a missing file. Replaced with `research/ablation/build_overleaf_bundle.py`,
   which keeps **no file list**: it reads `\input{}` and `\includegraphics{}` out of the
   manuscript, resolves each, and fails loudly on anything missing — the same check a compile
   performs, run where there is no compiler. Bundle is now **12 files, 3.44 MB** (was 9).
3. **The CLAIM checklist's section pointers were wrong, several of them before S10 ever ran.**
   Item 25 ("detailed description of the model") pointed at *Metrics*; item 31 ("metrics of
   model performance") pointed at *Selective prediction and conformal sets*; item 30
   ("ensembling") pointed at *Backbones and training*. S10 then shifted every letter again by
   adding three Methods subsections. All letter references are now **subsection names**, which
   do not rot. Item 26 also claimed the manuscript names its software stack; it did not, so a
   sentence naming Python 3.12 / PyTorch 2.11+cu128 / `timm` / scikit-learn 1.5 was added to
   §III *Backbones and training* rather than downgrading the item.

**CLAIM checklist brought current, and one item moved.** Item 35 (*validation or testing on
external data*) was **Not met — "in progress (S8)"**; S8b has since run, so it is now
**Partial**, with the honest wording that it is a dataset-shift benchmark on smartphone
photography rather than external validation of the intended use, and that no independent
same-modality dermoscopy cohort has been evaluated. Also refreshed: item 2 (abstract 260 →
**247** words), item 33 (the five negative levers and the NNB prevalence range), item 34 (the
quantitative lesion-interior fraction), item 38 (the external Fitzpatrick slice, and that it
says nothing about darker skin), item 41 (the real test rescue rates — the old row said
"33–39% in older bands", which the OOF-fitted gate does not support: it is 15.4% at 40-59 and
36.2% at 60+), item 42, item 43. **Counts moved 33/6/3/2 → 33 met / 7 partial / 2 not met /
2 N/A**, and the manuscript now states them rather than merely pointing at the file.

**New in `research/ablation/`:**

| Script | What it does |
|---|---|
| `build_supplementary.py` | renders `results/CLAIM_checklist.md` → `paper/supplementary.tex` (standalone `article`, `longtable`, 44 items, 336 lines). A generator, not a hand-edit: change the Markdown and re-run |
| `build_overleaf_bundle.py` | assembles the upload from the manuscript's own dependency list |
| `validate_structure.py` | promoted from S10 staging; stands in for the absent LaTeX compiler and now validates the supplementary too (unmapped unicode, unescaped specials, `longtable` row arity) |

`results/README.md`'s index went from 6 artifacts to 13 and now names the three scripts that
should pass before submission.

**Final state, all green:** `audit_manuscript.py` **300/300**; structural validation passes for
both documents (47 cites = 47 bibitems, 0 dangling, 0 duplicate labels; supplementary 14
longtables / 44 items); `run_part_a.py --table-only` still regenerates the 11-rung ladder and
the audit still passes afterwards; 34 frozen hashes unchanged.

⚠️ **Still not compiled.** There is no LaTeX toolchain on this machine and nothing here changes
that — the structural checks are a stand-in, not a substitute. Upload
`paper/manuscript_overleaf.zip` and compile `manuscript.tex` and `supplementary.tex` in
Overleaf. The 3 `TODO` markers (author list, repository URL, acknowledgements) are untouched
and remain pre-submission work, and the page-count question raised at the end of S10 is
unresolved until that compile happens.


### S12 — integrity remediation of the external battery (2026-09-05)

**No test read.** `results/test_pass_receipt.json` still records `n_executions: 2`, 19 quantities,
no rerun reason. This session's job was to make every external number data-derived and every
test-read claim true *before* any of it reaches the manuscript, per `DATASET_REFINING.md` §5.

#### What was actually wrong

An unlogged exploratory pass on the morning of 2026-09-05 produced seven external tables and two
figures under session labels "S12–S18, COMPLETED". The audit in `DATASET_REFINING.md` §2 found ten
defects; S12 closes the six it owns.

1. **A fabricated λ.** `eval_decision_curve.py` and `generate_case_atlas.py` both hardcoded
   `lambda_opt = 0.2818`, commented "from frozen analysis plan". The frozen file
   `research/agerule/results_oof/age_rule_lambda.json` holds per-band λ = **0.26 / 0.74 / 0.33**
   (pooled 0.65). That constant occurs in no `results/` file. Hard Rule 4 violation.
2. **A different rule under the frozen rule's name.** Both scripts implemented `S_esc ≥ λ`. The
   frozen rule is `argmax_c ( p_c + λ_band · 1[c escalates] )` — a per-band bonus inside the
   argmax, not a threshold on a summed score. These disagree on which lesions escalate.
3. **Unregistered test reads.** Five scripts read `research/predictions/{arch}_test.csv`. A sixth
   was found during this session and is *not* in the audit's list: `audit_shift_safety_nets.py`
   also read `research/selective/features/convnext_tiny_test.npz` for its Mahalanobis arm.
4. **Three pipelines in one battery.** Those reads were the *plain 1-view* matrix; every published
   number is 24-view TTA. E6/E7 applied no Dirichlet map at all.

#### Fixes

**1. One λ, loaded not typed — `research/external/frozen_params.py` (new).**
`load_lambda_by_band()` reads the frozen JSON; `apply_age_rule(probs, ages | bands)` delegates to
`research.agerule.lambda_rule.apply_lambda`, which is
`research.thresholds.optimize.apply_thresholds` with `theta_c = -λ` on the escalating classes — the
project's existing rule family, not a reimplementation. `age_bands()` reproduces
`research.selective.fairness.load_attributes`' banding and is asserted equal to it row-for-row.
`escalation_mass()` is still exported, because `S_esc` is a legitimate *ranking* score; what it is
not is a decision rule. Both fabricated-constant sites are gone and no `research/**/*.py` contains
the literal.

`$py -m research.external.frozen_params --selftest` — 6 checks, all passing — reproduces
`val_macro_f1_rule = 0.7713320988516693` and `val_macro_f1_base = 0.7638314390344817` exactly
(source: `research/agerule/results_oof/age_rule_lambda.json`).

**2. A finding the audit had not caught: three Dirichlet maps, and the plan named the wrong one.**
Reproducing the frozen val figure required identifying which OOF Dirichlet map the published system
actually deploys. There are three, and they are not close:

| File | vs. the map `run_session5_agerule` refits in process | val Macro-F1 of the λ rule |
|---|---|---|
| `research/selective/results_oof/fit_state.json` | max abs dW = max abs db = 0 — **bit-identical** | **0.7713320988516693** (the frozen figure) |
| `research/calibration/results_oof/fit_state.json` | max abs dW = 0.18 | 0.7715496085187014 |
| `research/conformal/results_oof/fit_state.json` | max abs dW = 1.01 | 0.7711978586371423 |

The first is also the map `research/session9/testpass.py` scores rung A7-oof and the age rule with,
so it is the only map under which the frozen λ and the S9 test anchor mean what they say. **The v1
pre-registration declared the second.** `frozen_params --selftest` now asserts the bit-identity, and
the v2 plan corrects the pointer (deviation D2). ⚠️ `research/xdomain/run_session8b.py` still loads
the calibration map, so its published PAD figures are not strictly comparable to this battery's PAD
panels until S8b is re-run — open for S13+.

**3. One pipeline.** Every HAM panel in `eval_decision_curve.py`, `generate_case_atlas.py`,
`eval_clinical_triage.py` and `audit_shift_safety_nets.py` now comes from
`frozen_params.load_ham_oof_panel()`: `research/predictions_oof_tta/{arch}_train.csv` (6,981 rows,
24-view TTA, cross-fitted by fold) → uniform soft-vote → the deployed Dirichlet map. The E4
Mahalanobis arm drops HAM test entirely and uses HAM **val** as the in-distribution reference, which
is what S8b already did and why its AUROC 0.913 / 18× separation figures are unchanged.

**4. A test anchor with no test read.** `eval_decision_curve.py:s9_test_anchor()` reconstructs net
benefit on HAM test from `results/session9/agerule_test.csv`, which records TP (`n_caught`),
`referral_rate` and `n` per band and rule — everything NB needs. Precedent: S10's
`_render_table_only` reading `age_gap_test.csv`. The column is labelled *"derived from frozen
Session 9 artifacts (no new read)"* in the table's own footnote.

**5. The pre-registration, amended honestly.** `scripts/external/freeze_analysis_plan_v2.py` writes
`results/external/analysis_plan_post_s11_v2.json` (sha256 `da3c2e1837a21cb3…`) and re-hashes
`post_s11_provenance.json` with **both** hashes. v1 is retained byte-for-byte — verified still
`e6193e191acba511…`. Seven deviations are recorded, and the note distinguishes the two made before
any corresponding data existed (D3 E0 withdrawn, D4 `E1_dose_response_skew_ordering` declared ahead
of S13) from the five that correct work already done under the v1 lock (D1 E6/E7 admitted post-hoc,
D2 the Dirichlet pointer, D5 the Level-0 panel, D6 the Mahalanobis test arm, D7 the conformal score
function). Those five are amendments to a broken record, not pre-registration, and say so.

**6. The DCA, corrected as statistics and not only as provenance.** Vickers' decision curve varies
the decision with `p_t`; the old code plotted one fixed rule at every `p_t` and compared it to
treat-all as though it were a curve. `eval_decision_curve.py` now draws three: the **risk model**
(biopsy iff `S_esc ≥ p_t`, the actual Vickers curve), and two **fixed operating points** (argmax,
and the frozen band-conditional λ rule) labelled as such. ΔNB against argmax carries a
**lesion-grouped paired bootstrap** interval (1,000 draws) via a new
`research/ablation/bootstrap.py:lesion_resample_indices()` generator — added rather than refactored
into the two existing functions, because changing their RNG consumption would silently move
published intervals. And the whole analysis is repeated **within the under-40 band**.

Source: `results/external/decision_curve_report.json`,
`paper/tables/external_table_decision_curve.tex`.

| Panel | p_t | Argmax NB | λ-rule NB | ΔNB [95% CI] | Test ΔNB (S9-derived) |
|---|---|---|---|---|---|
| All ages (N=6,981) | 0.05 | 0.1327 | 0.1589 | **+0.0263 [+0.0214, +0.0315]** | +0.0156 |
| All ages | 0.10 | 0.1306 | 0.1534 | **+0.0228 [+0.0178, +0.0282]** | +0.0114 |
| All ages | 0.20 | 0.1257 | 0.1403 | **+0.0146 [+0.0093, +0.0199]** | +0.0015 |
| Age <40 (N=1,319) | 0.05 | 0.0253 | 0.0280 | +0.0027 [−0.0002, +0.0065] | +0.0053 |
| Age <40 | 0.10 | 0.0239 | 0.0254 | +0.0015 [−0.0015, +0.0053] | +0.0034 |
| Age <40 | 0.20 | 0.0207 | 0.0193 | −0.0013 [−0.0053, +0.0029] | −0.0009 |

**The under-40 ΔNB is null at every reference threshold and turns negative by p_t = 0.20**, in the
OOF panel and in the S9-derived test column alike. That is consistent with everything the project
already knows — S5 measured the λ rule helping <40 *least* (val 0.545 → 0.591) — and it is reported
as measured. Only the λ-rule cells whose interval excludes zero are bolded in the LaTeX, so the
table cannot be read as claiming an under-40 win. PAD's ΔNB is large (+0.266 at p_t = 0.05) for the
reason S8b already gave: PAD's escalating prevalence is ~77%, the inverse prior, so a constant
escalating bonus helps mechanically.

**7. The case atlas, rebuilt on the real rule.** Exemplars are selected by *running* the rule
(`argmax` benign ∧ `apply_age_rule` escalating), not by comparing a score to a number, and
`_first_or_none` returns None rather than relaxing a filter — the old script silently dropped
Panel A's age constraint when its strict query came back empty. Panel A survives on the real
λ = 0.26: `ISIC_0030134`, one of **4 under-40 melanomas** the rule rescues out of **205 argmax
misses rescued overall**. Source: `results/external/case_atlas_report.json`.

**8. E0 withdrawn.** `train_comparison_arm.py` and `eval_comparison_arm.py` deleted (1.8 kB and
7.1 kB stubs for an arm that needs retraining, which the frozen-checkpoint rule forbids).
`results/external/comparison_arm/cell_a_control_report.json` is rewritten as a withdrawal notice
with the original numbers preserved under `withdrawn_payload` and marked uncitable — they came from
an unregistered test read. It enters Holm–Bonferroni at **p = 1.0** with the family denominator held
at 5; shrinking the denominator would inflate every survivor.
`paper/tables/appendix_table_tripod_ai.tex` row 20 claimed a comparison arm that reproduces the
blind spot — corrected to state the withdrawal.

**9. Traceability (A10).** `research/experiments.csv` had **0** rows matching `post_s11`; it now has
**30** under `session_post_s11`. `scripts/external/backfill_ledger.py` (idempotent) backfills the
retained workstreams from their own reports — E2 ×5 from `pad_prior_decoupling_report.json`, E5 ×7
from `fitzpatrick_slices.json` (V/VI marked suppressed), E0 ×1 withdrawal — and E3/E4/E6 now log at
run time.

**10. The pre-flight, rewritten so it can fail.** `scripts/external/preflight.py` replaces the
PowerShell heredoc in §8, which mixed a `need()` accumulator with bare `assert` (so the first
receipt problem threw before the list printed) and demanded S15's tables before S15 had run (so a
correct pre-S13 repo could only report failure). Now staged — `--stage pre_s13` / `--stage pre_s17`
— accumulating, and each failure names the session that owns it. Its fabricated-constant grep is
over `*.py`, because grepping all of `research/` matches ~70 prediction CSVs on coincidental float
digits and can never pass.

#### Acceptance (all met)

| Criterion | Result |
|---|---|
| fabricated constant in `research/**/*.py` | no matches |
| `predictions/*_test.csv` under `research/external/` | no matches |
| `results/test_pass_receipt.json` | `n_executions: 2`, no rerun reason — unchanged |
| `frozen_params --selftest` | reproduces `val_macro_f1_rule = 0.7713320988516693` |
| §8 pre-flight fails when a table is renamed | verified: exit 1, names the file |
| …and on a reintroduced constant, and a 1-byte plan edit | verified: exit 1 both times |
| `preflight.py --stage pre_s13` | 27 checks, **PASSED** |

#### Bugs found and fixed beyond the brief

- **APS conformal scoring was broken.** `audit_shift_safety_nets.py` reimplemented APS inline as a
  deterministic cumulative sum over sorted probabilities, while the stored quantiles were calibrated
  with the project's **randomised** `research.conformal.scores.aps_scores` (mass strictly above class
  `c` plus a uniform fraction of `c`'s own mass, plus the RAPS rank penalty). Comparing a threshold
  to a differently-scaled score gave APS Mondrian α = 0.05 an in-distribution marginal coverage of
  **0.3635** against a nominal 0.95. Importing the real score function and the stored RAPS
  hyperparameters gives **0.9465** marginal / **0.9495** serious. Same failure mode as the fabricated
  λ, one module over: a rule reimplemented beside its definition.
- **A sixth unregistered test read**, not in the audit's list of five: `convnext_tiny_test.npz` in
  the E4 Mahalanobis arm. Deleted; HAM val is the reference, as in S8b.
- **The conformal in-distribution reference was in-sample.** The quantiles are estimated on the
  calibration half, so the audit now scores the **tuning half**, which that estimate never saw. It is
  not fully held out — the Dirichlet map and RAPS hyperparameters were fitted on it — and the report
  states that rather than assuming it away. The claim E4 carries is the ID-to-shift *contrast*; S9
  owns absolute coverage, on test, once.

#### E3 / E4 numbers after the repoint

E3, source `results/external/clinical_triage_report.json` (HAM panel is now OOF, N = 6,981, not
test): tier-1 sensitivity **0.6578** calibrated / 0.6781 raw, specificity 0.9584 / 0.9509, point-FRR
**0.3289** / 0.2875, NNB(π = 0.03) **3.04** / 3.34. PAD unchanged in method.

E4, source `results/external/conformal_shift_audit.json`: Mahalanobis median 485 (HAM val) vs 8,717
(PAD), ratio **17.97×**, AUROC **0.9128** — the S8b figures. Conformal, ID (OOF tuning half) → PAD:
LAC marginal α = 0.10 coverage 0.9052 → 0.2379; LAC Mondrian α = 0.05 0.9365 → 0.6220; APS Mondrian
α = 0.05 0.9465 → 0.6548, mean set size 2.49 → 4.06. Sets widen under shift, as the workstream
predicted, but coverage collapses well below nominal — off-cohort conformal has no finite-sample
guarantee and this is what that costs.

#### Open items S12 did not close

- `research/xdomain/run_session8b.py` still loads the calibration-module Dirichlet map (see §2a of
  `DATASET_REFINING.md`). Re-running it would move published PAD figures; deferred deliberately.
- Three stubs remain for their owning sessions: `extract_external_predictions.py` and
  `assemble_external_ensemble.py` (S13), `eval_age_rule_transfer.py` (S14),
  `build_post_s11_artifacts.py` (S17).
- A9 (zero manuscript integration) is untouched — S15/S16 own it. `preflight.py --stage pre_s17`
  fails on exactly that, plus the two S15 composite tables, which is the correct state today.
- The ISIC-2019 download **is** complete (9.10 GB, 25,331 JPEGs in `data/external/isic2019_images/`,
  2026-09-05), closing audit item A7 ahead of S13.


## 11. Known findings that constrain later work

- **The soft-vote ensemble is under-confident, not over-confident.** Mean confidence 0.7048 vs.
  accuracy 0.8609 on test — signed gap −0.156, essentially all of the 0.1575 uncalibrated ECE.
  ConvNeXt-Tiny alone is also under-confident but less so (0.7606 vs 0.8296). This is the opposite
  of the typical single-network over-confidence result (Guo et al. 2017) and is *why* Dirichlet
  beats temperature scaling: soft-voting six members that disagree on the runner-up class
  systematically pulls the max probability down, and the resulting distortion is class-dependent.
  Per-band ECE (session 6, G.2): under-40 is simultaneously the *most* accurate band (0.914, easy
  nevi dominate) and the *worst* calibrated (ECE 0.1931, gap −0.193) — the calibration story and
  the fairness story are the same story.
- **Under-40 melanoma blind spot.** Escalation sensitivity 0.143 (3/21) under 40 vs. ~0.78 in older
  bands at full coverage, TTA+Dirichlet ensemble. Training prior: 4.9% escalating under 40 vs.
  35.5% at 60+ (`results/age_band_prior.csv`). Abstention does not rescue it — only 11.1% of
  under-40 misses get referred (vs. 33–39% in older bands) — the model is *confidently* wrong, not
  uncertain. See §10 S5 for the mitigation.
  **Three S5 amendments this bullet must be read with** (§10 S5):
  (1) the failure is **not purely a decision-rule failure** — the under-40 band has the lowest
  escalation-mass AUC as well as the lowest sensitivity (0.889 vs 0.953/0.933 out-of-fold), so part
  of the gap is weaker ranking that no threshold can recover, though the CI overlaps the other
  bands on 64 cases;
  (2) **0.143 rests on 21 positives** and the better-powered OOF estimate of the same band is 0.547
  — different split, different (fold) models, so not like-for-like, but the paper should stop
  treating 0.143 as a precise quantity;
  (3) the confidently-wrong property **reproduces under `msp`** (10.0% of misses referred) as well
  as under `margin` (11.1%), on validation as well as test — it is a property of the failure, not
  of the uncertainty score.
- **The ITA skin-tone proxy is invalid on dermoscopy** — `ml/ood/skin_tone_slice.py` documents its
  own failure (vignetting and erythema dominate the angle metric on dermoscopy images). Fitzpatrick
  slicing needs PAD-UFES-20 images restored locally or Fitzpatrick17k. `ml/ood/*` also imports a
  nonexistent `backend.` package and does not run in this repo copy.

## 12. Bugs found and fixed

| Bug | Symptom | Fix | Where |
|---|---|---|---|
| DeLong pooled-midrank bug | All 42 per-class AUCs ≈ 0.5 | Rank the reordered `[pos\|neg]` array, not the original case order | `research/ablation/delong.py`, session 5 |
| pandas ≥2.2 `groupby.apply` silently drops grouping column | `select_samples()` lost `class_code` | Re-select `[frame.columns]` before `.apply` | `ml/explainability/generate_gradcam.py`, session 5 |
| L1: Dirichlet fitted on the wrong half | Small coverage bias | Fit Dirichlet on the tuning half only, re-run | `run_session4_conformal.py`, session 5 |
| Selective logging mislabelled 20% abstention point as `abstain10` | One row per sweep point, wrong name | Log the correctly-named row per sweep point | `run_session4_selective.py`, session 5 |
| Hand-entered "N=1104 lesions" in a figure caption | Wrong lesion count | Recomputed from data: true count is 1121 | manuscript, session 5 |
| Calibration direction stated backwards | Discussion/Fig.2 caption/Related Work said "overconfident" | Corrected to "underconfident" everywhere, asserted in the audit script | post-compile pass |
| A.6 as specified would have read test twice before S9 | `run_comparison` runs each runner twice, and every runner fits *and* reads test in one pass | `--no-test` fit-only mode on all four runners, enforced by a process-wide loader lock rather than by inspection | `research/testguard.py`, session 6 S4 |
| Mahalanobis is doubly in-sample on the OOF fit split | Gaussians fitted on train features from the full-train checkpoints, then used to score those same training rows — the abstention quantiles would be meaningless | Disabled by default under `--fit-split oof`; `--allow-oof-mahalanobis` overrides and the report says which applied | `run_session4_selective.py`, session 6 S4 |
| Lambda grid was the binding constraint, not the data | Pooled and `40-59` fits both selected 0.59, the largest value on a `[0, 0.6]` grid; widening to `[0, 1.2]` moved them to 0.65 and 0.74 | `fit_lambda` raises `GridBoundaryError` on a boundary optimum, so a clipped parameter cannot be frozen into a pre-registration | `research/agerule/lambda_rule.py`, session 6 S5 |
| `msp` reimplemented instead of imported, inverting it | `research/selective/scores.py` defines `msp` as `1 - max p` with referral *above* threshold; the draft used `max p` and `<`, reporting a 0.7% abstention rate and nearly supporting a false claim that the pre-registered threshold fails to transfer (it transfers: 10.6% realised vs 10% nominal) | Import `predictive_scores`/`coverage_threshold`; record the realised rate beside the nominal one | `research/run_session5_agerule.py`, session 6 S5 |
| 7-parameter ablation read on an incomparable scale | Raw bootstrap SD 0.137 (theta) vs 0.150 (lambda) reads as the 7-vector being *more* stable, but theta is searched over a range of 0.6 and lambda over 1.2 | Report spread as a share of the range searched: 0.229 vs 0.125, ~1.8x wider from seven parameters instead of one | `research/agerule/lambda_rule.py`, session 6 S5 |
| Stability bootstrapped for bands that never used the fitted value | `unknown` band printed a bootstrap mean of 0.00 beside an applied (pooled fallback) lambda of 0.59 | Bootstrap only bands that cleared the positives gate | `research/run_session5_agerule.py`, session 6 S5 |
| Applied-rule ledger row logged `escalation_sens` as `nan` | Wrong metric key — `compute_metrics` nests it at `clinical.binary_sensitivity` | Correct key; pre-fix ledger rows pruned before the final run | `research/run_session5_agerule.py`, session 6 S5 |
| A val *fit-only* run would have inherited the published ledger session | Rows with no test split behind them, labelled `session2`/`session4` and indistinguishable from published ones | `resolve_fit` refuses the published session for any non-published run; fit-only val arms log as `session2_valfit`/`session4_valfit` | `research/fitsplit.py`, session 6 S4 |

Three latent bugs identified but not yet triggered (session 6 plan, item A.0 — fix-first before any
K-fold training run):

1. `research/tta/extract_tta_predictions.py:98` would silently overwrite the frozen
   `research/predictions_tta/{arch}_test.csv` if a fold run used the default `--out-dir` — highest
   severity, needs `--out-name`.
2. `ml/training/train.py:396` writes `{arch}_training_history.json` with no `--checkpoint-tag` —
   any two runs of one architecture with different tags already clobber each other's history file.
   **The damage was quantified in S7 and is permanent for the pre-S1 runs**: of 54 training runs
   in the project, only **41** have a recoverable epoch count (1,003 epochs, of which 754 are the
   30 OOF folds). Exactly **one untagged history survives per CNN**, and since HAM-only, PAD-fresh
   and PAD-warm all wrote to the same filename, *which run each surviving file describes is
   unknowable*. The 12 PAD-family runs and MaxViT's failed first attempt have no epoch count at
   all. This bug was caught before the 30 fold runs, which is the only reason the OOF histories
   are intact.
3. `research/run_session4_selective.py:60-69` silently falls back from `predictions_tta` to
   `predictions` on `FileNotFoundError` — a half-extracted OOF directory would emit val-fitted
   numbers mislabelled as OOF.

## 13. Rejected, and why

- **Colour constancy** — never implemented (no `shades_of_grey`/`color_constancy` anywhere in the
  repo); dropped from the ablation ladder rather than left as a phantom rung.
- **ITA skin-tone proxy** — documented invalid on dermoscopy by its own module docstring (§11).
- **Ridge stacking as the reported ensemble** — best test score, worst val_oof score; rejected as
  test-set selection (§4).
- **Cost-sensitive thresholds on the main ladder** — demoted in session 5: costs 0.062 Macro-F1 and
  has zero marginal effect once margin-abstention is already applied (§5).
- **8-member CNN+transformer ensemble, Caruana selection over 8, global/per-class prior
  correction** — all tested with cross-fitting in session 6 and came back wash/worse/dead/noise
  (§10) — reported as a negative result rather than omitted.

## 14. Number → source map

| Number | File |
|---|---|
| Per-rung Macro-F1 / Balanced Acc / Escalation Sens. + CIs | `results/ablation_table.csv` |
| Bootstrap 95% CIs | `results/bootstrap_cis.json` |
| McNemar / DeLong / Holm results | `results/mcnemar_delong.json` |
| Frozen prediction-matrix hashes (34 files) | `results/frozen_artifacts.json` |
| Age-band escalating-class prior | `results/age_band_prior.csv` |
| Ensembling method comparison | `research/ensembling/results/report.md` |
| Calibration / thresholds / DCA | `research/calibration/results/session2_report.md` |
| TTA gains | `research/tta/results/report.md` |
| Selective AURC, risk-coverage, fairness gaps | `research/selective/results/session4_report.md` |
| Conformal coverage, false reassurance | `research/conformal/results/session4_conformal_report.md` |
| Consolidated per-phase reports | `results/reports/` |
| Age-band lambda rule, diagnosis, stability, held-out check | `research/agerule/results_oof/session5_agerule_report.md`, `age_rule_lambda.json` |

**Correction made alongside this document**: CLAUDE.md's session-4 conformal line previously read
"Mondrian 89.3%/24, RAPS+Mondrian best 92.4%/12" — that predates the L1 Dirichlet-half-split fix
and is stale. The authoritative, current numbers (91.4%/17 and 94.1%/6) are in
`research/conformal/results/session4_conformal_report.md` and now in CLAUDE.md.

## 15. Likely viva questions with defensible answers

1. **Why not accuracy?** `nv` is 67% of the data; an always-`nv` classifier scores ~67% while
   missing every malignant case. See Hard Rule 3, §1.
2. **Why lesion-grouped splits?** Multiple images per lesion would otherwise let the same lesion's
   images leak across train/val/test — near-duplicate leakage, not genuine generalization. §1, §2.
3. **How do you know you didn't overfit test?** Every fitted quantity (ensemble weights,
   calibrator, thresholds, abstention cutoffs, λ for the age rule) is chosen on val or OOF, test is
   read once per experiment, and the one place this discipline was nearly violated in-project
   (ridge stacking, §4) is documented as the counter-example, not hidden.
4. **Why is the ensemble *under*-confident when Guo et al. found single networks over-confident?**
   Soft-voting six models that disagree on the runner-up class pulls max-probability down; the
   distortion is class- and subgroup-dependent (§11), which is also why Dirichlet (not temperature
   scaling) is the calibrator that wins.
5. **Why does margin-based abstention fail to rescue under-40 melanoma misses?** The model is
   *confidently* wrong in that band (only 11.1% of misses referred vs 33–39% elsewhere) — margin
   measures runner-up closeness, not correctness, and a wrong-but-confident prediction has a large
   margin. §10/§11.
6. **What does the conformal coverage guarantee actually promise?** Marginal coverage only, which
   is compatible with badly under-covering a minority subgroup (malignant lesions, §7) — hence
   Mondrian class-conditional calibration, and hence why the paper reports false-reassurance count
   as the clinically meaningful number instead of the marginal rate alone.
7. **Why does Mahalanobis distance rank last among uncertainty scores?** It detects distribution
   shift; HAM10000 test is in-distribution relative to train, so there is nothing for it to detect,
   and it ends up worse than the probability vector at ranking in-distribution difficulty. §7.
8. **What would you do with more compute?** Fold-bagged (K-fold OOF) ensembling — the one
   remaining credible Macro-F1 lever after four other post-hoc combination levers came back dead —
   plus restoring PAD-UFES-20 for a genuine cross-domain/shift benchmark. §10, active plan.
