# Session 5, Part B — Ablation Audit, Ladder Design & End-to-End Leak Review

Run 2026-09-03. Everything below was re-derived from `research/predictions/`,
`research/predictions_tta/`, `research/experiments.csv`, `ml/results/` and
`ml/configs/splits/split_v1.csv`. No new training. No new test-set experiment —
the two verification scripts re-computed numbers from prediction matrices that
were already extracted, so no additional test read was consumed.

---

## 0. Two corrections to `CLAUDE.md` before anything else

`CLAUDE.md` is stale on two points, and both change the ladder:

1. **MaxViT-Tiny was trained and evaluated.** `CLAUDE.md` says the smoke test
   failed and it was never trained. In fact `research/session3_logs/15_train_maxvit_tiny_retry.log`
   and `16_eval_maxvit_tiny_retry.log` exist, `ml/checkpoints/maxvit_tiny_best.pt`
   is a real 122 MB checkpoint, and `ml/results/maxvit_tiny/metrics.json` reports
   **test Macro-F1 0.7525** — the best single model in the project, ahead of
   ConvNeXt-Tiny's 0.7459.
2. **Gated fusion has a test read.** `CLAUDE.md` says "val Macro-F1 0.7643" only.
   `ml/results/gated_fusion_convnext_tiny/metrics.json` reports **test Macro-F1
   0.7411**, logged 2026-09-03 (`14_eval_fusion.log`).

Both were completed in session 3 after the `CLAUDE.md` phase notes were written.

---

## 1. Rung-by-rung audit of the roadmap's 9-step ladder

Verdict key: **OK** = a val-selected, test-read number exists now.
**FIX** = the artifact is missing but recoverable without retraining.
**BLOCKED** = requires new training.

| # | Roadmap rung | Verdict | Test Macro-F1 | Where the number comes from |
|---|---|---|---|---|
| 1 | Baseline ResNet-50 | **OK** | 0.7058 | `research/predictions/resnet50_test.csv` |
| 2 | Best single model (ConvNeXt-Tiny) | **OK** | 0.7459 | `research/predictions/convnext_tiny_test.csv` |
| 3 | + Shades-of-Grey color constancy | **BLOCKED** | — | *not implemented anywhere in the repo* |
| 4 | + Vision Transformer (Swin-V2) | **FIX** | 0.7273 (standalone only) | `ml/results/swinv2_tiny/` — no val or TTA predictions, so it is **not in the ensemble** |
| 5 | + Multimodal metadata fusion | **FIX** | 0.7411 (standalone only) | `ml/results/gated_fusion_convnext_tiny/metrics.json` — **no per-image predictions written at all** |
| 6 | + OOF stacking ensemble | **OK, but the roadmap picks the wrong method** | 0.7718 | see §2 |
| 7 | + TTA | **OK** | 0.7859 | `research/predictions_tta/`, uniform soft-vote |
| 8 | + Dirichlet calibration & cost-sensitive thresholds | **OK for Dirichlet; the threshold half is degenerate** | 0.8047 | see §3 |
| 9 | + Selective abstention (top 10% deferred) | **OK, but not a comparable row** | 0.8577 @ coverage 0.892 | see §4 |

### Detail on the three problem rungs

**Rung 3 — colour constancy does not exist.** `grep -i "shades\|grey_world\|gray_world\|color_constancy\|minkowski"` across `ml/` and `research/` returns nothing. This is not "never run as an isolated ablation"; the preprocessing step was never written. Populating it costs one new transform plus a full ConvNeXt-Tiny retrain, minimum, and honestly a retrain of every ensemble member if the claim is to be about the ensemble.

**Rung 4 — Swin-V2 is trained but architecturally outside the ensemble.** `research/ensembling/data.py:ARCHS` is a hardcoded 6-tuple of CNNs. Every downstream stage — soft-vote, TTA pooling, Dirichlet, selective, conformal — calls `load_split_matrix()` with that default. Swin-V2 and MaxViT have `ml/results/<arch>/predictions.csv`, but those are **test-split only**, carry probabilities without logits, and have no TTA counterpart. So the "+ Vision Transformer inclusion" rung currently measures nothing about the ensemble.

**Rung 5 — fusion has no per-image predictions.** `research/fusion/evaluate_fusion.py` is hardcoded to the test split (line 59) and writes a single `metrics.json` (line 73). There is no `predictions.csv`. Consequence: the fusion rung cannot be bootstrapped, cannot be McNemar-tested against anything, and cannot enter the ensemble. It is a bare point estimate.

---

## 2. Rung 6 is a test-set-selection trap

Session-1 results, all from `research/experiments.csv`:

| Method | val_oof Macro-F1 (selection) | test Macro-F1 |
|---|---:|---:|
| uniform soft-vote (arithmetic) | **0.7911** | 0.7718 |
| Nelder-Mead simplex | 0.7911 | 0.7718 |
| Caruana greedy | 0.7798 | 0.7688 |
| non-negative ridge stacking | 0.7583 | **0.7815** |

The ridge stacker has the **best test** score and the **worst val_oof** score. If the ablation table reports 0.7815 for "+ OOF stacking ensemble", that number was selected by looking at the test set — a direct violation of hard rule 5 and exactly what a TMI reviewer looks for. The val-selected winner is the uniform soft-vote at **0.7718**, and Nelder-Mead converging to uniform weights (`[0.167]×6`) is corroborating evidence, not a separate method.

**Decision: rung 6 reports 0.7718 (uniform soft-vote).** The ridge stacker's 0.7815 belongs in the text as a negative result — "the stacker that would have won on test was rejected on validation" — which is a *stronger* methodological claim than quietly reporting it.

The OOF machinery itself is sound: `research/ensembling/oof.py` uses `StratifiedGroupKFold` inside the val split with lesion groups, refits on full val, applies once to test. Generalization gaps (`val_oof − test`) are +0.019 (soft-vote), +0.011 (Caruana), −0.023 (ridge) — all inside the roadmap's E1 veto band of 0.02 except ridge, which is negative anyway.

---

## 3. Rung 8: Dirichlet earns its place, the cost thresholds do not

Re-derived, both preprocessing paths, uniform soft-vote base:

| Stage | plain val | plain test | TTA val | TTA test |
|---|---:|---:|---:|---:|
| soft-vote ×6 | 0.7911 | 0.7718 | 0.7986 | 0.7859 |
| + Dirichlet | 0.7825 | 0.7907 | 0.7969 | **0.8047** |
| + cost-sensitive thresholds | — | 0.7490 | — | 0.7429 |

The TTA+Dirichlet figure 0.8047 reproduces `selective_margin_abstain00` in `experiments.csv` to 4 dp, confirming the pipeline.

Cost-sensitive thresholding costs **−0.062 Macro-F1** on top of Dirichlet. It buys escalation sensitivity in the session-2 uncalibrated setting (0.783 → 0.845), but on the calibrated TTA ensemble it is a straight loss on the primary metric. Worse, it is **redundant with abstention**:

| Abstention rate | images kept | threshold-flipped predictions *inside the kept set* |
|---:|---:|---:|
| 0% | 1502 | 168 |
| 5% | 1433 | 128 |
| 10% | 1340 | 80 |
| 15% | 1255 | 27 |
| **20%** | **1170** | **0** |

All 168 predictions the cost rule changes are low-margin cases, and by 20% abstention every one of them has already been referred. The two mechanisms act on the same images.

**Two bugs found in `research/run_session4_selective.py` while establishing this:**

- **Mislabeled log row.** Line 219 builds the method name from `args.operating_point` (default 0.10) but logs `cs_points[-1]`, the last sweep point, which is the **0.20** rate. `experiments.csv` therefore contains a row named `selective_cost_sensitive_abstain10` whose contents are the 20% operating point — and whose own `notes` field correctly says "abstention at 20%". Any table generated by keying on the method name will be wrong.
- **That row's Macro-F1 (0.8956567253851093) is byte-identical to `selective_margin_abstain20`.** That is not a duplication bug — it is the zero-row of the table above, and it is a genuine finding. But as logged it looks like a copy-paste error, and it must be explained in the paper or it reads as one.

**Decision: split rung 8.** "+ Dirichlet calibration" is a real rung (+0.019). Cost-sensitive thresholding moves to a separate decision-rule sub-analysis reported on the escalation-sensitivity axis, where it actually wins, not on Macro-F1, where it loses.

---

## 4. Rung 9 changes the denominator

`selective_margin_abstain10` scores 0.8577 on **1340 images**; every rung above it scores on **1502**. Stacking them in one column implies a like-for-like gain of +0.053 that does not exist — the model got easier cases, it did not get better. This is the single most likely figure in the paper to be attacked.

**Decision:** the selective rows live in their own block, with a `coverage` column, below a horizontal rule, and the caption states the denominator change explicitly. The risk-coverage curve carries the real claim.

---

## 5. Redesigned ladder

Nine rungs become eight, in two blocks. All selection on validation; one test read per row.

**Block A — full coverage, N = 1502, monotone**

| # | Rung | Status | Test Macro-F1 |
|---|---|---|---|
| A1 | ResNet-50 (weakest CNN member) | ready | 0.7058 |
| A2 | ConvNeXt-Tiny (best single on val) | ready | 0.7459 |
| A3 | + transformer member (Swin-V2, MaxViT) | needs extraction | TBD |
| A4 | + metadata fusion member | needs prediction dump | TBD |
| A5 | uniform soft-vote over K members | ready | 0.7718 (K=6) |
| A6 | + 24-view multi-scale dihedral TTA | ready | 0.7859 |
| A7 | + Dirichlet calibration | ready | 0.8047 |

**Block B — selective, coverage-annotated**

| # | Rung | Coverage | Test Macro-F1 |
|---|---|---:|---:|
| B1 | + margin abstention @5% | 0.954 | 0.8298 |
| B2 | + margin abstention @10% | 0.892 | 0.8577 |
| B3 | + margin abstention @15% | 0.836 | 0.8696 |
| B4 | + margin abstention @20% | 0.779 | 0.8957 |

**Dropped from the roadmap ladder:** colour constancy (rung 3, not implemented — see §7 for the call), and cost-sensitive thresholds as a Macro-F1 rung (moved to the decision-rule sub-analysis).

**Note on A2 vs A3:** MaxViT-Tiny's standalone test Macro-F1 (0.7525) beats ConvNeXt-Tiny's (0.7459), so "best single model" may change once val predictions exist. It must be re-selected **on val**, which is precisely why the extraction in §8 has to happen before A2 is written down.

---

## 6. End-to-end leak review

### Clean — verified, no action

- **Split integrity.** `split_v1.csv`: 10015 rows, 7470 lesions, train 6981 / val 1532 / test 1502. **Zero lesion_ids span more than one split.** Hard rule 1 holds.
- **Base models are out-of-fold w.r.t. the meta-learners.** All six CNNs are frozen; their val predictions were never trained on.
- **Inner CV is group-aware.** `oof.py` uses `StratifiedGroupKFold` with lesion groups and asserts full fold coverage.
- **Mahalanobis is fit on train only.** `research/selective/mahalanobis.py` fits class means and a Ledoit-Wolf tied covariance on training-split features; val sets the threshold, test is scored once.
- **TTA uses eval transforms only.** `research/tta/transforms.py` composes Resize → CenterCrop → Normalize; the 24 views are deterministic D4 × 3 scales. No train-time randomness leaks into inference.
- **The headline ensemble is parameter-free.** Uniform soft-vote fits nothing, so its val score is honest by construction.

### Finding L1 — conformal calibration half was seen by the Dirichlet fit *(moderate; fix before final numbers)*

`research/run_session4_conformal.py` line 109 fits the Dirichlet calibrator on the **full** validation split, and *then* line 113 splits val into grouped tuning/calibration halves. The conformal calibration half therefore contributed to the calibrator whose outputs it is calibrating. Conformal's coverage guarantee requires the calibration set to be exchangeable with test w.r.t. the *fixed* score function; here the score function has seen half of its own calibration data.

The observed damage is small — LAC marginal coverage 0.9015 against a nominal 0.90 — but the guarantee as stated in the paper would be unsupported. **Fix: fit the Dirichlet calibrator on the tuning half only, then re-run.** One-line change, no retraining, and it costs the calibrator half its fitting data (766 images), which is acceptable.

### Finding L2 — validation is doing six jobs *(structural; disclose, partially fixable)*

The same 1532 val images are used for: (a) early stopping and checkpoint selection for all six backbones, (b) ensemble method selection, (c) the Dirichlet fit, (d) cost-threshold optimization, (e) abstention quantiles, (f) conformal calibration. Nothing here is a hard leak into test, but every val-derived threshold is fitted on data the backbones were already optimized against, so val-derived operating points are optimistic.

The signature is visible and consistent: requested abstention 5/10/15/20% yields test coverage 0.954 / 0.892 / 0.836 / 0.779 — the realized coverage runs ~1 pp below nominal at every point, i.e. the model abstains slightly *more* on test than the val quantile promised. That is exactly what an in-sample threshold looks like.

**Options, in order of cost:**
1. **Disclose** — report the nominal-vs-realized coverage table above as evidence the effect is bounded at ~1 pp. Zero cost, and it is honest. *Recommended for this paper.*
2. **Grouped val halving** — one half fits everything (Dirichlet, thresholds, conformal), the other selects methods. Cheap, no retraining, and it subsumes the L1 fix. Costs statistical power on `df`/`vasc`, which are already thin.
3. **Train-split OOF via K-fold retraining** — the correct fix, roughly 5× the training budget. Also the only thing that solves the `df`/`vasc` conformal problem in `CLAUDE.md`. Out of scope unless the schedule allows.

### Finding L3 — 41 logged test reads *(disclose)*

`research/experiments.csv` contains 41 rows with `split=test`. Hard rule 2 says the test set is touched once *per experiment*, which is literally satisfied, but no reviewer will accept 41 test evaluations without a statement. **Fix: freeze one final artifact set before Part A**, declare in Methods that all reported test numbers derive from that frozen set, and treat the rest of `experiments.csv` as the development log.

### Finding L4 — `results/` is empty *(process)*

The roadmap promises "100% of paper tables and plots compile directly from `results/`", and hard rule 4 says every number comes from `results/`. The top-level `results/` directory is **empty**. Real numbers are scattered across `ml/results/`, `research/*/results/`, and two different `experiments.csv` files (`research/experiments.csv` and `ml/results/experiments.csv`, with different schemas). `paper/figures/` and `paper/tables/` are also empty.

**Fix: Part A opens with an assembly step** that consolidates the frozen artifact set into `results/` and generates every table and figure from there. Otherwise the manuscript will be hand-transcribed, which is exactly what hard rule 4 exists to prevent.

### Finding L5 — checkpoint glob collision *(low, but it will bite)*

`ml/checkpoints/` holds both `resnet50_best.HAM-only.pt` and `resnet50_best.pt` (the deployed copy). A glob of `*_best.pt` matches `resnet50_best.pt`, `swinv2_tiny_best.pt`, `maxvit_tiny_best.pt`, `convnext_tiny-asl_best.pt` and `convnext_tiny-ldam_drw_best.pt` — so the "extract the transformers" command in §8 must name checkpoints one at a time, or ResNet-50 gets silently double-counted in the ensemble.

---

## 7. The colour-constancy call

Rung 3 needs a decision, and it is not a coding decision.

- **Implementing it properly** means a Shades-of-Grey transform plus retraining at least ConvNeXt-Tiny, and arguably all six members for the ensemble claim. That is the largest remaining compute item in the project.
- **Dropping it** costs one row and the E2 preprocessing narrative, but the paper already carries a much stronger preprocessing-adjacent result (the dermoscopy→smartphone domain collapse in `ml/results/RESULTS_SUMMARY.md`: Macro-F1 0.706 → 0.142 cross-domain).

**Recommendation: drop it from the ladder** and state in Methods that colour constancy was not evaluated. Do not leave the row in the table with a dash — an empty ablation row invites the reviewer to ask for it. If a single retrain is affordable, run it as ConvNeXt-Tiny only and report it as a single-model preprocessing ablation *outside* the main ladder, where it does not need to propagate through the ensemble.

---

## 8. Work queue before Part A can compute anything

Ordered. Items 1–3 are inference only, no training.

**1. Extract Swin-V2 and MaxViT predictions (plain, val + test)** — one arch per invocation, per L5:

```bash
python -m research.extract_predictions --checkpoints-glob "swinv2_tiny_best.pt"
```

```bash
python -m research.extract_predictions --checkpoints-glob "maxvit_tiny_best.pt"
```

**2. Patch the TTA extractor, then extract TTA predictions.** `research/tta/extract_tta_predictions.py:46` hardcodes `{arch}_best.HAM-only.pt`; it needs a fallback to `{arch}_best.pt`. Then:

```bash
python -m research.tta.extract_tta_predictions --archs swinv2_tiny maxvit_tiny
```

Budget: ~24 views × 3034 images. At the logged 24.1 ms/img for MaxViT and 12.1 ms/img for Swin-V2, expect roughly 30 and 15 minutes respectively.

**3. Give the fusion model a prediction dump.** `research/fusion/evaluate_fusion.py` needs a `--split` argument and a `predictions.csv` writer in the `research/predictions/` schema (`image_id, true_index, true_code, pred_index, pred_code, p_*, logit_*, arch, split, checkpoint`). Without this, rung A4 stays a bare point estimate that cannot be bootstrapped.

**4. Thread `--archs` through the session runners.** `research/ensembling/data.py:ARCHS` is the default in `run_session2_calibration.py`, `run_session4_selective.py`, `run_session4_conformal.py` and `run_ensembling.py`. Add the flag so the K=6 CNN ensemble and the K=8 mixed ensemble are both reproducible.

**5. Apply the L1 fix** — fit Dirichlet on the conformal tuning half — and re-run session 4's conformal block.

**6. Re-select A2 on validation** once step 1 lands. If MaxViT-Tiny wins on val, "best single model" changes and rungs A2/A3 need rewriting.

**7. Freeze the artifact set** (L3) and assemble `results/` (L4).

---

## 9. What Part A inherits

With items 1–7 done, Part A's statistics work has a well-defined target:

- **Bootstrap CIs (1000× stratified, lesion-grouped):** 11 ladder rows — 7 in block A, 4 in block B. Every one has, or will have, a per-image prediction file, which is what the bootstrap resamples.
- **McNemar:** the adjacent pairs within block A only (A1→A2, A2→A5, A5→A6, A6→A7, plus A3/A4 once populated). Block B pairs are not McNemar-testable against block A — different denominators.
- **DeLong:** multi-class ROC-AUC comparisons need probability vectors, so the same prediction-file requirement applies; the fusion model is the blocker here too.
- **Non-negotiable:** bootstrap must resample **lesions, not images**, or the CIs will be too narrow — 10015 images sit on only 7470 lesions, and the test split's repeated-lesion structure is exactly what group-aware splitting was protecting.
