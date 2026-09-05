# Session 6 — Hierarchical Bipartite Conformal & False Reassurance Rate

`fit on oof (train split, research/predictions_oof_tta, TTA=True); selection on val (research/predictions_tta); test=LOCKED; out=research/conformal/results_hierarchical_oof; session=session6_hier_oof`

Base system: 24-view TTA, uniform soft-vote over 6 backbones, Dirichlet calibration fitted on the tuning half. Calibration quantiles come from out-of-fold predictions over the 6981 training images (`research/predictions_oof_tta`), split into a tuning half (3513) and a calibration half (3468) grouped by `lesion_id`. Every number is measured on **val** (n=1532); **test is locked** by `research.testguard`.

## What is being compared

| Calibrator | Threshold granularity | Cells |
|---|---|---:|
| marginal | one for everything | 1 |
| class-conditional | one per class | 7 |
| bipartite | one per (age band x escalation requirement) | 8 |

The full cross (age band x class) would be 28 cells and is not evaluated: a finite threshold needs n >= 1/alpha - 1 calibration points (alpha=0.10 -> 9, alpha=0.05 -> 19), and the escalating classes in the `<40` band do not have them individually. Grouping the three escalating classes into one cell is what makes a band-conditional guarantee certifiable at all.

## Results

**FRR** is the share of truly escalating lesions whose set contains no escalating class (akiec, bcc, mel) — the failure marginal coverage averages away. **<40 esc. cov.** is coverage inside the `(<40, escalating)` cell, the paper's headline subgroup.

| alpha | Score | Calibrator | Marginal cov. | Serious cov. | <40 esc. cov. | Mean size | FRR | FRR 95% |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 0.10 | LAC | marginal | 0.9047 | 0.7630 | 0.5909 | 1.086 | 0.2208 (68/308) | [0.1596, 0.2857] boot |
| 0.10 | LAC | class-conditional | 0.8760 | 0.9188 | 0.8636 | 1.259 | 0.0649 (20/308) | [0.0401, 0.0985] CP |
| 0.10 | LAC | bipartite | 0.8923 | 0.9123 | 1.0000 | 1.226 | 0.0779 (24/308) | [0.0506, 0.1137] CP |
| 0.10 | APS | marginal | 0.8936 | 0.8409 | 0.7273 | 1.268 | 0.1396 (43/308) | [0.0942, 0.1873] boot |
| 0.10 | APS | class-conditional | 0.8864 | 0.9026 | 0.7273 | 1.435 | 0.0844 (26/308) | [0.0559, 0.1212] CP |
| 0.10 | APS | bipartite | 0.8890 | 0.8994 | 0.9545 | 1.366 | 0.0844 (26/308) | [0.0559, 0.1212] CP |
| 0.10 | RAPS | marginal | 0.9034 | 0.7630 | 0.5455 | 1.110 | 0.2078 (64/308) | [0.1489, 0.2678] boot |
| 0.10 | RAPS | class-conditional | 0.8884 | 0.9058 | 0.7727 | 1.383 | 0.0682 (21/308) | [0.0427, 0.1023] CP |
| 0.10 | RAPS | bipartite | 0.8975 | 0.9058 | 0.9545 | 1.285 | 0.0682 (21/308) | [0.0427, 0.1023] CP |
| 0.05 | LAC | marginal | 0.9550 | 0.8799 | 0.7727 | 1.315 | 0.1071 (33/308) | [0.0627, 0.1576] boot |
| 0.05 | LAC | class-conditional | 0.9380 | 0.9545 | 0.9091 | 2.229 | 0.0325 (10/308) | [0.0157, 0.0589] CP |
| 0.05 | LAC | bipartite | 0.9478 | 0.9643 | 1.0000 | 1.507 | 0.0260 (8/308) | [0.0113, 0.0505] CP |
| 0.05 | APS | marginal | 0.9497 | 0.9058 | 0.7273 | 1.473 | 0.0812 (25/308) | [0.0532, 0.1175] CP |
| 0.05 | APS | class-conditional | 0.9458 | 0.9545 | 0.9545 | 2.258 | 0.0357 (11/308) | [0.0180, 0.0630] CP |
| 0.05 | APS | bipartite | 0.9478 | 0.9545 | 0.9545 | 1.757 | 0.0325 (10/308) | [0.0157, 0.0589] CP |
| 0.05 | RAPS | marginal | 0.9478 | 0.8734 | 0.7273 | 1.359 | 0.1071 (33/308) | [0.0638, 0.1529] boot |
| 0.05 | RAPS | class-conditional | 0.9491 | 0.9513 | 0.9545 | 2.355 | 0.0325 (10/308) | [0.0157, 0.0589] CP |
| 0.05 | RAPS | bipartite | 0.9550 | 0.9708 | 0.9545 | 1.851 | 0.0097 (3/308) | [0.0020, 0.0282] CP |

Intervals: `CP` = Clopper-Pearson exact (leads when the numerator is small, where the percentile bootstrap under-covers); `boot` = 2000x lesion-grouped percentile bootstrap (leads otherwise, because it is the one that accounts for multiple images of a lesion). Which one leads is chosen by count, not by which is narrower.

## The price of an FRR bound

No FRR target is pre-committed. The sweep below reports what each candidate bound would cost in set size, so the bound can be chosen against a visible price rather than asserted. RAPS + bipartite, alpha re-tuned at each level.

| alpha | FRR | FRR upper 95% | Mean set size | Singletons |
|---:|---:|---:|---:|---:|
| 0.20 | 0.1331 | 0.1834 | 1.036 | 74.0% |
| 0.15 | 0.1006 | 0.1438 | 1.152 | 70.5% |
| 0.10 | 0.0682 | 0.1023 | 1.285 | 65.0% |
| 0.05 | 0.0097 | 0.0282 | 1.851 | 22.1% |
| 0.02 | 0.0000 | 0.0119 | 3.263 | 0.4% |
| 0.01 | 0.0000 | 0.0119 | 3.326 | 2.3% |

| Candidate bound on FRR (upper 95%) | Cheapest alpha that clears it | Mean set size |
|---|---|---:|
| <= 0.10 | alpha = 0.05 | 1.851 |
| <= 0.05 | alpha = 0.05 | 1.851 |
| <= 0.02 | alpha = 0.02 | 3.263 |
| <= 0.01 | not achieved at any swept alpha | — |

## Cells and classes that could not certify a threshold

Reported rather than clipped: an infinite quantile is the honest statement that the data cannot support a narrower claim at this level. A cell marked `class-conditional backoff` no longer carries a band-conditional guarantee — it carries the weaker all-ages class-conditional one.

- alpha=0.10 LAC bipartite: needs n>=9 per cell; (unknown, escalating) n=2 -> class-conditional backoff.
- alpha=0.10 APS bipartite: needs n>=9 per cell; (unknown, escalating) n=2 -> class-conditional backoff.
- alpha=0.10 RAPS bipartite: needs n>=9 per cell; (unknown, escalating) n=2 -> class-conditional backoff.
- alpha=0.05 LAC bipartite: needs n>=19 per cell; (unknown, escalating) n=2 -> class-conditional backoff.
- alpha=0.05 APS bipartite: needs n>=19 per cell; (unknown, escalating) n=2 -> class-conditional backoff.
- alpha=0.05 RAPS bipartite: needs n>=19 per cell; (unknown, escalating) n=2 -> class-conditional backoff.

## Achieved coverage per cell — RAPS bipartite a10

The guarantee is conditional on the *true* label's cell, so these are the numbers the calibrator actually promises.

| Cell (age band, group) | n | Coverage | Mean set size |
|---|---:|---:|---:|
| <40, benign | 245 | 0.8980 | 1.282 |
| <40, escalating | 22 | 0.9545 | 1.455 |
| 40-59, benign | 637 | 0.8823 | 1.187 |
| 40-59, escalating | 60 | 0.9667 | 1.533 |
| 60+, benign | 332 | 0.9157 | 1.343 |
| 60+, escalating | 226 | 0.8850 | 1.403 |
| unknown, benign | 10 | 1.0000 | 1.100 |

## Exchangeability caveat — this variant is approximate, not exact

Split conformal's finite-sample guarantee requires calibration and evaluation scores to be exchangeable under *one fixed* score function. These quantiles come from out-of-fold scores produced by five different fold models, none of which is the full-train model that will score test at S9, so the guarantee does not transfer as a theorem. What OOF buys is calibration sample size — which is the binding constraint for a band-conditional cell, and the only reason the `(<40, escalating)` cell is certifiable at all. Achieved coverage is therefore audited empirically above and must never be asserted from the construction. CV+ / cross-conformal would restore a (1-2*alpha) guarantee but requires scoring the evaluation split with all five fold models — noted as the rigorous follow-up.
