# V4 findings — what is settled (final, 2026-09-17)

Status: final findings document for V4, closed by S63. S53r, S59, S60–S62 and the final audit
are folded in. Every number carries the `results/` file it comes from. No number here was
computed for this document: the two audit-derived artefacts in `results/v4/audit/` were written
by `research/v4/audit_v4.py --emit` from development data only. No test split and no new
reserved quantity were read to write it.

Reproducibility check at close-out: `python -m research.v4.audit_v4 --check` gives
**76 passed, 0 failed** (`--emit` adds 3 artefact writes). It re-derives every headline below with sklearn from the frozen
prediction, feature and threshold files.

---

## 0. Recommended framing

**The runbook's planned shape no longer fits the evidence.** `V4_SESSION_RUNBOOK.md` §6 planned
"a solutions paper with a falsification section", organised around four mechanisms:
group-conditional selection, a continuous age-conditional rule, explicit metadata and domain
routing. The verdicts do not support that shape:
- the continuous age rule was rejected (S57b);
- per-hospital λ was rejected (S66);
- the combined policy was rejected (S65);
- the domain gate and router failed their own checks (S58);
- every attempt to change the model's under-40 *ranking* came back null (S51, S54, S58, S67).

The one mechanism that cleared its endpoint, per-band abstention (S56), did so by moving
referrals between age bands, at a measured cost to the oldest band.

**Recommendation: a pre-registered diagnostic paper about limits.** Working title:
*"Decision layers cannot fix a ranking failure: a pre-registered study of the under-40 melanoma
blind spot in dermoscopy triage."* The contribution has three parts:

1. **A located failure.** Under-40 escalating lesions are ranked worse than those of older
   patients. This survives restricting to melanoma, and no backbone, training corpus, head
   objective, data weighting or metadata input tested here moves it (§1).
2. **A cost law for decision layers.** Every threshold-, calibration- or abstention-based
   intervention that raised under-40 sensitivity did so by referring more under-40 patients.
   When the comparison is held at equal workload, the gain disappears (§2).
3. **What does transfer.** Per-band calibration closes the direction disagreement it targets
   (§3), and a domain-fitted head recovers most of the general-accuracy loss across archives
   (§4). Neither touches under-40 ranking.

**Why this framing and not the planned one.** It is the only one in which every pre-registered
verdict appears as itself, rather than being reframed after the fact. It turns the negative
results into a constraint a reader can use: the fix for this subgroup has to come from the data
(more under-40 escalating lesions, better image acquisition) or from a different signal, not
from the decision layer. It also keeps the honest positive result (S56) with its cost attached.
A "solutions" framing would have to present S56's reallocation as a fix, which §2 shows it is
not.

---

## 1. The under-40 blind spot is a ranking limit

### 1.1 The size of the gap (development data, HAM OOF)

| age band | escalating images / lesions | AUC of the λ-rule score *d* [95% CI] | referral needed for sensitivity 0.80 (constant λ) |
|---|---:|---:|---:|
| <40 | 64 / 34 | **0.878** [0.811, 0.942] | 0.259 |
| 40-59 | 397 / 224 | 0.948 [0.929, 0.963] | 0.160 |
| 60+ | 893 / 559 | 0.929 [0.916, 0.941] | 0.368 |

Source: `results/v4/s64/ceiling.json` (`bands.*.auc_d`, `auc_d_ci`, `referral_for_sens.0.80.constant`).

- **No cutoff rule can beat this.** The in-sample per-bin oracle, which bounds every λ(age)
  curve, still needs 0.168 under-40 referral for sensitivity 0.80
  (`ceiling.json`, `referral_for_sens.0.80.oracle_bins`).
- **Ensembling does not help.** It adds +0.004 [−0.042, +0.052] under-40 AUC over the best
  single model (`ceiling.json`, `ensemble_under40.delta_raw_vs_best_single`).

### 1.2 The gap is partly, not wholly, case mix

Under 40, escalating lesions are almost all melanoma: 51 mel, 12 bcc and 1 akiec, against
440 / 275 / 178 at 60+. Melanoma is the hardest escalating class, so the comparison is repeated
on melanoma alone.

| age band | all escalating vs benign | melanoma vs benign | melanoma vs nevus |
|---|---:|---:|---:|
| <40 | 0.889 | **0.873** | 0.872 |
| 40-59 | 0.953 | 0.940 | 0.948 |
| 60+ | 0.933 | 0.909 | 0.932 |

Source: `results/v4/audit/under40_case_mix.json` (escalation-mass AUC on the S5 cross-fitted OOF
panel).

- **The age gap survives on melanoma alone.** Against 60+ it narrows from 0.044 to 0.036; against
  40-59 it is 0.067.
- **The manuscript should state both comparisons.** Quoting only the all-escalating AUC
  overstates the age effect.

### 1.3 Nothing tested moves it

All values are under-40 escalation pAUC@0.20 (McClish-standardised).

| intervention | session | contrast | Δ under-40 pAUC [95% CI] | source |
|---|---|---|---:|---|
| Foundation backbone (PanDerm ViT-B/16) | S51 | vs ConvNeXt-Tiny, frozen features, reserved | **−0.036** [−0.083, +0.013] | `results/v4/backbone_probe_deltas.csv` |
| Foundation backbone (DINOv2 ViT-B/14) | S51 | vs ConvNeXt-Tiny | **−0.073** [−0.118, −0.029] | same |
| In-domain pooled training + 384 px + balanced sampler | S54 | composite − pooled control, 3 seeds, `_last` | **+0.009** [−0.020, +0.037] | `results/v4/s54/s54_gate.json` |
| same, `_best` checkpoint rule | S54 | sensitivity analysis | −0.011 [−0.037, +0.016] | `results/v4/s54/s54_contrasts.csv` |
| Domain-routed heads | S58 | H2 − H0 (checkpoint head), reserved | **+0.007** [−0.027, +0.040] | `results/v4/s58/stage3_contrasts.csv` |
| Pooled refit head | S58 | H1 − H0 | +0.020 [−0.012, +0.054] | same |
| Under-40 specialist head | S67 | vs pooled control head, dev rows | **−0.039** [−0.108, +0.034]† | `results/v4/s67/probes.json` |
| Hard-case reweighting | S67 | same | **−0.000** [−0.013, +0.012]† | same |
| Sex, site and age as inputs | S67 | same | **+0.007** [−0.002, +0.017]† | same |

† S67 intervals are Bonferroni-adjusted (1 − 0.05/3), as its plan declared
(`results/v4/s67/stage1_plan.json`).

- **Pre-registered targets, all missed.** Each row needed +0.05 (S51, S54) or +0.03 (S67), with
  an interval excluding zero.
- **Absolute levels sit in one narrow band on reserved:** 0.72–0.75 for every V4 pooled model and
  0.7335 for deployed V1 (`results/v4/s54/s54_marginals.csv`); 0.711 for the frozen checkpoint
  head (`results/v4/s58/stage3_marginals.csv`).
- **S54 shows training on the target archives fixes general accuracy but not this ranking.**
  - Macro-F1 moves by **+0.18** (composite − V1, `_last`: +0.1798 [+0.1297, +0.2254]).
  - Under-40 pAUC does not move (+0.004 [−0.053, +0.062]).
  - Source: `results/v4/s54/s54_contrasts.csv`, Gate A. That gate is confounded by corpus and
    declared as such.

**Reading.** Five independent levers leave under-40 ranking where it was:
- representation (S51);
- training data (S54);
- head fitted per domain (S58);
- head objective and weighting (S67);
- decision threshold (S64).

The runbook stated in advance what a null across S64–S67 would mean (§6b): a ranking limit of
dermoscopy models trained on these archives, which no decision layer removes. That is now the
finding.

---

## 2. Decision layers reallocate referrals; they do not create sensitivity

All rows are on the reserved cohort (4,733 images; 279 under-40 escalating images from 104
lesions), with the frozen V1 ensemble.

### 2.1 S56 — per-band abstention: the positive result, with its cost

Contrast: per-band CRC-floor thresholds minus one global threshold, same score (`msp`), nominal
budget R = 0.20.

| quantity | band − global [95% CI] |
|---|---:|
| under-40 system escalation sensitivity (**primary**, MCID 0.10) | **+0.154** [+0.104, +0.216] |
| under-40 referral rate | +0.222 [+0.186, +0.255] |
| 40-59 system sensitivity | +0.017 [+0.006, +0.030] |
| 60+ system sensitivity | **−0.100** [−0.118, −0.082] |
| all-ages system sensitivity | −0.040 [−0.055, −0.025] |
| all-ages referral rate | −0.017 [−0.031, −0.004] |
| retained Macro-F1 | −0.022 [−0.046, +0.001] |

Source: `results/v4/s56/contrasts_reserved.csv` (budget 0.2, contrast `band-global`) and
`results/v4/s56/s56_report.json`.

- **Budgets do not transfer.** At a nominal 20% the band arm refers **38.6%** of reserved
  (`results/v4/s56/frontier_reserved.csv`).
- **Neither does the floor.** The nominal OOF floor (0.855 at R = 0.20) is met in **0 of 15**
  budget × band cells: **12** violated (interval entirely below the floor) and 3 below it with the
  interval still reaching it (`s56_report.json`, `floor_transfer`). CRC's guarantee is
  exchangeability-bound; off HAM it is nominal.

### 2.2 The rejected arms all fail the same way

| session | intervention | under-40 sensitivity gain [95% CI] | under-40 referral | under-40 specificity | verdict |
|---|---|---:|---:|---:|---|
| S57b | A3 kernel λ(age) vs frozen 3-band rule | **+0.208** [+0.127, +0.296] | 0.145 → 0.278 (×1.92) | 0.960 → 0.857 | REJECT-cost |
| S66 | per-hospital λ (P2) vs frozen rule, BCN | **+0.216** [+0.125, +0.313] | 0.160 → 0.303 (×1.90) | 0.965 → 0.859 | REJECT-cost |
| S65 | S55 + λ + S56 vs S56 alone, R = 0.20 | **+0.082** [+0.042, +0.131] | 0.454 → 0.608 | — | REJECT-cost |

Sources: `results/v4/lambda_verdict.json` (A3 gates 1, 2, 4) and `results/v4/s57b/ablation_reserved.csv`;
`results/v4/s66/s66_report.json` (gates 1, 2, 4) and `results/v4/s66/reserved_arms.csv`;
`results/v4/s65/s65_report.json` and `results/v4/s65/contrasts_reserved.csv`.

**Where the gain actually comes from:**
- **S66: it is the pooled under-40 λ, not per-centre fitting.** PC and P2 give the same BCN
  under-40 sensitivity, 0.6157 (`reserved_arms.csv`). BCN's under-40 NNB at π = 0.03 rises from
  3.80 to 8.38.
- **S65: at equal workload the gain vanishes.**
  - S56 run at the budget that matches the combined policy's flag rate (R = 0.305) catches as
    many: COMB − S56_matched = **+0.004** [−0.014, +0.025] (`s65_report.json`,
    `workload_matched`).
  - The combined policy passes its 60+ non-inferiority test against the λ rule alone
    (+0.211 [+0.179, +0.243]). That test is easy by construction, because the comparator refers
    nobody.
- **S57b: λ(age) carries structure but no efficient gain.** Age does carry structure beyond
  three steps (deviance diagnostic in `lambda_verdict.json`), but no curve turned it into a gain
  at equal cost.

### 2.3 The deployed rule already breaks its own floor off HAM

The frozen 40-59 λ = 0.74 gives reserved specificity **0.748** [0.718, 0.775] in that band, and
60+ gives 0.827, against a nominal floor of 0.85. Source: `results/v4/s57a/reserved_audit.json`.
This is a finding about the deployed rule, not grounds to relax the floor.

**Reading.** Across S56, S57b, S65 and S66 the pattern is the same. Any rule that raises
under-40 sensitivity refers more under-40 patients, at about 1.9× the referral in the λ arms.
When workload is held equal, as in S65, nothing is left. This is what §1 predicts: with ranking
fixed, a decision layer can only choose a different point on the same ROC curve.

---

## 3. S55 — per-band calibration closes the direction disagreement, not the magnitude one

HAM OOF, cross-fitted within band:

| map | signed gap <40 | signed gap 40-59 | signed gap 60+ | signed-gap spread | ECE spread across bands | aggregate ECE |
|---|---:|---:|---:|---:|---:|---:|
| uncalibrated | −0.229 | −0.190 | −0.119 | 0.110 | 0.109 | 0.172 |
| global Dirichlet | −0.027 | +0.005 | +0.041 | **0.068** | 0.015 | 0.025 |
| per-band Dirichlet | +0.007 | +0.008 | +0.015 | **0.008** | **0.021** | 0.019 |

Sources: `research/multical/results_oof/band_calibration_multical.csv` and
`research/multical/results_oof/gap_summary.json`.

- **The sign flip is gone.** After the global map, under-40 was under-confident and 60+
  over-confident; per-band calibration removes that disagreement.
- **Magnitude disagreement widens slightly.** Under-40 ECE improves most (0.028 → 0.010), which
  leaves 60+ (0.030) as the clearer worst-calibrated band.
- **Calibration does not change ranking** (S64: +0.007 AUC from the calibrator,
  `ceiling.json`, `ensemble_under40.delta_cal_vs_raw`).
- **It does not compose with the frozen λ.** In S65 the per-band map cut under-40 λ-rule
  sensitivity on reserved from 0.409 to 0.262 (`results/v4/s65/frontier_reserved.csv`, arms
  `LAM` and `COMB_none`), because the rule was fit under the global map.

---

## 4. S58 — a domain-fitted head recovers general accuracy, and nothing else of what was asked

Reserved cohort, frozen HAM-only ConvNeXt-Tiny features:

| head | Macro-F1 | under-40 pAUC |
|---|---:|---:|
| H0 checkpoint layer | 0.419 | 0.711 |
| H1 pooled refit | **0.534** | 0.732 |
| H2 domain-routed | 0.526 | 0.718 |
| V4 pooled control (retrained trunk, 3 seeds, `_last`) | 0.588 / 0.609 / 0.611 | 0.720 / 0.733 / 0.733 |

Source: `results/v4/s58/stage3_marginals.csv`.

- **Primary SUPPORTED.** H2 − H0 Macro-F1 is **+0.107** [+0.075, +0.136], Holm p 0.002
  (`stage3_contrasts.csv`).
- **But routing adds nothing over one pooled head.** H2 − H1 is −0.008 [−0.024, +0.007], so
  stage 3 is the pooled head.
- **A refit head gets about 63% of the way to a retrained trunk.** The gap from 0.419 to roughly
  0.60 is mostly closed without touching the trunk.
- **The front-end checks fail** (`results/v4/s58/s58_report.json`):
  - the out-of-scope gate rejects only **0.378** of PAD-UFES-20 images;
  - the router's three-way accuracy is **0.932**.

---

## 5. The recipe ladder (S53r)

S53's ladder is superseded (§6.3, §6.4). S53r re-ran it with the colour step fixed, early
stopping off and three seeds per arm; plan `results/v4/s53r_plan.json` (sha256
`2ef2e91bf5db6df5…`), 21/21 runs banked.

Final-epoch HAM val Macro-F1 minus R0, paired by seed (`results/v4/s53r/s53r_report.json`):

| rung | change | mean Δ | seed range | reading (MCID 0.020) |
|---|---|---:|---|---|
| R1 | 224 → 384 px | **+0.0297** | +0.0097 … +0.0466 | **LEVER** |
| R4 | class-balanced sampler | +0.0134 | +0.0026 … +0.0295 | NULL |
| R7 | metadata branch | +0.0058 | −0.0133 … +0.0373 | NULL |
| R2 | shades-of-grey (fixed) | −0.0045 | −0.0224 … +0.0298 | NULL |
| R5 | Mixup/CutMix + RandAugment | −0.0112 | −0.0249 … +0.0095 | NULL |
| R6 | EMA + 60 epochs | −0.0139 | −0.0402 … +0.0154 | NULL |

- The control's own seed SD is **0.023**, the size of the MCID. R1's lowest seed (+0.0097)
  sits below it, so "LEVER" means "mean clears the bar and every seed is positive", not a
  certified effect.
- R2's earlier −0.102 was the colour bug. Fixed, colour constancy is a null.
- R7's age-flip span is 0.034–0.038 across seeds (`s53r_ladder.csv`), below its 0.05 mechanism
  MCID in every seed.
- This is a compute-allocation screen on HAM val and cannot change §1–§4. R1's resolution effect
  was already inside S54's pooled composite (R1+R4), which did not move under-40 pAUC on reserved.

---

## 6. Caveats the manuscript must carry

### 6.1 The reserved cohort has been read eight times

Receipts: `results/v4/{s54,s56,s57b,s58,s59,s65,s66}/reserved_receipt.json` and
`results/v4/s57a/reserved_audit_receipt.json`. Each records one completed execution per stage
and no rerun reason. S51's cross-fitted backbone probe also used reserved labels, before S54.

- **Each read answered a pre-registered question** under a plan frozen beforehand. Plan hashes
  match, and every plan's timestamp precedes its read.
- **But which session ran next depended on earlier results.** Reserved is therefore a reused
  evaluation set, not a pristine holdout, and multiplicity across sessions is uncorrected.
- **The risk falls on the positives:** S56's primary and S58's routed-vs-checkpoint contrast. The
  nulls in §1–§2 are, if anything, conservative. The HAM test split is untouched since S9
  (`results/test_pass_receipt.json`, `n_executions: 2`).

### 6.2 Several gates were hard to pass by construction

- **S57b and S66's 1.5× under-40 referral-blowout gate** fires for almost any λ increase that adds
  sensitivity. S66 declared in its plan that it expected the gate to fire.
- **S51's falsifier needed +0.05 with an interval excluding zero,** at 104 lesions where the
  marginal pAUC intervals are about ±0.07 wide (`results/v4/backbone_probe_marginals.csv`).
- These results are *failures to clear a demanding bar*. §2's workload-matched S65 contrast is
  the cleaner evidence that the gains are bought, not earned.

### 6.3 The colour-constancy step was wrong in S49 and S53

The original `research/v4/colour.py` scaled each channel by `‖e‖/e_c` instead of
`‖e‖/(√3·e_c)`. Measured on 60 training images (`results/v4/audit/colour_check.json`):

| | mean brightness | pixels clipped (median) |
|---|---:|---:|
| input | 149.3 | 0.000 |
| old step (used by S49, S53 R2) | **218.0** | **0.787** |
| fixed step | 149.3 | 0.000 |

**Consequences:**
- S53's R2 result (−0.102 val Macro-F1) and S49's "colour constancy reduces archive
  decodability" are **withdrawn**.
- The outputs were renamed `results/v4/archive_probe_pre_post.superseded_colour_bug.*`.
- **Both were re-run in S53r.** R2 is a null (§5). The S49 probe with the fixed step
  (`results/v4/archive_probe_pre_post.json`) gives HAM val vs BCN decodability
  0.9905 [0.987, 0.993] → **0.9716** [0.964, 0.978], Δ **−0.019**. The intervals do not
  overlap, but the change is below the probe's pre-existing −0.02 materiality rule, so the verdict
  is now *not material*. The honest reading: illuminant statistics carry a small, separable part
  of archive identity, and almost all of it (0.97) survives colour normalisation.
- **Nothing in §1–§4 used colour constancy.**

### 6.4 Early stopping truncated the S53 ladder and the S54 models

Patience 8 on a cosine schedule stopped runs while the learning rate was still high
(`results/v4/recipe_runs/*.json`, `epochs_run` and `history`):
- R6 (EMA + 60 epochs) stopped at 19/60;
- the pooled S54 runs stopped at 14–22/30.

What this does and does not affect:
- **S54's Gate B remains a valid comparison of the two procedures as run,** but "in-domain
  training does not fix under-40 ranking" is conditional on that truncated recipe.
- **S54's Macro-F1 contrast depends on the checkpoint rule:** −0.012 under `_last` and +0.032
  under `_best` (`s54_gate.json`, `checkpoint_sensitivity`). The manuscript reports both and
  claims neither.

### 6.5 S51's control used a different resize

S51's ConvNeXt control was extracted with a bicubic resize (S58's transform note). Under the
deployed bilinear transform the control's under-40 pAUC is 0.711 (`stage3_marginals.csv`, H0),
not 0.732. The PanDerm contrast becomes about −0.015 rather than −0.036. That is still far from
the +0.05 falsifier, so the S51 conclusion stands.

---

## 7. The deployed system and its contract (S59)

- **Stack:** frozen V1 + argmax + S56 per-band abstention at R = 0.20
  (`results/v4/s59/s59_report.json`, plan `e7ebc3f5…`).
  - The λ layer loses at matched workload: −0.018 [−0.025, −0.011] all ages, 0.000 under 40.
  - The conformal layer loses too: −0.069 [−0.086, −0.052] all ages.
- **Contract on reserved: CONTRACT_FAILS, every term NOT_MET**, as declared before the read.

  | term | value [95% CI] | target |
  |---|---|---|
  | `<40` system sensitivity | 0.763 [0.655, 0.853] | ≥ 0.855 |
  | `40-59` system sensitivity | 0.704 [0.631, 0.771] | ≥ 0.855 |
  | `60+` system sensitivity | 0.674 [0.638, 0.713] | ≥ 0.855 |
  | retained Macro-F1 | 0.465 [0.401, 0.516] | ≥ 0.888 |
  | referral | 0.386 [0.363, 0.408] | ≤ 0.25 |
  | NNB(π = 0.03) | 13.6 | ≤ 7.9 |

- **Selective coverage** (share decided without referral) is **0.614**. It is not a joint pass
  probability.
- **Against V1 at full coverage**, the stack adds +0.301 [+0.273, +0.328] system sensitivity
  overall and +0.423 [+0.310, +0.539] under 40, at +0.386 referral
  (`s59/contrasts_reserved.csv`).
- **Why V1 is still the base.** S56's thresholds are fit on cross-fitted OOF predictions, and no
  V4 model has them (`s59_plan.json:base`). It is not because V4 lost to V1: the V4 pooled models
  score about +0.18 Macro-F1 over V1 on reserved (S54 Gate A, confounded). A V4-based stack is
  untested.

## 8. Close-out (S63)

- **HAM test read: not spent.** S63's gate was that the cascade clears its per-band floor on
  reserved, and it did not. `results/test_pass_receipt.json` stays at `n_executions: 2`
  (`results/v4/final_verdict_v4.json`).
- **Not run, by design or by owner's choice:**
  - S57c (gated on S57b, which returned REJECT-cost);
  - S67 stage 2 (gated on stage 1, STAGE2_NO_GO);
  - S67 stage 0 (a descriptive reserved read, left to the owner);
  - the optional S66 V4 arm.
- **Decisions this leaves for a later version:**
  - a V4 base with cross-fitted predictions (GPU K-fold);
  - target-side threshold recalibration on BCN/MSKCC train rows (S56/S59);
  - a HAM-only or modality-classifier gate for smartphone images (S58);
  - more under-40 escalating data, which is the only lever §1 leaves open.
