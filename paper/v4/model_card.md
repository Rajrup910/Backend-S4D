# Model Card — Multimodal Skin Lesion Triage System (V4, draft)

**Status: complete for V4 (S61).** Written 2026-09-17 after the S53r re-run, the S59 freeze
(`results/v4/s59_plan.json`, sha256 `e7ebc3f5…`) and the S62 index
(`results/v4/analysis_plan_v4.json`, sha256 `bd320319…`, registered in
`results/frozen_artifacts.json`). Every number below is either (a) a frozen V1 artifact already
shipped in the manuscript, or (b) an OOF/reserved-cohort number from a closed V4 session
(S51–S67), cited to its file. The HAM test split has not been re-read (receipt still at 2).
S63's test-read gate ("the composed cascade clears its per-band floor on reserved") did **not**
open (§4.6), so no V4 number here comes from HAM test. This card follows CLAIM 2024 (Tejani et al., *Radiology: AI* 2024) and TRIPOD+AI
(Collins et al. 2024) as its structural backbone, per the V4 runbook §5 (S61).

---

## 1. Model details

| | |
|---|---|
| Task | 7-class skin lesion classification (HAM10000 taxonomy) + age-band-conditional escalation triage |
| Deployed model (frozen, shipped) | 6-CNN uniform soft-vote ensemble (ConvNeXt-Tiny/Small, EfficientNet-B0/B3, ResNet-50, DenseNet-121), 24-view TTA, global Dirichlet calibration (`research/selective/results_oof/fit_state.json`) — referred to below as **V1** |
| Candidate replacement | V4 in-domain-retrained ConvNeXt-Tiny recipe ladder (`research/v4/recipe.py`) — **not adopted**. On reserved, the V4 pooled models beat V1 by about +0.18 Macro-F1 (S54 Gate A, confounded by corpus) but not on under-40 pAUC (+0.004). The composite does not beat the V4 pooled control (Gate B, checkpoint-rule-dependent: report `_last` and `_best`, never `_best` alone). V1 stays the base because S56's thresholds are fit on cross-fitted OOF predictions, which no V4 model has (`s59_plan.json:base`) |
| Decision layers on top of the base model | (1) global Dirichlet calibration; (2) argmax classification; (3) S56 per-band CRC-floor abstention (deployed, R=0.20, floor S\*=0.855 nominal). S59 froze this stack (`S56`) as the deployed system. (4) The S6/S9 bipartite RAPS(α=0.05) conformal set is reported with each decision as information only. S59 declined it as a referral layer (at matched workload it lowers system sensitivity, −0.069 all ages) |
| Layers evaluated and **rejected** | frozen 3-band λ stacked on S56 (S59: at matched workload −0.018 all-ages system sensitivity, 0.000 under 40); per-band Dirichlet calibration stacked with abstention (S65: undoes the frozen λ); per-band λ(age) (S57b: fails Gate 5, band 0.37 vs distance 0.10); per-hospital λ (S66: gain is the pooled-λ effect, not per-centre fitting — REJECT-cost); domain admissibility gate + router (S58: PAD reject rate 0.378, rejects escalating lesions *more* often) |
| Model card owner | This repository; no external deployment yet |
| Version | v0.1.0-draft (tracks `research/v4/s60_api.py` version string) |

---

## 2. Intended use

- **Intended**: a decision-support triage aid over dermoscopy/clinical images already
  captured, producing a class, an uncertainty-conditioned referral recommendation, and a
  conformal shortlist — for a clinician to review, not to replace.
- **Not intended**: autonomous diagnosis, use as the sole basis for a treatment decision,
  or use on patients whose skin type is Fitzpatrick V/VI (see §5 — no evidence exists).
- **Population the model was developed on**: HAM10000 (dermoscopy, Austria) as the primary
  training/calibration corpus; PAD-UFES-20 (smartphone clinical, Brazil) and three external
  archives (BCN-20000, MSKCC via ISIC, PAD) as out-of-distribution evaluation cohorts, never
  used for fitting any deployed parameter.

---

## 3. Training and evaluation data

| Split | N | Role |
|---|---:|---|
| HAM10000 train (lesion-grouped) | 6,981 images / 5,229 lesions (OOF, cross-fitted) | Fits calibration, abstention thresholds, λ(age) |
| HAM10000 val | ~1,532 images | Score/method selection only (never a fitting split) |
| HAM10000 test | fixed, read once (`results/test_pass_receipt.json`, `n_executions: 2`) | S9's 19 pre-registered quantities; not re-read for V4 |
| Reserved cohort (BCN+MSKCC, V4-era) | 4,733 images, 0 HAM, 104 under-40 escalating lesions | V4's primary held-out endpoint; read 8 times across S54–S67 (S59 included), each receipted once — **treat as a reused evaluation set**, not a pristine holdout |
| PAD-UFES-20 | 2,106 images | Cross-domain probe only (S8b); never a fitting split |

All splits are lesion-grouped (Hard Rule 1); accuracy is never the selection criterion
(Hard Rule 3); every number here traces to a `results/` file (Hard Rule 4).

---

## 4. Performance

### 4.1 In-distribution (HAM10000, frozen V1, test — S9, `results/session9/`)
- Macro-F1 **0.7871** (soft-vote + 24-view TTA + deployed OOF Dirichlet map, rung A7-oof)
- Escalation sensitivity, all ages: **0.731** (argmax) → **0.831** with the frozen age-λ rule (referral 0.178 → 0.268)
- Escalation sensitivity, **under-40 only**: **0.143** (argmax) → **0.238 [0.082, 0.472]** with the λ rule —
  *the abstract-level headline (0.831) must never be quoted without this under-40 figure alongside it (S19 finding)*

### 4.2 V4 in-domain retraining (S53r on HAM val; S54 on reserved — Gate B composite vs V4 pooled control, Gate A vs deployed V1)
- Recipe ladder, re-run with the colour-constancy fix and 3 seeds (S53r, `results/v4/s53r/s53r_report.json`,
  HAM-only, val Macro-F1 paired against R0):
  **R1 (384 px) +0.0297 (seed range +0.010 to +0.047) is the only lever.** R2 colour constancy
  −0.0045 (−0.022 to +0.030), R4 +0.0134, R5 −0.0112, R6 −0.0139 and R7 +0.0058 all read NULL.
  R2's earlier −0.102 was the √3 bug, and R2 is now a null result, not a failure.
  The control's own seed SD is 0.023.
- On the reserved cohort, Gate B (`_last`): Macro-F1 **−0.0121 [−0.0321, +0.0098]**; under-40 pAUC **+0.0089 [−0.0200, +0.0367]** — **flat, not improved**
- Gate A (confounded, deployment delta): V4 composite − V1 Macro-F1 **+0.180 [+0.130, +0.225]**, under-40 pAUC **+0.004 [−0.052, +0.062]**
- Verdict: in-domain training lifts general accuracy but not under-40 ranking (S54 outcome 4 on Gate B; Gate A's flat pAUC; corroborated by S51 and S67)

### 4.3 Under-40 ranking ceiling (S51, S64, S67 — the closed question)
- Under-40 escalation pAUC@0.20 sits at **~0.66–0.75** across every representation and
  training regime tried (control ConvNeXt, PanDerm ViT-B/16, DINOv2 ViT-B/14, in-domain
  retraining, specialist/reweighted/metadata heads). On HAM OOF the under-40 AUC is **0.878**
  against **0.948** (40-59) and **0.929** (60+) (`results/v4/s64/ceiling.json`); restricted to
  melanoma vs benign it is 0.873 vs 0.940 / 0.909 (`results/v4/audit/under40_case_mix.json`). No intervention tested closes this gap. **This is reported as a measured
  ceiling, not a solved problem.**

### 4.4 Selective abstention (S56, deployed; `results/v4/s56/s56_report.json`)
- Primary endpoint SUPPORTED: under-40 system escalation sensitivity, band-conditional
  arm minus global arm, at R=0.20: **+0.154 [+0.104, +0.216]** (MCID 0.10)
- Cost: under-40 referral rises 0.232→0.454; 60+ sensitivity falls −0.100; all-ages falls −0.040
- ⚠️ **Nominal floors do not transfer**: on the reserved cohort, 0 of 15 budget×band floor
  cells were met. 12 were violated, and in the other 3 the point estimate is below the floor
  while the CI still reaches it (`s56_report.json:floor_transfer`). The nominal 20% budget
  realised ~39–40% referral there.

### 4.6 The deployed system's contract, measured end-to-end (S59, `results/v4/s59/s59_report.json`)
Stack **S56@0.20** on V1, measured on the reserved cohort (4,733 images, 104 under-40 escalating
lesions) with lesion-grouped intervals. This is a joint measurement, not a union bound.
The plan declared `CONTRACT_FAILS` as the expected outcome before the pass, and it failed.

| Term | Reserved | Target | Status |
|---|---:|---:|---|
| system sensitivity `<40` | 0.763 [0.655, 0.853] | ≥ 0.855 | not met |
| system sensitivity `40-59` | 0.704 [0.631, 0.771] | ≥ 0.855 | not met |
| system sensitivity `60+` | 0.674 [0.638, 0.713] | ≥ 0.855 | not met |
| retained Macro-F1 | 0.465 [0.401, 0.516] | ≥ 0.888 | not met |
| referral rate | 0.386 [0.363, 0.408] | ≤ 0.250 | not met |
| NNB (π=0.03) | 13.6 | ≤ 7.9 | not met |

The system decides 61.4% of reserved cases without referral (`selective_coverage` = 1 − referral
rate); every term's interval misses its target, so no bootstrap resample satisfies the contract.
Against V1
with no abstention, the deployed stack raises system sensitivity by **+0.301** overall and
**+0.423** under 40, but it refers **+0.386** more cases. The contract is therefore valid only for
HAM-like dermoscopy (the OOF figures in §8). **It does not transfer to BCN/MSKCC.**

### 4.5 Cross-domain / external (S8b, S13, S14)
- PAD-UFES-20: ensemble Macro-F1 **0.167**, below its own best single member (0.188) —
  correlated cross-domain errors erase the ensembling benefit
- Three-centre age-rule dose–response (S14): the pre-registered ordering **reversed**
  (HAM 0.547 > MSKCC 0.333 > BCN 0.279 sensitivity, opposite the predicted skew-driven order);
  the frozen λ still lifts sensitivity zero-shot in all three centres (+0.02–0.03 referral cost)

---

## 5. Fairness and subgroup performance — **mandatory disclosure**

**Fitzpatrick skin type.** Evaluated only on PAD-UFES-20 (n=1,302/2,106 labelled). Restricted
to the powered types I–IV, the true spread is **0.102 and non-monotonic**
(I 0.349, II 0.287, III 0.247, IV 0.278; n=59 at type IV). Types **V (n=8) and VI (n=1) are
suppressed** below the project's minimum group size (30) and **carry no result of any kind**.

> **This system has NO EVIDENCE for darker skin tones (Fitzpatrick V/VI).** It must not be
> presented, marketed, or deployed as validated for those skin types. This statement is
> reproduced verbatim in every `/predict` and `/contract` response of the S60 service
> (`research/v4/s60_contract.py:FITZPATRICK_NOTICE`) and must not be removed or softened by
> any downstream integration.

**Age.** The under-40 band is simultaneously the easiest-looking (highest raw accuracy, due
to a benign-mole-dominated case mix — 4.9% escalating prevalence vs 35.5% in 60+) and the
worst on every ranking and sensitivity metric measured. This is the system's most material
known failure mode; see §4.3.

**Intersectional (age × sex, S7/S9).** 5 of 8 cells usable; `<40 × male` suppressed
(9 escalating cases, below the gate of 10); `<40 × female` escalation sensitivity **0.250**.
TPR gap **0.598**.

---

## 6. Limitations (carried forward, not re-litigated)

1. Under-40 escalation ranking is a measured ceiling (§4.3) — five independent interventions
   failed to close it (S51, S54, S58, S64, S67).
2. Reserved cohort has been read 8 times across the V4 programme; treat its numbers as a
   reused evaluation set, not a pristine external test (V4 audit, 2026-09-17).
3. Two real defects were found and fixed in V4 (audit, 2026-09-17): a colour-constancy bug
   (`shades_of_grey` over-brightened by √3) invalidated R2/S49 — **re-run in S53r: R2 is NULL (−0.0045)**;
   early stopping (patience 8) truncated several high-LR cosine runs. S53r re-ran the HAM-only
   ladder without it; the pooled S54 models were not re-run, so S54 stays conditional on it.
4. No PAD-UFES-20 checkpoints or training histories survive from the original 12 PAD runs —
   any PAD number requiring a checkpoint must be retrained from scratch.
5. Nominal abstention floors and λ(age) operating points are OOF-fitted and do **not**
   transfer cleanly to the reserved or external cohorts (§4.4, §4.5) — quote them as nominal.
6. `_last` vs `_best` checkpoint choice can move a reported Macro-F1 delta by up to 0.057
   (S54) — a result reported from one checkpoint rule without stating the other is incomplete.

---

## 7. Reporting checklists

- **CLAIM 2024** (44 items): full crosswalk at `results/CLAIM_checklist.md` — **33 met /
  7 partial / 2 not met / 2 N/A** (V1; not yet re-run against V4's additions). Not
  registered as a clinical study; a frozen pre-registered analysis plan
  (`results/analysis_plan.json`, `results/v4/s56_plan.json`, etc., each sha256-hashed) is
  used as an imperfect substitute.
- **TRIPOD+AI**: 23-domain crosswalk in `paper/manuscript.tex` Appendix (transportability
  domain corresponds directly to §4.5's external-cohort results above).
- **V4-specific pre-registration**: `results/v4/s56_plan.json` (abstention),
  `results/v4/s57b_plan.json` (λ(age) freeze decision), `results/v4/s66_plan.json`, `results/v4/s65_plan.json`
  and `results/v4/s59_plan.json` (the composed system). Each plan was sha256-hashed when
  frozen. S62 indexes all eleven V4 gates (plan hash, receipt, verdict read from each report) in
  `results/v4/analysis_plan_v4.json` (`research/v4/s62_freeze.py --check`), registered under
  `analysis_plan_v4` in `results/frozen_artifacts.json`. It also declares their multiplicity rules
  as `v4_*` families in `research/stats/families.py`. That index was written after the reserved
  reads it lists, and it says so.

---

## 8. Deployment contract (S60)

Every live decision from `research/v4/s60_api.py` returns, alongside the predicted class:
the conformal set (bipartite RAPS α=0.05), the band-conditional abstention decision and the
threshold it was compared against, the age used (recorded or explicitly `"unknown"` — **this
service does not estimate age from the image**; S57a found the age estimator has a +19.5-year
bias under 40 and needs backbone features the stub does not compute), the λ(age) value
(informational only — not stacked into the decision, see §1), an honestly-`null` OOD score and
Grad-CAM overlay (both require the real model, not the stub), the frozen policy's plan hash,
and the fairness/under-40 notices from §5/§4.3 verbatim. The service also carries the S59 plan
hash and the reserved-cohort verdict (`external_contract.joint = CONTRACT_FAILS`, §4.6). It
refuses to start if S59's report names any stack other than the S56@0.20 it implements.

**In-distribution contract** (HAM OOF, band arm, R=0.20; `results/v4/s56/frontier_oof_insample.csv`):
system sensitivity 0.858 overall, 0.875 `<40`, 0.859 `40-59`, 0.857 `60+`. Referral is 0.200
overall and 0.317 `<40`. NNB at π=0.03 is 7.5. These figures describe HAM-like dermoscopy only.

---

## 9. Monitoring and drift hooks (S61)

Four hooks, all against **already-built instruments** — no new measurement machinery, only
their wiring into a monitoring loop:

| Hook | Instrument | Signal | Alerting logic |
|---|---|---|---|
| **Archive-probe drift** | `research.v3.probes.probe_archive` over `research.selective.features.extract_features` (used in S42/S43/S49 to show an embedding can tell dermoscopy archives apart) | Cross-fitted AUC of "which archive did this batch of incoming images come from" vs. the training archive | AUC rising materially above chance on a rolling window of incoming images means the input distribution has drifted from HAM/reserved; **not** itself a performance signal, an early-warning one |
| **Mahalanobis OOD drift** | `research/selective/mahalanobis.py` (`fit`/`score`), fit on HAM train ConvNeXt-Tiny features as in `research/xdomain/run_session8b.py:mahalanobis_shift` (S8b: AUROC 0.913 discriminating HAM val from PAD, 18× median-score separation) | Rolling median/95th-percentile Mahalanobis distance of incoming feature vectors against the frozen HAM-train Gaussian | Sustained shift in the rolling median toward the PAD-scale separation (baseline ~485 vs. PAD ~8,717 median score, S8b) flags a domain shift serious enough to distrust calibration |
| **Per-band coverage drift** | `research/multical/groupwise.py` (S55) + `research/conformal/hierarchical.py`'s cell coverage check | Realised empirical coverage of the deployed bipartite conformal set, recomputed per age band on a rolling window of cases with eventual ground truth (biopsy/follow-up) | This is **the highest-priority hook**: HG-CRC's central result is that group-composition shift breaks marginal coverage guarantees silently, and the reserved cohort already demonstrated this in-project (S56: 0/15 nominal floors held). Reference values for a shifted population are in S59: on reserved, bipartite set coverage is 0.854 (`<40`), 0.825 (`40-59`) and 0.779 (`60+`) against a nominal 0.95. A coverage drop in any band is the earliest warning, before outcome data catches up |
| **λ(age)-curve drift** | `research/v4/lambda_age.py` diagnostic (S57a) — the fitted λ(age) shape vs. the deployment population's age histogram | KL-divergence or simple histogram distance between the age distribution the frozen λ was fit on (HAM OOF) and the rolling age distribution of incoming cases | A shifting case-mix age distribution can silently move the effective operating point even though no parameter changed; flag when deployment age histogram diverges materially from the HAM OOF one used to fit the 3-band rule |

**Not yet implemented:** none of these four is wired into a running monitor. This section
specifies what each hook measures and against which frozen baseline, not a scheduled job.
Implementation should live beside `research/v4/s60_api.py` as a separate low-frequency batch
script, not inside the request path.

---

## 10. Contact / provenance

Generated as part of the V4 session sequence (`V4_SESSION_RUNBOOK.md` §5, S61). Every
numeric claim above is sourced to a file under `results/` or `research/`; see
`CHANGELOG.md` §S51–§S67 for the full narrative and `research/experiments.csv` for the
run ledger. The only HAM test numbers in this card come from S9's receipted pass
(`n_executions: 2`); every reserved-cohort number comes from a pre-registered V4 read (receipted from S54 on; S51's
probe predates receipts).
