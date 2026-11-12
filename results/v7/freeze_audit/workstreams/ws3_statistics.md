# WS3: promotion statistics and interval coverage (read-only audit, 11 Nov 2026)

Scope: `research/v7/promotion.py`, `research/v7/stats.py`, `research/v6/system_reads.py`, `research/v6/gate.py`,
`scripts/v7_interval_coverage.py`, `scripts/v7_p2_coverage.py` (lead's), `scripts/simulate_pauc_power.py`, the JSONs they write,
`tests/test_v7_decisions.py`, plan sections 7, 10.4, 10.7, 13.1 and the statistical preregistration doc.
No file other than this report was written. Scratch scripts only under the session scratchpad (`.../scratchpad/ws3/`).
No labels, MIDAS, HAM test or HIBA outcomes were read. No V7 prediction was generated. pytest was NOT run (economy).

## 1. Evidence inspected

Code read in full: promotion.py (200 lines), stats.py (109), system_reads.py (427), gate.py (143), v7_interval_coverage.py (143),
v7_p2_coverage.py (190), simulate_pauc_power.py (206), tests/test_v7_decisions.py (promotion block).
Data/JSON: `results/v6/anchor_noise_floor.json`, `results/v6/e_primary.json` (V6 vs V5 read, already banked),
`results/v7/interval_coverage.json` (+ `_95only_run1.json`), `results/v7/p2_coverage.json`, `results/v7/pauc_power_simulation.json`,
`results/v7/denominators.json`, `ml/data/v7/master_v7.csv` (id/split/link columns only).
Reproductions run (one python process at a time): fail-open probes, pAUC equivalence, arithmetic on SDs, a 1,200-trial P2 pAUC
sim of my own (150 trials x 8 scenarios, 250 replicates; superseded by the lead's 1,000-trial file, used only as a cross-check:
P2-pAUC pass 0.93/0.81/0.43/0.01 at rho 0.85 for shift 0/-0.25/-0.5/-1.0, and 0.93/0.56/0.06/0.00 at rho 0.95; agrees with the lead's file within MC error).

## 2. What the code implements (formulas)

Notation: V = V7 candidate, R = banked V6 vote, rows = folds 1-4 images (12,235 for all rows), bootstrap unit = `group_id`
(lesion connected component), images are the scored unit (no lesion-level averaging).

Escalating = {akiec, bcc, mel} = indices (0,1,4) in `head8.CLASSES_7` (verified; `promotion.ESC` and `system_reads.ESCALATING` agree).
Escalation score s(x) = p_akiec + p_bcc + p_mel (sum of the 7-class vote).
pAUC = McClish-standardised partial AUC, FPR <= 0.20, on the 0.5 (chance) to 1.0 (perfect) scale:
`0.5 * (1 + (raw - 0.02) / 0.18)`. P1/P2 reach it through `research.v2.frontier.partial_auc`; P3 and the simulations use
`sklearn.roc_auc_score(max_fpr=0.20)`. I confirmed the two are identical (max difference 0 over 200 tie-heavy draws). Docs agree
(plan 7.1 formula; +0.050 McClish = +0.018 raw). Doc wording "Melanoma-Specific Ranking" (plan 7.1 item 1) disagrees with the code, which is escalation (3 classes).

Delta direction everywhere: candidate minus reference (V - R; P3: V - L0 baseline). Positive = candidate better.

| Part | Estimand | Pass condition as coded | Strict? |
|---|---|---|---|
| P1 | D_pauc = pAUC_all(V) - pAUC_all(R), D_f1 = MacroF1_7(V) - MacroF1_7(R) (argmax, all 7 labels, zero_division=0) | `D_pauc > 0.008237` (POINT estimate, no interval) AND `D_f1 > 0` AND `lo95(D_f1) > 0` | all strict `>` |
| P2 | D_u = pAUC_u40(V) - pAUC_u40(R) on under-40 rows (unknown age counted as not under-40); D_s = ALL-AGE argmax escalation sensitivity delta (`_esc_sens` on every row, not on the under-40 band) | `hi95(D_u) >= 0` AND `hi95(D_s) >= 0`, hi95 = 97.5th percentile of 2,000 paired lesion-group resamples | non-strict `>=` |
| P3 (per archive) | delta of present-class Macro-F1 (classes with >= 50 lesion groups in the archive, from ground truth only, shared by both systems) and of escalation pAUC (all negatives), candidate vs L0 baseline, 2,000 resamples, seed 20261110 | `lo95(delta) > -0.02` on every read that is estimable, and at least one decisive read estimable | strict `>` |
| Gate A2 (secondary) | D_u | `D_u >= 0.050` AND `1.25th percentile of D_u > 0` (97.5% interval) | `>=` and `>` |

Percentile levels: 95% reads = (2.5, 97.5); under-40 gain claims (Gate A2, xAUC, E-histo under 40) = (1.25, 98.75). Bootstrap: seed 20261103
(sr.paired_bootstrap, every call, so all candidates and all row sets share the same random stream), 2,000 replicates; P3 seed 20261110.

Doc/code disagreements found:
1. Statistical preregistration doc section 5.1 and plan 7.1/7.2 specify a Holm family H1A (D_pauc >= +0.050), H1B (Macro-F1 NI margin -0.015), H1C (age 40+ sensitivity NI margin -0.010). Plan 10.4 and the code implement none of these (bar 0.0082, Macro-F1 lower bound > 0, P2 under-40 pAUC + all-age sensitivity, intersection-union, no alpha correction). Two incompatible promotion rules coexist in the plan.
2. Plan 10.4 says "all three (P1, P2, P3) on four row sets"; `verdict()` reads P3 once per archive with no row-set axis.
3. Plan 10.4 / promotion.py docstring: "under-40 pAUC delta and the escalation-sensitivity delta": ambiguous; code reads all-age sensitivity. Plan 329 says "under-40 escalation pAUC and escalation sensitivity".
4. Stat doc 2.2 says predictions are pre-aggregated to lesion level by arithmetic mean and patient clusters are used where verified; code scores image rows and clusters by `group_id`.
5. Stat doc 4.3 cites "bootstrap power of 81.2%" from `pauc_power_simulation.json`; that file contains no such number (see F10).
6. Plan 10.7 cites `pauc_power_simulation.json` for "at a true +0.020 the gate passes about 5%"; the file does not contain a +0.020 entry (see F10).

## 3. Findings

| ID | Finding | Sev | Conf. | Evidence | Recommended correction | Files | Tests needed |
|---|---|---|---|---|---|---|---|
| F1 | `verdict()` is fail-open on P3. `cross_site={}` gives `p3={}`, `all([])` is True, and `cross_site is None` is the only guard, so `{}` PROMOTES. No set of required holdouts (BCN, HAM, MSKCC) is enforced: a single passing archive promotes. Repro below. | BLOCKER | High (reproduced) | promotion.py:180, 186-189; test only covers `None` (test_v7_decisions.py:134-142) | Add `REQUIRED_HOLDOUTS` (owner-confirmed set; plan 13.5 decision 16), reason when any is missing or `cross_site` is empty, also fail when `p3_by_holdout` is empty. | promotion.py, tests | `{}`; subset of archives; unknown extra archive; all three present and one failing |
| F2 | `p3_holdout.passed` is true when only one decisive read is estimable. If S1 empties the counted-class set (or pAUC is not estimable), the remaining read alone decides and nothing in the output flags it. Repro: candidate that predicts nv for everything passes with `min_groups=10**6`; with `min_groups=50` the same candidate fails (lo95 -0.46). | HIGH | High | promotion.py:159-161, 168-169 | Require BOTH decisive reads estimable, else `passed=False` with `reason="not estimable"`. | promotion.py | both reads missing/one missing |
| F3 | P2 is "harm not statistically detected", not non-inferiority, and its power to catch real under-40 harm is low (section 4). It also protects under-40 sensitivity not at all (all-age sensitivity is read), though the plan/summary words suggest under-40. | HIGH | High | system_reads.py:128-130, 137-140, 186-188; p2_coverage.json | Pick an option in section 4 and relabel; decide explicitly whether sensitivity is all-age or under-40; word docs identically. | system_reads.py (V6 frozen: wrap in promotion.py instead), plan 10.4/7, docs | per-component pass-rate test on synthetic data |
| F4 | Two conflicting promotion/statistics rules in the same plan (7.1/7.2 and stat doc 5.1 vs 10.4 and code). A signed freeze would be ambiguous about which is binding. | HIGH | High | plan 7.1-7.3 (lines ~217-241) vs 10.4 (415-421); stat doc 5.1 | Delete or mark SUPERSEDED the 7.2/5.1 Holm family, or implement it; one rule only. | plan, stat doc | none |
| F5 | The Holm family for V7 secondaries (Gate A2, E-histo, xAUC, dark read, per-archive, per-band) is not implemented anywhere. `grep holm` finds only the legacy `research/ablation`, `research/external` code; `research/stats/families.py` has no V7 family. Bootstrap deltas carry intervals but no p-values, so Holm has nothing to adjust. Plan 10.4 promises "Holm-adjusted p-values". Family size is undeclared. | HIGH | High | grep over research/, scripts/; promotion.py:75-88, 127-129 | Declare the family (members, size) and implement a bootstrap p-value (two-sided from the resample distribution, or Wald z from SD) plus Holm, or reword to "intervals reported unadjusted, secondary, never promote". | promotion.py or new stats helper, families.py, plan | Holm step-down on known p-values |
| F6 | P1's pAUC condition is a point estimate against a noise bar. Bar = z80 x pair SD = 0.8416 x 0.009787 = 0.008237 (k=1), where pair SD = RMS of the 15 pairwise differences among 6 seeds of ONE single-model recipe on FOLD 0 (`anchor_noise_floor.py:38-42`, `n_pairs` 15). Under the null "difference is pure seed noise N(0, pair SD^2)" false-pass = 20.0% by construction. It does NOT control: (i) lesion-sampling error of the delta (V6-vs-V5 bootstrap SD of pAUC_all delta was 0.0039, so the bar is only ~2.1 such SDs; fine as a sampling test), (ii) seed noise of an ensemble candidate vs a banked ensemble (smaller than single-model noise, so conservative), (iii) the uncertainty in the SD itself (6 seeds, 5 df: roughly +-30%, bar 0.0056 to 0.0109), (iv) any selection among builds/candidates read on folds 1-4. The 4 row sets are nearly identical samples, so the intersection does not materially reduce the 20%. Macro-F1 lower-bound conjunct adds partial protection. Do not describe P1 as "significant". | MEDIUM | High | gate.py:53-54; anchor_noise_floor.py:38-42; system_reads.py:175-178; anchor_noise_floor.json (pair_sd 0.009787) | State in the plan: "P1 pAUC = point estimate exceeds a seed-noise bar; false-pass up to 20% under a pure-noise null"; optionally add a lower-bound > 0 condition. | plan 10.4, promotion docstring | none |
| F7 | F4 acceptance rule is weak and was applied loosely. Declared text: "wider interval ... at the level that restores 0.95 coverage". Applied: "0.93 floor in every scenario". With 300 trials the per-scenario MC SE is 0.009-0.016 (not the 0.013 quoted, which is at p=0.95), so 0.913 is only 1.0 SE below 0.93, and a nominal-exact procedure (true coverage 0.95) fails an all-four-scenarios >= 0.93 rule 18% of the time; a procedure with true coverage 0.92 passes it about 1% of the time (0.3 per scenario), and one at 0.93 about 10%. The 300-trial "failure" of the 95% interval was largely noise: the lead's 1,000-trial run gives 95% coverage 0.931, 0.929, 0.942, 0.914, 0.939, 0.942, 0.939, 0.931, 0.924, 0.922 (mean 0.931, MC SE 0.008): a real but mild under-coverage of about 1.5-2.5 points, not 0.913. The 97.5% interval covers 0.945-0.965 in the 1,000-trial run (300-trial run: 0.933-0.973), so it restores roughly the nominal 0.95 on average but not demonstrably in every scenario (0.945-0.947 in two). 0.933 vs 0.93 or vs 0.95 cannot be distinguished at 300 trials (z = +0.2 and -1.2). | MEDIUM | High | interval_coverage.json; p2_coverage.json; plan F4 row; v7_interval_coverage.py:119-127 | Replace the plan's F4 numbers with the 1,000-trial figures; state the acceptance criterion as "pooled coverage >= 0.94 with SE", or accept the 97.5% level on the basis "restores ~nominal, mild residual". Do not say the 95% interval "fails". | plan 13.1 F4 row | none |
| F8 | Simulation calibration defects (affect both coverage scripts). (a) `POS_SIZES` sums to 64 groups / 141 images (mean 2.20) though the comment says 65 / 158 (mean 2.43); `NEG_SIZES` 1,213 groups / 1,845 images. Slightly fewer images per positive group than measured: effective clustering slightly weaker, direction of bias on coverage unknown. (b) rho (0.85, 0.95) is never calibrated to a real candidate-vs-V6 pair; the implied delta SD 0.012-0.019 brackets V6-vs-V5's 0.0155. (c) Gaussian logits, no point masses at saturation, unlike real vote probabilities. (d) n_boot=400 for pAUC: 2.5/97.5 percentile jitter ~0.13 SD (vs 0.06 at 2,000), so simulated coverage is slightly pessimistic vs production. (e) The sensitivity model borrows ICC 0.93 from the pAUC positives; the argmax-call ICC was not measured. (f) P2 is a conjunction of two components, simulated separately; the joint false-pass is not given (independence assumption would multiply; positive correlation would raise it). (g) The sensitivity bootstrap SD of the simulation (0.0059 at rho 0.95) matches V6-vs-V5's 0.0057, supporting rho ~0.95 for sensitivity only. | MEDIUM | High | v7_interval_coverage.py:35-38; v7_p2_coverage.py:57-61, 104-110 | Fix constants to sum to 65/158, record the effective images; add a joint P2 simulation with correlated components, or state independence. | scripts, JSON | rerun |
| F9 | Row sets. `declared_lists()` returns image-id sets of declared at-risk rows (380 subset of 426, hash-checked, from V6's `read_leak_sensitivity.py`); `v7_linked_rows` = development rows (`frozen_split=='train'`) with `at_risk_links > 0` (771) or `status != 'kept'` (74), union 775 of 15,294 (5.1%). Both are EVALUATION exclusions only (applied as `every[~isin(ids)]`); no V7 runner builds the 4th row set yet (only the function exists; no test; `at_risk_links` counts any data-gate link, including links inside the same group? not verified). The names "without_*" read as removal from training too; V7 training removal is a separate gate and V6 (the reference) was trained with those rows. The two exclusions are not like-for-like. | MEDIUM | Medium | promotion.py:45, 107-118; read_leak_sensitivity.py:46-59 | Rename (`eval_excluding_380`, ...) and write one sentence: "rows removed from the scored set only, both systems scored on the same remaining rows"; add a runner and a test for `v7_linked_rows`. | promotion.py, plan, runner | row counts; both systems on same rows |
| F10 | Power statements are not traceable. `pauc_power_simulation.json` MC section simulates a +0.065 logit-like shift that yields mean delta 0.0145 (not 0.050; script comment says "~0.050"), empirical joint-gate power 0.0 to 0.004 and superiority power 0.20-0.41, which contradicts the headline "80.3% power at +0.050". The 80.3% is analytic power for lower bound > 0 only, at SE 0.0178 = 0.01858 x sqrt(65/71) (a planning SD from V5-vs-control, while V6's own read was 0.0155); Gate A2 as coded also needs the POINT estimate >= 0.050, so pass probability is at most 50% at a true +0.050 (stat doc 4.2 admits this). Stat doc 4.3 cites "bootstrap power 81.2%": not in the JSON. Plan 10.7's Gate A2 "5% at a true +0.020" is arithmetic (P(point >= 0.05 | 0.02, SD 0.0186) = 5.4%) but not in the cited file. Pass probabilities in 10.7 (P1 15-30, P2 80-90, promotion 10-25, A2 5-10) are labelled judgement; P1 cites `member_contribution.json` (exists), P2's 80-90% is not derived from P2's actual power: it is consistent only with "harmless candidate passes ~94-97%". The promotion product treats P1, P2 as independent and omits that all four row sets must pass and P3 is unknown, so 10-25% is a ceiling, not an estimate. Minimum detectable under-40 pAUC effect at 80% power, from the supported SD range: 2.80 x SD = 0.052 (SD 0.0186; two-sided 5% lower bound), 0.057 with the 97.5% interval, 0.034-0.037 if rho is 0.95 (SD 0.012). So +0.050 is about the MDE, not comfortably above it. | MEDIUM | High | simulate_pauc_power.py:123-127, 175-178; pauc_power_simulation.json; stat doc 4.2/4.3; plan 10.7 | Remove the MC block or fix it to simulate +0.050; delete the 81.2% sentence; add a +0.020/+0.035 row to the JSON; mark 10.7 numbers "judgement, ceiling". | script, JSON, docs | none |
| F11 | Bootstrap degenerate resamples: `_pauc` returns NaN when a resample lacks one class; `nanpercentile`/`nanstd` drop them silently, and the count is not recorded. At 65 positive groups of 1,294 the chance of a positive-free resample is about e^-65, so no practical bias for under-40; for small restricted sets (per-archive escalation, E-histo, per-band) the dropped fraction is unreported. Seeds are fixed (20261103, 20261110) and n_boot = 2,000 everywhere in promotion; the same seed in every call means identical resample streams across candidates and row sets (good for pairing, but the "one read" has no replicate-level independence). The simulation script skips such resamples silently too (`interval`, line 81). | LOW | High | system_reads.py:168; stats.py:108; v7_interval_coverage.py:81 | Record `n_valid` per read and fail if < 99% of n_boot. | system_reads wrapper, stats.py | NaN-rich synthetic |
| F12 | Under-40 gain claims and P2 use different levels (97.5% vs 95%). The stated reason ("narrow interval makes no-harm stricter") is right for false failures. Lead's file: P2-pAUC pass rates are lower with the 95% bound than the 97.5% bound (e.g. 0.43 vs 0.53 at true -0.04, rho 0.85), so 95% is the stricter choice. Residual: the 97.5% lower bound still sits above truth in 1.9-6.6% of trials (nominal 1.25%) in p2_coverage.json's 95% column; for gains the nominal one-sided error is not achieved. Gate A2 is secondary so this is low consequence. | LOW | Medium | promotion.py:49-52; p2_coverage.json | Keep; document the residual. | plan | none |
| F13 | P1's Macro-F1 uses all 7 labels while P3's uses present classes; P1 clause `delta > 0 and ci[0] > 0` is redundant (second implies first). P1 compares an argmax-based metric for a candidate whose calibration may shift class priors (e.g. under-40 data upweighting); fine but note. | INFO | High | system_reads.py:120-121, 177 | none | | |

### Reproduction (F1, F2), run with the repo venv from repo root

```
ok={"p1":{"passed":True},"p2":{"passed":True}}; full={s:ok for s in promotion.ROW_SETS}
promotion.verdict(full, {})["promoted"]                          # True, reasons []
promotion.verdict(full, {"bcn20000":{"passed":True}})["promoted"] # True (HAM, MSKCC never read)
promotion.verdict(full, None)["promoted"]                         # False (only guarded case)
promotion.p3_holdout(y,esc,groups,cand_all_nv,base,min_groups=10**6)["passed"]   # True; macro_f1 = {"status": "not estimable"}
promotion.p3_holdout(...,min_groups=50)["passed"]                # False, lo95 = -0.458
```
Also: `p3_holdout` with no escalating positives passes on Macro-F1 alone. Missing p1/p2/`passed` keys raise KeyError (fail-closed, acceptable).

## 4. P2 semantics and options

What passes: `hi95(D_u) >= 0` and `hi95(D_s) >= 0`. This is "zero is inside or above the 95% interval", i.e. the test of H0: D = 0 against D < 0 at one-sided 2.5% has NOT rejected. It is a superiority-style veto. Failure to reject is not evidence of equivalence: the probability of passing is Phi(D/SD + 1.96) (normal approximation), so even a true harm of 1.96 SD passes half the time.

Lead's 1,000-trial simulation (`results/v7/p2_coverage.json`; reviewed, see 5) gives the exact pass rates (MC SE 0.005-0.016):

| True delta | under-40 pAUC pass, rho 0.85 | rho 0.95 | all-age esc. sensitivity pass, rho 0.85 | rho 0.95 |
|---|---|---|---|---|
| 0 | 0.967 | 0.972 | 0.975 | 0.969 |
| -0.01 (sens only) | n/a | n/a | 0.785 | 0.606 |
| -0.02 | 0.827 | 0.555 | 0.354 | 0.100 |
| -0.03 (sens) | n/a | n/a | 0.089 | 0.001 |
| -0.04 | 0.430 | 0.073 | n/a | n/a |
| -0.05 (sens) | n/a | n/a | 0.000 | 0.000 |
| -0.082 / -0.086 | 0.017 | 0.000 | | |

Approximate true harm that still passes with probability 50% / 80% (normal approx., SD from the sim): under-40 pAUC (SD 0.012-0.019) -0.021 to -0.036 at 50%, and -0.012 to -0.021 at 80% (table: a true -0.02 passes 83% at rho 0.85, 55% at rho 0.95). All-age sensitivity SD 0.006-0.009 gives 50% at -0.011 to -0.017; 80% at -0.007 to -0.010. A candidate that is truly 0.04 worse than V6 under 40 (about V6's own gain over V5) still passes the pAUC component 43% of the time if rho is near 0.85. The two components are simulated separately; the joint pass is not reported (independence would multiply the rates, positive correlation would raise them above the product).

Testable non-inferiority margin (normal approximation, cross-checked against the file's NI table): for a truly equal candidate, 80% power needs m = 2.80 SD (two-sided 95% lower bound) = 0.052 / 0.043 / 0.031 for SD 0.0186 / 0.0155 / 0.0109 (file: m = 0.05 passes 0.75 at rho 0.85, 0.96 at rho 0.95; m = 0.03: 0.39 / 0.74; m = 0.02: 0.24 / 0.49). The one-sided 5% bound is about 11% smaller. For all-age sensitivity (SD 0.006-0.009) m = 0.02 gives 0.63 / 0.93 equal-candidate pass, m = 0.03 gives 0.93 / 1.00. I am NOT recommending a margin; the margin is a clinical decision for the owner, constrained by the numbers above.

Options:
A. Keep the rule, relabel as "no statistically detected harm veto (upper 95% bound >= 0)", and publish its operating characteristics (table above). Consequence: harmless candidates pass ~97% per component; a candidate truly 0.02-0.04 worse under 40 passes the pAUC component 43-83% at rho 0.85 (7-56% at rho 0.95). Under-40 gain claims cannot be backed by P2.
B. Replace by a formal NI test with an owner-chosen margin m and criterion `lo > -m`. Consequence: at m <= 0.03 most truly-equal candidates FAIL under 40 (24-74% pass), so P2 becomes a de facto superiority hurdle; at m ~ 0.05 it is testable but permits a harm as large as the Gate A2 effect the plan seeks. All-age sensitivity can be tested at m = 0.02-0.03.
C. Both: the relabelled veto (A) for under-40 pAUC, formal NI (B) on all-age sensitivity where it is powered, and add an under-40 sensitivity read with an exact Clopper-Pearson interval (as the stat doc requires for k < 30) reported but not gating, plus the conjunctive joint pass rate stated. Consequence: honest claim ("no harm detected, not excluded") for the under-40 band; the plan must say plainly that V7's under-40 safety cannot be certified on 65 groups.

## 5. Review of the lead's `scripts/v7_p2_coverage.py` and `results/v7/p2_coverage.json`

Verified: percentile unpacking (`PCTS` order 1.25, 2.5, 5, 95, 97.5, 98.75 maps to lo975, lo95, lo90, hi90, hi95, hi975) is correct; "one_sided_95_lower" uses the 5th percentile; P2 pass uses the 97.5th percentile as the code does. Sensitivity group sizes sum to 1,683 groups / 4,258 images as stated. Truth for sensitivity is the exact delta (bias +-0.0005, fine). pAUC truth from a 200x population (bias up to -0.005 at shift -1.0, rho 0.85: 0.0048, within 2 SD of the truth estimate). Results agree with my independent 150-trial run. `seconds` 2,081, 6 workers, results-not-overwritten guard present, raw trials stored.
Issues: (i) inherits F8 (a)-(f) because it reuses `v7_interval_coverage.draw`; (ii) P2's two components are not jointly simulated; (iii) sensitivity ICC borrowed; (iv) n_boot 400 for pAUC but 2,000 for sensitivity (production uses 2,000 for both), making pAUC rows slightly pessimistic on coverage and, marginally, optimistic on false-pass; (v) shift grid is logit shifts on positives only, so "true delta" depends on that mechanism (candidate worse only by lowering positives' scores; harm via raised negatives or non-uniform harm to melanoma vs bcc is not represented); (vi) the NI table is illustrative; no margin adopted: good; (vii) coverage_95 in this run (0.914-0.942) differs from the 300-trial file (0.913-0.957) in a way consistent with MC noise (SE 0.008 vs 0.013-0.016), which means the plan's F4 "fails the rule" statement should be restated (F7).

## 6. Task answers in brief

2. Single-seed bar: 0.008237 = z80 (0.8416) x pair SD 0.009787; pair SD = sqrt(mean of squared differences) over the 15 seed pairs of seeds 42-47, `youngdata` anchor recipe, fold 0, single model (not an ensemble). k=1. P1's pAUC test is a point estimate vs this bar (strict `>`). See F6.
4. Coverage script: 4 scenarios (rho 0.85/0.95 x shift 0/-0.5), 300 trials x 400 replicates, resampling unit = lesion group, coverage = two-sided percentile interval containing the population delta (200x population), plus one-sided miss rates. Stored JSON matches the code and plan F4 row (95%: 0.913, 0.957, 0.930, 0.913; 97.5%: 0.953, 0.973, 0.943, 0.933; lower-above-truth 97.5%: 0.020, 0.013, 0.047, 0.040). `coverage_mc_se_at_nominal` 0.0126 (plan rounds to 0.013) is at p=0.95; true SEs 0.009-0.016. Seed and trial streams are shared between levels, so the two levels are not independent evidence. `interval_coverage_95only_run1.json` is the same run (identical numbers) from before the 97.5% level was added.
5. See F1, F2. `classes_counted` uses `>=` and ground truth only, shared by both systems; `min_groups=0` counts every present class.
6. See F5.
7. See F9.
8. See F10.
9. See F11.

## 7. Issues I could not resolve

- Whether `at_risk_links` counts only links outside the row's own group (docstring claims so; `build_master_v7.py:296` not traced). 
- Which archives the owner will require for P3 (plan 13.5 decision 16 is open), so F1's required set cannot be fixed by me.
- True rho between a V7 candidate and banked V6; the coverage and false-pass numbers depend on it (0.85 vs 0.95 changes pAUC P2 pass at -0.02 from 0.83 to 0.56).
- Calibration constants (positive/negative logit mean, SD, ICC; sensitivity ICC) were taken from the scripts; I did not re-derive them from banked V6 predictions.
- pytest was not run; I did not verify the test-suite state.
