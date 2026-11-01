**Row set: non-HAM rows of folds 1–4 — same as `queue_Q9.txt` — minus one non-HAM image that is a near-duplicate of a HAM training image (`ISIC_0010852`). 6,672 rows / 2,838 lesions.**

# S84 power method (runsheet §9 "Power", A01 A9)

Declared 2026-11-01 21:47 +05:30, before any S84 or MILK10k read, before any V1 prediction on the
rows below exists, and before any V1-vs-V5 discordance has been computed. Registered as deviation
**D5** in `results/v5/protocol_deviations.json`. Inputs read to write this note:
`docs/V5_RUNSHEET.md` §9, `results/v5/v5_external_analysis_spec.md`,
`results/v4/kfold/fold_assignments.csv` (development rows only), `research/v4/s56_abstention.py`
(definitions only). No test, reserved or MILK10k label, and no value in
`results/v5/s75_milk10k_eligibility.json`, was read.

## 1. Decision

V1-vs-V5 discordance for the power calculation is estimated on the **non-HAM rows of OOF folds
1–4 only**. HAM rows are dropped, not kept with a stated bias.

## 2. Why

1. **The HAM rows are V1's own training rows.** The 6,981 `archive == "ham"` rows in
   `fold_assignments.csv` are exactly the HAM train split (the OOF matrix excludes HAM val, test
   and reserved, `v5_oof_matrix_spec.md` invariant 6). The six frozen V1 CNNs were trained on all of
   them, and V1's Dirichlet map and S56 thresholds were fit on HAM OOF. On these rows V1 is scored
   in-sample while V5 is scored out-of-fold, so V1's sensitivity is inflated, the V5-better
   discordant cell `p10` shrinks, and power comes out **too low** by an unknown amount. A bias
   whose size cannot be estimated cannot be "stated" usefully, only removed.
2. **The non-HAM rows match the S84 situation.** BCN20000 and MSKCC were never seen by V1, so V1
   meets them as an external domain, as it will meet MILK10k. V5 meets them out-of-fold from a
   training pool that contains the same archives, as it meets MILK10k (labelled "cohort-external,
   partly same-institution": Vienna and MSKCC). Both systems are therefore in the same position
   relative to the rows as at S84.
3. **The cost is precision, and it is reported, not hidden.** Under-40 escalating lesions in the
   row set: **37** (104 images; BCN20000 + MSKCC), against 64 had HAM been kept. The discordance
   estimate is noisy; §5 carries that noise into the power figure instead of quoting a single
   number.
4. Rejected alternative: V1's cross-fitted HAM OOF (`research/predictions_oof_tta/`) is not V1's
   *system* (fold models, not the six deployed checkpoints), its decision layer was fit on those
   same rows, and HAM is in-domain for V1 — it would estimate discordance under no domain shift.

## 3. Exact row set

```python
f = pd.read_csv("results/v4/kfold/fold_assignments.csv")
ham = f.archive.eq("ham")
touch = (f.group_id.isin(f.loc[ham, "group_id"])
         | f.dup_cluster.isin(f.loc[ham, "dup_cluster"])
         | f.effective_lesion_id.isin(f.loc[ham, "effective_lesion_id"]))
rows = f[f.fold.isin([1, 2, 3, 4]) & ~ham & ~touch]
```

| check (assert all) | value |
|---|---:|
| rows | **6,672** |
| lesions (`effective_lesion_id`) | **2,838** |
| archives | bcn20000, mskcc only |
| under-40 escalating lesions / images | **37 / 104** |
| excluded non-HAM rows that touch HAM | 1: `ISIC_0010852` (mskcc, fold 3, <40, escalating; shares `group_id` and `dup_cluster` 1063 with HAM image `ISIC_0032195`) |

The excluded image is a perceptual near-duplicate of an image V1 trained on, so V1 would score it
in-sample; it is also an under-40 escalating case, the exact stratum the power rests on.

V1 must be scored on all 6,672 rows (matching uses every row, §4), not only the under-40 subset.
V1 = the frozen V1 stack exactly as S84 will score it on MILK10k (six `*_best.HAM-only.pt`,
deployed TTA, frozen Dirichlet via `research/external/frozen_params.py`, frozen S56@0.20 policy
from `results/v4/s56/`), **no refit of anything**.

## 4. Per-lesion outcomes (identical definitions to the S84 primary)

- **Caught** (S56 system sensitivity, `s56_abstention.py` docstring): an escalating case is caught
  iff `pred_esc OR referred`.
- **Referral rate** = fraction of cases in S56's `referred` set (argmax-escalated cases are not
  counted as referrals), as S56's budget `R` is defined.
- **V1**: frozen policy as deployed.
- **V5**: seed-averaged composite OOF probabilities for folds 1–4 (TTA if the Q9 TTA outputs exist
  when the power is computed, single view otherwise — stated), the V5 Dirichlet refit, and the V5
  S56 refit. **Matched referral**: V5's S56 budget `R` is the only free value; it is found by
  bisection so that V5's referral rate on the row set equals V1's realised referral rate on the
  same rows. This uses no labels (referral rates only), so the identical procedure can run on
  MILK10k.
- **Lesion weighting**: MILK10k has one dermoscopic image per lesion; the OOF has several. Every
  rate and every discordance probability is lesion-weighted (each lesion weight 1, split equally
  across its images), so the estimate targets a one-image-per-lesion cohort.

Known residual bias (stated, small, direction known): the V5 Dirichlet and S56 refits are fit on
the whole V5 OOF, which includes these rows, so V5 is slightly in-sample in its decision layer →
`p10` slightly optimistic → power slightly optimistic. V1's decision layer never saw these rows.

## 5. Formula

From the 37 under-40 escalating lesions `j` with lesion-weighted indicators `a_j` = V5 caught,
`b_j` = V1 caught:

```
p10 = mean_j[ a_j (1 - b_j) ]      # V5 catches, V1 misses
p01 = mean_j[ (1 - a_j) b_j ]      # V1 catches, V5 misses
pd  = p10 + p01                    # discordance
pi  = p10 / pd                     # share of discordant pairs favouring V5
N   = under40_escalating_lesions   # from results/v5/s75_milk10k_eligibility.json (in-scope)

c(d)  = min{ c : P[Binom(d, 0.5) >= c] <= 0.05 }          # exact one-sided McNemar critical value
Power = sum_{d=1..N} Binom_pmf(d; N, pd) * P[Binom(d, pi) >= c(d)]   # d = 0 never rejects
```

This is the unconditional power of the exact conditional McNemar test (one-sided, α = 0.05) on
`N` paired lesions, the §9 primary.

Uncertainty: recompute `(p10, p01)` and Power on **2,000 lesion-grouped bootstrap resamples** of
the 37 lesions (RNG seed 20261004, as `confirm_read.py`); the matched `R` is held at its full-data
value. Report the plug-in Power, the bootstrap median and the 10th / 90th percentiles, the
observed discordant counts, `N`, and the archive split. If `pd = 0` on the plug-in, Power = 0 and
the report says the discordance was not estimable.

## 6. Reporting rule

Power is reported whatever it is (§9). Below 0.80 the S84 report and the manuscript say
**"underpowered"** next to the primary result, with this figure. Power never changes the S84
analysis, its α, its row definitions or whether S84 runs.

## 7. Precedence over the binomial power in `v5_external_analysis_spec.md`

`v5_external_analysis_spec.md` (sections 2.3 and 3–4) powers a **one-sample exact binomial** test
of under-40 sensitivity (H0 ≤ 0.50 vs 0.70, N ≥ 40) inside a fixed sequence that starts with
Macro-F1 ≥ 0.75. That is a different test and a different hierarchy from runsheet §9, which is the
hashed plan (`results/v5/v5_plan_freeze.json`). **§9 governs S84 wholly**: primary = paired exact
McNemar at matched referral, power by §5 above; key secondary = the S70 contract (Clopper–Pearson);
global = Macro-F1 non-inferiority, margin 0.02. The spec's binomial power figures (80.7% at
N = 40, 85.9% at N = 50), its N ≥ 40 floor, its Macro-F1 ≥ 0.75 success boundary, its τ80
threshold and its fixed-sequence gate are **not** used for any S84 decision. Where the two
documents disagree, §9 wins.
