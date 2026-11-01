# V5 addendum — DRAFT (confirmation, Q11, m4 secondary, deviations)

Status: sections 1–4 are final (data read, nothing pending); D5 (S84 power method) is included in §4. Section 5 (S84) is a placeholder until the
one-shot MILK10k read has run. Every number below is copied from the four source files named per
section (Hard Rule 4); nothing is hand-entered. Sources, abbreviated:
`CONF` = `results/v5/confirm_s42_s43_s44.json`, `Q11` = `results/v5/q11_decomposition.json`,
`DEV` = `results/v5/protocol_deviations.json`, `TAB` = `results/v5/review2_tables.md`.

## 1. Confirmation (folds 1–4, seeds 42/43/44)

Design (`CONF`: `question`, `lock`, `bootstrap`): locked V5 composite versus the in1k control, on
held-out folds 1–4; 12,235 images, 7,002 lesions, 65 under-40 escalating lesions (158 images; by
archive bcn20000 28, ham 27, mskcc 10). Lock `results/v5/composite_lock.json`, sha256
`a34bd78c…6059` (full hash in `CONF`). Uncertainty: 2,000 hierarchical bootstrap resamples (seed
pairs with replacement, then lesion clusters), RNG seed 20261004. No test, reserved or external read
(`CONF: test_read = False`).

| Endpoint | Δ composite − control | 95% CI | p | Gate |
|---|---|---|---|---|
| all-age escalation pAUC | +0.0020 | [−0.0056, +0.0093] | 0.596 | D: FAIL |
| under-40 escalation pAUC | +0.0351 | [+0.0125, +0.0596] | 0.003 | A: FAIL (needs ≥ +0.050 and CI > 0) |
| Macro-F1 | +0.0107 | [−0.0015, +0.0225] | 0.095 | B: pass |
| sensitivity @ spec 0.80, 40–59 | +0.0071 | [−0.0127, +0.0234] | 0.522 | C: pass |
| sensitivity @ spec 0.80, 60+ | −0.0007 | [−0.0101, +0.0083] | 0.872 | C: pass |
| pAUC, histopathology rows | +0.0054 | [−0.0053, +0.0153] | 0.295 | descriptive |
| balanced accuracy | +0.0042 | [−0.0085, +0.0147] | 0.550 | descriptive |
| escalation sensitivity (argmax) | +0.0167 | [+0.0074, +0.0252] | <0.001 | descriptive |
| B2 MEL-vs-NV AUC, under 40 | +0.0491 | [+0.0116, +0.0907] | 0.012 | descriptive |
| B2 class-standardised under-40 pAUC | +0.0220 | [−0.0006, +0.0461] | 0.060 | descriptive |
| B4 archive-fixed-effects under-40 pAUC | +0.0567 | [+0.0301, +0.0821] | <0.001 | descriptive |

Source: `CONF` (`primary.delta`, `gates`); same table as `TAB` §Confirmation.

Gate D per-seed Δ all-age pAUC (`CONF: gates.D`): −0.0002 (s42), −0.0012 (s43), +0.0075 (s44);
mean +0.0020, 1 of 3 seeds positive, min −0.0012. Gate D is evaluated and fails. The gate rule text is
in `docs/V5_RUNSHEET.md` (Gate D), not in `CONF`.

Per-seed under-40 pAUC Δ: +0.0398 / +0.0323 / +0.0332; Macro-F1 Δ: +0.0088 / +0.0169 / +0.0064
(`CONF: primary.per_seed`). All three seeds point the same way on both.

Archive split of the under-40 gain (`CONF: primary.delta.b4_*`): bcn20000 +0.0284 [−0.0083, +0.0679];
ham +0.0161 [−0.0230, +0.0529]; mskcc +0.2453 [+0.1429, +0.3441]. Only the mskcc stratum (10
lesions) has an interval that excludes zero; the pooled fixed-effects estimate (+0.0567) does too.

**Verdict as the pre-declared gates read it.** Gates A and D fail, B and C pass. The primary
under-40 endpoint has a positive, significant interval but falls short of the +0.050 bar; the
all-age endpoint is flat. The result is a **partial confirmation**: a real under-40 ranking gain
below the declared magnitude, with no all-age gain and no harm to older bands or Macro-F1.

Absolute values on the same rows (`TAB` §Absolute values; single models, last epoch): under-40 pAUC
control 0.7295 / 0.7432 / 0.7487 versus composite 0.7694 / 0.7755 / 0.7819 (s42 / s43 / s44);
all-age pAUC control 0.8202 / 0.8220 / 0.8190 versus composite 0.8200 / 0.8208 / 0.8265.

## 2. Q11 — trunk versus module decomposition (descriptive, no gate)

Declared as deviation D4 before any Q11 prediction was read (`Q11: declared`). Contrasts share the
same bootstrap draws (`Q11: bootstrap`). Source: `Q11` (`module_effect`, `trunk_effect`), `TAB`
§Q11.

| Contrast | Under-40 pAUC [95% CI] | p | Macro-F1 [95% CI] | p | All-age pAUC [95% CI] | p |
|---|---|---|---|---|---|---|
| Module effect: composite − Q11 control (same IN-22k trunk) | +0.0270 [+0.0045, +0.0488] | 0.023 | +0.0025 [−0.0073, +0.0118] | 0.649 | +0.0011 [−0.0057, +0.0083] | 0.762 |
| Trunk effect: Q11 control − in1k control | +0.0081 [−0.0127, +0.0301] | 0.432 | +0.0082 [−0.0045, +0.0191] | 0.188 | +0.0009 [−0.0047, +0.0067] | 0.776 |
| Total: composite − in1k control (primary) | +0.0351 [+0.0125, +0.0596] | 0.003 | +0.0107 [−0.0015, +0.0225] | 0.095 | +0.0020 [−0.0056, +0.0093] | 0.596 |

The two components sum to the total on under-40 pAUC (+0.0270 + 0.0081 = +0.0351). Taking the ratio of
point estimates, the module effect is 0.0270 / 0.0351 ≈ 77% of the under-40 gain (derived here; a
ratio of points, not an interval). The module-effect interval excludes zero; the trunk-effect interval
does not. The trunk swap alone carries a larger share of the Macro-F1 gain (+0.0082 of +0.0107) but
none of those Macro-F1 intervals excludes zero. `Q11: confirmation_recomputed` records that the
recomputed primary points and intervals match the stored confirmation (`points_match_stored` and
`intervals_match_stored` both true).

Claim limit: descriptive decomposition on 65 under-40 escalating lesions; the module effect bundles
the two-step head, m4 and the young-data set, so it does not attribute the gain to any one of them.

## 3. m4 secondary arm (deviation D1; seed 42 only, folds 1–4)

Source: `CONF: secondary_D1`; family Holm across the two contrasts on all-age pAUC, seed 42. Secondary
by declaration — it cannot change the lock, the primary verdict or the V5 system definition (`DEV: D1`
binding constraints).

| Contrast (seed 42) | All-age pAUC | Under-40 pAUC | pAUC histo | Macro-F1 |
|---|---|---|---|---|
| m4 − control | +0.0152 [+0.0052, +0.0254] p 0.004 | +0.0360 [+0.0040, +0.0709] p 0.025 | +0.0247 [+0.0088, +0.0396] p 0.001 | +0.0139 [−0.0048, +0.0338] p 0.146 |
| composite − m4 | −0.0154 [−0.0252, −0.0061] p 0.001 | +0.0038 [−0.0291, +0.0414] p 0.810 | −0.0239 [−0.0405, −0.0083] p 0.002 | −0.0051 [−0.0231, +0.0114] p 0.554 |

Holm-adjusted p on all-age pAUC: m4 − control 0.004; composite − m4 0.002 (`CONF: holm_adjusted_p`).
m4 vs control secondary gates (`CONF: m4_vs_control_gates_secondary`): A fail (under-40 +0.0360, bar
+0.050), B pass, C pass. Reading: the composite is **sub-additive** on all-age and histopathology
ranking relative to m4 alone (the pattern Q6 flagged on fold 0, `DEV: D1 why`), and no better than
m4 on under-40. One seed, so this is an exploratory comparison, not a replacement for the primary.
The lock stays as declared.

## 4. Protocol deviations (`DEV`)

Register of departures from the frozen plan; each was declared before the data it concerns were read.

| ID | Declared | What | Effect on the confirmatory claim |
|---|---|---|---|
| D1 | 2026-10-03T19:00+05:30 | m4 (in22k, 224 px) folds 1–4 × s42 added as a secondary arm (`queue_Q7b.txt`) | Secondary only; lock, primary contrast, gates unchanged |
| G1 | 2026-10-03T19:00+05:30 | Gap fix: score banked in1k control checkpoints `R0_kfold_f{1..4}_s42_last.pt` at last epoch before Q7 | None; avoids best-vs-last bias in the comparator |
| D2 | 2026-10-04T16:20+05:30 | Fold 0 of the 15-model OOF reuses Q6 runs `composite_f0_s{42,43,44}_in22k_v5stack` instead of retraining | Fold 0 stays flagged as the selection fold; confirmatory claims stay on folds 1–4 |
| D3 | 2026-10-04T16:20+05:30 | S84 MILK10k read and S56-on-V5 refit deferred to this post-Review-2 addendum | MILK10k unread, no receipt; V6 allocation unchanged |
| D4 | 2026-10-05T04:05+05:30 | Q11 extended from seed 42 only (4 runs) to seeds 42/43/44 (12 runs) | Descriptive, no gate; see §2 |
| D5 | 2026-11-01T21:47+05:30 | S84 power method fixed before any S84/MILK10k read: V1-vs-V5 discordance on non-HAM rows of OOF folds 1–4 only (bcn20000, mskcc; HAM-linked rows and ISIC_0010852 excluded): 6,672 rows, 2,838 lesions, 37 under-40 escalating lesions; unconditional power of one-sided exact McNemar (alpha 0.05), 2,000-resample lesion-grouped bootstrap band. Runsheet §9 governs S84 wholly; the spec's binomial power, N ≥ 40 floor, Macro-F1 ≥ 0.75 boundary, tau80 and fixed-sequence gate are not used (method: `results/v5/s84_power_method.md`) | Power reported whatever it is; below 0.80 the primary is labelled underpowered. Never changes whether S84 runs, alpha or analysis. Residual bias: V5 Dirichlet/S56 refits use the whole OOF, so power is slightly optimistic |

## 5. S84 external read (MILK10k) — PLACEHOLDER

Not run. Spec: `docs/V5_RUNSHEET.md` §9; single-read rule and receipt guard unchanged (`DEV: D3`).
This section is written only after the read and carries no numbers until then.
