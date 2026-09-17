# Research Session Log (S9 – V4 final audit)

Moved out of `CLAUDE.md` on 2026-09-17 to bring it under Claude Code's 40k-character limit.
This is the verbatim session-by-session narrative — nothing summarized, nothing reworded.
`CLAUDE.md` keeps the file map, the frozen Age-rule λ values, the Statistics (S7) tooling
description, the Known Findings section, and one pointer line back to this file.

---

- **S9 — frozen analysis plan + single test pass (`research/session9/`)**: `plan.py` holds the
  registry of **19 pre-registered test quantities** and writes `results/analysis_plan.json`
  (frozen 2026-09-04, sha256 `5c9bebcf255a87b7…`, hashed into `results/frozen_artifacts.json`
  under a **separate `analysis_plan` key** so the declared 34 prediction hashes stay untouched).
  `testpass.py` computes them; `receipt.py` writes append-only `results/test_pass_receipt.json`
  and **refuses a repeat execution without `--rerun-reason`**; `nnb.py` defines Number Needed to
  Biopsy with the benign-reweighting prevalence correction (primary π=0.03, range 0.01–0.05);
  `foldbag.py` + `research/oof/extract_foldbag_test.py` build rung A8 (uniform 30-member bag, **no
  member selection**, reusing the 6-member OOF Dirichlet map — a stated under-sharpening mismatch,
  so A8 is a lower bound); `attribution.py` is the GPU stage (Grad-CAM × Tschandl masks).
  Runners: `python -m research.run_session9_plan [--check]`, then
  `python -m research.run_session9_testpass --stage {tables,attribution} [--smoke]`.
  `--smoke` rehearses the whole pass on **val** into `results/session9_smoke/` with no receipt,
  ledger row or table — use it before ever spending the read.
  ✅ **The test split HAS now been read — once, on 2026-09-04, in two stages of one plan**
  (`results/test_pass_receipt.json`: `tables` 18 quantities, `attribution` 1, no rerun reasons).
  All 19 emitted. **Do not read test again** without a stated `--rerun-reason`, which the receipt
  keeps permanently. A7-val reproduced the published 0.8047238352979755 to **zero drift** and the
  conformal refit reproduced all 20 frozen OOF states bit-for-bit. Results in `results/session9/`;
  full narrative in CHANGELOG §S9 results.
  - **Both new rungs LOSE**: A7-oof 0.7871 (−0.0176 vs A7-val, Holm p=0.242), A8 30-member fold bag
    0.7810 (−0.0237, Holm p=0.242). Neither significant, both negative. **Fold-bagging — the last
    credible Macro-F1 lever — is dead**; Workstream E's exhaustion result is now 5 levers, 5 dead.
    A8 keeps its stated under-sharpening caveat, so it is a lower bound, not a refutation.
  - ⚠️ **The "decision-rule not information" mechanism claim WEAKENED on test.** Under-40
    escalation-mass AUC is **0.810** on test vs 0.927 val / 0.889 OOF, and vs 0.975 (40-59) /
    0.933 (60+) on the same split. On test the band also ranks worse. S10 must soften this, not
    restate it.
  - ⚠️ **Orthogonality FAILS in `<40`**: of 18 argmax misses, abstention refers 2 and λ catches
    2 — **the same 2** (Jaccard 1.00). It holds in 40-59 (Jaccard 0.18, λ rescues 11/13). Report as
    measured.
  - λ rule on test: escalation sens 0.731→0.831, missed 78→49, Macro-F1 0.7871→0.7492, referral
    0.178→0.268; NNB at π=0.03 3.0→6.2. `<40` 0.143→0.238 (still poor).
  - Confirmatory `<40` vs `60+` **−0.621 [−0.786, −0.386]**, Holm p<0.001 (vs −0.187 OOF). The
    `40-59` band swung 0.550 val → 0.612 OOF → **0.814** test — magnitude is split-unstable.
  - Conformal: **RAPS bipartite α=0.05 is best** (FRR 0.0138 [0.0038, 0.0349], serious coverage
    0.955, set size 1.87). LAC marginal α=0.10 gives 0.901 overall but 0.735 serious, FRR 0.228.
    **No α in the sweep bounds FRR below 0.01** — reported unreachable, not clipped.
  - Per-band calibration: the **sign-flip replicates** (40-59 −0.023, `<40` +0.015, 60+ +0.034 after
    Dirichlet, aggregate ECE 0.017) but **"`<40` is worst calibrated" does NOT** — on test 40-59 is
    (uncalibrated gap −0.197 vs −0.163).
  - Per-class F1: drag is `akiec` 0.697 / `mel` 0.699; widest CIs are `df` (0.389) / `vasc` (0.308).
    Attribute *level* to akiec/mel and *variance* to df/vasc — different claims.
  - Intersectional: 5/8 cells usable, `<40 × male` **suppressed** (9 escalating < gate of 10),
    `<40 × female` 0.250; TPR gap 0.598.
  - Attribution: lesion-interior Grad-CAM fraction **0.523 [0.509, 0.536]**, lesion area 0.256 →
    concentration ratio 3.45, 0 missing masks. Errors score *higher* (0.570) than correct (0.513) —
    the predicted-class artefact, exactly as the docstring warns. Supporting check only.
  - A declared-but-unevaluated family member enters Holm at **p=1.0** (keeping the pre-registered
    denominator), never by shrinking the family. Not needed in the end — both members ran.
- **S10 — manuscript revision (2026-09-05)**: `paper/manuscript.tex` rewritten against S7/S8b/S9
  (908 → 1,831 lines, 32 → 47 references, no test read). Title now "…Subgroup-Conditional
  Calibration, Abstention and Conformal Guarantees"; abstract 247 words. **All three claims S9
  flagged were revised, not restated**: the pure decision-rule reading of the `<40` failure is
  retracted (part decision rule, part lost ranking); orthogonality is reported as **failing** in
  `<40` (Jaccard 1.00); "under-40 is worst calibrated" is replaced by "the *sign disagreement*
  after a global map replicates, the *ranking* does not". New sections: exhaustion (5 levers),
  per-band calibration, class-conditional-is-not-enough + bipartite conformal (under-40 escalating
  coverage 0.238 → 0.571 → **0.952**), the age rule + NNB + orthogonality, intersectional, external
  PAD. Limitations closes 3 items, adds 4, and **corrects the rare-class attribution** (level drag
  = `akiec`/`mel`; variance = `df`/`vasc`).
  - Verified at session end: `audit_manuscript.py` **83/83 still pass**; structural validation
    passes (47 cites = 47 bibitems, 0 dangling/duplicate); **210** new literals resolved to
    `results/`; **25** hand-written table rows reconstructed row-for-row from the CSVs. The last
    two are staged as `research/ablation/verify_s10_numbers.py` and `verify_s10_tables.py`
    (plus `validate_structure.py`, worth keeping — no LaTeX toolchain here); **S11 should fold
    the first two into `audit_manuscript.py`** to reach ~100+ checks and delete them. Two FRR
    rounding errors were caught this way and fixed (0.034→0.035, 0.156→0.155).
  - Two real bugs fixed in `research/run_session7_stats.py`: Table IV is now `table*` (7 columns
    overflowed one IEEE column), and **`--table-only` no longer reverts Table IV's test column to
    `\textsc{pending}`** — `_render_table_only` now reads `results/session9/age_gap_test.csv`
    (a frozen artifact, not a test read).
  - ⚠️ Left for S11: `audit_manuscript.py` extension, `results/frozen_artifacts.json` regeneration,
    `paper/manuscript_overleaf.zip` rebuild, and attaching `results/CLAIM_checklist.md` as the
    supplementary appendix the manuscript now cites. The manuscript's 3 `TODO` markers (author
    list, repo URL, acknowledgements) are untouched.
  - ⚠️ The paper roughly doubled and no LaTeX toolchain exists here, so the page count is unknown;
    IEEE TMI charges over 10 pages. Cleanest supplementary split if needed: External evaluation +
    Intersectional + Explainability (self-contained, no headline claim in any of them).
- **S11 — audit, artifacts, supplementary, Overleaf bundle (2026-09-05)**. No test read.
  `research/ablation/audit_manuscript.py` **83 → 300 checks**: the S10 verifiers are folded in
  (helpers `present()` for artifact-derived literals and `row()` for whole-table-row
  reconstruction) and the staging files deleted, plus ~20 **directional** assertions that fire
  when a *paragraph* becomes wrong rather than a digit (both new rungs still negative; `<40`
  still worst on within-band AUC; under-40 conformal coverage still marginal < class-cond <
  bipartite; per-band residuals still sign-disagreeing; akiec/mel still lowest F1 and df/vasc
  still widest; attribution errors still scoring higher; PAD vote still below its best member;
  Fitzpatrick I–IV still non-monotonic; `<40` orthogonality still total).
  - **Three bugs.** (1) `build_paper_artifacts.py` rewrote `results/frozen_artifacts.json`
    wholesale and **would have deleted the `analysis_plan` hash** S9 wrote there — fixed to
    preserve sibling keys; re-run verified 34/34 prediction hashes byte-identical and the plan
    key intact. (2) **`paper/manuscript_overleaf.zip` was already unbuildable** — hand-assembled,
    it shipped 1 of the 3 `\input` tables. Replaced by `build_overleaf_bundle.py`, which keeps no
    file list and derives everything from the manuscript's own `\input`/`\includegraphics`
    (now 12 files, 3.44 MB). (3) **CLAIM section pointers were wrong, several from before S10**
    (item 25 → Metrics, item 31 → conformal, item 30 → Backbones); all letters replaced by
    subsection names. Item 26 claimed the manuscript named its software stack — it did not, so
    the sentence was added.
  - **CLAIM checklist now 33 met / 7 partial / 2 not met / 2 N/A** (was 33/6/3/2): item 35
    external validation moved *Not met* → *Partial* now that S8b has run, worded as a
    dataset-shift benchmark rather than validation of the intended use. Item 41's rescue rates
    were wrong ("33–39% in older bands"; the OOF-fitted gate gives 15.4% at 40-59, 36.2% at 60+).
    The manuscript now states the counts, and the audit asserts them against the file.
  - **New**: `research/ablation/build_supplementary.py` → `paper/supplementary.tex` (generated
    from the checklist Markdown, 44 items), `build_overleaf_bundle.py`, and
    `validate_structure.py` (LaTeX-compiler stand-in, covers both documents).
    `results/README.md` index 6 → 13 artifacts.
  - ⚠️ **Still never compiled here.** Upload the bundle and compile `manuscript.tex` +
    `supplementary.tex` in Overleaf. The 3 `TODO` markers and the page-count question from S10
    remain open.
- **S12 — integrity remediation of the external battery (2026-09-05)**. No test read; receipt still
  `n_executions: 2`, no rerun reason. The unlogged 2026-09-05 external pass had typed its
  parameters: a fabricated `lambda_opt = 0.2818` in `eval_decision_curve.py` and
  `generate_case_atlas.py`, implementing `S_esc >= lambda` — **a different rule** from the frozen
  `argmax_c(p_c + lambda_band*1[c esc])` — over **plain 1-view test predictions** read outside the
  S9 pass. All fixed. See `CHANGELOG.md` §S12 and `DATASET_REFINING.md` §2a for the full account.
  - **`research/external/frozen_params.py` is now the only loader for any frozen transfer
    parameter.** `load_lambda_by_band()`, `apply_age_rule(probs, ages|bands)` (delegates to
    `research.thresholds.optimize.apply_thresholds` with `theta = -lambda`), `age_bands()`,
    `load_dirichlet(source)`, `calibrate()`, `load_ham_oof_panel()`, `escalation_mass()`.
    **Never type a transfer parameter again — import it from here.**
    `$py -m research.external.frozen_params --selftest` → 6 checks, reproduces
    `val_macro_f1_rule = 0.7713320988516693`.
  - ⚠️ **Three OOF Dirichlet maps exist and they are not close.** The **deployed** one is
    `research/selective/results_oof/fit_state.json` — bit-identical (`max|dW| = max|db| = 0`) to the
    calibrator `run_session5_agerule` refits in process, and the map `research/session9/testpass.py`
    scores A7-oof and the age rule with. `research/calibration/results_oof/fit_state.json` differs
    by `max|dW| = 0.18` (**this is what pre-registration v1 wrongly declared**);
    `research/conformal/results_oof/fit_state.json` by 1.01 and is correct **only** for conformal
    scoring, whose quantiles were calibrated under it. Corrected in the v2 plan (deviation D2).
    ⚠️ `research/xdomain/run_session8b.py` still loads the calibration map, so its published PAD
    figures are not strictly comparable to the external battery's PAD panels until it is re-run.
  - **Every HAM panel is now `research/predictions_oof_tta/{arch}_train.csv`** (6,981 rows, 24-view
    TTA, cross-fitted) + soft-vote + the deployed Dirichlet map, via `load_ham_oof_panel()`. A
    **sixth** unregistered test read the audit had missed — `convnext_tiny_test.npz` in E4's
    Mahalanobis arm — was deleted; HAM **val** is the in-distribution reference, as in S8b.
  - **Test anchors come from frozen S9 artifacts, never a new read.** `s9_test_anchor()` in
    `eval_decision_curve.py` reconstructs net benefit from TP/FP/N in
    `results/session9/agerule_test.csv`. Same precedent as S10's `_render_table_only`.
  - **Pre-registration v2**: `results/external/analysis_plan_post_s11_v2.json`
    (sha256 `da3c2e1837a21cb3…`), v1 retained byte-for-byte (`e6193e191acba511…`), both hashed into
    `post_s11_provenance.json`. Seven deviations D1–D7. **E6/E7 are post-hoc secondary, in no Holm
    family**; **E0 is withdrawn** and enters Holm at p=1.0 with the denominator held at 5;
    **`E1_dose_response_skew_ordering`** is declared ahead of S13.
  - **E6 rebuilt**: three curves (Vickers risk model `S_esc >= p_t`, plus argmax and the λ rule as
    fixed operating points), lesion-grouped paired bootstrap on ΔNB
    (`research/ablation/bootstrap.py:lesion_resample_indices`, new), and a **within-under-40 panel**.
    All-ages ΔNB **+0.0228 [+0.0178, +0.0282]** at p_t=0.10; **under-40 ΔNB is null at every
    threshold** (+0.0015 [−0.0015, +0.0053]) and **negative by p_t=0.20**, in OOF and in the
    S9-derived test column alike. Only intervals excluding zero are bolded. Consistent with S5's
    "λ helps <40 least".
  - **E7 rebuilt**: exemplars selected by *running* the rule, never by a threshold; an unsatisfiable
    panel renders empty rather than relaxing its filter. Panel A survives on the real λ=0.26
    (`ISIC_0030134`); the rule rescues **205** argmax misses, **4** of them under-40 melanomas.
  - **Bug fixed beyond the brief: APS conformal scoring.** `audit_shift_safety_nets.py`
    reimplemented APS as a deterministic cumulative sum against **randomised**-APS quantiles →
    in-distribution marginal coverage **0.3635** at a nominal 0.95. Importing
    `research.conformal.scores` gives **0.9465**. The conformal ID reference is now the OOF
    **tuning half** (held out from the quantile estimate; the Dirichlet map and RAPS
    hyperparameters were fitted on it, and the report says so).
  - **E3 after the repoint** (`results/external/clinical_triage_report.json`): tier-1 sensitivity
    0.6578 calibrated / 0.6781 raw, point-FRR 0.3289 / 0.2875, NNB(π=0.03) 3.04 / 3.34.
  - **Traceability closed (A10)**: `research/experiments.csv` 0 → **30** `session_post_s11` rows.
    `scripts/external/backfill_ledger.py` (idempotent) covers E2 ×5, E5 ×7, E0 withdrawal ×1;
    E3/E4/E6 log at run time.
  - **Pre-flight is now `scripts/external/preflight.py`**, staged and able to fail:
    `--stage pre_s13` (27 checks, passing) and `--stage pre_s17` (adds S15 tables + A9 manuscript
    wiring; **fails today, correctly**). Verified to fail on a renamed table, a reintroduced
    constant, and a 1-byte plan edit. Its constant grep is over `*.py` — grepping all of `research/`
    matches ~70 prediction CSVs on coincidental float digits and can never pass.
  - **A7 closed ahead of schedule**: ISIC-2019 downloaded and unpacked 2026-09-05 — 9.10 GB,
    **25,331 JPEGs** in `data/external/isic2019_images/`. S13 can skip the download step.
  - Stubs remaining for their owner: `build_post_s11_artifacts.py` (S17). The other three are
    implemented — `extract_external_predictions.py` + `assemble_external_ensemble.py` (S13, the
    latter finished in S14) and `eval_age_rule_transfer.py` (S14).
- **S13 — external inference, run 2026-09-05.** 6 frozen HAM-only checkpoints × 24-view TTA over
  **11,982 BCN-20000** + **2,903 MSKCC** images → 12 matrices in `results/external/predictions/`.
  BCN is 11,982 not 12,413 because the pre-registered `scc` exclusion drops 431 images.
- **S14 — Workstream E1, three-centre dose–response (`research/external/eval_age_rule_transfer.py`)**.
  No test read. `assemble_external_ensemble.py` builds the deployed panel once
  (`results/external/predictions/ensemble_dirichlet_{cohort}.csv`: uniform 6-arch soft-vote +
  deployed HAM-OOF Dirichlet map, nulls → singleton `effective_lesion_id`); the analysis reads that
  frozen file. Point estimates image-level, intervals lesion-grouped via `research/stats/intervals.py`.
  Outputs `results/external/age_rule_transfer_report.json` + 4 CSVs,
  `paper/tables/external_table_bcn_age_replication.tex`, `paper/figures/external_figure_dose_response.png`,
  38 ledger rows (`write_ledger` prunes its own prior `E1_*` rows, so re-running replaces them).
  - ⚠️ **Both pre-registered claims FAIL.** Claim A (AUC flat across centres): under-40
    escalation-mass AUC **0.895** (HAM) / **0.791** (BCN) / **0.790** (MSKCC), spread **0.105**.
    Claim B (sensitivity tracks skew): predicted BCN > MSKCC ≈ HAM, observed the **exact reverse** —
    HAM **0.547** > MSKCC **0.333** > BCN **0.279**, HAM and BCN intervals non-overlapping.
    Contingency fired: *A fails and the ordering is non-monotonic — report the three point estimates
    with intervals, drop the trend language, offer no post-hoc explanation.*
  - ⚠️ **"Under-40 is the worst-ranked band" does not replicate.** Within BCN it is flat
    (0.791 vs 0.795 at 60+); within MSKCC under-40 is the **best**-ranked band (0.790 vs 0.694).
  - Neither confound explains the reversal: melanoma-only (mix control) HAM 0.490 / MSKCC 0.333 /
    BCN 0.274; lesion-level (robustness) HAM 0.500 / MSKCC 0.333 / **BCN 0.211** — reversal widens.
  - ⚠️ **The dose variable is entangled with transfer quality** — BCN has the lowest skew *and* the
    worst transfer (all-ages Macro-F1 **0.402** vs HAM 0.784). Recorded in `verdict.design_limitation`
    as a reason the test is non-decisive, **not** as grounds to keep the hypothesis. S16 must not
    write it up as one.
  - **The operating point does transfer.** Frozen per-band λ zero-shot lifts under-40 escalation
    sensitivity in all three centres: HAM 0.547→**0.625**, BCN 0.279→**0.352**, MSKCC 0.333→**0.389**,
    at +0.024/+0.027/+0.019 referral. First same-modality evidence (S8b's PAD probe had an inverted prior).
  - ⚠️ **The confirmatory member is structurally one-sided.** `E1_under40_sens_frozen_lambda_vs_argmax`
    on BCN: exact McNemar p **1.49e-08**, Holm bound `min(1,5p)` = **7.45e-08** (E0 at p=1.0,
    denominator held at 5). But λ≥0 can only *add* escalating predictions, so `only_argmax_caught`
    is structurally 0 and p reduces to 2·0.5^b in the rescue count. It certifies rescues happened,
    **not** that the rule is worth deploying — argue from the sensitivity/referral trade instead.
  - Descriptive λ sweep (target-fitted, not transfer): under-40 optimum 0.24 HAM (vs frozen 0.26 —
    a sanity check), 0.12 MSKCC, 0.91 BCN; older bands saturate at the grid ceiling on both external
    cohorts.
  - Cohort descriptives reproduce the plan's skew table: BCN <40 **114** escalating lesions/618,
    MSKCC **36**/756 — pooled **150**, not 151, after the SCC exclusion. Skew 3.83× / 6.54× / 8.45×
    (HAM's exceeds the plan's 7.33 because it is lesion-level on the OOF panel, not the training prior).
  - ⚠️ Left for S17: the external family is declared in the plan JSON but **not** in
    `research/stats/families.py` (frozen artifact). E2/E3/E4 have no p-values, which is why S14
    reports a Holm *bound* rather than the adjusted p.
- **S15/S16 — E2 + consolidation and manuscript integration (2026-09-05)**. The session logged
  as "S15" ran Workstream E2 (PAD-UFES-20 prior-shift decoupling) instead of the consolidation
  it was scoped for, so **S16 ran both halves**. `research/external/render_composite_tables.py`
  collapses six orphaned external tables into **two** composite `table*` floats
  (`external_table_validity_battery.tex`, `external_table_safety_nets.tex`) from one renderer;
  three external figures placed (dose–response + decision curve in the main text, case atlas in
  the supplement); new Results subsection IV-K; Discussion transportability paragraph; TRIPOD+AI
  cited. Related Work 6+1 → 4, Limitations 10 → 5, checklists to the supplement, all three
  `TODO` markers resolved. Closes audit finding **A9**.
- **S17 — close-out (2026-09-05)**. `build_post_s11_artifacts.py` implemented (was the last
  printf stub): hashes **35** external artifacts → `results/external/post_s11_artifacts.json`,
  verifies both pre-registration hashes, checks per-workstream ledger coverage, writes
  `reviewer_defense_package.md`. Closes **A5**; **A10** re-verified at **75** `session_post_s11`
  ledger rows. `preflight --stage pre_s17` passes 34/34.
  - ⚠️ **The page target was missed and is not reachable by compression.**
    `research/ablation/estimate_pages.py` puts the manuscript at ~24 pages (band 18–24) against
    IEEE TMI's 10. Prose is 16.2 of those and floats only 6.1 — **delete every float and it is
    still ~18 pages.** Every compression route in the plan was taken and is worth ~2 pages. The
    honest options are to split the paper or target a venue without a ten-page limit.
- **S19 — audit pass: the deployed-Dirichlet bug and four caption defects (2026-09-06)**. No test
  read. Two independent problems, neither catchable by the 348 numeric checks that were passing.
  - ⚠️ **`research/xdomain/run_session8b.py` was loading the WRONG Dirichlet map** — Session 2's
    `research/calibration/results_oof/fit_state.json` instead of the deployed
    `research/selective/results_oof/fit_state.json`. Repointed through `frozen_params`; its three
    duplicated implementations (`load_dirichlet`, `load_lambdas`, `apply_age_rule`) are deleted.
    **`DATASET_REFINING.md`'s claim that "no manuscript number depends on it any more" was
    false** — `audit_manuscript` failed on two Fitzpatrick literals the moment it was corrected.
    Superseded numbers: PAD Macro-F1 under the rule **0.177 → 0.1723**, escalation sensitivity
    **0.5851 → 0.5679**; Fitzpatrick pooled TPR gap **0.297 → 0.295**, type II **0.290 → 0.287**,
    type III **0.256 → 0.247**, unlabelled **0.051 → 0.054**, I–IV spread **0.093 → 0.102**.
    The qualitative finding **survives**: I–IV is still non-monotonic (I 0.349, II 0.287,
    III 0.247, IV 0.278), so the manuscript's reading is unchanged.
  - ⚠️ **The abstract implied the age rule fixes the failure it had just described.** It quoted
    escalation sensitivity "to 0.831" two sentences after the under-40 0.143 — but **0.831 is the
    all-ages figure**; under-40 reaches only **0.238 [0.082, 0.472]**
    (`results/session9/agerule_test.csv`). The body was already honest ("we do not present this
    as a solved problem"); the abstract now carries the same caveat.
  - **Four figure captions asserted things the figures do not show.** The load-bearing one:
    the DCA caption claimed the λ-rule and argmax are "therefore flat in $p_t$" — **false**, net
    benefit $= TP/N - (FP/N)\cdot p_t/(1-p_t)$ declines in $p_t$ even at a fixed operating point,
    which is what the figure plots and what `decision_curve_report.json` contains. Also fixed:
    dose–response error-bar scope, conformal over-claim (`akiec` is *at* target, not below), and
    the Grad-CAM caption arguing from the predicted-class artefact its own body text forbids.
  - **`audit_manuscript.py` 348 → 357 checks**, with an `absent()` helper for claims that must
    *not* return. **All five new guards were proved to fail** by reintroducing each defect
    (`scratchpad/prove_guards_fail.py` pattern) — the first draft of the S8b guard was silent
    because it searched the manuscript rather than the module, and because the docstring names
    the wrong map deliberately; it now inspects code with the docstring stripped.
  - Ledger hygiene: the four pre-fix `session8b` rows are retagged `__superseded_calibration_map`
    (two of them do not use a Dirichlet map and say so), and the S4 row named `abstain10` whose
    own note says "at 20%" — the documented S4 logging bug — is retagged too.
  - ⚠️ **Open, for the owner:** the abstract is **314 words** by
    `scratchpad`'s counter against IEEE TMI's 250 — it was already ~297 before the correction, so
    this is pre-existing and tied to the page-budget decision, not to this pass. Figure 5's
    per-class *marginal* coverage values exist only inside the PNG with **no backing table in
    `results/`** (a Hard Rule 4 gap; the one quoted number, `mel` 0.796, is sourced and checked).

- **S20 — Figure 5 provenance + orphan sweep (2026-09-06)**. No test read.
  - ⚠️ **The Fig. 5 Hard Rule 4 gap CANNOT be closed without a second test read.**
    `run_session4_conformal.py:129` is `eval_split = "test" if plan.read_test else "val"` and the
    published arm ran `read_test=True`, so every bar is a test quantity;
    `results/session9/conformal_cells_test.csv` slices by **age band × escalation, not by class**,
    so the seven per-class cells exist nowhere on disk. Leave it unless you deliberately spend a
    read with `--rerun-reason` (which makes the receipt say `n_executions: 3`). The one number the
    caption quotes — `mel` **0.796** — *is* sourced and checked.
  - **Mechanism fixed**: `research/conformal/plots.py` now emits `class_conditional_coverage.csv`
    beside the figure from the *same* `variants`/`codes` the bars use, so table and figure cannot
    diverge. Proved on the val arm (`--fit-split val --no-test`); the published **test** artifacts
    in `research/conformal/results/` were untouched and the val outputs reproduced field-for-field.
  - ⚠️ **Ledger hazard, seen twice**: a runner with no prune step appends a duplicate row set on
    every re-run (S19 `session8b` — values had changed, so they *conflicted*; S20
    `session4_valfit` — identical). Only `eval_age_rule_transfer.py:write_ledger` prunes its own
    prior rows. **Check the ledger diff after any re-run.**
  - **Orphans**: 6 orphan tables kept (per-workstream detail S16 collapsed into the composites;
    `build_overleaf_bundle` derives from `\input` so none ships). One dead figure deleted
    (`figure3_decision_curve_analysis.png`). ⚠️ **`figure4_risk_coverage.png` and
    `figure6_abstention_tradeoff.png` look orphaned but are BUILD INPUTS** —
    `assemble_figures.py:61-62` merges them into `figure4_selective.png`. Do not delete them.
  - `validate_structure.py` prints an **orphan inventory** every run (currently 6 tables,
    0 figures) plus the S19 `_stray_bytes` and `_swallowed` guards.

- **S51 — backbone probe (`research/v4/backbone_probe.py`, `research/v4/extract_backbone_features.py`)**.
  Frozen features from ConvNeXt-Tiny (control), **PanDerm ViT-B/16** and **DINOv2 ViT-B/14** over the
  4,733-image **reserved** cohort (104 under-40 escalating lesions, **zero HAM images** — the control
  is out-of-sample too, asserted by `--selftest` check 2), scored with `research/v3/ceiling.py`'s probe
  family **imported unchanged**. Plan frozen before extraction: `results/v4/backbone_probe_plan.json`,
  sha256 **`09d5795ef02eca14…`** (hash the *file*, not the string — see below).
  - ⚠️ **The falsifier FIRED.** PanDerm − ConvNeXt under-40 pAUC@0.20 = **−0.0363 [−0.0828, +0.0127]**
    against a required **≥ +0.05 with a CI excluding zero**. Sign negative, interval crosses zero.
    **`REPRESENTATION_BOTTLENECK_FALSIFIED` — the foundation-model arm is dropped and S52/S53 run on
    the in-repo trunk.** Do not reopen the backbone question without new evidence; §9 rule 6 of the V4
    runbook is now a finding, not a precaution.
  - Under-40 pAUC: `convnext_tiny` **0.7319 [0.6597, 0.7979]**, `panderm_vitb16` **0.6956
    [0.6251, 0.7688]**, `dinov2_vitb14` **0.6593 [0.5924, 0.7312]**. DINOv2 loses to the control with
    the interval excluding zero (−0.0725 [−0.1180, −0.0288]).
  - PanDerm **wins** at 60+ (+0.0382) and all-ages (+0.0138), neither certified — the sign flips across
    bands, so the honest statement is "indistinguishable overall, worse under 40", never "worse".
  - ⚠️ Both foundation models show the under-40-is-worst gradient; **the control does not** (its middle
    band is `<40`), matching S14's non-replication on BCN/MSKCC. Intervals overlap heavily — a pattern
    in point estimates, **not** a certified ordering.
  - Robustness arm (mean-pooled patch tokens, run because DermLIP's config does not say which pooling
    its head used): headline contrast **−0.0077 [−0.0489, +0.0364]**. No pooling choice passes.
  - **Licence decision (required before S53): moot.** PanDerm is CC-BY-NC-4.0 and product-blocking; the
    arm is dropped on evidence, so the product keeps no non-commercial dependency.
  - **D1**: DINOv2 pinned to **224 px** — timm's default 518 px would confound the arm with input
    resolution (rung R1's question, not S51's) and OOMs the host. **D2**: the PanDerm tower is DermLIP's
    CC-BY-4.0 **safetensors** `redlessone/DermLIP_PanDerm-base-w-PubMed-256`, remapped BEiT→timm
    (188 keys, assert-exhaustive); the official `.pth` exists only on unaffiliated 0-download mirrors and
    loading it would mean unpickling untrusted code.
  - Two defects fixed: the robustness arm **silently dropped the control** (`load_features` returned
    `None` for the backbone with no `features_alt`), and `--freeze-plan` printed the hash of the
    in-memory string while Windows wrote CRLF — the digest inside `backbone_probe.json` was always the
    correct one. Ledger verified idempotent across two runs (419 rows both times, 0 duplicates).

- **S52 — recipe ladder, built and pre-registered (`research/v4/recipe.py`, `research/v4/train_v4.py`)**.
  No test read, no GPU. Plan frozen at `results/v4/recipe_ladder_plan.json`, sha256
  **`1fb913413545b8e5…`** (hashed from the file, newlines pinned). `--selftest` = **37 checks**.
  Seven arms: **R0** control, **R1** 384 px, **R2** colour constancy, **R4** balanced sampler,
  **R5** Mixup/CutMix + colour-safe RandAugment, **R6** EMA + 60 ep, **R7** metadata branch;
  **R3 dropped** (S50 gate) and `--rungs R3` refuses with the reason.
  - **The screen ranks only R1/R2/R5/R6, on val Macro-F1, never on an under-40 quantity.** S44's
    same-data seed spread is **0.0027** on Macro-F1 (signal/noise ≈ 16×) but **0.2273** on under-40
    escalation sensitivity, where S48 recorded `noise_exceeds_signal: true`. Screening on the
    latter at n=22 would select on noise. The under-40 endpoint is read **once**, at S54.
    The screen is declared a **compute-allocation filter, not a test** — no p-values, and
    "screened out" ≠ falsified. **R4 and R7 are exempt by prior declaration** (S36 check 7 makes
    R4's null *predicted*; R7 is a mechanism and cannot be ranked against a lever).
  - MCID **0.020** val Macro-F1 = ~7× the seed spread and the size of the Dirichlet rung A6→A7
    (+0.019), larger than TTA A5→A6 (+0.014). Non-inferiority margin **0.005**. R4/R7 MCID **0.05**.
  - ⚠️ **The S54 gate as the runbook worded it is confounded.** Reserved is **4,733 images, 0 HAM**
    (3,856 BCN + 877 MSKCC); V4 trains on those archives and the V1 ensemble never did (S14: BCN
    transfer 0.402 vs 0.784 in-domain), so "beat the frozen V1 ensemble" fires on Macro-F1 for
    corpus reasons alone. Split into **Gate A** (vs V1, kept as declared, labelled confounded,
    reported as the *deployment* delta) and **Gate B** (vs the **V4 pooled control**, same corpus,
    seed-matched paired). **The four outcomes are read from Gate B.** Declared deviation.
  - ⚠️ **The V1 comparator is already frozen for 4,587 of 4,733** reserved images
    (`results/external/predictions/`, S13/S14) — no new pass needed. The **146** missing are exactly
    the `scc` rows S13 excluded; dropping them costs **104 → 103** under-40 escalating lesions,
    one below S48's target and no longer S51-comparable. **S54 must top up those 146.**
  - ⚠️ **V4 `val` also contains zero HAM images** — the screen selects on a HAM val distribution
    and Block 2 on a BCN/MSKCC one, so a rung can reverse. Declared, not a finding.
  - ⚠️ **Block 3 is three nights, not one, if R1 is promoted** (~5.3 h × 4 at 384 px vs ~1.8 h × 4
    at 224 px). S48's ≥3-seed rule binds both Block 2 arms unconditionally.
  - **Block 0 (GPU rehearsal) was run inside S52** and found three defects. (1) ⚠️ **4 dataloader
    workers die at 384 px pooled** — Windows error **1455** `ERROR_COMMITMENT_LIMIT`, not VRAM;
    host has ~7.5 GB commit headroom (30.4 limit / 22.9 committed) and each worker is a full
    spawned torch import. **Default is now 2**, free because at 384 px 2≈3 workers (306 vs 307
    ms/batch, GPU-bound). (2) ⚠️ **batch 16 at 384 px would have confounded R1** with a second
    factor (LR schedule); measured peak is **1.75 GB of 8.55 GB at batch 32**, so **batch is 32
    across the ladder**. (3) The runbook's **~50 min/rung is 3.4× too slow** — 30 ep over 6,981
    images at 224 px is **14.8 min**, 384 px costs 2.26× not 2.94×. **Whole programme ~9–13 h,
    not ~40 h**; the "three nights vs one" contingency does not arise.
  - ⚠️ **R2, not R1, is the expensive screen arm**: `shades_of_grey` is **18.8 ms/image** vs the
    control's 3.5 (S49's deliberate float64 power-6 — do not optimise it away), so R2 is
    data-bound at 224 px. The estimator takes `max(GPU, input pipeline)` per arm.
  - **Plan re-frozen** with the measured timings, `1fb913413545b8e5…` → **`eff9d79f36f351ff…`**.
    Legitimate — no data collected — and checkable: payload diff shows **exactly one key changed**
    (`/seeds/budget_contingency`), no MCID/endpoint/falsifier/gate/recipe field touched.
  - The recipe is reachable **only** by composing registry entries (`--rungs R1 R5`) — no parameter
    is typeable on the CLI, the `frozen_params.py` discipline applied to training. `write_ledger`
    prunes its own rows (S20 hazard). R7 needed a **new** tabular encoder: `research/fusion/tabular.py`
    carries HAM's localisation vocabulary, the V4 manifest carries ISIC's. `--smoke` writes real
    checkpoints to `results/v4/recipe_runs/smoke/` so the save path is rehearsed, not skipped.

- **S53 — Block 1 closed, Block 2 running (2026-09-16)**. Block 1 best val Macro-F1 (seed 42,
  HAM-only, `results/v4/recipe_runs/`): R0 **0.7773**, **R1 0.7998 (+0.0225, the only lever to clear
  the MCID)**, R4 0.7682, R7 0.7663, R6 0.7651, R5 0.7482, R2 **0.6751 (−0.102)**. R4 meets its own
  endpoint (esc. sens. +0.0617); R7's age-flip span 0.0475 misses 0.05. **Composite = R1 + R4**;
  the second ranking slot is unfilled (no other lever ≥ control).
  - **Control gate resolved by the pre-registered rule**: `ml/training/train.py` on the identical
    split scored **0.7765** vs train_v4's 0.7773 → trainers agree, band was stale →
    `FAILED_BUT_DEVIATION_ACCEPTED` (`results/v4/control_reference.json`). The published 0.7482 /
    0.7509 are single draws of a max-over-epochs statistic with ~0.02 jitter; the 0.0027 "seed
    spread" behind the band and MCID is **n=2**. Block 3's seeds are what measure it properly.
  - R1 ran at batch 16 × `--grad-accum 2` (effective 32), 1 worker. Its checkpoint was verified
    against its JSON after three concurrent launches.
  - ⚠️ **Error 1455 chain**: full disk → auto-managed pagefile can't grow → commit limit pinned →
    dataloader shared-memory fails. `train_v4` now refuses below 5 GB free (2 GB mid-run).
    `run_morning.ps1` has a named mutex (one copy only) and skips Block 2 arms already banked.
  - ⚠️ Scheduled tasks fired late (07:45 → 09:03) and the agent ignored its cutoff; both tasks are
    deleted. Gate time-critical rules in scripts, not prompts.
  - **Next**: Block 2 finishes ~17:00–17:40 → Block 3 (seeds 43/44, both arms, paired) → S54
    (reserved-cohort inference; Gate A confounded, **Gate B decides**; top up the 146 `scc`
    images for the V1 comparator; evaluate `_last.pt` for multi-stage rungs).

- **S54 build (2026-09-16 evening) — gate built and rehearsed, reserved NOT yet read.** Block 3
  banked all six pooled runs at 19:34. Plan `results/v4/s54_plan.json` sha256 **`990320b6ce22f92a…`**
  (`_last` primary / `_best` sensitivity, no TTA, `<40` pAUC@0.20 McClish, seed-mean paired delta +
  range, Gate B decides, Gate A confounded). Runners, in order: `research.v4.s54_topup_v1` →
  `s54_infer` → `s54_gate`; each has `--smoke` (V4 val → `results/v4/s54/smoke/`). Read-once
  receipt `results/v4/s54/reserved_receipt.json` (`s54_guard.py`): a repeat needs `--rerun-reason`,
  a crash resumes with `--resume`. `s54_gate --selftest` 16/16. ⚠️ Every `_last` is 8 epochs past
  the val peak (early stopping), and no `_best` lands in the head stage — read the `_best` row.

- **S54 result (2026-09-16 night) — reserved READ ONCE, outcome 4 "neither".** Receipt: v1_topup 1 /
  infer 1 / gate 1, no rerun reasons. **Do not read reserved again** without `--rerun-reason`.
  Gate B `_last`: Macro-F1 **−0.0121 [−0.0321, +0.0098]**, under-40 pAUC **+0.0089 [−0.0200, +0.0367]**.
  ⚠️ `_best` sensitivity gives outcome 2 (Macro-F1 +0.0316 [+0.0036, +0.0563]), but the gap comes
  from the *control's* `_best` scoring worse on reserved. Checkpoint choice alone moves one arm by up
  to 0.057, so the Macro-F1 result is **checkpoint-rule-dependent**: report both, never claim the
  `_best` row. Under-40 pAUC is flat under both rules (all models 0.72–0.75; V1 0.7335).
  Gate A (CONFOUNDED): V4 beats deployed V1 by **+0.18 Macro-F1** (V1 0.4108) and **+0.004 pAUC**.
  In-domain training fixes general accuracy but **not** under-40 ranking, so the representation route
  is closed (S51 + S54). §4 (S55–S59) is the whole contribution. S54 changed `s54_guard` to require
  commit headroom of `1.5 + 2.7 × workers` GB, and the S54 runners now default to `--num-workers 0`.
  S53 ledger rows now use best-epoch metrics throughout. ⛔ Stop and review before S55.

- **S55 — group-conditional (per-age-band) Dirichlet calibration (2026-09-16)**. No test read.
  Built `research/multical/groupwise.py` + `research/run_session55_multical.py`: one Dirichlet
  map per age band (`<40`/`40-59`/`60+`, each ≥300 OOF rows; `unknown`, 38 rows, falls back to
  the global map), K-fold cross-fitted within band so every reported number is leak-free.
  Closes the S7 sign-flip: signed-gap spread across bands **0.068 → 0.008**, `<40` ECE
  0.0276→**0.0095**, `60+` 0.0410→**0.0301**. ⚠️ **The ECE *spread* across bands widens, not
  narrows** (0.0150→0.0206) — `<40` had the most sign-flip room and improves most, leaving `60+`
  the clearer worst-calibrated band. Report both: per-band calibration fixes direction
  disagreement, not magnitude disagreement. Frozen artifact
  `research/multical/results_oof/fit_state.json`; `research/external/frozen_params.py` gained
  `load_group_dirichlet()`/`calibrate_by_band()` as the sole loader (selftest still 6/6, global
  map and lambdas unmoved). Not yet adopted as the deployed calibrator — a later session opts in
  explicitly. One ledger-duplication bug (S20's known hazard) caught and fixed with the
  `eval_age_rule_transfer.py` prune pattern before it compounded.

- **S56 — group-conditional selective abstention (2026-09-16)** (`research/v4/s56_abstention.py`,
  outputs `results/v4/s56/`). Frozen V1 ensemble; per-band CRC-floor referral thresholds fit on
  OOF, score `msp` selected on val (`esc_risk` failed the pre-declared F1 guard → secondary arm),
  plan `results/v4/s56_plan.json` sha256 **`4e142ad2c423a0ec…`**, reserved read **once**
  (`results/v4/s56/reserved_receipt.json`; repeat needs `--rerun-reason`). No test read.
  - **Primary SUPPORTED**: under-40 system escalation sensitivity band − global at R=0.20
    **+0.154 [+0.104, +0.216]** (MCID 0.10); positive at every budget (+0.125 … +0.219).
  - ⚠️ **It reallocates, it does not create**: under-40 referral **0.232 → 0.454**, `60+`
    sensitivity **−0.100**, all-ages **−0.040**, total referral −0.017. Always quote the cost.
  - ⚠️ **Budgets and floors do not transfer off HAM**: nominal 20% realises ~39–40% referral on
    reserved; nominal floor 0.855 is met in **0 of 15** budget×band cells (12 violated, 3 below with CI reaching it — `s56_report.json`; was misquoted as 10 until the 2026-09-17 audit). CRC is
    exchangeability-bound — call the floor nominal. V1's full-coverage reserved sensitivity is 0.392.

- **S57a — age resolution: diagnostic, baseline, candidate λ(age) fits (2026-09-16)**
  (`research/v4/lambda_age.py`, `research/v4/age_estimator.py`; outputs `results/v4/age_bin_diagnostic.csv`,
  `lambda_baseline_s5.json`, `lambda_candidates.json`, `age_estimator_report.json`, `s57a/`,
  `paper/figures/lambda_age_diagnostic.png`). No test read. Reserved read **once**, specificity only
  (`results/v4/s57a/reserved_audit_receipt.json`). S5 baseline reproduced exactly through `frozen_params`.
  - ⚠️ **No finer age resolution beats the frozen 3-band rule on leak-free nested-CV cost** (A1 0.398 ± 0.031;
    A2–A7 0.406–0.438). **A7 (shrunk 5-year) collapses to its A6 prior (τ=∞)**; A6 leaves under-40 sensitivity
    unchanged. A3 lifts under-40 OOF sensitivity 0.625→0.703 only by **doubling under-40 referral**. Only 35–39
    of the 8 under-40 bins clears the 30-positive gate; the gate's fallback puts pooled λ 0.65 on the other 7.
    Structure in λ(age) exists at 40+ (0.83 at 50–54 vs 0.04 at 80+), not under 40.
  - Every arm is **floor-repaired per band** (`repair_floor`) — the first fit let A3 "win" by breaking 60+
    specificity (0.77). Floors met exactly in-sample hold ~half the time held-out (A1 60+ 0.849).
  - ⚠️ **Deployed 40-59 λ=0.74 violates the 0.85 specificity floor on reserved: 0.748 [0.718, 0.775]**
    (60+ 0.827). A finding about the deployed rule, not grounds to relax the floor.
  - Age estimator (Tweedie GLM on frozen features): MAE 11.63 y, R² 0.249, beats least squares by 0.19 y, but
    **+19.5 y bias under 40** (3.8% of under-40 predicted in-band) — fallback/QC only, never the rule's age
    (`resolve_age`).

- **S57b — λ(age) freeze decision: REJECT-cost (2026-09-17)** (`research/v4/lambda_{crossfit,stability,freeze,transport}.py`;
  plan `results/v4/s57b_plan.json` sha256 `e00aafd837e62d74…`; verdict `results/v4/lambda_verdict.json`).
  No test read. Reserved read **once** (`results/v4/s57b/reserved_receipt.json`).
  - **Cross-fitting:** fully nested — calibrator, hyperparameters and λ never see the scored fold.
    κ flips between 0.1 and 0.001 across folds.
  - **Family:** A7 (primary) and A3 (dev-best by a declared rule).
  - **A3 passes Gate 1 but fails on cost.** Reserved under-40 sensitivity **+0.208 [+0.127, +0.296]**
    (Holm p 0.002), but under-40 referral **0.145→0.278** (×1.92) and under-40 specificity 0.960→0.857.
    ⚠️ A3 is close to the pooled λ below 55, so this is a bigger under-40 λ, not finer age resolution.
  - **A7 fails Gate 5:** band 0.37 against a distance of 0.10.
  - **Deviance test:** age *does* carry structure beyond three steps (LR 35.6/7 df, p 9e-6), but no
    curve turns it into a clinically efficient gain.
  - **Consequences:** nothing frozen, A1 stays deployed, **S57c skipped**, transport refuses to run.

- **S64 + Phase X (2026-09-17)** (`research/v4/ceiling_u40.py`, `results/v4/s64/`). OOF only.
  ⚠️ **No cutoff rule can fix under-40: it is a ranking limit.** Under-40 AUC **0.878** vs 0.948 / 0.929;
  a constant λ needs **0.259** under-40 referral for sensitivity 0.80, the in-sample per-bin oracle
  (bounds every λ(age)) 0.168. Ensembling adds **+0.004 [−0.042, +0.052]** under-40 AUC; pooled training
  (81 vs 34 under-40 lesions) was already flat in S54. Runbook §6b adds **S65** (S55+λ+S56 combined
  policy), **S66** (per-hospital λ; V1 arm GPU-free on non-reserved BCN/MSKCC train rows) and **S67**
  (under-40 ranking, target +0.05 AUC, GPU gated on a CPU probe). ⚠️ The runbook's **S57c** is the gated
  λ(age, margin) session, not this one.

- **S58 — three-stage front end (2026-09-17)** (`research/v4/s58_front_end.py`, `results/v4/s58/`,
  plan sha256 `cc4b205a2b60f1dc…`). Reserved read **once** (`results/v4/s58/reserved_receipt.json`);
  no test read.
  - **Setup.** Frozen HAM-only ConvNeXt-Tiny features; Mahalanobis gate, LR router, LR heads.
  - **Result: the pooled head (H1) is stage 3.**
    - Routed heads beat the checkpoint head: Macro-F1 **+0.107 [+0.075, +0.136]**, SUPPORTED.
    - But they are not non-inferior to one pooled head (−0.008 [−0.024, +0.007]).
    - Pooled 0.534 vs checkpoint head 0.419, about 63% of the way to retrained V4 (~0.60).
    - Under-40 pAUC is flat (+0.007): the ranking limit again.
  - ⚠️ **Gate fails.** PAD reject **0.378**, AUROC 0.799 once fit on the pooled corpus. It rejects
    escalating lesions more often (0.087 vs 0.067). Not usable for S59 as built.
  - ⚠️ **Router 0.932**: pairwise archive AUC 0.99 ≠ 3-way accuracy.
  - ⚠️ **Always use `ml.preprocessing.transforms.build_eval_transform` (bilinear)** for features.
    - S51's `extract_backbone_features` transform is bicubic: features sit 22% away and the
      control loses 0.021 HAM val Macro-F1.
    - S51's ConvNeXt control pAUC (0.7319) is not the deployed model's (0.7110 on the same rows).
    - The v3 BCN/MSKCC holdout caches overlap reserved (1,135 + 262): never fit on them.

- **S66 — λ per hospital: REJECT-cost (2026-09-17)** (`research/v4/s66_lambda_centre.py`,
  `results/v4/s66/`, plan sha256 `caf910c5a71962dd…`). V1 arm only, no GPU. Reserved read **once**
  (`results/v4/s66/reserved_receipt.json`); no test read.
  - Fit on HAM OOF + BCN/MSKCC V4 **train** rows (clean for V1). **τ\* = ∞**: the CV grid is flat,
    so P2 = pooled λ except where the per-centre 0.85 floor lowers a band. BCN's under-40 λ (0.66)
    is **floor-bound**; S14's 0.91 had no floor.
  - BCN reserved under-40 sensitivity P2 − P0 **+0.216 [+0.125, +0.313]**, but referral
    0.160→0.303 (×1.90, **1.95 referrals per extra catch**), `<40` specificity 0.965→0.859, older-band
    sensitivity below P0's CI → G2/G3/G4 fail. Routing (G5) passes.
  - ⚠️ **The gain is the pooled under-40 λ, not per-centre fitting**: PC − P0 is the same +0.216 and
    P2 − PC = 0.000. Same pattern as S57b's A3. P2 would also move HAM's under-40 λ 0.26 → 0.66.
  - ⚠️ P0's low CV cost comes from breaking the floor (BCN 40-59 spec 0.757 on reserved).
  - Nothing frozen, no `load_lambda_by_centre()`. **S65 takes the frozen 3-band rule.**

- **S65 — combined policy (S55 + frozen λ + S56): REJECT-cost (2026-09-17)**
  (`research/v4/s65_combined_policy.py`, `results/v4/s65/`, plan sha256 `22c964fdb5810ab6…`).
  Reserved read **once** (`results/v4/s65/reserved_receipt.json`); no test read. COMB score `entropy`
  (val-selected); its `S56` arm reproduces S56's frontiers exactly.
  - <40 system sensitivity COMB − S56 at R=0.20 **+0.082 [+0.042, +0.131]** (SUPPORTED, MCID 0.05);
    60+ vs λ-alone passes non-inferiority. **But S56 at the workload-matched budget (R=0.305) ties it:
    +0.004 [−0.014, +0.025]**, so the pre-declared rule gives REJECT-cost. <40 referral 0.454→0.608,
    <40 retained Macro-F1 0.459→0.134.
  - ⚠️ **The per-band map undoes the frozen <40 λ** (fit under the global map): <40 decision
    sensitivity 0.409→0.262 on reserved. Calibration alone moves <40 sensitivity −0.007 (S64 holds).
    Stacking S55 needs a λ refit, which would be a new, separately planned tool. S59 composes S56 alone.
- **S67 stage 1 — under-40 ranking probes: STAGE2_NO_GO (2026-09-17)** (`research/v4/s67_ranking_probe.py`,
  `results/v4/s67/`, plan sha256 `20405144a9884445…`). CPU only, frozen HAM-only ConvNeXt features on
  12,115 dev rows (BCN/MSKCC train+val, HAM val; 151 under-40 escalating images / 57 lesions). No reserved or test read.
  Under-40 pAUC Δ vs pooled control 0.728: specialist **−0.039**, hard-case reweight **−0.000**,
  metadata **+0.007 [−0.002, +0.017]**; the gate needs +0.03 → **no GPU fine-tune**. With S51/S54/S64 this
  closes the under-40 question: it is a ranking limit, not a head, weighting or decision-rule problem.
  Stage 0 (V4 ensemble on reserved) is unrun — a reserved read, owner's call.
- **V4 audit (2026-09-17)** — integrity clean (receipts, plan hashes, splits, frozen checkpoints) and every
  headline number from S54–S67 reproduces with independent code. **Two real defects** (CHANGELOG §V4 audit):
  - ⚠️ **`research/v4/colour.py:shades_of_grey` over-brightens by √3** (factor `‖e‖/e_c`, should be
    `‖e‖/(√3·e_c)`); a median 79% of pixels saturate. **R2's −0.102 and S49's archive-decodability verdict
    are invalid** — R2 is untested, not failed. **Fixed 2026-09-17** (S53r); old S49 outputs/ledger rows
    retagged `superseded_colour_bug`. Re-run: `scripts/run_s53r.ps1` (plan `results/v4/s53r_plan.json`
    `2ef2e91b…`, 19 runs ~11 h, patience off, `--run-tag rerun`, ledger `v4_s53r`); read with
    `python -m research.v4.ladder_rerun --report`.
  - ⚠️ **Early stopping (patience 8) truncates cosine runs at high LR**: R6 (EMA + 60 ep) stopped at 19/60,
    pooled S54 runs at 14–22/30. S53's ladder ranking and "training does not fix under-40" are conditional on this.
  - Reserved has been read 8 times across V4 (S54, S56, S57a, S57b, S58, S59, S65, S66; each receipted once), plus S51's pre-receipt probe — treat it as a reused evaluation set.
  - Under-40 AUC gap is partly case mix: melanoma-vs-benign 0.873 (<40) vs 0.940 / 0.909.
- **S53r → S62 close-out (2026-09-17)** (CHANGELOG §"S53r readout → S59 freeze …"). No test read.
  - **S53r**: R1 (384 px) is the only lever, +0.0297 [seed range +0.010, +0.047]. R2 is now
    NULL (−0.0045); its old −0.102 was the colour bug. The base stays **V1**.
  - **S59**: frozen plan `e7ebc3f5…`, reserved read once. The deployed stack is **S56@0.20 alone**;
    the λ and conformal layers both lose at matched workload.
    Contract on reserved: **CONTRACT_FAILS, every term NOT_MET**:
    - under-40 system sensitivity 0.763 against a floor of 0.855;
    - referral 0.386 against a limit of 0.25;
    - selective coverage 0.614 (= 1 − referral; **not** a joint bootstrap pass rate — misread until the final audit).
    The S63 test-read gate is therefore closed.
  - **S60**: `research/v4/s60_contract.py` and `s60_api.py`, tests in `tests/test_s60_service.py`
    (19 pass). The model is stubbed: `/predict` takes softmax probabilities. Every response carries
    the Fitzpatrick "no evidence for V/VI" notice and S59's external verdict. `torch` is imported
    on CPU via `research.calibration.methods`.
  - **S61**: `paper/v4/model_card.md`.
  - **S62**: `research/v4/s62_freeze.py`, which writes `results/v4/analysis_plan_v4.json`
    (`bd320319…`) and registers it under `analysis_plan_v4` in `frozen_artifacts.json`.
    `families.py` now declares the eight `v4_*` families.
  - **Left (closed by the final audit, see below):** S63 manuscript reframe; S59 in `audit_v4.py` `RECEIPTS`.
- **V4 final audit + S63 close-out (2026-09-17)** (CHANGELOG §"V4 final audit"). **V4 is done.** No test read, no new reserved read.
  - **S63 gate CLOSED**: no band meets 0.855 on reserved → no third HAM test read; receipt stays **2**.
    `research/v4/s63_final_verdict.py` → `results/v4/final_verdict_v4.json` (`--check`); reframe
    `paper/v4/manuscript_v4.tex`, audited by `research/v4/audit_manuscript_v4.py` (179 claims plus a
    closure check that fails on any unregistered decimal). Never compiled here.
  - `audit_v4 --check` now **76** checks (S59 receipt + recomputation, S53r, 8 receipted reads).
  - ⚠️ **S59 `contract.coverage` (0.614) is 1 − referral**, not a joint pass rate. The S60 field is now
    `selective_coverage`.
  - ⚠️ **V1 is the base only because S56 needs cross-fitted OOF predictions**, not because V4 lost:
    V4 pooled beats V1 by +0.18 Macro-F1 on reserved (Gate A, confounded). A V4 base is the main
    open decision (GPU K-fold + new plan).
  - ⚠️ **S49 re-run (fixed colour step): verdict flipped to "not material"**, Δ −0.019 (0.9905→0.9716),
    intervals separate but below the −0.02 rule.
  - ⚠️ The frozen S62 index says `reserved_reads: 7`; the true receipted count is **8** (S57a's audit).
  - ⚠️ S48's declared primary endpoint (sensitivity at V1-matched budget, lesion unit) was never read in
    that exact form; S54 used pAUC. Say so in any write-up.
