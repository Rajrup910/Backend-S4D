# results/

Canonical home for every number and figure in `paper/manuscript.tex` (hard rule 4).
Regenerate with `python -m research.ablation.run_part_a` followed by
`python -m research.ablation.build_paper_artifacts`.

| Artifact | Produced by | Used in |
|---|---|---|
| `ablation_table.csv` | `run_part_a.py` | Table II, the ablation ladder |
| `bootstrap_cis.json` | `run_part_a.py` | all lesion-grouped intervals |
| `mcnemar_delong.json` | `run_part_a.py` | the paired ladder tests |
| `age_band_prior.csv` | `build_paper_artifacts.py` | the under-40 mechanism |
| `frozen_artifacts.json` | `build_paper_artifacts.py` | the frozen-artifact declaration |
| `reports/` | `build_paper_artifacts.py` | per-phase source reports |
| `oof_vs_val_comparison.csv` | `research/oof/run_comparison.py` | Table III, val vs OOF fitting |
| `oof_provenance.json` | `research/oof/extract_oof.py` | fold-checkpoint provenance |
| `comparison_families.json` | `run_session7_stats.py` | the declared multiplicity families |
| `analysis_plan.json` | `run_session9_plan.py` | the pre-registered test quantities |
| `test_pass_receipt.json` | `run_session9_testpass.py` | append-only record of the test read |
| `session9/` | `run_session9_testpass.py` | every test number added after session 5 |
| `CLAIM_checklist.md` | maintained by hand | rendered to `paper/supplementary.tex` |
| `external/analysis_plan_post_s11_v2.json` | `freeze_analysis_plan_v2.py` | the external pre-registration |
| `external/post_s11_provenance.json` | `freeze_analysis_plan_v2.py` | both plan hashes, v1 retained |
| `external/*_report.json` | `research/external/eval_*.py` | Sec. IV-J and IV-K |
| `external/predictions/` | `extract_external_predictions.py` | BCN-20000 and MSKCC matrices |
| `external/post_s11_artifacts.json` | `build_post_s11_artifacts.py` | SHA-256 manifest of the battery |
| `external/reviewer_defense_package.md` | `build_post_s11_artifacts.py` | consolidated external provenance |

Four scripts read this directory rather than write it, and all four should pass
before the paper is submitted: `python -m research.ablation.audit_manuscript` checks
every number in the manuscript against the files above;
`python -m research.ablation.validate_structure` stands in for the LaTeX compiler
that is not installed here; `python scripts/external/preflight.py --stage pre_s17`
checks the external battery's integrity end to end; and
`python -m research.external.build_post_s11_artifacts --check` re-verifies the
pre-registration hashes and the ledger coverage without writing anything.
`python -m research.ablation.build_overleaf_bundle` then assembles the upload from
the manuscript's own dependency list, and
`python -m research.ablation.estimate_pages` estimates the compiled length, since
there is no compiler here to measure it.

Frozen artifact set: 34 prediction matrices, declared 2026-09-05T17:58:46+00:00.

---

## Phase V4 Artifacts (`results/v4/`)

Canonical storage for all V4 research session plans, receipts, run summaries, and diagnostic benchmarks (Sessions S48–S67). Verified by `python -m research.v4.audit_v4 --check` (76/76 checks passing).

| Artifact | Produced by / Session | Role & Contents |
|---|---|---|
| `analysis_plan_v4.json` | `s62_freeze.py` (S62) | Master V4 pre-registration index (SHA-256 registered in `results/frozen_artifacts.json`) |
| `final_verdict_v4.json` | `s63_final_verdict.py` (S63) | Comprehensive final verdicts, unrun accounting, 8 reserved receipts, and open decisions |
| `block3_status.json` | `scripts/run_block3.ps1` | Status tracker for S53 Block 3 multi-seed pooled training runs |
| `recipe_runs/*.json` | `train_v4.py` (S52, S53r) | R0–R7 rung training runs across seeds 42, 43, 44 on HAM and pooled corpora |
| `s53r_plan.json`, `s53r/` | `ladder_rerun.py` (S53r) | Schedule, smoke receipts, and final report for the colour-constancy corrected re-run |
| `s54_plan.json`, `s54/` | `s54_gate.py` (S54) | Representation Gate B evaluation, contrasts, marginals, and reserved receipt |
| `s56_plan.json`, `s56/` | `s56_abstention.py` (S56) | Band-conditional selective abstention frontiers, policies, and reserved receipt |
| `s57b_plan.json`, `lambda_verdict.json` | `lambda_crossfit.py` (S57b) | Continuous $\lambda(\text{age})$ candidate fits, cross-fitting, stability, and verdict (`REJECT-cost`) |
| `s58_plan.json`, `s58/` | `s58_front_end.py` (S58) | Domain-matched front-end heads (H0 vs H1), gate/router checks, and reserved receipt |
| `s59_plan.json`, `s59/` | `s59_cascade.py` (S59) | Composed cascade evaluation on reserved (`S56@0.20`, `CONTRACT_FAILS` verdict, 8th receipt) |
| `s64/ceiling.json`, `frontier_oof.csv` | `ceiling_u40.py` (S64) | Under-40 decision rule ROC frontier and ceiling audit |
| `s65_plan.json`, `s65/` | `s65_combined_policy.py` (S65) | Multi-mechanism combination policy (S55 + $\lambda$ + S56) verdict (`REJECT-cost`) |
| `s66_plan.json`, `s66/` | `s66_lambda_centre.py` (S66) | Per-hospital hierarchical $\lambda$ evaluation and verdict (`REJECT-cost`) |
| `s67/stage1_plan.json`, `probes.json` | `s67_ranking_probe.py` (S67) | Stage-1 ranking probes (specialist, reweighting, metadata; `STAGE2_NO_GO`) |
| `audit/` | `audit_v4.py --emit` | Supplementary audit artifacts (`dedupe_conventional_rule.json`, etc.) |

### V4 Verification Tools

- `python -m research.v4.audit_v4 --check`: Asserts 76 consistency, data-split, receipt, and numerical claims.
- `python -m research.v4.audit_manuscript_v4`: Asserts all 179 claims in `paper/v4/manuscript_v4.tex` against underlying JSON/CSV files with zero tolerance.
- `pytest tests/test_s60_service.py`: Runs 19 automated integration and unit tests for the V4 inference service.

