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
