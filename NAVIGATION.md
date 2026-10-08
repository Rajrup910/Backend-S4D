# Repository navigation

A fast way to find things in Backend-S4D. Use it alongside [`README.md`](README.md) (what the project
found) and [`CHANGELOG.md`](CHANGELOG.md) (the full session-by-session record).

**Rule of thumb:** code lives in `ml/` (the reusable engine) and `research/` (one package per research
question); every number lives in `results/`; plans and logs live in `docs/`; manuscripts live in `paper/`.

---

## 1 · Thirty-second map

```text
ml/          the engine: data loading, lesion-grouped splits, training, metrics, Grad-CAM
research/    one package per question; v2/ v3/ v4/ v5/ v6/ hold the post-Review-1 programme
results/     every number and figure (JSON / CSV / Markdown), one folder per version
paper/       LaTeX manuscripts, generated tables, figures
docs/        research plans, runsheets and the research log, in time order (see docs/README.md)
scripts/     environment check, dataset download, run launchers, GPU benchmark
tests/       unit and contract tests
```

---

## 2 · "Where is …?"

| I want to see … | Open |
|---|---|
| The headline results in one page | [`README.md`](README.md) |
| What happened in a given session, and why | [`CHANGELOG.md`](CHANGELOG.md) (search the session id, e.g. `S56` or `Q11`) |
| The plans, in the order they were written | [`docs/README.md`](docs/README.md) |
| How the train / val / test split is made and guarded | [`ml/preprocessing/split_dataset.py`](ml/preprocessing/split_dataset.py), frozen split [`ml/configs/splits/split_v1.csv`](ml/configs/splits/split_v1.csv) |
| How a model is trained | [`ml/training/train.py`](ml/training/train.py), shared helpers [`ml/training/common.py`](ml/training/common.py), losses [`ml/training/losses.py`](ml/training/losses.py) |
| How Macro-F1, balanced accuracy and escalation sensitivity are computed | [`ml/evaluation/metrics.py`](ml/evaluation/metrics.py) |
| The six-model ensemble | [`research/ensembling/methods.py`](research/ensembling/methods.py), runner [`research/ensembling/run_ensembling.py`](research/ensembling/run_ensembling.py) |
| 24-view test-time augmentation | [`research/tta/transforms.py`](research/tta/transforms.py), [`research/tta/predict.py`](research/tta/predict.py) |
| Dirichlet calibration | [`research/calibration/methods.py`](research/calibration/methods.py) |
| Abstention ("refer to a doctor when unsure") | [`research/selective/scores.py`](research/selective/scores.py), [`research/selective/risk_coverage.py`](research/selective/risk_coverage.py) |
| Conformal prediction sets | [`research/conformal/calibrate.py`](research/conformal/calibrate.py), [`research/conformal/scores.py`](research/conformal/scores.py) |
| The age-conditional decision rule | [`research/agerule/lambda_rule.py`](research/agerule/lambda_rule.py) |
| Confidence intervals, multiple-comparison families | [`research/stats/intervals.py`](research/stats/intervals.py), [`research/stats/families.py`](research/stats/families.py) |
| Bootstrap, McNemar and DeLong for the ablation ladder | [`research/ablation/bootstrap.py`](research/ablation/bootstrap.py), [`research/ablation/delong.py`](research/ablation/delong.py), [`research/ablation/run_part_a.py`](research/ablation/run_part_a.py) |
| The test-set lock | [`research/testguard.py`](research/testguard.py), receipt [`results/test_pass_receipt.json`](results/test_pass_receipt.json) |
| Every experiment ever run | [`research/experiments.csv`](research/experiments.csv) (append-only ledger) |
| Grad-CAM explanations | [`ml/explainability/gradcam.py`](ml/explainability/gradcam.py), figures [`paper/figures/gradcam/`](paper/figures/gradcam/) |
| The manuscript | [`paper/manuscript.tex`](paper/manuscript.tex) (full), [`paper/manuscript_edited.tex`](paper/manuscript_edited.tex) (condensed), [`paper/v4/`](paper/v4/) (V4 findings and model card) |

---

## 3 · By research version

| Version | Question | Code | Results | Plan / record |
|---|---|---|---|---|
| **V1** (Phases 0–5, S1–S9) | How far do CNNs, ensembling, TTA and calibration go? | `ml/`, [`research/ensembling/`](research/ensembling/), [`research/tta/`](research/tta/), [`research/calibration/`](research/calibration/), [`research/selective/`](research/selective/), [`research/conformal/`](research/conformal/), [`research/agerule/`](research/agerule/), [`research/stats/`](research/stats/), [`research/ablation/`](research/ablation/) | [`results/ablation_table.csv`](results/ablation_table.csv), [`results/mcnemar_delong.json`](results/mcnemar_delong.json), [`results/session9/`](results/session9/), [`results/reports/`](results/reports/) | [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md), [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) |
| **V1 external** (S12–S17) | Does V1 hold on other hospitals? | [`research/external/`](research/external/), [`research/xdomain/`](research/xdomain/), [`scripts/external/`](scripts/external/) | [`results/external/`](results/external/) | [`CHANGELOG.md`](CHANGELOG.md) |
| **V2** (S28–S39) | Is the under-40 miss a threshold problem? | [`research/v2/`](research/v2/) (`decompose.py`, `conformal_safety.py`, `transport.py`, `gate.py`) | [`results/v2/`](results/v2/) — verdict [`final_verdict.json`](results/v2/final_verdict.json) | [`CHANGELOG.md`](CHANGELOG.md) S28–S39 |
| **V3** (S40–S47) | Can more archives, a better head or removing age fix it? | [`research/v3/`](research/v3/) (`oos_probe.py`, `probes.py`, `eval_conditions.py`, `age_invariant.py`) | [`results/v3/`](results/v3/) — verdict [`final_verdict.json`](results/v3/final_verdict.json) | [`results/v3/analysis_plan_v3.json`](results/v3/analysis_plan_v3.json) |
| **V4** (S48–S75) | With a powered endpoint, does any lever work? | [`research/v4/`](research/v4/) (`power.py`, `build_corpus.py`, `recipe.py`, `train_v4.py`, `s56_abstention.py`, `s59_cascade.py`, `s71_kfold.py`) | [`results/v4/`](results/v4/) — verdict [`final_verdict_v4.json`](results/v4/final_verdict_v4.json) | [`results/v4/analysis_plan_v4.json`](results/v4/analysis_plan_v4.json), [`paper/v4/v4_findings.md`](paper/v4/v4_findings.md) |
| **V5** (Q0–Q11, E14) | Do dermatologist-style modules help young patients? | [`research/v5/`](research/v5/) (`train_v5.py`, `arms.py`, `modules.py`, `screen_gate.py`, `falsifiers.py`, `confirmation.py`, `q11_read.py`) | [`results/v5/`](results/v5/) — confirmation [`confirm_s42_s43_s44.json`](results/v5/confirm_s42_s43_s44.json), tables [`review2_tables.md`](results/v5/review2_tables.md) | [`docs/V5_RUNSHEET.md`](docs/V5_RUNSHEET.md), [`docs/v5_design/`](docs/v5_design/), [`docs/v5_record/`](docs/v5_record/) |
| **V6** (planned) | Counterexample-guided differential morphology (CG-DM) | [`research/v6/`](research/v6/) (pre-checks only) | [`results/v6/`](results/v6/) | [`docs/V6_RUNSHEET.md`](docs/V6_RUNSHEET.md) |

---

## 4 · Where each headline number lives

| Claim | Value | File |
|---|---|---|
| Best single CNN (ConvNeXt-Tiny), test Macro-F1 | 0.7459 | [`results/ablation_table.csv`](results/ablation_table.csv) (row A2) |
| Six-CNN soft vote → + TTA → + Dirichlet | 0.7718 → 0.7859 → 0.8047 | [`results/ablation_table.csv`](results/ablation_table.csv) (rows A5–A7) |
| Ensembling vs best single model, McNemar | p = 3.4 × 10⁻⁵ | [`results/mcnemar_delong.json`](results/mcnemar_delong.json) |
| Calibration lost 16 escalations, gained 0 | exact p = 3.1 × 10⁻⁵ | [`results/review2/escalation_mcnemar_a6_a7.json`](results/review2/escalation_mcnemar_a6_a7.json) |
| Under-40 escalation sensitivity on test | 3 of 21 (0.143) | [`research/selective/results/session4_report.md`](research/selective/results/session4_report.md) |
| Training prior: escalating share under 40 vs 60+ | 4.9% vs 35.5% | [`results/age_band_prior.csv`](results/age_band_prior.csv) |
| Bipartite conformal: under-40 false reassurance | 6/29 → 1/29 | [`results/v2/frr_by_group.csv`](results/v2/frr_by_group.csv) |
| Removing age from the model: under-40 AUC | 0.8249 → 0.7201 | [`results/v3/d2_strong_evaluation.json`](results/v3/d2_strong_evaluation.json) |
| "3 of 21" is 10 lesions; lesions needed | 104 single-arm, 81 paired | [`results/v4/power_audit.json`](results/v4/power_audit.json) |
| Per-age-band abstention, under-40 gain | +0.154 [+0.104, +0.216] | [`results/v4/s56/contrasts_reserved.csv`](results/v4/s56/contrasts_reserved.csv) |
| V5 under-40 ranking gain (3 seeds) | +0.0351 [+0.0125, +0.0596] | [`results/v5/confirm_s42_s43_s44.json`](results/v5/confirm_s42_s43_s44.json) |
| Share from modules + young data | +0.0270 (77%) | [`results/v5/q11_decomposition.json`](results/v5/q11_decomposition.json) |
| HAM test split reads | 2 (one plan, 4 Sep) | [`results/test_pass_receipt.json`](results/test_pass_receipt.json) |

---

## 5 · Verify the record

```bash
python -m research.ablation.audit_manuscript      # 366 manuscript claims vs results/
python -m research.v4.audit_v4 --check            # 76 V4 splits, receipts and claims
python -m research.v5.adopt_e1 --verify           # V5 pre-registration hashes
pytest tests -q                                   # unit and contract tests
```

None of these read the test split.

---

## 6 · Large files (don't open in a browser)

Prediction matrices hold one row per image and are several MB each:
[`research/predictions/`](research/predictions/), [`research/predictions_tta/`](research/predictions_tta/),
[`research/predictions_oof/`](research/predictions_oof/), [`results/external/predictions/`](results/external/predictions/),
[`results/v2/panels/`](results/v2/panels/), [`results/v4/kfold/`](results/v4/kfold/) and `results/v5/preds/`.
Read the summary files beside them instead. Checkpoints and image datasets are not in the repository.

---

## 7 · Glossary

| Term | Meaning |
|---|---|
| **Escalating classes** | Melanoma (MEL), basal-cell carcinoma (BCC), actinic keratosis (AKIEC): lesions that must go to a specialist |
| **Escalation sensitivity** | Share of escalating lesions the system sends to a specialist |
| **Macro-F1** | F1 averaged equally over the 7 classes, so rare cancers count as much as common moles |
| **pAUC** | Partial area under the ROC curve: how well the model *ranks* dangerous lesions above harmless ones |
| **TTA** | Test-time augmentation: average the prediction over 24 flipped / rotated / rescaled views |
| **Dirichlet calibration** | A per-class correction that makes predicted confidence match real accuracy |
| **OOF** | Out-of-fold: predictions on training images from models that never saw them |
| **MCID** | Minimum clinically important difference, set before each read |
| **Gate / falsifier** | A pass/fail rule, and a test that could prove an idea wrong, both written before the result |
| **HAM / BCN / MSKCC / PAD** | HAM10000, BCN20000, MSKCC (ISIC Archive) dermoscopy; PAD-UFES-20 smartphone photos |
| **Reserved cohort** | 4,733 BCN + MSKCC images held out in V4 for one-time reads |
| **MILK10k, HIBA** | External cohorts declared for future one-time confirmation reads |
| **S1 … S75** | Sessions V1–V4: S1–S9 V1 research, S10–S27 V1 manuscript and external validation, S28–S39 V2, S40–S47 V3, S48–S75 V4 |
| **Q0 … Q11, E14** | V5 GPU queue steps and the confirmation read |
| **CG-DM** | Counterexample-guided differential morphology, the proposed V6 mechanism |

---

## 8 · Search tips

```bash
git grep -n "def assert_no_leakage"          # find a function
git grep -n "S56" -- CHANGELOG.md             # every mention of a session
git grep -l "confirm_s42_s43_s44"             # who reads or writes a result file
```

Every CHANGELOG heading carries its session id and date, so searching `### S54` or `### Q11` jumps straight to it.
