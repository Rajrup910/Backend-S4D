# results/v6_audit: independent recomputation of V6's HIBA and internal results (12 Nov 2026)

Read-only with respect to V6: these scripts read `results/v6/` and write nothing back there.

- `backout.py`: reconstruct HIBA labels from `results/v6/archive_meta/hiba_175.csv`. 40 rows have a blank `diagnosis_3`; labelling them `vasc` reproduces all four reported Macro-F1s in `results/v6/hiba/hiba_report.json` to 6 decimals.
- `metrics.py`: HIBA Macro-F1, per-class metrics, confusion matrix, and patient- and lesion-bootstrap CIs, written to `hiba_recompute.json`.
- `members.py`: per-member HIBA Macro-F1 (from `results/v6/hiba/inference/*.npy`), raw vs Dirichlet vote, vote vs best member, HIBA ECE. Post hoc.
- `internal_members.py`: the same comparison on folds 1-4 from `results/v6/preds_tta/`. Post hoc.

These were run against copies of the files; paths now resolve relative to this folder. They need numpy, pandas and scikit-learn (`internal_members.py` needs only numpy and pandas).
Labels note: `metrics.py` assigns `vasc` to rows whose `diagnosis_3` is blank. That is valid for this cohort only (see `backout.py`).
