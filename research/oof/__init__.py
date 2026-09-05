"""K-fold out-of-fold predictions over the training split, and the machinery around them.

`make_folds` assigns lesion-grouped stratified folds, `train_folds` trains the six
architectures on each, `extract_oof` assembles the per-fold predictions into
`research/predictions_oof{,_tta}/{arch}_train.csv`, and `diagnose_shift` measures the
stacking mismatch those OOF scores carry against the frozen full-train models.

`run_comparison` then runs each downstream runner twice -- val-fitted and OOF-fitted --
and tabulates the difference, while `check_valfit_regression` proves the val-fitted path
still computes exactly what it computed before the fit-split rewiring. Neither reads test.
"""
