"""V7 data foundation (`results/leak_audit/V7_DATA_PLAN.md`): id normalisation, fingerprints, planted copies,
typed grouping and the overlap guard that every V7 entry point must pass before it reads a row.

Nothing here edits a V6 file or recomputes a frozen split. Spent sets (HAM test, reserved, HIBA) stay fixed: when a
link joins training and evaluation, the training-side row is removed.
"""
