"""Session 9 -- the frozen analysis plan and the single pre-registered test pass.

Everything the manuscript reports from the test split is emitted by one script, driven by
one file (`results/analysis_plan.json`) that is written and hashed **before** that script
is allowed to run. Nothing in this package may compute a test-split quantity that the plan
does not name.
"""
