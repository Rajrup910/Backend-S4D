"""Cross-domain (HAM10000 -> PAD-UFES-20) external evaluation, S8b.

The six frozen HAM-only CNNs, their uniform soft-vote ensemble, the frozen OOF
Dirichlet map and the frozen age-conditional lambda rule, all applied *inference-only*
to PAD-UFES-20 -- a smartphone clinical cohort, a genuine dermoscopy -> clinical
distribution shift. Nothing here trains, and nothing here reads the HAM10000 test split
(Hard Rule 2): PAD is external, so every PAD image is legitimately scoreable, and the
in-distribution reference for the Mahalanobis shift test is HAM *val*, not HAM test.
"""
