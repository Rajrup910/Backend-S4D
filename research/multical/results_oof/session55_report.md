# Session 55 -- group-conditional (per-age-band) Dirichlet calibration

Fit split: OOF (research/predictions_oof_tta, 6,981 rows). No test read.

Per-band maps fitted for **40-59, 60+, <40**; **unknown** fell back to the global map (n < 300).

## Aggregate (ALL rows) -- does the fix survive being averaged over?

            source  signed_gap      ece
      uncalibrated   -0.171847 0.172146
  dirichlet_global    0.011521 0.024602
dirichlet_per_band    0.010414 0.018979

## The two bands the global map pushed in opposite directions

### <40
- uncalibrated: signed_gap=-0.2290 [-0.2459, -0.2128], ece=0.2290
- dirichlet_global: signed_gap=-0.0269 [-0.0424, -0.0117], ece=0.0276
- dirichlet_per_band: signed_gap=0.0070 [-0.0070, 0.0212], ece=0.0095

### 60+
- uncalibrated: signed_gap=-0.1191 [-0.1367, -0.1008], ece=0.1201
- dirichlet_global: signed_gap=0.0410 [0.0239, 0.0588], ece=0.0410
- dirichlet_per_band: signed_gap=0.0154 [-0.0016, 0.0325], ece=0.0301

## Reading

The per-band arm is cross-fitted the same way the global arm is (K-fold within band, or the global cross-fit for an under-powered band), so none of the numbers above are in-sample. The deployed per-band maps (fit on the whole band, no cross-fitting) are written to `fit_state.json` for a later session to apply to a split this fit never saw.

**The sign-flip closes**: the max-min spread of the *signed* gap across bands drops 0.0679 -> 0.0084, and every band's own ECE improves individually (<40 0.0276 -> 0.0095, 60+ 0.0410 -> 0.0301).

**But the ECE *spread* across bands does not shrink** (0.0150 -> 0.0206, wider not narrower): <40 improves by far the most (its map had the most sign-flip room to close), so 60+ -- barely moved -- is left as the worst-calibrated band by a wider margin than before. Report both numbers: per-band calibration fixes the *direction* disagreement between bands, not the *magnitude* disagreement, and 60+ remains the band this project's calibration story is weakest on.

