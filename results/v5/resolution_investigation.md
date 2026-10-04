# Why did 384 px not beat 224 px in V5? — investigation (2026-10-04)

Owner's question: higher resolution should obviously help dermoscopy; V5's S01 said it did not. Is something wrong?
Every number is read from the file named. Nothing was trained for this report.

## 1. Ruled out: the pipeline
| Suspect | Check | Result |
|---|---|---|
| Images pre-shrunk on disk | 40 random development images per archive opened from `data/external/isic2019_images/ISIC_2019_Training_Input` | **Full resolution**: HAM 600×450, BCN20000 1024×1024, MSKCC ~1024×680 (919–1024 wide) |
| R1 changed more than resolution | `research/v4/recipe.py` rungs | R1 = `Recipe(image_size=384)` only; batch 16 × accum 2 = effective 32 (ConvNeXt has no batch norm) |
| Eval geometry differs | `recipe.py` eval transform | Resize to 256/224 × size, centre-crop: the same 87.5% field of view at both sizes |
| Train augmentation differs | `RandomResizedCrop(size, scale 0.8–1.0, ratio 0.9–1.11)` | identical at both sizes |

**No bug.** The images reach the model at the resolution asked for.

## 2. What the data actually say
**Resolution helps on HAM-type images, and has before.** V4 S53r (HAM-only training, HAM val, 3 seeds): R1 384 px
**+0.0297 Macro-F1, every seed positive** (+0.0097 to +0.0466); seed 42 showed a stable plateau, not a spike
(CHANGELOG, V4 R1 / S53r readouts; `results/v4/s53r/s53r_report.json`).

**On the pooled corpus the gain is confined to HAM rows** (S01, fold 0, 384 − 224, last epoch, per archive; from
`results/v5/preds/R1_kfold_f0_s4{2,3,4}_v5s01_last.csv` vs the R0 files):
| Archive (native size) | Δ Macro-F1 s42 / s43 / s44 | Δ balanced acc. | Δ escalation pAUC |
|---|---|---|---|
| HAM (600×450) | +0.008 / +0.006 / **+0.030** — 3/3 positive | +0.003 / +0.011 / +0.028 | −0.001 / +0.013 / +0.012 |
| BCN20000 (1024×1024) | −0.007 / +0.021 / −0.004 | −0.018 / +0.035 / −0.014 | −0.011 / −0.004 / +0.028 |
| MSKCC (~1024×680) | +0.019 / +0.003 / +0.006 | +0.039 / −0.002 / +0.006 | +0.017 / −0.004 / +0.019 |
| **All rows** | −0.005 / +0.011 / +0.005 (mean +0.004) | −0.007 / +0.016 / +0.004 | −0.009 / −0.001 / +0.014 |

BCN is 45% of fold 0 and is exactly where a resolution gain should be largest (4.0× downsampling to 256 vs 2.3× to
439) — yet it shows none. The pooled null is a **HAM gain diluted by a BCN non-gain**, not "resolution does nothing".

## 3. Why S01 could not see it — measurement, not biology
- **Underpowered by design.** Fold-0 Macro-F1 has a seed-pair SD of 0.016 (`results/v5/screens/noise_floor.json`);
  the observed per-seed Δ spread is ≈ 0.008. Three seeds on one fold cannot resolve a +0.005–0.010 effect, and the S01
  rule asked for a mean ≥ +0.015 **and** a CI above zero.
- **Last-epoch noise.** Validation Macro-F1 moves ≈ 0.006 per epoch at the end of training (median over the S01 runs),
  and every S01 run peaked at epochs 12–25 and finished 0.01–0.015 below its best (`results/v4/kfold/runs/*_v5s01.json`
  histories). The `_last` primary (Amendment 01) is the right guard against selection, but it adds noise of the
  same size as the effect.
- **Pretraining mismatch.** S01 fine-tuned 224-pretrained IN-1k weights at 384. IN-22k weights pretrained *at* 384
  (`convnext_tiny.fb_in22k_ft_in1k_384`) exist and were never tried at 384.

## 4. Why BCN might not gain (hypotheses, not findings)
1. BCN's hard cases are "in the wild" (nails, mucosa, hypopigmented, non-segmentable lesions — BCN20000 paper): the
   errors are about context and lesion type, not fine pigment structure.
2. BCN 1024² frames contain the dermatoscope vignette; at 384 more pixels go to the frame, and the lesion often fills a
   smaller part of the image than in HAM.
3. BCN labels include more diagnostically ambiguous lesions; resolution cannot fix label ambiguity.
Each is testable on existing predictions plus lesion masks (lesion-area fraction by archive; Δ by lesion size).

## 5. What this means for the module handicaps
The biology modules read 14×14 stride-16 tokens of a 224 px image. Q5's "0 young structure tokens" was measured there.
Running them at 384 (24×24 tokens) was planned only through the AU20 rescue, which needed S01 to pass. So the
dermatological modules were **never tested at a resolution where fine structure survives**, and HAM-type images —
where resolution demonstrably helps — are under half the pooled data. That is a real gap in V5, not a refutation.

## 6. Verdict
Nothing is broken. Resolution helps on HAM-type images (V4: +0.030, 3/3 seeds; V5 HAM rows: +0.015, 3/3 seeds), not on
BCN, and S01 had neither the power nor the per-archive endpoint to see it. The V6 response is in
`docs/V6_RUNSHEET.md` §A13.
