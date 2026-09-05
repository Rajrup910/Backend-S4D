# Skin-tone slice of the lesion gate — INVALID, DO NOT QUOTE

**Status: the measurement failed. The fairness question remains unanswered.**

This file previously contained a per-skin-tone table of gate pass rates. Those numbers
have been withdrawn because the skin-tone estimator they were binned by does not measure
skin tone on this dataset. Reproduce with `ml/ood/skin_tone_slice.py` only after reading
the failure analysis below.

## What was attempted

Individual Typology Angle (ITA), the standard image-based proxy for skin tone, computed
on the image border (away from the central lesion) and binned into the conventional
ranges, with the gate's lesion-pass rate reported per bin. Split `test` (n=1376 usable),
never seen by the gate.

## Why it is invalid

HAM10000 was collected in Austria and Australia. That population should contain almost
no dark skin, yet 4.8% of the test split landed in the "dark" ITA bin and a further 4.0%
in "brown". That discrepancy prompted manual inspection of the images the estimator
called darkest:

| Image | Estimated ITA | Bin assigned | What the image actually shows |
|---|---:|---|---|
| `ISIC_0025222` | −72.1 | dark | Light pink skin, brown macule |
| `ISIC_0025160` | −45.5 | dark | Very light/white skin, red vascular lesion |

Both are light skin. The estimator is wrong, not marginally but categorically.

The cause is structural, not a tuning problem. ITA = arctan((L\* − 50) / b\*) presumes
skin colour is carried by a yellow-brown component (b\*). Dermoscopy violates that in two
routine ways:

1. **Vignetting.** The dermatoscope aperture darkens the image corners — exactly the
   border ring being sampled. Low L\* with small b\* produces a large negative angle,
   indistinguishable from dark skin.
2. **Erythema.** Red and pink lesions have high a\* and near-zero b\*. A near-zero
   denominator makes the angle swing wildly on noise, and inflamed skin around a lesion
   extends into the border region.

Both artefacts are endemic to this dataset, so the darker bins are populated by
measurement failure rather than by darker-skinned patients. Any per-bin pass rate
computed on top of them describes nothing.

## What this does and does not tell us

* It does **not** show the gate is fair across skin tones.
* It does **not** show the gate is unfair.
* It **does** show that the aggregate "98.0% of real lesions kept" cannot be
  decomposed by skin tone using this data and this proxy.

## What would actually answer the question

* **Fitzpatrick-labelled data**, not an image-derived proxy — Fitzpatrick17k, or
  PAD-UFES-20 which carries Fitzpatrick labels and is smartphone-captured, matching the
  deployment domain.
* Failing that, a substantially more careful estimator: skin-pixel segmentation rather
  than a fixed border ring, rejection of vignetted regions, and an erythema guard
  (require a meaningful b\* magnitude) so red lesions cannot masquerade as dark skin.

Until one of those exists, the fairness of this component is **unmeasured**, and should
be stated as unmeasured rather than assumed from the aggregate figure.
