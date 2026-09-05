# OOD negative set

Curated from `data/ood_negatives` by `ml/ood/prepare_negatives.py` (seed 20260809).

Raw files with an image extension: 16066. Dropped 1817 (1817 mask/junk directory). Usable photographs: 14249.

| Source | Available | Selected | Role |
|---|---:|---:|---|
| Body Parts Dataset | 2092 | 2092 | near-OOD (skin/body) |
| Fasseg-DB-v2019 | 681 | 681 | near-OOD (skin/body) |
| Hand images | 408 | 300 | near-OOD (skin/body) |
| img_files | 11065 | 600 | far-OOD (scenes/objects) |
| phone_camera | 3 | 3 | near-OOD (skin/body) |
| **Total** | 14249 | **3676** | |

near-OOD 3076 / far-OOD 600.

<!-- analysis: preserved across regeneration -->

The three `phone_camera` images were captured on the deployment handset. On disk they are
written into the `Body_Parts_Dataset` folder rather than a directory of their own; they are
counted separately here because this table is the provenance record.

## What was dropped, and why it mattered

The 1817 dropped files were not merely redundant:

* **`__MACOSX/` AppleDouble stubs** (176-byte files carrying `.jpg`/`.png` extensions).
  These fail to decode, so they would have been skipped with warnings -- noisy, not harmful.
* **FASSEG segmentation masks** (`Labeled/`, `Test_Labels/`, `Train_labels/`). These are the
  dangerous ones: flat synthetic colour fields that decode perfectly well. Nothing would have
  failed. The gate would simply have learned, in part, that cartoon colour blocks are not
  lesions -- a boundary that transfers to nothing a user would ever photograph.

**COCO was capped 11065 -> 600.** Uncapped it is 78% of the usable pool. Since it represents
the *easy* far-OOD case (scenes, objects), leaving it uncapped would have let it dominate the
decision boundary while the case the gate exists for -- skin that is not a lesion -- was
outvoted roughly four to one.

`--neg-views` was set to 2 rather than the script default of 6. That default is documented for
"a small hand-collected set"; with 3676 genuine negatives, six correlated crops per image pads
the negative side with near-duplicates rather than adding information.

## Fitted gate

`ml/checkpoints/lesion_gate.npz`, from `resnet50_best.pt` (epoch 16), 2048-d features.

| Quantity | Value |
|---|---|
| Negatives (distinct images) | 3676 |
| Training vectors | 8882 (3000 pos / 5882 neg) |
| P(lesion), val lesions | mean 0.999 |
| P(lesion), val negatives | mean 0.004 |
| Threshold | 0.996 |
| Real lesions kept | 98.0% |
| Held-out negatives rejected | 100.0% |

## Validation beyond the fit script's own numbers

`fit_lesion_gate.py` holds out 20% of negatives *at random*, so every source appears in both
train and validation. That measures unseen **images**, not unseen **kinds of thing**, and a
100% reject rate under those conditions is weak evidence. Two independent checks were run with
`ml/ood/eval_lesion_gate.py`:

| Test | Result |
|---|---|
| 800 COCO scenes (far-OOD) | 800/800 rejected; max P(lesion) 0.33 vs threshold 0.996 |
| **Source-holdout**: gate retrained with FASSEG faces removed entirely, then shown 681 faces | **681/681 rejected**; max P(lesion) 0.65 vs threshold 0.994 |

The second test is the meaningful one: a gate that had never seen a human face rejected every
face it was shown, with a wide margin. The 100% figure therefore reflects genuine separation in
feature space rather than memorisation of the training sources.

### Known limitations of this negative set

* **Skin-tone coverage is unverified**, and the attempt to verify it failed -- see
  `ml/results/skin_tone_slice.md`. The fairness of this component is unmeasured, not
  established.
* Only 3 images come from the deployment camera. Smartphone-domain negatives are the scarcest
  and most valuable category; more would strengthen the gate where it matters most.
* The lesion positives are dermoscopy (HAM10000), so "real lesion kept" is measured on
  dermoscopy, not phone photos.
