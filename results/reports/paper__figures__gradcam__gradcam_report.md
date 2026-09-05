# Grad-CAM Validation

**Generated:** 2026-09-04  
**Model:** convnext_tiny — `ml/checkpoints/convnext_tiny_best.HAM-only.pt`  
**Target layer:** `Sequential` (last convolutional block)  
**Split:** test  
**Images:** 28

## Method

Gradients of the predicted-class score are taken with respect to the feature maps of
the last convolutional block, averaged spatially into per-channel importance weights,
used to weight and sum those feature maps, passed through ReLU, normalised, and
resized over the original image.

## Where the model looks

`border_mass_fraction` is the share of heatmap activation falling in the outer 15% frame
of the image. Values above **50%** are flagged: lesions in this dataset
are roughly centred, so a border-dominated heatmap suggests the model is responding to
framing artefacts rather than the lesion.

| Statistic | Value |
|---|---:|
| Mean border mass fraction | 0.192 |
| Median border mass fraction | 0.110 |
| Images flagged (> 50%) | **2 / 28** |
| Mean peak offset from centre | 0.435 |
| Degenerate (all-zero) maps | 0 |

## Per class

| Class | Images | Mean border mass | Flagged | Mean confidence |
|---|---:|---:|---:|---:|
| `akiec` | 4 | 0.236 | 0 | 0.940 |
| `bcc` | 4 | 0.266 | 0 | 0.686 |
| `bkl` | 4 | 0.023 | 0 | 0.777 |
| `df` | 4 | 0.089 | 0 | 0.896 |
| `mel` | 4 | 0.294 | 1 | 0.598 |
| `nv` | 4 | 0.190 | 0 | 0.728 |
| `vasc` | 4 | 0.244 | 1 | 0.789 |

## Flagged images — inspect these by eye

| Image | True | Predicted | Confidence | Border mass |
|---|---|---|---:|---:|
| `ISIC_0029502` | mel | df | 0.509 | 0.818 |
| `ISIC_0025244` | vasc | vasc | 0.988 | 0.583 |

## Interpretation limits

- Grad-CAM shows which regions influenced the score. It does **not** show that the model
  used medically correct features, and it must not be presented as proof that it did.
- The map is computed at the last conv layer's resolution (typically 7x7 for a 224x224
  input) and upsampled. The apparent precision of the overlay is interpolation, not
  fine-grained localisation.
- A convincing-looking heatmap on a wrong prediction is common. Always read the heatmap
  together with the predicted class and its confidence.
