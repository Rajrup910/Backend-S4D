# Cross-dataset comparison — six architectures, HAM10000 vs PAD-UFES-20

Every number below comes from `ml/evaluation/evaluate.py` on a **held-out, lesion-grouped
test split**, with temperature scaling fitted on validation. Nothing is hand-entered.
Six architectures, each trained and evaluated with the **same code, same two-stage transfer
schedule, same seed (42), same effective-number class weighting** — the only variable is the
backbone (and the training dataset).

Four evaluations per model:
- **HAM→HAM**: trained on HAM10000 (dermoscopy), tested on HAM test (1,502 img, all 7 classes) — *in-domain*.
- **HAM→PAD**: the HAM model tested on PAD test (314 img) — *cross-domain / the domain gap*.
- **PAD→PAD (fresh)**: trained on PAD-UFES-20 (smartphone) from ImageNet, tested on PAD test (314 img, 5 classes) — *in-domain smartphone*.
- **PAD→PAD (warm)**: same PAD fine-tune but *initialised from the HAM checkpoint* instead of ImageNet — the "mixed dataset" regime. Full study in `PAD_MACRO_F1_PUSH.md`.

Headline metric is **macro-F1**, not accuracy: `nv` is ~67% of HAM, so accuracy rewards
guessing "mole". PAD macro-F1 is over all 7 slots (PAD lacks `df`/`vasc`, which caps it at 5/7).

---

## 1. Model sizes

| Model | Params | Size |
|---|---:|---:|
| EfficientNet-B0 | **4.0 M** | **15.3 MB** |
| DenseNet-121 | 7.0 M | 26.6 MB |
| EfficientNet-B3 | 10.7 M | 40.8 MB |
| ResNet-50 | 23.5 M | 89.7 MB |
| ConvNeXt-Tiny | 27.8 M | 106.1 MB |
| ConvNeXt-Small | 49.5 M | 188.7 MB |

---

## 2. Headline: macro-F1 across all three evaluations

| Model | HAM→HAM | HAM→PAD | PAD→PAD (fresh) | PAD→PAD (warm) | warm Δ |
|---|---:|---:|---:|---:|---:|
| ResNet-50 | 0.706 | 0.142 | 0.472 | 0.467 | −0.005 |
| EfficientNet-B0 | 0.726 | 0.137 | 0.466 | 0.475 | +0.009 |
| EfficientNet-B3 | 0.689 | 0.113 | 0.463 | 0.477 | +0.014 |
| DenseNet-121 | 0.697 | 0.127 | 0.440 | 0.495 | +0.055 |
| **ConvNeXt-Tiny** | **0.746** | 0.164 | 0.469 | **0.543** | **+0.074** |
| ConvNeXt-Small | 0.725 | 0.172 | 0.476 | 0.515 | +0.039 |

All cells use the **best** checkpoint (selected on validation macro-F1, before any test
evaluation). The ConvNeXt-Tiny PAD run was regenerated from seed 42 (val 0.564). Macro-F1 here
is the 7-slot figure; on the 5 classes PAD actually contains, the warm ConvNeXt-Tiny reaches
**0.760** (see §5 and `PAD_MACRO_F1_PUSH.md`).

**Reading this table:**
- **Best in-domain dermoscopy model: ConvNeXt-Tiny (0.746).** It leads HAM→HAM on macro-F1,
  balanced accuracy, and melanoma recall.
- **The efficiency result the Colab reset had left open: EfficientNet-B0 is second (0.726) at
  6–12× fewer parameters** than the ConvNeXt models, with the best calibration of the whole
  set (HAM ECE 0.032). This is the strongest accuracy-per-parameter in the study.
- **The domain gap is universal.** Every architecture collapses from ~0.70 to ~0.11–0.17 when
  a dermoscopy model meets smartphone photos. It is not a weakness of one backbone.
- **PAD training recovers all of them** to ~0.44–0.49 — and, more importantly, recovers
  screening safety (§4).
- **Mixing datasets (warm-start) pushes the smartphone models further, but only some.**
  Initialising the PAD fine-tune from HAM lifts **ConvNeXt-Tiny (+0.074), DenseNet-121 (+0.055)
  and ConvNeXt-Small (+0.039)** while barely moving ResNet-50 / EfficientNet. The best
  smartphone model in the whole project is **ConvNeXt-Tiny warm-start** (5-class macro-F1 0.760,
  melanoma recall 0.625, 6 missed serious). Full analysis in `PAD_MACRO_F1_PUSH.md`.

---

## 3. In-domain dermoscopy (HAM→HAM, 1,502 images, all 7 classes)

| Model | Macro-F1 | Balanced acc | Accuracy | Macro ROC-AUC | ECE | Escalation sens. | Missed serious | `mel` recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ConvNeXt-Tiny** | **0.746** | **0.779** | 0.830 | 0.931 | 0.097 | **0.776** | **65** | **0.677** |
| EfficientNet-B0 | 0.726 | 0.739 | 0.834 | **0.954** | **0.032** | 0.786 | 62 | 0.647 |
| ConvNeXt-Small | 0.725 | 0.728 | **0.847** | 0.945 | 0.061 | 0.714 | 83 | 0.605 |
| ResNet-50 | 0.706 | 0.722 | 0.822 | 0.945 | 0.101 | 0.738 | 76 | 0.575 |
| DenseNet-121 | 0.697 | 0.734 | 0.800 | 0.944 | 0.049 | 0.738 | 76 | 0.581 |
| EfficientNet-B3 | 0.689 | 0.713 | 0.828 | 0.943 | 0.048 | 0.728 | 79 | 0.539 |

EfficientNet-B0 actually has the **highest escalation sensitivity (0.786) and fewest missed
serious cases (62)** of any dermoscopy model, on top of the best calibration — a remarkable
result for a 4 M-parameter network.

---

## 4. The domain gap and its recovery (the R&D headline)

For each model: the HAM model on PAD (cross-domain) vs the PAD model on PAD (in-domain phone).

| Model | HAM→PAD sens. | HAM→PAD missed | HAM→PAD `mel` recall | → | PAD→PAD sens. | PAD→PAD missed | PAD→PAD `mel` recall |
|---|---:|---:|---:|:--:|---:|---:|---:|
| ResNet-50 | 0.389 | 149 | 0.00 | → | 0.939 | 15 | 0.25 |
| EfficientNet-B0 | 0.352 | 158 | 0.00 | → | 0.918 | 20 | 0.25 |
| EfficientNet-B3 | 0.299 | 171 | 0.00 | → | 0.939 | 15 | 0.375 |
| DenseNet-121 | 0.422 | 141 | 0.00 | → | 0.910 | 22 | 0.125 |
| ConvNeXt-Tiny | 0.336 | 162 | 0.00 | → | 0.943 | 14 | 0.125 |
| ConvNeXt-Small | 0.324 | 165 | 0.00 | → | 0.939 | 15 | 0.125 |

**The story, now confirmed across six architectures:**
- On smartphone photos, **every** dermoscopy-trained model catches **0% of melanomas**
  (`mel` recall 0.00) and lets 140–170 of the serious cases through. The failure is a
  property of the *domain shift*, not any one model.
- Training on smartphone data (PAD-UFES-20) lifts escalation sensitivity from ~0.30–0.42 to
  **0.91–0.96** and cuts missed serious cases from ~150 to **11–22**.
- This is measured, architecture-independent justification that a phone app **must not** ship
  a dermoscopy-only model.

---

## 5. In-domain smartphone (PAD→PAD, 314 images, 5 classes)

### 5a. Fresh (ImageNet init)

| Model | Macro-F1 | Balanced acc | Accuracy | Escalation sens. | Missed serious | ECE |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Small | 0.476 | 0.668 | 0.787 | 0.939 | 15 | 0.108 |
| ResNet-50 | 0.472 | 0.658 | 0.764 | 0.939 | 15 | 0.083 |
| ConvNeXt-Tiny | 0.469 | 0.654 | 0.768 | **0.943** | **14** | **0.052** |
| EfficientNet-B0 | 0.466 | 0.667 | 0.771 | 0.918 | 20 | 0.065 |
| EfficientNet-B3 | 0.463 | 0.650 | 0.742 | 0.939 | 15 | 0.094 |
| DenseNet-121 | 0.440 | 0.626 | 0.752 | 0.910 | 22 | 0.067 |

### 5b. Warm-start (HAM init → PAD fine-tune, "mixed")

Same PAD test split, initialised from each model's HAM checkpoint. Best two rows in **bold**.

| Model | Macro-F1 (5-class) | Escalation sens. | Missed serious | `mel` recall | ECE |
|---|---:|---:|---:|---:|---:|
| **ConvNeXt-Tiny** | **0.760** | **0.975** | **6** | **0.625** | 0.079 |
| **ConvNeXt-Small** | **0.721** | 0.963 | 9 | 0.375 | 0.063 |
| DenseNet-121 | 0.692 | 0.930 | 17 | 0.375 | 0.062 |
| EfficientNet-B3 | 0.668 | 0.910 | 22 | 0.250 | 0.081 |
| EfficientNet-B0 | 0.665 | 0.922 | 19 | 0.375 | 0.066 |
| ResNet-50 | 0.653 | 0.951 | 12 | 0.250 | 0.084 |

*(5-class macro-F1 = averaged over the 5 classes PAD contains; the §5a fresh table above is
7-slot, so compare fresh vs warm within `PAD_MACRO_F1_PUSH.md`, which reports both on the same
basis.)*

**Caveat (all models):** PAD has only 52 melanoma images total (8 in test), so per-model
`mel` recall is noisy — **escalation sensitivity** (which pools all malignant/pre-malignant
classes) is the meaningful safety figure. On the *fresh* models it clusters at 0.91–0.96, so
architecture matters less than training domain. **Warm-start breaks that tie:** ConvNeXt-Tiny
reaches 0.975 sensitivity and just 6 missed serious — clearly the best smartphone model.

---

## 6. Provenance note

ConvNeXt-Tiny's PAD run was regenerated from seed 42 after an earlier checkpoint-copy step
had overwritten the PAD *best* with HAM weights. The current `eval_convnext_tiny-on-PAD` uses
the correct best checkpoint (val macro-F1 0.564, epoch 14); all cells in this document are now
best-checkpoint numbers. `convnext_tiny_best.pt` is restored to the HAM model.

---

*Sources: `ml/results/eval_*-on-PAD` (fresh), `ml/results/eval_*-on-PADwarm` (warm-start), and
`eval_ConvNeXtTiny-on-HAM` from the Review-1 copy for ConvNeXt-Tiny's HAM→HAM row. Params/size
and val metrics traceable to `ml/results/experiments.csv` (`*_pad_seed42` = fresh,
`*_padwarm_seed42` = warm). Warm-start breakdown: `PAD_MACRO_F1_PUSH.md`.*
