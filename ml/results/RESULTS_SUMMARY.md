# ResNet-50 results — HAM10000 vs PAD-UFES-20 (trained separately)

Generated 2026-08-09. Every number below comes from `ml/evaluation/evaluate.py` on a
**held-out, lesion-grouped test split**. Nothing here is a validation number quoted as a
test number, and nothing is hand-entered.

Two ResNet-50 models were trained independently on the same code, same two-stage transfer
schedule (frozen head → fine-tune), same seed (42), same augmentation:

| Model | Train data | Checkpoint | Classes present |
|---|---|---|---|
| **HAM model** | HAM10000 (dermoscopy) | `ml/checkpoints/resnet50_best.HAM-only.pt` | 7 (all) |
| **PAD model** | PAD-UFES-20 (smartphone) | `ml/checkpoints/resnet50_best.PAD-only.pt` | 5 (`df`, `vasc` absent in PAD) |

The **deployed** backend checkpoint `resnet50_best.pt` is the **HAM model** (7-class, the
complete label space). The PAD model is kept as a separate artifact for the domain-gap
analysis below.

---

## 1. Headline: the dermoscopy→smartphone domain gap

The same HAM model that scores well on dermoscopy **collapses** on smartphone photos; a
model trained on smartphone photos recovers most of that.

| Evaluation | Test set | Macro-F1 | Balanced acc | Accuracy | Escalation sensitivity | Missed serious |
|---|---|---:|---:|---:|---:|---:|
| HAM model on **HAM** (in-domain) | 1502 img | **0.706** | 0.722 | 0.822 | 0.738 | 76 |
| HAM model on **PAD** (cross-domain) | 314 img | **0.142** | 0.270 | 0.245 | 0.389 | 149 |
| PAD model on **PAD** (in-domain) | 314 img | **0.472** | 0.658 | 0.764 | **0.939** | 15 |

Macro-F1 above is computed over all 7 class slots (the two classes PAD lacks, `df`/`vasc`,
count as 0, which mechanically caps PAD's number at 5/7). On the **5 classes the two
datasets share**, the gap is even clearer:

| On the 5 shared classes | Macro-F1 |
|---|---:|
| HAM model on PAD | 0.199 |
| **PAD model on PAD** | **0.661** |

**Takeaway for the demo:** a dermoscopy-trained classifier is not safe to point at phone
photos — melanoma recall on PAD drops to **0.00** and escalation sensitivity to 0.39. This
is the concrete justification for either a PAD-specific model or domain adaptation before
any smartphone deployment.

---

## 2. HAM model on HAM test (in-domain, the deployed model)

`ml/results/eval_HAM-on-HAM/` — 1502 images, all 7 classes.

| Metric | Test |
|---|---:|
| Macro-F1 | 0.706 |
| Balanced accuracy | 0.722 |
| Accuracy | 0.822 |
| Weighted F1 | 0.823 |
| Cohen's κ | 0.664 |
| Macro ROC-AUC | 0.945 |
| Expected calibration error | 0.101 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `akiec` | 0.481 | 0.750 | 0.587 | 52 |
| `bcc`   | 0.709 | 0.789 | 0.747 | 71 |
| `bkl`   | 0.689 | 0.611 | 0.648 | 167 |
| `df`    | 0.762 | 0.800 | 0.780 | 20 |
| `mel`   | 0.589 | 0.575 | 0.582 | 167 |
| `nv`    | 0.918 | 0.908 | 0.913 | 1004 |
| `vasc`  | 0.765 | 0.619 | 0.684 | 21 |

Confusion matrix: `ml/results/eval_HAM-on-HAM/confusion_matrix.png`

---

## 3. PAD model on PAD test (in-domain smartphone)

`ml/results/eval_PAD-on-PAD/` — 314 images, 5 classes present.

| Metric | Test |
|---|---:|
| Macro-F1 (7-slot) | 0.472 |
| Macro-F1 (5 present classes) | 0.661 |
| Balanced accuracy | 0.658 |
| Accuracy | 0.764 |
| Weighted F1 | 0.762 |
| Cohen's κ | 0.660 |
| Expected calibration error | 0.083 |
| Escalation sensitivity | 0.939 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `akiec` | 0.777 | 0.813 | 0.795 | 107 |
| `bcc`   | 0.833 | 0.775 | 0.803 | 129 |
| `bkl`   | 0.688 | 0.647 | 0.667 | 34 |
| `mel`   | 0.500 | 0.250 | 0.333 | 8 |
| `nv`    | 0.630 | 0.806 | 0.707 | 36 |

Confusion matrix: `ml/results/eval_PAD-on-PAD/confusion_matrix.png`

**Caveat:** PAD has only 52 melanoma images total (8 in test), so `mel` recall (0.25 = 2/8)
carries very wide uncertainty. The escalation-sensitivity number (0.939) is the more
meaningful safety figure since it pools all malignant/pre-malignant classes.

---

## 4. PAD-UFES-20 ingestion summary

- Source: Mendeley `zr7vgbcyr2` (PAD-UFES-20), all 3 image parts + `metadata.csv`.
- 2298 metadata rows → **2106 usable** after dropping 192 `SCC` (no honest HAM equivalent).
- Class map: `BCC→bcc, MEL→mel, NEV→nv, ACK→akiec, SEK→bkl`.
- Leakage grouping: by **lesion** (`padles_<patient>_<lesion>`), 1746 groups; leakage check
  passed (no lesion crosses a split). Split 70/15/15 → 1474 / 318 / 314 images.
- Per class: `bcc` 845, `akiec` 730, `nv` 244, `bkl` 235, `mel` 52.
- Manifests: `ml/data/manifest_pad.csv`, `ml/data/manifest_combined.csv` (HAM+PAD, 12121 rows).

---

## 5. How to reproduce

```bash
.venv/Scripts/python.exe -m ml.preprocessing.prepare_pad_ufes
.venv/Scripts/python.exe -m ml.preprocessing.split_dataset --manifest ml/data/manifest_pad.csv --out ml/configs/splits/split_pad_only.csv
.venv/Scripts/python.exe -m ml.training.train --arch resnet50 --manifest ml/data/manifest_pad.csv --splits ml/configs/splits/split_pad_only.csv --run-name resnet50_pad_seed42
.venv/Scripts/python.exe -m ml.evaluation.evaluate --checkpoint ml/checkpoints/resnet50_best.PAD-only.pt --manifest ml/data/manifest_pad.csv --splits ml/configs/splits/split_pad_only.csv --out-dir ml/results/eval_PAD-on-PAD
.venv/Scripts/python.exe -m ml.evaluation.evaluate --checkpoint ml/checkpoints/resnet50_best.HAM-only.pt --manifest ml/data/manifest.csv --splits ml/configs/splits/split_v1.csv --out-dir ml/results/eval_HAM-on-HAM
.venv/Scripts/python.exe -m ml.evaluation.evaluate --checkpoint ml/checkpoints/resnet50_best.HAM-only.pt --manifest ml/data/manifest_pad.csv --splits ml/configs/splits/split_pad_only.csv --out-dir ml/results/eval_HAM-on-PAD
```

Hyperparameters, seeds and timings for both training runs are in `ml/results/experiments.csv`
(`resnet50_seed42` = HAM, `resnet50_pad_seed42` = PAD).
