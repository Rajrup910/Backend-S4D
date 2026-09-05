# Evaluation — resnet50

**Checkpoint:** `ml/checkpoints/resnet50_best.HAM-only.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.7401 (epoch 16)  
**Class mapping:** v1.0.0  
**Temperature:** 1.000 (uncalibrated)

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.1419** |
| Balanced accuracy | **0.2695** |
| Weighted F1 | 0.2446 |
| Accuracy | 0.2452 |
| Cohen's kappa | 0.0804 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.3857 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.3286 | **0.2150** | 0.2599 | 107 |
| `bcc` Basal cell carcinoma | malignant | 0.7600 | **0.1473** | 0.2468 | 129 |
| `bkl` Benign keratosis | benign | 0.1429 | **0.2353** | 0.1778 | 34 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.0000 | **0.0000** | 0.0000 | 8 |
| `nv` Mole | benign | 0.1942 | **0.7500** | 0.3086 | 36 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.3893** |
| Specificity | 0.9143 |
| Missed serious cases (false negatives) | **149** |
| False alarms | 6 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 4.7 ms |
| Median | 4.6 ms |
| p95 | 5.0 ms |
| Device | cuda |
| Parameters | 23,522,375 |
| Model size | 89.7 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.3286    0.2150    0.2599       107
         bcc     0.7600    0.1473    0.2468       129
         bkl     0.1429    0.2353    0.1778        34
          df     0.0000    0.0000    0.0000         0
         mel     0.0000    0.0000    0.0000         8
          nv     0.1942    0.7500    0.3086        36
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.2452       314
   macro avg     0.2037    0.1925    0.1419       314
weighted avg     0.4619    0.2452    0.2446       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
