# Evaluation — resnet50

**Checkpoint:** `ml/checkpoints/resnet50_best.PAD-only.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.5752 (epoch 16)  
**Class mapping:** v1.0.0  
**Temperature:** 1.000 (uncalibrated)

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.4722** |
| Balanced accuracy | **0.6582** |
| Weighted F1 | 0.7625 |
| Accuracy | 0.7643 |
| Cohen's kappa | 0.6601 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0832 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.7768 | **0.8131** | 0.7945 | 107 |
| `bcc` Basal cell carcinoma | malignant | 0.8333 | **0.7752** | 0.8032 | 129 |
| `bkl` Benign keratosis | benign | 0.6875 | **0.6471** | 0.6667 | 34 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.5000 | **0.2500** | 0.3333 | 8 |
| `nv` Mole | benign | 0.6304 | **0.8056** | 0.7073 | 36 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.9385** |
| Specificity | 0.9000 |
| Missed serious cases (false negatives) | **15** |
| False alarms | 7 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 4.9 ms |
| Median | 4.9 ms |
| p95 | 5.2 ms |
| Device | cuda |
| Parameters | 23,522,375 |
| Model size | 89.7 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.7768    0.8131    0.7945       107
         bcc     0.8333    0.7752    0.8032       129
         bkl     0.6875    0.6471    0.6667        34
          df     0.0000    0.0000    0.0000         0
         mel     0.5000    0.2500    0.3333         8
          nv     0.6304    0.8056    0.7073        36
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.7643       314
   macro avg     0.4897    0.4701    0.4722       314
weighted avg     0.7665    0.7643    0.7625       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
