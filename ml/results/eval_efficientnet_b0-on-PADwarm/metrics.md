# Evaluation — efficientnet_b0

**Checkpoint:** `ml/checkpoints/efficientnet_b0_best.PADwarm.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.5217 (epoch 11)  
**Class mapping:** v1.0.0  
**Temperature:** 0.995

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.4750** |
| Balanced accuracy | **0.6788** |
| Weighted F1 | 0.7656 |
| Accuracy | 0.7643 |
| Cohen's kappa | 0.6618 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0661 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.8523 | **0.7009** | 0.7692 | 107 |
| `bcc` Basal cell carcinoma | malignant | 0.8162 | **0.8605** | 0.8377 | 129 |
| `bkl` Benign keratosis | benign | 0.5814 | **0.7353** | 0.6494 | 34 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.3750 | **0.3750** | 0.3750 | 8 |
| `nv` Mole | benign | 0.6667 | **0.7222** | 0.6933 | 36 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.9221** |
| Specificity | 0.9000 |
| Missed serious cases (false negatives) | **19** |
| False alarms | 7 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 14.5 ms |
| Median | 15.2 ms |
| p95 | 24.5 ms |
| Device | cuda |
| Parameters | 4,016,515 |
| Model size | 15.3 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.8523    0.7009    0.7692       107
         bcc     0.8162    0.8605    0.8377       129
         bkl     0.5814    0.7353    0.6494        34
          df     0.0000    0.0000    0.0000         0
         mel     0.3750    0.3750    0.3750         8
          nv     0.6667    0.7222    0.6933        36
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.7643       314
   macro avg     0.4702    0.4848    0.4750       314
weighted avg     0.7747    0.7643    0.7656       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
