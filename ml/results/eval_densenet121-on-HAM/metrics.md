# Evaluation — densenet121

**Checkpoint:** `ml/checkpoints/densenet121_best.pt`  
**Split:** test (1,502 images)  
**Selected on:** validation macro_f1 = 0.7207 (epoch 13)  
**Class mapping:** v1.0.0  
**Temperature:** 0.815

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.6969** |
| Balanced accuracy | **0.7340** |
| Weighted F1 | 0.8042 |
| Accuracy | 0.8003 |
| Cohen's kappa | 0.6293 |
| Macro ROC-AUC (OvR) | 0.9442 |
| Expected calibration error | 0.0486 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.3962 | **0.8077** | 0.5316 | 52 |
| `bcc` Basal cell carcinoma | malignant | 0.7397 | **0.7606** | 0.7500 | 71 |
| `bkl` Benign keratosis | benign | 0.6833 | **0.4910** | 0.5714 | 167 |
| `df` Dermatofibroma | benign | 0.6818 | **0.7500** | 0.7143 | 20 |
| `mel` Melanoma | malignant | 0.5215 | **0.5808** | 0.5496 | 167 |
| `nv` Mole | benign | 0.9179 | **0.8904** | 0.9039 | 1,004 |
| `vasc` Vascular lesion | benign | 0.8571 | **0.8571** | 0.8571 | 21 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.7379** |
| Specificity | 0.8754 |
| Missed serious cases (false negatives) | **76** |
| False alarms | 151 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 12.3 ms |
| Median | 11.9 ms |
| p95 | 13.7 ms |
| Device | cuda |
| Parameters | 6,961,031 |
| Model size | 26.6 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.3962    0.8077    0.5316        52
         bcc     0.7397    0.7606    0.7500        71
         bkl     0.6833    0.4910    0.5714       167
          df     0.6818    0.7500    0.7143        20
         mel     0.5215    0.5808    0.5496       167
          nv     0.9179    0.8904    0.9039      1004
        vasc     0.8571    0.8571    0.8571        21

    accuracy                         0.8003      1502
   macro avg     0.6854    0.7340    0.6969      1502
weighted avg     0.8172    0.8003    0.8042      1502
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
