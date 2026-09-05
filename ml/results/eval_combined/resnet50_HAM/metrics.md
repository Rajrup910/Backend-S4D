# Evaluation — resnet50

**Checkpoint:** `ml/checkpoints/resnet50_best.combined.pt`  
**Split:** test (1,489 images)  
**Selected on:** validation macro_f1 = 0.7408 (epoch 8)  
**Class mapping:** v1.0.0  
**Temperature:** 1.000 (uncalibrated)

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.7008** |
| Balanced accuracy | **0.7199** |
| Weighted F1 | 0.8339 |
| Accuracy | 0.8355 |
| Cohen's kappa | 0.6821 |
| Macro ROC-AUC (OvR) | 0.9515 |
| Expected calibration error | 0.1104 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.5741 | **0.6200** | 0.5962 | 50 |
| `bcc` Basal cell carcinoma | malignant | 0.8033 | **0.7206** | 0.7597 | 68 |
| `bkl` Benign keratosis | benign | 0.6685 | **0.6879** | 0.6781 | 173 |
| `df` Dermatofibroma | benign | 0.4783 | **0.6875** | 0.5641 | 16 |
| `mel` Melanoma | malignant | 0.6176 | **0.5217** | 0.5657 | 161 |
| `nv` Mole | benign | 0.9199 | **0.9319** | 0.9258 | 998 |
| `vasc` Vascular lesion | benign | 0.7692 | **0.8696** | 0.8163 | 23 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.6308** |
| Specificity | 0.9380 |
| Missed serious cases (false negatives) | **103** |
| False alarms | 75 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 5.3 ms |
| Median | 4.5 ms |
| p95 | 9.8 ms |
| Device | cuda |
| Parameters | 23,522,375 |
| Model size | 89.7 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.5741    0.6200    0.5962        50
         bcc     0.8033    0.7206    0.7597        68
         bkl     0.6685    0.6879    0.6781       173
          df     0.4783    0.6875    0.5641        16
         mel     0.6176    0.5217    0.5657       161
          nv     0.9199    0.9319    0.9258       998
        vasc     0.7692    0.8696    0.8163        23

    accuracy                         0.8355      1489
   macro avg     0.6901    0.7199    0.7008      1489
weighted avg     0.8340    0.8355    0.8339      1489
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
