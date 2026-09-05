# Evaluation — resnet50

**Checkpoint:** `ml/checkpoints/resnet50_best.HAM-only.pt`  
**Split:** test (1,502 images)  
**Selected on:** validation macro_f1 = 0.7401 (epoch 16)  
**Class mapping:** v1.0.0  
**Temperature:** 1.000 (uncalibrated)

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.7058** |
| Balanced accuracy | **0.7217** |
| Weighted F1 | 0.8228 |
| Accuracy | 0.8216 |
| Cohen's kappa | 0.6636 |
| Macro ROC-AUC (OvR) | 0.9452 |
| Expected calibration error | 0.1010 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.4815 | **0.7500** | 0.5865 | 52 |
| `bcc` Basal cell carcinoma | malignant | 0.7089 | **0.7887** | 0.7467 | 71 |
| `bkl` Benign keratosis | benign | 0.6892 | **0.6108** | 0.6476 | 167 |
| `df` Dermatofibroma | benign | 0.7619 | **0.8000** | 0.7805 | 20 |
| `mel` Melanoma | malignant | 0.5890 | **0.5749** | 0.5818 | 167 |
| `nv` Mole | benign | 0.9184 | **0.9084** | 0.9134 | 1,004 |
| `vasc` Vascular lesion | benign | 0.7647 | **0.6190** | 0.6842 | 21 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.7379** |
| Specificity | 0.9101 |
| Missed serious cases (false negatives) | **76** |
| False alarms | 109 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 5.4 ms |
| Median | 5.0 ms |
| p95 | 7.0 ms |
| Device | cuda |
| Parameters | 23,522,375 |
| Model size | 89.7 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.4815    0.7500    0.5865        52
         bcc     0.7089    0.7887    0.7467        71
         bkl     0.6892    0.6108    0.6476       167
          df     0.7619    0.8000    0.7805        20
         mel     0.5890    0.5749    0.5818       167
          nv     0.9184    0.9084    0.9134      1004
        vasc     0.7647    0.6190    0.6842        21

    accuracy                         0.8216      1502
   macro avg     0.7019    0.7217    0.7058      1502
weighted avg     0.8270    0.8216    0.8228      1502
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
