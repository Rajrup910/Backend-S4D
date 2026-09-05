# Evaluation — convnext_tiny

**Checkpoint:** `ml/checkpoints/convnext_tiny-asl_best.pt`  
**Split:** test (1,502 images)  
**Selected on:** validation macro_f1 = 0.7485 (epoch 13)  
**Class mapping:** v1.0.0  
**Temperature:** 1.000 (uncalibrated)

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.7305** |
| Balanced accuracy | **0.7029** |
| Weighted F1 | 0.8307 |
| Accuracy | 0.8362 |
| Cohen's kappa | 0.6753 |
| Macro ROC-AUC (OvR) | 0.9689 |
| Expected calibration error | 0.0955 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.5636 | **0.5962** | 0.5794 | 52 |
| `bcc` Basal cell carcinoma | malignant | 0.6941 | **0.8310** | 0.7564 | 71 |
| `bkl` Benign keratosis | benign | 0.8137 | **0.4970** | 0.6171 | 167 |
| `df` Dermatofibroma | benign | 0.8667 | **0.6500** | 0.7429 | 20 |
| `mel` Melanoma | malignant | 0.6294 | **0.6407** | 0.6350 | 167 |
| `nv` Mole | benign | 0.8942 | **0.9432** | 0.9181 | 1,004 |
| `vasc` Vascular lesion | benign | 1.0000 | **0.7619** | 0.8649 | 21 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.7483** |
| Specificity | 0.9233 |
| Missed serious cases (false negatives) | **73** |
| False alarms | 93 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 4.8 ms |
| Median | 4.8 ms |
| p95 | 5.0 ms |
| Device | cuda |
| Parameters | 27,825,511 |
| Model size | 106.1 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.5636    0.5962    0.5794        52
         bcc     0.6941    0.8310    0.7564        71
         bkl     0.8137    0.4970    0.6171       167
          df     0.8667    0.6500    0.7429        20
         mel     0.6294    0.6407    0.6350       167
          nv     0.8942    0.9432    0.9181      1004
        vasc     1.0000    0.7619    0.8649        21

    accuracy                         0.8362      1502
   macro avg     0.7803    0.7029    0.7305      1502
weighted avg     0.8360    0.8362    0.8307      1502
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
