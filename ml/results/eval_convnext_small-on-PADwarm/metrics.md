# Evaluation — convnext_small

**Checkpoint:** `ml/checkpoints/convnext_small_best.PADwarm.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.5496 (epoch 14)  
**Class mapping:** v1.0.0  
**Temperature:** 1.266

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.5148** |
| Balanced accuracy | **0.7091** |
| Weighted F1 | 0.7982 |
| Accuracy | 0.8025 |
| Cohen's kappa | 0.7114 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0629 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.8736 | **0.7103** | 0.7835 | 107 |
| `bcc` Basal cell carcinoma | malignant | 0.7987 | **0.9225** | 0.8561 | 129 |
| `bkl` Benign keratosis | benign | 0.7188 | **0.6765** | 0.6970 | 34 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.6000 | **0.3750** | 0.4615 | 8 |
| `nv` Mole | benign | 0.7561 | **0.8611** | 0.8052 | 36 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.9631** |
| Specificity | 0.9143 |
| Missed serious cases (false negatives) | **9** |
| False alarms | 6 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 9.5 ms |
| Median | 8.0 ms |
| p95 | 13.7 ms |
| Device | cuda |
| Parameters | 49,460,071 |
| Model size | 188.7 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.8736    0.7103    0.7835       107
         bcc     0.7987    0.9225    0.8561       129
         bkl     0.7188    0.6765    0.6970        34
          df     0.0000    0.0000    0.0000         0
         mel     0.6000    0.3750    0.4615         8
          nv     0.7561    0.8611    0.8052        36
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.8025       314
   macro avg     0.5353    0.5065    0.5148       314
weighted avg     0.8056    0.8025    0.7982       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
