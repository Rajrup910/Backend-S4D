# Evaluation — densenet121

**Checkpoint:** `ml/checkpoints/densenet121_best.combined.pt`  
**Split:** test (1,489 images)  
**Selected on:** validation macro_f1 = 0.7430 (epoch 18)  
**Class mapping:** v1.0.0  
**Temperature:** 1.000 (uncalibrated)

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.7495** |
| Balanced accuracy | **0.7383** |
| Weighted F1 | 0.8370 |
| Accuracy | 0.8375 |
| Cohen's kappa | 0.6856 |
| Macro ROC-AUC (OvR) | 0.9493 |
| Expected calibration error | 0.0823 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.5962 | **0.6200** | 0.6078 | 50 |
| `bcc` Basal cell carcinoma | malignant | 0.7500 | **0.6618** | 0.7031 | 68 |
| `bkl` Benign keratosis | benign | 0.7310 | **0.6127** | 0.6667 | 173 |
| `df` Dermatofibroma | benign | 0.8667 | **0.8125** | 0.8387 | 16 |
| `mel` Melanoma | malignant | 0.5815 | **0.6646** | 0.6203 | 161 |
| `nv` Mole | benign | 0.9149 | **0.9269** | 0.9209 | 998 |
| `vasc` Vascular lesion | benign | 0.9091 | **0.8696** | 0.8889 | 23 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.7133** |
| Specificity | 0.9198 |
| Missed serious cases (false negatives) | **80** |
| False alarms | 97 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 16.8 ms |
| Median | 11.6 ms |
| p95 | 25.1 ms |
| Device | cuda |
| Parameters | 6,961,031 |
| Model size | 26.6 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.5962    0.6200    0.6078        50
         bcc     0.7500    0.6618    0.7031        68
         bkl     0.7310    0.6127    0.6667       173
          df     0.8667    0.8125    0.8387        16
         mel     0.5815    0.6646    0.6203       161
          nv     0.9149    0.9269    0.9209       998
        vasc     0.9091    0.8696    0.8889        23

    accuracy                         0.8375      1489
   macro avg     0.7642    0.7383    0.7495      1489
weighted avg     0.8387    0.8375    0.8370      1489
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
